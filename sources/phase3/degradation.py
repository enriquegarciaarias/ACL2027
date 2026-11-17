"""Utilidades del experimento de degradación (H1/H3/H4/H6).

Carga las features canónicas de cada sujeto y condición, construye la referencia
de estandarización intra-sujeto **solo con las ventanas limpias C0** y genera el
payload que ve el LLM. Se comparte entre el barrido (`run_degradation_sweep.py`)
y el análisis (`analyze_degradation.py`) para que ambos usen exactamente la
misma representación.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sources.ingestion.represent import LLM_ALLOWED_FEATURES, _opaque_window_id
from sources.phase2.binary_variants import DEFAULT_NONSTRESS_VARIANT, variant_classes
from sources.phase2.llm_config import model_slug

CHEST_MODALITIES = ("ecg", "eda", "resp", "temp")

#: Condiciones efectivamente distintas (C3 ≡ C1 a nivel de features).
EFFECTIVE_CONDITIONS: tuple[str, ...] = (
    "C0_clean",
    "C1_partial_resp",
    "C2_partial_eda",
    "C4_total_eda",
    "C5_total_both",
)

ALL_CONDITIONS: tuple[str, ...] = (
    "C0_clean",
    "C1_partial_resp",
    "C2_partial_eda",
    "C3_total_resp",
    "C4_total_eda",
    "C5_total_both",
)


def features_path(ingest_dir: Path, condition: str, subject: str) -> Path:
    """Ruta de features (degradadas si procede) de un sujeto."""
    if condition == "C0_clean":
        return ingest_dir / "features" / f"{subject}.features.jsonl"
    return ingest_dir / "degraded_features" / condition / f"{subject}.features.jsonl"


def load_features(
    ingest_dir: Path,
    condition: str,
    subject: str,
    include_temperature: bool = True,
) -> dict[str, dict[str, float | None]]:
    """Features canónicas (LLM allowlist) por ventana para una condición.

    Args:
        ingest_dir: raíz de artefactos de ``INGEST``.
        condition: identificador de condición (``C0_clean``, ``C1_partial_resp``…).
        subject: sujeto (``S5``…).
        include_temperature: si False, excluye ``temp`` de la vista del LLM.
            El experimento primario la excluye porque es un confusor verificado
            del orden del protocolo y la única modalidad de pecho que no se
            degrada; las sensibilidades pueden incluirla.
    """
    path = features_path(ingest_dir, condition, subject)
    rows: dict[str, dict[str, float | None]] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        values: dict[str, float | None] = {}
        for modality in CHEST_MODALITIES:
            if modality == "temp" and not include_temperature:
                continue
            for name, meta in record.get("features", {}).get(modality, {}).items():
                if name not in LLM_ALLOWED_FEATURES:
                    continue
                values[name] = meta.get("value")
        rows[record["window_id"]] = values
    return rows


def subject_reference(
    clean_rows: Mapping[str, Mapping[str, float | None]],
    feature_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mediana y escala (IQR) por feature usando SOLO las ventanas C0 limpias."""
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = np.asarray(
            [row[name] for row in clean_rows.values() if row.get(name) is not None],
            dtype=float,
        )
        if values.size == 0:
            centers[name], scales[name] = 0.0, 1.0
            continue
        centers[name] = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        scale = float(q3 - q1) or float(np.std(values)) or 1.0
        scales[name] = scale
    return centers, scales


def z_scores(
    values: Mapping[str, float | None],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
    clip: float = 3.0,
) -> dict[str, float | None]:
    """Convierte valores absolutos a z respecto a la referencia C0 del sujeto."""
    scores: dict[str, float | None] = {}
    for name, value in values.items():
        if value is None:
            scores[name] = None
        else:
            scores[name] = float(
                np.clip(
                    (value - centers.get(name, 0.0)) / scales.get(name, 1.0),
                    -clip,
                    clip,
                )
            )
    return scores


def build_payload(
    window_id: str,
    values: Mapping[str, float | None],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
    duration_s: float,
    clip: float = 3.0,
    include_temperature: bool = True,
) -> dict[str, Any]:
    """Payload ``numeric_llm`` con z-scores respecto a la referencia C0 del sujeto.

    ``include_temperature=False`` excluye la modalidad ``temp`` de la evidencia
    que ve el LLM (experimento primario: confusor de orden del protocolo y única
    modalidad no degradada). Las features canónicas siguen guardándose.
    """
    modalities: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        modality = name.split("_")[0]
        if modality not in CHEST_MODALITIES:
            continue
        if modality == "temp" and not include_temperature:
            continue
        if value is None:
            z = None
        else:
            z = float(
                np.clip(
                    (value - centers.get(name, 0.0)) / scales.get(name, 1.0),
                    -clip,
                    clip,
                )
            )
        modalities.setdefault(modality, {})[name] = {
            "value": z,
            "unit": "SD",
            "kind": "derived",
            "method": "within_subject_zscore_from_clean_reference",
        }
    return {
        "modalities": modalities,
        "feature_version": "degradation-e12",
        "window": {
            "window_id": _opaque_window_id(window_id),
            "duration_s": duration_s,
        },
    }


def condition_matrix(
    ingest_dir: Path,
    condition: str,
    subjects: list[str],
    feature_names: list[str],
    include_temperature: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Matriz de z-scores (referencia C0), sujetos y ventanas de una condición.

    Devuelve ``(X, groups, window_ids)``; las etiquetas se unen aparte desde el
    ground truth. Los valores ausentes se devuelven como ``nan`` para que el
    modelo de referencia los impute.

    ``include_temperature=False`` replica la vista del experimento primario: la
    referencia features-only debe ver **exactamente** las mismas features que el
    LLM, o la comparación H1 mezclaría un canal que el modelo no recibió.
    """
    matrix: list[list[float]] = []
    groups: list[str] = []
    window_ids: list[str] = []
    for subject in subjects:
        clean = load_features(
            ingest_dir, "C0_clean", subject, include_temperature=include_temperature
        )
        centers, scales = subject_reference(clean, feature_names)
        rows = load_features(
            ingest_dir, condition, subject, include_temperature=include_temperature
        )
        for window_id in sorted(rows):
            values = rows[window_id]
            row = []
            for name in feature_names:
                value = values.get(name)
                row.append(
                    float("nan")
                    if value is None
                    else (value - centers[name]) / scales[name]
                )
            matrix.append(row)
            groups.append(subject)
            window_ids.append(window_id)
    return np.asarray(matrix, dtype=float), groups, window_ids


#: Etiquetas legibles para la representación textual (mismas que Fase 1).
TEXT_LABELS: dict[str, str] = {
    "ecg_hr_bpm": "Heart rate",
    "ecg_n_beats": "Beats detected",
    "ecg_rr_mean_ms": "Mean RR interval",
    "ecg_sdnn_ms": "SDNN",
    "ecg_rmssd_ms": "RMSSD",
    "eda_mean_us": "Mean conductance",
    "eda_std_us": "Conductance variability",
    "eda_slope_us_per_s": "Conductance trend",
    "eda_tonic_mean_us": "Tonic level",
    "eda_phasic_std_us": "Phasic variability",
    "eda_scr_count": "SCR count",
    "eda_scr_rate_per_min": "SCR rate",
    "eda_scr_amplitude_mean_us": "Mean SCR amplitude",
    "eda_scr_amplitude_max_us": "Maximum SCR amplitude",
    "resp_rate_bpm": "Respiratory rate",
    "resp_interval_cv": "Breath interval variability",
    "resp_amplitude_pct": "Amplitude",
    "resp_std_pct": "Amplitude variability",
    "resp_n_cycles": "Breath intervals detected",
    "temp_mean_degc": "Mean temperature",
    "temp_std_degc": "Temperature variability",
    "temp_slope_degc_per_s": "Temperature trend",
    "temp_range_degc": "Temperature range",
}

TEXT_SECTIONS: dict[str, tuple[str, ...]] = {
    "CARDIOVASCULAR (ECG)": (
        "ecg_n_beats",
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
        "ecg_sdnn_ms",
        "ecg_rmssd_ms",
    ),
    "ELECTRODERMAL ACTIVITY (EDA)": (
        "eda_mean_us",
        "eda_std_us",
        "eda_slope_us_per_s",
        "eda_tonic_mean_us",
        "eda_phasic_std_us",
        "eda_scr_count",
        "eda_scr_rate_per_min",
        "eda_scr_amplitude_mean_us",
        "eda_scr_amplitude_max_us",
    ),
    "RESPIRATION": (
        "resp_rate_bpm",
        "resp_interval_cv",
        "resp_amplitude_pct",
        "resp_std_pct",
        "resp_n_cycles",
    ),
    "TEMPERATURE": (
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
    ),
}

FEATURE_SECTION: dict[str, str] = {
    feature: section for section, features in TEXT_SECTIONS.items() for feature in features
}


def render_text_summary(values: Mapping[str, float | None]) -> str:
    """Renderiza la representación textual de los mismos valores z (H3).

    Args:
        values: ``feature -> z`` (``None`` = no disponible).

    Returns:
        resumen textual con secciones y etiquetas legibles, en unidades ``SD``.
    """
    lines = ["PHYSIOLOGICAL SIGNAL SUMMARY", "=" * 50, ""]
    for section, features in TEXT_SECTIONS.items():
        present = [name for name in features if name in values]
        if not present:
            continue
        lines.append(section)
        for name in present:
            value = values[name]
            label = TEXT_LABELS.get(name, name)
            if value is None:
                lines.append(f"  {label}: not available")
            else:
                lines.append(f"  {label}: {value:+.2f} SD")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Rutas de artefactos: representación × tarea × variante × temperatura × modelo
# ---------------------------------------------------------------------------
#: Modos que corresponden al endpoint binario (CLI e interno).
BINARY_DECISION_MODES: tuple[str, ...] = ("binary", "stress_binary")

#: Prefijo de directorio por representación.
REPRESENTATION_PREFIXES: dict[str, str] = {
    "numeric": "degradation",
    "text": "degradation_text",
}


def is_binary_mode(decision_mode: str | None) -> bool:
    """True si ``decision_mode`` es el endpoint binario ``stress``/``non-stress``."""
    return str(decision_mode or "") in BINARY_DECISION_MODES


def degradation_dir_name(
    representation: str = "numeric",
    decision_mode: str = "affective",
    nonstress_variant: str | None = None,
    temperature_mode: str = "include",
    repetitions: int = 1,
) -> str:
    """Nombre del directorio de respuestas de un barrido.

    La tarea, la variante de ``non-stress`` y la presencia de temperatura forman
    parte del nombre: mezclar corpus de 4 clases, de binario con distintas
    clases negativas o con/sin temperatura invalidaría cualquier comparación.
    Las repeticiones añaden ``_rep{R}`` a partir de dos pasadas, para no
    mezclarse con el corpus de una sola pasada.

    Examples:
        >>> degradation_dir_name("numeric", "binary", "schmidt", "exclude")
        'degradation_binary_schmidt_notemp'
        >>> degradation_dir_name("text", "affective")
        'degradation_text'
        >>> degradation_dir_name("numeric", "binary", "schmidt", "exclude", 5)
        'degradation_binary_schmidt_notemp_rep5'
    """
    if representation not in REPRESENTATION_PREFIXES:
        raise ValueError(
            f"representación desconocida: {representation!r}. "
            f"Disponibles: {sorted(REPRESENTATION_PREFIXES)}"
        )
    name = REPRESENTATION_PREFIXES[representation]
    if is_binary_mode(decision_mode):
        variant = nonstress_variant or DEFAULT_NONSTRESS_VARIANT
        variant_classes(variant)  # valida la variante
        name += f"_binary_{variant}"
    if temperature_mode == "exclude":
        name += "_notemp"
    if int(repetitions) > 1:
        name += f"_rep{int(repetitions)}"
    return name


def degradation_output_dir(
    phase2_dir: Path,
    model: str,
    representation: str = "numeric",
    decision_mode: str = "affective",
    nonstress_variant: str | None = None,
    temperature_mode: str = "include",
    repetitions: int = 1,
) -> Path:
    """Directorio de salida de un modelo: ``{phase2}/{nombre}/{slug}``."""
    return (
        Path(phase2_dir)
        / degradation_dir_name(
            representation,
            decision_mode,
            nonstress_variant,
            temperature_mode,
            repetitions,
        )
        / model_slug(model)
    )


def degradation_candidate_dirs(
    phase2_dir: Path,
    representation: str = "numeric",
    decision_mode: str = "affective",
    nonstress_variant: str | None = None,
    temperature_mode: str | None = None,
) -> list[str]:
    """Nombres de directorio candidatos, en orden de preferencia.

    Sin variante ni temperatura explícitas se prefiere el **primario**
    (``schmidt`` sin temperatura) y se cae al histórico (``all`` con
    temperatura) solo si no existe. ``None`` significa "lo que haya".
    """
    if not is_binary_mode(decision_mode):
        return [
            degradation_dir_name(
                representation, "affective", None, temperature_mode or "include"
            )
        ]
    variants = (
        [nonstress_variant]
        if nonstress_variant
        else [DEFAULT_NONSTRESS_VARIANT, "all"]
    )
    temperatures = (
        [temperature_mode]
        if temperature_mode in ("exclude", "include")
        else ["exclude", "include"]
    )
    names: list[str] = []
    for variant in variants:
        for temperature in temperatures:
            name = degradation_dir_name(
                representation, "binary", variant, temperature
            )
            if name not in names:
                names.append(name)
    if not nonstress_variant and temperature_mode not in ("exclude", "include"):
        # Corpus anteriores al diseño por variantes: ``degradation_binary``
        # (``all`` con temperatura) sin sufijos. Solo en modo automático: una
        # petición explícita de variante/temperatura no debe caer aquí.
        legacy = f"{REPRESENTATION_PREFIXES[representation]}_binary"
        if legacy not in names:
            names.append(legacy)
    return names


def resolve_degradation_dir(
    phase2_dir: Path,
    model: str = "",
    representation: str = "numeric",
    decision_mode: str | None = None,
    nonstress_variant: str | None = None,
    temperature_mode: str | None = None,
) -> Path:
    """Localiza el directorio de respuestas de un barrido.

    Prueba, por orden, las combinaciones declaradas en
    :func:`degradation_candidate_dirs` y, si el nombre del modelo no existe,
    la carpeta histórica plana (ejecuciones anteriores al aislamiento por
    modelo) siempre que contenga ``*.responses.jsonl``.

    Raises:
        FileNotFoundError: si no hay respuestas en ninguna ruta candidata. El
            mensaje incluye las rutas probadas y la pista del barrido.
    """
    root = Path(phase2_dir)
    tried: list[Path] = []
    # ``decision_mode`` vacío = automático: primero el corpus binario (endpoint
    # primario) y después el de 4 clases.
    modes = [decision_mode] if decision_mode else ["binary", "affective"]
    for mode in modes:
        for name in degradation_candidate_dirs(
            root, representation, mode, nonstress_variant, temperature_mode
        ):
            base = root / name
            if model:
                candidates = [base / model_slug(model), base]  # subdirectorio y plano
            elif base.is_dir():
                # Sin modelo: primer subdirectorio (ordenado) con respuestas.
                candidates = sorted(p for p in base.iterdir() if p.is_dir()) or [base]
            else:
                candidates = [base]
            for candidate in candidates:
                tried.append(candidate)
                if candidate.is_dir() and any(candidate.glob("*.responses.jsonl")):
                    return candidate
    raise FileNotFoundError(
        "No se encontraron respuestas del barrido.\n"
        f"  Rutas probadas: {', '.join(str(p) for p in tried) or '(ninguna)'}\n"
        "  Genera primero el corpus con:\n"
        "    --proc GENERATE --experiment degradation --decision-mode binary \\\n"
        "      --nonstress-variant "
        f"{nonstress_variant or DEFAULT_NONSTRESS_VARIANT} "
        f"--temperature {temperature_mode or 'exclude'}"
    )


def resolve_repetitions_dir(
    phase2_dir: Path,
    model: str = "",
    representation: str = "numeric",
    decision_mode: str | None = None,
    nonstress_variant: str | None = None,
    temperature_mode: str | None = None,
    repetitions: int | None = None,
) -> Path:
    """Localiza el corpus de repeticiones (``..._rep{R}/{slug}``).

    Con ``repetitions`` explícito busca exactamente ese corpus; sin él recorre
    los directorios ``*_rep*`` y devuelve el primero con respuestas. Se separa
    del resolutor del barrido porque el sufijo ``_rep{R}`` no debe caer nunca al
    corpus de una sola pasada: analizar repeticiones sobre un corpus de una
    pasada daría «varianza cero» por construcción.

    Raises:
        FileNotFoundError: con las rutas probadas y el comando que genera el
            corpus.
    """
    root = Path(phase2_dir)
    mode = decision_mode or "binary"
    variant = (
        nonstress_variant if nonstress_variant is not None else DEFAULT_NONSTRESS_VARIANT
    )
    temperature = temperature_mode or "exclude"

    if repetitions:
        candidates = [
            root
            / degradation_dir_name(
                representation, mode, variant, temperature, repetitions
            )
        ]
    else:
        candidates = (
            sorted(p for p in root.iterdir() if p.is_dir() and "_rep" in p.name)
            if root.is_dir()
            else []
        )

    tried: list[Path] = []
    for base in candidates:
        ordered = ([base / model_slug(model)] if model else []) + [base]
        for directory in ordered:
            tried.append(directory)
            if directory.is_dir() and any(directory.glob("*.responses.jsonl")):
                return directory

    suffix = f" --repetitions {repetitions}" if repetitions else ""
    raise FileNotFoundError(
        "No se encontraron respuestas del corpus de repeticiones.\n"
        f"  Rutas probadas: {', '.join(str(p) for p in tried) or '(ninguna)'}\n"
        "  Genera primero el corpus con:\n"
        "    --proc GENERATE --experiment degradation --decision-mode "
        f"{mode} --nonstress-variant {variant} --temperature {temperature}{suffix}"
    )
