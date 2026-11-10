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
) -> dict[str, dict[str, float | None]]:
    """Features canónicas (LLM allowlist) por ventana para una condición."""
    path = features_path(ingest_dir, condition, subject)
    rows: dict[str, dict[str, float | None]] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        values: dict[str, float | None] = {}
        for modality in CHEST_MODALITIES:
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
) -> dict[str, Any]:
    """Payload ``numeric_llm`` con z-scores respecto a la referencia C0 del sujeto."""
    modalities: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        modality = name.split("_")[0]
        if modality not in CHEST_MODALITIES:
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
) -> tuple[np.ndarray, list[str], list[str]]:
    """Matriz de z-scores (referencia C0), sujetos y ventanas de una condición.

    Devuelve ``(X, groups, window_ids)``; las etiquetas se unen aparte desde el
    ground truth. Los valores ausentes se devuelven como ``nan`` para que el
    modelo de referencia los impute.
    """
    matrix: list[list[float]] = []
    groups: list[str] = []
    window_ids: list[str] = []
    for subject in subjects:
        clean = load_features(ingest_dir, "C0_clean", subject)
        centers, scales = subject_reference(clean, feature_names)
        rows = load_features(ingest_dir, condition, subject)
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
