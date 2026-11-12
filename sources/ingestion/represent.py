"""
Generación de representaciones listas para el LLM (H5).

Dos formatos versionados:

- ``numeric``: JSON autodescriptivo por ventana, con ``value`` y ``unit`` por
  feature. No incluye métodos ni parámetros (quedan en ``features/``).
- ``text``: texto estructurado por secciones de modalidad, con unidades y un
  bloque de calidad.

Vistas:

- Auditoría (completa): incluye tiempos absolutos, IDs reales, SQI, cobertura,
  flags y procedencia. Se usa para análisis y depuración (H6).
- LLM (filtrada por lista de campos permitidos): solo valores de features con
  unidades. Sin tiempos absolutos, sin IDs reales, sin SQI, sin cobertura,
  sin flags. Identificadores de ventana opacos. Esto evita que el modelo copie
  la señalización de calidad en lugar de detectar degradación por sí mismo, y
  evita fuga indirecta de etiqueta mediante tiempos o identificadores.

Invariante anti-leakage: ninguna representación contiene la etiqueta.
La unión con la etiqueta se hará en la fase de evaluación.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from sources.common.common import logger, writeLog
from sources.common.paths import (
    atomic_write_json,
    atomic_write_jsonl,
    corpus_dir,
    ensure_dir,
    read_json,
    read_jsonl,
)
from sources.common.provenance import (
    build_fingerprint,
    code_fingerprint,
    sha256_json,
    utc_now,
)
from sources.common.validation import ValidationError
from sources.common.llm_client import (
    create_ollama_client,
    ollama_embeddings,
    ollama_warm_up,
)
from sources.ingestion import features, loader, window
from sources.ingestion.config import representations_config
from sources.ingestion.features.common import FEATURE_VERSION
from sources.ingestion.schemas import RepresentationRecord, SCHEMA_VERSION

#: Código que produce las representaciones (vistas numérica y textual).
_CODE_SCOPE: tuple[str, ...] = ("sources/ingestion/represent.py",)

REPRESENTATION_VERSION = "1.3"
KINDS = ("numeric", "text", "embeddings")

MODALITY_TITLES = {
    "ecg": "CARDIOVASCULAR (ECG)",
    "eda": "ELECTRODERMAL ACTIVITY (EDA)",
    "resp": "RESPIRATION",
    "temp": "TEMPERATURE",
    "acc": "MOTION (ACC)",
    "emg": "MUSCLE ACTIVITY (EMG)",
    "bvp": "PULSE (BVP/PPG)",
    "wrist_eda": "WRIST ELECTRODERMAL ACTIVITY (EDA)",
    "wrist_temp": "WRIST TEMPERATURE",
    "wrist_acc": "WRIST MOTION (ACC)",
}

FEATURE_LABELS = {
    "ecg_n_beats": "Beats detected",
    "ecg_amplitude_std_mv": "Amplitude variability",
    "ecg_hr_bpm": "Heart rate",
    "ecg_rr_mean_ms": "Mean RR interval",
    "ecg_sdnn_ms": "SDNN",
    "ecg_rmssd_ms": "RMSSD",
    "ecg_pnn50_pct": "pNN50",
    "ecg_rr_min_ms": "Minimum RR interval",
    "ecg_rr_max_ms": "Maximum RR interval",
    "ecg_rr_outlier_fraction": "RR outlier fraction",
    "eda_mean_us": "Mean conductance",
    "eda_std_us": "Conductance variability",
    "eda_slope_us_per_s": "Conductance trend",
    "eda_tonic_mean_us": "Tonic level",
    "eda_phasic_std_us": "Phasic variability",
    "eda_scr_candidates": "SCR candidates",
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
    "acc_mean_g": "Mean acceleration magnitude",
    "acc_std_g": "Acceleration variability",
    "acc_p95_g": "95th percentile acceleration",
    "acc_energy_g2": "Acceleration energy",
    "acc_motion_fraction": "Motion fraction",
    "emg_mean_abs_mv": "Mean absolute amplitude",
    "emg_std_mv": "Amplitude variability",
    "emg_rms_mv": "RMS amplitude",
    "emg_p95_abs_mv": "95th percentile amplitude",
    "bvp_n_beats": "Pulses detected",
    "bvp_amplitude_std": "Pulse amplitude variability",
    "bvp_hr_bpm": "Pulse rate",
    "bvp_rr_mean_ms": "Mean pulse interval",
    "bvp_rmssd_ms": "Pulse interval RMSSD",
    "bvp_rr_outlier_fraction": "Pulse interval outlier fraction",
}


def representations_dir() -> Path:
    return corpus_dir() / "representations"


def kind_dir(kind: str) -> Path:
    return representations_dir() / kind


def representation_path(kind: str, subject: str) -> Path:
    return kind_dir(kind) / f"{subject}.jsonl"


def llm_representation_path(kind: str, subject: str) -> Path:
    """Ruta de la representación filtrada para el LLM (vista restringida)."""
    return kind_dir(f"{kind}_llm") / f"{subject}.jsonl"


def report_path(kind: str, subject: str) -> Path:
    return kind_dir(kind) / f"{subject}.report.json"


def llm_report_path(kind: str, subject: str) -> Path:
    """Ruta del informe de la representación filtrada para el LLM."""
    return kind_dir(f"{kind}_llm") / f"{subject}.report.json"


def llm_motion_representation_path(kind: str, subject: str) -> Path:
    """Ruta de la representación LLM con canales de movimiento (vista con confusor)."""
    return kind_dir(f"{kind}_llm_motion") / f"{subject}.jsonl"


def llm_motion_report_path(kind: str, subject: str) -> Path:
    """Ruta del informe de la representación LLM con movimiento."""
    return kind_dir(f"{kind}_llm_motion") / f"{subject}.report.json"


def _filter_motion_channels(feature_block: dict[str, Any]) -> dict[str, Any]:
    """Filtra canales de movimiento (acc, emg) de un bloque de features.

    Devuelve una copia del bloque sin los features de acelerómetro ni EMG,
    que actúan como confusor natural del estrés en WESAD.
    """
    return {
        name: meta for name, meta in feature_block.items()
        if not (name.startswith("acc") or name.startswith("emg"))
    }


# Modalidades visibles en la vista LLM. Se excluyen las de muñeca
# (wrist_eda, wrist_temp, wrist_acc) para que ``numeric_llm`` y ``text_llm``
# compartan el mismo conjunto de canales (pecho). El canal de movimiento
# (acc/emg) se controla aparte con ``exclude_motion_channels``.
LLM_CHEST_MODALITIES: frozenset[str] = frozenset({"ecg", "eda", "resp", "temp"})
LLM_MOTION_MODALITIES: frozenset[str] = frozenset({"acc", "emg"})
# ``bvp`` se extrae de ``wrist_BVP`` (ver registry.py): es una modalidad de
# muñeca, no de pecho, y por tanto queda fuera del conjunto experimental
# mínimo (se reserva para el experimento de sobre-confianza en medidas
# derivadas, donde se compara HR de BVP con el ECG de referencia).
WRIST_MODALITIES: frozenset[str] = frozenset(
    {"bvp", "wrist_eda", "wrist_temp", "wrist_acc"}
)

# Conjunto curado de features que forma la vista LLM. Se aplica tanto a
# ``numeric_llm`` como a ``text_llm`` para que la comparación entre
# representaciones no esté confundida por un conjunto distinto de features.
LLM_ALLOWED_FEATURES: frozenset[str] = frozenset(
    {
        # EDA
        "eda_mean_us",
        "eda_std_us",
        "eda_slope_us_per_s",
        "eda_tonic_mean_us",
        "eda_phasic_std_us",
        "eda_scr_count",
        "eda_scr_rate_per_min",
        "eda_scr_amplitude_mean_us",
        "eda_scr_amplitude_max_us",
        # ECG / cardiovascular
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
        "ecg_sdnn_ms",
        "ecg_rmssd_ms",
        "ecg_n_beats",
        # Respiración
        "resp_rate_bpm",
        "resp_interval_cv",
        "resp_amplitude_pct",
        "resp_std_pct",
        "resp_n_cycles",
        # Temperatura
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
        # Movimiento (solo en la vista *_llm_motion)
        "acc_mean_g",
        "acc_std_g",
        "acc_p95_g",
        "acc_energy_g2",
        "acc_motion_fraction",
        "emg_mean_abs_mv",
        "emg_std_mv",
        "emg_rms_mv",
        "emg_p95_abs_mv",
    }
)


def _llm_view_features(
    feature_row: Mapping[str, Any],
    exclude_motion_channels: bool,
) -> dict[str, dict[str, Any]]:
    """Devuelve las features visibles en la vista LLM (pecho, sin muñeca).

    Unifica el conjunto de canales y de features entre ``numeric_llm`` y
    ``text_llm``: solo modalidades de pecho (ecg, eda, resp, temp), solo
    features de ``LLM_ALLOWED_FEATURES`` y, opcionalmente, los canales de
    movimiento (acc/emg) cuando no se excluyen. Las modalidades de muñeca
    (incluida ``bvp``, derivada de ``wrist_BVP``) nunca forman parte del
    conjunto experimental mínimo.
    """
    allowed = LLM_CHEST_MODALITIES
    if not exclude_motion_channels:
        allowed = allowed | LLM_MOTION_MODALITIES
    result: dict[str, dict[str, Any]] = {}
    for modality, block in feature_row.get("features", {}).items():
        if modality not in allowed:
            continue
        kept = {
            name: meta
            for name, meta in block.items()
            if name in LLM_ALLOWED_FEATURES
        }
        if kept:
            result[modality] = kept
    return result


def index_path(kind: str) -> Path:
    return kind_dir(kind) / f"{kind}_index.jsonl"


def global_report_path(kind: str) -> Path:
    return kind_dir(kind) / f"{kind}.report.json"


def spec_path() -> Path:
    return representations_dir() / "representation_spec.json"


def _window_meta(window_row: Mapping[str, Any]) -> dict:
    """Copia solo metadatos permitidos (nunca la etiqueta)."""
    keys = ("window_id", "subject", "t_start_s", "t_end_s", "duration_s", "coverage")
    return {key: window_row.get(key) for key in keys}


def _opaque_window_id(window_id: str) -> str:
    """Devuelve un identificador opaco de 8 caracteres para la vista del LLM.

    Evita fuga indirecta de etiqueta mediante IDs reales (w00001) o nombres
    de sujeto que permiten memorización o correlación con el protocolo fijo
    de WESAD.
    """
    return "win_" + hashlib.sha256(window_id.encode()).hexdigest()[:8]


def _round(value: Any, digits: int) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, digits)
    return value


def _fmt(value: Any, fmt: str) -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def build_numeric(
    feature_row: Mapping[str, Any],
    window_meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
    llm_view: bool = False,
    exclude_motion_channels: bool | None = None,
) -> dict:
    """Construye la representación numérica de una ventana.

    Cuando ``llm_view`` es True, filtra los campos según la lista de campos
    permitidos para el LLM: sin metadatos de ventana (tiempos, IDs), sin
    calidad, sin cobertura, sin procedencia. Solo ``value`` y ``unit`` por
    feature, con identificador de ventana opaco.

    Cuando ``exclude_motion_channels`` es True, elimina los canales de
    acelerómetro (acc.*) y EMG (emg.*) para evitar el confusor natural
    del movimiento en WESAD.
    """
    options = representations_config(cfg)["numeric"]
    digits = int(options.get("round_digits", 4))
    include_units = bool(options.get("include_units", True))

    if llm_view:
        llm_opts = options.get("llm_view", {})
        include_units = bool(llm_opts.get("allowed_fields", ["value", "unit"])) and "unit" in llm_opts.get("allowed_fields", ["value", "unit"])
        include_provenance = False
        include_quality = False
        include_window_meta = False
        opaque_id = bool(llm_opts.get("opaque_window_id", True))
        # El parámetro explícito tiene prioridad sobre la config; si no se
        # pasa, se usa el valor de llm_view.exclude_motion_channels.
        if exclude_motion_channels is None:
            exclude_motion_channels = bool(llm_opts.get("exclude_motion_channels", False))
    else:
        include_provenance = bool(options.get("include_provenance", True))
        include_quality = bool(options.get("include_quality", True))
        include_window_meta = True
        opaque_id = False

    if llm_view:
        features_source = _llm_view_features(feature_row, exclude_motion_channels)
    else:
        features_source = feature_row.get("features", {})

    modalities: dict[str, dict] = {}
    for modality, feature_block in features_source.items():
        modalities[modality] = {}
        for name, meta in feature_block.items():
            item: dict[str, Any] = {"value": _round(meta.get("value"), digits)}
            if include_units:
                item["unit"] = meta.get("unit")
            if include_provenance:
                item["kind"] = meta.get("kind", "derived")
                item["method"] = meta.get("method")
            modalities[modality][name] = item

    payload: dict[str, Any] = {"modalities": modalities,
                               "feature_version": feature_row.get("feature_version", FEATURE_VERSION)}
    if include_window_meta:
        payload["window"] = dict(window_meta)
    elif opaque_id:
        payload["window"] = {
            "window_id": _opaque_window_id(str(window_meta.get("window_id", ""))),
            "duration_s": window_meta.get("duration_s"),
        }
    if include_quality:
        payload["quality"] = dict(feature_row.get("quality", {}))
    return payload


def build_text(
    feature_row: Mapping[str, Any],
    window_meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
    llm_view: bool = False,
    exclude_motion_channels: bool | None = None,
) -> str:
    """Construye la representación textual estructurada de una ventana.

    Cuando ``llm_view`` es True, filtra los campos según la lista de campos
    permitidos para el LLM: sin header de ventana (Window, Subject, Time),
    sin bloque SIGNAL QUALITY, sin Coverage, sin Concerns. Solo secciones por
    modalidad con valor y unidad, e identificador de ventana opaco al inicio.

    Cuando ``exclude_motion_channels`` es True, elimina los canales de
    acelerómetro (acc.*) y EMG (emg.*) para evitar el confusor natural
    del movimiento en WESAD.
    """
    options = representations_config(cfg)["text"]
    fmt = str(options.get("float_format", ".4g"))
    include_methods = bool(options.get("include_methods", False))

    if llm_view:
        llm_opts = options.get("llm_view", {})
        include_quality = False
        include_times = False
        include_ids = False
        opaque_id = bool(llm_opts.get("opaque_window_id", True))
        # El parámetro explícito tiene prioridad sobre la config.
        if exclude_motion_channels is None:
            exclude_motion_channels = bool(llm_opts.get("exclude_motion_channels", False))
    else:
        include_quality = bool(options.get("include_quality", True))
        include_times = True
        include_ids = True
        opaque_id = False

    max_flags = int(options.get("max_flags", 10))

    lines: list[str] = [
        "PHYSIOLOGICAL SIGNAL SUMMARY",
        "=" * 50,
        "",
    ]
    if not llm_view:
        if include_ids:
            lines.append(f"Window: {window_meta.get('window_id')}")
            lines.append(f"Subject: {window_meta.get('subject')}")
        else:
            lines.append(f"Window: {_opaque_window_id(str(window_meta.get('window_id', '')))}")
        if include_times:
            lines.append(
                f"Time: {_fmt(window_meta.get('t_start_s'), fmt)} s - "
                f"{_fmt(window_meta.get('t_end_s'), fmt)} s "
                f"(duration {_fmt(window_meta.get('duration_s'), fmt)} s)"
            )
        lines.append("")

    if llm_view:
        features_source = _llm_view_features(feature_row, exclude_motion_channels)
    else:
        features_source = feature_row.get("features", {})

    for modality, feature_block in features_source.items():
        lines.append("")
        lines.append(MODALITY_TITLES.get(modality, modality.upper()))
        for name, meta in feature_block.items():
            label = FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())
            value = _fmt(meta.get("value"), fmt)
            unit = meta.get("unit")
            suffix = f" {unit}" if unit else ""
            method = ""
            if include_methods:
                method = f"  [{meta.get('method')} {meta.get('method_version')}]"
            lines.append(f"  {label}: {value}{suffix}{method}")

    if not llm_view:
        quality = feature_row.get("quality", {})
        if include_quality and quality:
            lines.append("")
            lines.append("SIGNAL QUALITY")
            lines.append(
                f"  Overall SQI: {_fmt(quality.get('sqi_overall'), fmt)} "
                f"({quality.get('category', 'unknown')})"
            )
            lines.append(
                f"  SQI range: {_fmt(quality.get('sqi_min'), fmt)} - "
                f"{_fmt(quality.get('sqi_max'), fmt)}"
            )
            per_modality = quality.get("per_modality", {})
            if per_modality:
                summary = ", ".join(
                    f"{name}={_fmt(block.get('sqi'), fmt)}"
                    for name, block in per_modality.items()
                )
                lines.append(f"  Per modality: {summary}")
            coverage = window_meta.get("coverage") or {}
            if coverage:
                summary = ", ".join(
                    f"{name}={_fmt(value, fmt)}" for name, value in coverage.items()
                )
                lines.append(f"  Coverage: {summary}")
            flags = list(quality.get("flags", []))
            if flags:
                shown = ", ".join(flags[:max_flags])
                if len(flags) > max_flags:
                    shown += f" ... (+{len(flags) - max_flags})"
                lines.append(f"  Concerns: {shown}")
            else:
                lines.append("  Concerns: none")

    return "\n".join(lines) + "\n"


def build_embeddings(
    feature_row: Mapping[str, Any],
    window_meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
    ollama_client: Mapping[str, Any],
) -> dict:
    """Genera el embedding de una ventana usando Ollama.

    Vectoriza la representación ``text`` de la misma ventana (misma fuente de
    datos, misma configuración de floats) para mantener coherencia entre las
    representaciones textual y vectorial.

    Args:
        feature_row: fila de features de la ventana.
        window_meta: metadatos de la ventana (sin etiqueta).
        cfg: configuración resuelta de ingesta.
        ollama_client: configuración de cliente Ollama (ver ``llm_client``).

    Returns:
        dict con claves ``vector``, ``dimension``, ``model``, ``source``.
    """
    rep_cfg = representations_config(cfg)
    emb_cfg = rep_cfg.get("embeddings", {})
    model = str(emb_cfg.get("model", "nomic-embed-text"))
    source = str(emb_cfg.get("source", "text"))

    if source != "text":
        raise ValidationError(
            f"representaciones.embeddings.source='{source}' no soportado; se requiere 'text'",
            where="represent",
        )

    text_payload = build_text(feature_row, window_meta, cfg)
    vector = ollama_embeddings(ollama_client, model, text_payload)

    return {
        "vector": vector,
        "dimension": len(vector),
        "model": model,
        "source": source,
    }


def representation_fingerprint(
    meta: Mapping[str, Any],
    windows_report: Mapping[str, Any],
    features_report: Mapping[str, Any],
    cfg: Mapping[str, Any],
    kind: str,
) -> str:
    return build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "representation": kind,
            "representation_version": REPRESENTATION_VERSION,
            "feature_version": FEATURE_VERSION,
            "features_fingerprint": features_report.get("fingerprint"),
            "windows_fingerprint": windows_report.get("fingerprint"),
            "canonical_sha256": meta.get("canonical_sha256"),
            "options": representations_config(cfg).get(kind),
            "code": code_fingerprint(_CODE_SCOPE),
        }
    )


def _write_spec(cfg: Mapping[str, Any]) -> None:
    rep_cfg = representations_config(cfg)
    spec = {
        "schema_version": SCHEMA_VERSION,
        "representation_version": REPRESENTATION_VERSION,
        "feature_version": FEATURE_VERSION,
        "generated_at": utc_now(),
        "enabled": rep_cfg.get("enabled", []),
        "options": {kind: rep_cfg.get(kind, {}) for kind in KINDS},
        "formats": {
            "numeric": "JSON autodescriptivo: value + unit por feature + quality",
            "text": "texto estructurado por modalidad + bloque de calidad",
            "numeric_llm": "JSON filtrado para LLM: solo value + unit, sin quality, sin coverage, sin tiempos, sin IDs reales",
            "text_llm": "Texto filtrado para LLM: solo secciones por modalidad + valor + unidad, sin header de ventana, sin SIGNAL QUALITY, sin Coverage, sin Concerns",
        },
        "notes": {
            "labels": "ninguna representación contiene etiquetas (anti-leakage)",
            "methods": "los métodos y parámetros quedan en features/feature_spec.json",
            "llm_view": "vista filtrada por lista de campos permitidos, no por exclusión. Por defecto sin calidad ni cobertura. Se activa con representations.{kind}.llm_view.enabled=true",
        },
    }
    ensure_dir(representations_dir())
    atomic_write_json(spec_path(), spec)


def run_represent(cfg: Mapping[str, Any], subject: str, *, force: bool = False) -> dict:
    """Genera las representaciones habilitadas de un sujeto."""
    _, meta_path = loader.canonical_paths(subject)
    meta = read_json(meta_path)
    if meta is None:
        raise ValidationError(
            f"no existe el canónico de {subject}; ejecute primero '--step ingest'",
            where="represent",
        )
    windows_report = read_json(window.report_path(subject))
    if windows_report is None:
        raise ValidationError(
            f"no existen ventanas de {subject}; ejecute primero '--step window'",
            where="represent",
        )
    features_report = read_json(features.report_path(subject))
    if features_report is None:
        raise ValidationError(
            f"faltan features de {subject}; ejecute primero '--step features'",
            where="represent",
        )

    rep_cfg = representations_config(cfg)
    enabled = list(rep_cfg.get("enabled", []))
    window_rows = {
        row["window_id"]: row for row in read_jsonl(window.windows_path(subject))
    }
    feature_rows = read_jsonl(features.features_path(subject))

    statuses: dict[str, str] = {}
    counters: dict[str, int] = {}

    # Cliente Ollama (se crea solo si se va a generar embeddings).
    ollama_client: Mapping[str, Any] | None = None
    ollama_ready = False
    if "embeddings" in enabled:
        ollama_client = create_ollama_client()
        emb_model = str(
            rep_cfg.get("embeddings", {}).get("model", "nomic-embed-text")
        )
        writeLog(
            "info",
            logger,
            f"[represent] {subject}: Ollama client listo para embeddings (modelo={emb_model})",
        )
        # Warm-up: cargar el modelo en Ollama antes del bucle de ventanas.
        # Los modelos grandes (mxbai-embed-large ~669 MB) tardan en cargarse
        # la primera vez y pueden devolver 500 si se consulta inmediatamente.
        try:
            ollama_warm_up(ollama_client, emb_model)
            ollama_ready = True
            writeLog(
                "info",
                logger,
                f"[represent] {subject}: Ollama warm-up completado (modelo={emb_model})",
            )
        except Exception as exc:
            # Los embeddings son material complementario (la Fase 2 usa
            # numeric/text): si Ollama no está, se omite la representación en
            # lugar de tumbar la ingesta. Reintentar por ventana multiplicaría
            # los timeouts y el backoff.
            enabled = [kind for kind in enabled if kind != "embeddings"]
            cfg["representations"]["enabled"] = list(enabled)
            statuses["embeddings"] = "skipped"
            counters["embeddings"] = 0
            writeLog(
                "warning",
                logger,
                f"[represent] {subject}: embeddings omitidos, Ollama no responde "
                f"({exc}); se generan solo {enabled}",
            )

    for kind in enabled:
        if kind not in KINDS:
            raise ValidationError(f"representación no soportada: {kind}", where="represent")

        fingerprint = representation_fingerprint(
            meta, windows_report, features_report, cfg, kind
        )
        target = representation_path(kind, subject)
        report_target = report_path(kind, subject)
        previous = read_json(report_target)
        if (
            not force
            and previous is not None
            and previous.get("fingerprint") == fingerprint
            and target.is_file()
        ):
            statuses[kind] = "reused"
            counters[kind] = int(previous.get("n_windows", 0))
            continue

        records: list[dict] = []
        embedding_error: str | None = None
        for feature_row in feature_rows:
            window_row = window_rows.get(feature_row["window_id"])
            if window_row is None:
                raise ValidationError(
                    f"ventana {feature_row['window_id']} presente en features pero no en windows",
                    where="represent",
                )
            window_meta = _window_meta(window_row)
            if kind == "numeric":
                payload = build_numeric(feature_row, window_meta, cfg)
            elif kind == "text":
                payload = build_text(feature_row, window_meta, cfg)
            else:  # kind == "embeddings"
                if ollama_client is None:
                    raise ValidationError(
                        "cliente Ollama no inicializado para embeddings",
                        where="represent",
                    )
                try:
                    payload = build_embeddings(
                        feature_row, window_meta, cfg, ollama_client
                    )
                except Exception as exc:  # Ollama caído o modelo descargado
                    embedding_error = str(exc)
                    break
            records.append(
                RepresentationRecord(
                    window_id=feature_row["window_id"],
                    subject=subject,
                    representation=kind,
                    representation_version=REPRESENTATION_VERSION,
                    payload=payload,
                    config_fingerprint=fingerprint,
                    feature_version=FEATURE_VERSION,
                    schema_version=SCHEMA_VERSION,
                ).to_dict()
            )

        if embedding_error is not None:
            cfg["representations"]["enabled"] = [
                item for item in enabled if item != "embeddings"
            ]
            statuses["embeddings"] = "skipped"
            counters["embeddings"] = 0
            writeLog(
                "warning",
                logger,
                f"[represent] {subject}: embeddings omitidos ({embedding_error}); "
                "se generan solo las vistas numérica y textual",
            )
            continue

        ensure_dir(kind_dir(kind))
        atomic_write_jsonl(target, records)
        atomic_write_json(
            report_target,
            {
                "schema_version": SCHEMA_VERSION,
                "representation": kind,
                "representation_version": REPRESENTATION_VERSION,
                "feature_version": FEATURE_VERSION,
                "subject": subject,
                "created_at": utc_now(),
                "fingerprint": fingerprint,
                "config_fingerprint": sha256_json({"representations": rep_cfg}),
                "n_windows": len(records),
                "file": str(target),
            },
        )
        statuses[kind] = "ok"
        counters[kind] = len(records)

        # Vista filtrada para el LLM (llm_view). Se genera en directorio
        # separado para no mezclar con la vista de auditoría completa.
        llm_opts = rep_cfg.get(kind, {}).get("llm_view", {})
        if bool(llm_opts.get("enabled", False)):
            # Vista LLM sin canales de movimiento (limpia, por defecto)
            llm_records: list[dict] = []
            for feature_row in feature_rows:
                window_row = window_rows.get(feature_row["window_id"])
                if window_row is None:
                    raise ValidationError(
                        f"ventana {feature_row['window_id']} presente en features pero no en windows",
                        where="represent",
                    )
                window_meta = _window_meta(window_row)
                if kind == "numeric":
                    payload = build_numeric(
                        feature_row, window_meta, cfg,
                        llm_view=True,
                        exclude_motion_channels=bool(
                            llm_opts.get("exclude_motion_channels", False)
                        ),
                    )
                elif kind == "text":
                    payload = build_text(
                        feature_row, window_meta, cfg,
                        llm_view=True,
                        exclude_motion_channels=bool(
                            llm_opts.get("exclude_motion_channels", False)
                        ),
                    )
                else:
                    payload = build_embeddings(
                        feature_row, window_meta, cfg, ollama_client
                    )
                llm_records.append(
                    RepresentationRecord(
                        window_id=feature_row["window_id"],
                        subject=subject,
                        representation=f"{kind}_llm",
                        representation_version=REPRESENTATION_VERSION,
                        payload=payload,
                        config_fingerprint=fingerprint,
                        feature_version=FEATURE_VERSION,
                        schema_version=SCHEMA_VERSION,
                    ).to_dict()
                )
            llm_target = llm_representation_path(kind, subject)
            llm_report_target = llm_report_path(kind, subject)
            ensure_dir(kind_dir(f"{kind}_llm"))
            atomic_write_jsonl(llm_target, llm_records)
            atomic_write_json(
                llm_report_target,
                {
                    "schema_version": SCHEMA_VERSION,
                    "representation": f"{kind}_llm",
                    "representation_version": REPRESENTATION_VERSION,
                    "feature_version": FEATURE_VERSION,
                    "subject": subject,
                    "created_at": utc_now(),
                    "fingerprint": fingerprint,
                    "config_fingerprint": sha256_json(
                        {"representations": rep_cfg, "llm_view": llm_opts,
                         "exclude_motion_channels": llm_opts.get("exclude_motion_channels", False)}
                    ),
                    "n_windows": len(llm_records),
                    "file": str(llm_target),
                },
            )
            statuses[f"{kind}_llm"] = "ok"
            counters[f"{kind}_llm"] = len(llm_records)
            writeLog(
                "info",
                logger,
                f"[represent] {subject}: {kind}_llm generado ({len(llm_records)} ventanas)",
            )

            # Vista LLM con canales de movimiento (con confusor) cuando
            # control_motion_confounder está activo.
            exp_cfg = cfg.get("experiment", {})
            if bool(exp_cfg.get("control_motion_confounder", False)):
                llm_motion_records: list[dict] = []
                for feature_row in feature_rows:
                    window_row = window_rows.get(feature_row["window_id"])
                    if window_row is None:
                        raise ValidationError(
                            f"ventana {feature_row['window_id']} presente en features pero no en windows",
                            where="represent",
                        )
                    window_meta = _window_meta(window_row)
                    if kind == "numeric":
                        payload = build_numeric(
                            feature_row, window_meta, cfg,
                            llm_view=True,
                            exclude_motion_channels=False,
                        )
                    elif kind == "text":
                        payload = build_text(
                            feature_row, window_meta, cfg,
                            llm_view=True,
                            exclude_motion_channels=False,
                        )
                    else:
                        payload = build_embeddings(
                            feature_row, window_meta, cfg, ollama_client
                        )
                    llm_motion_records.append(
                        RepresentationRecord(
                            window_id=feature_row["window_id"],
                            subject=subject,
                            representation=f"{kind}_llm_motion",
                            representation_version=REPRESENTATION_VERSION,
                            payload=payload,
                            config_fingerprint=fingerprint,
                            feature_version=FEATURE_VERSION,
                            schema_version=SCHEMA_VERSION,
                        ).to_dict()
                    )
                llm_motion_target = llm_motion_representation_path(kind, subject)
                llm_motion_report_target = llm_motion_report_path(kind, subject)
                ensure_dir(kind_dir(f"{kind}_llm_motion"))
                atomic_write_jsonl(llm_motion_target, llm_motion_records)
                atomic_write_json(
                    llm_motion_report_target,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "representation": f"{kind}_llm_motion",
                        "representation_version": REPRESENTATION_VERSION,
                        "feature_version": FEATURE_VERSION,
                        "subject": subject,
                        "created_at": utc_now(),
                        "fingerprint": fingerprint,
                        "config_fingerprint": sha256_json(
                            {"representations": rep_cfg, "llm_view": llm_opts,
                             "exclude_motion_channels": False}
                        ),
                        "n_windows": len(llm_motion_records),
                        "file": str(llm_motion_target),
                    },
                )
                statuses[f"{kind}_llm_motion"] = "ok"
                counters[f"{kind}_llm_motion"] = len(llm_motion_records)
                writeLog(
                    "info",
                    logger,
                    f"[represent] {subject}: {kind}_llm_motion generado ({len(llm_motion_records)} ventanas)",
                )

    _write_spec(cfg)
    writeLog(
        "info",
        logger,
        f"[represent] {subject}: {statuses} ({sum(counters.values())} ventanas-representación)",
    )
    return {
        "subject": subject,
        "status": statuses,
        "n_windows": len(feature_rows),
        "counts": counters,
    }


def build_global_index(cfg: Mapping[str, Any], subjects: Iterable[str]) -> dict:
    """Concatena las representaciones por tipo en sus índices globales."""
    subjects = list(subjects)
    rep_cfg = representations_config(cfg)
    enabled = list(rep_cfg.get("enabled", []))
    summary: dict[str, dict] = {}

    for kind in enabled:
        all_rows: list[dict] = []
        per_subject: dict[str, int] = {}
        for subject in subjects:
            path = representation_path(kind, subject)
            if not path.is_file():
                raise ValidationError(
                    f"faltan representaciones '{kind}' de {subject}; ejecute 'represent' antes del índice global",
                    where="represent",
                )
            rows = read_jsonl(path)
            per_subject[subject] = len(rows)
            all_rows.extend(rows)

        ensure_dir(kind_dir(kind))
        atomic_write_jsonl(index_path(kind), all_rows)
        report = {
            "schema_version": SCHEMA_VERSION,
            "representation": kind,
            "representation_version": REPRESENTATION_VERSION,
            "feature_version": FEATURE_VERSION,
            "created_at": utc_now(),
            "n_subjects": len(subjects),
            "n_windows": len(all_rows),
            "per_subject": per_subject,
            "index_file": str(index_path(kind)),
            "config_fingerprint": sha256_json({"representations": rep_cfg}),
        }
        atomic_write_json(global_report_path(kind), report)
        summary[kind] = report
        writeLog(
            "info",
            logger,
            f"[represent] índice global '{kind}': {len(all_rows)} registros de {len(subjects)} sujetos",
        )

    _write_spec(cfg)
    return summary
