"""
Configuración resuelta del pipeline de ingesta.

La configuración vive en ``params.ingestion`` del documento fusionado
(global + subject). Aquí se aplican los valores por defecto del proyecto, se
valida el resultado y se normalizan las listas de sujetos.

Cualquier parámetro que afecte a un artefacto debe estar en este bloque: el
fingerprint del corpus se calcula sobre la configuración resuelta, no sobre el
texto del ``config.json``.
"""
from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Optional

from sources.common.common import processControl
from sources.common.paramsManager import deep_merge
from sources.common.validation import ValidationError, require

REPRESENTATION_DEFAULTS: dict = {
    "enabled": ["numeric", "text", "embeddings"],
    "numeric": {
        "round_digits": 4,
        "include_quality": True,
        "include_units": True,
        "include_provenance": True,
        "llm_view": {
            "enabled": False,
            "include_quality": False,
            "include_coverage": False,
            "include_provenance": False,
            "include_window_meta": False,
            "opaque_window_id": True,
            "allowed_fields": ["value", "unit"],
            "exclude_motion_channels": True,
        },
    },
    "text": {
        "include_quality": True,
        "include_methods": False,
        "float_format": ".4g",
        "max_flags": 10,
        "llm_view": {
            "enabled": False,
            "include_quality": False,
            "include_times": False,
            "include_ids": False,
            "opaque_window_id": True,
            "exclude_motion_channels": True,
        },
    },
    "embeddings": {
        "model": "mxbai-embed-large",
        "source": "text",
        "batch_size": 1,
        "include_quality": True,
        "llm_view": {
            "enabled": False,
        },
    },
}

DEFAULTS: dict = {
    "dataset": "WESAD",    "source_dir": ".",
    "source_file": "{subject}.pkl",
    "subject_glob": r"^S\d+$",
    "subjects": "auto",
    "seed": 42,
    "sampling_rates": {
        "label": 700.0,
        "chest": 700.0,
        "wrist_acc": 32.0,
        "wrist_bvp": 64.0,
        "wrist_eda": 4.0,
        "wrist_temp": 4.0,
    },
    "labels": {
        "keep": [1, 2, 3, 4],
        "names": {
            "0": "undefined",
            "1": "baseline",
            "2": "stress",
            "3": "amusement",
            "4": "meditation",
            "6": "ignore",
            "7": "ignore",
        },
    },
    "windows": {
        "length_s": 60.0,
        "stride_s": 60.0,
        "purity_min": 0.999,
        "guard_s": 5.0,
        "min_valid_fraction": 0.9,
        "drop_low_coverage": True,
    },
    "units": {
        "wrist_acc_scale": 0.015625,  # cuentas E4 -> g (1/64)
        "bvp_unit": "raw",
        "verify_ranges": True,
        "strict_ranges": False,
    },
    "qc": {
        "flatline_std_eps": 1e-9,
        "max_nan_fraction": 0.01,
        "saturation_quantile": 0.999,
        "max_zero_diff_fraction": 0.99,
        "saturation_fraction_threshold": 0.05,
        "min_duration_s": 60.0,
    },
    "features": {
        "version": "1.0",
        "enabled": [
            "ecg",
            "eda",
            "resp",
            "temp",
            "acc",
            "emg",
            "bvp",
            "wrist_eda",
            "wrist_temp",
            "wrist_acc",
        ],
        "min_window_duration_s": 10.0,
        "ecg": {
            "clean_method": "neurokit",
            "rpeak_method": "neurokit",
            "min_hr_bpm": 30.0,
            "max_hr_bpm": 220.0,
            "outlier_low_ms": 300.0,
            "outlier_high_ms": 2000.0,
            "min_beats": 10,
        },
        "eda": {
            "clean_method": "neurokit",
            "phasic_method": "highpass",
            "highpass_hz": 0.05,
            "min_scr_amplitude_us": 0.05,
        },
        "resp": {
            "clean_method": "khodadad2018",
            "min_cycles": 3,
        },
        "acc": {
            "motion_threshold_g": 0.05,
        },
        "bvp": {
            "clean_method": "elgendi",
            "peak_method": "elgendi",
            "outlier_low_ms": 300.0,
            "outlier_high_ms": 2000.0,
            "min_beats": 10,
        },
        "sqi": {
            "weights": {},
            "good_threshold": 0.8,
            "poor_threshold": 0.5,
        },
    },
    "representations": {
        "enabled": ["numeric", "text"],
        "numeric": {
            "round_digits": 4,
            "include_quality": True,
            "include_units": True,
        },
        "text": {
            "include_quality": True,
            "include_methods": False,
            "float_format": ".4g",
            "max_flags": 10,
        },
    },
    "persist": {
        "canonical_arrays": True,
        "compression": False,
        "windows_arrays": False,
    },
    "hashing": {
        "algorithm": "sha256",
        "chunk_size": 1048576,
    },
    "package": {
        "corpus_version": "0.1.0",
    },
    "experiment": {
        "minimal_channels": [],
        "degradation_conditions": [],
        "degradation_enabled": False,
    },
}


class IngestionConfigError(ValidationError):
    """Configuración de ingesta inválida."""


def get_ingestion_config() -> dict:
    """Devuelve la configuración de ingesta resuelta y validada."""
    defaults = getattr(processControl, "defaults", {}) or {}
    parms = getattr(processControl, "parms", {}) or {}

    raw = {}
    if isinstance(defaults, dict) and isinstance(defaults.get("ingestion"), dict):
        raw = deep_merge(raw, defaults["ingestion"])
    if isinstance(parms, dict) and isinstance(parms.get("ingestion"), dict):
        raw = deep_merge(raw, parms["ingestion"])

    cfg = deep_merge(copy.deepcopy(DEFAULTS), raw)
    validate_config(cfg)
    return cfg


def representations_config(cfg: Mapping[str, Any]) -> dict:
    """Normaliza ``representations`` admitiendo lista (formato antiguo) o dict."""
    raw = cfg.get("representations", {})
    if isinstance(raw, list):
        raw = {"enabled": raw}
    merged = deep_merge(copy.deepcopy(REPRESENTATION_DEFAULTS), raw or {})
    return merged


def validate_config(cfg: dict) -> None:
    """Valida los invariantes de la configuración de ingesta."""
    windows = cfg["windows"]
    require(float(windows["length_s"]) > 0, "windows.length_s debe ser > 0", where="ingestion")
    require(float(windows["stride_s"]) > 0, "windows.stride_s debe ser > 0", where="ingestion")
    require(
        float(windows["stride_s"]) <= float(windows["length_s"]),
        "windows.stride_s no puede ser mayor que windows.length_s",
        where="ingestion",
    )
    require(
        0.0 < float(windows["purity_min"]) <= 1.0,
        "windows.purity_min debe estar en (0, 1]",
        where="ingestion",
    )
    require(float(windows["guard_s"]) >= 0, "windows.guard_s debe ser >= 0", where="ingestion")
    require(
        0.0 <= float(windows["min_valid_fraction"]) <= 1.0,
        "windows.min_valid_fraction debe estar en [0, 1]",
        where="ingestion",
    )

    qc = cfg.get("qc", {})
    for key in ("max_nan_fraction", "max_zero_diff_fraction", "saturation_fraction_threshold"):
        value = float(qc.get(key, 0.0))
        require(0.0 <= value <= 1.0, f"qc.{key} debe estar en [0, 1]", where="ingestion")

    keep = cfg["labels"]["keep"]
    require(isinstance(keep, list) and bool(keep), "labels.keep debe ser una lista no vacía", where="ingestion")
    require(all(isinstance(v, int) for v in keep), "labels.keep debe contener enteros", where="ingestion")

    representations = representations_config(cfg)
    enabled = representations.get("enabled")
    require(
        isinstance(enabled, list) and bool(enabled),
        "representations.enabled debe ser una lista no vacía",
        where="ingestion",
    )
    require(
        all(item in ("numeric", "text", "embeddings") for item in enabled),
        f"representaciones no soportadas: {enabled}",
        where="ingestion",
    )

    package = cfg.get("package", {})
    require(
        isinstance(package.get("corpus_version"), str) and bool(package["corpus_version"].strip()),
        "package.corpus_version debe ser una cadena no vacía",
        where="ingestion",
    )

    rates = cfg["sampling_rates"]
    for name in ("label", "chest", "wrist_acc", "wrist_bvp", "wrist_eda", "wrist_temp"):
        require(
            isinstance(rates.get(name), (int, float)) and float(rates[name]) > 0,
            f"sampling_rates.{name} debe ser > 0",
            where="ingestion",
        )


def requested_subjects(cli_value: Optional[str], cfg: dict) -> Optional[list[str]]:
    """Devuelve el subconjunto de sujetos pedido por CLI/config.

    Returns:
        ``None`` si se deben procesar todos los sujetos descubiertos.
    """
    if cli_value:
        parts = [p.strip() for p in str(cli_value).split(",") if p.strip()]
        return parts or None
    configured = cfg.get("subjects", "auto")
    if configured in (None, "", "auto", "all"):
        return None
    if isinstance(configured, str):
        parts = [p.strip() for p in configured.split(",") if p.strip()]
        return parts or None
    if isinstance(configured, (list, tuple)):
        return [str(p) for p in configured] or None
    raise IngestionConfigError(f"subjects no válido: {configured!r}", where="ingestion")


def filter_available(
    requested: Optional[Iterable[str]],
    available: Iterable[str],
) -> list[str]:
    """Filtra/ordena los sujetos pedidos contra los disponibles.

    Mantiene el orden de ``available`` (orden natural del dataset) y falla si
    se pide un sujeto inexistente.
    """
    available = list(available)
    if requested is None:
        return available
    requested = [str(s) for s in requested]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise IngestionConfigError(
            f"sujetos no encontrados en el dataset: {missing}. Disponibles: {available}",
            where="ingestion",
        )
    wanted = set(requested)
    return [s for s in available if s in wanted]
