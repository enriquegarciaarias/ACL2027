"""
Normalización y verificación de unidades de WESAD.

Hechos verificados sobre el dataset:

- ``chest`` (RespiBAN) ya viene en unidades físicas: ECG/EMG en mV, EDA en µS,
  Temp en °C, Resp en % y ACC en g.
- ``wrist`` (Empatica E4): EDA en µS y TEMP en °C, pero ``ACC`` viene en
  cuentas de 1/64 g y ``BVP`` en crudo.

Este módulo aplica únicamente conversiones documentadas y registra un informe
de verificación por canal. La verificación **no descarta** datos por defecto:
marca violaciones para que el paso ``qc`` decida con una política explícita.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from sources.common.validation import ValidationError, range_report
from sources.ingestion.schemas import UnitReport

# (modalidad, canal) -> (min, max, unidad)
CHANNEL_RANGES: dict[tuple[str, str], tuple[float, float, str]] = {
    ("chest", "ECG"): (-10.0, 10.0, "mV"),
    ("chest", "EMG"): (-10.0, 10.0, "mV"),
    ("chest", "EDA"): (0.0, 100.0, "uS"),
    ("chest", "Temp"): (15.0, 45.0, "degC"),
    ("chest", "Resp"): (-100.0, 100.0, "%"),
    ("chest", "ACC"): (-16.0, 16.0, "g"),
    ("wrist", "ACC"): (-16.0, 16.0, "g"),
    ("wrist", "BVP"): (-1.0e6, 1.0e6, "raw"),
    ("wrist", "EDA"): (0.0, 100.0, "uS"),
    ("wrist", "TEMP"): (15.0, 45.0, "degC"),
}

# Unidades declaradas para cada canal (se registran en el artefacto canónico).
CHANNEL_UNITS: dict[tuple[str, str], str] = {
    key: value[2] for key, value in CHANNEL_RANGES.items()
}


def to_float32(array: Any, *, name: str) -> np.ndarray:
    """Convierte a float32 conservando la forma (sin colapsar canales)."""
    arr = np.asarray(array)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim not in (1, 2):
        raise ValidationError(f"forma no soportada: {arr.shape}", where=name)
    return np.ascontiguousarray(arr, dtype=np.float32)


def normalize_units(
    signal: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], list[UnitReport]]:
    """Convierte y verifica las unidades de todas las modalidades.

    Args:
        signal: diccionario ``{"chest": {...}, "wrist": {...}}`` del ``.pkl``.
        cfg: configuración de ingesta (sección ``units`` y ``sampling_rates``).

    Returns:
        (arrays, informes): arrays normalizados a float32 y un informe de
        verificación por canal.
    """
    units_cfg = cfg.get("units", {})
    scale = float(units_cfg.get("wrist_acc_scale", 0.015625))
    verify = bool(units_cfg.get("verify_ranges", True))
    strict = bool(units_cfg.get("strict_ranges", False))

    normalized: dict[str, dict[str, np.ndarray]] = {}
    reports: list[UnitReport] = []

    for modality in ("chest", "wrist"):
        if modality not in signal:
            raise ValidationError(f"falta la modalidad '{modality}'", where="WESAD.signal")
        normalized[modality] = {}
        for channel, values in signal[modality].items():
            key = (modality, channel)
            if key not in CHANNEL_RANGES:
                raise ValidationError(
                    f"canal no reconocido '{modality}.{channel}'", where="WESAD.signal"
                )

            arr = to_float32(values, name=f"{modality}.{channel}")

            if modality == "wrist" and channel == "ACC":
                arr = arr * np.float32(scale)

            normalized[modality][channel] = arr

            if verify:
                lo, hi, unit = CHANNEL_RANGES[key]
                report = range_report(
                    f"{modality}.{channel}",
                    arr,
                    lo,
                    hi,
                    unit=unit,
                    strict=strict,
                )
                reports.append(UnitReport(**report))

    return normalized, reports
