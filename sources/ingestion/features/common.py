"""Utilidades compartidas por los extractores de características."""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from sources.ingestion.schemas import FeatureValue

FEATURE_VERSION = "1.0"


def _json_scalar(value: Any) -> Any:
    """Convierte tipos numpy a tipos JSON y normaliza NaN/inf a ``None``."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    return value


class FeatureBuilder:
    """Acumula características con una procedencia común.

    Evita repetir ``method``/``method_version``/``params`` en cada feature.
    """

    def __init__(self, method: str, method_version: str, params: Optional[dict] = None):
        self.method = method
        self.method_version = str(method_version)
        self.params = dict(params or {})
        self.features: dict[str, dict] = {}

    def add(
        self,
        name: str,
        value: Any,
        unit: str,
        *,
        kind: str = "derived",
        method: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> None:
        # Una feature con valor None no se pudo calcular: no es un cero real,
        # es una ausencia de evidencia. Se marca como ``error`` para que el
        # consumidor (LLM, evaluador) pueda distinguirla y abstenerse si es
        # necesario. Si el llamador pasa un ``kind`` explícito, se respeta.
        if value is None and kind == "derived":
            kind = "error"
        self.features[name] = FeatureValue(
            value=_json_scalar(value),
            unit=unit,
            method=method or self.method,
            method_version=self.method_version,
            params=dict(params if params is not None else self.params),
            kind=kind,
        ).to_dict()


def finite(values: np.ndarray) -> np.ndarray:
    """Devuelve solo los valores finitos de un array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def linear_slope(values: np.ndarray, fs: float) -> Optional[float]:
    """Pendiente de una regresión lineal (unidades de la señal por segundo)."""
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if series.size < 2:
        return None
    t = np.arange(series.size, dtype=np.float64) / float(fs)
    return float(np.polyfit(t, series, 1)[0])


def clip01(value: float) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def duration_from(values: np.ndarray, fs: float) -> float:
    return float(np.asarray(values).shape[0] / float(fs))
