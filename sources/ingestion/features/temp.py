"""Características de temperatura: nivel, variabilidad y tendencia."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from sources.ingestion.features.common import FeatureBuilder, linear_slope
from sources.ingestion.features.sqi import modality_quality


def extract(
    values: np.ndarray,
    *,
    fs: float,
    cfg: Mapping[str, Any],
    coverage: float = 1.0,
    duration_s: Optional[float] = None,
) -> tuple[dict, dict]:
    temp = np.asarray(values, dtype=float).reshape(-1)
    finite = temp[np.isfinite(temp)]
    if finite.size == 0:
        raise ValueError("temperatura sin valores finitos")

    builder = FeatureBuilder(
        "descriptive+linear_regression",
        "1.0",
        {"sampling_rate": float(fs), "slope_unit": "degC/s"},
    )
    flags: list[str] = []

    mean = float(finite.mean())
    std = float(finite.std())
    builder.add("temp_mean_degc", mean, "degC", kind="observed")
    builder.add("temp_std_degc", std, "degC", kind="observed")
    builder.add("temp_slope_degc_per_s", linear_slope(temp, fs), "degC/s", kind="observed")
    builder.add("temp_range_degc", float(finite.max() - finite.min()), "degC", kind="observed")

    sqi = coverage
    if std <= 1e-12:
        # Una temperatura cuantizada puede ser constante en 60 s; se penaliza
        # suavemente, no se considera señal inválida.
        sqi *= 0.8
        flags.append("constant")

    return builder.features, modality_quality(sqi, flags, mean_degc=mean, std_degc=std)
