"""Características de acelerometría: actividad y movimiento.

La señal de entrada puede ser 1D o 2D (3 ejes). En el segundo caso se calcula
la magnitud vectorial ``sqrt(x² + y² + z²)`` en unidades de g.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from sources.ingestion.features.common import FeatureBuilder
from sources.ingestion.features.sqi import modality_quality


def _magnitude(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.sqrt((arr ** 2).sum(axis=1))


def extract(
    values: np.ndarray,
    *,
    fs: float,
    cfg: Mapping[str, Any],
    coverage: float = 1.0,
    duration_s: Optional[float] = None,
) -> tuple[dict, dict]:
    p = cfg["features"]["acc"]
    magnitude = _magnitude(values)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        raise ValueError("ACC sin valores finitos")

    builder = FeatureBuilder(
        "vector_magnitude+descriptive",
        "1.0",
        {
            "sampling_rate": float(fs),
            "motion_threshold_g": float(p["motion_threshold_g"]),
        },
    )
    flags: list[str] = []

    mean = float(finite.mean())
    std = float(finite.std())
    median = float(np.median(finite))
    motion_fraction = float(np.mean(np.abs(finite - median) > float(p["motion_threshold_g"])))

    builder.add("acc_mean_g", mean, "g", kind="observed")
    builder.add("acc_std_g", std, "g", kind="observed")
    builder.add("acc_p95_g", float(np.percentile(finite, 95)), "g", kind="observed")
    builder.add("acc_energy_g2", float(np.mean(finite ** 2)), "g^2")
    builder.add("acc_motion_fraction", motion_fraction, "ratio")

    sqi = coverage
    if std <= 1e-12:
        sqi *= 0.5
        flags.append("constant")

    return builder.features, modality_quality(sqi, flags, mean_g=mean, std_g=std)
