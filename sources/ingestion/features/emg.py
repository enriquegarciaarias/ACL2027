"""Características EMG: amplitud y energía muscular."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from sources.ingestion.features.common import FeatureBuilder
from sources.ingestion.features.sqi import modality_quality


def extract(
    values: np.ndarray,
    *,
    fs: float,
    cfg: Mapping[str, Any],
    coverage: float = 1.0,
    duration_s: Optional[float] = None,
) -> tuple[dict, dict]:
    emg = np.asarray(values, dtype=float).reshape(-1)
    finite = emg[np.isfinite(emg)]
    if finite.size == 0:
        raise ValueError("EMG sin valores finitos")

    builder = FeatureBuilder(
        "descriptive", "1.0", {"sampling_rate": float(fs)}
    )
    flags: list[str] = []

    absolute = np.abs(finite)
    std = float(finite.std())
    builder.add("emg_mean_abs_mv", float(absolute.mean()), "mV", kind="observed")
    builder.add("emg_std_mv", std, "mV", kind="observed")
    builder.add("emg_rms_mv", float(np.sqrt(np.mean(finite ** 2))), "mV")
    builder.add("emg_p95_abs_mv", float(np.percentile(absolute, 95)), "mV", kind="observed")

    sqi = coverage
    if std <= 1e-12:
        sqi *= 0.5
        flags.append("constant")

    return builder.features, modality_quality(sqi, flags, std_mv=std)
