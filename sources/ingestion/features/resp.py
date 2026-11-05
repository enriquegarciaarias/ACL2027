"""Características de respiración: frecuencia, regularidad y amplitud."""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import neurokit2 as nk

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
    p = cfg["features"]["resp"]
    resp = np.asarray(values, dtype=float).reshape(-1)

    params = {
        "sampling_rate": float(fs),
        "clean_method": p["clean_method"],
        "min_cycles": int(p["min_cycles"]),
    }
    builder = FeatureBuilder("neurokit2.rsp_peaks", nk.__version__, params)
    flags: list[str] = []

    cleaned = nk.rsp_clean(resp, sampling_rate=fs, method=p["clean_method"])
    _, info = nk.rsp_peaks(cleaned, sampling_rate=fs)
    troughs = np.asarray(info.get("RSP_Troughs", []), dtype=int)

    n_cycles = int(max(0, troughs.size - 1))
    intervals = np.diff(troughs) / float(fs)
    rate = cv = None
    if intervals.size >= 1:
        rate = float(60.0 / np.mean(intervals))
    if intervals.size >= 2 and np.mean(intervals) > 0:
        cv = float(np.std(intervals, ddof=1) / np.mean(intervals))

    finite_clean = cleaned[np.isfinite(cleaned)]
    if finite_clean.size:
        p5, p95 = np.percentile(finite_clean, (5, 95))
        amplitude = float(p95 - p5)
        std = float(finite_clean.std())
    else:
        amplitude = std = None

    if n_cycles < int(p["min_cycles"]):
        flags.append("few_cycles")

    builder.add("resp_rate_bpm", rate, "1/min")
    builder.add("resp_interval_cv", cv, "ratio")
    builder.add("resp_amplitude_pct", amplitude, "%")
    builder.add("resp_std_pct", std, "%")
    builder.add("resp_n_cycles", n_cycles, "count", kind="observed")

    sqi = coverage
    if cv is not None:
        sqi *= max(0.3, 1.0 - min(cv, 1.0))
    else:
        sqi *= 0.5
    if n_cycles < int(p["min_cycles"]):
        sqi *= 0.5

    return builder.features, modality_quality(
        sqi, flags, n_cycles=n_cycles, rate_bpm=rate, interval_cv=cv
    )
