"""Características BVP/PPG de muñeca: pulso y HRV básica."""
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
    p = cfg["features"]["bvp"]
    bvp = np.asarray(values, dtype=float).reshape(-1)

    params = {
        "sampling_rate": float(fs),
        "clean_method": p["clean_method"],
        "peak_method": p["peak_method"],
        "outlier_low_ms": float(p["outlier_low_ms"]),
        "outlier_high_ms": float(p["outlier_high_ms"]),
        "min_beats": int(p["min_beats"]),
    }
    builder = FeatureBuilder("neurokit2.ppg_peaks", nk.__version__, params)
    flags: list[str] = []

    clean = nk.ppg_clean(bvp, sampling_rate=fs, method=p["clean_method"])
    _, info = nk.ppg_peaks(clean, sampling_rate=fs, method=p["peak_method"])
    peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
    n_beats = int(peaks.size)

    builder.add("bvp_n_beats", n_beats, "count", kind="observed")
    builder.add("bvp_amplitude_std", float(np.std(clean)), "raw", kind="observed")

    hr = rr_mean = rmssd = outlier = None
    mean_quality = 0.0

    if n_beats >= 2:
        rr = np.diff(peaks) / float(fs) * 1000.0
        valid = rr[(rr >= float(p["outlier_low_ms"])) & (rr <= float(p["outlier_high_ms"]))]
        outlier = float(1.0 - valid.size / rr.size) if rr.size else None
        if valid.size >= 1:
            rr_mean = float(valid.mean())
            hr = float(60000.0 / rr_mean)
        if valid.size >= 3:
            rmssd = float(np.sqrt(np.mean(np.diff(valid) ** 2)))

        try:
            quality = nk.ppg_quality(
                clean, peaks=peaks, sampling_rate=fs, method="templatematch"
            )
            quality = np.asarray(quality, dtype=float)
            mean_quality = float(np.nanmean(quality)) if quality.size else 0.0
        except Exception:
            mean_quality = 0.5

    if n_beats < int(p["min_beats"]):
        flags.append("few_beats")
    if outlier is not None and outlier > 0.10:
        flags.append("rr_outliers")

    builder.add("bvp_hr_bpm", hr, "bpm")
    builder.add("bvp_rr_mean_ms", rr_mean, "ms")
    builder.add("bvp_rmssd_ms", rmssd, "ms")
    builder.add("bvp_rr_outlier_fraction", outlier, "ratio")

    sqi = coverage * max(0.0, min(1.0, mean_quality))
    if outlier is not None:
        sqi *= (1.0 - outlier)
    if n_beats < int(p["min_beats"]):
        sqi *= 0.5

    return builder.features, modality_quality(
        sqi, flags, n_beats=n_beats, mean_quality=mean_quality
    )
