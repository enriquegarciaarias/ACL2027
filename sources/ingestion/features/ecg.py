"""Características ECG: frecuencia cardiaca y HRV en dominio temporal.

No se calculan métricas frecuenciales (LF/HF) porque una ventana de 60 s no
tiene resolución suficiente para estimarlas de forma fiable. Esa decisión es
metodológica y debe documentarse en cualquier análisis posterior.
"""
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
    p = cfg["features"]["ecg"]
    ecg = np.asarray(values, dtype=float).reshape(-1)

    params = {
        "sampling_rate": float(fs),
        "clean_method": p["clean_method"],
        "rpeak_method": p["rpeak_method"],
        "outlier_low_ms": float(p["outlier_low_ms"]),
        "outlier_high_ms": float(p["outlier_high_ms"]),
        "min_beats": int(p["min_beats"]),
    }
    builder = FeatureBuilder("neurokit2.ecg_peaks", nk.__version__, params)
    flags: list[str] = []

    clean = nk.ecg_clean(ecg, sampling_rate=fs, method=p["clean_method"])
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method=p["rpeak_method"])
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    n_beats = int(peaks.size)

    builder.add("ecg_n_beats", n_beats, "count", kind="observed")
    builder.add("ecg_amplitude_std_mv", float(np.std(clean)), "mV", kind="observed")

    hr = rr_mean = sdnn = rmssd = pnn50 = rr_min = rr_max = outlier = None
    mean_quality = 0.0

    if n_beats >= 2:
        rr = np.diff(peaks) / float(fs) * 1000.0
        valid = rr[(rr >= float(p["outlier_low_ms"])) & (rr <= float(p["outlier_high_ms"]))]
        outlier = float(1.0 - valid.size / rr.size) if rr.size else None

        if valid.size >= 1:
            rr_mean = float(valid.mean())
            hr = float(60000.0 / rr_mean)
            rr_min = float(valid.min())
            rr_max = float(valid.max())
        if valid.size >= 2:
            sdnn = float(valid.std(ddof=1))
        if valid.size >= 3:
            diffs = np.diff(valid)
            rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0)

        try:
            quality = nk.ecg_quality(
                clean, rpeaks=peaks, sampling_rate=fs, method="averageQRS"
            )
            quality = np.asarray(quality, dtype=float)
            mean_quality = float(np.nanmean(quality)) if quality.size else 0.0
        except Exception:
            mean_quality = 0.0

    if n_beats < int(p["min_beats"]):
        flags.append("few_beats")
    if outlier is not None and outlier > 0.10:
        flags.append("rr_outliers")

    builder.add("ecg_hr_bpm", hr, "bpm")
    builder.add("ecg_rr_mean_ms", rr_mean, "ms")
    builder.add("ecg_sdnn_ms", sdnn, "ms")
    builder.add("ecg_rmssd_ms", rmssd, "ms")
    builder.add("ecg_pnn50_pct", pnn50, "%")
    builder.add("ecg_rr_min_ms", rr_min, "ms")
    builder.add("ecg_rr_max_ms", rr_max, "ms")
    builder.add("ecg_rr_outlier_fraction", outlier, "ratio")

    sqi = coverage * max(0.0, min(1.0, mean_quality))
    if outlier is not None:
        sqi *= (1.0 - outlier)
    if n_beats < int(p["min_beats"]):
        sqi *= 0.5

    return builder.features, modality_quality(
        sqi,
        flags,
        n_beats=n_beats,
        mean_quality=mean_quality,
        rr_outlier_fraction=outlier,
    )
