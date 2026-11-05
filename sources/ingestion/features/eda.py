"""Características EDA: nivel tónico, componente fásica y SCR.

Se usa ``neurokit2`` para limpieza, descomposición tónica/fásica y detección de
respuestas de conductancia (SCR). Las amplitudes SCR se expresan en µS de la
componente fásica y deben interpretarse con cautela: la descomposición depende
del filtro y del umbral configurados.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import neurokit2 as nk

from sources.ingestion.features.common import FeatureBuilder, duration_from, linear_slope
from sources.ingestion.features.sqi import modality_quality


def extract(
    values: np.ndarray,
    *,
    fs: float,
    cfg: Mapping[str, Any],
    coverage: float = 1.0,
    duration_s: Optional[float] = None,
) -> tuple[dict, dict]:
    p = cfg["features"]["eda"]
    eda = np.asarray(values, dtype=float).reshape(-1)
    finite = eda[np.isfinite(eda)]
    if finite.size == 0:
        raise ValueError("EDA sin valores finitos")

    params = {
        "sampling_rate": float(fs),
        "clean_method": p["clean_method"],
        "phasic_method": p["phasic_method"],
        "highpass_hz": float(p["highpass_hz"]),
        "min_scr_amplitude_us": float(p["min_scr_amplitude_us"]),
    }
    builder = FeatureBuilder("neurokit2.eda_phasic", nk.__version__, params)
    flags: list[str] = []

    mean = float(finite.mean())
    std = float(finite.std())
    builder.add("eda_mean_us", mean, "uS", kind="observed")
    builder.add("eda_std_us", std, "uS", kind="observed")
    builder.add("eda_slope_us_per_s", linear_slope(eda, fs), "uS/s", kind="observed")

    tonic_mean = phasic_std = scr_count = scr_candidates = scr_rate = amp_mean = amp_max = None
    try:
        cleaned = nk.eda_clean(eda, sampling_rate=fs, method=p["clean_method"])
        phasic = nk.eda_phasic(
            cleaned,
            sampling_rate=fs,
            method=p["phasic_method"],
            cutoff=float(p["highpass_hz"]),
        )
        tonic = np.asarray(phasic["EDA_Tonic"], dtype=float)
        phasic_signal = np.asarray(phasic["EDA_Phasic"], dtype=float)
        tonic_mean = float(np.nanmean(tonic))
        phasic_std = float(np.nanstd(phasic_signal))

        _, peak_info = nk.eda_peaks(
            phasic_signal,
            sampling_rate=fs,
            method="neurokit",
            amplitude_min=float(p["min_scr_amplitude_us"]),
        )
        detected = np.asarray(peak_info.get("SCR_Peaks", []), dtype=int)
        amplitudes_all = np.asarray(peak_info.get("SCR_Amplitude", []), dtype=float)
        scr_candidates = int(detected.size)

        # Filtro explícito y auditable por amplitud de la componente fásica.
        # ``eda_peaks`` no siempre aplica ``amplitude_min`` con la semántica
        # esperada, así que el umbral se aplica aquí y queda en ``params``.
        if amplitudes_all.size == detected.size:
            mask = np.isfinite(amplitudes_all) & (
                amplitudes_all >= float(p["min_scr_amplitude_us"])
            )
            amplitudes = amplitudes_all[mask]
        else:
            amplitudes = amplitudes_all[np.isfinite(amplitudes_all)]
        scr_count = int(amplitudes.size)

        duration = float(duration_s) if duration_s else duration_from(eda, fs)
        scr_rate = float(scr_count / (duration / 60.0)) if duration > 0 else None

        if amplitudes.size:
            amp_mean = float(amplitudes.mean())
            amp_max = float(amplitudes.max())
    except Exception:
        flags.append("phasic_failed")

    builder.add("eda_tonic_mean_us", tonic_mean, "uS")
    builder.add("eda_phasic_std_us", phasic_std, "uS")
    builder.add("eda_scr_candidates", scr_candidates, "count", kind="observed")
    builder.add("eda_scr_count", scr_count, "count")
    builder.add("eda_scr_rate_per_min", scr_rate, "1/min")
    builder.add("eda_scr_amplitude_mean_us", amp_mean, "uS")
    builder.add("eda_scr_amplitude_max_us", amp_max, "uS")

    sqi = coverage
    if std <= 1e-12:
        sqi *= 0.5
        flags.append("constant")
    if "phasic_failed" in flags:
        sqi *= 0.5

    return builder.features, modality_quality(
        sqi,
        flags,
        scr_count=scr_count,
        mean_us=mean,
        std_us=std,
    )
