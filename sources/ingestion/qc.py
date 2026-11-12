"""
Control de calidad (QC) por sujeto y modalidad.

El objetivo es **describir** la calidad sin descartar datos en silencio. Cada
canal produce estadísticos y flags; el paso ``window`` decide después, con una
política explícita, si una ventana es utilizable.

Métricas por canal:

- cobertura: fracción de muestras finitas (NaN/inf);
- rango observado y violaciones de rango fisiológico (de ``units``);
- variabilidad, cuantiles, curtosis y asimetría;
- ``zero_diff_fraction``: proporción de diferencias nulas (sensor atascado);
- ``railing_fraction``: proporción de muestras en el mínimo/máximo (saturación).

Los flags no implican exclusión automática.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sources.common.common import logger, writeLog
from sources.common.paths import atomic_write_json, corpus_dir, ensure_dir, read_json
from sources.common.provenance import (
    build_fingerprint,
    code_fingerprint,
    sha256_json,
    utc_now,
)
from sources.common.validation import ValidationError, require
from sources.ingestion import loader
from sources.ingestion.schemas import SCHEMA_VERSION

#: Código que produce el informe QC (sin el historial de git).
_CODE_SCOPE: tuple[str, ...] = ("sources/ingestion/qc.py",)

CHANNEL_ORDER = (
    "chest.ACC",
    "chest.ECG",
    "chest.EMG",
    "chest.EDA",
    "chest.Temp",
    "chest.Resp",
    "wrist.ACC",
    "wrist.BVP",
    "wrist.EDA",
    "wrist.TEMP",
)


def qc_dir() -> Path:
    return corpus_dir() / "qc"


def qc_path(subject: str) -> Path:
    return qc_dir() / f"{subject}.qc.json"


def qc_fingerprint(meta: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    """Fingerprint del informe QC de un sujeto."""
    return build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_sha256": meta.get("canonical_sha256"),
            "qc": cfg.get("qc"),
            "sampling_rates": cfg.get("sampling_rates"),
            "code": code_fingerprint(_CODE_SCOPE),
        }
    )


def _as_series(values: np.ndarray) -> np.ndarray:
    """Reduce un canal 2D (ACC) a su magnitud; deja los 1D intactos."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return np.sqrt((arr ** 2).sum(axis=1))


def compute_channel_qc(
    name: str,
    values: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    range_violations: int = 0,
) -> dict:
    """Calcula el informe QC de un canal."""
    qc_cfg = cfg.get("qc", {})
    series = _as_series(values)

    n = int(series.size)
    nan_count = int(np.count_nonzero(np.isnan(series)))
    inf_count = int(np.count_nonzero(np.isinf(series)))
    finite = series[np.isfinite(series)]
    require(finite.size > 0, "canal sin valores finitos", where=f"qc.{name}")

    mean = float(finite.mean())
    std = float(finite.std())
    vmin = float(finite.min())
    vmax = float(finite.max())
    percentiles = {
        str(p): float(v)
        for p, v in zip((1, 5, 50, 95, 99), np.percentile(finite, (1, 5, 50, 95, 99)))
    }

    diffs = np.diff(finite)
    zero_diff_fraction = float(np.count_nonzero(diffs == 0) / diffs.size) if diffs.size else 0.0

    if vmin == vmax:
        railing_fraction = 1.0
    else:
        railing_fraction = float(
            (np.count_nonzero(finite == vmin) + np.count_nonzero(finite == vmax)) / finite.size
        )

    kurtosis = skewness = None
    if std > 0 and finite.size > 3:
        try:
            from scipy import stats as scipy_stats

            kurtosis = float(scipy_stats.kurtosis(finite, fisher=True, bias=False))
            skewness = float(scipy_stats.skew(finite, bias=False))
        except Exception:
            kurtosis = skewness = None

    flags: list[str] = []
    nan_fraction = nan_count / n if n else 0.0
    if nan_fraction > float(qc_cfg.get("max_nan_fraction", 0.01)):
        flags.append("high_nan")
    if inf_count > 0:
        flags.append("infinite")
    if std <= float(qc_cfg.get("flatline_std_eps", 1e-9)):
        flags.append("constant")
    if zero_diff_fraction >= float(qc_cfg.get("max_zero_diff_fraction", 0.99)):
        flags.append("flatline")
    if railing_fraction >= float(qc_cfg.get("saturation_fraction_threshold", 0.05)):
        flags.append("saturated")
    if range_violations > 0:
        flags.append("range_violation")

    return {
        "channel": name,
        "n_samples": n,
        "nan_count": nan_count,
        "nan_fraction": float(nan_fraction),
        "inf_count": inf_count,
        "mean": mean,
        "std": std,
        "min": vmin,
        "max": vmax,
        "percentiles": percentiles,
        "zero_diff_fraction": zero_diff_fraction,
        "railing_fraction": railing_fraction,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "range_violations": int(range_violations),
        "flags": flags,
    }


def compute_subject_qc(
    subject: str,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict:
    """Construye el informe QC completo de un sujeto."""
    violations = {
        report.get("channel"): int(report.get("violations", 0))
        for report in meta.get("unit_reports", [])
    }

    channels: dict[str, dict] = {}
    for key, values in arrays.items():
        if key == "label":
            continue
        modality, channel = key.split("_", 1)
        name = f"{modality}.{channel}"
        channels[name] = compute_channel_qc(
            name,
            values,
            cfg,
            range_violations=violations.get(name, 0),
        )

    label = np.asarray(arrays["label"])
    values, counts = np.unique(label, return_counts=True)
    total = int(label.size)
    label_counts = {str(int(v)): int(c) for v, c in zip(values, counts)}
    label_fractions = {str(int(v)): float(c / total) for v, c in zip(values, counts)}

    flags = [
        f"{channel}:{flag}"
        for channel, report in channels.items()
        for flag in report["flags"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "created_at": utc_now(),
        "fingerprint": qc_fingerprint(meta, cfg),
        "config_fingerprint": sha256_json({"qc": cfg.get("qc"), "schema_version": SCHEMA_VERSION}),
        "n_samples": int(meta.get("n_samples", total)),
        "duration_s": float(meta.get("duration_s", 0.0)),
        "channels": {name: channels[name] for name in CHANNEL_ORDER if name in channels},
        "labels": {
            "counts": label_counts,
            "fractions": label_fractions,
            "transitions": int(np.count_nonzero(np.diff(label) != 0)) if label.size > 1 else 0,
        },
        "flags": flags,
        "quality_ok": not flags,
        "summary": {
            "n_channels": len(channels),
            "n_flagged": sum(1 for report in channels.values() if report["flags"]),
            "n_flags": len(flags),
        },
    }


def run_qc(cfg: Mapping[str, Any], subject: str, *, force: bool = False) -> dict:
    """Ejecuta el QC de un sujeto y escribe ``qc/{SX}.qc.json``."""
    _, meta_path = loader.canonical_paths(subject)
    meta = read_json(meta_path)
    if meta is None:
        raise ValidationError(
            f"no existe el canónico de {subject}; ejecute primero '--step ingest'",
            where="qc",
        )

    fingerprint = qc_fingerprint(meta, cfg)
    target = qc_path(subject)
    previous = read_json(target)
    if not force and previous is not None and previous.get("fingerprint") == fingerprint:
        writeLog("info", logger, f"[qc] {subject}: informe reutilizado (fingerprint {fingerprint[:12]}…)")
        return {
            "subject": subject,
            "status": "reused",
            "fingerprint": fingerprint,
            "n_flags": len(previous.get("flags", [])),
        }

    arrays = loader.load_canonical(subject)
    report = compute_subject_qc(subject, arrays, meta, cfg)
    ensure_dir(qc_dir())
    atomic_write_json(target, report)

    writeLog(
        "info",
        logger,
        f"[qc] {subject}: {report['summary']['n_flags']} flags, "
        f"{report['summary']['n_flagged']}/{report['summary']['n_channels']} canales marcados",
    )
    return {
        "subject": subject,
        "status": "ok",
        "fingerprint": fingerprint,
        "n_flags": len(report["flags"]),
        "report": str(target),
    }
