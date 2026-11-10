"""Adaptador de PhysioNet/CinC 2017 (ECG de una derivación, 4 clases).

Control de generalidad de la tesis: un dataset con **etiquetas de experto** y
una clase **"Too Noisy to Process"** que representa evidencia naturalmente
inutilizable (ideal para H4/H5). No tiene afecto ni EDA/respiración: solo ECG.

Diseño:

- Cada registro (9–61 s, 300 Hz) es una ventana; no hay línea base por sujeto,
  así que la dirección de los claims se define respecto a una **referencia
  poblacional** calculada solo con registros de desarrollo.
- Split determinista por CRC32 del identificador: 30 % desarrollo (referencia de
  estandarización y modelo de referencia), 70 % evaluación.
- Features: las de ECG del pipeline (HR, RR, SDNN, RMSSD, latidos), sin
  amplitud (depende de la ganancia del dispositivo).
"""

from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.io import loadmat

from sources.ingestion.features import ecg as ecg_features

#: Etiquetas de REFERENCE.csv -> clases del prompt.
LABEL_MAP: dict[str, str] = {
    "N": "normal",
    "A": "af",
    "O": "other-rhythm",
    "~": "noisy",
}

#: Clases en orden canónico.
RHYTHM_CLASSES: tuple[str, ...] = ("normal", "af", "other-rhythm", "noisy")

#: Features canónicas que usamos del extractor de ECG.
CANONICAL_FEATURES: tuple[str, ...] = (
    "ecg_hr_bpm",
    "ecg_rr_mean_ms",
    "ecg_sdnn_ms",
    "ecg_rmssd_ms",
    "ecg_n_beats",
)

#: Configuración mínima del extractor de ECG (defaults del proyecto).
ECG_CFG: dict[str, Any] = {
    "features": {
        "ecg": {
            "clean_method": "neurokit",
            "rpeak_method": "neurokit",
            "min_hr_bpm": 30.0,
            "max_hr_bpm": 220.0,
            "outlier_low_ms": 300.0,
            "outlier_high_ms": 2000.0,
            "min_beats": 10,
        }
    }
}

UNIT: dict[str, str] = {
    "ecg_hr_bpm": "bpm",
    "ecg_rr_mean_ms": "ms",
    "ecg_sdnn_ms": "ms",
    "ecg_rmssd_ms": "ms",
    "ecg_n_beats": "count",
}


@dataclass(frozen=True)
class RhythmRecord:
    """Registro de PhysioNet 2017 con sus features."""

    record_id: str
    label: str
    fs: float
    duration_s: float
    values: dict[str, float | None]


def load_reference(path: Path) -> dict[str, str]:
    """Carga ``REFERENCE.csv`` -> ``record_id -> label`` (canónica)."""
    labels: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            raw = row[1].strip()
            if raw in LABEL_MAP:
                labels[row[0].strip()] = LABEL_MAP[raw]
    return labels


def read_record(record_id: str, data_dir: Path) -> tuple[np.ndarray, float]:
    """Lee un registro (``.mat``) y devuelve la señal y la frecuencia de muestreo."""
    data = loadmat(Path(data_dir) / f"{record_id}.mat")
    signal = np.asarray(data["val"], dtype=float).reshape(-1)
    header = (Path(data_dir) / f"{record_id}.hea").read_text().splitlines()[0]
    fs = float(header.split()[2])
    return signal, fs


def extract_features(signal: np.ndarray, fs: float) -> dict[str, float | None]:
    """Extrae las features canónicas de ECG de un registro."""
    duration_s = signal.size / fs
    features, _quality = ecg_features.extract(
        signal,
        fs=fs,
        cfg=ECG_CFG,
        coverage=1.0,
        duration_s=duration_s,
    )
    values: dict[str, float | None] = {}
    for name in CANONICAL_FEATURES:
        entry = features.get(name) if isinstance(features, Mapping) else None
        values[name] = entry.get("value") if isinstance(entry, Mapping) else None
    return values


def select_records(
    labels: Mapping[str, str],
    per_class: int = 150,
    seed: int = 42,
) -> list[str]:
    """Muestreo estratificado por clase (determinista)."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[str]] = {label: [] for label in RHYTHM_CLASSES}
    for record_id, label in sorted(labels.items()):
        if label in by_class:
            by_class[label].append(record_id)
    selected: list[str] = []
    for label in RHYTHM_CLASSES:
        candidates = by_class[label]
        if not candidates:
            continue
        take = min(per_class, len(candidates))
        indices = rng.choice(len(candidates), size=take, replace=False)
        selected.extend(candidates[i] for i in sorted(indices))
    return sorted(selected)


def _is_development(record_id: str, dev_fraction: float) -> bool:
    return (zlib.crc32(record_id.encode()) % 1000) / 1000.0 < dev_fraction


def split_records(
    record_ids: Sequence[str],
    dev_fraction: float = 0.3,
) -> tuple[list[str], list[str]]:
    """Split determinista desarrollo/evaluación por CRC32 del identificador."""
    dev = [r for r in record_ids if _is_development(r, dev_fraction)]
    test = [r for r in record_ids if r not in set(dev)]
    return dev, test


def population_reference(
    records: Sequence[RhythmRecord],
) -> tuple[dict[str, float], dict[str, float]]:
    """Mediana e IQR por feature sobre los registros de desarrollo."""
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in CANONICAL_FEATURES:
        values = np.asarray(
            [r.values[name] for r in records if r.values.get(name) is not None],
            dtype=float,
        )
        if values.size == 0:
            centers[name], scales[name] = 0.0, 1.0
            continue
        centers[name] = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        scales[name] = float(q3 - q1) or float(np.std(values)) or 1.0
    return centers, scales


def z_scores(
    values: Mapping[str, float | None],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
    clip: float = 3.0,
) -> dict[str, float | None]:
    """Convierte a z poblacional (referencia de desarrollo)."""
    scores: dict[str, float | None] = {}
    for name in CANONICAL_FEATURES:
        value = values.get(name)
        if value is None:
            scores[name] = None
        else:
            scores[name] = float(
                np.clip(
                    (value - centers.get(name, 0.0)) / scales.get(name, 1.0),
                    -clip,
                    clip,
                )
            )
    return scores


def build_payload(
    record_id: str,
    scores: Mapping[str, float | None],
    duration_s: float,
) -> dict[str, Any]:
    """Payload ``numeric_llm`` con z poblacional y solo la modalidad ECG."""
    modalities: dict[str, dict[str, Any]] = {"ecg": {}}
    for name in CANONICAL_FEATURES:
        modalities["ecg"][name] = {
            "value": scores.get(name),
            "unit": "SD",
            "kind": "derived",
            "method": "population_zscore_from_development",
        }
    return {
        "modalities": modalities,
        "feature_version": "physionet2017-population-z",
        "window": {
            "window_id": "win_" + format(zlib.crc32(record_id.encode()) & 0xFFFFFFFF, "08x"),
            "duration_s": duration_s,
        },
    }
