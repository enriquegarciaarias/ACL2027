"""Registro de extractores por modalidad.

Cada modalidad declara qué array del canónico consume, a qué frecuencia,
qué rango de la ventana usar y qué extractor aplica. Añadir una modalidad
nueva solo requiere un extractor y una entrada en ``MODALITIES``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from sources.ingestion.features import acc, bvp, ecg, eda, emg, resp, sqi, temp


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    array_key: str
    fs_key: str
    range_key: str
    coverage_key: str
    extractor: Callable


MODALITIES: tuple[ModalitySpec, ...] = (
    ModalitySpec("ecg", "chest_ECG", "chest", "chest", "chest", ecg.extract),
    ModalitySpec("eda", "chest_EDA", "chest", "chest", "chest", eda.extract),
    ModalitySpec("resp", "chest_Resp", "chest", "chest", "chest", resp.extract),
    ModalitySpec("temp", "chest_Temp", "chest", "chest", "chest", temp.extract),
    ModalitySpec("acc", "chest_ACC", "chest", "chest", "chest", acc.extract),
    ModalitySpec("emg", "chest_EMG", "chest", "chest", "chest", emg.extract),
    ModalitySpec("bvp", "wrist_BVP", "wrist_bvp", "wrist_bvp", "wrist_bvp", bvp.extract),
    ModalitySpec("wrist_eda", "wrist_EDA", "wrist_eda", "wrist_eda", "wrist_eda", eda.extract),
    ModalitySpec("wrist_temp", "wrist_TEMP", "wrist_temp", "wrist_temp", "wrist_temp", temp.extract),
    ModalitySpec("wrist_acc", "wrist_ACC", "wrist_acc", "wrist_acc", "wrist_acc", acc.extract),
)


def extract_window(
    arrays: Mapping[str, np.ndarray],
    window: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Extrae todas las modalidades habilitadas de una ventana.

    Un fallo en una modalidad no invalida el resto: se registra un SQI de 0 con
    flag ``extractor_error`` y se continúa.
    """
    enabled = set(cfg["features"].get("enabled", []))
    features: dict[str, dict] = {}
    quality: dict[str, dict] = {}

    for spec in MODALITIES:
        if spec.name not in enabled:
            continue
        start, end = window["ranges"][spec.range_key]
        values = np.asarray(arrays[spec.array_key][start:end])
        coverage = float(window.get("coverage", {}).get(spec.coverage_key, 1.0))
        fs = float(cfg["sampling_rates"][spec.fs_key])
        try:
            modality_features, modality_quality = spec.extractor(
                values,
                fs=fs,
                cfg=cfg,
                coverage=coverage,
                duration_s=window.get("duration_s"),
            )
        except Exception as exc:  # robustez: una modalidad no rompe la ventana
            modality_features = {}
            modality_quality = sqi.modality_quality(
                0.0,
                ["extractor_error"],
                error=f"{type(exc).__name__}: {exc}",
            )
        features[spec.name] = modality_features
        quality[spec.name] = modality_quality

    return features, sqi.combine(quality, cfg)


def modality_specs(cfg: Mapping[str, Any]) -> list[dict]:
    """Describe las modalidades activas (para ``feature_spec.json``)."""
    enabled = set(cfg["features"].get("enabled", []))
    return [
        {
            "name": spec.name,
            "array": spec.array_key,
            "fs_key": spec.fs_key,
            "range_key": spec.range_key,
            "extractor": spec.extractor.__module__,
        }
        for spec in MODALITIES
        if spec.name in enabled
    ]
