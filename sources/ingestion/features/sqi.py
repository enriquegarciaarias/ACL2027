"""Indicadores de calidad de señal (SQI) por modalidad y agregado.

El SQI es una variable **experimental** de calidad, no una validación clínica.
Cada extractor devuelve un valor en ``[0, 1]`` y una lista de flags. El
agregado es una media ponderada (pesos configurables, por defecto iguales) y
se publica junto a ``sqi_min`` y ``sqi_max`` para no ocultar modalidades malas.
"""
from __future__ import annotations

from typing import Mapping, Optional

from sources.ingestion.features.common import clip01


def modality_quality(
    sqi: float,
    flags: Optional[list[str]] = None,
    **extra,
) -> dict:
    """Construye el bloque de calidad de una modalidad."""
    payload = {
        "sqi": clip01(sqi),
        "flags": list(flags or []),
    }
    payload.update(extra)
    return payload


def combine(quality: Mapping[str, dict], cfg: Mapping) -> dict:
    """Agrega los SQI de todas las modalidades extraídas."""
    sqi_cfg = dict((cfg.get("features", {}) or {}).get("sqi", {}) or {})
    weights = dict(sqi_cfg.get("weights", {}) or {})
    good_threshold = float(sqi_cfg.get("good_threshold", 0.8))
    poor_threshold = float(sqi_cfg.get("poor_threshold", 0.5))

    available = {
        name: block
        for name, block in quality.items()
        if block.get("sqi") is not None
    }
    per_modality = {
        name: {
            "sqi": block.get("sqi"),
            "flags": list(block.get("flags", [])),
        }
        for name, block in quality.items()
    }
    flags = [
        f"{name}:{flag}"
        for name, block in quality.items()
        for flag in block.get("flags", [])
    ]

    if not available:
        return {
            "per_modality": per_modality,
            "sqi_overall": 0.0,
            "sqi_min": 0.0,
            "sqi_max": 0.0,
            "category": "poor",
            "flags": flags + ["no_modalities"],
        }

    total_weight = sum(float(weights.get(name, 1.0)) for name in available)
    values = [float(block["sqi"]) for block in available.values()]
    overall = sum(
        float(weights.get(name, 1.0)) * float(block["sqi"])
        for name, block in available.items()
    ) / total_weight if total_weight > 0 else 0.0

    if overall >= good_threshold:
        category = "good"
    elif overall >= poor_threshold:
        category = "fair"
    else:
        category = "poor"

    return {
        "per_modality": per_modality,
        "sqi_overall": clip01(overall),
        "sqi_min": clip01(min(values)),
        "sqi_max": clip01(max(values)),
        "category": category,
        "flags": flags,
    }
