"""Tests de normalización y verificación de unidades."""
from __future__ import annotations

import numpy as np
import pytest

from sources.common.validation import ValidationError
from sources.ingestion import units


def test_wrist_acc_is_scaled_from_counts_to_g():
    signal = {
        "chest": {
            "ECG": np.zeros((10, 1)),
            "EMG": np.zeros((10, 1)),
            "EDA": np.ones((10, 1)),
            "Temp": np.full((10, 1), 33.0),
            "Resp": np.zeros((10, 1)),
            "ACC": np.zeros((10, 3)),
        },
        "wrist": {
            "ACC": np.full((4, 3), 64.0),
            "BVP": np.zeros((8, 1)),
            "EDA": np.ones((4, 1)),
            "TEMP": np.full((4, 1), 34.0),
        },
    }
    cfg = {"units": {"wrist_acc_scale": 0.015625, "verify_ranges": True, "strict_ranges": False}}

    normalized, reports = units.normalize_units(signal, cfg)

    np.testing.assert_allclose(normalized["wrist"]["ACC"], 1.0)
    assert normalized["chest"]["ECG"].ndim == 1
    assert normalized["chest"]["ACC"].ndim == 2
    assert all(report.violations == 0 for report in reports)


def test_range_violations_are_reported_but_not_raised():
    signal = {
        "chest": {
            "ECG": np.zeros((10, 1)),
            "EMG": np.zeros((10, 1)),
            "EDA": np.full((10, 1), 500.0),  # fuera de rango
            "Temp": np.full((10, 1), 33.0),
            "Resp": np.zeros((10, 1)),
            "ACC": np.zeros((10, 3)),
        },
        "wrist": {
            "ACC": np.zeros((4, 3)),
            "BVP": np.zeros((8, 1)),
            "EDA": np.zeros((4, 1)),
            "TEMP": np.full((4, 1), 34.0),
        },
    }
    cfg = {"units": {"verify_ranges": True, "strict_ranges": False}}
    _, reports = units.normalize_units(signal, cfg)

    eda = next(r for r in reports if r.channel == "chest.EDA")
    assert eda.violations == 10


def test_strict_ranges_raise():
    signal = {
        "chest": {
            "ECG": np.zeros((10, 1)),
            "EMG": np.zeros((10, 1)),
            "EDA": np.full((10, 1), 500.0),
            "Temp": np.full((10, 1), 33.0),
            "Resp": np.zeros((10, 1)),
            "ACC": np.zeros((10, 3)),
        },
        "wrist": {
            "ACC": np.zeros((4, 3)),
            "BVP": np.zeros((8, 1)),
            "EDA": np.zeros((4, 1)),
            "TEMP": np.full((4, 1), 34.0),
        },
    }
    cfg = {"units": {"verify_ranges": True, "strict_ranges": True}}
    with pytest.raises(ValidationError, match="fuera de"):
        units.normalize_units(signal, cfg)


def test_unknown_channel_is_rejected():
    signal = {"chest": {"UNKNOWN": np.zeros((4, 1))}, "wrist": {}}
    with pytest.raises(ValidationError, match="canal no reconocido"):
        units.normalize_units(signal, {"units": {"verify_ranges": False}})
