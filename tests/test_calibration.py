"""Tests de calibración y predicción selectiva (P4)."""

from __future__ import annotations

import numpy as np
import pytest

from sources.phase3.calibration import (
    apply_temperature,
    brier_score,
    conformal_risk_threshold,
    evaluate_selective,
    expected_calibration_error,
    fit_temperature,
    group_kfold_indices,
    log_loss,
    maximum_calibration_error,
    reliability_table,
    risk_coverage_curve,
    wilson_interval,
)


def test_brier_and_log_loss_perfect_predictions():
    y = [1, 1, 0, 0]
    p = [0.99, 0.99, 0.01, 0.01]
    assert brier_score(y, p) < 0.001
    assert log_loss(y, p) < 0.02


def test_brier_score_known_value():
    assert brier_score([1, 0], [0.9, 0.1]) == pytest.approx(0.01)


def test_reliability_table_bins():
    rows = reliability_table([1, 1, 0, 0], [0.9, 0.9, 0.1, 0.1], n_bins=10)
    assert len(rows) == 2
    high = next(r for r in rows if r["mean_confidence"] > 0.5)
    assert high["n"] == 2
    assert high["accuracy"] == pytest.approx(1.0)


def test_ece_known_value():
    # Dos bins, cada uno con |acc - conf| = 0,1 y 2 muestras de 4.
    y = [1, 1, 0, 0]
    p = [0.9, 0.9, 0.1, 0.1]
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.1)
    assert maximum_calibration_error(y, p, n_bins=10) == pytest.approx(0.1)


def test_temperature_relaxes_overconfidence():
    # El modelo dice 0,9 pero acierta el 70 % => T > 1 debe mejorar el log-loss.
    y = [1] * 70 + [0] * 30
    p = [0.9] * 100
    fit = fit_temperature(p, y)
    assert fit["temperature"] > 1.0
    assert fit["log_loss_after"] < fit["log_loss_before"]


def test_apply_temperature_identity_at_one():
    probs = np.array([0.2, 0.5, 0.8])
    assert np.allclose(apply_temperature(probs, 1.0), probs, atol=1e-6)


def test_risk_coverage_curve_monotonic_coverage():
    conf = [0.6, 0.7, 0.8, 0.9, 0.95]
    correct = [True, False, True, True, True]
    curve = risk_coverage_curve(conf, correct, thresholds=[0.5, 0.8, 0.9, 0.99])
    coverages = [entry["coverage"] for entry in curve]
    assert coverages == sorted(coverages, reverse=True)
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[-1]["coverage"] == pytest.approx(0.0)


def test_conformal_threshold_reachable():
    conf = [0.9] * 10
    correct = [True] * 9 + [False]
    result = conformal_risk_threshold(conf, correct, alpha=0.2, thresholds=[0.5])
    assert result["threshold"] == 0.5
    assert result["calibration_risk"] == pytest.approx(0.1)


def test_conformal_threshold_unreachable_for_strict_alpha():
    # Con n=10, la cota mínima es 1/11 ≈ 0,091 > 0,05.
    conf = [0.9] * 10
    correct = [True] * 10
    result = conformal_risk_threshold(conf, correct, alpha=0.05, thresholds=[0.5])
    assert result["threshold"] is None


def test_evaluate_selective_without_threshold_accepts_all():
    metrics = evaluate_selective([0.6, 0.9], [True, False], threshold=None)
    assert metrics["coverage"] == 1.0
    assert metrics["selective_accuracy"] == pytest.approx(0.5)


def test_evaluate_selective_with_threshold():
    metrics = evaluate_selective([0.6, 0.9], [False, True], threshold=0.8)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["selective_accuracy"] == pytest.approx(1.0)
    assert metrics["selective_risk"] == pytest.approx(0.0)


def test_group_kfold_has_no_subject_leakage():
    groups = [f"S{i}" for i in range(10) for _ in range(3)]
    splits = group_kfold_indices(groups, n_splits=5)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        train_groups = {groups[i] for i in train_idx}
        test_groups = {groups[i] for i in test_idx}
        assert not (train_groups & test_groups)


def test_wilson_interval_contains_proportion():
    low, high = wilson_interval(90, 100)
    assert low < 0.9 < high
    assert 0.0 <= low <= high <= 1.0
