"""Tests de la comprobación de consistencia fisiológica (P1 de Fase 2)."""

from __future__ import annotations

import copy

import pytest

from sources.phase2.consistency import (
    CHECK_SEVERITY,
    check_consistency,
    should_abstain,
)
from sources.phase2.negative_controls import _shuffle_values

# Valores reales de una ventana limpia de S3 (features absolutas del corpus).
CLEAN_VALUES = {
    "ecg_hr_bpm": 55.13418903150525,
    "ecg_rr_mean_ms": 1088.2539682539682,
    "ecg_n_beats": 55.0,
    "ecg_sdnn_ms": 103.59151832919115,
    "ecg_rmssd_ms": 108.91066310303744,
    "eda_mean_us": 8.130270994277227,
    "eda_std_us": 0.4523894300381929,
    "eda_slope_us_per_s": 0.02588072766845622,
    "eda_tonic_mean_us": 8.124745393224783,
    "eda_phasic_std_us": 0.03516448850790846,
    "eda_scr_count": 2.0,
    "eda_scr_rate_per_min": 2.0,
    "eda_scr_amplitude_mean_us": 0.204143835406533,
    "eda_scr_amplitude_max_us": 0.35522474054892095,
    "resp_rate_bpm": 11.149784673470592,
    "resp_interval_cv": 0.39399234032244856,
    "resp_amplitude_pct": 11.899008633290975,
    "resp_std_pct": 4.223799114838562,
    "resp_n_cycles": 9.0,
    "temp_mean_degc": 32.46962416149321,
    "temp_std_degc": 0.24897757150810032,
    "temp_slope_degc_per_s": -0.014240846595107288,
    "temp_range_degc": 0.976715087890625,
}

# Modalidades en el orden del corpus (ecg, eda, resp, temp) para replicar la
# permutación fija que aplica ``NC_shuffled``.
MODALITIES = {
    "ecg": [
        "ecg_n_beats",
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
        "ecg_sdnn_ms",
        "ecg_rmssd_ms",
    ],
    "eda": [
        "eda_mean_us",
        "eda_std_us",
        "eda_slope_us_per_s",
        "eda_tonic_mean_us",
        "eda_phasic_std_us",
        "eda_scr_count",
        "eda_scr_rate_per_min",
        "eda_scr_amplitude_mean_us",
        "eda_scr_amplitude_max_us",
    ],
    "resp": [
        "resp_rate_bpm",
        "resp_interval_cv",
        "resp_amplitude_pct",
        "resp_std_pct",
        "resp_n_cycles",
    ],
    "temp": [
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
    ],
}


def _with(name: str, value: float) -> dict:
    values = dict(CLEAN_VALUES)
    values[name] = value
    return values


def _shuffled_values() -> dict:
    modalities = {
        modality: {name: {"value": CLEAN_VALUES[name]} for name in names}
        for modality, names in MODALITIES.items()
    }
    shuffled = _shuffle_values(modalities)
    return {name: meta["value"] for block in shuffled.values() for name, meta in block.items()}


def test_clean_window_passes_all_checks():
    report = check_consistency(CLEAN_VALUES, 60.0)
    assert report.failures == ()
    assert report.consistency_score == pytest.approx(1.0)
    assert report.missing_key_features == ()
    assert should_abstain(report) is False
    assert len(report.checks) == len(CHECK_SEVERITY)


def test_shuffled_window_is_flagged_and_gated():
    report = check_consistency(_shuffled_values(), 60.0)
    assert report.n_hard_failures >= 2
    assert "ecg_n_beats_vs_hr" in report.failed_names()
    assert "eda_scr_rate_vs_count" in report.failed_names()
    assert report.consistency_score < 0.5
    assert should_abstain(report) is True


def test_hr_does_not_match_rr_interval():
    report = check_consistency(_with("ecg_hr_bpm", 120.0), 60.0)
    outcome = {c.name: c for c in report.checks}["ecg_hr_vs_rr_mean"]
    assert outcome.ok is False
    assert CHECK_SEVERITY["ecg_hr_vs_rr_mean"] == "hard"
    assert should_abstain(report) is True


def test_zero_scr_count_with_nonzero_rate_is_incoherent():
    report = check_consistency(
        _with("eda_scr_count", 0.0), 60.0
    )
    outcome = {c.name: c for c in report.checks}["eda_scr_rate_vs_count"]
    assert outcome.ok is False


def test_zero_scr_count_and_zero_rate_is_coherent():
    values = _with("eda_scr_count", 0.0)
    values["eda_scr_rate_per_min"] = 0.0
    values["eda_scr_amplitude_mean_us"] = 0.0
    values["eda_scr_amplitude_max_us"] = 0.0
    report = check_consistency(values, 60.0)
    outcome = {c.name: c for c in report.checks}["eda_scr_rate_vs_count"]
    assert outcome.ok is True


def test_n_beats_does_not_match_heart_rate():
    report = check_consistency(_with("ecg_n_beats", 90.0), 60.0)
    outcome = {c.name: c for c in report.checks}["ecg_n_beats_vs_hr"]
    assert outcome.ok is False


def test_scr_amplitude_mean_cannot_exceed_max():
    report = check_consistency(
        _with("eda_scr_amplitude_mean_us", 5.0), 60.0
    )
    outcome = {c.name: c for c in report.checks}["eda_scr_amplitude_mean_vs_max"]
    assert outcome.ok is False


def test_missing_key_features_force_abstention():
    values = dict(CLEAN_VALUES)
    values["ecg_hr_bpm"] = None
    report = check_consistency(values, 60.0)
    assert "ecg_hr_bpm" in report.missing_key_features
    assert should_abstain(report) is True


def test_missing_key_features_can_be_tolerated():
    values = dict(CLEAN_VALUES)
    values["ecg_hr_bpm"] = None
    report = check_consistency(values, 60.0)
    assert should_abstain(report, require_key_features=False) is False


def test_single_soft_failure_is_tolerated_by_default():
    # Rango de temperatura por debajo de 2 DE: solo falla una comprobación soft.
    values = _with("temp_range_degc", 0.10)
    report = check_consistency(values, 60.0)
    assert report.n_hard_failures == 0
    assert report.n_soft_failures == 1
    assert should_abstain(report) is False
    assert should_abstain(report, max_soft_failures=0) is True


def test_two_soft_failures_force_abstention():
    values = _with("temp_range_degc", 0.10)
    values["eda_std_us"] = 0.001
    report = check_consistency(values, 60.0)
    assert report.n_hard_failures == 0
    assert report.n_soft_failures == 2
    assert should_abstain(report) is True


def test_tolerance_scale_relaxes_noise_sensitivity():
    # Ruido del +10 % en HR (y en n_beats, que se mantiene coherente con ella):
    # falla con la tolerancia estricta del 5 % y pasa al relajarla.
    values = dict(CLEAN_VALUES)
    values["ecg_hr_bpm"] *= 1.10
    values["ecg_n_beats"] *= 1.10
    strict = check_consistency(values, 60.0, tolerance_scale=1.0)
    relaxed = check_consistency(values, 60.0, tolerance_scale=2.5)
    assert should_abstain(strict) is True
    assert should_abstain(relaxed) is False


def test_report_serialization_contains_expected_fields():
    payload = check_consistency(CLEAN_VALUES, 60.0).to_dict()
    for key in (
        "consistency_score",
        "n_evaluated",
        "n_hard_failures",
        "n_soft_failures",
        "failed_checks",
        "missing_key_features",
        "checks",
    ):
        assert key in payload
    assert len(payload["checks"]) == len(CHECK_SEVERITY)
    assert payload["checks"][0]["severity"] in {"hard", "soft"}


def test_check_consistency_does_not_mutate_input():
    values = copy.deepcopy(CLEAN_VALUES)
    check_consistency(values, 60.0)
    assert values == CLEAN_VALUES


def test_duration_scales_count_relationships():
    # Con ventana de 30 s, 55 latidos implican 110 bpm y dejan de cuadrar.
    report = check_consistency(CLEAN_VALUES, 30.0)
    outcome = {c.name: c for c in report.checks}["ecg_n_beats_vs_hr"]
    assert outcome.ok is False
