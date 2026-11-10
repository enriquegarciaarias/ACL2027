"""Tests de las utilidades del experimento de degradación."""

from __future__ import annotations

from pathlib import Path

import pytest

from sources.phase3.degradation import (
    EFFECTIVE_CONDITIONS,
    build_payload,
    features_path,
    subject_reference,
)


def test_features_path_clean_vs_degraded():
    ingest = Path("/tmp/ingest")
    clean = features_path(ingest, "C0_clean", "S5")
    degraded = features_path(ingest, "C4_total_eda", "S5")
    assert clean == ingest / "features" / "S5.features.jsonl"
    assert degraded == ingest / "degraded_features" / "C4_total_eda" / "S5.features.jsonl"


def test_effective_conditions_exclude_c3():
    assert "C3_total_resp" not in EFFECTIVE_CONDITIONS
    assert len(EFFECTIVE_CONDITIONS) == 5


def test_subject_reference_uses_provided_rows_only():
    rows = {
        "w1": {"ecg_hr_bpm": 60.0, "eda_mean_us": 2.0},
        "w2": {"ecg_hr_bpm": 70.0, "eda_mean_us": 4.0},
        "w3": {"ecg_hr_bpm": 80.0, "eda_mean_us": 6.0},
    }
    centers, scales = subject_reference(rows, ["ecg_hr_bpm", "eda_mean_us"])
    assert centers["ecg_hr_bpm"] == pytest.approx(70.0)
    assert centers["eda_mean_us"] == pytest.approx(4.0)
    assert scales["ecg_hr_bpm"] > 0
    assert scales["eda_mean_us"] > 0


def test_subject_reference_handles_missing_feature():
    rows = {"w1": {"ecg_hr_bpm": 60.0}}
    centers, scales = subject_reference(rows, ["ecg_hr_bpm", "eda_mean_us"])
    assert centers["eda_mean_us"] == 0.0
    assert scales["eda_mean_us"] == 1.0


def test_build_payload_clips_and_marks_missing():
    payload = build_payload(
        window_id="WESAD:S5:w00001",
        values={"ecg_hr_bpm": 300.0, "eda_mean_us": None},
        centers={"ecg_hr_bpm": 70.0, "eda_mean_us": 3.0},
        scales={"ecg_hr_bpm": 10.0, "eda_mean_us": 1.0},
        duration_s=60.0,
    )
    assert payload["modalities"]["ecg"]["ecg_hr_bpm"]["value"] == 3.0  # recortado
    assert payload["modalities"]["eda"]["eda_mean_us"]["value"] is None
    assert payload["feature_version"] == "degradation-e12"
    assert payload["window"]["duration_s"] == 60.0


def test_build_payload_reference_is_condition_independent():
    """Con la misma referencia, degradar EDA baja su z (no se re-normaliza)."""
    centers = {"eda_mean_us": 4.0}
    scales = {"eda_mean_us": 2.0}
    clean = build_payload("w", {"eda_mean_us": 4.0}, centers, scales, 60.0)
    degraded = build_payload("w", {"eda_mean_us": 0.0}, centers, scales, 60.0)
    assert clean["modalities"]["eda"]["eda_mean_us"]["value"] == pytest.approx(0.0)
    assert degraded["modalities"]["eda"]["eda_mean_us"]["value"] == pytest.approx(-2.0)
