"""Tests de las utilidades del experimento de degradación."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

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


# ---------------------------------------------------------------------------
# Desglose por clase y endpoint binario mapeado (analyze_degradation)
# ---------------------------------------------------------------------------
def _analysis_records() -> list[dict]:
    """Cuatro ventanas, una por clase; el modelo nunca predice amusement ni
    meditation (patrón real del 7b) y se abstiene en una."""
    return [
        {
            "window_id": "WESAD:S5:w00001",
            "window_id_opaque": "win_00000001",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "baseline",
            "confidence": 80,
            "claims": [],
            "reasoning": "Stable.",
            "prompt": "",
        },
        {
            "window_id": "WESAD:S5:w00002",
            "window_id_opaque": "win_00000002",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "stress",
            "confidence": 90,
            "claims": [],
            "reasoning": "Elevated.",
            "prompt": "",
        },
        {
            # amusement real, predicho baseline: fallo de 4 clases y acierto binario
            "window_id": "WESAD:S5:w00003",
            "window_id_opaque": "win_00000003",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "baseline",
            "confidence": 70,
            "claims": [],
            "reasoning": "Calm.",
            "prompt": "",
        },
        {
            # meditation real, abstención
            "window_id": "WESAD:S5:w00004",
            "window_id_opaque": "win_00000004",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_abstention",
            "decision": "abstain",
            "confidence": 30,
            "claims": [],
            "reasoning": "Insufficient.",
            "prompt": "",
        },
    ]


def _analysis_labels() -> dict[str, str]:
    return {
        "WESAD:S5:w00001": "baseline",
        "WESAD:S5:w00002": "stress",
        "WESAD:S5:w00003": "amusement",
        "WESAD:S5:w00004": "meditation",
    }


def test_analyze_group_per_class_recall():
    """El recall por clase distingue 'no predice la clase' de 'la confunde'."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    per_class = metrics["per_class"]

    assert per_class["baseline"]["recall"] == 1.0
    assert per_class["stress"]["recall"] == 1.0
    assert per_class["amusement"]["recall"] == 0.0
    assert per_class["amusement"]["predictions"] == {"baseline": 1}
    assert per_class["meditation"]["decided"] == 0
    assert per_class["meditation"]["recall_decided"] == 0.0


def test_analyze_group_binary_mapped_endpoint():
    """Con el mapeo 4→2, amusement/baseline cuentan como acierto binario."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    binary = metrics["binary_mapped"]

    assert binary["positive_label"] == "stress"
    assert binary["n_decided"] == 3          # 3 decididas, 1 abstención
    assert binary["abstention_rate"] == pytest.approx(0.25)
    assert binary["coverage"] == pytest.approx(0.75)
    assert binary["accuracy"] == pytest.approx(1.0)
    assert binary["recall_stress"] == pytest.approx(1.0)


def test_analyze_group_splits_certainty_by_decision():
    """Una abstención habla de la evidencia, no del afecto: se separa (H2b)."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    certainty = metrics["certainty"]

    assert certainty["n"] == 4
    assert certainty["decided"]["n"] == 3
    assert certainty["abstain"]["n"] == 1
    assert certainty["decided"]["n"] + certainty["abstain"]["n"] == certainty["n"]


def test_certainty_stats_empty_is_nan():
    from analyze_degradation import _certainty_stats

    stats = _certainty_stats([])
    assert stats["n"] == 0
    assert stats["certainty_index_mean"] != stats["certainty_index_mean"]  # NaN
