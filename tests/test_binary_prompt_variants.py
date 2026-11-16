"""Tests de las variantes del prompt binario y de la exclusión de temperatura."""

from __future__ import annotations

import json

from sources.phase2.prompt_builder import build_numeric_prompt, build_text_prompt
from sources.phase2.response_schema import DECISION_MODE_STRESS_BINARY
from sources.phase3.degradation import build_payload, load_features


def _payload() -> dict:
    return {
        "modalities": {
            "ecg": {"ecg_hr_bpm": {"value": 70.0, "unit": "bpm"}},
            "eda": {"eda_mean_us": {"value": 2.0, "unit": "uS"}},
            "resp": {"resp_rate_bpm": {"value": 15.0, "unit": "bpm"}},
            "temp": {"temp_mean_degc": {"value": 36.5, "unit": "°C"}},
        },
        "feature_version": "test",
        "window": {"window_id": "win_x", "duration_s": 60.0},
    }


def _prompt(**kwargs) -> str:
    params = dict(
        payload=_payload(),
        window_id_opaque="win_x",
        duration_s=60.0,
        num_ctx=32768,
        use_few_shot=True,
        include_additional=True,
        standardized=True,
        decision_mode=DECISION_MODE_STRESS_BINARY,
    )
    params.update(kwargs)
    return build_numeric_prompt(**params)


def test_default_binary_prompt_is_schmidt():
    prompt = _prompt()
    assert "calm baseline and amusement" in prompt
    assert "meditative states" not in prompt
    assert "Non-stress (amusement)" in prompt
    assert "Non-stress (meditation)" not in prompt
    assert 'Use "non-stress" for calm baseline and amusement.' in prompt


def test_all_variant_includes_meditation():
    prompt = _prompt(nonstress_classes=("baseline", "amusement", "meditation"))
    assert "meditative states" in prompt
    assert "Non-stress (meditation)" in prompt
    assert "Non-stress (amusement)" in prompt


def test_baseline_only_variant():
    prompt = _prompt(nonstress_classes=("baseline",))
    assert "covers calm baseline." in prompt
    assert "amusement" not in prompt
    assert "Non-stress (calm baseline)" in prompt


def test_text_prompt_accepts_variant():
    prompt = build_text_prompt(
        text_representation="Heart rate: +0.1 SD\nConductance: +0.2 SD",
        window_id_opaque="win_x",
        duration_s=60.0,
        num_ctx=32768,
        use_few_shot=True,
        standardized=True,
        decision_mode=DECISION_MODE_STRESS_BINARY,
        nonstress_classes=("baseline", "amusement"),
    )
    assert "calm baseline and amusement" in prompt
    assert "meditative states" not in prompt


def test_build_payload_can_exclude_temperature():
    values = {
        "ecg_hr_bpm": 70.0,
        "eda_mean_us": 2.0,
        "temp_mean_degc": 36.5,
    }
    without = build_payload(
        "w1", values, {}, {}, 60.0, include_temperature=False
    )
    assert "temp" not in without["modalities"]
    assert "ecg" in without["modalities"]
    with_temp = build_payload("w1", values, {}, {}, 60.0, include_temperature=True)
    assert "temp" in with_temp["modalities"]
    assert "temp_mean_degc" in with_temp["modalities"]["temp"]


def test_load_features_can_exclude_temperature(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    record = {
        "window_id": "WESAD:S5:w00001",
        "features": {
            "ecg": {"ecg_hr_bpm": {"value": 70.0}},
            "temp": {"temp_mean_degc": {"value": 36.5}},
        },
    }
    (features_dir / "S5.features.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )

    without = load_features(tmp_path, "C0_clean", "S5", include_temperature=False)
    assert "ecg_hr_bpm" in without["WESAD:S5:w00001"]
    assert "temp_mean_degc" not in without["WESAD:S5:w00001"]

    with_temp = load_features(tmp_path, "C0_clean", "S5", include_temperature=True)
    assert "temp_mean_degc" in with_temp["WESAD:S5:w00001"]
