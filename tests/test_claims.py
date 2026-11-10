"""Tests del verificador de claims (H5) y de la certeza lingüística (H2b)."""

from __future__ import annotations

import pytest

from sources.phase3.claims import (
    ClaimVerdict,
    direction_compatible,
    expected_direction,
    magnitude_bucket,
    parse_prompt_features,
    summarize_verdicts,
    verify_claim,
    verify_claims,
)
from sources.phase3.certainty import linguistic_certainty

PROMPT = """You are a physiological signal analysis assistant.

Input features (within-subject z-scores; 0 = subject mean, ±1 = 1 standard deviation):

Modality: ECG
  - ecg_hr_bpm: 2.4 SD
  - ecg_rmssd_ms: -1.3 SD
  - ecg_n_beats: 1.9 SD

Modality: EDA
  - eda_mean_us: 0.3 SD
  - eda_scr_count: not available
  - eda_scr_rate_per_min: 2.1 SD

Window ID (opaque): win_12345678
Window duration: 60.0 s

Rules:
- Use only the features provided above. Do not invent data.
"""


def _claim(metric: str, direction: str, magnitude: str = "medium", channel: str = "ecg"):
    return {
        "metric": metric,
        "channel": channel,
        "direction": direction,
        "magnitude": magnitude,
    }


def test_parse_prompt_features():
    features = parse_prompt_features(PROMPT)
    assert features["ecg_hr_bpm"].value == pytest.approx(2.4)
    assert features["ecg_hr_bpm"].unit == "SD"
    assert features["ecg_hr_bpm"].is_standardized
    assert features["ecg_rmssd_ms"].value == pytest.approx(-1.3)
    assert features["eda_scr_count"].value is None
    assert features["eda_scr_count"].available is False
    # Las reglas no se confunden con features.
    assert "Use" not in features


def test_direction_and_magnitude_helpers():
    assert expected_direction(2.0, True) == "increase"
    assert expected_direction(-2.0, True) == "decrease"
    assert expected_direction(0.1, True) == "stable"
    assert expected_direction(0.2, True) == "increase"
    assert expected_direction(2.0, False) is None
    assert magnitude_bucket(0.5, True) == "low"
    assert magnitude_bucket(1.5, True) == "medium"
    assert magnitude_bucket(2.5, True) == "high"


def test_slight_deviation_is_compatible_with_direction():
    """-0,26 SD es compatible con ``decrease`` (banda de ambigüedad)."""
    assert direction_compatible("decrease", -0.26, True) is True
    assert direction_compatible("increase", 0.31, True) is True
    assert direction_compatible("stable", 0.3, True) is True
    # Incompatibles.
    assert direction_compatible("increase", -0.8, True) is False
    assert direction_compatible("stable", 1.5, True) is False
    assert direction_compatible("decrease", 0.8, True) is False


def test_supported_claim():
    verdict = verify_claim(_claim("ecg_hr_bpm", "increase", "high"), parse_prompt_features(PROMPT))
    assert verdict.category == "supported"
    assert verdict.direction_ok is True
    assert verdict.magnitude_ok is True


def test_contradicted_direction():
    verdict = verify_claim(
        _claim("ecg_hr_bpm", "decrease", "high"), parse_prompt_features(PROMPT)
    )
    assert verdict.category == "contradicted"
    assert verdict.direction_ok is False


def test_contradicted_magnitude_by_two_levels():
    verdict = verify_claim(
        _claim("ecg_hr_bpm", "increase", "low"), parse_prompt_features(PROMPT)
    )
    # |z|=2.4 => high; declarar low difiere en 2 niveles.
    assert verdict.category == "contradicted"
    assert verdict.direction_ok is True
    assert verdict.magnitude_ok is False


def test_magnitude_close_enough_is_supported():
    verdict = verify_claim(
        _claim("ecg_hr_bpm", "increase", "medium"), parse_prompt_features(PROMPT)
    )
    assert verdict.category == "supported"


def test_unsupported_metric():
    verdict = verify_claim(
        _claim("resp_rate_bpm", "decrease", "high", channel="resp"),
        parse_prompt_features(PROMPT),
    )
    assert verdict.category == "unsupported"


def test_indeterminate_when_value_missing():
    verdict = verify_claim(
        _claim("eda_scr_count", "increase", "high", channel="eda"),
        parse_prompt_features(PROMPT),
    )
    assert verdict.category == "indeterminate"


def test_stable_claim_supported():
    verdict = verify_claim(
        _claim("eda_mean_us", "stable", "low", channel="eda"),
        parse_prompt_features(PROMPT),
    )
    assert verdict.category == "supported"


def test_absolute_scale_is_indeterminate():
    prompt = "Modality: ECG\n  - ecg_hr_bpm: 72.4 bpm\n"
    verdict = verify_claim(_claim("ecg_hr_bpm", "increase", "medium"), parse_prompt_features(prompt))
    assert verdict.category == "indeterminate"
    assert "absoluta" in verdict.detail


def test_summarize_verdicts():
    verdicts = verify_claims(
        [
            _claim("ecg_hr_bpm", "increase", "high"),
            _claim("ecg_hr_bpm", "decrease", "high"),
            _claim("resp_rate_bpm", "decrease", "high", channel="resp"),
            _claim("eda_scr_count", "decrease", "high", channel="eda"),
        ],
        PROMPT,
    )
    summary = summarize_verdicts(verdicts)
    assert summary["n_claims"] == 4
    assert summary["categories"]["supported"] == 1
    assert summary["categories"]["contradicted"] == 1
    assert summary["categories"]["unsupported"] == 1
    assert summary["categories"]["indeterminate"] == 1
    assert summary["direction_accuracy"] == 0.5  # 1 de 2 evaluables


def test_empty_claims():
    summary = summarize_verdicts([])
    assert summary["n_claims"] == 0
    assert summary["categories"]["supported"] == 0


def test_linguistic_certainty_counts():
    result = linguistic_certainty(
        "The pattern clearly indicates stress; EDA is markedly elevated, which probably reflects arousal."
    )
    assert result["boosters"] == 2  # clearly, markedly
    assert result["hedges"] == 1  # probably
    assert result["certainty_index"] > 0
    assert result["n_words"] > 0


def test_linguistic_certainty_hedged():
    result = linguistic_certainty("This may possibly suggest stress, but it is unclear.")
    assert result["hedges"] >= 3  # may, possibly, suggest, unclear
    assert result["certainty_index"] < 0


def test_linguistic_certainty_empty():
    result = linguistic_certainty("")
    assert result["n_words"] == 0
    assert result["hedges"] == 0
    assert result["boosters"] == 0


def test_recovery_of_invalid_claims():
    import json

    from sources.phase3.recovery import recover_llm_record

    raw = json.dumps(
        {
            "decision": "baseline",
            "confidence": 75,
            "claims": [
                {"metric": "Heart rate", "channel": "cardiovascular", "direction": "slight decrease", "magnitude": "low"},
                {"metric": "SCR count", "channel": "not available", "direction": "increase", "magnitude": "none"},
                {"metric": "Respiratory rate", "channel": "resp", "direction": "not available", "magnitude": "high"},
                {"metric": "Heart rate", "channel": "ecg", "direction": "increase", "magnitude": "medium"},
            ],
            "reasoning": "Slightly decreased heart rate.",
        }
    )
    record = {"status": "invalid_output", "raw_output": raw}
    recovered = recover_llm_record(record)
    assert recovered is not None
    assert recovered["decision"] == "baseline"
    assert recovered["confidence"] == 75
    assert recovered["dropped_claims"] == 2
    assert len(recovered["claims"]) == 2
    assert recovered["claims"][0]["metric"] == "ecg_hr_bpm"
    assert recovered["claims"][0]["direction"] == "decrease"


def test_recovery_rejects_unparseable_decision():
    import json

    from sources.phase3.recovery import recover_llm_record

    raw = json.dumps({"decision": "unknown-state", "confidence": 50, "claims": [], "reasoning": ""})
    assert recover_llm_record({"status": "invalid_output", "raw_output": raw}) is None


def test_apply_recovery_sets_status():
    import json

    from sources.phase3.recovery import apply_recovery

    raw = json.dumps({"decision": "abstain", "confidence": 53, "claims": [], "reasoning": ""})
    records, n = apply_recovery([{"status": "invalid_output", "raw_output": raw}])
    assert n == 1
    assert records[0]["status"] == "valid_abstention"
    assert records[0]["recovered"] is True
