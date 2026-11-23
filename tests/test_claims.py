"""Tests del verificador de claims (H5) y de la certeza lingüística (H2b)."""

from __future__ import annotations

import pytest

from sources.phase3.claims import (
    COHERENCE_CATEGORIES,
    MIN_CLAIMS_FOR_RATE,
    ClaimVerdict,
    claim_coherence,
    coherence_side,
    direction_compatible,
    expected_direction,
    magnitude_bucket,
    parse_prompt_features,
    response_coherence,
    summarize_coherence,
    summarize_verdicts,
    verify_claim,
    verify_claims,
)
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.signatures import (
    PRIMARY_FEATURES,
    SENSITIVITY_FEATURES,
)

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


def test_recovery_respects_decision_mode():
    """``noisy`` es recuperable en ``rhythm4`` pero no en el modo afectivo."""
    import json

    from sources.phase3.recovery import recover_llm_record

    raw = json.dumps(
        {
            "decision": "noisy",
            "confidence": 85,
            "claims": [
                {"metric": "ecg_sdnn_ms", "channel": "ecg", "direction": "increase", "magnitude": "high"},
                {"metric": "ecg_hr_bpm", "channel": "ecg", "direction": "high", "magnitude": "low"},
            ],
            "reasoning": "The tracing is dominated by artifact.",
        }
    )
    record = {"status": "invalid_output", "raw_output": raw}
    assert recover_llm_record(record) is None  # modo afectivo por defecto
    recovered = recover_llm_record(record, decision_mode="rhythm4")
    assert recovered is not None
    assert recovered["decision"] == "noisy"
    assert recovered["dropped_claims"] == 1
    assert len(recovered["claims"]) == 1


def test_recovery_rejects_unknown_decision_mode():
    import json

    import pytest

    from sources.phase3.recovery import recover_llm_record

    raw = json.dumps({"decision": "normal", "confidence": 85, "claims": []})
    with pytest.raises(ValueError):
        recover_llm_record(
            {"status": "invalid_output", "raw_output": raw},
            decision_mode="no-existe",
        )


def test_recovery_binary_mode_recovers_missing_magnitude():
    """Job 2731: 50 inválidas del 7b binario por magnitud ausente."""
    import json

    from sources.phase3.recovery import recover_llm_record

    payload = {
        "decision": "stress",
        "confidence": 85,
        "claims": [
            {"metric": "ecg_hr_bpm", "channel": "ecg", "direction": "increase"}
        ],
        "reasoning": "HR is high.",
    }
    record = {"status": "invalid_output", "raw_output": json.dumps(payload)}

    recovered = recover_llm_record(record, decision_mode="stress_binary")

    assert recovered is not None
    assert recovered["decision"] == "stress"
    assert recovered["claims"][0]["magnitude"] == "low"


def test_recovery_binary_decision_not_in_affective_mode():
    """Sin el modo correcto, 'stress'/'non-stress' no se recuperaban."""
    import json

    from sources.phase3.recovery import recover_llm_record

    payload = {"decision": "non-stress", "confidence": 85, "claims": [], "reasoning": "Calm."}
    record = {"status": "invalid_output", "raw_output": json.dumps(payload)}

    assert recover_llm_record(record, decision_mode="stress_binary") is not None
    assert recover_llm_record(record, decision_mode="affective") is None


def test_recovery_accepts_cli_binary_alias():
    """El CLI pasa 'binary'; el vocabulario interno es 'stress_binary'."""
    import json

    from sources.phase3.recovery import recover_llm_record

    payload = {"decision": "non-stress", "confidence": 80, "claims": [], "reasoning": "Calm."}
    record = {"status": "invalid_output", "raw_output": json.dumps(payload)}

    recovered = recover_llm_record(record, decision_mode="binary")

    assert recovered is not None and recovered["decision"] == "non-stress"


# ---------------------------------------------------------------------------
# Relevancia: coherencia del claim con la decisión (preregistro H5)
# ---------------------------------------------------------------------------


def _coherence_claim(metric: str = "ecg_hr_bpm", direction: str = "increase") -> dict:
    return {
        "metric": metric,
        "channel": "ecg",
        "direction": direction,
        "magnitude": "medium",
    }


def _verdict(category: str = "supported", metric: str = "ecg_hr_bpm") -> ClaimVerdict:
    return ClaimVerdict(
        metric=metric,
        channel="ecg",
        direction="increase",
        magnitude="medium",
        category=category,
        observed=1.0,
        observed_unit="SD",
        direction_ok=True,
        magnitude_ok=True,
    )


def _pair(
    subject: str = "S5",
    side: str = "stress",
    truth: str = "supported",
    coherence: str = "coherent",
) -> dict:
    return {
        "subject": subject,
        "side": side,
        "truth": truth,
        "coherence": coherence,
    }


def test_coherence_categories_catalog():
    assert COHERENCE_CATEGORIES == (
        "coherent",
        "contrary",
        "neutral",
        "no_signature",
        "not_applicable",
    )


def test_coherence_side_maps_binary_and_affective_labels():
    assert coherence_side("stress") == "stress"
    assert coherence_side("non-stress") == "non-stress"
    # Las 4 clases se reinterpretan con el mapa de la jerarquía Schmidt.
    assert coherence_side("baseline") == "non-stress"
    assert coherence_side("amusement") == "non-stress"
    assert coherence_side("meditation") == "non-stress"
    assert coherence_side("abstain") is None
    assert coherence_side(None) is None
    assert coherence_side("") is None


def test_claim_coherence_stress_side():
    coherent = claim_coherence(_coherence_claim("ecg_hr_bpm", "increase"), "stress")
    contrary = claim_coherence(_coherence_claim("ecg_hr_bpm", "decrease"), "stress")
    assert (coherent.category, coherent.sign, coherent.alignment) == ("coherent", 1, 1)
    assert (contrary.category, contrary.alignment) == ("contrary", -1)
    # Las features con firma negativa se invierten igual: SDNN baja es coherente con stress.
    assert claim_coherence(_coherence_claim("ecg_sdnn_ms", "decrease"), "stress").category == (
        "coherent"
    )
    assert claim_coherence(_coherence_claim("ecg_sdnn_ms", "increase"), "stress").category == (
        "contrary"
    )


def test_claim_coherence_non_stress_is_mirror():
    assert claim_coherence(_coherence_claim("ecg_hr_bpm", "decrease"), "non-stress").category == (
        "coherent"
    )
    assert claim_coherence(_coherence_claim("ecg_hr_bpm", "increase"), "non-stress").category == (
        "contrary"
    )
    assert claim_coherence(_coherence_claim("ecg_sdnn_ms", "increase"), "non-stress").category == (
        "coherent"
    )


def test_claim_coherence_stable_is_neutral_on_both_sides():
    for decision in ("stress", "non-stress"):
        block = claim_coherence(_coherence_claim("ecg_hr_bpm", "stable"), decision)
        assert block.category == "neutral"
        assert block.alignment == 0


def test_claim_coherence_metric_without_signature():
    # 'temperature' se normaliza a temp_mean_degc, que está fuera de la firma.
    block = claim_coherence(_coherence_claim("temperature", "increase"), "stress")
    assert block.metric == "temp_mean_degc"
    assert block.category == "no_signature"
    assert block.sign == 0
    assert block.alignment is None


def test_claim_coherence_respects_the_feature_set():
    claim = _coherence_claim("resp_interval_cv", "increase")
    assert claim_coherence(claim, "stress", features=PRIMARY_FEATURES).category == "no_signature"
    assert claim_coherence(claim, "stress", features=SENSITIVITY_FEATURES).category == "coherent"


def test_claim_coherence_abstain_is_not_applicable():
    block = claim_coherence(_coherence_claim(), "abstain")
    assert block.category == "not_applicable"
    assert block.side is None


def test_claim_coherence_unknown_direction_is_not_signature():
    block = claim_coherence(_coherence_claim("ecg_hr_bpm", "sideways"), "stress")
    assert block.category == "no_signature"


def test_response_coherence_counts_and_pairs():
    claims = [
        _coherence_claim("ecg_hr_bpm", "increase"),  # coherente
        _coherence_claim("ecg_hr_bpm", "decrease"),  # contrario y verdadero
        _coherence_claim("temperature", "increase"),  # sin firma
    ]
    verdicts = [
        _verdict("supported"),
        _verdict("supported"),
        _verdict("supported", metric="temp_mean_degc"),
    ]
    block = response_coherence(claims, verdicts, "stress", subject="S7")

    assert block["side"] == "stress"
    assert block["n_claims"] == 3
    assert block["n_signature_claims"] == 2
    assert block["n_confirming"] == 1
    assert block["n_against"] == 1
    assert block["n_no_signature"] == 1
    assert block["n_against_supported"] == 1
    assert block["has_confirmation"] is True
    assert block["evidence_conflict"] is False
    assert [pair["subject"] for pair in block["pairs"]] == ["S7", "S7"]
    assert [pair["coherence"] for pair in block["pairs"]] == ["coherent", "contrary"]


def test_response_coherence_abstain_and_conflict_flags():
    claims = [_coherence_claim("ecg_hr_bpm", "decrease")]
    block = response_coherence(claims, [_verdict("supported")], "abstain")
    assert block["n_not_applicable"] == 1
    assert block["pairs"] == []
    assert block["side"] is None

    conflict = response_coherence(
        [_coherence_claim("ecg_hr_bpm", "decrease")],
        [_verdict("supported")],
        "stress",
    )
    assert conflict["evidence_conflict"] is True
    assert conflict["has_confirmation"] is False


def test_summarize_coherence_rate_and_joint_table():
    pairs = [
        _pair(truth="supported", coherence="coherent"),
        _pair(truth="supported", coherence="coherent"),
        _pair(truth="supported", coherence="contrary"),
        _pair(truth="contradicted", coherence="coherent"),
    ]
    block = summarize_coherence(pairs, min_claims=1)

    assert block["n_pairs"] == 4
    assert block["categories"] == {"coherent": 3, "contrary": 1, "neutral": 0}
    assert block["coherence_rate"] == pytest.approx(0.75)
    assert block["coherence_denominator"] == 4
    assert block["insufficient"] is False
    # 3 claims verdaderos; 1 de ellos contradice la propia decisión.
    assert block["n_supported"] == 3
    assert block["against_supported_share"] == pytest.approx(1 / 3)
    assert block["joint"]["supported"] == {"coherent": 2, "contrary": 1, "neutral": 0}
    assert block["joint"]["contradicted"] == {"coherent": 1, "contrary": 0, "neutral": 0}


def test_summarize_coherence_insufficient_below_min_claims():
    pairs = [_pair() for _ in range(MIN_CLAIMS_FOR_RATE - 1)]
    assert summarize_coherence(pairs)["insufficient"] is True
    pairs.append(_pair())
    assert summarize_coherence(pairs)["insufficient"] is False
    # El umbral se puede rebajar explícitamente (no se publica si es insuficiente).
    assert summarize_coherence(pairs[:3], min_claims=3)["insufficient"] is False


def test_summarize_coherence_by_side_does_not_aggregate():
    pairs = [
        _pair(side="stress", coherence="coherent"),
        _pair(side="stress", coherence="coherent"),
        _pair(side="stress", coherence="coherent"),
        _pair(side="stress", coherence="contrary"),
        _pair(side="non-stress", coherence="contrary"),
        _pair(side="non-stress", coherence="contrary"),
    ]
    block = summarize_coherence(pairs, min_claims=1)

    assert block["coherence_rate"] == pytest.approx(3 / 6)
    assert block["by_side"]["stress"]["coherence_rate"] == pytest.approx(0.75)
    assert block["by_side"]["stress"]["n_pairs"] == 4
    assert block["by_side"]["non-stress"]["coherence_rate"] == pytest.approx(0.0)
    assert block["by_side"]["stress"]["insufficient"] is False


def test_summarize_coherence_response_rates():
    responses = [
        {
            "side": "stress",
            "n_signature_claims": 2,
            "n_confirming": 1,
            "n_against": 1,
            "n_against_supported": 1,
            "has_confirmation": True,
            "evidence_conflict": False,
            "n_no_signature": 0,
            "n_not_applicable": 0,
        },
        {
            "side": "stress",
            "n_signature_claims": 1,
            "n_confirming": 0,
            "n_against": 1,
            "n_against_supported": 0,
            "has_confirmation": False,
            "evidence_conflict": True,
            "n_no_signature": 2,
            "n_not_applicable": 0,
        },
        {
            # Sin claims con firma: no cuenta en el denominador de confirmación.
            "side": "stress",
            "n_signature_claims": 0,
            "n_confirming": 0,
            "n_against": 0,
            "n_against_supported": 0,
            "has_confirmation": False,
            "evidence_conflict": False,
            "n_no_signature": 1,
            "n_not_applicable": 0,
        },
        {
            "side": "non-stress",
            "n_signature_claims": 1,
            "n_confirming": 1,
            "n_against": 0,
            "n_against_supported": 0,
            "has_confirmation": True,
            "evidence_conflict": False,
            "n_no_signature": 0,
            "n_not_applicable": 1,
        },
    ]
    block = summarize_coherence([_pair()], responses=responses, min_claims=1)

    assert block["responses"]["n"] == 4
    assert block["responses"]["n_stress"] == 3
    assert block["responses"]["n_non_stress"] == 1
    # 1 de 2 respuestas de stress con claims con firma no tiene confirmación.
    assert block["responses"]["zero_confirmation_share"] == pytest.approx(0.5)
    assert block["responses"]["evidence_conflict_rate"] == pytest.approx(0.25)
    # 1 claim contrario verdadero sobre 2 claims contrarios en total.
    assert block["responses"]["contrary_supported_rate"] == pytest.approx(0.5)
    assert block["n_no_signature_total"] == 3
    assert block["n_not_applicable_total"] == 1


def test_summarize_coherence_empty_is_defined():
    block = summarize_coherence([])
    assert block["n_pairs"] == 0
    assert block["insufficient"] is True
    assert block["coherence_denominator"] == 0
