"""Pruebas del esquema JSON de salida del LLM (Fase 2, Paso 2)."""

from __future__ import annotations

import pytest

from sources.phase2.response_schema import (
    ALLOWED_DECISIONS,
    Claim,
    LlmResponse,
    make_timeout_response,
    schema_example,
    validate_llm_output,
)


def _valid_response_str() -> str:
    return (
        '{"decision": "stress", "confidence": 87, '
        '"claims": [{"metric": "eda_phasic_std_us", "channel": "eda", '
        '"direction": "increase", "magnitude": "high"}], '
        '"reasoning": "EDA phasic variability is elevated."}'
    )


def test_validate_valid_response():
    """Respuesta JSON válida con decision normal."""
    resp = validate_llm_output(_valid_response_str())
    assert resp.status == "valid_response"
    assert resp.decision == "stress"
    assert resp.confidence == 87
    assert len(resp.claims) == 1
    assert resp.claims[0].metric == "eda_phasic_std_us"
    assert resp.claims[0].channel == "eda"
    assert resp.claims[0].direction == "increase"
    assert resp.claims[0].magnitude == "high"
    assert "EDA phasic variability" in resp.reasoning
    assert resp.error_message is None


def test_validate_valid_abstention():
    """Respuesta válida con decision 'abstain'."""
    raw = (
        '{"decision": "abstain", "confidence": 30, '
        '"claims": [], "reasoning": "Insuficiente evidencia."}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "valid_abstention"
    assert resp.decision == "abstain"
    assert resp.confidence == 30
    assert resp.claims == []


def test_validate_all_allowed_decisions():
    """Todas las decisions permitidas se aceptan."""
    for decision in sorted(ALLOWED_DECISIONS):
        raw = f'{{"decision": "{decision}", "confidence": 50, "claims": [], "reasoning": "ok"}}'
        resp = validate_llm_output(raw)
        assert resp.status in ("valid_response", "valid_abstention")
        assert resp.decision == decision


def test_validate_malformed_json():
    """JSON mal formado se clasifica como invalid_output."""
    resp = validate_llm_output("{not valid json")
    assert resp.status == "invalid_output"
    assert resp.error_message is not None
    assert "JSON mal formado" in resp.error_message


def test_validate_missing_field():
    """Campo obligatorio faltante -> invalid_output."""
    raw = '{"decision": "stress", "confidence": 80}'
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "claims" in resp.error_message or "reasoning" in resp.error_message


def test_validate_invalid_decision():
    """Decisión no permitida -> invalid_output."""
    raw = '{"decision": "unknown", "confidence": 50, "claims": [], "reasoning": "x"}'
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "no permitido" in resp.error_message


def test_validate_confidence_out_of_range():
    """Confidence fuera de 0-100 -> invalid_output."""
    for bad in (-1, 101, 200):
        raw = f'{{"decision": "stress", "confidence": {bad}, "claims": [], "reasoning": "x"}}'
        resp = validate_llm_output(raw)
        assert resp.status == "invalid_output", f"confidence {bad} debería fallar"
        assert "0-100" in resp.error_message


def test_validate_confidence_is_bool():
    """Confidence como boolean no es válido."""
    raw = '{"decision": "stress", "confidence": true, "claims": [], "reasoning": "x"}'
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "no es entero" in resp.error_message


def test_validate_confidence_is_float():
    """Confidence como float no es válido."""
    raw = '{"decision": "stress", "confidence": 87.5, "claims": [], "reasoning": "x"}'
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "no es entero" in resp.error_message


def test_validate_claim_invalid_channel():
    """Claim con canal de movimiento (acc) -> invalid_output."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "acc_mean_g", "channel": "acc", '
        '"direction": "increase", "magnitude": "high"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "canales permitidos" in resp.error_message


def test_validate_claim_alias_cardiovascular():
    """El título descriptivo 'cardiovascular' se mapea a 'ecg'."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "ecg_hr_bpm", "channel": "cardiovascular", '
        '"direction": "increase", "magnitude": "high"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.claims[0].channel == "ecg"


def test_validate_claim_alias_electrodermal():
    """El título 'electrodermal activity' se mapea a 'eda'."""
    raw = (
        '{"decision": "baseline", "confidence": 70, '
        '"claims": [{"metric": "eda_mean_us", "channel": "electrodermal activity", '
        '"direction": "stable", "magnitude": "low"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.claims[0].channel == "eda"


def test_validate_claim_additional_channels_allowed():
    """Los canales adicionales (resp, temp, bvp) se aceptan."""
    for channel, metric in [
        ("respiration", "resp_rate_bpm"),
        ("temperature", "temp_mean_degc"),
        ("pulse", "bvp_hr_bpm"),
    ]:
        raw = (
            '{"decision": "stress", "confidence": 80, '
            f'"claims": [{{"metric": "{metric}", "channel": "{channel}", '
            '"direction": "increase", "magnitude": "high"}], '
            '"reasoning": "x"}'
        )
        resp = validate_llm_output(raw)
        assert resp.status == "valid_response", f"{channel} debería aceptarse"


def test_validate_claim_wrist_channel_rejected():
    """Un título de muñeca no se mapea a pecho y se rechaza."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "eda_mean_us", "channel": "wrist electrodermal activity", '
        '"direction": "increase", "magnitude": "high"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "canales permitidos" in resp.error_message


def test_validate_claim_invalid_direction():
    """Claim con direction no permitido -> invalid_output."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "eda_mean_us", "channel": "eda", '
        '"direction": "fluctuate", "magnitude": "high"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "direction" in resp.error_message


def test_validate_claim_invalid_magnitude():
    """Claim con magnitud no permitida -> invalid_output."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "eda_mean_us", "channel": "eda", '
        '"direction": "increase", "magnitude": "extreme"}], '
        '"reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "magnitude" in resp.error_message


def test_validate_claims_not_array():
    """claims no es array -> invalid_output."""
    raw = '{"decision": "stress", "confidence": 80, "claims": "no array", "reasoning": "x"}'
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "no es un array" in resp.error_message


def test_validate_reasoning_not_string():
    """reasoning no es string pero se convierte (tolerante)."""
    raw = '{"decision": "stress", "confidence": 80, "claims": [], "reasoning": 123}'
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.reasoning == "123"


def test_validate_claim_missing_field():
    """Claim con campo faltante -> invalid_output."""
    raw = (
        '{"decision": "stress", "confidence": 80, '
        '"claims": [{"metric": "eda_mean_us"}], "reasoning": "x"}'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "invalid_output"
    assert "falta campo" in resp.error_message


def test_make_timeout_response():
    """make_timeout_response crea respuesta con status timeout."""
    resp = make_timeout_response()
    assert resp.status == "timeout"
    assert resp.error_message == "Tiempo de espera agotado"
    assert resp.decision == ""
    assert resp.confidence == 0


def test_schema_example_is_valid():
    """El ejemplo del esquema pasa la validación."""
    import json

    example = schema_example()
    raw = json.dumps(example)
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.decision == example["decision"]
    assert resp.confidence == example["confidence"]


def test_validate_json_embedded_in_text():
    """JSON dentro de texto (markdown, explicación) se extrae y valida."""
    raw = (
        "Aquí está el resultado:\n```json\n"
        '{"decision": "stress", "confidence": 85, '
        '"claims": [{"metric": "eda_mean_us", "channel": "eda", '
        '"direction": "increase", "magnitude": "high"}], '
        '"reasoning": "EDA elevada."}\n```'
    )
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.decision == "stress"
    assert resp.confidence == 85


def test_validate_confidence_as_string():
    """confidence como string numérico se acepta."""
    raw = '{"decision": "baseline", "confidence": "90", "claims": [], "reasoning": "ok"}'
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.confidence == 90


def test_validate_decision_case_insensitive():
    """decision con mayúsculas/minúsculas se acepta."""
    raw = '{"decision": "STRESS", "confidence": 80, "claims": [], "reasoning": "ok"}'
    resp = validate_llm_output(raw)
    assert resp.status == "valid_response"
    assert resp.decision == "stress"
