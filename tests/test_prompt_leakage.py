"""Pruebas de fugas (leakage) del constructor de prompts (Fase 2).

Verifica que los prompts generados por ``prompt_builder`` nunca contengan:
- etiquetas de estado afectivo (baseline, stress, amusement, meditation)
- identificadores reales de sujeto (S2, S3, ...)
- identificadores reales de ventana (window_00001, w00001, ...)
- tiempos absolutos o marcas de tiempo
- términos de calidad (sqi, quality, coverage, flag, good, fair, poor)
"""

from __future__ import annotations

import pytest

from sources.phase2.prompt_builder import (
    FEW_SHOT_EXAMPLES,
    assert_no_leakage,
    build_numeric_prompt,
    check_prompt_leakage,
    estimate_tokens,
    filter_eda_hr_modalities,
)


def _make_numeric_payload() -> dict:
    """Crea un payload numérico LLM sintético con features EDA + HR."""
    return {
        "modalities": {
            "ecg": {
                "ecg_hr_bpm": {"value": 72.5, "unit": "bpm", "kind": "derived"},
                "ecg_rr_mean_ms": {"value": 827.0, "unit": "ms", "kind": "derived"},
                "ecg_sdnn_ms": {"value": 35.2, "unit": "ms", "kind": "derived"},
                "ecg_rmssd_ms": {"value": 28.1, "unit": "ms", "kind": "derived"},
                "ecg_n_beats": {"value": 44, "unit": "count", "kind": "observed"},
            },
            "eda": {
                "eda_mean_us": {"value": 12.4, "unit": "uS", "kind": "observed"},
                "eda_std_us": {"value": 3.1, "unit": "uS", "kind": "observed"},
                "eda_phasic_std_us": {"value": 1.8, "unit": "uS", "kind": "derived"},
                "eda_scr_count": {"value": 2, "unit": "count", "kind": "derived"},
                "eda_scr_rate_per_min": {"value": 2.0, "unit": "1/min", "kind": "derived"},
            },
            # Modalidades que deben ser filtradas (no EDA ni HR)
            "acc": {
                "acc_mean_g": {"value": 0.02, "unit": "g", "kind": "observed"},
                "acc_std_g": {"value": 0.05, "unit": "g", "kind": "observed"},
            },
            "emg": {
                "emg_mean_abs_mv": {"value": 0.5, "unit": "mV", "kind": "observed"},
            },
        },
        "feature_version": "1.0",
        "window": {
            "window_id": "win_a1b2c3d4",
            "duration_s": 60.0,
        },
    }


def test_filter_eda_hr_removes_non_eda_hr_modalities():
    """Las modalidades que no son EDA ni HR deben ser filtradas."""
    payload = _make_numeric_payload()
    filtered = filter_eda_hr_modalities(payload["modalities"])

    assert "ecg" in filtered
    assert "eda" in filtered
    assert "acc" not in filtered
    assert "emg" not in filtered
    assert "resp" not in filtered
    assert "temp" not in filtered
    assert "bvp" not in filtered

    # Dentro de ecg, solo features HR permitidas
    ecg_names = set(filtered["ecg"].keys())
    assert ecg_names.issubset({
        "ecg_hr_bpm", "ecg_rr_mean_ms", "ecg_sdnn_ms",
        "ecg_rmssd_ms", "ecg_n_beats",
    })

    # Dentro de eda, solo features EDA permitidas
    eda_names = set(filtered["eda"].keys())
    assert eda_names.issubset({
        "eda_mean_us", "eda_std_us", "eda_slope_us_per_s",
        "eda_tonic_mean_us", "eda_phasic_std_us",
        "eda_scr_candidates", "eda_scr_count",
        "eda_scr_rate_per_min", "eda_scr_amplitude_mean_us",
        "eda_scr_amplitude_max_us",
    })


def test_build_numeric_prompt_no_leakage():
    """El prompt generado no contiene información prohibida."""
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_a1b2c3d4",
        duration_s=60.0,
        num_ctx=32768,
    )

    leaks = check_prompt_leakage(prompt)
    assert leaks == {}, f"Fugas detectadas: {leaks}"

    # También debe pasar assert_no_leakage sin lanzar excepción
    assert_no_leakage(prompt)


def test_build_numeric_prompt_contains_required_elements():
    """El prompt contiene los elementos obligatorios para el LLM."""
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_a1b2c3d4",
        duration_s=60.0,
        num_ctx=32768,
    )

    assert "You are a physiological signal analysis assistant." in prompt
    assert "Classify the affective state" in prompt
    assert "EDA + HR" in prompt
    assert "win_a1b2c3d4" in prompt
    assert "ecg_hr_bpm" in prompt
    assert "eda_mean_us" in prompt
    assert '"decision"' in prompt
    assert '"confidence"' in prompt
    assert '"claims"' in prompt
    assert '"reasoning"' in prompt
    assert "abstain" in prompt
    assert "0-100" in prompt


def test_build_numeric_prompt_excludes_acc_and_emg():
    """ACC y EMG nunca aparecen en el prompt del conjunto mínimo."""
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
    )

    assert "acc_mean_g" not in prompt
    assert "acc_std_g" not in prompt
    assert "emg_mean_abs_mv" not in prompt
    assert "emg_std_mv" not in prompt


def test_check_prompt_leakage_detects_labels():
    """Se detectan etiquetas de estado afectivo."""
    prompt_ok = "The heart rate is 72 bpm and EDA is 12 uS."
    assert check_prompt_leakage(prompt_ok) == {}

    prompt_bad = "The subject is in stress state with high EDA."
    leaks = check_prompt_leakage(prompt_bad)
    assert "labels" in leaks


def test_check_prompt_leakage_detects_subject_ids():
    """Se detectan identificadores de sujeto reales."""
    prompt_bad = "Subject S2 shows increased heart rate."
    leaks = check_prompt_leakage(prompt_bad)
    assert "subject_ids" in leaks


def test_check_prompt_leakage_detects_window_ids():
    """Se detectan identificadores de ventana reales."""
    prompt_bad = "Window window_00042 has high EDA."
    leaks = check_prompt_leakage(prompt_bad)
    assert "window_ids" in leaks


def test_check_prompt_leakage_detects_quality_terms():
    """Se detectan términos de calidad (sqi, coverage, flag, etc.)."""
    prompt_bad = "The SQI is 0.93 and coverage is 1.0."
    leaks = check_prompt_leakage(prompt_bad)
    assert "quality_terms" in leaks


def test_check_prompt_leakage_detects_timestamps():
    """Se detectan marcas de tiempo absolutas."""
    prompt_bad = "At 14:30:00 the EDA increased."
    leaks = check_prompt_leakage(prompt_bad)
    assert "timestamps" in leaks


def test_assert_no_leakage_raises_on_leak():
    """assert_no_leakage lanza ValueError cuando hay fuga."""
    prompt_bad = "Subject S5 is in stress with SQI 0.8."
    with pytest.raises(ValueError, match="Fuga de información"):
        assert_no_leakage(prompt_bad)


def test_estimate_tokens_basic():
    """La estimación de tokens es determinista y positiva."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 2  # 5 chars / 4 = 1.25 -> ceil = 2
    assert estimate_tokens("a" * 100) == 25


def test_build_numeric_prompt_rejects_excessive_length():
    """Si el prompt excede num_ctx, se lanza ValueError (no se trunca)."""
    payload = _make_numeric_payload()
    with pytest.raises(ValueError, match="excede num_ctx"):
        build_numeric_prompt(
            payload=payload,
            window_id_opaque="win_x",
            num_ctx=10,  # Muy pequeño para forzar el error
        )


def test_few_shot_examples_reflect_real_scr_values():
    """Los ejemplos few-shot usan valores SCR reales (umbral 0.05 µS)."""
    # Valores inflados del ejemplo anterior (SCR 8 stress, 5 amusement, 2 baseline)
    # no deben aparecer.
    assert "eda_scr_count=8" not in FEW_SHOT_EXAMPLES
    assert "eda_scr_count=2 count" not in FEW_SHOT_EXAMPLES
    # El "amusement" elevado (HR 85) era contradictorio con los datos reales.
    assert "ecg_hr_bpm=85.0" not in FEW_SHOT_EXAMPLES
    # Valores reales: baseline/amusement SCR≈1, stress≈5, meditation≈0.
    assert "eda_scr_count=5 count" in FEW_SHOT_EXAMPLES  # stress
    assert "eda_scr_count=1 count" in FEW_SHOT_EXAMPLES  # baseline/amusement
    assert "eda_scr_count=0 count" in FEW_SHOT_EXAMPLES  # meditation


def test_few_shot_examples_do_not_leak_labels():
    """El prompt con few-shot no introduce fugas de etiqueta."""
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_a1b2c3d4",
        num_ctx=32768,
        use_few_shot=True,
    )
    assert_no_leakage(prompt)
