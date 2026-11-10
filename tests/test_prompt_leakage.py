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
            "resp": {
                "resp_rate_bpm": {"value": 14.0, "unit": "1/min", "kind": "derived"},
            },
            "temp": {
                "temp_mean_degc": {"value": 33.5, "unit": "degC", "kind": "observed"},
            },
            "bvp": {
                "bvp_hr_bpm": {"value": 70.0, "unit": "bpm", "kind": "derived"},
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


def test_build_numeric_prompt_with_additional_features_describes_chest_channels():
    """Con ``include_additional`` el prompt declara los canales de pecho reales.

    Evita que el modelo reciba un encabezado que diga "EDA + HR only" mientras
    se le envían también respiración y temperatura. BVP es de muñeca y no debe
    aparecer.
    """
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
        include_additional=True,
    )

    assert "chest physiological features" in prompt
    assert "respiration" in prompt
    assert "resp_rate_bpm" in prompt
    assert "temp_mean_degc" in prompt
    assert "EDA + HR only" not in prompt
    assert "wrist_" not in prompt
    assert "bvp" not in prompt


def test_build_numeric_prompt_standardized_uses_zscore_scale():
    """La variante E12 declara la escala z y usa ejemplos en la misma escala."""
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
        use_few_shot=True,
        include_additional=True,
        standardized=True,
    )
    assert "within-subject z-scores" in prompt
    assert "0 is the subject's session reference" in prompt
    assert "ecg_hr_bpm=+2.1 SD" in prompt  # mediana de dev (stress)
    assert "EDA + HR only" not in prompt
    assert "wrist_" not in prompt
    assert_no_leakage(prompt)


def test_build_numeric_prompt_consistency_rule_is_optional():
    """La comprobación de consistencia solo aparece si se pide."""
    payload = _make_numeric_payload()
    base = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
        include_additional=True,
        standardized=True,
    )
    with_rule = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
        include_additional=True,
        standardized=True,
        consistency_check=True,
    )
    assert "Consistency check" not in base
    assert "Consistency check" in with_rule
    assert "eda_scr_count and eda_scr_rate_per_min" in with_rule
    assert "ecg_n_beats and ecg_hr_bpm" in with_rule
    assert_no_leakage(with_rule)


def test_build_text_prompt_accepts_scale_options():
    """``build_text_prompt`` admite la escala y la comprobación (regresión)."""
    from sources.phase2.prompt_builder import build_text_prompt

    prompt = build_text_prompt(
        text_representation="ecg_hr_bpm: 72 bpm\neda_mean_us: 4 uS",
        window_id_opaque="win_x",
        num_ctx=32768,
        standardized=True,
        rubric=True,
        consistency_check=True,
    )
    assert "Consistency check" in prompt
    assert_no_leakage(prompt)


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


def test_few_shot_examples_come_from_development_subjects():
    """Los ejemplos usan medianas de S2-S4, nunca estadísticos del test."""
    from sources.phase2.development_split import DEV_SUBJECTS
    from sources.phase2.prompt_builder import (
        FEW_SHOT_EXEMPLARS,
        FEW_SHOT_PROVENANCE,
    )

    assert FEW_SHOT_PROVENANCE["subjects"] == list(DEV_SUBJECTS)
    assert FEW_SHOT_PROVENANCE["statistic"].startswith("median")
    # Valores de desarrollo: stress con SCR alto y baseline/amusement/meditation ~0.
    assert FEW_SHOT_EXEMPLARS["stress"]["eda_scr_count"] == 4
    for label in ("baseline", "amusement", "meditation"):
        assert FEW_SHOT_EXEMPLARS[label]["eda_scr_count"] == 0
    # La respiración separa meditation del resto.
    assert FEW_SHOT_EXEMPLARS["meditation"]["resp_rate_bpm"] == 12.8
    assert FEW_SHOT_EXEMPLARS["baseline"]["resp_rate_bpm"] == 17.6
    # Guarda anti-fuga: si alguien reintroduce las medias de las 705 ventanas
    # evaluadas (HR baseline 72.4, EDA baseline 3.9), el test falla.
    assert FEW_SHOT_EXEMPLARS["baseline"]["ecg_hr_bpm"] != 72.4
    assert FEW_SHOT_EXEMPLARS["baseline"]["eda_mean_us"] != 3.9


def test_few_shot_values_match_development_features():
    """Los valores del prompt coinciden con las medianas recalculadas en S2-S4."""
    import json
    from pathlib import Path

    import numpy as np
    import pytest

    from sources.phase2.development_split import DEV_SUBJECTS
    from sources.phase2.prompt_builder import FEW_SHOT_EXEMPLARS

    ingest = Path("results/output/WESAD/INGEST")
    if not (ingest / "windows" / "S2.windows.jsonl").is_file():
        pytest.skip("corpus WESAD no disponible")

    modality = {
        "ecg_hr_bpm": "ecg",
        "ecg_rmssd_ms": "ecg",
        "eda_mean_us": "eda",
        "eda_scr_count": "eda",
        "resp_rate_bpm": "resp",
    }
    values: dict[str, dict[str, list[float]]] = {}
    for subject in DEV_SUBJECTS:
        windows = {
            json.loads(line)["window_id"]: json.loads(line)["label"]["name"]
            for line in (ingest / "windows" / f"{subject}.windows.jsonl").open()
        }
        for line in (ingest / "features" / f"{subject}.features.jsonl").open():
            record = json.loads(line)
            label = windows.get(record["window_id"])
            if not label:
                continue
            for name, mod in modality.items():
                value = record["features"].get(mod, {}).get(name, {}).get("value")
                if value is not None:
                    values.setdefault(label, {}).setdefault(name, []).append(value)

    for label, exemplar in FEW_SHOT_EXEMPLARS.items():
        for name, expected in exemplar.items():
            median = float(np.median(values[label][name]))
            # Comparamos con el redondeo publicado en el prompt.
            assert float(f"{median:.1f}") == pytest.approx(expected, abs=0.1), (
                label,
                name,
            )


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


# ---------------------------------------------------------------------------
# Prompt binario stress vs non-stress (P2)
# ---------------------------------------------------------------------------

from sources.phase2.response_schema import DECISION_MODE_STRESS_BINARY  # noqa: E402


def test_build_numeric_prompt_binary_mode_schema_and_leakage():
    payload = _make_numeric_payload()
    prompt = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        num_ctx=32768,
        use_few_shot=True,
        include_additional=True,
        standardized=True,
        decision_mode=DECISION_MODE_STRESS_BINARY,
    )
    assert '"decision": "stress" | "non-stress" | "abstain"' in prompt
    assert "acute stress response" in prompt
    # El prompt binario no ofrece las etiquetas de 4 clases como decisión.
    assert '"amusement" | "meditation"' not in prompt
    assert "Decision: non-stress" in prompt
    assert_no_leakage(prompt)


def test_build_text_prompt_binary_mode():
    from sources.phase2.prompt_builder import build_text_prompt

    prompt = build_text_prompt(
        text_representation="ecg_hr_bpm: 72 bpm\neda_mean_us: 4 uS",
        window_id_opaque="win_x",
        num_ctx=32768,
        decision_mode=DECISION_MODE_STRESS_BINARY,
    )
    assert '"stress" | "non-stress" | "abstain"' in prompt
    assert "Calm baseline" not in prompt
    assert_no_leakage(prompt)


def test_invalid_decision_mode_raises_in_prompt():
    import pytest

    with pytest.raises(ValueError, match="decision_mode"):
        build_numeric_prompt(
            payload=_make_numeric_payload(),
            window_id_opaque="win_x",
            decision_mode="ternary",
        )


# ---------------------------------------------------------------------------
# Rúbrica de confianza graduada (P4)
# ---------------------------------------------------------------------------


def test_graded_confidence_rubric_is_optional_and_leakage_free():
    payload = _make_numeric_payload()
    base = build_numeric_prompt(
        payload=payload, window_id_opaque="win_x", standardized=True
    )
    graded = build_numeric_prompt(
        payload=payload,
        window_id_opaque="win_x",
        standardized=True,
        graded_confidence=True,
    )
    assert "Confidence guidance" not in base
    assert "Confidence guidance" in graded
    assert "90-100" in graded
    assert "do not default" in graded
    assert_no_leakage(graded)
