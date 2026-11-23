"""Tests de la firma fisiológica preregistrada (H5 · relevancia).

La tabla está congelada en ``docs/h5_relevancia.md`` (2026-11-21). Estos tests
bloquean cambios silenciosos: mover un signo o sacar una feature del conjunto
primario **exige** actualizar el preregistro y, por tanto, fallar aquí primero.
"""

from __future__ import annotations

import pytest

from sources.phase3.signatures import (
    COMPACT_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_SETS,
    PRIMARY_FEATURES,
    SENSITIVITY_FEATURES,
    SENSITIVITY_SIGNATURE,
    STRESS_SIGNATURE,
    excluded_table,
    feature_set,
    has_signature,
    sign_of,
    signature_table,
)

#: Firma fijada en el preregistro §3.2 (12 features).
EXPECTED_PRIMARY_SIGNS = {
    "eda_mean_us": +1,
    "eda_tonic_mean_us": +1,
    "eda_phasic_std_us": +1,
    "eda_scr_count": +1,
    "eda_scr_rate_per_min": +1,
    "eda_std_us": +1,
    "eda_scr_amplitude_mean_us": +1,
    "eda_scr_amplitude_max_us": +1,
    "ecg_hr_bpm": +1,
    "ecg_sdnn_ms": -1,
    "ecg_rmssd_ms": -1,
    "resp_rate_bpm": +1,
}

#: Las 23 features canónicas que puede ver la vista LLM (``LLM_ALLOWED_FEATURES``
#: de ``represent.py``, replicadas aquí para no arrastrar ``neurokit2``).
EXPECTED_LLM_FEATURES = frozenset(EXPECTED_PRIMARY_SIGNS) | frozenset(
    {
        "eda_slope_us_per_s",
        "ecg_rr_mean_ms",
        "ecg_n_beats",
        "resp_interval_cv",
        "resp_amplitude_pct",
        "resp_std_pct",
        "resp_n_cycles",
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
    }
)


def test_primary_set_has_the_twelve_preregistered_features():
    assert set(PRIMARY_FEATURES) == set(EXPECTED_PRIMARY_SIGNS)
    assert len(PRIMARY_FEATURES) == 12


def test_primary_signs_match_the_preregistration():
    assert {metric: sign_of(metric, PRIMARY_FEATURES) for metric in PRIMARY_FEATURES} == (
        EXPECTED_PRIMARY_SIGNS
    )


def test_primary_features_have_strong_or_moderate_level_and_rationale():
    for metric in PRIMARY_FEATURES:
        signature = STRESS_SIGNATURE[metric]
        assert signature.level in {"strong", "moderate"}
        assert signature.primary is True
        assert signature.rationale.strip()
        assert signature.channel in {"eda", "ecg", "resp"}


def test_excluded_features_have_a_reason_and_no_signature():
    assert len(EXCLUDED_FEATURES) == 10
    for metric, reason in EXCLUDED_FEATURES.items():
        assert reason.strip()
        assert not has_signature(metric)
        assert sign_of(metric) == 0
    # Las que más fácilmente se colarían por parecer informativas.
    for metric in ("ecg_rr_mean_ms", "ecg_n_beats", "resp_n_cycles", "temp_mean_degc"):
        assert metric in EXCLUDED_FEATURES


def test_eight_of_twelve_primary_features_are_eda():
    """Se declara en el preregistro: EDA aporta 8 de 12 (de ahí el conjunto de 5)."""
    eda = [metric for metric in PRIMARY_FEATURES if STRESS_SIGNATURE[metric].channel == "eda"]
    assert len(eda) == 8


def test_compact_set_is_one_feature_per_mechanism_and_subset_of_primary():
    assert COMPACT_FEATURES <= PRIMARY_FEATURES
    assert COMPACT_FEATURES == frozenset(
        {
            "eda_mean_us",
            "eda_phasic_std_us",
            "ecg_hr_bpm",
            "ecg_rmssd_ms",
            "resp_rate_bpm",
        }
    )


def test_sensitivity_set_adds_only_the_weak_signature():
    assert SENSITIVITY_FEATURES == PRIMARY_FEATURES | frozenset(SENSITIVITY_SIGNATURE)
    assert frozenset(SENSITIVITY_SIGNATURE) == frozenset({"resp_interval_cv"})
    assert SENSITIVITY_SIGNATURE["resp_interval_cv"].level == "weak"
    assert sign_of("resp_interval_cv", SENSITIVITY_FEATURES) == +1


def test_sets_partition_the_llm_feature_space():
    """12 primarias + 10 sin firma + 1 de sensibilidad = las 23 que ve el LLM."""
    assert PRIMARY_FEATURES.isdisjoint(EXCLUDED_FEATURES)
    assert PRIMARY_FEATURES.isdisjoint(SENSITIVITY_SIGNATURE)
    assert frozenset(EXCLUDED_FEATURES).isdisjoint(SENSITIVITY_SIGNATURE)
    union = PRIMARY_FEATURES | frozenset(EXCLUDED_FEATURES) | frozenset(SENSITIVITY_SIGNATURE)
    assert union == EXPECTED_LLM_FEATURES
    assert len(union) == 23


def test_sign_of_respects_the_feature_set():
    # ``resp_interval_cv`` tiene firma débil: fuera del primario, dentro de sensibilidad.
    assert sign_of("resp_interval_cv", PRIMARY_FEATURES) == 0
    assert sign_of("resp_interval_cv", SENSITIVITY_FEATURES) == +1
    # Sin filtro, cualquier firma registrada vale.
    assert sign_of("resp_interval_cv") == +1
    # Una feature del primario no está en el conjunto compacto si no se eligió.
    assert sign_of("ecg_sdnn_ms", COMPACT_FEATURES) == 0
    assert sign_of("ecg_sdnn_ms", PRIMARY_FEATURES) == -1


def test_sign_of_unknown_metric_is_zero():
    assert sign_of("no_existe_esta_feature") == 0
    assert has_signature("no_existe_esta_feature") is False


def test_feature_set_lookup_and_error():
    assert feature_set("primary") is PRIMARY_FEATURES
    assert feature_set("compact") is COMPACT_FEATURES
    assert feature_set("sensitivity") is SENSITIVITY_FEATURES
    assert set(FEATURE_SETS) == {"primary", "compact", "sensitivity"}
    with pytest.raises(ValueError):
        feature_set("todas")


def test_tables_are_serializable_for_the_report():
    table = signature_table(PRIMARY_FEATURES)
    assert len(table) == 12
    assert all({"metric", "sign", "level", "rationale"} <= set(row) for row in table)
    excluded = excluded_table()
    assert len(excluded) == 10
    assert all({"metric", "reason"} <= set(row) for row in excluded)
