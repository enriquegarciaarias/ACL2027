"""Tests de la deriva de decisión bajo degradación (`sources/phase3/drift.py`).

Cubren las tres piezas que el módulo añade sobre los informes por condición:
marginal frente al prior (con tasa de reclamación), matriz pareada con
descomposición del error (inducido frente a preexistente) y flujos de
abstención, más el cruce con claims en las ventanas que derivan.
"""

from __future__ import annotations

import pytest

from sources.phase3.drift import (
    build_decision_table,
    coherence_summary,
    normalize_decision,
    prediction_marginal,
    resolve_decision_mode,
    table_from_predictions,
    transition_analysis,
    truth_in_space,
    wilson_interval,
    window_groups,
)

SCHMIDT = ("baseline", "amusement")


def _record(
    window: str,
    condition: str,
    decision: str,
    *,
    subject: str = "S5",
    status: str = "valid_response",
    confidence: int | None = 85,
) -> dict:
    return {
        "window_id_opaque": window,
        "condition_id": condition,
        "decision": decision,
        "confidence": confidence,
        "status": status,
        "subject": subject,
    }


def _table(rows: list[dict], classes=("stress", "non-stress")) -> dict:
    """Tabla construida a mano (misma forma que ``build_decision_table``)."""
    return {
        "decision_mode": "stress_binary",
        "classes": list(classes),
        "windows": {
            row["window_id"]: {
                "truth": row["truth"],
                "subject": row.get("subject", "S5"),
                "decisions": row["decisions"],
                "confidence": {},
                "n_reps": {},
            }
            for row in rows
        },
    }


# ---------------------------------------------------------------------------
# Espacios de decisión
# ---------------------------------------------------------------------------
def test_resolve_decision_mode_accepts_cli_aliases():
    assert resolve_decision_mode("binary") == "stress_binary"
    assert resolve_decision_mode("4class") == "affective"
    assert resolve_decision_mode("stress_binary") == "stress_binary"
    assert resolve_decision_mode(None) == "stress_binary"


def test_resolve_decision_mode_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_decision_mode("sleep4")


def test_truth_in_space_binary_schmidt_excludes_meditation():
    assert truth_in_space("stress", "binary", SCHMIDT) == "stress"
    assert truth_in_space("baseline", "binary", SCHMIDT) == "non-stress"
    assert truth_in_space("amusement", "binary", SCHMIDT) == "non-stress"
    # Fuera del endpoint de la variante.
    assert truth_in_space("meditation", "binary", SCHMIDT) is None


def test_truth_in_space_affective_keeps_four_classes():
    for label in ("baseline", "stress", "amusement", "meditation"):
        assert truth_in_space(label, "affective") == label
    assert truth_in_space("recovery", "affective") is None
    assert truth_in_space(None, "affective") is None


def test_normalize_decision_aliases():
    assert normalize_decision("Non_Stress") == "non-stress"
    assert normalize_decision("STRESS") == "stress"
    assert normalize_decision("abstention") == "abstain"
    assert normalize_decision("baseline") == "baseline"
    assert normalize_decision("tal vez") is None
    assert normalize_decision(None) is None


# ---------------------------------------------------------------------------
# Tabla de decisiones
# ---------------------------------------------------------------------------
def test_build_table_drops_windows_outside_variant():
    records = [
        _record("w1", "C0_clean", "stress"),
        _record("w2", "C0_clean", "non-stress"),
    ]
    labels = {"w1": "stress", "w2": "meditation"}
    table = build_decision_table(
        records, labels, decision_mode="binary", nonstress_classes=SCHMIDT
    )
    assert table["n_windows"] == 1
    assert table["dropped"]["fuera_de_la_variante_o_sin_etiqueta"] == 1


def test_build_table_excludes_gate_abstentions_by_default():
    records = [
        _record("w1", "C0_clean", "stress"),
        _record("w1", "C4_total_eda", "abstain", status="gate_abstention"),
    ]
    labels = {"w1": "stress"}
    default = build_decision_table(
        records, labels, decision_mode="binary", nonstress_classes=SCHMIDT
    )
    assert "C4_total_eda" not in default["windows"]["w1"]["decisions"]
    assert default["dropped"]["estado_no_usable"] == 1

    gated = build_decision_table(
        records,
        labels,
        decision_mode="binary",
        nonstress_classes=SCHMIDT,
        include_gate=True,
    )
    assert gated["windows"]["w1"]["decisions"]["C4_total_eda"] == "abstain"


def test_build_table_votes_across_repetitions():
    records = [
        _record("w1", "C0_clean", "stress"),
        _record("w1", "C0_clean", "stress"),
        _record("w1", "C0_clean", "non-stress"),
    ]
    table = build_decision_table(
        records, {"w1": "stress"}, decision_mode="binary", nonstress_classes=SCHMIDT
    )
    window = table["windows"]["w1"]
    assert window["decisions"]["C0_clean"] == "stress"
    assert window["n_reps"]["C0_clean"] == 3


def test_build_table_counts_unusable_records():
    records = [
        _record("w1", "C0_clean", "stress"),
        _record("w1", "C4_total_eda", "stress", status="invalid_output"),
        _record("w1", "C4_total_eda", "quizá"),
    ]
    table = build_decision_table(
        records, {"w1": "stress"}, decision_mode="binary", nonstress_classes=SCHMIDT
    )
    assert table["dropped"]["estado_no_usable"] == 1
    assert table["dropped"]["decision_desconocida"] == 1


# ---------------------------------------------------------------------------
# Marginal y tasa de reclamación
# ---------------------------------------------------------------------------
def test_prediction_marginal_and_claim_rate():
    table = _table(
        [
            {"window_id": "w1", "truth": "stress", "decisions": {"C0": "stress"}},
            {"window_id": "w2", "truth": "non-stress", "decisions": {"C0": "stress"}},
            {"window_id": "w3", "truth": "non-stress", "decisions": {"C0": "non-stress"}},
            {"window_id": "w4", "truth": "stress", "decisions": {"C0": "abstain"}},
        ]
    )
    marginal = prediction_marginal(table, ["C0"])["conditions"]["C0"]
    assert marginal["n"] == 4
    assert marginal["n_decided"] == 3
    assert marginal["abstain_share"] == pytest.approx(1 / 4)
    # Decididas: 2 stress de 3.
    assert marginal["predicted_decided"]["stress"] == pytest.approx(2 / 3)
    assert marginal["truth"]["stress"] == pytest.approx(2 / 4)
    # Tasa de reclamación: (2/3) / (2/4) = 4/3.
    assert marginal["claim_rate"]["stress"] == pytest.approx(4 / 3)
    assert marginal["claim_rate"]["non-stress"] == pytest.approx((1 / 3) / (1 / 2))


def test_prediction_marginal_claim_rate_none_without_prior():
    table = _table(
        [
            {"window_id": "w1", "truth": "non-stress", "decisions": {"C0": "non-stress"}},
        ]
    )
    marginal = prediction_marginal(table, ["C0"])["conditions"]["C0"]
    assert marginal["claim_rate"]["stress"] is None


# ---------------------------------------------------------------------------
# Transiciones
# ---------------------------------------------------------------------------
def _mixed_table() -> dict:
    return _table(
        [
            # Acierta en limpio, falla al degradar.
            {"window_id": "w1", "truth": "stress", "subject": "S5",
             "decisions": {"C0_clean": "stress", "C4_total_eda": "non-stress"}},
            # Falla en las dos, con la misma clase equivocada.
            {"window_id": "w2", "truth": "stress", "subject": "S5",
             "decisions": {"C0_clean": "non-stress", "C4_total_eda": "non-stress"}},
            # Estable correcta.
            {"window_id": "w3", "truth": "non-stress", "subject": "S6",
             "decisions": {"C0_clean": "non-stress", "C4_total_eda": "non-stress"}},
            # Acierta en limpio y se rompe hacia stress.
            {"window_id": "w4", "truth": "non-stress", "subject": "S6",
             "decisions": {"C0_clean": "non-stress", "C4_total_eda": "stress"}},
            # Se abstenía en limpio y decide al degradar.
            {"window_id": "w5", "truth": "stress", "subject": "S6",
             "decisions": {"C0_clean": "abstain", "C4_total_eda": "stress"}},
            # Decidía en limpio y se abstiene al degradar.
            {"window_id": "w6", "truth": "non-stress", "subject": "S5",
             "decisions": {"C0_clean": "non-stress", "C4_total_eda": "abstain"}},
        ]
    )


def test_transition_decomposition_separates_induced_from_preexisting():
    report = transition_analysis(_mixed_table(), base="C0_clean")
    entry = report["conditions"]["C4_total_eda"]
    decomposition = entry["decomposition"]
    assert entry["n_windows_both"] == 6
    assert entry["n_decided_pairs"] == 4
    assert decomposition["correct_correct"] == 1
    assert decomposition["correct_wrong"] == 2
    assert decomposition["wrong_correct"] == 0
    assert decomposition["wrong_wrong"] == 1
    assert decomposition["induced_error_share"] == pytest.approx(2 / 3)
    assert decomposition["preexisting_error_share"] == pytest.approx(1 / 3)
    assert decomposition["repaired_share"] == pytest.approx(0.0)


def test_transition_matrix_and_direction():
    report = transition_analysis(_mixed_table(), base="C0_clean")
    entry = report["conditions"]["C4_total_eda"]
    # La matriz completa incluye la abstención como estado.
    assert entry["matrix_all"]["abstain"] == {"stress": 1}
    assert entry["matrix_all"]["stress"] == {"non-stress": 1}
    assert entry["matrix_all"]["non-stress"] == {
        "non-stress": 2,
        "stress": 1,
        "abstain": 1,
    }
    # La matriz decidida no.
    assert entry["matrix_decided"]["stress"] == {"non-stress": 1}
    assert entry["n_changed"] == 2
    assert entry["changed_share"] == pytest.approx(2 / 4)
    assert {(item["from"], item["to"], item["n"]) for item in entry["direction"]} == {
        ("stress", "non-stress", 1),
        ("non-stress", "stress", 1),
    }


def test_transition_by_truth_destination():
    report = transition_analysis(_mixed_table(), base="C0_clean")
    by_truth = report["conditions"]["C4_total_eda"]["by_truth"]
    # De las ventanas realmente de estrés, una acierta en limpio y se rompe.
    assert by_truth["stress"]["drifted_from_correct"] == 1
    assert by_truth["stress"]["drifted_destination"] == {"non-stress": 1}
    assert by_truth["stress"]["destination_of_errors"] == {"non-stress": 2}
    assert by_truth["non-stress"]["drifted_destination"] == {"stress": 1}


def test_transition_abstain_flow():
    report = transition_analysis(_mixed_table(), base="C0_clean")
    flow = report["conditions"]["C4_total_eda"]["abstain_flow"]
    assert flow["abstain_to_stress"] == 1
    assert flow["non-stress_to_abstain"] == 1


def test_transition_counts_missing_base_and_ignores_absent_condition():
    table = _table(
        [
            {"window_id": "w1", "truth": "stress", "decisions": {"C0_clean": "stress"}},
        ]
    )
    report = transition_analysis(table, base="C0_clean", conditions=["C0_clean", "C9"])
    assert report["conditions"] == {}
    assert report["missing_base"] == 0


def test_transition_by_subject_counts():
    report = transition_analysis(_mixed_table(), base="C0_clean")
    by_subject = report["conditions"]["C4_total_eda"]["by_subject"]
    assert by_subject["S5"]["n_pairs"] == 2
    assert by_subject["S6"]["n_pairs"] == 2
    assert by_subject["S5"]["n_correct_wrong"] == 1
    assert by_subject["S6"]["n_correct_wrong"] == 1
    assert by_subject["S5"]["n_wrong_wrong"] == 1


def test_table_from_predictions_reproduces_transition_tables():
    llm = _mixed_table()
    predictions = {
        condition: {
            window_id: window["decisions"][condition]
            for window_id, window in llm["windows"].items()
            if condition in window["decisions"]
        }
        for condition in ("C0_clean", "C4_total_eda")
    }
    truth = {window_id: window["truth"] for window_id, window in llm["windows"].items()}
    reference = table_from_predictions(predictions, truth, decision_mode="binary")
    expected = transition_analysis(llm, base="C0_clean")
    got = transition_analysis(reference, base="C0_clean")
    assert (
        got["conditions"]["C4_total_eda"]["decomposition"]
        == expected["conditions"]["C4_total_eda"]["decomposition"]
    )
    assert (
        got["conditions"]["C4_total_eda"]["matrix_decided"]
        == expected["conditions"]["C4_total_eda"]["matrix_decided"]
    )


def test_window_groups_partition_by_effect():
    groups = window_groups(_mixed_table(), base="C0_clean", condition="C4_total_eda")
    assert groups["drifted"] == ["w1", "w4"]
    assert groups["stable_correct"] == ["w3"]
    assert groups["stable_wrong"] == ["w2"]
    assert groups["abstained_before"] == ["w5"]
    assert groups["abstained_after"] == ["w6"]


# ---------------------------------------------------------------------------
# Coherencia de claims
# ---------------------------------------------------------------------------
def test_coherence_summary_empty():
    block = coherence_summary([])
    assert block["n_responses"] == 0
    assert block["insufficient"] is True


def test_coherence_summary_aggregates_confirmation_and_conflict():
    blocks = [
        {
            "n_signature_claims": 2,
            "n_confirming": 2,
            "n_against": 0,
            "has_confirmation": True,
            "evidence_conflict": False,
            "n_supported": 2,
            "n_against_supported": 0,
        },
        {
            "n_signature_claims": 1,
            "n_confirming": 0,
            "n_against": 1,
            "has_confirmation": False,
            "evidence_conflict": True,
            "n_supported": 1,
            "n_against_supported": 1,
        },
    ]
    block = coherence_summary(blocks, min_responses=2)
    assert block["insufficient"] is False
    assert block["coherence_rate"] == pytest.approx(2 / 3)
    assert block["zero_confirmation_share"] == pytest.approx(1 / 2)
    assert block["evidence_conflict_rate"] == pytest.approx(1 / 2)
    assert block["against_supported_share"] == pytest.approx(1 / 3)


def test_wilson_interval_bounds():
    low, high = wilson_interval(0, 10)
    assert 0.0 <= low <= high < 0.35
    low, high = wilson_interval(10, 10)
    assert 0.65 < low <= high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Cruce de claims del script
# ---------------------------------------------------------------------------
PROMPT = """Measurements for this window.

- ecg_hr_bpm: 1.5 SD
- eda_mean_us: 1.2 SD
"""


def _response(
    window: str,
    condition: str,
    decision: str,
    *,
    prompt: str = PROMPT,
    status: str = "valid_response",
) -> dict:
    return {
        "window_id_opaque": window,
        "condition_id": condition,
        "decision": decision,
        "status": status,
        "prompt": prompt,
        "claims": [
            {"metric": "ecg_hr_bpm", "channel": "ecg", "direction": "increase",
             "magnitude": "high"},
            {"metric": "eda_mean_us", "channel": "eda", "direction": "increase",
             "magnitude": "high"},
        ],
    }


def test_claims_by_group_uses_only_the_condition_and_group():
    from scripts.analyze_drift import claims_by_group

    records = [
        _response("w1", "C4_total_eda", "stress"),
        _response("w2", "C4_total_eda", "stress"),
        # Otra condición: no debe entrar.
        _response("w1", "C0_clean", "non-stress"),
        # Sin prompt: se ignora.
        _response("w3", "C4_total_eda", "stress", prompt=""),
    ]
    block = claims_by_group(
        records,
        {"drifted": ["w1", "w2", "w3"], "stable_correct": []},
        "C4_total_eda",
        min_responses=2,
    )
    assert block["drifted"]["n_responses"] == 2
    # Ambos claims (HR y EDA al alza) apoyan `stress`.
    assert block["drifted"]["coherence_rate"] == pytest.approx(1.0)
    assert block["drifted"]["against_supported_share"] == pytest.approx(0.0)
    assert "stable_correct" not in block


def test_claims_by_group_reports_contradiction():
    from scripts.analyze_drift import claims_by_group

    records = [_response("w1", "C4_total_eda", "non-stress")]
    block = claims_by_group(
        records, {"drifted": ["w1"]}, "C4_total_eda", min_responses=1
    )
    # Con decisión `non-stress`, dos claims al alza son contrarios.
    assert block["drifted"]["coherence_rate"] == pytest.approx(0.0)
