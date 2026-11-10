"""Tests de las métricas a nivel de registro (Fase 2/3)."""

from __future__ import annotations

from sources.phase3.records import evaluate_records

LABELS = {
    "op1": "baseline",
    "op2": "stress",
    "op3": "baseline",
    "op4": "stress",
}


def _record(opaque: str, decision: str | None, status: str = "valid_response", conf: int = 80):
    return {
        "window_id_opaque": opaque,
        "status": status,
        "decision": decision,
        "confidence": conf,
    }


def test_accuracy_excludes_abstentions():
    records = [
        _record("op1", "baseline"),
        _record("op2", "stress"),
        _record("op3", "abstain"),
        _record("op4", "abstain"),
    ]
    metrics = evaluate_records(records, LABELS)
    assert metrics["n_valid"] == 4
    assert metrics["n_abstain"] == 2
    assert metrics["n_decided"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["abstention_rate"] == 0.5


def test_invalid_outputs_are_reported_separately():
    records = [
        _record("op1", "baseline"),
        _record("op2", "stress"),
        _record("op3", None, status="invalid_output"),
        _record("op4", None, status="timeout"),
    ]
    metrics = evaluate_records(records, LABELS)
    assert metrics["n"] == 4
    assert metrics["n_invalid"] == 1
    assert metrics["n_valid"] == 3


def test_binary_subset_uses_only_stress_and_baseline():
    records = [
        _record("op1", "baseline"),
        _record("op2", "baseline"),
        _record("op3", "baseline"),
        _record("op4", "stress"),
    ]
    metrics = evaluate_records(records, LABELS)
    binary = metrics["binary"]
    assert binary["n"] == 4
    assert binary["tp"] == 1 and binary["fn"] == 1
    assert binary["tn"] == 2 and binary["fp"] == 0
    assert binary["recall_stress"] == 0.5


def test_unknown_label_counts_as_error():
    records = [_record("desconocido", "baseline")]
    metrics = evaluate_records(records, LABELS)
    assert metrics["n_decided"] == 1
    assert metrics["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Endpoint binario stress vs non-stress (P2)
# ---------------------------------------------------------------------------

from sources.phase3.records import evaluate_binary_records  # noqa: E402

BINARY_LABELS = {
    "w1": "stress",
    "w2": "baseline",
    "w3": "amusement",
    "w4": "meditation",
    "w5": "stress",
}


def _bin_record(opaque, decision, status="valid_response", conf=80):
    return {
        "window_id_opaque": opaque,
        "status": status,
        "decision": decision,
        "confidence": conf,
    }


def test_binary_metrics_map_other_classes_to_non_stress():
    records = [
        _bin_record("w1", "stress"),
        _bin_record("w2", "non-stress"),
        _bin_record("w3", "non-stress"),
        _bin_record("w4", "non-stress"),
        _bin_record("w5", "non-stress"),
    ]
    m = evaluate_binary_records(records, BINARY_LABELS)
    assert m["n"] == 5
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 3, 0)
    assert m["accuracy"] == 0.8
    assert m["recall_stress"] == 0.5
    assert m["specificity"] == 1.0


def test_binary_metrics_count_abstentions_and_invalids():
    records = [
        _bin_record("w1", "stress"),
        _bin_record("w2", "abstain", status="valid_abstention"),
        _bin_record("w3", None, status="invalid_output"),
    ]
    m = evaluate_binary_records(records, BINARY_LABELS)
    assert m["n_valid"] == 2
    assert m["n_invalid"] == 1
    assert m["n_abstain"] == 1
    assert m["n_decided"] == 1
    assert m["coverage"] == 1 / 3


def test_binary_metrics_auroc_perfect_ranking():
    # stress predicho con confianza alta; no-stress con confianza alta en la
    # dirección opuesta => score de stress perfectamente ordenado.
    records = [
        _bin_record("w1", "stress", conf=90),
        _bin_record("w5", "stress", conf=80),
        _bin_record("w2", "non-stress", conf=90),
        _bin_record("w3", "non-stress", conf=80),
        _bin_record("w4", "non-stress", conf=70),
    ]
    m = evaluate_binary_records(records, BINARY_LABELS)
    assert m["auroc_stress"] == 1.0
    assert m["brier"] < 0.1


def test_binary_metrics_exclude_unknown_labels():
    records = [_bin_record("desconocido", "stress")]
    m = evaluate_binary_records(records, BINARY_LABELS)
    assert m["n"] == 0
    assert m["n_unknown_label"] == 1


def test_binary_metrics_map_other_predictions():
    records = [
        _bin_record("w1", "stress"),
        _bin_record("w2", "baseline"),      # 4 clases reinterpretado
        _bin_record("w4", "meditation"),    # 4 clases reinterpretado
        _bin_record("w5", "amusement"),     # 4 clases reinterpretado
    ]
    mapped = {"baseline": "non-stress", "amusement": "non-stress", "meditation": "non-stress"}
    m = evaluate_binary_records(records, BINARY_LABELS, other_decisions_map=mapped)
    assert m["n_decided"] == 4
    assert m["tp"] == 1 and m["fn"] == 1
    assert m["tn"] == 2 and m["fp"] == 0
    assert m["n_abstain"] == 0
