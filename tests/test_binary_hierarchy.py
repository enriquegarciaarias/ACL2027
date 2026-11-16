"""Tests de la jerarquía binaria stress vs non-stress (Fase 1)."""

from __future__ import annotations

import pytest

from sources.phase2.binary_variants import (
    DEFAULT_NONSTRESS_VARIANT,
    NONSTRESS_VARIANTS,
    variant_classes,
)
from sources.phase3.binary_hierarchy import (
    base_rate,
    condition_hierarchy_report,
    evaluate_variant,
    filter_records,
    grouped_bootstrap_ci,
    hierarchy_report,
    provenance,
)


def _record(
    subject: str,
    window: str,
    label: str,
    decision: str,
    condition: str = "C0_clean",
    confidence: int = 85,
) -> dict:
    return {
        "subject": subject,
        "window_id": f"WESAD:{subject}:{window}",
        "window_id_opaque": f"win_{window}",
        "condition_id": condition,
        "decision": decision,
        "confidence": confidence,
        "status": "valid_response",
        "split": "evaluation",
        "label": label,
    }


def _labels(records: list[dict]) -> dict[str, str]:
    return {record["window_id_opaque"]: record["label"] for record in records}


def _sample() -> list[dict]:
    return [
        _record("S5", "w1", "stress", "stress"),
        _record("S5", "w2", "stress", "stress"),
        _record("S5", "w3", "baseline", "non-stress"),
        _record("S5", "w4", "baseline", "non-stress"),
        _record("S5", "w5", "amusement", "non-stress"),
        # Esta ventana es un falso positivo si se incluye (variante all).
        _record("S5", "w6", "meditation", "stress"),
    ]


def test_default_variant_is_schmidt_and_declares_four_variants():
    assert DEFAULT_NONSTRESS_VARIANT == "schmidt"
    assert set(NONSTRESS_VARIANTS) == {
        "schmidt",
        "baseline",
        "no_amusement",
        "all",
    }
    assert variant_classes("schmidt") == ("baseline", "amusement")
    assert variant_classes("baseline") == ("baseline",)
    assert variant_classes("no_amusement") == ("baseline", "meditation")
    assert variant_classes("all") == ("baseline", "amusement", "meditation")


def test_invalid_variant_raises():
    with pytest.raises(ValueError, match="variante binaria"):
        variant_classes("ternary")


def test_filter_drops_meditation_in_schmidt_but_not_in_all():
    records = _sample()
    labels = _labels(records)
    schmidt = filter_records(records, labels, "schmidt")
    assert [record["window_id"] for record in schmidt] == [
        "WESAD:S5:w1",
        "WESAD:S5:w2",
        "WESAD:S5:w3",
        "WESAD:S5:w4",
        "WESAD:S5:w5",
    ]
    all_records = filter_records(records, labels, "all")
    assert len(all_records) == 6
    baseline_only = filter_records(records, labels, "baseline")
    assert len(baseline_only) == 4


def test_filter_drops_unknown_labels():
    records = _sample()
    labels = _labels(records)
    labels["win_w1"] = "UNKNOWN"
    kept = filter_records(records, labels, "schmidt")
    assert "WESAD:S5:w1" not in [record["window_id"] for record in kept]


def test_base_rate_depends_on_variant():
    records = _sample()
    labels = _labels(records)
    assert base_rate(filter_records(records, labels, "schmidt"), labels) == pytest.approx(2 / 5)
    assert base_rate(filter_records(records, labels, "baseline"), labels) == pytest.approx(2 / 4)
    assert base_rate(filter_records(records, labels, "all"), labels) == pytest.approx(2 / 6)


def test_evaluate_variant_schmidt_is_clean():
    records = _sample()
    labels = _labels(records)
    metrics = evaluate_variant(records, labels, "schmidt")
    assert metrics["n"] == 5
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["precision_stress"] == pytest.approx(1.0)
    assert metrics["recall_stress"] == pytest.approx(1.0)
    assert metrics["base_rate"] == pytest.approx(2 / 5)
    assert metrics["n_dropped"] == 1


def test_evaluate_variant_all_counts_the_meditation_false_positive():
    records = _sample()
    labels = _labels(records)
    metrics = evaluate_variant(records, labels, "all")
    assert metrics["n"] == 6
    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["precision_stress"] == pytest.approx(2 / 3)
    assert metrics["base_rate"] == pytest.approx(2 / 6)


def _two_subjects_opposite() -> list[dict]:
    """S5 siempre acierta; S6 siempre falla (para probar el agrupamiento)."""
    return [
        _record("S5", "a1", "stress", "stress"),
        _record("S5", "a2", "stress", "stress"),
        _record("S5", "a3", "baseline", "non-stress"),
        _record("S5", "a4", "baseline", "non-stress"),
        _record("S6", "b1", "stress", "non-stress"),
        _record("S6", "b2", "stress", "non-stress"),
        _record("S6", "b3", "baseline", "stress"),
        _record("S6", "b4", "baseline", "stress"),
    ]


def test_grouped_bootstrap_is_deterministic_and_brackets_the_point():
    records = _two_subjects_opposite()
    labels = _labels(records)
    first = grouped_bootstrap_ci(
        records, labels, "schmidt", metrics=("accuracy",), n_boot=200, seed=7
    )
    second = grouped_bootstrap_ci(
        records, labels, "schmidt", metrics=("accuracy",), n_boot=200, seed=7
    )
    assert first == second
    entry = first["accuracy"]
    assert entry["point"] == pytest.approx(0.5)
    assert entry["lo"] <= entry["point"] <= entry["hi"]
    assert entry["lo"] < entry["hi"]
    assert entry["n"] > 0


def test_grouped_bootstrap_changes_with_seed_but_stays_in_range():
    records = _two_subjects_opposite()
    labels = _labels(records)
    a = grouped_bootstrap_ci(
        records, labels, "schmidt", metrics=("accuracy",), n_boot=100, seed=1
    )["accuracy"]
    b = grouped_bootstrap_ci(
        records, labels, "schmidt", metrics=("accuracy",), n_boot=100, seed=2
    )["accuracy"]
    assert 0.0 <= a["lo"] <= a["hi"] <= 1.0
    assert 0.0 <= b["lo"] <= b["hi"] <= 1.0
    assert a["point"] == b["point"] == pytest.approx(0.5)


def test_bootstrap_point_matches_point_estimate_with_invalid_record():
    records = _sample() + [_record("S5", "w7", "stress", "stress")]
    records[-1]["status"] = "invalid_output"
    labels = _labels(records)
    point = evaluate_variant(records, labels, "schmidt")["accuracy"]
    boot = grouped_bootstrap_ci(
        records, labels, "schmidt", metrics=("accuracy",), n_boot=10, seed=0
    )["accuracy"]["point"]
    assert point == pytest.approx(boot)


def test_hierarchy_report_has_all_variants():
    records = _sample()
    labels = _labels(records)
    report = hierarchy_report(records, labels, n_boot=20, seed=0)
    assert set(report) == set(NONSTRESS_VARIANTS)
    for variant, block in report.items():
        assert block["variant"] == variant
        assert block["n_records"] == len(filter_records(records, labels, variant))
        assert "bootstrap" in block


def test_condition_hierarchy_report_groups_and_pools():
    records = _sample()
    for record in records:
        record["condition_id"] = "C0_clean"
    extra = [
        {**record, "condition_id": "C5_total_both"}
        for record in _two_subjects_opposite()
    ]
    labels = _labels(records + extra)
    report = condition_hierarchy_report(
        records + extra,
        labels,
        conditions=["C0_clean", "C5_total_both"],
        variants=["schmidt"],
        n_boot=10,
        seed=0,
    )
    assert set(report) == {"C0_clean", "C5_total_both", "pooled"}
    assert report["C0_clean"]["schmidt"]["n_records"] == 5
    assert report["C5_total_both"]["schmidt"]["n_records"] == 8
    assert report["pooled"]["schmidt"]["n_records"] == 13


def test_provenance_reads_first_record():
    records = _sample()
    records[0].update(
        {
            "model": "qwen3:14b",
            "representation": "numeric",
            "decision_mode": "stress_binary",
            "nonstress_variant": "schmidt",
            "include_temperature": False,
        }
    )
    info = provenance(records)
    assert info["model"] == "qwen3:14b"
    assert info["nonstress_variant"] == "schmidt"
    assert info["include_temperature"] is False
    assert info["n_records"] == 6
