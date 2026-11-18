"""Tests del análisis de degradación en el diseño primario (variante y temperatura).

El corpus primario es binario ``schmidt`` **sin** temperatura: 424 ventanas por
condición. Evaluarlo con la variante ``all`` (que incluye ``meditation``) o con
la referencia features-only que sí ve temperatura mezclaría dos diseños.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_degradation as ad  # noqa: E402


def _record(window_id: str, decision: str, confidence: int = 80, **extra):
    return {
        "window_id": window_id,
        "window_id_opaque": f"win_{window_id}",
        "condition_id": "C0_clean",
        "subject": "S5",
        "decision": decision,
        "confidence": confidence,
        "status": "valid_response",
        "claims": [],
        "reasoning": "",
        "prompt": "",
        **extra,
    }


def test_infer_variant_defaults_to_all_for_legacy_corpus():
    """Los corpus sin procedencia de variante eran ``all`` (4 clases mapeadas)."""
    assert ad.infer_variant([{"window_id": "w1"}]) == "all"
    assert ad.infer_variant([]) == "all"


def test_infer_variant_reads_provenance():
    records = [{"nonstress_variant": "schmidt"}, {"nonstress_variant": "baseline"}]
    assert ad.infer_variant(records) == "schmidt"


def test_infer_variant_ignores_unknown_values():
    assert ad.infer_variant([{"nonstress_variant": "inventada"}]) == "all"


def test_infer_include_temperature():
    assert ad.infer_include_temperature([{}]) is True
    assert ad.infer_include_temperature([{"include_temperature": False}]) is False
    assert ad.infer_include_temperature([{"include_temperature": True}]) is True


def test_filter_primary_variant_drops_meditation():
    records = [{"window_id": "w1"}, {"window_id": "w2"}, {"window_id": "w3"}]
    labels = {"w1": "stress", "w2": "baseline", "w3": "meditation"}

    kept = ad.filter_primary_variant(records, labels, "schmidt")

    assert [r["window_id"] for r in kept] == ["w1", "w2"]


def test_filter_primary_variant_all_keeps_every_class():
    records = [{"window_id": "w1"}, {"window_id": "w2"}, {"window_id": "w3"}]
    labels = {"w1": "stress", "w2": "baseline", "w3": "meditation"}

    kept = ad.filter_primary_variant(records, labels, "all")

    assert len(kept) == 3


def test_filter_primary_variant_baseline_only_keeps_baseline_and_stress():
    records = [{"window_id": "w1"}, {"window_id": "w2"}, {"window_id": "w3"}]
    labels = {"w1": "stress", "w2": "baseline", "w3": "amusement"}

    kept = ad.filter_primary_variant(records, labels, "baseline")

    assert [r["window_id"] for r in kept] == ["w1", "w2"]


def test_analyze_group_binary_reports_accuracy_including_abstention():
    """Con 2 abstenciones de 6 ventanas: acc 1.000 sobre decididas, 0.667 con acc*."""
    records = [
        _record("w1", "stress"),
        _record("w2", "stress"),
        _record("w3", "non-stress"),
        _record("w4", "non-stress"),
        _record("w5", "abstain"),
        _record("w6", "abstain"),
    ]
    labels = {
        "w1": "stress",
        "w2": "stress",
        "w3": "baseline",
        "w4": "amusement",
        "w5": "stress",
        "w6": "baseline",
    }

    block = ad.analyze_group_binary(records, labels)

    assert block["n"] == 6
    assert block["n_abstain"] == 2
    assert block["accuracy"] == pytest.approx(1.0)
    assert block["accuracy_incl_abstain"] == pytest.approx(4 / 6)
    assert block["coverage"] == pytest.approx(4 / 6)
    assert block["balanced_accuracy"] == pytest.approx(1.0)


def test_incremental_quality_auroc_understands_binary_decisions():
    """H6 debe evaluar ``stress`` y ``non-stress``, no solo ``stress``."""
    records = []
    labels = {}
    for index in range(40):
        correct = index % 2 == 0
        truth = "stress" if index % 4 == 0 else "baseline"
        predicted_stress = (truth == "stress") == correct
        decision = "stress" if predicted_stress else "non-stress"
        window_id = f"w{index}"
        records.append(
            _record(
                window_id,
                decision,
                confidence=90 if correct else 55,
                subject=f"S{5 + index % 6}",
            )
        )
        labels[window_id] = truth
    quality = {record["window_id"]: 0.9 for record in records}

    block = ad.incremental_quality_auroc(
        records, labels, quality, decision_mode="binary"
    )

    assert block, "el bloque no puede quedar vacío en modo binario"
    assert 0.0 <= block["auroc_confidence"] <= 1.0


def test_h3_comparison_reports_coverage_and_invalid():
    """El contraste H3 debe mostrar cobertura e inválidas de ambos brazos.

    Sin ellas, una diferencia de accuracy entre representaciones puede ser solo
    un subconjunto distinto de ventanas: el brazo textual produce inválidas al
    copiar "not available" en los claims (155 en el 7b, concentradas en C4).
    """
    from pathlib import Path

    source = Path("scripts/analyze_degradation.py").read_text(encoding="utf-8")
    assert '"coverage": base.get("coverage")' in source
    assert '"coverage": other.get("coverage")' in source
    assert '"n_invalid": other.get("n_invalid")' in source
    assert "'cob_num':>8s" in source and "'inv_txt':>8s" in source
