"""Tests del análisis de repeticiones (varianza de muestreo).

El analizador decide si merece la pena gastar horas de LLM repitiendo prompts.
Sus cantidades clave son la **proporción modal** (¿es determinista?), la
**dispersión de la confianza** (¿hay resolución?) y la **convergencia** de la
métrica con ``k`` repeticiones.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_repetitions as ar  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _record(
    window: str,
    decision: str,
    confidence: int,
    repetition: int,
    condition: str = "C0_clean",
    status: str = "valid_response",
) -> dict:
    return {
        "window_id_opaque": window,
        "condition_id": condition,
        "decision": decision,
        "confidence": confidence,
        "status": status,
        "repetition": repetition,
        "model": "qwen2.5:7b",
        "decision_mode": "stress_binary",
        "nonstress_variant": "schmidt",
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Voto
# ---------------------------------------------------------------------------
def test_vote_takes_the_majority():
    assert ar.vote(["stress", "stress", "non-stress"]) == "stress"


def test_vote_tie_is_abstention():
    """Un empate no se resuelve a favor de ninguna clase."""
    assert ar.vote(["stress", "non-stress"]) == "abstain"
    assert ar.vote(["stress", "stress", "non-stress", "non-stress"]) == "abstain"


def test_vote_single_sample_returns_it():
    assert ar.vote(["non-stress"]) == "non-stress"


# ---------------------------------------------------------------------------
# Estabilidad por ventana
# ---------------------------------------------------------------------------
def test_window_stability_deterministic_model():
    records = [
        _record("w1", "stress", 85, rep) for rep in range(5)
    ]
    stats = ar.window_stability(records)
    assert stats["modal_share"] == 1.0
    assert stats["n_distinct_decisions"] == 1
    assert stats["entropy_bits"] == 0.0
    assert stats["confidence_std"] == 0.0
    assert stats["confidence_n_distinct"] == 1


def test_window_stability_split_decisions():
    records = [
        _record("w1", "stress", 85, 0),
        _record("w1", "stress", 85, 1),
        _record("w1", "non-stress", 75, 2),
    ]
    stats = ar.window_stability(records)
    assert stats["modal_decision"] == "stress"
    assert stats["modal_share"] == pytest.approx(2 / 3)
    assert stats["n_distinct_decisions"] == 2
    assert stats["entropy_bits"] > 0
    assert stats["confidence_std"] > 0


def test_window_stability_ignores_transport_failures():
    records = [
        _record("w1", "stress", 85, 0),
        _record("w1", None, None, 1, status="timeout"),
    ]
    stats = ar.window_stability(records)
    assert stats["n_usable"] == 1
    assert stats["n_invalid"] == 1
    assert stats["modal_share"] == 1.0


def test_window_stability_all_failures_does_not_crash():
    records = [_record("w1", None, None, 0, status="timeout")]
    stats = ar.window_stability(records)
    assert stats["n_usable"] == 0
    assert stats["modal_share"] == 0.0


# ---------------------------------------------------------------------------
# Agregación y voto
# ---------------------------------------------------------------------------
def test_aggregate_window_averages_confidence_of_the_winner():
    records = [
        _record("w1", "stress", 80, 0),
        _record("w1", "stress", 90, 1),
        _record("w1", "non-stress", 60, 2),
    ]
    aggregated = ar.aggregate_window(records, [0, 1, 2])
    assert aggregated["decision"] == "stress"
    assert aggregated["confidence"] == pytest.approx(85.0)
    assert aggregated["status"] == "valid_response"


def test_aggregate_window_tie_becomes_abstention():
    records = [
        _record("w1", "stress", 85, 0),
        _record("w1", "non-stress", 85, 1),
    ]
    aggregated = ar.aggregate_window(records, [0, 1])
    assert aggregated["decision"] == "abstain"
    assert aggregated["status"] == "valid_abstention"


def test_aggregate_window_uses_only_requested_reps():
    records = [_record("w1", "stress", 85, 0), _record("w1", "non-stress", 85, 1)]
    assert ar.aggregate_window(records, [0])["decision"] == "stress"
    assert ar.aggregate_window(records, [1])["decision"] == "non-stress"


def test_aggregate_window_without_usable_records_returns_none():
    records = [_record("w1", None, None, 0, status="timeout")]
    assert ar.aggregate_window(records, [0]) is None


# ---------------------------------------------------------------------------
# Extrapolación
# ---------------------------------------------------------------------------
def test_extrapolate_sqrt_law():
    result = ar.extrapolate(0.10)
    assert result["implied_k"]["2pp"] == 25
    assert result["implied_k"]["1pp"] == 100
    assert result["implied_k"]["3pp"] == 12


def test_extrapolate_without_variance_says_no_repetitions_needed():
    result = ar.extrapolate(0.0)
    assert result["implied_k"]["1pp"] == 1
    assert "no cambiarían" in result["note"]


# ---------------------------------------------------------------------------
# Etiquetas permitidas
# ---------------------------------------------------------------------------
def test_allowed_labels_binary_schmidt_excludes_meditation():
    allowed = ar.allowed_labels("stress_binary", "schmidt")
    assert allowed == {"stress", "baseline", "amusement"}
    assert "meditation" not in allowed


def test_allowed_labels_affective_keeps_every_class():
    allowed = ar.allowed_labels("affective", "schmidt")
    assert {"baseline", "stress", "amusement", "meditation"} <= allowed


# ---------------------------------------------------------------------------
# Análisis completo
# ---------------------------------------------------------------------------
def _corpus(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """3 ventanas × 4 repeticiones; una de ellas fuera de la variante."""
    rows = []
    for rep in range(4):
        rows.append(_record("win_stress", "stress", 85, rep))
        rows.append(_record("win_base", "non-stress", 85, rep))
        rows.append(_record("win_amuse", "non-stress", 85, rep))
        rows.append(_record("win_medit", "stress", 85, rep))
    _write(tmp_path / "S5.responses.jsonl", rows)
    labels = {
        "win_stress": "stress",
        "win_base": "baseline",
        "win_amuse": "amusement",
        "win_medit": "meditation",
    }
    return tmp_path, labels


def test_analyze_drops_windows_outside_the_variant(tmp_path):
    directory, labels = _corpus(tmp_path)
    report = ar.analyze([directory], draws=20, labels=labels)
    condition = report["conditions"]["C0_clean"]
    assert report["n_dropped_outside_variant"] == 4
    assert condition["n_windows"] == 3


def test_analyze_reports_deterministic_model(tmp_path):
    directory, labels = _corpus(tmp_path)
    report = ar.analyze([directory], draws=20, labels=labels)
    stability = report["conditions"]["C0_clean"]["stability"]
    assert stability["frac_deterministic"] == 1.0
    assert stability["mean_confidence_std"] == 0.0
    extrapolation = report["conditions"]["C0_clean"]["extrapolation"]["accuracy"]
    assert extrapolation["implied_k"]["2pp"] == 1


def test_analyze_convergence_covers_every_k(tmp_path):
    directory, labels = _corpus(tmp_path)
    report = ar.analyze([directory], draws=20, labels=labels)
    curve = report["conditions"]["C0_clean"]["convergence"]
    assert [row["k"] for row in curve] == [1, 2, 3, 4]
    assert curve[-1]["n_combinations"] == 1          # k = R: una sola combinación
    assert curve[0]["accuracy_sd"] == 0.0            # modelo determinista


def test_analyze_creates_confidence_resolution_with_averaging(tmp_path):
    """Promediar réplicas convierte dos valores (75/85) en una escala fina."""
    rows = []
    labels = {}
    mixes = {
        "win_a": [75, 75, 75],
        "win_b": [75, 75, 85],
        "win_c": [75, 85, 85],
        "win_d": [85, 85, 85],
    }
    for window, confidences in mixes.items():
        for rep, confidence in enumerate(confidences):
            rows.append(_record(window, "stress", confidence, rep))
        labels[window] = "stress"
    _write(tmp_path / "S5.responses.jsonl", rows)
    report = ar.analyze([tmp_path], draws=20, labels=labels)
    confidence = report["conditions"]["C0_clean"]["confidence"]
    assert confidence["single"]["n_distinct"] == 2        # 75 y 85
    assert confidence["aggregated"]["n_distinct"] == 4    # 75, 78,3, 81,7, 85
    assert confidence["aggregated"]["n_distinct"] > confidence["single"]["n_distinct"]


def test_analyze_single_run_variability_reflects_sampling(tmp_path):
    """Los sujetos no deterministas deben mostrar sd > 0 entre repeticiones."""
    rows = []
    labels = {}
    for index in range(6):
        window = f"win_{index}"
        # Cada ventana alterna: la mitad de las repeticiones cambian de decisión.
        for rep in range(4):
            decision = "stress" if (index + rep) % 2 == 0 else "non-stress"
            rows.append(_record(window, decision, 85, rep))
        labels[window] = "stress" if index % 2 == 0 else "baseline"
    _write(tmp_path / "S5.responses.jsonl", rows)
    report = ar.analyze([tmp_path], draws=50, labels=labels)
    single = report["conditions"]["C0_clean"]["single_run"]
    assert single["accuracy"]["sd"] > 0
    assert not math.isnan(single["accuracy"]["sd"])


def test_analyze_raises_without_responses(tmp_path):
    with pytest.raises(SystemExit):
        ar.analyze([tmp_path], labels={})


def test_infer_metadata_reads_provenance():
    records = [_record("w", "stress", 85, 0)]
    assert ar.infer_metadata(records, None, None) == ("stress_binary", "schmidt")


def test_infer_metadata_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        ar.infer_metadata([_record("w", "stress", 85, 0)], "rhythm4", None)


def test_analyzer_avoids_the_heavy_ingestion_import_chain():
    """No debe importar ``represent``: arrastra ``neurokit2``.

    En nodos de login con una CPU distinta al nodo de cálculo, importar
    ``sources.ingestion.represent`` aborta con ``Illegal instruction``. El
    analizador solo necesita el ID opaco, que vive en ``provenance``.
    """
    source = Path(ar.__file__).read_text(encoding="utf-8")
    assert "sources.ingestion.represent" not in source
    assert "neurokit" not in source
