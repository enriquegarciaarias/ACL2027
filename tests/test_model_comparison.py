"""Tests del comparador de modelos (`scripts/compare_models.py`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_models as cm  # noqa: E402


def _write(path: Path, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_model_dir_prefers_slug(tmp_path):
    """Si existe el subdirectorio del modelo, se usa."""
    slug_dir = tmp_path / "qwen3_14b"
    _write(slug_dir / "S5.responses.jsonl", "")
    _write(tmp_path / "S5.responses.jsonl", "")

    assert cm._model_dir(tmp_path, "qwen3:14b") == slug_dir


def test_model_dir_falls_back_to_legacy(tmp_path, capsys):
    """Sin subdirectorio, se usa la carpeta histórica con aviso."""
    _write(tmp_path / "S5.responses.jsonl", "")

    resolved = cm._model_dir(tmp_path, "qwen3:14b")

    assert resolved == tmp_path
    assert "carpeta histórica" in capsys.readouterr().out


def test_model_dir_missing_raises(tmp_path):
    """Sin datos en ningún sitio, error explícito."""
    with pytest.raises(SystemExit):
        cm._model_dir(tmp_path, "qwen3:14b")


def test_models_found():
    """Extrae el modelo del campo directo o de ``ollama_meta``."""
    records = [
        {"model": "qwen3:14b"},
        {"ollama_meta": {"model": "qwen2.5:7b"}},
        {},
    ]
    assert cm._models_found(records) == {"qwen3:14b", "qwen2.5:7b"}


def test_summarize_reports_coverage_and_accuracy_including_abstention():
    """La tabla del comparador debe exponer la cobertura y la acc* (H4)."""
    metrics = {
        "n": 10,
        "n_valid": 10,
        "n_invalid": 0,
        "n_decided": 8,
        "abstention_rate": 0.2,
        "coverage": 0.8,
        "accuracy": 0.875,
        "accuracy_incl_abstain": 0.7,
        "balanced_accuracy": 0.8,
        "ece": 0.1,
        "claims": {},
        "certainty": {},
        "confidence_mean": 80.0,
    }

    summary = cm.summarize(metrics)

    assert summary["coverage"] == pytest.approx(0.8)
    assert summary["accuracy_incl_abstain"] == pytest.approx(0.7)
    assert summary["accuracy"] == pytest.approx(0.875)


def test_load_degradation_filters_variant_and_infers_task(tmp_path, monkeypatch):
    """Sin banderas: deduce la tarea del corpus y descarta ``meditation``."""
    import analyze_degradation as ad

    labels = {"w1": "stress", "w2": "baseline", "w3": "amusement", "w4": "meditation"}
    records = [
        {
            "window_id": window_id,
            "condition_id": "C0_clean",
            "subject": "S5",
            "split": "evaluation",
            "decision": "non-stress",
            "confidence": 80,
            "status": "valid_response",
            "claims": [],
            "reasoning": "",
            "prompt": "",
            "decision_mode": "stress_binary",
            "nonstress_variant": "schmidt",
            "include_temperature": False,
            "model": "qwen2.5:7b",
        }
        for window_id in labels
    ]
    directory = tmp_path / "corpus"
    directory.mkdir()
    with (directory / "S5.responses.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")

    monkeypatch.setattr(cm, "_degradation_dir", lambda *a, **k: directory)
    monkeypatch.setattr(ad, "load_labels", lambda: labels)

    kept, evaluator, info = cm.load_degradation(
        "qwen2.5:7b", "numeric", {"C0_clean"}, decision_mode=""
    )

    assert info["decision_mode"] == "binary"
    assert info["nonstress_variant"] == "schmidt"
    assert info["include_temperature"] is False
    assert info["n_dropped_variant"] == 1
    assert {record["window_id"] for record in kept} == {"w1", "w2", "w3"}
    block = evaluator(kept)
    assert block["n"] == 3


def _counts(hits: int, decided: int, total: int | None = None) -> dict[str, int]:
    """Entrada de ``per_subject_counts`` para tests (sin clases binarias)."""
    return {
        "hits": hits,
        "decided": decided,
        "total": decided if total is None else total,
        "pos": 0,
        "pos_dec": 0,
        "pos_tp": 0,
        "neg": 0,
        "neg_dec": 0,
        "neg_tn": 0,
    }


def test_paired_bootstrap_delta_zero_for_identical():
    """Dos modelos idénticos dan delta 0 con IC degenerado en 0."""
    counts = {"S5": _counts(10, 20), "S6": _counts(12, 20), "S7": _counts(8, 20)}
    stats = cm.paired_bootstrap_delta(counts, counts, n_boot=200, seed=42)

    assert stats["delta"] == pytest.approx(0.0)
    assert stats["ci_low"] == pytest.approx(0.0)
    assert stats["ci_high"] == pytest.approx(0.0)
    assert stats["n_subjects"] == 3


def test_paired_bootstrap_delta_detects_improvement():
    """Un modelo que acierta más da delta positivo y fuera de 0."""
    base = {s: _counts(h, 20) for s, h in (("S5", 5), ("S6", 6), ("S7", 4), ("S8", 5))}
    better = {s: _counts(h, 20) for s, h in (("S5", 15), ("S6", 14), ("S7", 16), ("S8", 13))}
    stats = cm.paired_bootstrap_delta(base, better, n_boot=500, seed=42)

    assert stats["delta"] > 0.35
    assert stats["ci_low"] > 0


def test_paired_bootstrap_delta_without_shared_subjects():
    """Sin sujetos comunes no hay comparación posible."""
    stats = cm.paired_bootstrap_delta({"S5": _counts(1, 2)}, {"S6": _counts(1, 2)})
    assert stats["n_subjects"] == 0


def test_per_subject_counts_separates_decided_abstain_and_classes():
    """``acc*`` cuenta las abstenciones; ``pos``/``neg`` habilitan balanced."""
    records = [
        {"subject": "S5", "_truth": "stress", "decision": "stress"},
        {"subject": "S5", "_truth": "stress", "decision": "abstain"},
        {"subject": "S5", "_truth": "non-stress", "decision": "non-stress"},
        {"subject": "S5", "_truth": "non-stress", "decision": "stress"},
    ]

    counts = cm.per_subject_counts(records)["S5"]

    assert counts["decided"] == 3
    assert counts["hits"] == 2
    assert counts["total"] == 4
    assert counts["pos"] == 2 and counts["pos_tp"] == 1
    assert counts["neg"] == 2 and counts["neg_tn"] == 1
    # La abstención cuenta como positivo no decidido (recall estricta).
    assert counts["pos_dec"] == 1
    assert counts["neg_dec"] == 2


def test_paired_deltas_expose_the_degenerate_classifier():
    """Accuracy puede subir mientras balanced baja (el caso del 14b en C4)."""
    # Modelo A (7b): acierta estrés y no-estrés; 100 ventanas, 30 de estrés.
    base = {
        "S5": {"hits": 60, "decided": 100, "total": 100,
               "pos": 30, "pos_dec": 30, "pos_tp": 15,
               "neg": 70, "neg_dec": 70, "neg_tn": 66},
    }
    # Modelo B (14b): casi nunca dice estrés y se abstiene en el estrés.
    better = {
        "S5": {"hits": 75, "decided": 85, "total": 100,
               "pos": 30, "pos_dec": 15, "pos_tp": 2,
               "neg": 70, "neg_dec": 70, "neg_tn": 69},
    }

    deltas = cm.paired_bootstrap_all_metrics(base, better, n_boot=50, seed=42)

    assert deltas["accuracy"]["delta"] > 0
    assert deltas["balanced_accuracy"]["delta"] < 0
    assert deltas["balanced_accuracy_incl_abstain"]["delta"] < 0
    # La versión estricta es más severa: la abstención pesa como fallo.
    assert (
        deltas["balanced_accuracy_incl_abstain"]["delta"]
        < deltas["balanced_accuracy"]["delta"]
    )


def test_balanced_with_abstention_as_error_differs_from_decided():
    """La abstención selectiva baja la recall estricta, no la de decididas."""
    counts = {
        "S5": {
            "hits": 6,
            "decided": 8,
            "total": 8,
            "pos": 8,
            "pos_dec": 4,
            "pos_tp": 2,
            "neg": 4,
            "neg_dec": 4,
            "neg_tn": 4,
        }
    }

    decided = cm.metric_value(counts, ["S5"], "balanced_accuracy")
    strict = cm.metric_value(counts, ["S5"], "balanced_accuracy_incl_abstain")

    assert decided == pytest.approx(0.75)
    assert strict == pytest.approx(0.625)


def test_paired_bootstrap_all_metrics_skips_unavailable_balanced():
    """En 4 clases no hay balanced binaria: se omite en lugar de dar nan."""
    base = {"S5": _counts(5, 10)}
    stats = cm.paired_bootstrap_all_metrics(base, base, n_boot=20, seed=42)
    assert "balanced_accuracy" not in stats
    assert "balanced_accuracy_incl_abstain" not in stats
    assert "accuracy" in stats and "accuracy_incl_abstain" in stats


def test_summarize_handles_missing_claims():
    """Un bloque sin claims evaluables se marca como tal, sin romper."""
    metrics = {
        "n": 10,
        "n_decided": 10,
        "abstention_rate": 0.0,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "ece": 0.2,
        "claims": {
            "n_claims": 40,
            "supported_rate": 0.0,
            "contradicted_rate": 0.0,
            "indeterminate_rate": 1.0,
            "unsupported_rate": 0.0,
        },
        "certainty": {"certainty_index_mean": -0.4},
    }
    summary = cm.summarize(metrics)
    assert summary["claims_n"] == 40
    assert summary["claims_evaluable"] == 0
    assert summary["certainty_index"] == pytest.approx(-0.4)


def test_summarize_counts_evaluable_claims():
    """Con claims verificables, ``claims_evaluable`` los cuenta."""
    metrics = {
        "n": 10,
        "claims": {
            "n_claims": 20,
            "supported_rate": 0.75,
            "contradicted_rate": 0.25,
            "indeterminate_rate": 0.0,
            "unsupported_rate": 0.0,
        },
    }
    summary = cm.summarize(metrics)
    assert summary["claims_evaluable"] == 20
    assert summary["claims_supported"] == pytest.approx(0.75)


def test_check_distinct_models_rejects_shared_directory(tmp_path, monkeypatch):
    """Job 2703: `qwen3:14b` llegó truncado como `qwen3` y ambos cayeron en
    la carpeta histórica; el comparador debe pararlo en vez de comparar un
    corpus consigo mismo."""
    _write(tmp_path / "S5.responses.jsonl", "")
    monkeypatch.setattr(cm, "_base_dir", lambda *args, **kwargs: tmp_path)

    with pytest.raises(SystemExit) as exc:
        cm.check_distinct_models(["qwen2.5:7b", "qwen3"], "degradation", "numeric")
    assert "mismo directorio" in str(exc.value)


def test_check_distinct_models_accepts_separate_directories(tmp_path, monkeypatch):
    """Con los dos subdirectorios por modelo no hay colisión."""
    dirs = {
        "qwen2.5:7b": tmp_path / "qwen2_5_7b",
        "qwen3:14b": tmp_path / "qwen3_14b",
    }
    for directory in dirs.values():
        _write(directory / "S5.responses.jsonl", "")
    monkeypatch.setattr(cm, "_degradation_dir", lambda model, *a, **k: dirs[model])

    cm.check_distinct_models(
        ["qwen2.5:7b", "qwen3:14b"], "degradation", "numeric"
    )


def test_check_distinct_models_rejects_shared_legacy(tmp_path, monkeypatch):
    """Dos modelos en la misma carpeta histórica se rechazan."""
    legacy = tmp_path / "degradation"
    _write(legacy / "S5.responses.jsonl", "")
    monkeypatch.setattr(cm, "_degradation_dir", lambda *a, **k: legacy)

    with pytest.raises(SystemExit):
        cm.check_distinct_models(
            ["qwen2.5:7b", "qwen3:14b"], "degradation", "numeric"
        )


def test_model_dir_quiet_suppresses_legacy_warning(tmp_path, capsys):
    """El preflight no repite el aviso de la carpeta histórica."""
    _write(tmp_path / "S5.responses.jsonl", "")
    cm._model_dir(tmp_path, "qwen2.5:7b", quiet=True)
    assert capsys.readouterr().out == ""


def _report_stub(abstention: float, accuracy: float, n: int = 100) -> dict:
    return {
        "mode": "degradation",
        "representation": "numeric",
        "models": ["a", "b"],
        "models_in_data": {"a": ["a"], "b": ["b"]},
        "conditions": ["C0_clean"],
        "tables": {
            "C0_clean": [
                {
                    "model": "b",
                    "n": n,
                    "n_valid": n,
                    "n_abstain": int(round(n * abstention)),
                    "abstention_rate": abstention,
                    "accuracy": accuracy,
                    "balanced_accuracy": accuracy,
                    "ece": 0.1,
                    "claims_evaluable": 10,
                    "claims_supported": 0.8,
                    "claims_contradicted": 0.2,
                    "certainty_index": 0.0,
                    "claims_n": 40,
                }
            ]
        },
        "deltas": {},
    }


def test_print_report_shows_decided_count(capsys):
    cm.print_report(_report_stub(abstention=0.963, accuracy=0.571, n=567))
    out = capsys.readouterr().out
    assert "dec" in out
    assert "21" in out          # 567 × (1 − 0.963) ≈ 21 decididas
    assert "DECIDIDAS" in out   # aviso de la trampa de la abstención


def test_print_report_warns_about_deltas_with_one_subject(capsys):
    report = _report_stub(abstention=0.998, accuracy=1.0)
    stats = {
        "delta": 0.457,
        "ci_low": 0.457,
        "ci_high": 0.457,
        "n_subjects": 1,
    }
    report["deltas"] = {
        "C2_partial_eda": {
            "b - a": {
                "accuracy": dict(stats),
                "balanced_accuracy": {**stats, "delta": -0.2},
            }
        }
    }
    cm.print_report(report)
    out = capsys.readouterr().out
    assert "+0.457" in out
    assert "accuracy (decididas)" in out
    assert "balanced" in out
    assert "remuestrea sujetos" in out


def test_accuracy_at_coverage_uses_most_confident():
    """A cobertura 0.5 se queda con la mitad más confiada, no con cualquiera."""
    pairs = [(90.0, 1.0), (80.0, 1.0), (20.0, 0.0), (10.0, 0.0)]
    accuracy, n_eval = cm.accuracy_at_coverage(pairs, coverage=0.5, n_total=4)
    assert n_eval == 2
    assert accuracy == pytest.approx(1.0)


def test_accuracy_at_coverage_empty_and_zero():
    import math

    for pairs, coverage in (([], 0.5), ([(90.0, 1.0)], 0.0)):
        accuracy, n_eval = cm.accuracy_at_coverage(pairs, coverage, 10)
        assert math.isnan(accuracy)
        assert n_eval == 0


def test_decided_pairs_excludes_abstain_but_counts_total():
    rows = [
        {"status": "valid_response", "decision": "stress", "confidence": 90, "_truth": "stress"},
        {"status": "valid_abstention", "decision": "abstain", "confidence": 30, "_truth": "stress"},
        {"status": "invalid_output", "decision": None, "confidence": 0, "_truth": "baseline"},
    ]
    pairs, n_total = cm._decided_pairs(rows)
    assert pairs == [(90.0, 1.0)]
    assert n_total == 2          # la inválida no cuenta como ventana evaluable


def test_base_dir_binary_uses_own_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(cm, "REPO_ROOT", tmp_path)
    affective = cm._base_dir("degradation", "numeric", "affective")
    binary = cm._base_dir("degradation", "numeric", "binary")
    assert affective.name == "degradation"
    assert binary.name == "degradation_binary"


def test_binary_truth_maps_four_class_labels():
    assert cm.binary_truth("stress") == "stress"
    assert cm.binary_truth("baseline") == "non-stress"
    assert cm.binary_truth("amusement") == "non-stress"
    assert cm.binary_truth("meditation") == "non-stress"
    assert cm.binary_truth(None) is None


def test_risk_coverage_with_binary_truth():
    """Con etiquetas mapeadas, un acierto 'non-stress' cuenta como acierto."""
    rows = {
        "m": [
            {"status": "valid_response", "decision": "non-stress", "confidence": 90,
             "_truth": "non-stress"},
            {"status": "valid_response", "decision": "stress", "confidence": 80,
             "_truth": "non-stress"},
        ]
    }
    block = cm.risk_coverage_by_condition(rows, {"m": 1.0})
    assert block["models"]["m"]["accuracy"] == pytest.approx(0.5)


def test_print_report_decided_excludes_invalid(capsys):
    """La columna 'dec' no debe contar las respuestas inválidas."""
    report = _report_stub(abstention=0.0, accuracy=0.9, n=100)
    report["tables"]["C0_clean"][0].update({"n_valid": 90, "n_abstain": 0, "n_invalid": 10})
    cm.print_report(report)
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("C0_clean")][0]
    assert " 90 " in line
