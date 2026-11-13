"""Tests del comparador de modelos (`scripts/compare_models.py`)."""

from __future__ import annotations

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


def test_paired_bootstrap_delta_zero_for_identical():
    """Dos modelos idénticos dan delta 0 con IC degenerado en 0."""
    counts = {"S5": (10, 20), "S6": (12, 20), "S7": (8, 20)}
    stats = cm.paired_bootstrap_delta(counts, counts, n_boot=200, seed=42)

    assert stats["delta"] == pytest.approx(0.0)
    assert stats["ci_low"] == pytest.approx(0.0)
    assert stats["ci_high"] == pytest.approx(0.0)
    assert stats["n_subjects"] == 3


def test_paired_bootstrap_delta_detects_improvement():
    """Un modelo que acierta más da delta positivo y fuera de 0."""
    base = {"S5": (5, 20), "S6": (6, 20), "S7": (4, 20), "S8": (5, 20)}
    better = {"S5": (15, 20), "S6": (14, 20), "S7": (16, 20), "S8": (13, 20)}
    stats = cm.paired_bootstrap_delta(base, better, n_boot=500, seed=42)

    assert stats["delta"] > 0.35
    assert stats["ci_low"] > 0


def test_paired_bootstrap_delta_without_shared_subjects():
    """Sin sujetos comunes no hay comparación posible."""
    stats = cm.paired_bootstrap_delta({"S5": (1, 2)}, {"S6": (1, 2)})
    assert stats["n_subjects"] == 0


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
    _write(tmp_path / "qwen2_5_7b" / "S5.responses.jsonl", "")
    _write(tmp_path / "qwen3_14b" / "S5.responses.jsonl", "")
    monkeypatch.setattr(cm, "_base_dir", lambda *args, **kwargs: tmp_path)

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
    report["deltas"] = {
        "C2_partial_eda": {
            "b - a": {
                "delta": 0.457,
                "ci_low": 0.457,
                "ci_high": 0.457,
                "n_subjects": 1,
            }
        }
    }
    cm.print_report(report)
    out = capsys.readouterr().out
    assert "+0.457" in out
    assert "no es inferencia" in out


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
