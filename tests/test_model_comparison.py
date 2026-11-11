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
