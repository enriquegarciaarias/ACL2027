"""Tests del soporte de repeticiones del barrido de degradación.

Cubren lo que puede invalidar el experimento en silencio: que las semillas no
sean distintas, que la caché mezcle repeticiones, que el piloto corto recoja una
sola clase del protocolo o que la estimación de llamadas del ``--dry-run`` no
cuente las repeticiones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_degradation_sweep as sweep  # noqa: E402
from sources.pipelines import _estimate_calls, build_generate_argv  # noqa: E402


# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------
def test_repetition_seeds_start_at_base():
    assert sweep.repetition_seeds(42, 1) == [42]


def test_repetition_seeds_are_consecutive_and_distinct():
    seeds = sweep.repetition_seeds(42, 10)
    assert seeds == [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    assert len(set(seeds)) == len(seeds)


def test_repetition_one_matches_single_pass_corpus():
    """La repetición 0 debe ser la del corpus de una pasada (--seed)."""
    assert sweep.repetition_seeds(42, 3)[0] == 42


@pytest.mark.parametrize("bad", [0, -1])
def test_repetition_seeds_reject_non_positive(bad):
    with pytest.raises(ValueError):
        sweep.repetition_seeds(42, bad)


# ---------------------------------------------------------------------------
# Caché / reanudación
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_existing_pairs_distinguishes_repetitions(tmp_path):
    path = _write_jsonl(
        tmp_path / "S5.responses.jsonl",
        [
            {"window_id": "W1", "condition_id": "C0_clean", "repetition": 0},
            {"window_id": "W1", "condition_id": "C0_clean", "repetition": 1},
            {"window_id": "W1", "condition_id": "C5_total_both", "repetition": 0},
        ],
    )
    existing, n_retry = sweep.existing_pairs(path)
    assert existing == {
        ("W1", "C0_clean", 0),
        ("W1", "C0_clean", 1),
        ("W1", "C5_total_both", 0),
    }
    assert n_retry == 0


def test_existing_pairs_reads_legacy_corpus_as_repetition_zero(tmp_path):
    """Los corpus anteriores a las repeticiones no tienen el campo."""
    path = _write_jsonl(
        tmp_path / "S5.responses.jsonl",
        [{"window_id": "W1", "condition_id": "C0_clean"}],
    )
    existing, _ = sweep.existing_pairs(path)
    assert existing == {("W1", "C0_clean", 0)}


def test_existing_pairs_still_retries_timeouts(tmp_path):
    path = _write_jsonl(
        tmp_path / "S5.responses.jsonl",
        [
            {"window_id": "W1", "condition_id": "C0_clean", "status": "timeout"},
            {"window_id": "W1", "condition_id": "C0_clean", "repetition": 1},
        ],
    )
    existing, n_retry = sweep.existing_pairs(path)
    assert n_retry == 1
    assert ("W1", "C0_clean", 0) not in existing
    assert ("W1", "C0_clean", 1) in existing


def test_existing_pairs_tolerates_half_written_line(tmp_path):
    path = tmp_path / "S5.responses.jsonl"
    path.write_text('{"window_id": "W1", "condition_id": "C0_clean"}\n{"window_',
                    encoding="utf-8")
    existing, _ = sweep.existing_pairs(path)
    assert existing == {("W1", "C0_clean", 0)}


# ---------------------------------------------------------------------------
# Muestra de ventanas
# ---------------------------------------------------------------------------
def test_sample_windows_spans_the_whole_record():
    windows = [f"W{i:03d}" for i in range(20)]
    sample = sweep.sample_window_ids(windows, 5)
    assert len(sample) == 5
    assert sample[0] == "W000"
    assert sweep.sample_window_ids(windows, 5) != windows[:5]
    assert any(w >= "W010" for w in sample)


def test_sample_windows_is_deterministic_and_sorted():
    windows = [f"W{i:03d}" for i in range(30)]
    first = sweep.sample_window_ids(windows, 7)
    assert first == sweep.sample_window_ids(windows, 7)
    assert first == sorted(first)


def test_sample_windows_without_limit_returns_everything():
    windows = ["W2", "W1", "W3"]
    assert sweep.sample_window_ids(windows, None) == ["W1", "W2", "W3"]
    assert sweep.sample_window_ids(windows, 10) == ["W1", "W2", "W3"]


def test_sample_windows_does_not_use_labels():
    """La selección es ciega: mismo resultado con etiquetas invertidas."""
    windows = [f"W{i:03d}" for i in range(12)]
    assert sweep.sample_window_ids(windows, 4) == sweep.sample_window_ids(
        list(reversed(windows)), 4
    )


# ---------------------------------------------------------------------------
# Salida por modelo y por repeticiones
# ---------------------------------------------------------------------------
def test_output_directory_matches_shared_resolver():
    expected = sweep.degradation_output_dir(
        sweep.OUTPUT_DIR.parent,
        "qwen2.5:7b",
        representation="numeric",
        decision_mode="stress_binary",
        nonstress_variant="schmidt",
        temperature_mode="exclude",
    )
    assert sweep.output_directory(
        "numeric", "stress_binary", "qwen2.5:7b", "schmidt", "exclude"
    ) == expected


# ---------------------------------------------------------------------------
# CLI y estimación
# ---------------------------------------------------------------------------
def _args(**overrides):
    base = dict(
        subjects="S5,S6",
        representation=None,
        conditions="",
        decision_mode=None,
        nonstress_variant=None,
        temperature=None,
        think=False,
        allowDev=False,
        repetitions=None,
        sample_windows=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_argv_forwards_repetitions_and_sample():
    argv = build_generate_argv("degradation", _args(repetitions=10, sample_windows=12))
    assert argv[argv.index("--repetitions") + 1] == "10"
    assert argv[argv.index("--sample-windows") + 1] == "12"


def test_generate_argv_omits_repetitions_by_default():
    argv = build_generate_argv("degradation", _args())
    assert "--repetitions" not in argv
    assert "--sample-windows" not in argv


def test_estimate_calls_multiplies_by_repetitions(monkeypatch):
    monkeypatch.setattr("sources.pipelines._n_windows", lambda subject: 10)
    base = _estimate_calls("degradation", ["S5", "S6"], _args())
    with_reps = _estimate_calls(
        "degradation", ["S5", "S6"], _args(repetitions=5)
    )
    assert base == 20 * 5
    assert with_reps == 20 * 5 * 5


def test_estimate_calls_respects_sample_windows(monkeypatch):
    monkeypatch.setattr("sources.pipelines._n_windows", lambda subject: 40)
    calls = _estimate_calls(
        "degradation", ["S5", "S6"], _args(sample_windows=12, repetitions=10)
    )
    assert calls == 2 * 12 * 5 * 10


# ---------------------------------------------------------------------------
# Identidad del motor
# ---------------------------------------------------------------------------
def test_engine_conflicts_detects_version_and_digest():
    stored = {"model": "qwen2.5:7b", "model_digest": "aaa", "ollama_version": "0.40.6"}
    current = {"model": "qwen2.5:7b", "model_digest": "bbb", "ollama_version": "0.42.0"}
    conflicts = sweep.engine_conflicts(stored, current)
    assert len(conflicts) == 2
    assert any("model_digest" in c for c in conflicts)
    assert any("ollama_version" in c for c in conflicts)


def test_engine_conflicts_ignores_missing_fields():
    """Sin dato en uno de los dos lados no se puede afirmar que haya cambiado."""
    assert sweep.engine_conflicts({"ollama_version": "0.40.6"}, {"model": "x"}) == []
    assert sweep.engine_conflicts(None, {"ollama_version": "0.40.6"}) == []
    assert sweep.engine_conflicts({}, {}) == []


def test_engine_conflicts_same_engine_is_clean():
    provenance = {"model": "qwen2.5:7b", "model_digest": "aaa", "ollama_version": "0.40.6"}
    assert sweep.engine_conflicts(provenance, dict(provenance)) == []


def test_sweep_aborts_on_engine_change(tmp_path, monkeypatch):
    """Reanudar tras cambiar de motor mezclaría corpora en silencio."""
    (tmp_path / "run_provenance.json").write_text(
        json.dumps({"ollama": {"model_digest": "viejo", "ollama_version": "0.40.6"}}),
        encoding="utf-8",
    )
    (tmp_path / "S5.responses.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(sweep, "ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        sweep,
        "ollama_run_provenance",
        lambda model: {"model": model, "model_digest": "nuevo", "ollama_version": "0.40.6"},
    )
    with pytest.raises(SystemExit) as excinfo:
        sweep.main([
            "--subjects", "S5", "--conditions", "C5_total_both",
            "--output-dir", str(tmp_path), "--model", "qwen2.5:7b",
            "--decision-mode", "binary",
        ])
    assert "otro motor" in str(excinfo.value)
