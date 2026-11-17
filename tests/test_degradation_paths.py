"""Tests de las rutas del barrido de degradación (tarea × variante × temperatura).

El nombre del directorio forma parte del diseño: mezclar corpus de 4 clases, de
binario con distintas clases ``non-stress`` o con/sin temperatura invalidaría
cualquier comparación. Barrido y analizadores deben resolver **la misma** ruta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sources.phase3.degradation import (  # noqa: E402
    degradation_candidate_dirs,
    degradation_dir_name,
    degradation_output_dir,
    is_binary_mode,
    resolve_degradation_dir,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def test_dir_name_primary_binary_without_temperature():
    assert (
        degradation_dir_name("numeric", "binary", "schmidt", "exclude")
        == "degradation_binary_schmidt_notemp"
    )


def test_dir_name_binary_with_temperature_and_variant():
    assert (
        degradation_dir_name("numeric", "binary", "baseline", "include")
        == "degradation_binary_baseline"
    )


def test_dir_name_text_binary():
    assert (
        degradation_dir_name("text", "binary", "schmidt", "exclude")
        == "degradation_text_binary_schmidt_notemp"
    )


def test_dir_name_affective_ignores_variant():
    """En 4 clases la variante no aplica; la temperatura sí nombra el corpus."""
    assert degradation_dir_name("numeric", "affective") == "degradation"
    assert degradation_dir_name("numeric", "affective", "schmidt", "include") == (
        "degradation"
    )
    assert degradation_dir_name("text", "affective", "schmidt", "exclude") == (
        "degradation_text_notemp"
    )


def test_dir_name_rejects_unknown_representation():
    with pytest.raises(ValueError):
        degradation_dir_name("audio", "binary", "schmidt", "exclude")


def test_is_binary_mode_accepts_cli_and_internal_names():
    assert is_binary_mode("binary")
    assert is_binary_mode("stress_binary")
    assert not is_binary_mode("affective")
    assert not is_binary_mode(None)


def test_candidates_prefer_primary_without_temperature():
    names = degradation_candidate_dirs(REPO_ROOT, "numeric", "binary")
    assert names == [
        "degradation_binary_schmidt_notemp",
        "degradation_binary_schmidt",
        "degradation_binary_all_notemp",
        "degradation_binary_all",
        "degradation_binary",
    ]


def test_candidates_respect_explicit_variant_and_temperature():
    names = degradation_candidate_dirs(
        REPO_ROOT, "numeric", "binary", "no_amusement", "exclude"
    )
    assert names == ["degradation_binary_no_amusement_notemp"]


def test_resolve_prefers_primary_notemp(tmp_path):
    _touch(tmp_path / "degradation_binary_schmidt_notemp" / "qwen3_14b" / "S5.responses.jsonl")
    _touch(tmp_path / "degradation_binary" / "qwen3_14b" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(
        tmp_path, model="qwen3:14b", decision_mode="binary"
    )

    assert resolved == tmp_path / "degradation_binary_schmidt_notemp" / "qwen3_14b"


def test_resolve_falls_back_to_historical_corpus(tmp_path):
    _touch(tmp_path / "degradation_binary" / "qwen3_14b" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(
        tmp_path, model="qwen3:14b", decision_mode="binary"
    )

    assert resolved == tmp_path / "degradation_binary" / "qwen3_14b"


def test_resolve_accepts_flat_legacy_directory(tmp_path):
    """Ejecuciones anteriores al aislamiento por modelo: carpeta plana."""
    _touch(tmp_path / "degradation" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(tmp_path, model="qwen3:14b")

    assert resolved == tmp_path / "degradation"


def test_resolve_without_model_picks_a_model_subdirectory(tmp_path):
    _touch(tmp_path / "degradation_binary" / "qwen3_14b" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(tmp_path, decision_mode="binary")

    assert resolved == tmp_path / "degradation_binary" / "qwen3_14b"


def test_resolve_auto_prefers_binary_then_affective(tmp_path):
    """Sin ``--decision-mode``: primero el corpus binario (endpoint primario)."""
    _touch(tmp_path / "degradation" / "qwen3_14b" / "S5.responses.jsonl")
    _touch(tmp_path / "degradation_binary" / "qwen3_14b" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(tmp_path, model="qwen3:14b")

    assert resolved == tmp_path / "degradation_binary" / "qwen3_14b"


def test_resolve_auto_falls_back_to_affective(tmp_path):
    _touch(tmp_path / "degradation" / "qwen3_14b" / "S5.responses.jsonl")

    resolved = resolve_degradation_dir(tmp_path, model="qwen3:14b")

    assert resolved == tmp_path / "degradation" / "qwen3_14b"


def test_resolve_raises_with_actionable_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_degradation_dir(tmp_path, model="qwen3:14b", decision_mode="binary")

    message = str(excinfo.value)
    assert "GENERATE" in message
    assert "degradation_binary_schmidt_notemp" in message


def test_sweep_and_analyzers_share_the_same_directory():
    """El barrido delega en el helper compartido (evita rutas divergentes)."""
    import run_degradation_sweep as sweep

    expected = degradation_output_dir(
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
