"""Tests de la procedencia del motor LLM (versión de Ollama y digest del modelo).

El nombre del modelo no identifica el corpus: `qwen2.5:7b` con Ollama 0.40.6 y
con 0.30.7 dio 0 y 28 abstenciones respectivamente con el mismo prompt y la
misma semilla (2026-11-18). Sin registrar versión y digest, un corpus no es
reproducible ni auditable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sources.common.llm_client import ollama_run_provenance


class _Response:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_records_version_digest_and_parameters():
    with patch("sources.common.llm_client.requests.get") as get, patch(
        "sources.common.llm_client.requests.post"
    ) as post:
        get.side_effect = [
            _Response({"version": "0.40.6"}),
            _Response(
                {
                    "models": [
                        {"name": "qwen2.5:7b", "digest": "abc123", "size": 4700},
                        {"name": "qwen3:14b", "digest": "def456", "size": 9000},
                    ]
                }
            ),
        ]
        post.return_value = _Response(
            {
                "parameters": "temperature 0.7\ntop_k 40\nrepeat_penalty 1.1",
                "details": {
                    "family": "qwen2",
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            }
        )

        provenance = ollama_run_provenance("qwen2.5:7b")

    assert provenance["ollama_version"] == "0.40.6"
    assert provenance["model_digest"] == "abc123"
    assert provenance["model_size"] == 4700
    assert "repeat_penalty" in provenance["model_parameters"]
    assert provenance["model_details"]["quantization_level"] == "Q4_K_M"
    assert "errors" not in provenance


def test_picks_the_requested_model_among_several():
    with patch("sources.common.llm_client.requests.get") as get, patch(
        "sources.common.llm_client.requests.post"
    ) as post:
        get.side_effect = [
            _Response({"version": "0.40.6"}),
            _Response(
                {
                    "models": [
                        {"name": "qwen2.5:7b", "digest": "abc123"},
                        {"name": "qwen3:14b", "digest": "def456"},
                    ]
                }
            ),
        ]
        post.return_value = _Response({"parameters": []})

        provenance = ollama_run_provenance("qwen3:14b")

    assert provenance["model_digest"] == "def456"


def test_failures_are_recorded_not_raised():
    """Si Ollama no responde, el corpus puede generarse igual: solo se anota."""

    def boom(*args, **kwargs):
        raise ConnectionError("sin conexión")

    with patch("sources.common.llm_client.requests.get", side_effect=boom), patch(
        "sources.common.llm_client.requests.post", side_effect=boom
    ):
        provenance = ollama_run_provenance("qwen2.5:7b")

    assert provenance["ollama_version"] is None
    assert provenance["model_digest"] is None
    assert provenance["model_parameters"] == {}
    assert len(provenance["errors"]) == 3


def test_missing_model_leaves_digest_empty():
    with patch("sources.common.llm_client.requests.get") as get, patch(
        "sources.common.llm_client.requests.post"
    ) as post:
        get.side_effect = [
            _Response({"version": "0.40.6"}),
            _Response({"models": [{"name": "otro:1b", "digest": "zzz"}]}),
        ]
        post.return_value = _Response({})

        provenance = ollama_run_provenance("qwen2.5:7b")

    assert provenance["model_digest"] is None
    assert provenance["ollama_version"] == "0.40.6"


def test_sweep_writes_run_provenance_before_calling(tmp_path, monkeypatch):
    """El registro se escribe al arrancar: un job cortado también deja la huella."""
    import json
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    import run_degradation_sweep as sweep

    out = tmp_path / "salida"
    monkeypatch.setattr(sweep, "ensure_model_available", lambda model: None)
    monkeypatch.setattr(
        sweep,
        "ollama_run_provenance",
        lambda model: {"model": model, "ollama_version": "0.40.6", "model_digest": "abc"},
    )
    monkeypatch.setattr(sweep, "build_experiment_config", lambda **kw: MagicMock(
        model="qwen2.5:7b", model_slug="qwen2_5_7b", think=False, num_ctx=32768,
        to_dict=lambda: {"model": "qwen2.5:7b", "temperature": 0.3},
    ))
    monkeypatch.setattr(sweep, "create_ollama_chat_client", lambda config: {})
    monkeypatch.setattr(sweep, "run_subject", lambda **kw: {"n_calls": 0, "n_cached": 0, "n_errors": 0})

    sweep.main(
        [
            "--subjects", "S5",
            "--conditions", "C5_total_both",
            "--decision-mode", "binary",
            "--repetitions", "3",
            "--output-dir", str(out),
        ]
    )

    provenance = json.loads((out / "run_provenance.json").read_text(encoding="utf-8"))
    assert provenance["ollama"]["ollama_version"] == "0.40.6"
    assert provenance["ollama"]["model_digest"] == "abc"
    assert provenance["repetitions"] == 3
    assert provenance["seeds"] == [42, 43, 44]
