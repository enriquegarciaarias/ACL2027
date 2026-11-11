"""Pruebas de configuración experimental de LLM (Fase 2, Paso 3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sources.common.common import processControl
from sources.phase2.llm_config import (
    LlmExperimentConfig,
    build_experiment_config,
    create_ollama_chat_client,
)


def _set_llm_defaults(monkeypatch, llm_section: dict):
    """Establece processControl.defaults con una sección llm dada."""
    defaults = {"llm": llm_section}
    monkeypatch.setattr(processControl, "defaults", defaults)


def test_build_experiment_config_defaults(monkeypatch):
    """Con valores por defecto de config.json (vacíos), se usan los hardcoded."""
    _set_llm_defaults(monkeypatch, {})
    cfg = build_experiment_config()

    assert cfg.model == "qwen2.5:7b"
    assert cfg.think is False
    assert cfg.temperature == 0.3
    assert cfg.num_ctx == 32768
    assert cfg.seed == 42
    assert cfg.top_p == 0.9
    assert cfg.max_tokens == 800
    assert cfg.timeout == 120.0
    assert cfg.keep_alive == "5m"
    assert cfg.max_retries == 3
    assert cfg.cascade_enabled is False
    assert cfg.fallback_models == []


def test_build_experiment_config_from_config_json(monkeypatch):
    """Se leen los valores de config.json > defaults.llm."""
    _set_llm_defaults(
        monkeypatch,
        {
            "primary_model": "qwen2.5:7b",
            "temperature": 0.7,
            "max_tokens": 400,
            "llm_timeout": 60.0,
            "llm_keep_alive": "10m",
            "fallback_model": ["llama3.1:8b"],
        },
    )
    cfg = build_experiment_config()

    assert cfg.model == "qwen2.5:7b"
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 400
    assert cfg.timeout == 60.0
    assert cfg.keep_alive == "10m"


def test_build_experiment_config_explicit_overrides(monkeypatch):
    """Los parámetros explícitos sobrescriben config.json."""
    _set_llm_defaults(
        monkeypatch,
        {
            "primary_model": "qwen3:8b",
            "temperature": 0.3,
            "fallback_model": ["qwen2.5:7b"],
        },
    )
    cfg = build_experiment_config(
        model="llama3.1:8b",
        temperature=0.5,
        seed=123,
        max_retries=5,
        cascade_enabled=True,
    )

    assert cfg.model == "llama3.1:8b"
    assert cfg.temperature == 0.5
    assert cfg.seed == 123
    assert cfg.max_retries == 5
    assert cfg.cascade_enabled is True
    assert cfg.fallback_models == ["qwen2.5:7b"]


def test_build_experiment_config_cascade_disabled_no_fallbacks(monkeypatch):
    """Con cascade_enabled=False, fallback_models queda vacío aunque haya en config."""
    _set_llm_defaults(
        monkeypatch,
        {
            "primary_model": "qwen3:8b",
            "fallback_model": ["qwen2.5:7b", "llama3.1:8b"],
        },
    )
    cfg = build_experiment_config(cascade_enabled=False)

    assert cfg.cascade_enabled is False
    assert cfg.fallback_models == []


def test_build_experiment_config_cascade_enabled_with_fallbacks(monkeypatch):
    """Con cascade_enabled=True, se incluyen los fallbacks de config."""
    _set_llm_defaults(
        monkeypatch,
        {
            "primary_model": "qwen3:8b",
            "fallback_model": ["qwen2.5:7b"],
        },
    )
    cfg = build_experiment_config(cascade_enabled=True)

    assert cfg.cascade_enabled is True
    assert cfg.fallback_models == ["qwen2.5:7b"]


def test_config_to_dict_contains_all_fields(monkeypatch):
    """to_dict serializa todos los campos para procedencia."""
    _set_llm_defaults(monkeypatch, {})
    cfg = build_experiment_config(model="qwen3:8b", seed=7, max_retries=3)
    d = cfg.to_dict()

    expected_keys = {
        "model", "model_slug", "temperature", "num_ctx", "seed", "top_p",
        "max_tokens", "timeout", "keep_alive", "max_retries",
        "cascade_enabled", "think", "fallback_models",
    }
    assert set(d.keys()) == expected_keys
    assert d["model"] == "qwen3:8b"
    assert d["seed"] == 7
    assert d["max_retries"] == 3
    assert d["cascade_enabled"] is False


def test_create_ollama_chat_client_structure(monkeypatch):
    """El cliente chat tiene todos los campos necesarios para llamadas a /api/chat."""
    _set_llm_defaults(monkeypatch, {})
    cfg = build_experiment_config(seed=99, num_ctx=16384)
    client = create_ollama_chat_client(cfg)

    assert client["host"] == "http://localhost:11434"
    assert client["model"] == "qwen2.5:7b"
    assert client["think"] is False
    assert client["seed"] == 99
    assert client["num_ctx"] == 16384
    assert client["cascade_enabled"] is False
    assert client["fallback_models"] == []
    assert "timeout" in client
    assert "keep_alive" in client
    assert "top_p" in client
    assert "max_tokens" in client
    assert "max_retries" in client


def test_config_is_dataclass(monkeypatch):
    """LlmExperimentConfig es un dataclass con eq=True por defecto."""
    _set_llm_defaults(monkeypatch, {})
    cfg1 = build_experiment_config(seed=1)
    cfg2 = build_experiment_config(seed=1)
    cfg3 = build_experiment_config(seed=2)

    assert cfg1 == cfg2
    assert cfg1 != cfg3


def test_think_resolved_from_config(monkeypatch):
    """``defaults.llm.think`` se respeta y se propaga al cliente."""
    _set_llm_defaults(monkeypatch, {"think": True})
    cfg = build_experiment_config()
    assert cfg.think is True
    assert create_ollama_chat_client(cfg)["think"] is True


def test_think_explicit_override(monkeypatch):
    """Un ``think`` explícito gana sobre config.json."""
    _set_llm_defaults(monkeypatch, {"think": True})
    cfg = build_experiment_config(think=False)
    assert cfg.think is False


def test_model_slug():
    """El slug es estable y apto para rutas."""
    from sources.phase2.llm_config import model_slug

    assert model_slug("qwen3:14b") == "qwen3_14b"
    assert model_slug("Qwen2.5:7B") == "qwen2_5_7b"
    assert model_slug("llama3.1:8b-instruct") == "llama3_1_8b_instruct"
    assert model_slug("") == "model"


def test_supports_thinking():
    """Solo la familia qwen3 recibe el parámetro ``think``."""
    from sources.phase2.llm_config import supports_thinking

    assert supports_thinking("qwen3:14b") is True
    assert supports_thinking("qwen3:8b") is True
    assert supports_thinking("qwen2.5:7b") is False
    assert supports_thinking("llama3.1:8b") is False


def test_ollama_available_models_and_check(monkeypatch):
    """La comprobación previa detecta modelos instalados y ausentes."""
    from sources.common import llm_client

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": "qwen2.5:7b"},
                    {"name": "qwen3:14b-q4_K_M"},
                ]
            }

    monkeypatch.setattr(llm_client.requests, "get", lambda *a, **k: _Response())
    assert llm_client.ollama_available_models() == [
        "qwen2.5:7b", "qwen3:14b-q4_K_M",
    ]
    assert llm_client.check_model_available("qwen2.5:7b") is True
    assert llm_client.check_model_available("qwen3:14b") is True  # por familia/tag
    assert llm_client.check_model_available("llama3.1:8b") is False


def test_ollama_available_models_returns_empty_on_error(monkeypatch):
    """Si Ollama no responde, la comprobación no rompe la ejecución."""
    from sources.common import llm_client

    def _boom(*args, **kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(llm_client.requests, "get", _boom)
    assert llm_client.ollama_available_models() == []
    assert llm_client.check_model_available("qwen3:14b") is None
