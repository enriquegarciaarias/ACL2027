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

    assert cfg.model == "qwen3:8b"
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
        "model", "temperature", "num_ctx", "seed", "top_p", "max_tokens",
        "timeout", "keep_alive", "max_retries", "cascade_enabled",
        "fallback_models",
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
    assert client["model"] == "qwen3:8b"
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
