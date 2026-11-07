"""Configuración experimental de llamadas a LLM (Fase 2, Paso 3).

Define dos modos de ejecución:

- **Modo experimento**: un solo modelo (qwen3:8b), misma semilla fija,
  N reintentos fijos con la misma semilla, SIN cascada de modelos.
  Esto garantiza que cada condición experimental sea reproducible y que no
  haya contaminación entre modelos dentro de una misma condición.

- **Modo desarrollo**: cascada de modelos permitida (primary → fallbacks),
  solo para debugging y pruebas rápidas.

Todos los parámetros se leen de `config.json > defaults.llm` y se registran
para procedencia completa (modelo, temperatura, num_ctx, seed, top_p,
keep_alive, versión de Ollama).

No se implementa aquí la llamada HTTP real a Ollama; se define la
configuración del cliente y se delega la ejecución al orquestador
(Fase 2, Paso 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sources.common.common import processControl, logger, writeLog

# Valores por defecto si no están en config.json.
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_LLM_TIMEOUT = 120.0
_DEFAULT_LLM_KEEP_ALIVE = "5m"
_DEFAULT_NUM_CTX = 32768
_DEFAULT_SEED = 42
_DEFAULT_TOP_P = 0.9
_DEFAULT_MAX_RETRIES = 3

# Modelo por defecto para experimento (según decisión del Paso 0).
_DEFAULT_PRIMARY_MODEL = "qwen3:8b"


@dataclass
class LlmExperimentConfig:
    """Configuración de una ejecución experimental de LLM.

    Atributos registrados para procedencia completa de cada llamada.
    """

    model: str
    temperature: float
    num_ctx: int
    seed: int
    top_p: float
    max_tokens: int
    timeout: float
    keep_alive: str
    max_retries: int
    cascade_enabled: bool
    fallback_models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "keep_alive": self.keep_alive,
            "max_retries": self.max_retries,
            "cascade_enabled": self.cascade_enabled,
            "fallback_models": self.fallback_models,
        }


def _llm_cfg() -> dict[str, Any]:
    """Devuelve la sección `defaults.llm` de config.json."""
    defaults = processControl.defaults or {}
    return defaults.get("llm", {}) or {}


def build_experiment_config(
    *,
    model: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    keep_alive: str | None = None,
    max_retries: int | None = None,
    cascade_enabled: bool = False,
) -> LlmExperimentConfig:
    """Construye la configuración experimental de LLM.

    Args:
        model: nombre del modelo (default: primary_model de config.json,
            o "qwen3:8b" si no está definido).
        temperature: temperatura de muestreo (default: 0.3).
        num_ctx: tamaño de contexto en tokens (default: 32768).
        seed: semilla aleatoria fija (default: 42).
        top_p: probabilidad acumulada para muestreo (default: 0.9).
        max_tokens: tokens máximos de respuesta (default: 800).
        timeout: timeout por llamada en segundos (default: 120.0).
        keep_alive: tiempo de permanencia del modelo en Ollama (default: "5m").
        max_retries: reintentos fijos con la misma semilla (default: 3).
        cascade_enabled: si True, permite fallback en cascada (solo desarrollo).
            En experimento debe ser False.

    Returns:
        LlmExperimentConfig con todos los parámetros resueltos.
    """
    cfg = _llm_cfg()

    resolved_model = (
        model
        if model is not None
        else str(cfg.get("primary_model", _DEFAULT_PRIMARY_MODEL))
    )
    resolved_temperature = (
        float(temperature)
        if temperature is not None
        else float(cfg.get("temperature", _DEFAULT_TEMPERATURE))
    )
    resolved_num_ctx = (
        int(num_ctx) if num_ctx is not None else int(_DEFAULT_NUM_CTX)
    )
    resolved_seed = (
        int(seed) if seed is not None else int(_DEFAULT_SEED)
    )
    resolved_top_p = (
        float(top_p) if top_p is not None else float(_DEFAULT_TOP_P)
    )
    resolved_max_tokens = (
        int(max_tokens)
        if max_tokens is not None
        else int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
    )
    resolved_timeout = (
        float(timeout) if timeout is not None
        else float(cfg.get("llm_timeout", _DEFAULT_LLM_TIMEOUT))
    )
    resolved_keep_alive = (
        str(keep_alive)
        if keep_alive is not None
        else str(cfg.get("llm_keep_alive", _DEFAULT_LLM_KEEP_ALIVE))
    )
    resolved_max_retries = (
        int(max_retries) if max_retries is not None else int(_DEFAULT_MAX_RETRIES)
    )

    fallback_models: list[str] = []
    if cascade_enabled:
        fallbacks = cfg.get("fallback_model", [])
        if isinstance(fallbacks, list):
            fallback_models = [str(f) for f in fallbacks]

    config = LlmExperimentConfig(
        model=resolved_model,
        temperature=resolved_temperature,
        num_ctx=resolved_num_ctx,
        seed=resolved_seed,
        top_p=resolved_top_p,
        max_tokens=resolved_max_tokens,
        timeout=resolved_timeout,
        keep_alive=resolved_keep_alive,
        max_retries=resolved_max_retries,
        cascade_enabled=cascade_enabled,
        fallback_models=fallback_models,
    )

    writeLog(
        "info",
        logger,
        f"[llm_config] configuración experimental: model={config.model}, "
        f"seed={config.seed}, retries={config.max_retries}, "
        f"cascade={'enabled' if config.cascade_enabled else 'disabled'}, "
        f"num_ctx={config.num_ctx}, temperature={config.temperature}",
    )
    return config


def create_ollama_chat_client(config: LlmExperimentConfig) -> dict[str, Any]:
    """Crea un diccionario de cliente Ollama para llamadas de chat (/api/chat).

    Args:
        config: configuración experimental resuelta.

    Returns:
        dict con host, modelo, timeout, keep_alive, num_ctx, seed, top_p,
        max_tokens, max_retries, cascade_enabled, fallback_models.
    """
    client = {
        "host": "http://localhost:11434",
        "model": config.model,
        "timeout": config.timeout,
        "keep_alive": config.keep_alive,
        "num_ctx": config.num_ctx,
        "seed": config.seed,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "max_retries": config.max_retries,
        "cascade_enabled": config.cascade_enabled,
        "fallback_models": config.fallback_models,
    }
    writeLog(
        "debug",
        logger,
        f"[llm_config] cliente Ollama chat creado: model={client['model']}, "
        f"seed={client['seed']}, num_ctx={client['num_ctx']}",
    )
    return client
