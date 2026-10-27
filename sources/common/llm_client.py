"""
Cliente LLM con reintentos, timeout y fallback en cascada.

Mejoras propuestas incorporadas al esqueleto común (heredadas de la experiencia
de SLIP):
  - Timeout por llamada para evitar bloqueos indefinidos si el equipo se
    suspende o el servicio LLM deja de responder.
  - Fallback en cascada entre modelos (primary → fallbacks).
  - Configuración vía `config.json > defaults.llm` (llm_timeout, llm_keep_alive).
  - factories `create_ollama_client()` y `create_resilient_ollama_client()`.

Este módulo es opcional: los proyectos que no usen Ollama/LLM pueden simplemente
no importarlo.
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.utils import read_json

import json
import time
from pathlib import Path

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_KEEP_ALIVE = "5m"


class LLMClientError(Exception):
    """Error genérico del cliente LLM."""
    pass


def _llm_config():
    """Devuelve la configuración LLM de `processControl.defaults`."""
    defaults = processControl.defaults or {}
    return defaults.get("llm", {})


def create_ollama_client(timeout=None, keep_alive=None):
    """Crea un cliente Ollama simple.

    Esta es una factory ligera. La implementación real de llamadas a Ollama
    debe añadirse en el proyecto concreto (p. ej. usando la librería `ollama`
    o peticiones HTTP a `localhost:11434`). Se devuelve un diccionario de
    configuración listo para usar.

    Args:
        timeout (float, optional): Timeout en segundos por llamada.
        keep_alive (str, optional): Tiempo de permanencia del modelo en Ollama.

    Returns:
        dict: Configuración del cliente ({host, timeout, keep_alive}).
    """
    cfg = _llm_config()
    timeout = timeout if timeout is not None else cfg.get("llm_timeout", _DEFAULT_TIMEOUT)
    keep_alive = keep_alive if keep_alive is not None else cfg.get("llm_keep_alive", _DEFAULT_KEEP_ALIVE)

    client = {
        "host": "http://localhost:11434",
        "timeout": float(timeout),
        "keep_alive": keep_alive,
    }
    writeLog("debug", logger, f"Ollama client creado: timeout={timeout}s, keep_alive={keep_alive}")
    return client


def create_resilient_ollama_client(timeout=None, keep_alive=None, max_retries=None):
    """Crea un cliente Ollama resiliente con reintentos y fallback de modelos.

    Lee `primary_model` y `fallback_model` de `config.json > defaults.llm`.

    Args:
        timeout (float, optional): Timeout en segundos por llamada.
        keep_alive (str, optional): Tiempo de permanencia del modelo.
        max_retries (int, optional): Número máximo de reintentos por modelo.

    Returns:
        dict: Configuración del cliente resiliente.
    """
    cfg = _llm_config()
    timeout = timeout if timeout is not None else cfg.get("llm_timeout", _DEFAULT_TIMEOUT)
    keep_alive = keep_alive if keep_alive is not None else cfg.get("llm_keep_alive", _DEFAULT_KEEP_ALIVE)
    max_retries = max_retries if max_retries is not None else cfg.get("max_retries", 1)

    client = {
        "host": "http://localhost:11434",
        "timeout": float(timeout),
        "keep_alive": keep_alive,
        "max_retries": int(max_retries),
        "primary_model": cfg.get("primary_model"),
        "fallback_models": cfg.get("fallback_model", []),
    }
    writeLog("debug", logger,
             f"Resilient Ollama client creado: primary={client['primary_model']}, "
             f"fallbacks={client['fallback_models']}, timeout={timeout}s, retries={max_retries}")
    return client
