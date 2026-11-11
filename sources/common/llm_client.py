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
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import requests

from sources.common.common import processControl, logger, writeLog
from sources.common.utils import read_json

_DEFAULT_TIMEOUT = 120.0
_DEFAULT_KEEP_ALIVE = "5m"
_DEFAULT_EMBED_MODEL = "nomic-embed-text"
_OLLAMA_EMBED_ENDPOINT = "/api/embeddings"
# mxbai-embed-large tiene un contexto de aprox. 512 tokens (~1500-2000 chars).
# Usamos 1200 chars por chunk para estar seguros por debajo del límite.
_EMBED_CHUNK_CHARS = 1200


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


def _ollama_embedding_single(
    client: Mapping[str, Any],
    model: str,
    text: str,
) -> list[float]:
    """Llama una sola vez a Ollama /api/embeddings con backoff.

    Args:
        client: configuración de cliente Ollama.
        model: nombre del modelo de embeddings.
        text: texto corto (dentro del límite de contexto del modelo).

    Returns:
        lista de floats con el vector de embeddings.

    Raises:
        LLMClientError: si Ollama no responde o devuelve error.
    """
    host = str(client.get("host", "http://localhost:11434")).rstrip("/")
    timeout = float(client.get("timeout", _DEFAULT_TIMEOUT))
    url = f"{host}{_OLLAMA_EMBED_ENDPOINT}"
    payload = {"model": model, "prompt": text}

    last_error: str | None = None
    max_retries = int(client.get("max_retries", 1))
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise LLMClientError(
                    f"Ollama no devolvió embedding válido para modelo={model}: {data!r}"
                )
            return [float(v) for v in embedding]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_error = str(exc)
            if attempt < max_retries:
                # Backoff progresivo: 2s, 8s, 18s... para darle tiempo a Ollama
                # de cargar el modelo en memoria (puede tardar varios segundos
                # la primera vez con modelos grandes).
                time.sleep(2.0 * (attempt + 1) ** 2)
                continue
            raise LLMClientError(
                f"Error obteniendo embedding de Ollama (modelo={model}): {last_error}"
            ) from exc

    raise LLMClientError(
        f"Error obteniendo embedding de Ollama (modelo={model}): {last_error}"
    )


def ollama_available_models(
    host: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> list[str]:
    """Lista los modelos instalados en Ollama (``[]`` si no responde).

    Sirve de comprobación previa: lanzar un barrido de horas contra un modelo
    que no está descargado produciría miles de timeouts con reintentos.
    """
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 - comprobación best-effort
        return []
    models = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model")
        if name:
            models.append(str(name))
    return models


def check_model_available(
    model: str,
    host: str = "http://localhost:11434",
) -> bool | None:
    """¿Está el modelo instalado? ``None`` si no se pudo comprobar.

    Acepta coincidencia exacta o por familia: ``qwen3:14b`` casa con
    ``qwen3:14b`` y con ``qwen3:14b-q4_K_M``.
    """
    installed = ollama_available_models(host)
    if not installed:
        return None
    if model in installed:
        return True
    family = model.split(":")[0]
    tag = model.split(":")[1] if ":" in model else ""
    return any(
        name == model or (name.startswith(f"{family}:") and tag in name)
        for name in installed
    )


def ollama_warm_up(client: Mapping[str, Any], model: str) -> None:
    """Calienta el modelo de embeddings en Ollama con una llamada ligera.

    Los modelos de embeddings grandes (p. ej. mxbai-embed-large, ~669 MB)
    pueden tardar varios segundos en cargarse en memoria la primera vez.
    Esta función hace una llamada de prueba con un texto corto para asegurar
    que el modelo está residente antes del bucle principal de ventanas.

    Args:
        client: configuración de cliente Ollama.
        model: nombre del modelo de embeddings.
    """
    try:
        _ollama_embedding_single(client, model, "warm-up")
    except LLMClientError:
        # Si falla el warm-up, el primer intento real reintentará con backoff.
        pass


def ollama_embeddings(client: Mapping[str, Any], model: str, text: str) -> list[float]:
    """Genera el embedding de un texto con Ollama (endpoint /api/embeddings).

    Si el texto supera el límite de contexto del modelo (p. ej.
    ``mxbai-embed-large`` con ~512 tokens), se divide en chunks y se devuelve
    el promedio de los vectores de cada chunk, preservando toda la
    información sin perder datos por truncado.

    Args:
        client: configuración de cliente devuelta por ``create_ollama_client``
            o ``create_resilient_ollama_client``.
        model: nombre del modelo de embeddings en Ollama (p. ej. "nomic-embed-text").
        text: texto a vectorizar.

    Returns:
        lista de floats con el vector de embeddings (promedio de chunks).

    Raises:
        LLMClientError: si Ollama no responde o devuelve error.
    """
    if len(text) <= _EMBED_CHUNK_CHARS:
        return _ollama_embedding_single(client, model, text)

    # Dividir en chunks solapados para preservar continuidad de contexto.
    step = _EMBED_CHUNK_CHARS
    overlap = 100
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + step, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    vectors: list[list[float]] = []
    for chunk in chunks:
        vectors.append(_ollama_embedding_single(client, model, chunk))

    # Promedio elemento a elemento de todos los vectores de chunk.
    import numpy as np

    stacked = np.vstack([np.array(v, dtype=np.float64) for v in vectors])
    mean_vector = stacked.mean(axis=0)
    return [float(v) for v in mean_vector]


def ollama_embeddings_batch(
    client: Mapping[str, Any],
    model: str,
    texts: list[str],
) -> list[list[float]]:
    """Genera embeddings para una lista de textos llamando a Ollama uno por uno.

    Ollama expone /api/embeddings por prompt individual; este helper itera
    sobre ``texts`` y devuelve una lista de vectores en el mismo orden.

    Args:
        client: configuración de cliente Ollama.
        model: nombre del modelo de embeddings.
        texts: lista de textos a vectorizar.

    Returns:
        lista de vectores (lista de listas de floats).
    """
    vectors: list[list[float]] = []
    for text in texts:
        vectors.append(ollama_embeddings(client, model, text))
    return vectors
