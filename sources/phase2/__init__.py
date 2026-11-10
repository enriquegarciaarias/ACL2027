"""Fase 2: constructores de prompts, llamadas a LLM y compuerta de consistencia.

Componentes:
- ``prompt_builder``: prompts en inglés a partir de las vistas ``numeric_llm`` y
  ``text_llm``, con verificación anti-leakage.
- ``response_schema``: esquema y validación de la salida JSON del LLM.
- ``llm_config``: configuración experimental sin cascada (modelo, semilla,
  reintentos) con procedencia completa.
- ``orchestrator``: ejecución factorial (ventana × representación × variante ×
  repetición × condición) con caché por hash de prompt.
- ``negative_controls``: controles negativos (shuffled, zero, nan, absent).
- ``consistency``: comprobación determinista de coherencia fisiológica interna
  y compuerta de abstención previa al LLM (P1).
"""

from __future__ import annotations

from .consistency import (
    ConsistencyReport,
    check_consistency,
    should_abstain,
)

__all__ = [
    "ConsistencyReport",
    "check_consistency",
    "should_abstain",
]
