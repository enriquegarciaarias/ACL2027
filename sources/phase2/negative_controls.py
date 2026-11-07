"""Controles negativos para el experimento de Fase 2.

Genera variantes de ventanas con señales alteradas para verificar que el LLM
no responde al azar ni detecta patrones espurios cuando no hay información
fisiológica válida.

Tipos de control negativo:

- ``shuffled``: los valores de las features se barajan manteniendo la
  distribución original, destruyendo cualquier estructura temporal o relación
  entre canales. El LLM debería abstenerse o dar decisiones inconsistentes.
- ``zero``: todas las features numéricas se ponen a cero. Señal de ausencia.
- ``nan``: todas las features numéricas se ponen a NaN (not available).
- ``absent``: las features se dejan como "not available" (sin valor).

Estos controles se generan a partir de las representaciones LLM existentes
(numeric_llm o text_llm) y se añaden al manifiesto de ejecución como entradas
adicionales con ``condition_id`` del tipo ``NC_<tipo>``.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Mapping

# Identificadores de condición para controles negativos.
NC_SHUFFLED = "NC_shuffled"
NC_ZERO = "NC_zero"
NC_NAN = "NC_nan"
NC_ABSENT = "NC_absent"

ALL_NEGATIVE_CONTROLS: tuple[str, ...] = (
    NC_SHUFFLED,
    NC_ZERO,
    NC_NAN,
    NC_ABSENT,
)

# Expresión para detectar el valor numérico de una feature en la representación
# textual (formato "  Label: VALUE [unit]"). Captura el prefijo (hasta el dos
# puntos y espacios) y el número, dejando la unidad intacta.
_TEXT_VALUE_RE = re.compile(
    r"^(\s*[^:\n]+:\s+)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)

# Igual que _TEXT_VALUE_RE pero captura también la unidad (para el control
# "absent", que elimina valor y unidad).
_TEXT_VALUE_UNIT_RE = re.compile(
    r"^(\s*[^:\n]+:\s+)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)([^\n]*)",
    re.MULTILINE,
)


def _shuffle_text_values(text: str, seed: int = 42) -> str:
    """Baraja los valores numéricos de una representación textual.

    Mantiene etiquetas y unidades, pero permuta los números, destruyendo la
    correspondencia feature->valor sin revelar que se trata de un control.
    """
    import random

    rng = random.Random(seed)
    values = [m.group(2) for m in _TEXT_VALUE_RE.finditer(text)]
    shuffled = values[:]
    rng.shuffle(shuffled)
    it = iter(shuffled)
    return _TEXT_VALUE_RE.sub(lambda m: m.group(1) + next(it), text)


def _zero_text_values(text: str) -> str:
    """Pone todos los valores numéricos de la representación textual a cero."""
    return _TEXT_VALUE_RE.sub(lambda m: m.group(1) + "0", text)


def _nan_text_values(text: str) -> str:
    """Sustituye los valores numéricos por 'not available' (mantiene la unidad)."""
    return _TEXT_VALUE_RE.sub(lambda m: m.group(1) + "not available", text)


def _absent_text_values(text: str) -> str:
    """Elimina valor y unidad, dejando solo 'not available'."""
    return _TEXT_VALUE_UNIT_RE.sub(lambda m: m.group(1) + "not available", text)


def _shuffle_values(features: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Baraja los valores de las features dentro de cada modalidad.

    Mantiene las claves y unidades, pero asigna valores de otras features
    de la misma modalidad, destruyendo la estructura original.
    """
    import random

    rng = random.Random(42)
    result: dict[str, dict[str, Any]] = {}

    for modality, feature_block in features.items():
        items = list(feature_block.items())
        if not items:
            result[modality] = {}
            continue

        # Barajar solo los valores, manteniendo claves y unidades.
        values = [meta.get("value") for _, meta in items]
        rng.shuffle(values)

        shuffled_block: dict[str, dict[str, Any]] = {}
        for (name, meta), new_value in zip(items, values):
            new_meta = dict(meta)
            new_meta["value"] = new_value
            shuffled_block[name] = new_meta
        result[modality] = shuffled_block

    return result


def _zero_values(features: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pone todas las valores numéricos a cero."""
    result: dict[str, dict[str, Any]] = {}
    for modality, feature_block in features.items():
        zero_block: dict[str, dict[str, Any]] = {}
        for name, meta in feature_block.items():
            new_meta = dict(meta)
            new_meta["value"] = 0.0
            zero_block[name] = new_meta
        result[modality] = zero_block
    return result


def _nan_values(features: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pone todas las valores a None (not available)."""
    result: dict[str, dict[str, Any]] = {}
    for modality, feature_block in features.items():
        nan_block: dict[str, dict[str, Any]] = {}
        for name, meta in feature_block.items():
            new_meta = dict(meta)
            new_meta["value"] = None
            nan_block[name] = new_meta
        result[modality] = nan_block
    return result


def _absent_values(features: Mapping[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deja las features como 'not available' (sin valor ni unidad)."""
    result: dict[str, dict[str, Any]] = {}
    for modality, feature_block in features.items():
        absent_block: dict[str, dict[str, Any]] = {}
        for name, meta in feature_block.items():
            new_meta = dict(meta)
            new_meta["value"] = None
            new_meta["unit"] = None
            absent_block[name] = new_meta
        result[modality] = absent_block
    return result


def generate_negative_control_rep_row(
    rep_row: Mapping[str, Any],
    control_type: str,
) -> dict[str, Any]:
    """Genera una fila de representación con control negativo.

    Args:
        rep_row: fila original de la representación LLM (numeric_llm o text_llm).
        control_type: uno de NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT.

    Returns:
        dict con la misma estructura que rep_row pero con features alteradas
        según el tipo de control negativo.

    Raises:
        ValueError: si control_type no es válido.
    """
    import random

    rng = random.Random(42)
    new_row = dict(rep_row)

    if control_type == NC_SHUFFLED:
        # Para numeric_llm: barajar valores de features.
        # Para text_llm: el payload es texto; no se puede barajar fácilmente,
        # así que se marca como absent.
        payload = rep_row.get("payload", {})
        if isinstance(payload, dict) and "modalities" in payload:
            new_payload = dict(payload)
            new_payload["modalities"] = _shuffle_values(payload["modalities"])
            new_row["payload"] = new_payload
        elif isinstance(payload, str):
            # text_llm: barajar los valores numéricos manteniendo etiquetas y
            # unidades (sin revelar que es un control).
            new_row["payload"] = _shuffle_text_values(payload)
        else:
            new_row["payload"] = _shuffle_values(payload) if isinstance(payload, dict) else payload

    elif control_type == NC_ZERO:
        payload = rep_row.get("payload", {})
        if isinstance(payload, dict) and "modalities" in payload:
            new_payload = dict(payload)
            new_payload["modalities"] = _zero_values(payload["modalities"])
            new_row["payload"] = new_payload
        elif isinstance(payload, str):
            # text_llm: poner todos los valores numéricos a cero.
            new_row["payload"] = _zero_text_values(payload)
        else:
            new_row["payload"] = _zero_values(payload) if isinstance(payload, dict) else payload

    elif control_type == NC_NAN:
        payload = rep_row.get("payload", {})
        if isinstance(payload, dict) and "modalities" in payload:
            new_payload = dict(payload)
            new_payload["modalities"] = _nan_values(payload["modalities"])
            new_row["payload"] = new_payload
        elif isinstance(payload, str):
            # text_llm: sustituir los valores numéricos por 'not available'.
            new_row["payload"] = _nan_text_values(payload)
        else:
            new_row["payload"] = _nan_values(payload) if isinstance(payload, dict) else payload

    elif control_type == NC_ABSENT:
        payload = rep_row.get("payload", {})
        if isinstance(payload, dict) and "modalities" in payload:
            new_payload = dict(payload)
            new_payload["modalities"] = _absent_values(payload["modalities"])
            new_row["payload"] = new_payload
        elif isinstance(payload, str):
            # text_llm: eliminar valor y unidad, dejando solo 'not available'.
            new_row["payload"] = _absent_text_values(payload)
        else:
            new_row["payload"] = _absent_values(payload) if isinstance(payload, dict) else payload

    else:
        raise ValueError(
            f"Tipo de control negativo no válido: {control_type}. "
            f"Valores permitidos: {ALL_NEGATIVE_CONTROLS}"
        )

    # Marcar en el representation que es un control negativo.
    new_row["_negative_control"] = control_type
    new_row["_original_window_id"] = rep_row.get("window_id", "")

    return new_row


def is_negative_control(rep_row: Mapping[str, Any]) -> bool:
    """Devuelve True si la fila es un control negativo."""
    return rep_row.get("_negative_control") is not None


def negative_control_type(rep_row: Mapping[str, Any]) -> str | None:
    """Devuelve el tipo de control negativo o None si no lo es."""
    return rep_row.get("_negative_control")
