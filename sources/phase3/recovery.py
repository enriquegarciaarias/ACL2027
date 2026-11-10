"""Recuperación tolerante de respuestas con claims mal formados.

El validador estricto (`validate_llm_output`) marca como ``invalid_output``
toda respuesta que contenga un claim con vocabulario no canónico (p. ej.
``channel: "not available"``, ``magnitude: "none"`` o
``direction: "slight increase"``). En esas respuestas la **decisión y la
confianza suelen ser válidas**, y descartarlas sesga la comparación entre
representaciones (el texto produce más variantes de vocabulario que el JSON
numérico).

Este módulo recupera, **solo para el análisis**, la decisión, la confianza y los
claims bien formados. Los datos originales no se modifican: la respuesta
recuperada se marca con ``recovered = True`` y ``dropped_claims``.
"""

from __future__ import annotations

from typing import Any, Mapping

from sources.phase3.claims import CANONICAL_METRICS, normalize_metric
from sources.phase2.response_schema import (
    ALLOWED_CLAIM_CHANNELS,
    ALLOWED_DECISIONS,
    ALLOWED_DIRECTIONS,
    ALLOWED_MAGNITUDES,
    CHANNEL_ALIASES,
    DECISION_ALIASES,
    DIRECTION_ALIASES,
    MAGNITUDE_ALIASES,
    _coerce_int,
    _coerce_str,
    _extract_json_from_text,
)

import json

_EXTRA_DIRECTION_ALIASES = {
    "slight increase": "increase",
    "slightly increased": "increase",
    "slight decrease": "decrease",
    "slightly decreased": "decrease",
    "slightly elevated": "increase",
    "slightly reduced": "decrease",
    "unchanged ": "stable",
}
_EXTRA_MAGNITUDE_ALIASES = {"none": "low", "no change": "low", "negligible": "low"}
_INVALID_CHANNEL_VALUES = {"not available", "missing", "none", "n/a", "na", ""}


def _normalize_channel(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in _INVALID_CHANNEL_VALUES:
        return None
    if key in ALLOWED_CLAIM_CHANNELS:
        return key
    if key in CHANNEL_ALIASES:
        mapped = CHANNEL_ALIASES[key]
        return mapped if mapped in ALLOWED_CLAIM_CHANNELS else None
    # Prefijos de ubicación: wrist_eda -> eda
    if "_" in key:
        candidate = key.split("_")[-1]
        if candidate in ALLOWED_CLAIM_CHANNELS:
            return candidate
    return None


def _normalize_direction(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in ALLOWED_DIRECTIONS:
        return key
    if key in DIRECTION_ALIASES:
        return DIRECTION_ALIASES[key]
    if key in _EXTRA_DIRECTION_ALIASES:
        return _EXTRA_DIRECTION_ALIASES[key]
    return None


def _normalize_magnitude(raw: str) -> str | None:
    key = raw.strip().lower()
    if key in ALLOWED_MAGNITUDES:
        return key
    if key in MAGNITUDE_ALIASES:
        return MAGNITUDE_ALIASES[key]
    if key in _EXTRA_MAGNITUDE_ALIASES:
        return _EXTRA_MAGNITUDE_ALIASES[key]
    return None


def _parse_json(raw_output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    extracted = _extract_json_from_text(raw_output)
    if extracted is None:
        return None
    try:
        parsed = json.loads(extracted)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def recover_llm_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Recupera decisión, confianza y claims válidos de una respuesta inválida.

    Returns:
        dict con ``decision``, ``confidence``, ``claims``, ``reasoning``,
        ``dropped_claims`` y ``recovered=True``; ``None`` si la decisión o la
        confianza no son recuperables.
    """
    if record.get("status") not in ("invalid_output", "timeout"):
        return None
    data = _parse_json(record.get("raw_output") or "")
    if data is None:
        return None

    decision_raw = _coerce_str(data.get("decision"))
    if decision_raw is None:
        return None
    decision_key = DECISION_ALIASES.get(decision_raw.strip().lower(), decision_raw.strip().lower())
    decision_map = {d.lower(): d for d in ALLOWED_DECISIONS}
    decision = decision_map.get(decision_key)
    confidence = _coerce_int(data.get("confidence"))
    if decision is None or confidence is None or not 0 <= confidence <= 100:
        return None

    claims: list[dict[str, Any]] = []
    dropped = 0
    claims_raw = data.get("claims")
    if isinstance(claims_raw, list):
        for claim in claims_raw:
            if not isinstance(claim, Mapping):
                dropped += 1
                continue
            metric = normalize_metric(str(claim.get("metric", "")))
            channel = _normalize_channel(str(claim.get("channel", "")))
            direction = _normalize_direction(str(claim.get("direction", "")))
            magnitude = _normalize_magnitude(str(claim.get("magnitude", "")))
            if metric is None or channel is None or direction is None or magnitude is None:
                dropped += 1
                continue
            claims.append(
                {
                    "metric": metric,
                    "channel": channel,
                    "direction": direction,
                    "magnitude": magnitude,
                }
            )
    elif claims_raw is not None:
        dropped += 1

    return {
        "decision": decision,
        "confidence": float(confidence),
        "claims": claims,
        "reasoning": str(data.get("reasoning", "") or ""),
        "dropped_claims": dropped,
        "recovered": True,
    }


def apply_recovery(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Aplica la recuperación a una lista de respuestas (in situ, copiadas).

    Returns:
        ``(records_recuperados, n_recuperados)``. Las respuestas no recuperables
        se dejan intactas (siguen siendo inválidas).
    """
    output: list[dict[str, Any]] = []
    recovered_count = 0
    for record in records:
        recovered = recover_llm_record(record)
        if recovered is None:
            output.append(record)
            continue
        merged = dict(record)
        merged.update(recovered)
        merged["status"] = (
            "valid_abstention" if recovered["decision"] == "abstain" else "valid_response"
        )
        output.append(merged)
        recovered_count += 1
    return output, recovered_count
