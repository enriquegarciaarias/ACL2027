"""Esquema JSON de salida del LLM y validador de respuestas (Fase 2, Paso 2).

Define el contrato de respuesta que el LLM debe devolver y valida que las
salidas crudas cumplan con él. Todo el esquema y la validación usan solo la
stdlib (sin dependencias externas como pydantic o jsonschema).

Esquema de salida (orden de campos: decision antes de reasoning):

{
  "decision": "baseline" | "stress" | "amusement" | "meditation" | "abstain",
  "confidence": <integer 0-100>,
  "claims": [
    {"metric": "<feature name>", "channel": "<modality>", "direction": "increase|decrease|stable", "magnitude": "low|medium|high"}
  ],
  "reasoning": "<brief explanation>"
}

Categorías de estado de una llamada al LLM:
- valid_response: JSON válido, decision en el conjunto permitido, confidence en 0-100.
- valid_abstention: JSON válido con decision == "abstain".
- invalid_output: JSON mal formado, campos faltantes, valores fuera de rango, etc.
- timeout: el modelo no respondió a tiempo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

# Decisiones permitidas (semánticas, no códigos numéricos de WESAD).
ALLOWED_DECISIONS: frozenset[str] = frozenset(
    {"baseline", "stress", "amusement", "meditation", "abstain"}
)

# Modo binario (P2): el endpoint primario es ``stress`` vs ``no-stress``.
BINARY_DECISIONS: frozenset[str] = frozenset({"stress", "non-stress", "abstain"})

# Validación externa con K-EmoCon (P3): arousal y valencia alta vs baja.
AROUSAL_DECISIONS: frozenset[str] = frozenset(
    {"high-arousal", "low-arousal", "abstain"}
)
VALENCE_DECISIONS: frozenset[str] = frozenset(
    {"high-valence", "low-valence", "abstain"}
)

# Control de generalidad (PhysioNet/CinC 2017): ritmo en ECG de una derivación.
RHYTHM_DECISIONS: frozenset[str] = frozenset(
    {"normal", "af", "other-rhythm", "noisy", "abstain"}
)

#: Modos de decisión soportados por el prompt y el validador.
DECISION_MODE_AFFECTIVE = "affective"
DECISION_MODE_STRESS_BINARY = "stress_binary"
DECISION_MODE_AROUSAL_BINARY = "arousal_binary"
DECISION_MODE_VALENCE_BINARY = "valence_binary"
DECISION_MODE_RHYTHM4 = "rhythm4"
DECISION_MODES: dict[str, frozenset[str]] = {
    DECISION_MODE_AFFECTIVE: ALLOWED_DECISIONS,
    DECISION_MODE_STRESS_BINARY: BINARY_DECISIONS,
    DECISION_MODE_AROUSAL_BINARY: AROUSAL_DECISIONS,
    DECISION_MODE_VALENCE_BINARY: VALENCE_DECISIONS,
    DECISION_MODE_RHYTHM4: RHYTHM_DECISIONS,
}

#: Variantes de escritura de ``non-stress`` que produce el modelo.
DECISION_ALIASES: dict[str, str] = {
    "nonstress": "non-stress",
    "non_stress": "non-stress",
    "no-stress": "non-stress",
    "no stress": "non-stress",
    "not stress": "non-stress",
    "notstress": "non-stress",
    "non-stressful": "non-stress",
    "high arousal": "high-arousal",
    "high_arousal": "high-arousal",
    "higharousal": "high-arousal",
    "low arousal": "low-arousal",
    "low_arousal": "low-arousal",
    "lowarousal": "low-arousal",
    "high valence": "high-valence",
    "high_valence": "high-valence",
    "low valence": "low-valence",
    "low_valence": "low-valence",
    "normal sinus rhythm": "normal",
    "nsr": "normal",
    "atrial fibrillation": "af",
    "a-fib": "af",
    "other rhythm": "other-rhythm",
    "other_rhythm": "other-rhythm",
    "other": "other-rhythm",
    "too noisy": "noisy",
    "noisy recording": "noisy",
}

# Direcciones permitidas para los claims.
ALLOWED_DIRECTIONS: frozenset[str] = frozenset({"increase", "decrease", "stable"})

# Magnitudes permitidas para los claims.
ALLOWED_MAGNITUDES: frozenset[str] = frozenset({"low", "medium", "high"})

# Alias de dirección/magnitud que el modelo produce con frecuencia. Se
# normalizan al conjunto canónico para que ``invalid_output`` mida fallos
# reales de formato y no variaciones de vocabulario.
DIRECTION_ALIASES: dict[str, str] = {
    "increased": "increase",
    "increasing": "increase",
    "rise": "increase",
    "rising": "increase",
    "up": "increase",
    "decreased": "decrease",
    "decreasing": "decrease",
    "fall": "decrease",
    "falling": "decrease",
    "down": "decrease",
    "unchanged": "stable",
    "no change": "stable",
    "same": "stable",
    "flat": "stable",
}
MAGNITUDE_ALIASES: dict[str, str] = {
    "very low": "low",
    "very_low": "low",
    "slightly low": "low",
    "slight": "low",
    "mild": "low",
    "moderate": "medium",
    "medium-low": "medium",
    "medium low": "medium",
    "medium-high": "medium",
    "medium high": "medium",
    "very high": "high",
    "very_high": "high",
    "slightly high": "high",
    "strong": "high",
    "severe": "high",
}

# Campos obligatorios del esquema JSON (en orden: decision antes de reasoning).
REQUIRED_FIELDS: tuple[str, ...] = ("decision", "confidence", "claims", "reasoning")

# Conjunto mínimo de canales experimental (EDA + HR). Los claims solo pueden
# referenciar features de estas modalidades.
EDA_HR_MODALITIES: frozenset[str] = frozenset({"ecg", "eda"})

# Canales de pecho que el prompt puede contener (EDA + HR + las modalidades
# adicionales resp/temp/bvp). Los claims solo pueden referenciar estos
# canales; los de muñeca y de movimiento (acc/emg) quedan excluidos.
ALLOWED_CLAIM_CHANNELS: frozenset[str] = frozenset(
    {"ecg", "eda", "resp", "temp"}
)

# Mapeo de títulos descriptivos de modalidad (usados en la representación
# textual) a su canal canónico. Solo canales de pecho; los títulos de muñeca
# o movimiento no se mapean para que sigan siendo rechazados.
CHANNEL_ALIASES: dict[str, str] = {
    "cardiovascular": "ecg",
    "heart rate": "ecg",
    "heartrate": "ecg",
    "electrodermal activity": "eda",
    "electrodermal": "eda",
    "skin conductance": "eda",
    "respiration": "resp",
    "respiratory": "resp",
    "breathing": "resp",
    "temperature": "temp",
    "skin temperature": "temp",
    "pulse": "bvp",
    "pulse rate": "bvp",
    "blood volume pulse": "bvp",
    "photoplethysmogram": "bvp",
    "ppg": "bvp",
}


@dataclass
class Claim:
    """Un claim estructurado del LLM sobre una feature específica."""

    metric: str
    channel: str
    direction: str
    magnitude: str

    def to_dict(self) -> dict[str, str]:
        return {
            "metric": self.metric,
            "channel": self.channel,
            "direction": self.direction,
            "magnitude": self.magnitude,
        }


@dataclass
class LlmResponse:
    """Respuesta parseada y validada del LLM."""

    decision: str
    confidence: int
    claims: list[Claim]
    reasoning: str
    raw_output: str
    status: str  # "valid_response" | "valid_abstention" | "invalid_output" | "timeout"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "claims": [c.to_dict() for c in self.claims],
            "reasoning": self.reasoning,
            "status": self.status,
            "error_message": self.error_message,
        }


def _is_int(value: Any) -> bool:
    """Devuelve True si value es un int (no bool, no float)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_claim(claim: Any, index: int) -> tuple[Claim | None, str | None]:
    """Valida un claim individual.

    Returns:
        (Claim, None) si es válido; (None, mensaje_error) si no.
    """
    if not isinstance(claim, dict):
        return None, f"claim[{index}] no es un objeto"

    for required in ("metric", "channel", "direction", "magnitude"):
        if required not in claim:
            return None, f"claim[{index}] falta campo '{required}'"

    metric = claim["metric"]
    channel = claim["channel"]
    direction = claim["direction"]
    magnitude = claim["magnitude"]

    if not isinstance(metric, str) or not metric:
        return None, f"claim[{index}].metric debe ser string no vacío"
    if not isinstance(channel, str) or not channel:
        return None, f"claim[{index}].channel debe ser string no vacío"
    if not isinstance(direction, str) or not direction:
        return None, f"claim[{index}].direction debe ser string no vacío"
    if not isinstance(magnitude, str) or not magnitude:
        return None, f"claim[{index}].magnitude debe ser string no vacío"

    # Normalizar a minúsculas para tolerar variaciones del modelo (ej. "EDA" vs "eda").
    channel = channel.lower().strip()
    direction = direction.lower().strip()
    magnitude = magnitude.lower().strip()
    direction = DIRECTION_ALIASES.get(direction, direction)
    magnitude = MAGNITUDE_ALIASES.get(magnitude, magnitude)

    # Extraer el nombre base del canal eliminando prefijos de ubicación
    # (ej. "wrist_eda" -> "eda", "chest_ecg" -> "ecg").
    if "_" in channel:
        channel = channel.rsplit("_", 1)[-1]

    # Extraer el canal de entre paréntesis en nombres descriptivos
    # (ej. "cardiovascular (ecg)" -> "ecg", "heart rate (ecg)" -> "ecg",
    #  "pulse (bvp/ppg)" -> "bvp").
    import re
    paren_match = re.search(r"\(([^)]+)\)", channel)
    if paren_match:
        channel = paren_match.group(1)
        # Si hay barra (ej. "bvp/ppg"), tomar solo la primera parte.
        if "/" in channel:
            channel = channel.split("/")[0]

    # Mapear títulos descriptivos sin paréntesis a su canal canónico
    # (ej. "cardiovascular" -> "ecg", "electrodermal activity" -> "eda").
    channel = CHANNEL_ALIASES.get(channel, channel)

    if direction not in ALLOWED_DIRECTIONS:
        return None, f"claim[{index}].direction '{direction}' no permitido"
    if magnitude not in ALLOWED_MAGNITUDES:
        return None, f"claim[{index}].magnitude '{magnitude}' no permitido"

    # Los claims solo pueden referenciar canales del conjunto de pecho permitido.
    if channel not in ALLOWED_CLAIM_CHANNELS:
        return None, (
            f"claim[{index}].channel '{channel}' no está en el conjunto "
            f"de canales permitidos {sorted(ALLOWED_CLAIM_CHANNELS)}"
        )

    return (
        Claim(metric=metric, channel=channel, direction=direction, magnitude=magnitude),
        None,
    )


def _extract_json_from_text(text: str) -> str | None:
    """Extrae el primer objeto JSON válido de un texto.

    Los modelos de lenguaje a menudo envuelven el JSON en texto explicativo,
    bloques de markdown (````json ... ````) o añaden prefijos/sufijos.
    Esta función busca el primer par de llaves balanceadas que forme un
    JSON válido.

    Args:
        text: texto crudo del modelo.

    Returns:
        string con el JSON extraído, o None si no se encontró ninguno.
    """
    # Buscar todos los posibles objetos JSON en el texto.
    # Los modelos pueden poner texto explicativo (con llaves) antes del JSON real.
    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        found_end = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_string:
                if ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            # No era un JSON válido; buscar el siguiente '{'
                            found_end = True
                            break
        if not found_end:
            # No se encontró llave de cierre balanceada para este '{'
            return None
        pos = start + 1


def _coerce_int(value: Any) -> int | None:
    """Intenta convertir un valor a entero.

    Acepta int (no bool), float que sea entero, o string numérico.
    Devuelve None si no se puede convertir.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, TypeError):
            return None
    return None


def _coerce_str(value: Any) -> str | None:
    """Intenta convertir un valor a string."""
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value).strip()


def validate_llm_output(
    raw_output: str,
    decision_mode: str = DECISION_MODE_AFFECTIVE,
) -> LlmResponse:
    """Valida una salida cruda del LLM contra el esquema JSON.

    Args:
        raw_output: string con la respuesta cruda del modelo (puede contener
            texto adicional, markdown, etc.).
        decision_mode: ``"affective"`` (4 clases + abstain) o
            ``"stress_binary"`` (stress / non-stress / abstain).

    Returns:
        LlmResponse con el estado de validación y, si es válido, los datos
        parseados. Nunca lanza excepción: los errores de validación se
        capturan y se reportan en el campo ``status``.

    Raises:
        ValueError: si ``decision_mode`` no es un modo conocido.
    """
    if decision_mode not in DECISION_MODES:
        raise ValueError(
            f"decision_mode no válido: {decision_mode}. "
            f"Valores permitidos: {sorted(DECISION_MODES)}"
        )
    allowed_decisions = DECISION_MODES[decision_mode]
    # 1. Intentar parsear el texto directamente; si falla, extraer JSON.
    data: dict | None = None
    parse_error: str | None = None

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            data = parsed
        else:
            parse_error = "La raíz del JSON no es un objeto"
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = f"JSON mal formado: {exc}"
        # Intentar extraer JSON embebido en texto.
        extracted = _extract_json_from_text(raw_output)
        if extracted is not None:
            try:
                parsed2 = json.loads(extracted)
                if isinstance(parsed2, dict):
                    data = parsed2
                    parse_error = None
                else:
                    parse_error = "El JSON embebido no es un objeto"
            except (json.JSONDecodeError, ValueError) as exc2:
                parse_error = f"JSON embebido inválido: {exc2}"

    if data is None:
        return LlmResponse(
            decision="",
            confidence=0,
            claims=[],
            reasoning="",
            raw_output=raw_output,
            status="invalid_output",
            error_message=parse_error or "No se pudo extraer JSON",
        )

    # 2. Campos obligatorios presentes.
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return LlmResponse(
            decision="",
            confidence=0,
            claims=[],
            reasoning="",
            raw_output=raw_output,
            status="invalid_output",
            error_message=f"Campos faltantes: {missing}",
        )

    # 3. Validar decision (aceptar case-insensitive y con espacios).
    decision_raw = _coerce_str(data["decision"])
    if decision_raw is None:
        return LlmResponse(
            decision="",
            confidence=0,
            claims=[],
            reasoning=str(data.get("reasoning", "")),
            raw_output=raw_output,
            status="invalid_output",
            error_message=f"decision no es string válida",
        )
    decision = decision_raw.strip().lower()
    decision = DECISION_ALIASES.get(decision, decision)
    allowed_lower = {d.lower() for d in allowed_decisions}
    if decision not in allowed_lower:
        return LlmResponse(
            decision=decision_raw,
            confidence=0,
            claims=[],
            reasoning=str(data.get("reasoning", "")),
            raw_output=raw_output,
            status="invalid_output",
            error_message=f"decision '{decision_raw}' no permitido. Valores: {sorted(allowed_decisions)}",
        )
    # Restaurar forma canónica (primera letra mayúscula para abstain, etc.)
    decision_map = {d.lower(): d for d in allowed_decisions}
    decision = decision_map[decision]

    # 4. Validar confidence (entero 0-100, aceptar string o float entero).
    confidence_raw = data["confidence"]
    confidence = _coerce_int(confidence_raw)
    if confidence is None or confidence < 0 or confidence > 100:
        return LlmResponse(
            decision=decision,
            confidence=0,
            claims=[],
            reasoning=str(data.get("reasoning", "")),
            raw_output=raw_output,
            status="invalid_output",
            error_message=f"confidence '{confidence_raw}' no es entero en 0-100",
        )

    # 5. Validar claims (array de objetos).
    claims_raw = data["claims"]
    if not isinstance(claims_raw, list):
        return LlmResponse(
            decision=decision,
            confidence=confidence,
            claims=[],
            reasoning=str(data.get("reasoning", "")),
            raw_output=raw_output,
            status="invalid_output",
            error_message="claims no es un array",
        )

    claims: list[Claim] = []
    for i, claim_raw in enumerate(claims_raw):
        claim, err = _validate_claim(claim_raw, i)
        if err:
            return LlmResponse(
                decision=decision,
                confidence=confidence,
                claims=[],
                reasoning=str(data.get("reasoning", "")),
                raw_output=raw_output,
                status="invalid_output",
                error_message=err,
            )
        claims.append(claim)

    # 6. Validar reasoning (string).
    reasoning = _coerce_str(data["reasoning"])
    if reasoning is None:
        return LlmResponse(
            decision=decision,
            confidence=confidence,
            claims=claims,
            reasoning="",
            raw_output=raw_output,
            status="invalid_output",
            error_message="reasoning no es un string",
        )

    # 7. Todo válido.
    status = "valid_abstention" if decision == "abstain" else "valid_response"
    return LlmResponse(
        decision=decision,
        confidence=confidence,
        claims=claims,
        reasoning=reasoning,
        raw_output=raw_output,
        status=status,
        error_message=None,
    )


def make_timeout_response(raw_output: str = "") -> LlmResponse:
    """Crea una respuesta de timeout (modelo no respondió a tiempo)."""
    return LlmResponse(
        decision="",
        confidence=0,
        claims=[],
        reasoning="",
        raw_output=raw_output,
        status="timeout",
        error_message="Tiempo de espera agotado",
    )


def schema_example() -> dict[str, Any]:
    """Devuelve un ejemplo del esquema JSON de salida."""
    return {
        "decision": "stress",
        "confidence": 87,
        "claims": [
            {
                "metric": "eda_phasic_std_us",
                "channel": "eda",
                "direction": "increase",
                "magnitude": "high",
            },
            {
                "metric": "ecg_hr_bpm",
                "channel": "ecg",
                "direction": "increase",
                "magnitude": "medium",
            },
        ],
        "reasoning": "EDA phasic variability and heart rate are both elevated "
        "relative to baseline, consistent with stress.",
    }
