"""Verificación reproducible de afirmaciones del LLM (H5 / RQ3).

Cada respuesta del LLM incluye ``claims`` con ``metric``, ``channel``,
``direction`` y ``magnitude``. Este módulo contrasta cada afirmación con los
valores que el modelo **realmente recibió**, parseados del propio prompt.

Categorías por claim:

- ``supported``: la dirección coincide con el valor y la magnitud no difiere en
  más de un nivel.
- ``contradicted``: la dirección se opone o la magnitud difiere en dos o más
  niveles (alucinación de patrón).
- ``unsupported``: la métrica no aparece en la entrada de esa ventana/condición
  (el modelo referencia datos que no recibió).
- ``indeterminate``: la métrica aparece pero el valor no está disponible
  (``not available``) o la escala no permite interpretar la dirección.

La verificación es determinista y no usa etiquetas. Se aplica a la escala
estandarizada (z intra-sujeto, unidad ``SD``); las representaciones absolutas
requieren una referencia por sujeto y, si no se aporta, quedan como
``indeterminate``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sources.phase3.signatures import NON_STRESS, STRESS, sign_of

#: Alias de métricas que usa la representación textual (etiquetas legibles)
#: y variantes que produce el modelo. Se normalizan al nombre canónico.
METRIC_ALIASES: dict[str, str] = {
    # ECG / cardiovascular
    "heart rate": "ecg_hr_bpm",
    "hr": "ecg_hr_bpm",
    "beats detected": "ecg_n_beats",
    "beat count": "ecg_n_beats",
    "beats": "ecg_n_beats",
    "mean rr interval": "ecg_rr_mean_ms",
    "mean rr": "ecg_rr_mean_ms",
    "rr interval": "ecg_rr_mean_ms",
    "mean ibi": "ecg_rr_mean_ms",
    "sdnn": "ecg_sdnn_ms",
    "rmssd": "ecg_rmssd_ms",
    # EDA
    "mean conductance": "eda_mean_us",
    "mean eda": "eda_mean_us",
    "eda mean": "eda_mean_us",
    "conductance variability": "eda_std_us",
    "eda variability": "eda_std_us",
    "eda std": "eda_std_us",
    "conductance trend": "eda_slope_us_per_s",
    "eda trend": "eda_slope_us_per_s",
    "eda slope": "eda_slope_us_per_s",
    "tonic level": "eda_tonic_mean_us",
    "eda tonic": "eda_tonic_mean_us",
    "tonic eda": "eda_tonic_mean_us",
    "phasic variability": "eda_phasic_std_us",
    "eda phasic": "eda_phasic_std_us",
    "scr count": "eda_scr_count",
    "number of scr": "eda_scr_count",
    "scr rate": "eda_scr_rate_per_min",
    "scr frequency": "eda_scr_rate_per_min",
    "mean scr amplitude": "eda_scr_amplitude_mean_us",
    "scr amplitude mean": "eda_scr_amplitude_mean_us",
    "maximum scr amplitude": "eda_scr_amplitude_max_us",
    "max scr amplitude": "eda_scr_amplitude_max_us",
    # Respiración
    "respiratory rate": "resp_rate_bpm",
    "respiration rate": "resp_rate_bpm",
    "breathing rate": "resp_rate_bpm",
    "resp rate": "resp_rate_bpm",
    "breath interval variability": "resp_interval_cv",
    "respiratory interval variability": "resp_interval_cv",
    "resp interval cv": "resp_interval_cv",
    "amplitude": "resp_amplitude_pct",
    "respiratory amplitude": "resp_amplitude_pct",
    "amplitude variability": "resp_std_pct",
    "respiratory amplitude variability": "resp_std_pct",
    "breath intervals detected": "resp_n_cycles",
    "breathing cycles": "resp_n_cycles",
    "resp cycles": "resp_n_cycles",
    # Temperatura
    "mean temperature": "temp_mean_degc",
    "temperature": "temp_mean_degc",
    "temperature variability": "temp_std_degc",
    "temp std": "temp_std_degc",
    "temperature trend": "temp_slope_degc_per_s",
    "temp slope": "temp_slope_degc_per_s",
    "temperature range": "temp_range_degc",
    "temp range": "temp_range_degc",
}

#: Nombres canónicos conocidos (los que aparecen en la vista numérica).
CANONICAL_METRICS: frozenset[str] = frozenset(
    {
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
        "ecg_sdnn_ms",
        "ecg_rmssd_ms",
        "ecg_n_beats",
        "eda_mean_us",
        "eda_std_us",
        "eda_slope_us_per_s",
        "eda_tonic_mean_us",
        "eda_phasic_std_us",
        "eda_scr_count",
        "eda_scr_rate_per_min",
        "eda_scr_amplitude_mean_us",
        "eda_scr_amplitude_max_us",
        "resp_rate_bpm",
        "resp_interval_cv",
        "resp_amplitude_pct",
        "resp_std_pct",
        "resp_n_cycles",
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
    }
)


def normalize_metric(
    name: str,
    extra_aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Normaliza un nombre de métrica (canónico o etiqueta legible).

    Args:
        name: nombre tal como aparece en el claim o en el prompt.
        extra_aliases: alias adicionales (``etiqueta -> canónica``) para
            vocabularios de otros datasets (p. ej. las features de movimiento
            de DREAMT). Sin este argumento el comportamiento es exactamente el
            de siempre, de modo que el corpus de WESAD no cambia.
    """
    key = str(name).strip().lower()
    if key in CANONICAL_METRICS:
        return key
    if extra_aliases:
        extra = {str(alias).strip().lower(): target for alias, target in extra_aliases.items()}
        if key in extra:
            return extra[key]
        # Un vocabulario de otro dataset escribe la misma etiqueta con guiones
        # bajos o con espacios («acc_mean» / «acc mean»). Se reintenta con la
        # otra forma antes de rendirse: si no, un nombre reconocible se contaría
        # como ``unsupported`` (métrica ausente) y la tasa de respaldo mediría el
        # vocabulario del verificador, no la conducta del modelo. Alcance
        # limitado a los vocabularios con ``extra_aliases``: el corpus de WESAD
        # no cambia.
        swapped = key.replace("_", " ") if "_" in key else key.replace(" ", "_")
        if swapped in extra:
            return extra[swapped]
        # Un vocabulario de otro dataset puede aportar nombres canónicos que no
        # existen en el de WESAD: ``acc_std_g`` estaba excluido de la vista del
        # LLM allí, así que no figura en ``CANONICAL_METRICS``. Si el nombre es
        # el destino de algún alias, es canónico para ese dataset; sin esto,
        # todos los claims de movimiento se contarían como ``unsupported``.
        if key in {str(target).strip().lower() for target in extra_aliases.values()}:
            return key
    return METRIC_ALIASES.get(key)


#: Umbral mínimo de |z| para que una dirección ``increase``/``decrease`` sea
#: compatible con el valor observado. Por debajo, la desviación es ambigua y
#: también sería compatible con ``stable``.
DIRECTION_LOW = 0.2
#: Umbral máximo de |z| para que ``stable`` sea compatible (banda de ambigüedad).
DIRECTION_STABLE = 0.5
#: Cortes de magnitud sobre |z|: low < 1 <= medium < 2 <= high.
MAGNITUDE_CUTS = (1.0, 2.0)
#: Diferencia máxima de nivel de magnitud tolerada para ``supported``.
MAGNITUDE_TOLERANCE = 1

_MAGNITUDE_LEVEL = {"low": 0, "medium": 1, "high": 2}
_DIRECTIONS = ("increase", "decrease", "stable")
_CLAIM_LINE = re.compile(r"^\s*(?:-\s+)?([A-Za-z][A-Za-z0-9 _/()\-]*?):\s*(.+?)\s*$")
_NUMBER = re.compile(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s+(\S+))?$")
_UNAVAILABLE = re.compile(r"^not available$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFeature:
    """Feature tal como apareció en el prompt."""

    name: str
    value: float | None
    unit: str | None
    raw: str

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def is_standardized(self) -> bool:
        return (self.unit or "").upper() == "SD"


@dataclass(frozen=True)
class ClaimVerdict:
    """Resultado de verificar un claim."""

    metric: str
    channel: str
    direction: str
    magnitude: str
    category: str  # supported | contradicted | unsupported | indeterminate
    observed: float | None
    observed_unit: str | None
    direction_ok: bool | None
    magnitude_ok: bool | None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "channel": self.channel,
            "direction": self.direction,
            "magnitude": self.magnitude,
            "category": self.category,
            "observed": self.observed,
            "observed_unit": self.observed_unit,
            "direction_ok": self.direction_ok,
            "magnitude_ok": self.magnitude_ok,
            "detail": self.detail,
        }


def parse_prompt_features(
    prompt: str,
    extra_aliases: Mapping[str, str] | None = None,
) -> dict[str, ParsedFeature]:
    """Extrae las features mostradas por el prompt (``- nombre: valor unidad``).

    ``extra_aliases`` amplía el vocabulario reconocido (features de otros
    datasets). Los nombres canónicos ya conocidos se resuelven antes.
    """
    features: dict[str, ParsedFeature] = {}
    for line in prompt.splitlines():
        match = _CLAIM_LINE.match(line)
        if not match:
            continue
        canonical = normalize_metric(match.group(1), extra_aliases)
        if canonical is None:
            continue  # línea no métrica (reglas, títulos, JSON del esquema)
        raw = match.group(2)
        if _UNAVAILABLE.match(raw):
            features[canonical] = ParsedFeature(
                name=canonical, value=None, unit=None, raw=raw
            )
            continue
        number = _NUMBER.match(raw)
        if not number:
            continue
        features[canonical] = ParsedFeature(
            name=canonical,
            value=float(number.group(1)),
            unit=number.group(2),
            raw=raw,
        )
    return features


def expected_direction(
    value: float,
    standardized: bool,
) -> str | None:
    """Dirección que implica un valor (``None`` si la escala no es comparable).

    Es la lectura con bandas disjuntas (``|z| < 0.2`` → ``stable``). Para
    verificar claims se usa ``direction_compatible``, que admite la banda de
    ambigüedad.
    """
    if not standardized:
        return None
    if value >= DIRECTION_LOW:
        return "increase"
    if value <= -DIRECTION_LOW:
        return "decrease"
    return "stable"


def direction_compatible(
    direction: str,
    value: float,
    standardized: bool,
) -> bool:
    """¿Es la dirección declarada compatible con el valor observado?

    Intervalos (en unidades SD):

    - ``increase``: ``z >= +0.2``
    - ``decrease``: ``z <= -0.2``
    - ``stable``: ``|z| <= 0.5``

    Los intervalos se solapan a propósito: una desviación de +0.3 SD es
    compatible tanto con ``increase`` como con ``stable``. Solo se marca
    ``contradicted`` cuando la dirección declarada es incompatible.
    """
    if not standardized:
        return False
    if direction == "increase":
        return value >= DIRECTION_LOW
    if direction == "decrease":
        return value <= -DIRECTION_LOW
    if direction == "stable":
        return abs(value) <= DIRECTION_STABLE
    return False


def magnitude_bucket(value: float, standardized: bool) -> str | None:
    """Nivel de magnitud (low/medium/high) a partir de |valor|."""
    if not standardized:
        return None
    magnitude = abs(value)
    if magnitude < MAGNITUDE_CUTS[0]:
        return "low"
    if magnitude < MAGNITUDE_CUTS[1]:
        return "medium"
    return "high"


def verify_claim(
    claim: Mapping[str, Any],
    features: Mapping[str, ParsedFeature],
    extra_aliases: Mapping[str, str] | None = None,
) -> ClaimVerdict:
    """Verifica un claim contra las features parseadas del prompt."""
    metric_raw = str(claim.get("metric", "")).strip()
    metric = normalize_metric(metric_raw, extra_aliases) or metric_raw
    channel = str(claim.get("channel", "")).strip()
    direction = str(claim.get("direction", "")).strip().lower()
    magnitude = str(claim.get("magnitude", "")).strip().lower()

    base = dict(
        metric=metric,
        channel=channel,
        direction=direction,
        magnitude=magnitude,
    )

    feature = features.get(metric)
    if feature is None:
        return ClaimVerdict(
            **base,
            category="unsupported",
            observed=None,
            observed_unit=None,
            direction_ok=None,
            magnitude_ok=None,
            detail="métrica ausente en la entrada",
        )
    if not feature.available:
        return ClaimVerdict(
            **base,
            category="indeterminate",
            observed=None,
            observed_unit=feature.unit,
            direction_ok=None,
            magnitude_ok=None,
            detail="valor no disponible en la entrada",
        )

    assert feature.value is not None
    # ``expected`` solo se usa en el detalle; la verificación usa compatibilidad.
    observed_bucket = magnitude_bucket(feature.value, feature.is_standardized)
    if observed_bucket is None:
        return ClaimVerdict(
            **base,
            category="indeterminate",
            observed=feature.value,
            observed_unit=feature.unit,
            direction_ok=None,
            magnitude_ok=None,
            detail="escala absoluta sin referencia; dirección no verificable",
        )

    direction_ok = direction_compatible(direction, feature.value, feature.is_standardized)
    if not direction_ok:
        return ClaimVerdict(
            **base,
            category="contradicted",
            observed=feature.value,
            observed_unit=feature.unit,
            direction_ok=False,
            magnitude_ok=None,
            detail=f"dirección {direction} incompatible con z={feature.value:+.2f}",
        )

    # Para claims de estabilidad la magnitud no aporta información discriminante.
    if direction == "stable":
        return ClaimVerdict(
            **base,
            category="supported",
            observed=feature.value,
            observed_unit=feature.unit,
            direction_ok=True,
            magnitude_ok=None,
            detail="",
        )

    magnitude_ok = (
        magnitude in _MAGNITUDE_LEVEL
        and abs(_MAGNITUDE_LEVEL[magnitude] - _MAGNITUDE_LEVEL[observed_bucket])
        <= MAGNITUDE_TOLERANCE
    )
    if magnitude_ok:
        return ClaimVerdict(
            **base,
            category="supported",
            observed=feature.value,
            observed_unit=feature.unit,
            direction_ok=True,
            magnitude_ok=True,
            detail="",
        )
    return ClaimVerdict(
        **base,
        category="contradicted",
        observed=feature.value,
        observed_unit=feature.unit,
        direction_ok=True,
        magnitude_ok=False,
        detail=f"magnitud declarada {magnitude}, observada {observed_bucket}",
    )


def verify_claims(
    claims: Sequence[Mapping[str, Any]],
    prompt: str,
    extra_aliases: Mapping[str, str] | None = None,
) -> list[ClaimVerdict]:
    """Verifica una lista de claims contra el prompt que las originó.

    El vocabulario del dataset tiene que llegar a las **dos** partes: al parsear
    el prompt (que convierte etiquetas legibles en nombres canónicos) y al
    normalizar la métrica de cada claim. Se olvidó lo segundo hasta el piloto
    DREAMT del 2026-11-23: el modelo citaba «Movement fraction» —una etiqueta
    que sí estaba en la entrada— y el verificador la contaba como ``unsupported``
    (métrica ausente). Inflaba el ``unsupported`` del brazo textual y no tocaba
    el numérico, donde los nombres ya son canónicos.
    """
    features = parse_prompt_features(prompt, extra_aliases)
    return [verify_claim(claim, features, extra_aliases) for claim in claims]


def summarize_verdicts(verdicts: Sequence[ClaimVerdict]) -> dict[str, Any]:
    """Resumen de categorías y de aciertos de dirección/magnitud."""
    total = len(verdicts)
    categories = {"supported": 0, "contradicted": 0, "unsupported": 0, "indeterminate": 0}
    for verdict in verdicts:
        categories[verdict.category] = categories.get(verdict.category, 0) + 1
    evaluable_direction = [v for v in verdicts if v.direction_ok is not None]
    evaluable_magnitude = [v for v in verdicts if v.magnitude_ok is not None]
    return {
        "n_claims": total,
        "categories": categories,
        "supported_rate": categories["supported"] / total if total else float("nan"),
        "contradicted_rate": categories["contradicted"] / total if total else float("nan"),
        "unsupported_rate": categories["unsupported"] / total if total else float("nan"),
        "indeterminate_rate": categories["indeterminate"] / total if total else float("nan"),
        "direction_accuracy": (
            float(np.mean([v.direction_ok for v in evaluable_direction]))
            if evaluable_direction
            else float("nan")
        ),
        "magnitude_accuracy": (
            float(np.mean([v.magnitude_ok for v in evaluable_magnitude]))
            if evaluable_magnitude
            else float("nan")
        ),
        "n_evaluable_direction": len(evaluable_direction),
    }


# ---------------------------------------------------------------------------
# Relevancia: el claim favorece o contradice la decisión (preregistro H5)
# ---------------------------------------------------------------------------
#
# La verdad del claim (``supported``/``contradicted``) y su **coherencia** con
# la decisión son ejes distintos. Este bloque implementa el segundo tal como
# quedó fijado en ``docs/h5_relevancia.md`` (2026-11-21): firma por clase en
# ``sources/phase3/signatures.py``, alineamiento ``a = signo × dirección`` y
# regla espejo para ``non-stress``.

#: Categorías de coherencia de un claim.
COHERENCE_CATEGORIES: tuple[str, ...] = (
    "coherent",
    "contrary",
    "neutral",
    "no_signature",
    "not_applicable",
)

#: Mínimo de claims con firma para publicar una tasa (preregistro §4.2).
MIN_CLAIMS_FOR_RATE = 30

_DIRECTION_SIGN: dict[str, int] = {"increase": +1, "decrease": -1, "stable": 0}

#: Lado binario de cada etiqueta de decisión.
#:
#: El preregistro define el eje para la tarea binaria. Las etiquetas de 4
#: clases se reinterpretan con el mismo mapa que la jerarquía ``schmidt``
#: (``stress`` frente al resto); es una **extensión declarada**, no parte del
#: experimento primario.
DECISION_SIDES: dict[str, str] = {
    STRESS: STRESS,
    NON_STRESS: NON_STRESS,
    "baseline": NON_STRESS,
    "amusement": NON_STRESS,
    "meditation": NON_STRESS,
}


@dataclass(frozen=True)
class ClaimCoherence:
    """Coherencia de un claim con la decisión de la respuesta."""

    metric: str
    channel: str
    direction: str
    sign: int
    alignment: int | None
    category: str  # coherent | contrary | neutral | no_signature | not_applicable
    side: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "channel": self.channel,
            "direction": self.direction,
            "sign": self.sign,
            "alignment": self.alignment,
            "coherence": self.category,
            "side": self.side,
        }


def coherence_side(decision: str | None) -> str | None:
    """Lado binario de una decisión (``None`` para ``abstain`` o desconocida)."""
    return DECISION_SIDES.get(str(decision or "").strip().lower())


def claim_coherence(
    claim: Mapping[str, Any],
    decision: str | None,
    *,
    features: Iterable[str] | None = None,
    signatures: Mapping[str, Mapping[str, int]] | None = None,
    extra_aliases: Mapping[str, str] | None = None,
) -> ClaimCoherence:
    """Coherencia de un claim con la decisión tomada.

    Dos modos:

    **Binario (por defecto).** ``a = signo(metric) × dirección`` y:

    - decisión ``stress``: ``a > 0`` coherente, ``a < 0`` contrario;
    - decisión ``non-stress``: ``a < 0`` coherente, ``a > 0`` contrario;
    - ``stable`` produce ``a = 0`` → **neutro** (no se penaliza hedgear);
    - métrica sin firma o fuera de ``features`` → ``no_signature``;
    - ``abstain`` → ``not_applicable`` (se cuenta aparte).

    **Multiclase** (``signatures`` no es ``None``). La tabla es
    ``{clase: {métrica: signo}}``; el claim es coherente con la clase
    predicha si ``signo(clase, métrica) × dirección > 0``. Una clase con tabla
    vacía (``n1`` en la tarea de sueño) devuelve ``no_signature``, no
    ``contrary``: no tener firma no es evidencia en contra.
    """
    metric_raw = str(claim.get("metric", "")).strip()
    metric = normalize_metric(metric_raw, extra_aliases) or metric_raw
    channel = str(claim.get("channel", "")).strip()
    direction = str(claim.get("direction", "")).strip().lower()
    declared = _DIRECTION_SIGN.get(direction)

    if signatures is not None:
        klass = str(decision or "").strip().lower()
        if not klass:
            return ClaimCoherence(
                metric, channel, direction, 0, None, "not_applicable", None
            )
        # Una decisión que no es una clase de la tabla (``abstain``) no se puede
        # juzgar: es «no aplicable», no «sin firma». La diferencia importa al
        # resumir: ``no_signature`` describe una clase sin tabla (``n1``).
        if klass not in signatures:
            return ClaimCoherence(
                metric, channel, direction, 0, None, "not_applicable", klass
            )
        sign = int(signatures.get(klass, {}).get(metric, 0) or 0)
        if features is not None and metric not in set(features):
            sign = 0
        if sign == 0 or declared is None:
            return ClaimCoherence(
                metric, channel, direction, sign, None, "no_signature", klass
            )
        alignment = sign * declared
        if alignment == 0:
            category = "neutral"
        else:
            category = "coherent" if alignment > 0 else "contrary"
        return ClaimCoherence(
            metric, channel, direction, sign, alignment, category, klass
        )

    side = coherence_side(decision)
    sign = sign_of(metric, features)

    if side is None:
        return ClaimCoherence(
            metric, channel, direction, sign, None, "not_applicable", side
        )
    if sign == 0 or declared is None:
        return ClaimCoherence(
            metric, channel, direction, sign, None, "no_signature", side
        )

    alignment = sign * declared
    if alignment == 0:
        category = "neutral"
    elif side == STRESS:
        category = "coherent" if alignment > 0 else "contrary"
    else:
        category = "coherent" if alignment < 0 else "contrary"
    return ClaimCoherence(
        metric, channel, direction, sign, alignment, category, side
    )


def response_coherence(
    claims: Sequence[Mapping[str, Any]],
    verdicts: Sequence[ClaimVerdict],
    decision: str | None,
    *,
    features: Iterable[str] | None = None,
    signatures: Mapping[str, Mapping[str, int]] | None = None,
    extra_aliases: Mapping[str, str] | None = None,
    subject: str = "",
) -> dict[str, Any]:
    """Bloque de coherencia de **una** respuesta.

    ``verdicts`` debe venir alineado con ``claims`` (misma longitud); solo se
    usa su categoría de verdad para la tabla conjunta. Con ``signatures`` se
    evalúa en modo multiclase y el «lado» pasa a ser la clase predicha.
    """
    pairs: list[dict[str, Any]] = []
    counts = {
        "coherent": 0,
        "contrary": 0,
        "neutral": 0,
        "no_signature": 0,
        "not_applicable": 0,
    }
    against_supported = 0
    sides: list[str | None] = []
    for index, claim in enumerate(claims):
        coherence = claim_coherence(
            claim,
            decision,
            features=features,
            signatures=signatures,
            extra_aliases=extra_aliases,
        )
        sides.append(coherence.side)
        counts[coherence.category] = counts.get(coherence.category, 0) + 1
        if coherence.category in {"no_signature", "not_applicable"}:
            continue
        truth = verdicts[index].category if index < len(verdicts) else "indeterminate"
        pair = {
            "subject": subject,
            "side": coherence.side,
            "truth": truth,
            "coherence": coherence.category,
        }
        pairs.append(pair)
        if coherence.category == "contrary" and truth == "supported":
            against_supported += 1

    with_signature = counts["coherent"] + counts["contrary"] + counts["neutral"]
    return {
        "side": (sides[0] if sides else None) or coherence_side(decision),
        "decision": decision,
        "n_claims": len(claims),
        "n_signature_claims": with_signature,
        "n_confirming": counts["coherent"],
        "n_against": counts["contrary"],
        "n_neutral": counts["neutral"],
        "n_no_signature": counts["no_signature"],
        "n_not_applicable": counts["not_applicable"],
        "n_against_supported": against_supported,
        "has_confirmation": counts["coherent"] > 0,
        "evidence_conflict": counts["contrary"] > counts["coherent"],
        "pairs": pairs,
    }


def _coherence_aggregate(
    pairs: Sequence[Mapping[str, Any]],
    *,
    min_claims: int = MIN_CLAIMS_FOR_RATE,
) -> dict[str, Any]:
    """Agregado de un conjunto de pares (claim × decisión)."""
    categories = {"coherent": 0, "contrary": 0, "neutral": 0}
    joint = {
        truth: {"coherent": 0, "contrary": 0, "neutral": 0}
        for truth in ("supported", "contradicted", "unsupported", "indeterminate")
    }
    n_supported = 0
    against_supported = 0
    for pair in pairs:
        coherence = pair.get("coherence")
        if coherence not in categories:
            continue
        categories[coherence] += 1
        truth = pair.get("truth", "indeterminate")
        if truth not in joint:
            truth = "indeterminate"
        joint[truth][coherence] += 1
        if truth == "supported":
            n_supported += 1
            if coherence == "contrary":
                against_supported += 1

    decided = categories["coherent"] + categories["contrary"]
    rate = categories["coherent"] / decided if decided else float("nan")
    return {
        "n_pairs": sum(categories.values()),
        "categories": categories,
        "coherence_rate": rate,
        "coherence_denominator": decided,
        "insufficient": decided < int(min_claims),
        "min_claims": int(min_claims),
        "n_supported": n_supported,
        "against_supported_share": (
            against_supported / n_supported if n_supported else float("nan")
        ),
        "joint": joint,
    }


def summarize_coherence(
    pairs: Sequence[Mapping[str, Any]],
    *,
    responses: Sequence[Mapping[str, Any]] = (),
    min_claims: int = MIN_CLAIMS_FOR_RATE,
    classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Resumen de relevancia por grupo (condición × modelo × representación).

    ``pairs`` son los pares emitidos por :func:`response_coherence` (uno por
    claim con firma y decisión). ``responses`` son esos mismos bloques
    respondidos, necesarios para ``zero_confirmation_share``.

    Con ``classes`` se evalúa en modo multiclase (una celda por clase) y
    ``responses`` pasa a tener un bloque ``per_class``. Sin ``classes`` el
    comportamiento es el binario de siempre.

    Si una celda tiene menos de ``min_claims`` claims decididos, se marca
    ``insufficient``: el preregistro prohíbe publicar esa tasa (la capa de
    presentación la oculta).
    """
    block = _coherence_aggregate(pairs, min_claims=min_claims)
    block["n_no_signature_total"] = sum(
        int(response.get("n_no_signature") or 0) for response in responses
    )
    block["n_not_applicable_total"] = sum(
        int(response.get("n_not_applicable") or 0) for response in responses
    )

    groups = tuple(classes) if classes else (STRESS, NON_STRESS)
    by_side: dict[str, dict[str, Any]] = {}
    for side in groups:
        by_side[side] = _coherence_aggregate(
            [pair for pair in pairs if pair.get("side") == side],
            min_claims=min_claims,
        )
    block["by_side"] = by_side

    if classes:
        per_class: dict[str, dict[str, Any]] = {}
        for klass in groups:
            decided = [r for r in responses if r.get("side") == klass]
            with_signature = [
                r for r in decided if (r.get("n_signature_claims") or 0) > 0
            ]
            per_class[klass] = {
                "n": len(decided),
                "zero_confirmation_share": (
                    sum(1 for r in with_signature if not r.get("has_confirmation"))
                    / len(with_signature)
                    if with_signature
                    else float("nan")
                ),
                "evidence_conflict_rate": (
                    sum(1 for r in decided if r.get("evidence_conflict"))
                    / len(decided)
                    if decided
                    else float("nan")
                ),
            }
        block["responses"] = {
            "n": len([r for r in responses if r.get("side") in groups]),
            "per_class": per_class,
        }
        return block

    decided_responses = [
        response for response in responses if response.get("side") in (STRESS, NON_STRESS)
    ]
    stress_responses = [
        response for response in decided_responses if response.get("side") == STRESS
    ]
    stress_with_signature = [
        response for response in stress_responses if (response.get("n_signature_claims") or 0) > 0
    ]
    block["responses"] = {
        "n": len(decided_responses),
        "n_stress": len(stress_responses),
        "n_non_stress": len(decided_responses) - len(stress_responses),
        "zero_confirmation_share": (
            sum(1 for response in stress_with_signature if not response.get("has_confirmation"))
            / len(stress_with_signature)
            if stress_with_signature
            else float("nan")
        ),
        "evidence_conflict_rate": (
            sum(1 for response in decided_responses if response.get("evidence_conflict"))
            / len(decided_responses)
            if decided_responses
            else float("nan")
        ),
        "contrary_supported_rate": (
            sum(int(response.get("n_against_supported") or 0) for response in decided_responses)
            / max(1, sum(int(response.get("n_against") or 0) for response in decided_responses))
        ),
        # Share de los claims *contrarios* que además son verdaderos. El
        # agregado principal (``against_supported_share``) divide por los
        # ``supported``, no por los contrarios: no son el mismo número.
    }
    return block
