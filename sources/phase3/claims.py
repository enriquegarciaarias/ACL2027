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
from typing import Any, Mapping, Sequence

import numpy as np

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


def normalize_metric(name: str) -> str | None:
    """Normaliza un nombre de métrica (canónico o etiqueta legible)."""
    key = str(name).strip().lower()
    if key in CANONICAL_METRICS:
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


def parse_prompt_features(prompt: str) -> dict[str, ParsedFeature]:
    """Extrae las features mostradas por el prompt (``- nombre: valor unidad``)."""
    features: dict[str, ParsedFeature] = {}
    for line in prompt.splitlines():
        match = _CLAIM_LINE.match(line)
        if not match:
            continue
        canonical = normalize_metric(match.group(1))
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
) -> ClaimVerdict:
    """Verifica un claim contra las features parseadas del prompt."""
    metric_raw = str(claim.get("metric", "")).strip()
    metric = normalize_metric(metric_raw) or metric_raw
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
) -> list[ClaimVerdict]:
    """Verifica una lista de claims contra el prompt que las originó."""
    features = parse_prompt_features(prompt)
    return [verify_claim(claim, features) for claim in claims]


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
