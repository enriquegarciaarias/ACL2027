"""Certeza lingüística: hedges y boosters (H2b / RQ2).

El índice de certeza se calcula sobre el campo ``reasoning`` de cada respuesta:
``(boosters - hedges) / palabras * 100``. Un valor alto indica lenguaje
asertivo; uno bajo, lenguaje cauteloso. La hipótesis H2b predice que los hedges
aumentarán y los boosters disminuirán con la degradación, pero menos de lo que
cae la corrección (desacoplamiento).

Los marcadores son listas cerradas y documentadas para que el cálculo sea
reproducible y auditable. Se recomienda validar una muestra con anotación
humana (previsto en H2b).
"""

from __future__ import annotations

import re
from typing import Any

#: Marcadores de cautela (multi-palabra antes que simples).
HEDGES: tuple[str, ...] = (
    "might",
    "may",
    "could",
    "possibly",
    "possible",
    "perhaps",
    "probably",
    "likely",
    "unlikely",
    "appears",
    "appear",
    "seems",
    "seem",
    "suggests",
    "suggest",
    "approximately",
    "about",
    "around",
    "roughly",
    "somewhat",
    "relatively",
    "tends to",
    "tend to",
    "uncertain",
    "unclear",
    "estimate",
    "estimated",
    "potential",
    "potentially",
    "borderline",
    "partially",
    "moderately",
    "mildly",
    "not available",
    "insufficient",
    "inconclusive",
    "ambiguous",
)

#: Marcadores de certeza/asertividad.
BOOSTERS: tuple[str, ...] = (
    "clearly",
    "obviously",
    "definitely",
    "certainly",
    "undoubtedly",
    "unquestionably",
    "strongly",
    "markedly",
    "substantially",
    "significantly",
    "consistently",
    "unequivocally",
    "always",
    "never",
    "must",
    "proves",
    "prove",
    "confirms",
    "confirm",
    "demonstrates",
    "demonstrate",
    "evident",
    "evidently",
    "unambiguous",
    "undeniably",
    "conclusive",
    "conclusively",
    "exactly",
    "completely",
    "entirely",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _count_markers(text: str, markers: tuple[str, ...]) -> dict[str, int]:
    """Cuenta marcadores (frases y palabras) sin doble conteo de subfrases."""
    lowered = text.lower()
    counts: dict[str, int] = {}
    remaining = lowered
    # Primero las expresiones multi-palabra para no contarlas dos veces.
    for marker in sorted(markers, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])"
        found = re.findall(pattern, remaining)
        if found:
            counts[marker] = len(found)
            remaining = re.sub(pattern, " ", remaining)
    return counts


def linguistic_certainty(text: str) -> dict[str, Any]:
    """Calcula hedges, boosters y el índice de certeza de un texto.

    Args:
        text: campo ``reasoning`` (u otro texto) del LLM.

    Returns:
        dict con ``n_words``, ``hedges``, ``boosters`` y ``certainty_index``
        (positivo = asertivo, negativo = cauteloso; por 100 palabras).
    """
    if not text:
        return {
            "n_words": 0,
            "hedges": 0,
            "boosters": 0,
            "hedge_terms": {},
            "booster_terms": {},
            "certainty_index": float("nan"),
        }
    words = _WORD.findall(text)
    hedge_terms = _count_markers(text, HEDGES)
    booster_terms = _count_markers(text, BOOSTERS)
    hedges = sum(hedge_terms.values())
    boosters = sum(booster_terms.values())
    n_words = len(words)
    index = (boosters - hedges) / n_words * 100.0 if n_words else float("nan")
    return {
        "n_words": n_words,
        "hedges": hedges,
        "boosters": boosters,
        "hedge_terms": hedge_terms,
        "booster_terms": booster_terms,
        "certainty_index": index,
    }
