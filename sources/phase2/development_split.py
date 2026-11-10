"""Split congelado de desarrollo y evaluación (WESAD).

Motivo: durante la puesta a punto se iteró sobre S3 y el corpus completo
(rúbrica, patrón dominante, regla de consistencia, prompt binario, centrado) y
los ejemplos few-shot se calcularon con las **705 ventanas evaluadas**. Ambos
hechos inflan los resultados. Este módulo congela la partición y sirve de
fuente única de verdad para el código y la documentación.

Regla:
- **Desarrollo:** ``S2``, ``S3``, ``S4``. Sirven para diseñar prompts, reglas y
  ejemplos few-shot (incluidos sus estadísticos).
- **Evaluación:** el resto (``S5``–``S17`` salvo ``S12``, que no existe en el
  corpus). Solo estos sujetos cuentan para los resultados finales.

Ningún artefacto de evaluación debe usar estadísticos de los sujetos de
evaluación; los pocos ejemplos few-shot se calculan exclusivamente con los de
desarrollo (ver ``FEW_SHOT_PROVENANCE`` en ``prompt_builder``).
"""

from __future__ import annotations

from typing import Iterable, Sequence

SPLIT_VERSION = "1.0"

#: Sujetos de desarrollo (diseño de prompts, reglas y few-shot).
DEV_SUBJECTS: tuple[str, ...] = ("S2", "S3", "S4")

#: Sujetos de evaluación (resultados finales). S12 no está en el corpus WESAD.
TEST_SUBJECTS: tuple[str, ...] = (
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17",
)

ALL_SUBJECTS: tuple[str, ...] = DEV_SUBJECTS + TEST_SUBJECTS

#: Número de ventanas de 60 s de los sujetos de desarrollo y su distribución
#: por clase (usado para los ejemplos few-shot).
DEV_N_WINDOWS = 138
DEV_CLASS_COUNTS = {
    "baseline": 55,
    "stress": 30,
    "amusement": 17,
    "meditation": 36,
}


def normalize_subjects(subjects: Iterable[str]) -> list[str]:
    """Normaliza una lista de sujetos (mayúsculas, sin espacios, sin vacíos)."""
    return [s.strip().upper() for s in subjects if s and s.strip()]


def split_subjects(
    subjects: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Devuelve ``(desarrollo, evaluación)`` para la lista indicada."""
    candidates = normalize_subjects(subjects) if subjects else list(ALL_SUBJECTS)
    dev = [s for s in candidates if s in DEV_SUBJECTS]
    test = [s for s in candidates if s in TEST_SUBJECTS]
    return dev, test


def development_subjects(subjects: Iterable[str] | None = None) -> list[str]:
    """Sujetos de la lista que pertenecen al conjunto de desarrollo."""
    return split_subjects(subjects)[0]


def evaluation_subjects(subjects: Iterable[str] | None = None) -> list[str]:
    """Sujetos de la lista que pertenecen al conjunto de evaluación."""
    return split_subjects(subjects)[1]


def is_development_subject(subject: str) -> bool:
    return subject.strip().upper() in DEV_SUBJECTS


def contains_development_subjects(subjects: Sequence[str]) -> bool:
    return any(is_development_subject(s) for s in subjects)


def split_warning(subjects: Sequence[str]) -> str | None:
    """Mensaje de advertencia si la lista incluye sujetos de desarrollo."""
    dev = [s for s in subjects if is_development_subject(s)]
    if not dev:
        return None
    return (
        f"la lista incluye sujetos de DESARROLLO {dev}; los resultados no son de "
        "evaluación final (split congelado v"
        f"{SPLIT_VERSION}: dev={list(DEV_SUBJECTS)})"
    )
