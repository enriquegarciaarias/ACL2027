"""Variantes del endpoint binario ``stress`` vs ``non-stress``.

La variante **``schmidt``** reproduce la tarea binaria del paper original de
WESAD (Schmidt et al., 2018): ``stress`` frente a ``baseline`` + ``amusement``.
``meditation`` se excluye porque en el diseño original es una fase de
"de-excitación", no una clase afectiva. Las demás variantes son
**sensibilidades declaradas** para mostrar que la conclusión no depende de la
composición de clases:

- ``schmidt``: ``baseline`` + ``amusement`` (primario).
- ``baseline``: solo ``baseline`` (constructo homogéneo; máxima exposición al
  confusor de orden del protocolo).
- ``no_amusement``: ``baseline`` + ``meditation`` (aísla ``amusement``).
- ``all``: ``baseline`` + ``amusement`` + ``meditation`` (corpus histórico).

Este módulo no depende de ``phase3``: lo usan tanto el constructor de prompts
como el analizador de la jerarquía.
"""

from __future__ import annotations

DEFAULT_NONSTRESS_VARIANT = "schmidt"

#: Clases de WESAD que forman ``non-stress`` en cada variante.
NONSTRESS_VARIANTS: dict[str, tuple[str, ...]] = {
    "schmidt": ("baseline", "amusement"),
    "baseline": ("baseline",),
    "no_amusement": ("baseline", "meditation"),
    "all": ("baseline", "amusement", "meditation"),
}

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "schmidt": (
        "stress vs baseline+amusement (tarea binaria de Schmidt et al., 2018; "
        "excluye meditation)"
    ),
    "baseline": (
        "stress vs baseline (constructo homogéneo; máxima exposición al "
        "confusor de orden del protocolo)"
    ),
    "no_amusement": (
        "stress vs baseline+meditation (aísla el efecto de amusement)"
    ),
    "all": (
        "stress vs baseline+amusement+meditation (corpus binario histórico)"
    ),
}

_HUMAN_LABEL: dict[str, str] = {
    "baseline": "calm baseline",
    "amusement": "amusement",
    "meditation": "meditative states",
}

_HEADING_LABEL: dict[str, str] = {
    "baseline": "Non-stress (calm baseline)",
    "amusement": "Non-stress (amusement)",
    "meditation": "Non-stress (meditation)",
}


def variant_classes(variant: str = DEFAULT_NONSTRESS_VARIANT) -> tuple[str, ...]:
    """Devuelve las clases ``non-stress`` de una variante.

    Raises:
        ValueError: si la variante no existe.
    """
    try:
        return NONSTRESS_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"variante binaria desconocida: {variant!r}. "
            f"Disponibles: {sorted(NONSTRESS_VARIANTS)}"
        ) from None


def _join_human(classes: tuple[str, ...]) -> str:
    """Une etiquetas legibles: ``a``, ``a and b``, ``a, b and c``."""
    human = [_HUMAN_LABEL.get(name, name) for name in classes]
    if len(human) == 1:
        return human[0]
    if len(human) == 2:
        return f"{human[0]} and {human[1]}"
    return ", ".join(human[:-1]) + f" and {human[-1]}"


def render_task_note(classes: tuple[str, ...]) -> str:
    """Nota de tarea del prompt binario para un conjunto de clases ``non-stress``."""
    return (
        "Task: decide whether this 60-second window shows an acute stress "
        'response (decision "stress") or not (decision "non-stress"). The '
        f'"non-stress" class covers {_join_human(classes)}.'
    )


def render_rules(classes: tuple[str, ...]) -> tuple[str, ...]:
    """Reglas del prompt binario coherentes con las clases ``non-stress``."""
    return (
        '- Use "stress" only when the pattern indicates an acute stress response.',
        f'- Use "non-stress" for {_join_human(classes)}.',
    )


def heading_for(label: str) -> str:
    """Encabezado legible de un ejemplo few-shot ``non-stress``."""
    return _HEADING_LABEL.get(label, f"Non-stress ({label})")
