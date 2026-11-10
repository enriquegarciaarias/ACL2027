"""Validación externa sobre datasets distintos de WESAD (P3).

- ``kemocon``: adaptador de K-EmoCon (EDA/BVP/temperatura de muñeca, tarea
  ``pre`` vs ``debate``).
"""

from __future__ import annotations

from .kemocon import (
    KEMOCON_FEATURES,
    KemoconWindow,
    aggregate_windows,
    build_payload,
    load_feature_rows,
    load_period_assignments,
    subject_centered,
)

__all__ = [
    "KEMOCON_FEATURES",
    "KemoconWindow",
    "aggregate_windows",
    "build_payload",
    "load_feature_rows",
    "load_period_assignments",
    "subject_centered",
]
