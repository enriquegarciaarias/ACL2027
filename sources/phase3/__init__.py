"""Fase 3: Evaluación de respuestas del LLM.

Métricas de evaluación:
- Corrección (accuracy)
- AUROC por clase
- ECE (Expected Calibration Error)
- Brier score
- Tasa de abstención
- Matriz de confusión

Todo el código usa solo la stdlib (sin dependencias externas como scikit-learn
o scipy) para mantener el proyecto ligero, consistente con el resto del
proyecto ACL2027.
"""

from __future__ import annotations

from .evaluator import (
    evaluate_subject,
    evaluate_all_subjects,
    load_ground_truth,
    load_responses,
    match_opaque_to_window,
    EvaluationResult,
)

__all__ = [
    "evaluate_subject",
    "evaluate_all_subjects",
    "load_ground_truth",
    "load_responses",
    "match_opaque_to_window",
    "EvaluationResult",
]
