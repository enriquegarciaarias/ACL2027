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
from .claims import (
    parse_prompt_features,
    summarize_verdicts,
    verify_claim,
    verify_claims,
)
from .certainty import linguistic_certainty
from .records import evaluate_binary_records, evaluate_records

__all__ = [
    "evaluate_subject",
    "evaluate_all_subjects",
    "load_ground_truth",
    "load_responses",
    "match_opaque_to_window",
    "EvaluationResult",
    "evaluate_records",
    "evaluate_binary_records",
    "parse_prompt_features",
    "summarize_verdicts",
    "verify_claim",
    "verify_claims",
    "linguistic_certainty",
]
