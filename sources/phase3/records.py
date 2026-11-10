"""Métricas a nivel de registro para experimentos de Fase 2/3.

Funciones compartidas por los scripts de experimentación (``run_e12_pilot.py``,
``analyze_consistency.py``) para no duplicar el cálculo de accuracy, balanced
accuracy, AUROC, ECE, Brier y del subconjunto binario stress/baseline.

Un *registro* es un dict con al menos ``status``, ``decision`` y ``confidence``;
el emparejamiento con la etiqueta real se hace por identificador opaco de
ventana (``window_id_opaque``), nunca por el ID real.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np

from sources.phase3.evaluator import (
    LABEL_ORDER,
    _compute_auroc_one_vs_rest,
    _compute_brier_score,
    _compute_ece,
)


def evaluate_records(
    records: list[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
) -> dict[str, Any]:
    """Calcula métricas de clasificación sobre una lista de registros.

    Args:
        records: registros con ``status``, ``decision``, ``confidence`` y
            ``window_id_opaque``. Los registros sin etiqueta conocida se
            tratan como ``UNKNOWN`` (cuentan en el denominador de accuracy).
        label_by_opaque: mapping ``window_id_opaque -> etiqueta real``.

    Returns:
        dict con recuentos, accuracy, balanced accuracy, abstención, AUROC por
        clase, ECE, Brier, matriz de confusión y métricas del subconjunto
        binario stress/baseline.
    """
    rows = []
    for record in records:
        true_label = label_by_opaque.get(record.get("window_id_opaque", ""), "UNKNOWN")
        rows.append(
            {
                "true_label": true_label,
                "predicted": record.get("decision") or "",
                "confidence": record.get("confidence") or 0,
                "status": record.get("status") or "",
            }
        )
    valid = [r for r in rows if r["status"] != "invalid_output"]
    abstain = [r for r in valid if r["predicted"] == "abstain"]
    decided = [r for r in valid if r["predicted"] != "abstain"]
    correct = sum(1 for r in decided if r["predicted"] == r["true_label"])
    accuracy = correct / len(decided) if decided else 0.0
    recalls = []
    for label in LABEL_ORDER:
        total = sum(1 for r in decided if r["true_label"] == label)
        hit = sum(
            1
            for r in decided
            if r["true_label"] == label and r["predicted"] == label
        )
        if total:
            recalls.append(hit / total)
    balanced = float(np.mean(recalls)) if recalls else 0.0
    y_true = [r["true_label"] for r in decided]
    y_probs = {
        c: [
            r["confidence"] / 100.0 if r["predicted"] == c else 0.0
            for r in decided
        ]
        for c in LABEL_ORDER
    }
    if decided:
        auroc = {
            c: _compute_auroc_one_vs_rest(y_true, y_probs, c) for c in LABEL_ORDER
        }
        ece = _compute_ece(
            y_true,
            [r["predicted"] for r in decided],
            [r["confidence"] / 100.0 for r in decided],
        )
        brier = _compute_brier_score(y_true, y_probs, LABEL_ORDER)
    else:
        auroc = {c: float("nan") for c in LABEL_ORDER}
        ece = float("nan")
        brier = float("nan")
    confusion = {
        tl: {
            pl: sum(
                1 for r in valid if r["true_label"] == tl and r["predicted"] == pl
            )
            for pl in list(LABEL_ORDER) + ["abstain"]
        }
        for tl in LABEL_ORDER
    }
    binary_rows = [
        r
        for r in decided
        if r["true_label"] in ("stress", "baseline")
        and r["predicted"] in ("stress", "baseline")
    ]
    tp = sum(
        1
        for r in binary_rows
        if r["true_label"] == "stress" and r["predicted"] == "stress"
    )
    fn = sum(
        1
        for r in binary_rows
        if r["true_label"] == "stress" and r["predicted"] == "baseline"
    )
    tn = sum(
        1
        for r in binary_rows
        if r["true_label"] == "baseline" and r["predicted"] == "baseline"
    )
    fp = sum(
        1
        for r in binary_rows
        if r["true_label"] == "baseline" and r["predicted"] == "stress"
    )
    n_bin = len(binary_rows)
    binary = {
        "n": n_bin,
        "accuracy": (tp + tn) / n_bin if n_bin else 0.0,
        "precision_stress": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall_stress": tp / (tp + fn) if (tp + fn) else 0.0,
        "balanced_accuracy": (
            (tp / (tp + fn) if (tp + fn) else 0.0)
            + (tn / (tn + fp) if (tn + fp) else 0.0)
        )
        / 2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "n_abstain": len(abstain),
        "n_decided": len(decided),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "abstention_rate": len(abstain) / len(valid) if valid else 0.0,
        "auroc": auroc,
        "ece": ece,
        "brier": brier,
        "decisions": dict(Counter(r["predicted"] for r in valid)),
        "confusion": confusion,
        "binary": binary,
    }


def evaluate_binary_records(
    records: list[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    positive_label: str = "stress",
    negative_label: str = "non-stress",
    other_decisions_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Métricas del endpoint binario ``stress`` vs ``non-stress`` (P2).

    Convierte la etiqueta real de WESAD en binaria (``stress`` frente al resto:
    baseline, amusement, meditation) y evalúa las decisiones
    ``stress`` / ``non-stress`` / ``abstain``.

    La probabilidad de ``stress`` se deriva de la confianza del modelo:
    ``confidence/100`` si predijo ``stress`` y ``1 - confidence/100`` si predijo
    ``non-stress``. Así la confianza se usa como score ordenado para AUROC,
    Brier y ECE.

    Args:
        records: registros con ``status``, ``decision`` y ``confidence``.
        label_by_opaque: mapping ``window_id_opaque -> etiqueta real``.
        positive_label: etiqueta (real y predicha) de la clase positiva.
        negative_label: etiqueta de la clase negativa.
        other_decisions_map: mapeo opcional de decisiones fuera del conjunto
            binario a ``stress``/``non-stress`` (p. ej. para reinterpretar las
            respuestas del prompt de 4 clases: baseline, amusement y meditation
            → ``non-stress``). Si es ``None``, esas decisiones se ignoran.

    Returns:
        dict con recuentos, accuracy, balanced accuracy, precisión/recall/F1,
        cobertura, abstención, AUROC, Brier, ECE y matriz de confusión 2×3.
    """

    def to_binary(true_label: str) -> str | None:
        if true_label == "UNKNOWN":
            return None
        return positive_label if true_label == positive_label else negative_label

    def normalize_prediction(predicted: str) -> str:
        if predicted in (positive_label, negative_label, "abstain"):
            return predicted
        if other_decisions_map and predicted in other_decisions_map:
            return other_decisions_map[predicted]
        return predicted

    rows = []
    n_unknown = 0
    for record in records:
        true_label = label_by_opaque.get(record.get("window_id_opaque", ""), "UNKNOWN")
        binary_label = to_binary(true_label)
        if binary_label is None:
            n_unknown += 1
            continue
        rows.append(
            {
                "true_label": binary_label,
                "predicted": normalize_prediction(record.get("decision") or ""),
                "confidence": float(record.get("confidence") or 0),
                "status": record.get("status") or "",
            }
        )

    valid = [r for r in rows if r["status"] != "invalid_output"]
    abstain = [r for r in valid if r["predicted"] == "abstain"]
    decided = [
        r for r in valid if r["predicted"] in (positive_label, negative_label)
    ]

    tp = sum(
        1
        for r in decided
        if r["true_label"] == positive_label and r["predicted"] == positive_label
    )
    fn = sum(
        1
        for r in decided
        if r["true_label"] == positive_label and r["predicted"] == negative_label
    )
    tn = sum(
        1
        for r in decided
        if r["true_label"] == negative_label and r["predicted"] == negative_label
    )
    fp = sum(
        1
        for r in decided
        if r["true_label"] == negative_label and r["predicted"] == positive_label
    )

    n = len(decided)
    accuracy = (tp + tn) / n if n else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    y_true = [r["true_label"] for r in decided]
    p_positive = [
        r["confidence"] / 100.0
        if r["predicted"] == positive_label
        else 1.0 - r["confidence"] / 100.0
        for r in decided
    ]
    auroc = _compute_auroc_one_vs_rest(
        y_true,
        {positive_label: p_positive, negative_label: [1.0 - p for p in p_positive]},
        positive_label,
    )
    brier = (
        sum(
            (p - (1.0 if t == positive_label else 0.0)) ** 2
            for p, t in zip(p_positive, y_true)
        )
        / n
        if n
        else float("nan")
    )
    ece = _compute_ece(
        y_true,
        [r["predicted"] for r in decided],
        [r["confidence"] / 100.0 for r in decided],
    )

    return {
        "n": len(rows),
        "n_unknown_label": n_unknown,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "n_abstain": len(abstain),
        "n_decided": n,
        "coverage": n / len(rows) if rows else 0.0,
        "abstention_rate": len(abstain) / len(valid) if valid else 0.0,
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2 if n else 0.0,
        "precision_stress": precision,
        "recall_stress": recall,
        "specificity": specificity,
        "f1_stress": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auroc_stress": auroc,
        "brier": brier,
        "ece": ece,
        "decisions": dict(Counter(r["predicted"] for r in valid)),
        "confusion": {
            true_label: {
                predicted: sum(
                    1
                    for r in valid
                    if r["true_label"] == true_label
                    and r["predicted"] == predicted
                )
                for predicted in (positive_label, negative_label, "abstain")
            }
            for true_label in (positive_label, negative_label)
        },
    }
