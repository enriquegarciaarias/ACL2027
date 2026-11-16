"""Jerarquía del endpoint binario ``stress`` vs ``non-stress``.

Implementa el análisis primario (variante ``schmidt``: ``baseline`` +
``amusement``) y las sensibilidades declaradas (``baseline``, ``no_amusement``,
``all``). Dos decisiones metodológicas importantes:

- **Filtrado por clase real, no por resultado.** Cada variante decide qué
  clases de WESAD entran en ``non-stress``; las ventanas de las clases
  excluidas se descartan antes de calcular métricas. Nunca se elige la variante
  mirando el acierto.
- **Bootstrap agrupado por sujeto.** Las ventanas de un mismo sujeto están
  correlacionadas; remuestrear ventanas como si fueran independientes
  subestimaría los intervalos. El bootstrap remuestrea **sujetos** con
  reemplazo y conserva todas sus ventanas.

La temperatura no aparece aquí: el experimento primario la excluye de la
evidencia del LLM (confusor de orden y única modalidad no degradada). El
analizador filtra las respuestas ya generadas, así que este módulo es agnóstico
a la representación.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sources.phase2.binary_variants import (
    DEFAULT_NONSTRESS_VARIANT,
    NONSTRESS_VARIANTS,
    variant_classes,
)
from sources.phase3.records import evaluate_binary_records

POSITIVE_LABEL = "stress"
NEGATIVE_LABEL = "non-stress"
UNKNOWN_LABEL = "UNKNOWN"

#: Métricas que se bootstrapean por defecto (las que no dependen de la
#: prevalencia de forma problemática y sostienen las conclusiones).
DEFAULT_BOOTSTRAP_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "auroc_stress",
)


def _opaque(record: Mapping[str, Any]) -> str:
    """Identificador opaco de la ventana (nunca el ID real)."""
    return str(
        record.get("window_id_opaque")
        or record.get("window_id")
        or ""
    )


def _subject(record: Mapping[str, Any]) -> str:
    """Sujeto del registro (campo explícito o prefijo de ``window_id``)."""
    subject = record.get("subject")
    if subject:
        return str(subject)
    window_id = str(record.get("window_id") or "")
    parts = window_id.split(":")
    return parts[1] if len(parts) >= 2 else "UNKNOWN"


def _with_opaque(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copia los registros asegurando ``window_id_opaque``."""
    return [
        {**record, "window_id_opaque": _opaque(record)}
        for record in records
    ]


def true_label(
    record: Mapping[str, Any],
    label_by_opaque: Mapping[str, str],
) -> str:
    """Etiqueta real de un registro (``UNKNOWN`` si no está mapeada)."""
    return label_by_opaque.get(_opaque(record), UNKNOWN_LABEL)


def filter_records(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    variant: str = DEFAULT_NONSTRESS_VARIANT,
) -> list[dict[str, Any]]:
    """Conserva solo ``stress`` y las clases ``non-stress`` de la variante.

    Las ventanas de clases excluidas (p. ej. ``meditation`` en la variante
    ``schmidt``) y las de etiqueta desconocida se descartan.
    """
    allowed = {POSITIVE_LABEL, *variant_classes(variant)}
    return [
        {**record, "window_id_opaque": _opaque(record)}
        for record in records
        if true_label(record, label_by_opaque) in allowed
    ]


def base_rate(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
) -> float:
    """Prevalencia de ``stress`` entre los registros con etiqueta conocida.

    Es la referencia de un clasificador constante que siempre predice
    ``non-stress`` (accuracy = 1 - base_rate). Depende de la variante, así que
    se reporta en todas las tablas junto a accuracy y Brier.
    """
    known = [
        record
        for record in records
        if true_label(record, label_by_opaque) != UNKNOWN_LABEL
    ]
    if not known:
        return float("nan")
    n_stress = sum(
        1
        for record in known
        if true_label(record, label_by_opaque) == POSITIVE_LABEL
    )
    return n_stress / len(known)


def evaluate_variant(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    variant: str = DEFAULT_NONSTRESS_VARIANT,
) -> dict[str, Any]:
    """Métricas de una variante (filtrado por clase real + métricas binarias)."""
    kept = _with_opaque(filter_records(records, label_by_opaque, variant))
    metrics = dict(evaluate_binary_records(kept, label_by_opaque))
    metrics["variant"] = variant
    metrics["nonstress_classes"] = list(variant_classes(variant))
    metrics["base_rate"] = base_rate(kept, label_by_opaque)
    metrics["n_kept"] = len(kept)
    metrics["n_dropped"] = max(len(_with_opaque(records)) - len(kept), 0)
    return metrics


# ---------------------------------------------------------------------------
# Bootstrap agrupado por sujeto
# ---------------------------------------------------------------------------
def _subject_arrays(
    records: Sequence[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
) -> dict[str, dict[str, np.ndarray]]:
    """Arrays por sujeto: ``y`` (0/1), ``pred`` (0/1), ``p`` (score), ``valid``."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"y": [], "pred": [], "p": [], "valid": []}
    )
    for record in records:
        label = true_label(record, label_by_opaque)
        if label == UNKNOWN_LABEL:
            continue
        subject = _subject(record)
        module = groups[subject]
        module["y"].append(1.0 if label == POSITIVE_LABEL else 0.0)
        if record.get("status") == "invalid_output":
            module["pred"].append(0.0)
            module["p"].append(float("nan"))
            module["valid"].append(0.0)
            continue
        decision = record.get("decision")
        confidence = float(record.get("confidence") or 0.0) / 100.0
        if decision == POSITIVE_LABEL:
            module["pred"].append(1.0)
            module["p"].append(confidence)
            module["valid"].append(1.0)
        elif decision == NEGATIVE_LABEL:
            module["pred"].append(0.0)
            module["p"].append(1.0 - confidence)
            module["valid"].append(1.0)
        else:  # abstain / invalid / desconocido
            module["pred"].append(0.0)
            module["p"].append(float("nan"))
            module["valid"].append(0.0)
    return {
        subject: {key: np.asarray(values, dtype=float) for key, values in block.items()}
        for subject, block in groups.items()
    }


def _auroc_from_scores(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC vía estadística U de Mann-Whitney (empates a 0,5)."""
    positive = np.asarray([score for score, label in zip(p, y) if label == 1.0], dtype=float)
    negative = np.asarray([score for score, label in zip(p, y) if label == 0.0], dtype=float)
    positive = positive[~np.isnan(positive)]
    negative = negative[~np.isnan(negative)]
    n_pos, n_neg = positive.size, negative.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_positive = float(ranks[:n_pos].sum())
    u = rank_sum_positive - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _metrics_from_arrays(
    y: np.ndarray,
    pred: np.ndarray,
    p: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Métricas binarias a partir de arrays ya concatenados."""
    y_decided = y[valid == 1.0]
    pred_decided = pred[valid == 1.0]
    p_decided = p[valid == 1.0]
    n = int(y_decided.size)
    if n == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision_stress": float("nan"),
            "recall_stress": float("nan"),
            "f1_stress": float("nan"),
            "auroc_stress": float("nan"),
            "brier": float("nan"),
        }
    tp = int(np.sum((y_decided == 1.0) & (pred_decided == 1.0)))
    fn = int(np.sum((y_decided == 1.0) & (pred_decided == 0.0)))
    tn = int(np.sum((y_decided == 0.0) & (pred_decided == 0.0)))
    fp = int(np.sum((y_decided == 0.0) & (pred_decided == 1.0)))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) and not (
            math.isnan(precision) or math.isnan(recall)
        )
        else float("nan")
    )
    return {
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision_stress": precision,
        "recall_stress": recall,
        "f1_stress": f1,
        "auroc_stress": _auroc_from_scores(y_decided, p_decided),
        "brier": float(np.mean((p_decided - y_decided) ** 2)),
    }


def grouped_bootstrap_ci(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    variant: str = DEFAULT_NONSTRESS_VARIANT,
    metrics: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """IC 95 % por bootstrap agrupado por sujeto.

    Cada remuestra elige sujetos con reemplazo (no ventanas) y conserva todas
    sus ventanas. Devuelve, por métrica, ``point``, ``lo``, ``hi`` y ``n``
    (número de remuestras válidas).
    """
    kept = _with_opaque(filter_records(records, label_by_opaque, variant))
    groups = _subject_arrays(kept, label_by_opaque)
    subjects = sorted(groups)
    if not subjects:
        return {
            metric: {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
            for metric in metrics
        }

    stacked = {
        key: np.concatenate([groups[subject][key] for subject in subjects])
        for key in ("y", "pred", "p", "valid")
    }
    point = _metrics_from_arrays(**stacked)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(max(int(n_boot), 0)):
        chosen = [subjects[rng.randrange(len(subjects))] for _ in subjects]
        boot = {
            key: np.concatenate([groups[subject][key] for subject in chosen])
            for key in ("y", "pred", "p", "valid")
        }
        values = _metrics_from_arrays(**boot)
        for metric in metrics:
            value = values.get(metric, float("nan"))
            if value is not None and not math.isnan(value):
                samples[metric].append(float(value))

    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray(samples[metric], dtype=float)
        if values.size == 0:
            low = high = float("nan")
        else:
            low, high = np.percentile(values, [2.5, 97.5])
        result[metric] = {
            "point": float(point.get(metric, float("nan"))),
            "lo": float(low),
            "hi": float(high),
            "n": int(values.size),
        }
    return result


# ---------------------------------------------------------------------------
# Informes
# ---------------------------------------------------------------------------
def hierarchy_report(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    variants: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Informe de todas las variantes sobre el mismo conjunto de registros."""
    records = list(records)
    variants = list(variants) if variants else list(NONSTRESS_VARIANTS)
    report: dict[str, Any] = {}
    for variant in variants:
        point = evaluate_variant(records, label_by_opaque, variant)
        report[variant] = {
            "variant": variant,
            "nonstress_classes": list(variant_classes(variant)),
            "base_rate": point["base_rate"],
            "n_records": point["n_kept"],
            "n_dropped": point["n_dropped"],
            "metrics": point,
            "bootstrap": grouped_bootstrap_ci(
                records,
                label_by_opaque,
                variant=variant,
                metrics=metrics,
                n_boot=n_boot,
                seed=seed,
            ),
        }
    return report


def condition_hierarchy_report(
    records: Iterable[Mapping[str, Any]],
    label_by_opaque: Mapping[str, str],
    conditions: Sequence[str],
    variants: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_BOOTSTRAP_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Informe por condición (y un bloque ``pooled`` descriptivo)."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pooled: list[dict[str, Any]] = []
    for record in records:
        condition = str(record.get("condition_id") or "")
        by_condition[condition].append(dict(record))
        pooled.append(dict(record))

    report: dict[str, Any] = {}
    for condition in conditions:
        report[condition] = hierarchy_report(
            by_condition.get(condition, []),
            label_by_opaque,
            variants=variants,
            metrics=metrics,
            n_boot=n_boot,
            seed=seed,
        )
    report["pooled"] = hierarchy_report(
        pooled,
        label_by_opaque,
        variants=variants,
        metrics=metrics,
        n_boot=n_boot,
        seed=seed,
    )
    return report


def provenance(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Campos de procedencia presentes en los registros (modelo, variante…)."""
    records = list(records)
    first = records[0] if records else {}
    return {
        "model": first.get("model"),
        "representation": first.get("representation"),
        "decision_mode": first.get("decision_mode"),
        "nonstress_variant": first.get("nonstress_variant"),
        "include_temperature": first.get("include_temperature"),
        "n_records": len(records),
        "subjects": sorted({_subject(record) for record in records}),
    }
