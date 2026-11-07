"""Evaluación de respuestas del LLM (Fase 3).

Calcula métricas de evaluación para comparar las predicciones del LLM
contra el ground truth del corpus WESAD.

Métricas implementadas (solo stdlib):
- Corrección (accuracy)
- AUROC por clase (one-vs-rest, usando trapecio para el área bajo la curva)
- ECE (Expected Calibration Error)
- Brier score
- Tasa de abstención
- Matriz de confusión

Todo usa solo la biblioteca estándar de Python para mantener consistencia
con el resto del proyecto ACL2027.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Etiquetas permitidas (semánticas, no códigos numéricos de WESAD).
ALLOWED_DECISIONS: frozenset[str] = frozenset(
    {"baseline", "stress", "amusement", "meditation", "abstain"}
)

# Orden canónico para matriz de confusión.
LABEL_ORDER: tuple[str, ...] = ("baseline", "stress", "amusement", "meditation")


def _opaque_window_id(window_id: str) -> str:
    """Devuelve el identificador opaco de 8 caracteres (misma función que Fase 1)."""
    return "win_" + hashlib.sha256(window_id.encode()).hexdigest()[:8]


@dataclass
class EvaluationResult:
    """Resultado de evaluación para un sujeto."""

    subject: str
    n_windows: int
    n_valid: int  # respuestas válidas (valid_response + valid_abstention)
    n_abstain: int
    n_invalid: int
    accuracy: float  # solo sobre respuestas no-abstain
    accuracy_incl_abstain: float  # abstain cuenta como error
    brier_score: float
    ece: float
    auroc: dict[str, float]  # AUROC por clase (one-vs-rest)
    confusion_matrix: dict[str, dict[str, int]]
    decisions: dict[str, int]  # conteo de decisiones
    confidence_avg: float  # confianza promedio de respuestas no-abstain
    generated_at: str


def load_ground_truth(path: Path) -> dict[str, str]:
    """Carga el ground truth desde labels.jsonl.

    Returns:
        dict mapeando window_id -> label (string semántica).
    """
    gt: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            window_id = record.get("window_id", "")
            label = record.get("label", "")
            if isinstance(label, dict):
                label = label.get("name", str(label))
            gt[window_id] = str(label)
    return gt


def load_responses(path: Path) -> list[dict[str, Any]]:
    """Carga las respuestas del LLM desde S3.responses.jsonl."""
    responses: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            responses.append(json.loads(line))
    return responses


def match_opaque_to_window(
    responses: list[dict[str, Any]],
    ground_truth: dict[str, str],
) -> list[dict[str, Any]]:
    """Une respuestas con ground truth usando el ID opaco.

    Para cada respuesta, calcula el opaque ID de cada window_id en el ground
    truth y busca coincidencia con window_id_opaco de la respuesta.

    Returns:
        lista de dict con campos: true_label, predicted, confidence, status,
        representation, condition_id, window_id_opaco.
    """
    # Índice: opaque_id -> true_label
    opaque_to_label: dict[str, str] = {}
    for window_id, label in ground_truth.items():
        opaque_to_label[_opaque_window_id(window_id)] = label

    matched: list[dict[str, Any]] = []
    for resp in responses:
        opaque_id = resp.get("window_id_opaque", "")
        true_label = opaque_to_label.get(opaque_id, "UNKNOWN")
        matched.append(
            {
                "true_label": true_label,
                "predicted": resp.get("decision", ""),
                "confidence": resp.get("confidence", 0),
                "status": resp.get("status", ""),
                "representation": resp.get("representation", ""),
                "condition_id": resp.get("condition_id", ""),
                "window_id_opaco": opaque_id,
            }
        )
    return matched


def _compute_auroc_one_vs_rest(
    y_true: list[str],
    y_scores: dict[str, list[float]],
    positive_class: str,
) -> float:
    """Calcula AUROC para una clase (one-vs-rest) usando Mann-Whitney U.

    Args:
        y_true: lista de etiquetas verdaderas.
        y_scores: dict etiqueta -> lista de scores (confianzas normalizadas).
        positive_class: la clase para la cual calcular AUROC.

    Returns:
        valor de AUROC en [0, 1]. Devuelve 0.5 si no hay suficientes datos.
    """
    # Construir listas binarias: 1 si es la clase positiva, 0 en otro caso.
    y_binary = [1 if t == positive_class else 0 for t in y_true]
    scores = y_scores.get(positive_class, [])

    if len(scores) != len(y_binary):
        return 0.5

    # Separar los scores de positivos y negativos.
    pos_scores = [s for s, y in zip(scores, y_binary) if y == 1]
    neg_scores = [s for s, y in zip(scores, y_binary) if y == 0]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # AUROC como la estadística U de Mann-Whitney (equivalente al área bajo
    # la curva ROC). Los empates se resuelven con 0.5 por par igual.
    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5

    return wins / (n_pos * n_neg)


def _compute_brier_score(
    y_true: list[str],
    y_probs: dict[str, list[float]],
    labels: tuple[str, ...],
) -> float:
    """Calcula el Brier score multi-clase.

    Brier score = (1/N) * sum_i sum_c (y_ic - p_ic)^2

    donde y_ic es 1 si la muestra i pertenece a la clase c, y p_ic es la
    probabilidad predicha para la clase c en la muestra i.

    Nota: las decisiones "abstain" no tienen probabilidades de clase, así
    que se excluyen del cálculo (o se tratan como probabilidad uniforme).
    Aquí las excluimos.
    """
    total = 0.0
    count = 0
    for i, true_label in enumerate(y_true):
        if true_label not in labels:
            continue
        for c in labels:
            y_ic = 1.0 if true_label == c else 0.0
            p_ic = y_probs.get(c, [0.0] * len(y_true))[i]
            total += (y_ic - p_ic) ** 2
            count += 1
    return total / count if count > 0 else 0.0


def _compute_ece(
    y_true: list[str],
    y_pred: list[str],
    y_conf: list[float],
    n_bins: int = 10,
) -> float:
    """Calcula el Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    donde B_b es el conjunto de muestras cuyo caído de confianza cae en el
    bin b, acc es la precisión dentro del bin, y conf es la confianza media.

    Nota: las decisiones "abstain" se excluyen del cálculo.
    """
    # Crear bins de confianza.
    bins: list[dict[str, Any]] = [
        {"count": 0, "correct": 0, "conf_sum": 0.0} for _ in range(n_bins)
    ]

    for true_label, pred_label, conf in zip(y_true, y_pred, y_conf):
        if pred_label == "abstain":
            continue
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx]["count"] += 1
        bins[bin_idx]["conf_sum"] += conf
        if true_label == pred_label:
            bins[bin_idx]["correct"] += 1

    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0

    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        acc = b["correct"] / b["count"]
        avg_conf = b["conf_sum"] / b["count"]
        ece += (b["count"] / total) * abs(acc - avg_conf)

    return ece


def evaluate_subject(
    subject: str,
    responses_path: Path,
    ground_truth_path: Path,
) -> EvaluationResult:
    """Evalúa las respuestas de un sujeto contra el ground truth.

    Args:
        subject: identificador del sujeto (ej. "S3").
        responses_path: ruta al archivo S3.responses.jsonl.
        ground_truth_path: ruta al archivo labels.jsonl.

    Returns:
        EvaluationResult con todas las métricas calculadas.
    """
    from datetime import timezone, datetime

    ground_truth = load_ground_truth(ground_truth_path)
    responses = load_responses(responses_path)
    matched = match_opaque_to_window(responses, ground_truth)

    # Contadores generales: solo ventanas de este sujeto en el ground truth.
    subject_windows = {wid: lab for wid, lab in ground_truth.items()
                       if wid.startswith(f"{subject}:") or wid.startswith(f"WESAD:{subject}:")}
    n_windows = len(subject_windows)
    n_valid = 0
    n_abstain = 0
    n_invalid = 0
    correct = 0
    correct_incl_abstain = 0

    # Para AUROC, ECE, Brier: solo respuestas válidas (no abstain, no invalid).
    y_true: list[str] = []
    y_pred: list[str] = []
    y_conf: list[float] = []
    y_probs: dict[str, list[float]] = {c: [] for c in LABEL_ORDER}

    decisions: Counter = Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for m in matched:
        status = m["status"]
        true_label = m["true_label"]
        predicted = m["predicted"]
        confidence = m.get("confidence", 0) or 0

        decisions[predicted] += 1

        if status == "invalid_output":
            n_invalid += 1
            continue

        n_valid += 1

        if predicted == "abstain":
            n_abstain += 1
            correct_incl_abstain += 0  # abstain cuenta como error
            # Para AUROC/ECE/Brier, usar probabilidad uniforme (1/4) por clase.
            for c in LABEL_ORDER:
                y_probs[c].append(0.25)
            y_true.append(true_label)
            y_pred.append("abstain")
            y_conf.append(confidence)
            confusion[true_label]["abstain"] += 1
            continue

        # Respuesta válida con decisión concreta.
        y_true.append(true_label)
        y_pred.append(predicted)
        y_conf.append(confidence)

        # Probabilidad: confidence/100 para la clase predicha, 0 para las demás.
        for c in LABEL_ORDER:
            if c == predicted:
                y_probs[c].append(confidence / 100.0)
            else:
                y_probs[c].append(0.0)

        if true_label == predicted:
            correct += 1
            correct_incl_abstain += 1
        confusion[true_label][predicted] += 1

    # Métricas.
    # accuracy: solo sobre respuestas con decisión concreta (excluye abstain).
    # accuracy_incl_abstain: la abstención cuenta como error (denominador
    # incluye las abstenciones).
    n_decided = n_valid - n_abstain
    accuracy = correct / n_decided if n_decided > 0 else 0.0
    accuracy_incl_abstain = correct_incl_abstain / n_valid if n_valid > 0 else 0.0

    # AUROC por clase (one-vs-rest).
    auroc: dict[str, float] = {}
    for c in LABEL_ORDER:
        auroc[c] = _compute_auroc_one_vs_rest(y_true, y_probs, c)

    # Brier score (sobre respuestas válidas, incluyendo abstain con prob uniforme).
    brier = _compute_brier_score(y_true, y_probs, LABEL_ORDER)

    # ECE (solo decisiones no-abstain, confianza normalizada a [0,1]).
    y_true_non_abstain = [t for t, p in zip(y_true, y_pred) if p != "abstain"]
    y_pred_non_abstain = [p for p in y_pred if p != "abstain"]
    y_conf_non_abstain = [c / 100.0 for p, c in zip(y_pred, y_conf) if p != "abstain"]
    ece = _compute_ece(y_true_non_abstain, y_pred_non_abstain, y_conf_non_abstain)

    # Confianza promedio (sobre decisiones no-abstain, normalizada a [0,1]).
    non_abstain_confs = [c / 100.0 for p, c in zip(y_pred, y_conf) if p != "abstain"]
    confidence_avg = sum(non_abstain_confs) / len(non_abstain_confs) if non_abstain_confs else 0.0

    # Convertir confusion a dict normal.
    confusion_dict: dict[str, dict[str, int]] = {}
    for tl in LABEL_ORDER:
        confusion_dict[tl] = {pl: confusion[tl].get(pl, 0) for pl in list(LABEL_ORDER) + ["abstain"]}

    return EvaluationResult(
        subject=subject,
        n_windows=n_windows,
        n_valid=n_valid,
        n_abstain=n_abstain,
        n_invalid=n_invalid,
        accuracy=accuracy,
        accuracy_incl_abstain=accuracy_incl_abstain,
        brier_score=brier,
        ece=ece,
        auroc=auroc,
        confusion_matrix=confusion_dict,
        decisions=dict(decisions),
        confidence_avg=confidence_avg,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def evaluate_all_subjects(
    subjects: list[str],
    responses_dir: Path,
    ground_truth_path: Path,
) -> dict[str, EvaluationResult]:
    """Evalúa múltiples sujetos y devuelve un dict sujeto -> EvaluationResult."""
    results: dict[str, EvaluationResult] = {}
    for subject in subjects:
        responses_path = responses_dir / f"{subject}.responses.jsonl"
        if not responses_path.is_file():
            continue
        results[subject] = evaluate_subject(subject, responses_path, ground_truth_path)
    return results


def format_result(result: EvaluationResult) -> str:
    """Formatea un EvaluationResult como string legible."""
    lines: list[str] = []
    lines.append(f"=== Evaluación: {result.subject} ===")
    lines.append(f"  Ventanas totales: {result.n_windows}")
    lines.append(f"  Respuestas válidas: {result.n_valid}")
    lines.append(f"  Abstenciones: {result.n_abstain}")
    lines.append(f"  Invalid output: {result.n_invalid}")
    lines.append(f"  Accuracy (sólo válidas): {result.accuracy:.3f}")
    lines.append(f"  Accuracy (abstain=error): {result.accuracy_incl_abstain:.3f}")
    lines.append(f"  Brier score: {result.brier_score:.4f}")
    lines.append(f"  ECE: {result.ece:.4f}")
    lines.append(f"  Confianza promedio: {result.confidence_avg:.3f}")
    lines.append("")
    lines.append("  AUROC por clase:")
    for c in LABEL_ORDER:
        lines.append(f"    {c}: {result.auroc.get(c, 0.0):.3f}")
    lines.append("")
    lines.append("  Matriz de confusión (filas=real, cols=predicho):")
    header = "        " + " ".join(f"{c:>10}" for c in list(LABEL_ORDER) + ["abstain"])
    lines.append(header)
    for tl in LABEL_ORDER:
        row = confusion_dict = result.confusion_matrix.get(tl, {})
        cells = " ".join(f"{row.get(pl, 0):>10}" for pl in list(LABEL_ORDER) + ["abstain"])
        lines.append(f"  {tl:>10}: {cells}")
    lines.append("")
    lines.append("  Decisiones:")
    for d in sorted(result.decisions.keys()):
        lines.append(f"    {d}: {result.decisions[d]}")
    return "\n".join(lines)
