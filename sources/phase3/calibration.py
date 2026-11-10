"""Calibración y predicción selectiva para el endpoint binario (P4).

Proporciona las herramientas que faltaban para evaluar y corregir la confianza
del LLM:

- Tabla de fiabilidad (reliability) y ECE/MCE.
- Brier y log-loss binarios.
- Escalado de temperatura (un parámetro) ajustado por validación cruzada
  agrupada por sujeto.
- Curva riesgo-cobertura (riesgo selectivo) y umbral por control conformal del
  riesgo (Angelopoulos et al., 2024), con garantía marginal sobre sujetos
  intercambiables.

Todo es stdlib+numpy (sin scikit-learn) y no usa etiquetas para la predicción:
las etiquetas solo se emplean para medir calibración y para ajustar el umbral de
abstención en el conjunto de calibración.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

_EPS = 1e-6


def _clip(probs: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)


def brier_score(y_true: Sequence[int], probs: Sequence[float]) -> float:
    """Brier score binario: media de (p - y)²."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: Sequence[int], probs: Sequence[float]) -> float:
    """Log-loss binario (entropía cruzada)."""
    y = np.asarray(y_true, dtype=float)
    p = _clip(probs)
    if y.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def reliability_table(
    y_true: Sequence[int],
    probs: Sequence[float],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Tabla de fiabilidad con bins de igual anchura en ``probs``.

    Args:
        y_true: 1 si la clase positiva (stress), 0 en otro caso.
        probs: probabilidad predicha de la clase positiva en [0, 1].
        n_bins: número de intervalos en [0, 1].

    Returns:
        lista de dicts con ``lo``, ``hi``, ``n``, ``mean_confidence`` y
        ``accuracy`` por bin (solo bins no vacíos).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        rows.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(np.sum(mask)),
                "mean_confidence": float(np.mean(p[mask])),
                "accuracy": float(np.mean(y[mask])),
            }
        )
    return rows


def expected_calibration_error(
    y_true: Sequence[int],
    probs: Sequence[float],
    n_bins: int = 10,
) -> float:
    """ECE: media ponderada de |accuracy - confianza| por bin."""
    rows = reliability_table(y_true, probs, n_bins)
    total = sum(r["n"] for r in rows)
    if total == 0:
        return float("nan")
    return float(
        sum(r["n"] / total * abs(r["accuracy"] - r["mean_confidence"]) for r in rows)
    )


def maximum_calibration_error(
    y_true: Sequence[int],
    probs: Sequence[float],
    n_bins: int = 10,
) -> float:
    """MCE: máximo |accuracy - confianza| entre bins no vacíos."""
    rows = reliability_table(y_true, probs, n_bins)
    if not rows:
        return float("nan")
    return float(max(abs(r["accuracy"] - r["mean_confidence"]) for r in rows))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(probs: Sequence[float], temperature: float) -> np.ndarray:
    """Aplica escalado de temperatura en el espacio logit."""
    p = _clip(probs)
    logits = np.log(p / (1.0 - p))
    return _sigmoid(logits / max(float(temperature), 1e-3))


def fit_temperature(
    probs: Sequence[float],
    y_true: Sequence[int],
    grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Ajusta la temperatura por búsqueda en rejilla minimizando log-loss.

    Args:
        probs: probabilidades predichas en [0, 1].
        y_true: etiquetas binarias.
        grid: valores de temperatura a probar (default: 0,25 a 4,0).

    Returns:
        dict con ``temperature``, ``log_loss_before``, ``log_loss_after`` y
        ``brier_before``/``brier_after``.
    """
    if grid is None:
        grid = tuple(np.round(np.arange(0.25, 4.01, 0.05), 2))
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in grid:
        loss = log_loss(y_true, apply_temperature(probs, temperature))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    return {
        "temperature": best_temperature,
        "log_loss_before": log_loss(y_true, probs),
        "log_loss_after": best_loss,
        "brier_before": brier_score(y_true, probs),
        "brier_after": brier_score(y_true, apply_temperature(probs, best_temperature)),
    }


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Curva riesgo-cobertura de predicción selectiva.

    Se acepta la predicción si ``confidence >= threshold``; en otro caso el
    sistema se abstiene.

    Args:
        confidences: confianza en la predicción emitida (en [0, 1]).
        correct: True si la predicción es correcta.
        thresholds: umbrales a evaluar (default: 0,50 a 0,99).

    Returns:
        lista de dicts con ``threshold``, ``coverage``, ``selective_accuracy``,
        ``n_accepted`` y ``n_total``.
    """
    if thresholds is None:
        thresholds = tuple(np.round(np.arange(0.50, 1.00, 0.01), 2))
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    total = conf.size
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = conf >= threshold
        n_accepted = int(np.sum(accepted))
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": n_accepted / total if total else 0.0,
                "selective_accuracy": (
                    float(np.mean(ok[accepted])) if n_accepted else float("nan")
                ),
                "n_accepted": n_accepted,
                "n_total": total,
            }
        )
    return curve


def conformal_risk_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    alpha: float = 0.1,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Umbral de abstención por control conformal del riesgo selectivo.

    Sigue el procedimiento de control conformal del riesgo de Angelopoulos et
    al. (2024): para cada umbral λ se estima el riesgo empírico
    ``R̂(λ) = (1/n) Σ L_i``, donde ``L_i = 1`` si la predicción se acepta y es
    incorrecta (0 si se abstiene o si acierta). Se elige el menor λ tal que
    ``n/(n+1) R̂(λ) + 1/(n+1) ≤ α``; con datos intercambiables, el riesgo
    selectivo queda acotado por α con probabilidad ≥ 1-δ.

    Args:
        confidences: confianza en la predicción emitida (en [0, 1]).
        correct: True si la predicción es correcta.
        alpha: riesgo selectivo objetivo (1 - precisión selectiva).
        thresholds: rejilla de umbrales (default: 0,50 a 0,99).

    Returns:
        dict con ``alpha``, ``threshold`` (``None`` si no hay umbral válido),
        ``calibration_risk``, ``coverage`` y ``n``.
    """
    if thresholds is None:
        thresholds = tuple(np.round(np.arange(0.50, 1.00, 0.01), 2))
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = conf.size
    if n == 0:
        return {
            "alpha": float(alpha),
            "threshold": None,
            "calibration_risk": float("nan"),
            "coverage": 0.0,
            "n": 0,
        }
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        accepted = conf >= threshold
        loss = np.where(accepted & ~ok, 1.0, 0.0)
        risk = float(np.mean(loss))
        bound = n / (n + 1) * risk + 1 / (n + 1)
        if bound <= alpha:
            best = {
                "alpha": float(alpha),
                "threshold": float(threshold),
                "calibration_risk": risk,
                "coverage": float(np.mean(accepted)),
                "n": n,
            }
            break
    if best is None:
        return {
            "alpha": float(alpha),
            "threshold": None,
            "calibration_risk": float("nan"),
            "coverage": 0.0,
            "n": n,
        }
    return best


def evaluate_selective(
    confidences: Sequence[float],
    correct: Sequence[bool],
    threshold: float | None,
) -> dict[str, Any]:
    """Métricas selectivas para un umbral dado (None = no abstenerse nunca)."""
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    total = conf.size
    if threshold is None:
        accepted = np.ones(total, dtype=bool)
    else:
        accepted = conf >= threshold
    n_accepted = int(np.sum(accepted))
    return {
        "threshold": None if threshold is None else float(threshold),
        "coverage": n_accepted / total if total else 0.0,
        "selective_accuracy": (
            float(np.mean(ok[accepted])) if n_accepted else float("nan")
        ),
        "selective_risk": (
            float(np.mean(~ok[accepted])) if n_accepted else float("nan")
        ),
        "n_accepted": n_accepted,
        "n_total": total,
    }


def group_kfold_indices(
    groups: Sequence[str],
    n_splits: int = 5,
) -> list[tuple[list[int], list[int]]]:
    """Índices de validación cruzada agrupada (sin fuga entre sujetos).

    Args:
        groups: identificador de grupo (sujeto) por muestra.
        n_splits: número de particiones.

    Returns:
        lista de ``(train_indices, test_indices)``.
    """
    unique = sorted(set(groups))
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for index, group in enumerate(unique):
        folds[index % n_splits].append(group)
    splits: list[tuple[list[int], list[int]]] = []
    for fold_groups in folds:
        test_set = set(fold_groups)
        test_idx = [i for i, g in enumerate(groups) if g in test_set]
        train_idx = [i for i, g in enumerate(groups) if g not in test_set]
        if train_idx and test_idx:
            splits.append((train_idx, test_idx))
    return splits


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))
