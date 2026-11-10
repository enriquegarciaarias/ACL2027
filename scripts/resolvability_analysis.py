#!/usr/bin/env python
"""Análisis de resolubilidad estadística del corpus WESAD (Fase 3).

Objetivo: determinar si las clases afectivas son separables con el mismo
conjunto de features que ve el LLM y si el LLM está por debajo de un modelo
de referencia con la misma información. Sirve para justificar (o refutar) el
paso a un experimento binario stress vs no-stress.

Diseño:
- Features: las 23 features de pecho de ``LLM_ALLOWED_FEATURES``.
- Etiquetas: ``baseline``, ``stress``, ``amusement``, ``meditation``.
- Validación: ``GroupKFold(5)`` por sujeto (sin fuga entre sujetos).
- Referencia: regresión logística (lineal) y Random Forest (no lineal), con
  imputación de mediana y estandarización; ``class_weight='balanced'``.
- Métricas: balanced accuracy, macro-F1, AUROC; IC 95 % bootstrap agrupado
  por sujeto; p-valor de permutación.

Controles temporales (WESAD es un protocolo secuencial):
- ``time_only``: la fase del protocolo (0-1 por sujeto) como único predictor.
  Cuantifica el confusor de orden.
- ``features_plus_time``: features + fase; mide el valor incremental.
- ``time_residualized``: features residualizadas sobre fase² por sujeto
  (normalización temporal no supervisada; sin usar etiquetas).
- ``subject_centered``: z-score intra-sujeto (elimina identidad del sujeto).

Regla de decisión (margen preespecificado 0.60 para balanced accuracy):
- separable: IC95 inferior > 0.60.
- equivalente al azar en el margen: IC95 superior < 0.60.
- inconcluso: en otro caso.

Uso:
    python scripts/resolvability_analysis.py
    python scripts/resolvability_analysis.py --permutations 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sources.ingestion.represent import LLM_ALLOWED_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
REPORT_PATH = (
    REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "resolvability_report.json"
)
EVALUATION_PATH = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "evaluation_report.json"

CHEST_MODALITIES = ("ecg", "eda", "resp", "temp")
LABELS = ("baseline", "stress", "amusement", "meditation")
PAIRS = [
    ("baseline", "stress"),
    ("baseline", "amusement"),
    ("baseline", "meditation"),
    ("stress", "amusement"),
    ("stress", "meditation"),
    ("amusement", "meditation"),
]
CHANCE_4 = 1 / len(LABELS)
MARGIN_4 = CHANCE_4 + 0.10
MARGIN_BINARY = 0.60


def load_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """Carga features, etiquetas, tiempo y fase por sujeto."""
    rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
    for features_path in sorted(OUTPUT_DIR.glob("features/S*.features.jsonl")):
        subject = features_path.name.split(".")[0]
        windows_path = OUTPUT_DIR / "windows" / f"{subject}.windows.jsonl"
        windows = {
            json.loads(line)["window_id"]: json.loads(line)
            for line in windows_path.open(encoding="utf-8")
        }
        subject_rows: list[dict[str, Any]] = []
        for line in features_path.open(encoding="utf-8"):
            record = json.loads(line)
            window = windows.get(record["window_id"])
            if window is None:
                continue
            values: dict[str, float] = {}
            for modality in CHEST_MODALITIES:
                for name, meta in record.get("features", {}).get(modality, {}).items():
                    if name not in LLM_ALLOWED_FEATURES:
                        continue
                    value = meta.get("value")
                    values[name] = float(value) if value is not None else np.nan
                    feature_names.add(name)
            subject_rows.append(
                {
                    "subject": subject,
                    "window_id": record["window_id"],
                    "label": window["label"]["name"],
                    "t_start_s": float(window["t_start_s"]),
                    "values": values,
                }
            )
        # Fase 0-1 dentro del sujeto.
        times = [r["t_start_s"] for r in subject_rows]
        lo, hi = min(times), max(times)
        for row in subject_rows:
            row["phase"] = (row["t_start_s"] - lo) / (hi - lo) if hi > lo else 0.0
        rows.extend(subject_rows)
    return rows, sorted(feature_names)


def to_matrix(rows: list[dict[str, Any]], feature_names: list[str], key: str) -> np.ndarray:
    return np.array(
        [
            [row["phase"] if name == "phase" else row[key].get(name, np.nan) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )


def subject_center(rows: list[dict[str, Any]], feature_names: list[str]) -> None:
    """Añade ``centered``: z-score intra-sujeto."""
    by_subject: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_subject[row["subject"]].append(i)
    for idx in by_subject.values():
        block = np.array(
            [[rows[i]["values"].get(f, np.nan) for f in feature_names] for i in idx],
            dtype=float,
        )
        mean = np.nanmean(block, axis=0)
        std = np.nanstd(block, axis=0)
        std[std == 0] = 1.0
        centered = (block - mean) / std
        for k, i in enumerate(idx):
            rows[i]["centered"] = {
                f: float(centered[k, j]) for j, f in enumerate(feature_names)
            }


def residualize_on_phase(rows: list[dict[str, Any]], feature_names: list[str]) -> None:
    """Añade ``resid``: residuo de cada feature tras ajustar fase² por sujeto.

    Es una normalización temporal no supervisada (no usa etiquetas). Se ajusta
    por sujeto, por lo que asume que en despliegue se dispone de la sesión del
    sujeto para estimar su tendencia temporal.
    """
    by_subject: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_subject[row["subject"]].append(i)
    for idx in by_subject.values():
        phase = np.array([rows[i]["phase"] for i in idx], dtype=float)
        design = np.vstack([np.ones_like(phase), phase, phase**2]).T
        for feature in feature_names:
            values = np.array(
                [rows[i]["values"].get(feature, np.nan) for i in idx], dtype=float
            )
            mask = ~np.isnan(values)
            if mask.sum() < 5:
                residual = np.full_like(values, np.nan)
            else:
                coef, *_ = np.linalg.lstsq(design[mask], values[mask], rcond=None)
                residual = values - design @ coef
            for k, i in enumerate(idx):
                rows[i].setdefault("resid", {})[feature] = float(residual[k])


def make_model(kind: str) -> Pipeline:
    if kind == "logreg":
        classifier = LogisticRegression(
            max_iter=5000, class_weight="balanced", solver="lbfgs"
        )
    elif kind == "rf":
        classifier = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def grouped_predictions(
    X: np.ndarray, y: np.ndarray, groups: list[str], kind: str
) -> tuple[np.ndarray, np.ndarray]:
    cv = GroupKFold(n_splits=5)
    proba = cross_val_predict(
        make_model(kind), X, y, groups=groups, cv=cv, method="predict_proba", n_jobs=1
    )
    pred = np.argmax(proba, axis=1)
    positive = proba[:, 1] if proba.shape[1] == 2 else proba[:, -1]
    return pred, positive


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    groups: list[str],
    metric: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(groups))
    indices_by_group = {g: np.where(np.array(groups) == g)[0] for g in unique}
    stats: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([indices_by_group[g] for g in sampled])
        y_b, p_b = y[idx], pred[idx]
        if len(set(y_b)) < 2:
            continue
        if metric == "balanced_accuracy":
            stats.append(balanced_accuracy_score(y_b, p_b))
        else:
            stats.append(f1_score(y_b, p_b, average="macro"))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def permutation_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    observed: float,
    n_perm: int,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        pred, _ = grouped_predictions(X, y_perm, groups, "logreg")
        null.append(balanced_accuracy_score(y_perm, pred))
    return float((1 + np.sum(np.array(null) >= observed)) / (n_perm + 1))


def univariate_auroc(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 10 or len(set(y[mask])) < 2:
            continue
        try:
            auc = roc_auc_score(y[mask], col[mask])
        except ValueError:
            continue
        result[name] = float(max(auc, 1 - auc))
    return result


def call_at_margin(ci: tuple[float, float], margin: float) -> str:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return "sin_datos"
    if lo > margin:
        return "separable"
    if hi < margin:
        return "equivalente_al_azar_en_margen"
    return "inconcluso"


def evaluate_pair(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    label_a: str,
    label_b: str,
    key: str,
    n_perm: int,
) -> dict[str, Any]:
    subset = [r for r in rows if r["label"] in (label_a, label_b)]
    X = to_matrix(subset, feature_names, key)
    subjects = [r["subject"] for r in subset]
    y = np.array([0 if r["label"] == label_a else 1 for r in subset])

    pred_lr, proba_lr = grouped_predictions(X, y, subjects, "logreg")
    pred_rf, _ = grouped_predictions(X, y, subjects, "rf")
    bal_lr = float(balanced_accuracy_score(y, pred_lr))
    bal_rf = float(balanced_accuracy_score(y, pred_rf))
    try:
        auc_lr = float(roc_auc_score(y, proba_lr))
    except ValueError:
        auc_lr = float("nan")
    ci = bootstrap_ci(y, pred_lr, subjects, "balanced_accuracy")
    p = permutation_pvalue(X, y, subjects, bal_lr, n_perm) if n_perm else None
    uni = univariate_auroc(X, y, feature_names)
    best_feature = max(uni, key=uni.get) if uni else None
    return {
        "pair": [label_a, label_b],
        "n": int(len(y)),
        "n_per_class": {label_a: int(np.sum(y == 0)), label_b: int(np.sum(y == 1))},
        "logreg_balanced_accuracy": bal_lr,
        "logreg_ci95": list(ci),
        "logreg_macro_f1": float(f1_score(y, pred_lr, average="macro")),
        "logreg_auroc": auc_lr,
        "rf_balanced_accuracy": bal_rf,
        "permutation_p": p,
        "call_at_margin": call_at_margin(ci, MARGIN_BINARY),
        "best_univariate_feature": best_feature,
        "best_univariate_auroc": float(uni[best_feature]) if best_feature else None,
    }


def evaluate_multiclass(
    rows: list[dict[str, Any]], feature_names: list[str], key: str, kind: str = "logreg"
) -> dict[str, Any]:
    subset = [r for r in rows if r["label"] in LABELS]
    X = to_matrix(subset, feature_names, key)
    subjects = [r["subject"] for r in subset]
    y = np.array([LABELS.index(r["label"]) for r in subset])
    pred, _ = grouped_predictions(X, y, subjects, kind)
    bal = float(balanced_accuracy_score(y, pred))
    ci = bootstrap_ci(y, pred, subjects, "balanced_accuracy")
    return {
        "n": int(len(y)),
        "balanced_accuracy": bal,
        "ci95": list(ci),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "call_at_margin": call_at_margin(ci, MARGIN_4),
        "predicted_counts": {LABELS[i]: int(np.sum(pred == i)) for i in range(len(LABELS))},
        "true_counts": {LABELS[i]: int(np.sum(y == i)) for i in range(len(LABELS))},
    }


def transition_gaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Gaps temporales entre condiciones adyacentes (diagnóstico)."""
    gaps: list[float] = []
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    for subject, subject_rows in by_subject.items():
        ordered = sorted(subject_rows, key=lambda r: r["t_start_s"])
        for prev, curr in zip(ordered, ordered[1:]):
            if prev["label"] != curr["label"]:
                gaps.append(curr["t_start_s"] - prev["t_start_s"])
                _ = subject
    return {
        "n_transitions": len(gaps),
        "median_gap_s": float(np.median(gaps)) if gaps else None,
        "min_gap_s": float(np.min(gaps)) if gaps else None,
        "max_gap_s": float(np.max(gaps)) if gaps else None,
        "note": (
            "Los huecos entre condiciones incluyen fases de preparación "
            "(cientos de segundos), por lo que no hay ventanas fisiológicamente "
            "adyacentes para un análisis de transición estricto. Se reporta como "
            "limitación y se recomienda un diseño con transiciones continuas."
        ),
    }


def llm_comparison(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    binary = report.get("binary_stress_vs_baseline", {})
    result: dict[str, Any] = {}
    for representation, metrics in binary.items():
        tp, fp = metrics["tp"], metrics["fp"]
        tn, fn = metrics["tn"], metrics["fn"]
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        result[representation] = {
            "accuracy": metrics["accuracy"],
            "precision_stress": metrics["precision_stress"],
            "recall_stress": recall,
            "specificity_baseline": specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "coverage": metrics["coverage"],
            "abstention_rate": metrics["abstention_rate"],
            "n_decided": metrics["n_decided"],
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args(argv)

    rows, feature_names = load_rows()
    subject_center(rows, feature_names)
    residualize_on_phase(rows, feature_names)
    print(f"Ventanas: {len(rows)} | features: {len(feature_names)}")
    print("Features:", ", ".join(feature_names))

    report: dict[str, Any] = {
        "n_windows": len(rows),
        "features": feature_names,
        "labels": list(LABELS),
        "method": {
            "cv": "GroupKFold(5) por sujeto",
            "models": "LogisticRegression(balanced) y RandomForest(balanced_subsample)",
            "bootstrap": "1000 remuestreos agrupados por sujeto",
            "permutation": "etiquetas globales, misma CV",
            "margins": {"binary": MARGIN_BINARY, "multiclass": MARGIN_4},
            "decision_rule": (
                "separable si IC95 inferior > margen; equivalente al azar en el "
                "margen si IC95 superior < margen; inconcluso en otro caso"
            ),
        },
        "reference": {},
        "transitions": transition_gaps(rows),
        "llm_comparison": llm_comparison(EVALUATION_PATH),
    }

    # 1. Referencia con las mismas features que ve el LLM (sin tiempo).
    print("\n=== Referencia: features solamente (mismas que el LLM) ===")
    report["reference"]["features_only"] = {
        "pairs": [],
        "multiclass_logreg": evaluate_multiclass(rows, feature_names, "values", "logreg"),
        "multiclass_rf": evaluate_multiclass(rows, feature_names, "values", "rf"),
    }
    print(
        f"4 clases: logreg={report['reference']['features_only']['multiclass_logreg']['balanced_accuracy']:.3f} "
        f"RF={report['reference']['features_only']['multiclass_rf']['balanced_accuracy']:.3f}"
    )
    print(f"{'par':32s} {'bal_acc':>8s} {'IC95':>18s} {'p':>7s} {'RF':>7s} {'max_uni':>8s} {'call':>28s}")
    for a, b in PAIRS:
        result = evaluate_pair(
            rows, feature_names, a, b, "values", args.permutations
        )
        report["reference"]["features_only"]["pairs"].append(result)
        lo, hi = result["logreg_ci95"]
        print(
            f"{a + ' vs ' + b:32s} {result['logreg_balanced_accuracy']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  {result['permutation_p']:.3f} "
            f"{result['rf_balanced_accuracy']:.3f} {result['best_univariate_auroc']:.3f} "
            f"{result['call_at_margin']:>28s}"
        )

    # 2. Confusor temporal: fase como único predictor.
    print("\n=== Confusor temporal: solo fase del protocolo (RF) ===")
    report["reference"]["time_only"] = {
        "multiclass_rf": evaluate_multiclass(rows, ["phase"], "phase", "rf"),
        "pairs": [],
    }
    m = report["reference"]["time_only"]["multiclass_rf"]
    print(f"4 clases: bal_acc={m['balanced_accuracy']:.3f} IC95={m['ci95']}")
    for a, b in PAIRS:
        result = evaluate_pair(rows, ["phase"], a, b, "phase", 0)
        report["reference"]["time_only"]["pairs"].append(result)
        print(f"  {a + ' vs ' + b:32s} RF fase={result['rf_balanced_accuracy']:.3f}")

    # 3. Features + fase (valor incremental).
    print("\n=== Features + fase (valor incremental) ===")
    plus = feature_names + ["phase"]
    report["reference"]["features_plus_time"] = {
        "multiclass_rf": evaluate_multiclass(rows, plus, "values", "rf"),
    }
    m = report["reference"]["features_plus_time"]["multiclass_rf"]
    print(f"4 clases: bal_acc={m['balanced_accuracy']:.3f} IC95={m['ci95']}")

    # 4. Control temporal: residuales sobre fase².
    print("\n=== Control temporal: features residualizadas en fase² por sujeto ===")
    report["reference"]["time_residualized"] = {
        "pairs": [],
        "multiclass_logreg": evaluate_multiclass(rows, feature_names, "resid", "logreg"),
        "multiclass_rf": evaluate_multiclass(rows, feature_names, "resid", "rf"),
    }
    print(
        f"4 clases: logreg={report['reference']['time_residualized']['multiclass_logreg']['balanced_accuracy']:.3f} "
        f"RF={report['reference']['time_residualized']['multiclass_rf']['balanced_accuracy']:.3f}"
    )
    print(f"{'par':32s} {'bal_acc':>8s} {'IC95':>18s} {'RF':>7s} {'call':>28s}")
    for a, b in PAIRS:
        result = evaluate_pair(rows, feature_names, a, b, "resid", 0)
        report["reference"]["time_residualized"]["pairs"].append(result)
        lo, hi = result["logreg_ci95"]
        print(
            f"{a + ' vs ' + b:32s} {result['logreg_balanced_accuracy']:.3f} "
            f"[{lo:.3f}, {hi:.3f}]  {result['rf_balanced_accuracy']:.3f} "
            f"{result['call_at_margin']:>28s}"
        )

    # 5. Centrado por sujeto (personalización).
    print("\n=== Centrado por sujeto (personalización) ===")
    report["reference"]["subject_centered"] = {
        "multiclass_logreg": evaluate_multiclass(rows, feature_names, "centered", "logreg"),
    }
    m = report["reference"]["subject_centered"]["multiclass_logreg"]
    print(f"4 clases: bal_acc={m['balanced_accuracy']:.3f} IC95={m['ci95']}")

    # 6. Comparación con el LLM.
    print("\n=== Comparación LLM vs referencia (binario stress vs baseline) ===")
    if report["llm_comparison"]:
        ref = next(
            r
            for r in report["reference"]["features_only"]["pairs"]
            if r["pair"] == ["baseline", "stress"]
        )
        print(f"  Referencia features-only: bal_acc={ref['logreg_balanced_accuracy']:.3f} "
              f"IC95={ref['logreg_ci95']}")
        for rep, metrics in report["llm_comparison"].items():
            print(
                f"  LLM {rep}: bal_acc={metrics['balanced_accuracy']:.3f} "
                f"acc={metrics['accuracy']:.3f} recall={metrics['recall_stress']:.3f} "
                f"coverage={metrics['coverage']:.3f}"
            )
            metrics["reference_features_only_balanced_accuracy"] = ref["logreg_balanced_accuracy"]
            metrics["excess_cost_balanced_accuracy"] = ref["logreg_balanced_accuracy"] - metrics["balanced_accuracy"]
    else:
        print("  (no hay evaluation_report.json)")

    # 7. Decisión.
    print("\n=== Lectura para la decisión ===")
    stress_pair = next(
        r for r in report["reference"]["features_only"]["pairs"]
        if r["pair"] == ["baseline", "stress"]
    )
    amusement_pair = next(
        r for r in report["reference"]["features_only"]["pairs"]
        if r["pair"] == ["baseline", "amusement"]
    )
    print(
        f"  stress vs baseline: referencia={stress_pair['logreg_balanced_accuracy']:.3f} "
        f"({stress_pair['call_at_margin']}); LLM="
        f"{report['llm_comparison'].get('numeric_llm', {}).get('balanced_accuracy', float('nan')):.3f}"
        f"–{report['llm_comparison'].get('text_llm', {}).get('balanced_accuracy', float('nan')):.3f}"
    )
    print(
        f"  baseline vs amusement: referencia={amusement_pair['logreg_balanced_accuracy']:.3f} "
        f"({amusement_pair['call_at_margin']}) -> par más débil"
    )
    print(
        "  El LLM no usa la fase; su referencia justa es 'features_only'. "
        "La fase sola es un confusor del protocolo, no información disponible."
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReporte guardado en: {args.output}")


if __name__ == "__main__":
    main()
