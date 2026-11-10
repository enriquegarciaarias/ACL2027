#!/usr/bin/env python
"""Análisis de calibración y predicción selectiva del endpoint binario (P4).

Sin llamadas al LLM. Mide:

1. Calibración de la confianza del modelo (fiabilidad, ECE, MCE, Brier,
   log-loss) y su **diversidad** (cuántos valores distintos emite).
2. Escalado de temperatura ajustado con validación cruzada agrupada por sujeto.
3. Predicción selectiva: curva riesgo-cobertura y umbral por control conformal
   del riesgo (garantía marginal sobre sujetos).

Uso:
    python scripts/analyze_calibration.py
    python scripts/analyze_calibration.py --alphas 0.05,0.1,0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.ingestion.represent import _opaque_window_id
from sources.phase3.calibration import (
    apply_temperature,
    brier_score,
    conformal_risk_threshold,
    evaluate_selective,
    expected_calibration_error,
    fit_temperature,
    group_kfold_indices,
    log_loss,
    maximum_calibration_error,
    reliability_table,
    risk_coverage_curve,
    wilson_interval,
)
from sources.phase3.evaluator import _compute_auroc_one_vs_rest

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
E12_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "e12"
DEFAULT_OUTPUT = E12_DIR / "calibration_report.json"

# Reinterpretación del prompt de 4 clases como binario.
FOUR_CLASS_MAP = {
    "baseline": "non-stress",
    "amusement": "non-stress",
    "meditation": "non-stress",
    "stress": "stress",
}


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in (INGEST_DIR / "groundtruth" / "labels.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        labels[_opaque_window_id(record["window_id"])] = label
    return labels


def load_rows(path: Path, decision_map: dict[str, str] | None = None):
    rows = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("condition_id") != "C0_clean":
            continue
        decision = record.get("decision") or ""
        if decision not in ("stress", "non-stress", "abstain"):
            decision = (decision_map or {}).get(decision, decision)
        if decision not in ("stress", "non-stress"):
            continue
        confidence = float(record.get("confidence") or 0) / 100.0
        p_stress = confidence if decision == "stress" else 1.0 - confidence
        rows.append(
            {
                "subject": record.get("subject", ""),
                "window_id_opaque": record.get("window_id_opaque", ""),
                "decision": decision,
                "p_stress": p_stress,
                "confidence_pred": max(p_stress, 1.0 - p_stress),
                "status": record.get("status", ""),
            }
        )
    return rows


def build_arrays(
    rows: list[dict[str, Any]], labels: dict[str, str]
) -> tuple[list[int], list[float], list[bool], list[str], list[float]]:
    y_true: list[int] = []
    p_stress: list[float] = []
    correct: list[bool] = []
    groups: list[str] = []
    conf: list[float] = []
    for row in rows:
        label = labels.get(row["window_id_opaque"], "UNKNOWN")
        if label == "UNKNOWN":
            continue
        truth = 1 if label == "stress" else 0
        y_true.append(truth)
        p_stress.append(row["p_stress"])
        correct.append((row["decision"] == "stress") == (truth == 1))
        groups.append(row["subject"])
        conf.append(row["confidence_pred"])
    return y_true, p_stress, correct, groups, conf


def analyze(name: str, rows: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    y_true, p_stress, correct, groups, conf = build_arrays(rows, labels)
    y = np.asarray(y_true)
    p = np.asarray(p_stress)
    ok = np.asarray(correct, dtype=bool)
    g = np.asarray(groups)
    c = np.asarray(conf)

    if y.size == 0:
        return {"name": name, "n": 0}

    report: dict[str, Any] = {
        "name": name,
        "n": int(y.size),
        "n_subjects": int(len(set(groups))),
        "base_rate_stress": float(np.mean(y)),
        "confidence_values": {
            str(k): int(v) for k, v in Counter(np.round(c, 4)).most_common()
        },
        "n_distinct_confidence": int(len(set(np.round(c, 4)))),
        "accuracy": float(np.mean(ok)),
        "auroc_stress": _compute_auroc_one_vs_rest(
            [str(int(v)) for v in y],
            {"1": list(p), "0": list(1.0 - p)},
            "1",
        ),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "ece_10": expected_calibration_error(y, p, 10),
        "ece_5": expected_calibration_error(y, p, 5),
        "mce_10": maximum_calibration_error(y, p, 10),
        "reliability_10": reliability_table(y, p, 10),
        "risk_coverage": risk_coverage_curve(c, ok),
    }

    # Escalado de temperatura con validación cruzada agrupada por sujeto.
    cv = []
    for train_idx, test_idx in group_kfold_indices(groups, n_splits=5):
        fit = fit_temperature([p[i] for i in train_idx], [y[i] for i in train_idx])
        p_test = np.asarray(p)[test_idx]
        y_test = np.asarray(y)[test_idx]
        p_cal = apply_temperature(p_test, fit["temperature"])
        cv.append(
            {
                "n_test": int(len(test_idx)),
                "temperature": fit["temperature"],
                "log_loss_before": log_loss(y_test, p_test),
                "log_loss_after": log_loss(y_test, p_cal),
                "brier_before": brier_score(y_test, p_test),
                "brier_after": brier_score(y_test, p_cal),
                "ece_before": expected_calibration_error(y_test, p_test, 10),
                "ece_after": expected_calibration_error(y_test, p_cal, 10),
            }
        )
    report["temperature_cv"] = cv
    if cv:
        report["temperature_cv_mean"] = {
            key: float(np.nanmean([entry[key] for entry in cv]))
            for key in (
                "temperature",
                "log_loss_before",
                "log_loss_after",
                "brier_before",
                "brier_after",
                "ece_before",
                "ece_after",
            )
        }

    # Control conformal del riesgo con validación cruzada agrupada por sujeto.
    alphas = [0.05, 0.10, 0.20]
    conformal = {f"alpha_{alpha:g}": [] for alpha in alphas}
    for train_idx, test_idx in group_kfold_indices(groups, n_splits=5):
        conf_train = [c[i] for i in train_idx]
        ok_train = [ok[i] for i in train_idx]
        conf_test = [c[i] for i in test_idx]
        ok_test = [ok[i] for i in test_idx]
        for alpha in alphas:
            fit = conformal_risk_threshold(conf_train, ok_train, alpha=alpha)
            test_metrics = evaluate_selective(conf_test, ok_test, fit["threshold"])
            if test_metrics["n_accepted"]:
                successes = int(
                    round(
                        test_metrics["selective_accuracy"]
                        * test_metrics["n_accepted"]
                    )
                )
                low, high = wilson_interval(successes, test_metrics["n_accepted"])
            else:
                low, high = (float("nan"), float("nan"))
            conformal[f"alpha_{alpha:g}"].append(
                {
                    "threshold": fit["threshold"],
                    "calibration_coverage": fit["coverage"],
                    "test_coverage": test_metrics["coverage"],
                    "test_selective_accuracy": test_metrics["selective_accuracy"],
                    "test_selective_risk": test_metrics["selective_risk"],
                    "test_n_accepted": test_metrics["n_accepted"],
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    report["conformal_cv"] = conformal
    report["conformal_cv_mean"] = {
        key: {
            field: float(np.nanmean([e[field] for e in entries if e[field] is not None]))
            for field in (
                "calibration_coverage",
                "test_coverage",
                "test_selective_accuracy",
                "test_selective_risk",
            )
        }
        for key, entries in conformal.items()
        if entries
    }

    # Referencia in-sample (sin partición) para el umbral conformal.
    report["conformal_insample"] = {
        f"alpha_{alpha:g}": conformal_risk_threshold(conf, ok, alpha=alpha)
        for alpha in alphas
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary-responses", default=str(E12_DIR / "all.numeric_centered_binary.responses.jsonl"))
    parser.add_argument(
        "--graded-responses",
        default=str(E12_DIR / "all.numeric_centered_binary_graded.responses.jsonl"),
        help="Variante con rúbrica de confianza graduada (P4).",
    )
    parser.add_argument("--four-class-responses", default=str(E12_DIR / "all.numeric_centered.responses.jsonl"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    labels = load_labels()
    report: dict[str, Any] = {"systems": {}}

    binary_path = Path(args.binary_responses)
    if binary_path.is_file():
        rows = load_rows(binary_path)
        report["systems"]["binary_prompt_e12"] = analyze(
            "binary_prompt_e12", rows, labels
        )
    else:
        print(f"[calibración] no existe {binary_path}")

    graded_path = Path(args.graded_responses)
    if graded_path.is_file():
        rows = load_rows(graded_path)
        report["systems"]["binary_prompt_graded"] = analyze(
            "binary_prompt_graded", rows, labels
        )

    four_path = Path(args.four_class_responses)
    if four_path.is_file():
        rows = load_rows(four_path, decision_map=FOUR_CLASS_MAP)
        report["systems"]["four_class_prompt_mapped"] = analyze(
            "four_class_prompt_mapped", rows, labels
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("=== Calibración (C0_clean) ===")
    header = (
        f"{'sistema':28s} {'n':>4s} {'conf. distintas':>15s} {'acc':>6s} "
        f"{'ECE10':>6s} {'MCE10':>6s} {'Brier':>6s} {'logloss':>8s} {'AUROC':>6s}"
    )
    print(header)
    for name, entry in report["systems"].items():
        if not entry.get("n"):
            continue
        print(
            f"{name:28s} {entry['n']:4d} {entry['n_distinct_confidence']:15d} "
            f"{entry['accuracy']:6.3f} {entry['ece_10']:6.3f} "
            f"{entry['mce_10']:6.3f} {entry['brier']:6.3f} "
            f"{entry['log_loss']:8.3f} {entry['auroc_stress']:6.3f}"
        )
        print(f"    valores de confianza: {entry['confidence_values']}")
        mean = entry.get("temperature_cv_mean")
        if mean:
            print(
                f"    temperatura CV: T={mean['temperature']:.2f} | "
                f"logloss {mean['log_loss_before']:.3f}->{mean['log_loss_after']:.3f} | "
                f"Brier {mean['brier_before']:.3f}->{mean['brier_after']:.3f} | "
                f"ECE {mean['ece_before']:.3f}->{mean['ece_after']:.3f}"
            )
        for alpha_key, stats in entry.get("conformal_cv_mean", {}).items():
            print(
                f"    conformal {alpha_key}: cobertura={stats['test_coverage']:.3f} "
                f"precisión selectiva={stats['test_selective_accuracy']:.3f} "
                f"riesgo={stats['test_selective_risk']:.3f}"
            )
    print(f"\nReporte: {output_path}")


if __name__ == "__main__":
    main()
