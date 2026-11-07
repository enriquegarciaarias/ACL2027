"""Evaluación agregada de Fase 3 sobre el corpus completo (15 sujetos).

Calcula métricas separando:
  - Condición: C0_clean (datos reales) vs controles negativos (NC_*).
  - Representación: numeric_llm vs text_llm.

Métricas por grupo (pooled entre sujetos):
  - n, válidas, inválidas, abstenciones, decididas
  - accuracy (solo decididas) y accuracy_incl_abstain
  - tasa de abstención
  - AUROC por clase (one-vs-rest, estadística U de Mann-Whitney)
  - ECE, Brier, confianza promedio
  - matriz de confusión y distribución de decisiones

Uso:
    source .venv/bin/activate
    python scripts/evaluate_phase3.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sources.phase3.evaluator import (
    LABEL_ORDER,
    _compute_auroc_one_vs_rest,
    _compute_brier_score,
    _compute_ece,
    load_ground_truth,
    load_responses,
)

# Rutas relativas al repositorio.
RESPONSES_DIR = Path("results/output/WESAD/PHASE2/llm_responses")
GROUND_TRUTH_PATH = Path("results/output/WESAD/INGEST/groundtruth/labels.jsonl")
REPORT_PATH = Path("results/output/WESAD/PHASE2/evaluation_report.json")

CONDITIONS = ["C0_clean", "NC_shuffled", "NC_zero", "NC_nan", "NC_absent"]
REPRESENTATIONS = ["numeric_llm", "text_llm"]


def _opaque_window_id(window_id: str) -> str:
    """ID opaco (misma función que Fase 1)."""
    return "win_" + hashlib.sha256(window_id.encode()).hexdigest()[:8]


def build_matched(
    responses_dir: Path,
    ground_truth_path: Path,
) -> list[dict[str, Any]]:
    """Une todas las respuestas con el ground truth usando el ID opaco."""
    ground_truth = load_ground_truth(ground_truth_path)
    opaque_to_label: dict[str, str] = {}
    for window_id, label in ground_truth.items():
        opaque_to_label[_opaque_window_id(window_id)] = label

    matched: list[dict[str, Any]] = []
    for resp_path in sorted(responses_dir.glob("*.responses.jsonl")):
        for resp in load_responses(resp_path):
            opaque_id = resp.get("window_id_opaque", "")
            matched.append(
                {
                    "subject": resp.get("subject", ""),
                    "true_label": opaque_to_label.get(opaque_id, "UNKNOWN"),
                    "predicted": resp.get("decision", ""),
                    "confidence": resp.get("confidence", 0) or 0,
                    "status": resp.get("status", ""),
                    "representation": resp.get("representation", ""),
                    "condition_id": resp.get("condition_id", ""),
                }
            )
    return matched


def compute_group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula métricas para un grupo filtrado de respuestas (pooled)."""
    n = len(rows)
    valid = [r for r in rows if r["status"] != "invalid_output"]
    invalid = [r for r in rows if r["status"] == "invalid_output"]
    n_valid = len(valid)
    n_invalid = len(invalid)

    abstain = [r for r in valid if r["predicted"] == "abstain"]
    decided = [r for r in valid if r["predicted"] != "abstain"]
    n_abstain = len(abstain)
    n_decided = len(decided)

    correct = sum(1 for r in decided if r["predicted"] == r["true_label"])
    accuracy = correct / n_decided if n_decided > 0 else 0.0
    accuracy_incl_abstain = correct / n_valid if n_valid > 0 else 0.0
    abstention_rate = n_abstain / n_valid if n_valid > 0 else 0.0

    # AUROC (decididas): score = confianza si predice la clase, 0 en otro caso.
    y_true = [r["true_label"] for r in decided]
    y_pred = [r["predicted"] for r in decided]
    y_conf = [r["confidence"] for r in decided]
    y_probs: dict[str, list[float]] = {c: [] for c in LABEL_ORDER}
    for r in decided:
        for c in LABEL_ORDER:
            if c == r["predicted"]:
                y_probs[c].append(r["confidence"] / 100.0)
            else:
                y_probs[c].append(0.0)

    auroc: dict[str, float] = {}
    for c in LABEL_ORDER:
        auroc[c] = _compute_auroc_one_vs_rest(y_true, y_probs, c)

    brier = _compute_brier_score(y_true, y_probs, LABEL_ORDER)
    ece = _compute_ece(y_true, y_pred, [c / 100.0 for c in y_conf])

    conf_non_abstain = [r["confidence"] / 100.0 for r in decided]
    confidence_avg = (
        sum(conf_non_abstain) / len(conf_non_abstain) if conf_non_abstain else 0.0
    )

    # Matriz de confusión y distribución de decisiones.
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in valid:
        confusion[r["true_label"]][r["predicted"]] += 1
    confusion_dict = {
        tl: {pl: confusion[tl].get(pl, 0) for pl in list(LABEL_ORDER) + ["abstain"]}
        for tl in LABEL_ORDER
    }

    decisions = Counter(r["predicted"] for r in valid)

    return {
        "n": n,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_abstain": n_abstain,
        "n_decided": n_decided,
        "accuracy": accuracy,
        "accuracy_incl_abstain": accuracy_incl_abstain,
        "abstention_rate": abstention_rate,
        "auroc": auroc,
        "brier": brier,
        "ece": ece,
        "confidence_avg": confidence_avg,
        "confusion_matrix": confusion_dict,
        "decisions": dict(decisions),
    }


def main() -> None:
    if not RESPONSES_DIR.is_dir():
        raise SystemExit(f"No existe el directorio de respuestas: {RESPONSES_DIR}")
    if not GROUND_TRUTH_PATH.is_file():
        raise SystemExit(f"No existe ground truth: {GROUND_TRUTH_PATH}")

    matched = build_matched(RESPONSES_DIR, GROUND_TRUTH_PATH)
    subjects = sorted({r["subject"] for r in matched})
    print(f"Respuestas totales: {len(matched)}")
    print(f"Sujetos: {subjects}")
    print()

    # Reporte por (representación × condición).
    report: dict[str, Any] = {"subjects": subjects, "groups": {}}

    print("=" * 100)
    print("MÉTRICAS POR GRUPO (representación × condición)")
    print("=" * 100)

    for rep in REPRESENTATIONS:
        for cond in CONDITIONS:
            rows = [
                r
                for r in matched
                if r["representation"] == rep and r["condition_id"] == cond
            ]
            if not rows:
                continue
            m = compute_group_metrics(rows)
            key = f"{rep} / {cond}"
            report["groups"][key] = m

            print(f"\n--- {key} ---")
            print(
                f"  n={m['n']} válidas={m['n_valid']} inválidas={m['n_invalid']} "
                f"abstain={m['n_abstain']} decididas={m['n_decided']}"
            )
            print(
                f"  accuracy={m['accuracy']:.3f}  acc_incl_abstain={m['accuracy_incl_abstain']:.3f}  "
                f"abstention={m['abstention_rate']:.3f}"
            )
            print(
                "  AUROC: "
                + "  ".join(f"{c}={m['auroc'][c]:.3f}" for c in LABEL_ORDER)
            )
            print(
                f"  ECE={m['ece']:.3f}  Brier={m['brier']:.3f}  conf_avg={m['confidence_avg']:.3f}"
            )

    # Matrices de confusión para C0_clean (lo relevante).
    print("\n" + "=" * 100)
    print("MATRIZ DE CONFUSIÓN — C0_clean (filas=real, columnas=predicho)")
    print("=" * 100)
    for rep in REPRESENTATIONS:
        rows = [
            r
            for r in matched
            if r["representation"] == rep and r["condition_id"] == "C0_clean"
        ]
        if not rows:
            continue
        m = compute_group_metrics(rows)
        print(f"\n  {rep}:")
        header = "    " + "".join(f"{c:>12}" for c in list(LABEL_ORDER) + ["abstain"])
        print(header)
        for tl in LABEL_ORDER:
            cells = "".join(
                f"{m['confusion_matrix'][tl].get(pl, 0):>12}"
                for pl in list(LABEL_ORDER) + ["abstain"]
            )
            print(f"    {tl:>10}{cells}")

    # Distribución de decisiones en controles negativos.
    print("\n" + "=" * 100)
    print("CONTROLES NEGATIVOS — distribución de decisiones (válidas)")
    print("=" * 100)
    for rep in REPRESENTATIONS:
        print(f"\n  {rep}:")
        for cond in CONDITIONS:
            if cond == "C0_clean":
                continue
            rows = [
                r
                for r in matched
                if r["representation"] == rep and r["condition_id"] == cond
            ]
            if not rows:
                continue
            m = compute_group_metrics(rows)
            decisions = ", ".join(f"{k}={v}" for k, v in sorted(m["decisions"].items()))
            print(
                f"    {cond:12s} abstention={m['abstention_rate']:.3f}  ->  {decisions}"
            )

    # Guardar reporte JSON.
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReporte JSON guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
