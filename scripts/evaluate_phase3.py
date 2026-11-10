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
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Permite ejecutar el script directamente (`python scripts/evaluate_phase3.py`)
# sin depender de PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    raw_decisions: bool = False,
) -> list[dict[str, Any]]:
    """Une todas las respuestas con el ground truth usando el ID opaco.

    Args:
        responses_dir: directorio con ``*.responses.jsonl``.
        ground_truth_path: ruta a ``labels.jsonl``.
        raw_decisions: si True evalúa ``decision_llm`` (salida cruda del modelo,
            antes de la compuerta de consistencia). Si False (default) evalúa
            ``decision``, que es la salida final del sistema.
    """
    ground_truth = load_ground_truth(ground_truth_path)
    opaque_to_label: dict[str, str] = {}
    for window_id, label in ground_truth.items():
        opaque_to_label[_opaque_window_id(window_id)] = label

    matched: list[dict[str, Any]] = []
    for resp_path in sorted(responses_dir.glob("*.responses.jsonl")):
        for resp in load_responses(resp_path):
            opaque_id = resp.get("window_id_opaque", "")
            decision = resp.get("decision", "")
            confidence = resp.get("confidence", 0) or 0
            status = resp.get("status", "")
            if raw_decisions and "decision_llm" in resp:
                decision = resp.get("decision_llm") or ""
                confidence = resp.get("confidence_llm", 0) or 0
                # La compuerta reetiqueta el estado; para el modelo crudo se
                # recupera el estado de validación previo.
                if status == "gate_abstention":
                    status = (
                        "valid_abstention"
                        if decision == "abstain"
                        else "valid_response"
                    )
            matched.append(
                {
                    "subject": resp.get("subject", ""),
                    "true_label": opaque_to_label.get(opaque_id, "UNKNOWN"),
                    "predicted": decision,
                    "confidence": confidence,
                    "status": status,
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


def parse_args(argv: list[str] | None = None):
    """Argumentos de línea de comandos del evaluador."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluación agregada de Fase 3 (todas las respuestas por defecto)."
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Lista de sujetos separada por comas (p. ej. S3 o S2,S3). "
        "Por defecto evalúa todos los que tengan respuestas.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Ruta alternativa del reporte JSON. Por defecto "
        "evaluation_report.json (o evaluation_report_<sujetos>.json si se filtra).",
    )
    parser.add_argument(
        "--raw-decisions",
        action="store_true",
        help="Evaluar decision_llm (modelo crudo, sin compuerta de consistencia) "
        "en lugar de la decisión final del sistema.",
    )
    return parser.parse_args(argv)


def compute_binary_stress_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Métricas para el problema binario stress vs baseline (C0_clean).

    El LLM nunca predice ``amusement`` ni ``meditation`` en el corpus actual;
    esta vista mide la separabilidad real que el modelo usa. Las predicciones
    de otra clase se excluyen del cálculo y se reportan aparte.
    """
    subset = [r for r in rows if r["true_label"] in ("stress", "baseline")]
    valid = [r for r in subset if r["status"] != "invalid_output"]
    abstain = [r for r in valid if r["predicted"] == "abstain"]
    other = [r for r in valid if r["predicted"] in ("amusement", "meditation")]
    decided = [r for r in valid if r["predicted"] in ("stress", "baseline")]

    tp = sum(1 for r in decided if r["true_label"] == "stress" and r["predicted"] == "stress")
    fn = sum(1 for r in decided if r["true_label"] == "stress" and r["predicted"] == "baseline")
    tn = sum(1 for r in decided if r["true_label"] == "baseline" and r["predicted"] == "baseline")
    fp = sum(1 for r in decided if r["true_label"] == "baseline" and r["predicted"] == "stress")

    n = len(decided)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_subset": len(subset),
        "n_decided": n,
        "n_abstain": len(abstain),
        "n_other_class": len(other),
        "coverage": n / len(subset) if subset else 0.0,
        "abstention_rate": len(abstain) / len(valid) if valid else 0.0,
        "accuracy": accuracy,
        "precision_stress": precision,
        "recall_stress": recall,
        "f1_stress": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not RESPONSES_DIR.is_dir():
        raise SystemExit(f"No existe el directorio de respuestas: {RESPONSES_DIR}")
    if not GROUND_TRUTH_PATH.is_file():
        raise SystemExit(f"No existe ground truth: {GROUND_TRUTH_PATH}")

    wanted = [s.strip() for s in str(args.subjects).split(",") if s.strip()]

    matched = build_matched(
        RESPONSES_DIR, GROUND_TRUTH_PATH, raw_decisions=args.raw_decisions
    )
    if wanted:
        unknown = sorted(set(wanted) - {r["subject"] for r in matched})
        if unknown:
            raise SystemExit(f"Sujetos sin respuestas: {unknown}")
        matched = [r for r in matched if r["subject"] in set(wanted)]

    subjects = sorted({r["subject"] for r in matched})
    print(f"Respuestas totales: {len(matched)}")
    print(f"Sujetos: {subjects}")
    print()

    if args.output:
        report_path = Path(args.output)
    elif wanted:
        report_path = REPORT_PATH.with_name(
            "evaluation_report_" + "-".join(subjects) + ".json"
        )
    else:
        report_path = REPORT_PATH

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

    # Binario stress vs baseline en C0_clean (el modelo no predice
    # amusement/meditation en este corpus).
    print("\n" + "=" * 100)
    print("BINARIO stress vs baseline — C0_clean (decididas stress/baseline)")
    print("=" * 100)
    report["binary_stress_vs_baseline"] = {}
    for rep in REPRESENTATIONS:
        rows = [
            r
            for r in matched
            if r["representation"] == rep and r["condition_id"] == "C0_clean"
        ]
        if not rows:
            continue
        b = compute_binary_stress_metrics(rows)
        report["binary_stress_vs_baseline"][rep] = b
        print(
            f"  {rep}: acc={b['accuracy']:.3f}  precision={b['precision_stress']:.3f}  "
            f"recall={b['recall_stress']:.3f}  F1={b['f1_stress']:.3f}  "
            f"coverage={b['coverage']:.3f}  abstention={b['abstention_rate']:.3f}  "
            f"(n={b['n_decided']}, otras clases={b['n_other_class']})"
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nReporte JSON guardado en: {report_path}")


if __name__ == "__main__":
    main()
