#!/usr/bin/env python
"""Análisis del barrido de degradación (H1, H2, H5).

Lee las respuestas de ``results/output/WESAD/PHASE2/degradation/`` y calcula,
por condición:

- Corrección (accuracy, balanced accuracy) y abstención.
- Calibración (ECE, Brier) y confianza media.
- **Respaldo de afirmaciones** (H5): proporciones de claims respaldados,
  contradichos, no respaldados e indeterminados, y acierto de dirección.
- **Certeza lingüística** (H2b): hedges, boosters e índice por 100 palabras.

Sin llamadas al LLM; se puede ejecutar con el barrido en marcha (ignora líneas
incompletas).

Uso:
    python scripts/analyze_degradation.py
    python scripts/analyze_degradation.py --conditions C0_clean,C4_total_eda
"""

from __future__ import annotations

import argparse
import warnings
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# sklearn avisa de columnas sin valores observados (p. ej. SCR en C4); es esperado.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from sources.ingestion.represent import _opaque_window_id
from sources.phase2.llm_config import model_slug
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.claims import summarize_verdicts, verify_claims
from sources.phase3.records import evaluate_binary_records, evaluate_records
from sources.phase3.recovery import apply_recovery

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
DEGRADATION_DIR = (
    REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "degradation"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "degradation_report.json"
)
DEFAULT_CONDITIONS = (
    "C0_clean",
    "C1_partial_resp",
    "C2_partial_eda",
    "C4_total_eda",
    "C5_total_both",
)


def load_labels() -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in (INGEST_DIR / "groundtruth" / "labels.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        labels[record["window_id"]] = label
    return labels


def load_records(
    degradation_dir: Path,
    conditions: set[str],
    split: str = "evaluation",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(degradation_dir.glob("*.responses.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # línea incompleta si el barrido está escribiendo
            if record.get("condition_id") not in conditions:
                continue
            if split and record.get("split") != split:
                continue
            records.append(record)
    return records


def _certainty_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen de certeza lingüística de un conjunto de respuestas."""
    if not items:
        return {
            "n": 0,
            "hedges_mean": float("nan"),
            "boosters_mean": float("nan"),
            "certainty_index_mean": float("nan"),
            "words_mean": float("nan"),
        }
    return {
        "n": len(items),
        "hedges_mean": float(np.mean([c["hedges"] for c in items])),
        "boosters_mean": float(np.mean([c["boosters"] for c in items])),
        "certainty_index_mean": float(np.mean([c["certainty_index"] for c in items])),
        "words_mean": float(np.mean([c["n_words"] for c in items])),
    }


def _claims_and_certainty(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Respaldo de claims (H5) y certeza lingüística (H2b) de un grupo."""
    verdicts: list[dict[str, Any]] = []
    certainties: list[dict[str, Any]] = []
    certainties_decided: list[dict[str, Any]] = []
    certainties_abstain: list[dict[str, Any]] = []
    n_claims = 0
    for record in records:
        claims = record.get("claims") or []
        n_claims += len(claims)
        if claims:
            verdicts.extend(verify_claims(claims, record.get("prompt", "")))
        certainty = linguistic_certainty(record.get("reasoning", "") or "")
        if certainty["n_words"] > 0:
            certainties.append(certainty)
            if (record.get("decision") or "") == "abstain":
                certainties_abstain.append(certainty)
            else:
                certainties_decided.append(certainty)

    claims_block = {**summarize_verdicts(verdicts), "n_claims_total": n_claims}
    certainty_block = {
        **_certainty_stats(certainties),
        # La certeza de una abstención habla de la evidencia («the channel is
        # clearly missing»), no del afecto: se separa para no confundir H2b con
        # la redacción del veto.
        "decided": _certainty_stats(certainties_decided),
        "abstain": _certainty_stats(certainties_abstain),
    }
    return claims_block, certainty_block


def _confidence_auroc(
    records: list[dict[str, Any]], labels: dict[str, str], binary: bool
) -> float:
    """AUROC de la confianza prediciendo corrección (H2a)."""
    from sources.phase3.evaluator import _compute_auroc_one_vs_rest

    y_true: list[str] = []
    scores: list[float] = []
    for record in records:
        decision = record.get("decision")
        valid = ("stress", "non-stress") if binary else (
            "baseline", "stress", "amusement", "meditation"
        )
        if decision not in valid:
            continue
        truth = labels.get(record["window_id"], "UNKNOWN")
        if truth == "UNKNOWN":
            continue
        correct = (
            (decision == "stress") == (truth == "stress")
            if binary
            else decision == truth
        )
        y_true.append("correct" if correct else "incorrect")
        scores.append(float(record.get("confidence") or 0) / 100.0)
    if not y_true or len(set(y_true)) < 2:
        return float("nan")
    return _compute_auroc_one_vs_rest(
        y_true,
        {"correct": scores, "incorrect": [1.0 - s for s in scores]},
        "correct",
    )


def analyze_group_binary(
    records: list[dict[str, Any]], labels: dict[str, str]
) -> dict[str, Any]:
    """Métricas de un grupo cuando el prompt es binario stress/non-stress.

    Las decisiones ya son ``stress`` / ``non-stress`` / ``abstain``: no hay
    mapeo desde 4 clases. El resto de bloques (claims, certeza, confianza,
    indicadores) se calcula igual que en el modo afectivo.
    """
    if not records:
        return {"n": 0}

    records = [
        {
            **record,
            "window_id_opaque": record.get("window_id_opaque")
            or _opaque_window_id(record["window_id"]),
        }
        for record in records
    ]
    label_by_opaque = {
        record["window_id_opaque"]: labels.get(record["window_id"], "UNKNOWN")
        for record in records
    }
    metrics = evaluate_binary_records(records, label_by_opaque)
    claims_block, certainty_block = _claims_and_certainty(records)
    confidences = [
        float(record["confidence"])
        for record in records
        if record.get("confidence") is not None
    ]

    return {
        "n": metrics["n"],
        "n_valid": metrics["n_valid"],
        "n_invalid": metrics["n_invalid"],
        "n_abstain": metrics["n_abstain"],
        "n_decided": metrics["n_decided"],
        "coverage": metrics["coverage"],
        "abstention_rate": metrics["abstention_rate"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_stress": metrics["precision_stress"],
        "recall_stress": metrics["recall_stress"],
        "specificity": metrics["specificity"],
        "ece": metrics["ece"],
        "brier": metrics["brier"],
        "auroc": metrics.get("auroc_stress", float("nan")),
        "f1_stress": metrics.get("f1_stress", float("nan")),
        "decisions": metrics.get("decisions", {}),
        "confidence_mean": float(np.mean(confidences)) if confidences else float("nan"),
        "confidence_std": float(np.std(confidences)) if confidences else float("nan"),
        "n_distinct_confidence": len({round(c, 2) for c in confidences}),
        "confidence_auroc_correctness": _confidence_auroc(records, labels, binary=True),
        "claims": claims_block,
        "certainty": certainty_block,
        # Sin desglose por clase: el endpoint binario ya lo resume la tabla propia.
        "per_class": {},
        "binary_mapped": metrics,
    }


def analyze_group(records: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    """Métricas de corrección, calibración, claims y certeza de un grupo."""
    if not records:
        return {"n": 0}

    records = [
        {
            **record,
            "window_id_opaque": record.get("window_id_opaque")
            or _opaque_window_id(record["window_id"]),
        }
        for record in records
    ]
    label_by_opaque = {
        record["window_id_opaque"]: labels.get(record["window_id"], "UNKNOWN")
        for record in records
    }
    metrics = evaluate_records(records, label_by_opaque)

    claims_block, certainty_block = _claims_and_certainty(records)
    confidence_auroc = _confidence_auroc(records, labels, binary=False)

    confidences = [
        float(record["confidence"])
        for record in records
        if record.get("confidence") is not None
    ]

    # Desglose por clase real: ¿el modelo llega a predecir las cuatro?
    per_class: dict[str, dict[str, Any]] = {}
    for record in records:
        truth = label_by_opaque.get(record["window_id_opaque"], "UNKNOWN")
        if truth == "UNKNOWN":
            continue
        bucket = per_class.setdefault(
            truth, {"n": 0, "decided": 0, "correct": 0, "predictions": {}}
        )
        bucket["n"] += 1
        decision = record.get("decision")
        if not decision or decision == "abstain":
            continue
        bucket["decided"] += 1
        bucket["predictions"][decision] = bucket["predictions"].get(decision, 0) + 1
        if decision == truth:
            bucket["correct"] += 1
    for bucket in per_class.values():
        bucket["recall"] = bucket["correct"] / bucket["n"] if bucket["n"] else 0.0
        bucket["recall_decided"] = (
            bucket["correct"] / bucket["decided"] if bucket["decided"] else 0.0
        )

    # Endpoint primario en modo binario: se reinterpreta la decisión de 4 clases
    # (baseline, amusement y meditation → non-stress). Mide al LLM con la misma
    # vara que la referencia binaria, sin gastar llamadas.
    binary_mapped = evaluate_binary_records(
        records,
        label_by_opaque,
        other_decisions_map={
            "baseline": "non-stress",
            "amusement": "non-stress",
            "meditation": "non-stress",
        },
    )
    return {
        "n": metrics["n"],
        "n_valid": metrics["n_valid"],
        "n_invalid": metrics["n_invalid"],
        "n_abstain": metrics["n_abstain"],
        "abstention_rate": metrics["abstention_rate"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "ece": metrics["ece"],
        "brier": metrics["brier"],
        "auroc": metrics["auroc"],
        "decisions": metrics["decisions"],
        "confidence_mean": float(np.mean(confidences)) if confidences else float("nan"),
        "confidence_std": float(np.std(confidences)) if confidences else float("nan"),
        "n_distinct_confidence": len({round(c, 2) for c in confidences}),
        "confidence_auroc_correctness": confidence_auroc,
        "per_class": per_class,
        "binary_mapped": binary_mapped,
        "claims": claims_block,
        "certainty": certainty_block,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--model",
        default="",
        help="Modelo a analizar (p. ej. 'qwen3:14b'). Lee de "
        "degradation{,_text}/{modelo}/; vacío = carpeta histórica.",
    )
    parser.add_argument(
        "--representation",
        choices=["numeric", "text"],
        default="numeric",
        help="Brazo de representación del subdirectorio por modelo.",
    )
    parser.add_argument(
        "--degradation-dir",
        default="",
        help="Directorio explícito (tiene prioridad sobre --model).",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--no-recovery",
        action="store_true",
        help="No recuperar respuestas inválidas solo por claims mal formados.",
    )
    parser.add_argument(
        "--compare-with",
        default="",
        help="Directorio del otro brazo de representación (p. ej. degradation_text) "
        "para el contraste H3.",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["affective", "binary"],
        default="affective",
        help="Tarea con la que se generaron las respuestas: 4 clases o binario "
        "stress/non-stress. Debe coincidir con el barrido (--decision-mode).",
    )
    parser.add_argument(
        "--split",
        default="evaluation",
        choices=["evaluation", "development", "all"],
        help="Conjunto a analizar (por defecto, evaluación).",
    )
    args = parser.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    split = "" if args.split == "all" else args.split
    labels = load_labels()

    binary = args.decision_mode == "binary"
    analyze_fn = analyze_group_binary if binary else analyze_group

    if args.degradation_dir:
        degradation_dir = Path(args.degradation_dir)
    else:
        name = "degradation" if args.representation == "numeric" else "degradation_text"
        if binary:
            name += "_binary"
        base = DEGRADATION_DIR.parent / name
        degradation_dir = base / model_slug(args.model) if args.model else base
        if args.model and not degradation_dir.is_dir() and base.is_dir():
            # Ejecuciones anteriores al aislamiento por modelo: se leen de la
            # carpeta histórica, avisando de a quién pertenecen.
            print(
                f"[AVISO] No existe {degradation_dir}; se usa la carpeta "
                f"histórica {base}."
            )
            degradation_dir = base
    if not degradation_dir.is_dir():
        raise SystemExit(f"No existe el directorio de respuestas: {degradation_dir}")
    records = load_records(degradation_dir, set(conditions), split)
    n_loaded = len(records)
    models_found = {
        record.get("model") or (record.get("ollama_meta") or {}).get("model")
        for record in records
    }
    models_found.discard(None)
    if models_found:
        print(f"Modelos en los datos: {sorted(models_found)}")
    if args.model and models_found and args.model not in models_found:
        raise SystemExit(
            f"Los datos de {degradation_dir} pertenecen a {sorted(models_found)}, "
            f"no a {args.model}. Ejecuta primero el barrido con --model {args.model}."
        )
    if not args.no_recovery:
        records, n_recovered = apply_recovery(records)
        dropped = sum(int(r.get("dropped_claims") or 0) for r in records if r.get("recovered"))
        print(
            f"Respuestas cargadas: {n_loaded} (split={args.split}); "
            f"recuperadas: {n_recovered}; claims descartados en recuperadas: {dropped}"
        )
    else:
        print(f"Respuestas cargadas: {n_loaded} (split={args.split})")

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[record["condition_id"]].append(record)

    report = {
        "model": args.model or "legacy",
        "degradation_dir": str(degradation_dir),
        "representation": args.representation if args.model else "legacy",
        "decision_mode": args.decision_mode,
        "split": args.split,
        "n_records": len(records),
        "recovery": not args.no_recovery,
        "conditions": {},
    }
    for condition in conditions:
        report["conditions"][condition] = analyze_fn(
            by_condition.get(condition, []), labels
        )

    if not records:
        raise SystemExit(
            f"No hay respuestas en {degradation_dir} (split={args.split})."
        )
    subjects = sorted({record["subject"] for record in records})
    report["subjects"] = subjects
    report["reference_by_condition"] = reference_by_condition(
        conditions, subjects, labels
    )
    gate_scores = gate_scores_by_condition(conditions, subjects, by_condition)
    sqi_scores = sqi_by_condition(conditions, subjects)
    report["indicator_incremental"] = {
        "gate": {
            condition: incremental_quality_auroc(
                by_condition.get(condition, []), labels, gate_scores.get(condition, {})
            )
            for condition in conditions
        },
        "sqi": {
            condition: incremental_quality_auroc(
                by_condition.get(condition, []), labels, sqi_scores.get(condition, {})
            )
            for condition in conditions
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\nTarea: {args.decision_mode} | representación: {args.representation}")
    print("\n=== Degradación: corrección, abstención y calibración ===")
    header = (
        f"{'condición':16s} {'n':>5s} {'abst':>6s} {'acc':>6s} {'bal':>6s} "
        f"{'ECE':>6s} {'conf':>6s} {'#conf':>6s}"
    )
    print(header)
    for condition in conditions:
        entry = report["conditions"][condition]
        if not entry.get("n"):
            print(f"{condition:16s} (sin datos)")
            continue
        print(
            f"{condition:16s} {entry['n']:5d} {entry['abstention_rate']:6.3f} "
            f"{entry['accuracy']:6.3f} {entry['balanced_accuracy']:6.3f} "
            f"{entry['ece']:6.3f} {entry['confidence_mean']:6.1f} "
            f"{entry['n_distinct_confidence']:6d}"
        )

    print("\n=== Respaldo de afirmaciones (H5) ===")
    print(
        f"{'condición':16s} {'claims':>7s} {'suport':>7s} {'contra':>7s} "
        f"{'unsup':>7s} {'indet':>7s} {'dir_acc':>8s}"
    )
    for condition in conditions:
        entry = report["conditions"][condition]
        claims = entry.get("claims") or {}
        if not claims.get("n_claims"):
            print(f"{condition:16s} (sin claims)")
            continue
        print(
            f"{condition:16s} {claims['n_claims']:7d} "
            f"{claims['supported_rate']:7.3f} {claims['contradicted_rate']:7.3f} "
            f"{claims['unsupported_rate']:7.3f} {claims['indeterminate_rate']:7.3f} "
            f"{claims['direction_accuracy']:8.3f}"
        )

    print("\n=== Certeza lingüística (H2b) ===")
    print(
        f"{'condición':16s} {'hedges':>7s} {'boost':>7s} {'índice':>8s} "
        f"{'palabras':>9s} {'índ_dec':>8s} {'n_dec':>6s} {'índ_abs':>8s} {'n_abs':>6s}"
    )
    for condition in conditions:
        entry = report["conditions"][condition]
        certainty = entry.get("certainty") or {}
        if not certainty.get("n"):
            continue
        decided = certainty.get("decided") or {}
        abstain = certainty.get("abstain") or {}
        print(
            f"{condition:16s} {certainty['hedges_mean']:7.2f} "
            f"{certainty['boosters_mean']:7.2f} {certainty['certainty_index_mean']:8.2f} "
            f"{certainty['words_mean']:9.1f} "
            f"{decided.get('certainty_index_mean', float('nan')):8.2f} "
            f"{decided.get('n', 0):6d} "
            f"{abstain.get('certainty_index_mean', float('nan')):8.2f} "
            f"{abstain.get('n', 0):6d}"
        )
    print(
        "  índice: global. índ_dec / índ_abs: solo decididas / solo abstenciones "
        "(una abstención habla de la evidencia, no del afecto)."
    )

    print("\n=== H1/H4: resolubilidad (referencia features-only) y abstención ===")
    print(
        f"{'condición':16s} {'ref_4c':>7s} {'ref_bin':>8s} {'resoluble':>10s} "
        f"{'LLM_acc':>8s} {'LLM_abst':>9s}"
    )
    for condition in conditions:
        ref = report["reference_by_condition"].get(condition, {})
        entry = report["conditions"][condition]
        if not ref or not entry.get("n"):
            continue
        print(
            f"{condition:16s} {ref['balanced_accuracy_4class']:7.3f} "
            f"{ref['balanced_accuracy_binary']:8.3f} {str(ref['resolvable']):>10s} "
            f"{entry['accuracy']:8.3f} {entry['abstention_rate']:9.3f}"
        )

    if not binary:
        print("\n=== Por clase real: recall (todas / decididas) ===")
    classes = ("baseline", "stress", "amusement", "meditation")
    print(f"{'condición':16s} " + " ".join(f"{c[:7]:>13s}" for c in classes))
    for condition in conditions:
        entry = report["conditions"][condition]
        if binary or not entry.get("n"):
            continue
        cells = []
        for class_name in classes:
            bucket = (entry.get("per_class") or {}).get(class_name)
            if not bucket:
                cells.append(f"{'-':>13s}")
                continue
            cells.append(
                f"{bucket['recall']:5.3f}/{bucket['recall_decided']:5.3f}".rjust(13)
            )
        print(f"{condition:16s} " + " ".join(cells))
    print(
        "  baseline/amusement/meditation: si el modelo nunca las predice, su "
        "recall es 0 aunque la accuracy de 4 clases parezca alta."
    )

    print(
        "\n=== Endpoint binario (prompt binario) ==="
        if binary
        else "\n=== Endpoint binario (4 clases mapeadas a stress / non-stress) ==="
    )
    print(
        f"{'condición':16s} {'cob':>6s} {'abst':>6s} {'acc':>6s} {'bal':>6s} "
        f"{'prec':>6s} {'rec':>6s} {'ref_bin':>8s}"
    )
    for condition in conditions:
        entry = report["conditions"][condition]
        binary_entry = entry.get("binary_mapped") or {}
        ref = report["reference_by_condition"].get(condition, {})
        if not binary_entry.get("n"):
            continue
        print(
            f"{condition:16s} {binary_entry['coverage']:6.3f} "
            f"{binary_entry['abstention_rate']:6.3f} {binary_entry['accuracy']:6.3f} "
            f"{binary_entry['balanced_accuracy']:6.3f} "
            f"{binary_entry['precision_stress']:6.3f} {binary_entry['recall_stress']:6.3f} "
            f"{ref.get('balanced_accuracy_binary', float('nan')):8.3f}"
        )
    print(
        "  acc/bal/prec/rec se miden sobre las decididas ('cob' = cobertura). "
        "La columna ref_bin es la referencia features-only."
    )

    print("\n=== H2a: confianza como predictor de corrección ===")
    print(f"{'condición':16s} {'AUROC_conf':>10s} {'conf':>6s} {'acc':>6s} {'sobreconf':>10s}")
    for condition in conditions:
        entry = report["conditions"][condition]
        if not entry.get("n"):
            continue
        print(
            f"{condition:16s} {entry['confidence_auroc_correctness']:10.3f} "
            f"{entry['confidence_mean']:6.1f} {entry['accuracy']:6.3f} "
            f"{entry['confidence_mean'] / 100 - entry['accuracy']:10.3f}"
        )

    print("\n=== H6: AUROC incremental de indicadores de calidad (CV por sujeto) ===")
    print(
        f"{'condición':16s} {'conf':>7s} {'+gate':>7s} {'Δgate':>7s} "
        f"{'+SQI':>7s} {'ΔSQI':>7s}"
    )
    for condition in conditions:
        gate = report["indicator_incremental"]["gate"].get(condition) or {}
        sqi = report["indicator_incremental"]["sqi"].get(condition) or {}
        if not gate and not sqi:
            continue
        print(
            f"{condition:16s} {gate.get('auroc_confidence', float('nan')):7.3f} "
            f"{gate.get('auroc_confidence_plus_quality', float('nan')):7.3f} "
            f"{gate.get('delta_auroc', float('nan')):7.3f} "
            f"{sqi.get('auroc_confidence_plus_quality', float('nan')):7.3f} "
            f"{sqi.get('delta_auroc', float('nan')):7.3f}"
        )

    if args.compare_with:
        other_dir = Path(args.compare_with)
        other_records = load_records(other_dir, set(conditions), split)
        if not args.no_recovery:
            other_records, _ = apply_recovery(other_records)
        other_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in other_records:
            other_by_condition[record["condition_id"]].append(record)
        report["comparison"] = {
            "other_dir": str(other_dir),
            "conditions": {},
        }
        print("\n=== H3: contraste de representaciones (numérica vs otra) ===")
        print(
            f"{'condición':16s} {'acc_num':>8s} {'acc_txt':>8s} {'Δacc':>7s} "
            f"{'abst_num':>9s} {'abst_txt':>9s} {'Δabst':>7s} {'Δclaims':>8s} {'Δcerteza':>9s}"
        )
        for condition in conditions:
            base = report["conditions"][condition]
            other = analyze_fn(other_by_condition.get(condition, []), labels)
            if not base.get("n") or not other.get("n"):
                continue
            d_acc = other["accuracy"] - base["accuracy"]
            d_abst = other["abstention_rate"] - base["abstention_rate"]
            d_claims = (
                other["claims"]["supported_rate"] - base["claims"]["supported_rate"]
            )
            d_cert = (
                other["certainty"]["certainty_index_mean"]
                - base["certainty"]["certainty_index_mean"]
            )
            report["comparison"]["conditions"][condition] = {
                "numeric": {
                    "accuracy": base["accuracy"],
                    "abstention_rate": base["abstention_rate"],
                    "claims_supported": base["claims"]["supported_rate"],
                    "certainty_index": base["certainty"]["certainty_index_mean"],
                },
                "other": {
                    "accuracy": other["accuracy"],
                    "abstention_rate": other["abstention_rate"],
                    "claims_supported": other["claims"]["supported_rate"],
                    "certainty_index": other["certainty"]["certainty_index_mean"],
                },
                "delta": {
                    "accuracy": d_acc,
                    "abstention_rate": d_abst,
                    "claims_supported": d_claims,
                    "certainty_index": d_cert,
                },
            }
            print(
                f"{condition:16s} {base['accuracy']:8.3f} {other['accuracy']:8.3f} "
                f"{d_acc:7.3f} {base['abstention_rate']:9.3f} "
                f"{other['abstention_rate']:9.3f} {d_abst:7.3f} "
                f"{d_claims:8.3f} {d_cert:9.3f}"
            )
        # Interacción: efecto de la degradación en cada brazo (C0 -> peor condición)
        base_condition = conditions[0]
        worst = conditions[-1]
        if base_condition in report["comparison"]["conditions"] and worst in report["comparison"]["conditions"]:
            num = report["comparison"]["conditions"]
            print(
                f"\nEfecto de la degradación {base_condition} -> {worst}: "
                f"numérico Δacc={num[worst]['numeric']['accuracy'] - num[base_condition]['numeric']['accuracy']:+.3f}, "
                f"texto Δacc={num[worst]['other']['accuracy'] - num[base_condition]['other']['accuracy']:+.3f}"
            )

    print(f"\nReporte: {output_path}")




# ---------------------------------------------------------------------------
# H1 (referencia por condición), H2a (AUROC de confianza) y H6 (gate)
# ---------------------------------------------------------------------------


def reference_by_condition(
    conditions: list[str],
    subjects: list[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Resolubilidad por condición con un modelo features-only (CV por sujeto)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from sources.phase3.degradation import condition_matrix, load_features

    feature_names = sorted(
        {
            name
            for window in load_features(INGEST_DIR, "C0_clean", subjects[0]).values()
            for name in window
        }
    )
    results: dict[str, Any] = {}
    for condition in conditions:
        matrix, groups, window_ids = condition_matrix(
            INGEST_DIR, condition, subjects, feature_names
        )
        y4 = np.array([labels.get(window_id, "UNKNOWN") for window_id in window_ids])
        y2 = np.array([1 if label == "stress" else 0 for label in y4])
        if len(set(groups)) < 2:
            continue
        splitter = GroupKFold(n_splits=min(5, len(set(groups))))
        bal4, bal2 = [], []
        for train_idx, test_idx in splitter.split(matrix, y4, groups):
            if len(set(y4[train_idx])) < 2 or len(set(y4[test_idx])) < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
            model.fit(matrix[train_idx], y4[train_idx])
            bal4.append(
                balanced_accuracy_score(y4[test_idx], model.predict(matrix[test_idx]))
            )
            model.fit(matrix[train_idx], y2[train_idx])
            bal2.append(
                balanced_accuracy_score(y2[test_idx], model.predict(matrix[test_idx]))
            )
        results[condition] = {
            "n": int(len(y4)),
            "balanced_accuracy_4class": float(np.mean(bal4)) if bal4 else float("nan"),
            "balanced_accuracy_4class_std": float(np.std(bal4)) if bal4 else float("nan"),
            "balanced_accuracy_binary": float(np.mean(bal2)) if bal2 else float("nan"),
            "resolvable": bool(bal4 and np.mean(bal4) > 0.55),
        }
    return results


def confidence_auroc(records: list[dict[str, Any]], labels: dict[str, str]) -> float:
    """AUROC de la confianza como predictor de corrección (solo decididas)."""
    from sources.phase3.evaluator import _compute_auroc_one_vs_rest

    y_true: list[str] = []
    scores: list[float] = []
    for record in records:
        decision = record.get("decision")
        if decision not in ("baseline", "stress", "amusement", "meditation"):
            continue
        truth = labels.get(record["window_id"], "UNKNOWN")
        if truth == "UNKNOWN":
            continue
        y_true.append("correct" if decision == truth else "incorrect")
        scores.append(float(record.get("confidence") or 0) / 100.0)
    if not y_true or len(set(y_true)) < 2:
        return float("nan")
    return _compute_auroc_one_vs_rest(
        y_true, {"correct": scores, "incorrect": [1.0 - s for s in scores]}, "correct"
    )


def sqi_by_condition(
    conditions: list[str],
    subjects: list[str],
) -> dict[str, dict[str, float]]:
    """SQI agregado (``quality.sqi_overall``) por ventana y condición."""
    from sources.phase3.degradation import features_path

    scores: dict[str, dict[str, float]] = {condition: {} for condition in conditions}
    for subject in subjects:
        for condition in conditions:
            path = features_path(INGEST_DIR, condition, subject)
            for line in path.open(encoding="utf-8"):
                record = json.loads(line)
                quality = record.get("quality", {})
                value = quality.get("sqi_overall")
                if isinstance(value, (int, float)):
                    scores[condition][record["window_id"]] = float(value)
    return scores


def gate_scores_by_condition(
    conditions: list[str],
    subjects: list[str],
    records_by_condition: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Score de consistencia (P1) por ventana y condición (usa valores absolutos)."""
    from sources.phase2.consistency import check_consistency
    from sources.phase3.degradation import load_features

    cache: dict[tuple[str, str], float] = {}
    for subject in subjects:
        for condition in conditions:
            rows = load_features(INGEST_DIR, condition, subject)
            for window_id, values in rows.items():
                report = check_consistency(values, duration_s=60.0)
                cache[(window_id, condition)] = report.consistency_score
    return {
        condition: {
            record["window_id"]: cache.get((record["window_id"], condition), float("nan"))
            for record in records
        }
        for condition, records in records_by_condition.items()
    }


def incremental_quality_auroc(
    records: list[dict[str, Any]],
    labels: dict[str, str],
    quality_scores: dict[str, float],
) -> dict[str, Any]:
    """AUROC de confianza vs confianza+gate con CV agrupada por sujeto (H6)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    rows = []
    for record in records:
        decision = record.get("decision")
        if decision not in ("baseline", "stress", "amusement", "meditation"):
            continue
        truth = labels.get(record["window_id"], "UNKNOWN")
        score = quality_scores.get(record["window_id"], float("nan"))
        if truth == "UNKNOWN" or np.isnan(score):
            continue
        rows.append(
            (
                float(record.get("confidence") or 0) / 100.0,
                float(score),
                1 if decision == truth else 0,
                record["subject"],
            )
        )
    if len(rows) < 30 or len({r[3] for r in rows}) < 2:
        return {}
    confidence = np.array([[r[0]] for r in rows])
    combined = np.array([[r[0], r[1]] for r in rows])
    y = np.array([r[2] for r in rows])
    groups = np.array([r[3] for r in rows])

    splitter = GroupKFold(n_splits=min(5, len(set(groups))))
    auroc_conf, auroc_comb = [], []
    for train_idx, test_idx in splitter.split(confidence, y, groups):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            continue
        for matrix, target in ((confidence, auroc_conf), (combined, auroc_comb)):
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(matrix[train_idx], y[train_idx])
            target.append(roc_auc_score(y[test_idx], model.decision_function(matrix[test_idx])))
    if not auroc_conf:
        return {}
    return {
        "auroc_confidence": float(np.mean(auroc_conf)),
        "auroc_confidence_plus_quality": float(np.mean(auroc_comb)),
        "delta_auroc": float(np.mean(auroc_comb) - np.mean(auroc_conf)),
        "n": len(rows),
    }


if __name__ == "__main__":
    main()
