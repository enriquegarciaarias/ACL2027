#!/usr/bin/env python
"""Piloto E12: representación numérica estandarizada por sujeto.

Genera z-scores intra-sujeto de las 23 features de pecho (0 = media de la
sesión del sujeto, ±1 = 1 DE), construye el prompt numérico en esa escala y
compara con la vista absoluta del mismo sujeto.

Diseño:
- Sujeto piloto por defecto: S3 (46 ventanas).
- Condición por defecto: C0_clean. ``--conditions`` admite NC_*.
- Mismo modelo/semilla/temperatura que Fase 2 (qwen2.5:7b, seed 42, T=0.3).
- La normalización usa la distribución de la sesión del sujeto (sin etiquetas).
  Es una aproximación transductiva; en despliegue se usaría una referencia
  inicial o una ventana móvil.

Uso:
    python scripts/run_e12_pilot.py --subject S3
    python scripts/run_e12_pilot.py --subject S3 --conditions C0_clean,NC_shuffled
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

from sources.common.provenance import utc_now
from sources.ingestion.represent import (
    LLM_ALLOWED_FEATURES,
    _opaque_window_id,
)
from sources.phase2.development_split import (
    DEV_SUBJECTS,
    TEST_SUBJECTS,
    split_warning,
)
from sources.phase2.llm_config import build_experiment_config, create_ollama_chat_client
from sources.phase2.negative_controls import generate_negative_control_rep_row
from sources.phase2.orchestrator import _call_ollama_chat
from sources.phase2.prompt_builder import (
    assert_no_leakage,
    build_numeric_prompt,
)
from sources.phase2.response_schema import (
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_STRESS_BINARY,
    validate_llm_output,
)
from sources.phase3.evaluator import LABEL_ORDER
from sources.phase3.records import evaluate_binary_records, evaluate_records as metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
OUTPUT_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "e12"
CHEST_MODALITIES = ("ecg", "eda", "resp", "temp")


def load_subject(subject: str) -> tuple[list[dict[str, Any]], list[str]]:
    features_path = INGEST_DIR / "features" / f"{subject}.features.jsonl"
    windows_path = INGEST_DIR / "windows" / f"{subject}.windows.jsonl"
    windows = {
        json.loads(line)["window_id"]: json.loads(line)
        for line in windows_path.open(encoding="utf-8")
    }
    rows: list[dict[str, Any]] = []
    feature_names: set[str] = set()
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
        rows.append(
            {
                "window_id": record["window_id"],
                "window": window,
                "label": window["label"]["name"],
                "values": values,
            }
        )
    return rows, sorted(feature_names)


def add_subject_scaling(
    rows: list[dict[str, Any]], feature_names: list[str], method: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Estandariza cada feature con la distribución del sujeto (sin etiquetas).

    ``zscore``: (x - media) / DE.
    ``robust``: (x - mediana) / IQR, recortado a [-3, 3] para limitar colas.
    """
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = np.array([row["values"].get(name, np.nan) for row in rows], dtype=float)
        finite = values[~np.isnan(values)]
        if finite.size == 0:
            centers[name] = 0.0
            scales[name] = 1.0
            continue
        if method == "robust":
            center = float(np.median(finite))
            q1, q3 = np.percentile(finite, [25, 75])
            scale = float(q3 - q1)
            if scale == 0:
                scale = float(np.std(finite)) or 1.0
        else:
            center = float(np.mean(finite))
            scale = float(np.std(finite)) or 1.0
        centers[name] = center
        scales[name] = scale
    for row in rows:
        row["zscores"] = {}
        for name in feature_names:
            value = row["values"].get(name, np.nan)
            if np.isnan(value):
                row["zscores"][name] = None
            else:
                z = (value - centers[name]) / scales[name]
                if method == "robust":
                    z = float(np.clip(z, -3.0, 3.0))
                row["zscores"][name] = float(z)
    return centers, scales


def build_payload(row: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    modalities: dict[str, dict[str, Any]] = defaultdict(dict)
    for name in feature_names:
        if not name.startswith(tuple(CHEST_MODALITIES)):
            continue
        modality = name.split("_")[0]
        value = row["zscores"].get(name)
        modalities[modality][name] = {
            "value": value,
            "unit": "SD",
            "kind": "derived",
            "method": "within_subject_zscore",
        }
    return {
        "modalities": dict(modalities),
        "feature_version": "e12",
        "window": {
            "window_id": _opaque_window_id(row["window_id"]),
            "duration_s": row["window"].get("duration_s", 60.0),
        },
    }


def call_model(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    conditions: list[str],
    model: str,
    temperature: float,
    seed: int,
    rubric: bool = False,
    consistency_check: bool = False,
    decision_mode: str = DECISION_MODE_AFFECTIVE,
    graded_confidence: bool = False,
) -> list[dict[str, Any]]:
    config = build_experiment_config(
        model=model,
        temperature=temperature,
        seed=seed,
        max_retries=3,
        cascade_enabled=False,
    )
    client = create_ollama_chat_client(config)
    records: list[dict[str, Any]] = []

    for row in rows:
        for condition in conditions:
            rep_row = {
                "window_id": row["window_id"],
                "payload": build_payload(row, feature_names),
            }
            payload = rep_row["payload"]
            if condition.startswith("NC_"):
                nc_row = generate_negative_control_rep_row(rep_row, condition)
                payload = nc_row["payload"]
            window_id_opaque = _opaque_window_id(row["window_id"])
            prompt = build_numeric_prompt(
                payload=payload,
                window_id_opaque=window_id_opaque,
                duration_s=row["window"].get("duration_s", 60.0),
                num_ctx=config.num_ctx,
                use_few_shot=True,
                include_additional=True,
                standardized=True,
                rubric=rubric,
                consistency_check=consistency_check,
                decision_mode=decision_mode,
                graded_confidence=graded_confidence,
            )
            assert_no_leakage(prompt)
            try:
                raw_output, ollama_meta = _call_ollama_chat(
                    client, prompt, config, seed=seed
                )
                validated = validate_llm_output(raw_output, decision_mode=decision_mode)
                status = validated.status
                decision = validated.decision
                confidence = validated.confidence
                claims = [c.to_dict() for c in validated.claims]
                reasoning = validated.reasoning
                error = validated.error_message
            except Exception as exc:  # noqa: BLE001
                raw_output, ollama_meta = "", {}
                status, decision, confidence = "timeout", None, None
                claims, reasoning, error = [], "", str(exc)
            records.append(
                {
                    "subject": row["window_id"].split(":")[1],
                    "window_id": row["window_id"],
                    "window_id_opaque": window_id_opaque,
                    "condition_id": condition,
                    "representation": "numeric_llm_centered",
                    "normalization": "subject_centered",
                    "prompt_variant": (
                        "graded_confidence"
                        if graded_confidence
                        else ("consistency" if consistency_check else "base")
                    ),
                    "split": "development" if row["window_id"].split(":")[1] in DEV_SUBJECTS else "evaluation",
                    "decision_mode": decision_mode,
                    "status": status,
                    "decision": decision,
                    "confidence": confidence,
                    "claims": claims,
                    "reasoning": reasoning,
                    "error_message": error,
                    "raw_output": raw_output,
                    "prompt": prompt,
                    "ollama_meta": ollama_meta,
                    "generated_at": utc_now(),
                }
            )
            print(
                f"[E12] {row['window_id']} {condition}: {status} "
                f"decision={decision} conf={confidence}"
            )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="",
        help="Sujeto único (deprecado; usar --subjects).",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Lista separada por comas. Por defecto, los sujetos de EVALUACIÓN "
        "(split congelado); los de desarrollo requieren --allow-dev.",
    )
    parser.add_argument(
        "--allow-dev",
        action="store_true",
        help="Permitir sujetos de desarrollo (S2-S4): resultados no son de evaluación.",
    )
    parser.add_argument("--conditions", default="C0_clean")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scaling", choices=["robust", "zscore"], default="robust")
    parser.add_argument("--rubric", action="store_true")
    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Añade la comprobación de consistencia interna al prompt (P1).",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Endpoint binario stress vs non-stress (P2).",
    )
    parser.add_argument(
        "--graded-confidence",
        action="store_true",
        help="Rúbrica de confianza graduada para evitar la saturación (P4).",
    )
    parser.add_argument("--tag", default="", help="Sufijo para los ficheros de salida.")
    args = parser.parse_args(argv)

    subjects = [
        s.strip().upper()
        for s in (args.subjects or args.subject).split(",")
        if s.strip()
    ]
    if not subjects:
        subjects = list(TEST_SUBJECTS)
    warning = split_warning(subjects)
    if warning:
        if not args.allow_dev:
            raise SystemExit(
                f"[E12] {warning}.\n"
                "Usa --allow-dev para ejecutar en desarrollo (resultados exploratorios) "
                "o elige sujetos de evaluación."
            )
        print(f"[E12] AVISO: {warning}")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    decision_mode = (
        DECISION_MODE_STRESS_BINARY if args.binary else DECISION_MODE_AFFECTIVE
    )
    print(f"Sujetos: {subjects} | condiciones: {conditions} | scaling={args.scaling} "
          f"| rubric={args.rubric} | consistency={args.consistency} "
          f"| decision_mode={decision_mode} | graded={args.graded_confidence}")

    all_records: list[dict[str, Any]] = []
    absolute_records: list[dict[str, Any]] = []
    label_by_opaque: dict[str, str] = {}
    per_subject: dict[str, Any] = {}

    for subject in subjects:
        rows, feature_names = load_subject(subject)
        if not rows:
            print(f"[E12] {subject}: sin ventanas, se omite")
            continue
        add_subject_scaling(rows, feature_names, args.scaling)
        label_by_opaque.update(
            {_opaque_window_id(row["window_id"]): row["label"] for row in rows}
        )
        records = call_model(
            rows, feature_names, conditions, args.model, args.temperature, args.seed,
            rubric=args.rubric, consistency_check=args.consistency,
            decision_mode=decision_mode,
            graded_confidence=args.graded_confidence,
        )
        suffix = (
            ("_" + args.tag if args.tag else "")
            + ("_rubric" if args.rubric else "")
            + ("_consistency" if args.consistency else "")
            + ("_binary" if args.binary else "")
            + ("_graded" if args.graded_confidence else "")
        )
        output_path = OUTPUT_DIR / f"{subject}.numeric_centered{suffix}.responses.jsonl"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        all_records.extend(records)

        absolute_path = (
            REPO_ROOT
            / "results"
            / "output"
            / "WESAD"
            / "PHASE2"
            / "llm_responses"
            / f"{subject}.responses.jsonl"
        )
        if absolute_path.is_file():
            for line in absolute_path.open(encoding="utf-8"):
                record = json.loads(line)
                if (
                    record.get("representation") == "numeric_llm"
                    and record.get("condition_id") in conditions
                ):
                    absolute_records.append(record)
        print(f"[E12] {subject}: {len(records)} respuestas -> {output_path}")

    def build_condition_metrics(condition: str) -> dict[str, Any]:
        centered = [r for r in all_records if r["condition_id"] == condition]
        absolute = [r for r in absolute_records if r["condition_id"] == condition]
        return {
            "centered": metrics(centered, label_by_opaque) if centered else None,
            "absolute": metrics(absolute, label_by_opaque) if absolute else None,
            "centered_binary": (
                evaluate_binary_records(centered, label_by_opaque) if centered else None
            ),
            "absolute_binary": (
                evaluate_binary_records(
                    absolute,
                    label_by_opaque,
                    # La vista absoluta se generó con el prompt de 4 clases:
                    # se reinterpreta como binario para poder comparar.
                    other_decisions_map={
                        "baseline": "non-stress",
                        "amusement": "non-stress",
                        "meditation": "non-stress",
                    },
                )
                if absolute
                else None
            ),
        }

    condition_metrics = {c: build_condition_metrics(c) for c in conditions}
    centered_metrics = condition_metrics.get("C0_clean", {}).get("centered")
    absolute_metrics = condition_metrics.get("C0_clean", {}).get("absolute")
    centered_binary = condition_metrics.get("C0_clean", {}).get("centered_binary")
    absolute_binary = condition_metrics.get("C0_clean", {}).get("absolute_binary")

    def show_binary(name: str, m: dict[str, Any] | None) -> None:
        if not m:
            print(f"{name}: sin datos")
            return
        print(
            f"\n{name} (stress vs non-stress): n={m['n']} válidas={m['n_valid']} "
            f"inválidas={m['n_invalid']} abstain={m['n_abstain']} "
            f"decididas={m['n_decided']} cobertura={m['coverage']:.3f}"
        )
        print(
            f"  accuracy={m['accuracy']:.3f} balanced={m['balanced_accuracy']:.3f} "
            f"precision={m['precision_stress']:.3f} recall={m['recall_stress']:.3f} "
            f"F1={m['f1_stress']:.3f}"
        )
        print(
            f"  abstención={m['abstention_rate']:.3f} AUROC={m['auroc_stress']:.3f} "
            f"ECE={m['ece']:.3f} Brier={m['brier']:.3f}"
        )
        print(f"  decisiones: {m['decisions']}")
        print("  confusión (real -> predicho):")
        for label in ("stress", "non-stress"):
            print(
                f"    {label:10s} "
                + " ".join(
                    f"{p[:7]}={m['confusion'][label][p]}"
                    for p in ("stress", "non-stress", "abstain")
                )
            )

    def show(name: str, m: dict[str, Any] | None) -> None:
        if not m:
            print(f"{name}: sin datos")
            return
        print(
            f"\n{name}: n={m['n']} válidas={m['n_valid']} inválidas={m['n_invalid']} "
            f"abstain={m['n_abstain']} decididas={m['n_decided']}"
        )
        print(
            f"  accuracy={m['accuracy']:.3f} balanced_acc={m['balanced_accuracy']:.3f} "
            f"abstention={m['abstention_rate']:.3f} ECE={m['ece']:.3f} Brier={m['brier']:.3f}"
        )
        print("  AUROC: " + " ".join(f"{c}={m['auroc'][c]:.3f}" for c in LABEL_ORDER))
        print(
            "  binario stress/baseline: "
            f"n={m['binary']['n']} acc={m['binary']['accuracy']:.3f} "
            f"precision={m['binary']['precision_stress']:.3f} "
            f"recall={m['binary']['recall_stress']:.3f} "
            f"balanced={m['binary']['balanced_accuracy']:.3f}"
        )
        print(f"  decisiones: {m['decisions']}")
        print("  confusión (real -> predicho):")
        for label in LABEL_ORDER:
            print(
                f"    {label:10s} "
                + " ".join(
                    f"{p[:7]}={m['confusion'][label][p]}"
                    for p in list(LABEL_ORDER) + ["abstain"]
                )
            )

    if args.binary:
        print("\n=== P2: prompt binario stress vs non-stress (C0_clean) ===")
        show_binary("Centrado", centered_binary)
        show_binary("Absoluto", absolute_binary)
    else:
        print("\n=== E12: centrado por sujeto (C0_clean) ===")
        show("Centrado", centered_metrics)
        print("\n=== Línea base: vista absoluta (C0_clean) ===")
        show("Absoluto", absolute_metrics)

    print("\n=== Métricas por condición ===")
    if args.binary:
        print(
            f"{'cond':12s} {'abs_abst':>9s} {'cnt_abst':>9s} "
            f"{'abs_acc':>8s} {'cnt_acc':>8s} {'abs_bal':>8s} {'cnt_bal':>8s}"
        )
        for condition in conditions:
            entry = condition_metrics[condition]
            a, c = entry["absolute_binary"], entry["centered_binary"]
            def fmt_binary(value: float | None) -> str:
                return "  —  " if value is None or np.isnan(value) else f"{value:.3f}"
            print(
                f"{condition:12s} "
                f"{fmt_binary(a['abstention_rate'] if a else None):>9s} "
                f"{fmt_binary(c['abstention_rate'] if c else None):>9s} "
                f"{fmt_binary(a['accuracy'] if a else None):>8s} "
                f"{fmt_binary(c['accuracy'] if c else None):>8s} "
                f"{fmt_binary(a['balanced_accuracy'] if a else None):>8s} "
                f"{fmt_binary(c['balanced_accuracy'] if c else None):>8s}"
            )
            if c:
                print(f"    centrado decisiones: {c['decisions']}")
            if a:
                print(f"    absoluto decisiones: {a['decisions']}")
    else:
        print(
            f"{'cond':12s} {'abs_abst':>9s} {'cnt_abst':>9s} "
            f"{'abs_acc':>8s} {'cnt_acc':>8s} {'abs_bal':>8s} {'cnt_bal':>8s}"
        )
        for condition in conditions:
            entry = condition_metrics[condition]
            a, c = entry["absolute"], entry["centered"]
            def fmt(value: float | None) -> str:
                return "  —  " if value is None or np.isnan(value) else f"{value:.3f}"
            print(
                f"{condition:12s} "
                f"{fmt(a['abstention_rate'] if a else None):>9s} "
                f"{fmt(c['abstention_rate'] if c else None):>9s} "
                f"{fmt(a['accuracy'] if a else None):>8s} "
                f"{fmt(c['accuracy'] if c else None):>8s} "
                f"{fmt(a['balanced_accuracy'] if a else None):>8s} "
                f"{fmt(c['balanced_accuracy'] if c else None):>8s}"
            )
            if c:
                print(f"    centrado decisiones: {c['decisions']}")
            if a:
                print(f"    absoluto decisiones: {a['decisions']}")

    summary = {
        "subjects": subjects,
        "split_version": "1.0",
        "development_subjects": list(DEV_SUBJECTS),
        "conditions": conditions,
        "scaling": args.scaling,
        "rubric": args.rubric,
        "consistency_check": args.consistency,
        "graded_confidence": args.graded_confidence,
        "decision_mode": decision_mode,
        "condition_metrics": condition_metrics,
        "c0_centered": centered_metrics,
        "c0_absolute": absolute_metrics,
        "c0_centered_binary": centered_binary,
        "c0_absolute_binary": absolute_binary,
    }
    suffix = (
        ("_" + args.tag if args.tag else "")
        + ("_rubric" if args.rubric else "")
        + ("_consistency" if args.consistency else "")
        + ("_binary" if args.binary else "")
        + ("_graded" if args.graded_confidence else "")
    )
    summary_path = OUTPUT_DIR / f"e12_summary{suffix}.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    combined_path = OUTPUT_DIR / f"all.numeric_centered{suffix}.responses.jsonl"
    with combined_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nResumen: {summary_path}")
    print(f"Respuestas combinadas: {combined_path}")


if __name__ == "__main__":
    main()
