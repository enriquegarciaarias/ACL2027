#!/usr/bin/env python
"""Análisis de la compuerta de consistencia fisiológica (P1 de Fase 2).

Sin llamadas al LLM, mide dos cosas sobre el corpus completo:

1. **Discriminación limpio vs corrupto.** Aplica ``check_consistency`` a las
   features absolutas de cada ventana y compara la tasa de violaciones y de
   abstención de la compuerta entre ``C0_clean`` y los controles negativos.
2. **Sistema compuesto.** Reutiliza las respuestas E12 ya generadas y aplica
   la compuerta como veto previo: si la compuerta se abstiene, la predicción
   final es ``abstain`` (sin llamar al LLM). Compara las métricas del LLM solo
   frente al sistema compuesto.

Salidas:
    results/output/WESAD/PHASE2/e12/consistency_gate_report.json

Uso:
    python scripts/analyze_consistency.py
    python scripts/analyze_consistency.py --subjects S3 --conditions C0_clean,NC_shuffled
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.ingestion.represent import LLM_ALLOWED_FEATURES, _opaque_window_id
from sources.phase2.consistency import (
    CHECK_SEVERITY,
    check_consistency,
    should_abstain,
)
from sources.phase2.negative_controls import generate_negative_control_rep_row
from sources.phase3.records import evaluate_binary_records, evaluate_records

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
PHASE2_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2"
E12_DIR = PHASE2_DIR / "e12"
DEFAULT_OUTPUT = E12_DIR / "consistency_gate_report.json"
CHEST_MODALITIES = ("ecg", "eda", "resp", "temp")
ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


def stress_values(
    modalities: dict[str, dict[str, Any]], kind: str, seed: int = 0
) -> dict[str, float | None]:
    """Aplica una corrupción controlada a los valores absolutos.

    Sirve para medir la sensibilidad de la compuerta más allá del control
    ``NC_shuffled`` (que usa una permutación fija). Tipos:

    - ``shuffle_modality_random``: permuta los valores dentro de cada
      modalidad con una semilla distinta por ventana.
    - ``shuffle_global``: permuta todos los valores entre modalidades.
    - ``swap_one``: intercambia un único par de valores de una modalidad.
    - ``noise_10pct``: ruido multiplicativo ±10 % (coherente; la compuerta no
      debería dispararse).
    """
    import random

    rng = random.Random(seed)
    items = [
        (modality, name, meta.get("value"))
        for modality, block in modalities.items()
        for name, meta in block.items()
    ]
    values = {name: value for _, name, value in items}

    if kind.startswith("shuffle_modality"):
        result = dict(values)
        for modality, block in modalities.items():
            names = list(block)
            permuted = [values[name] for name in names]
            rng.shuffle(permuted)
            for name, value in zip(names, permuted):
                result[name] = value
        return result

    if kind == "shuffle_global":
        permuted = [value for _, _, value in items]
        rng.shuffle(permuted)
        return {name: value for (_, name, _), value in zip(items, permuted)}

    if kind == "swap_one":
        result = dict(values)
        candidates = [
            (modality, list(block))
            for modality, block in modalities.items()
            if len(block) >= 2
        ]
        if candidates:
            _, names = rng.choice(candidates)
            first, second = rng.sample(names, 2)
            result[first], result[second] = result[second], result[first]
        return result

    if kind == "noise_10pct":
        result = {}
        for name, value in values.items():
            if value is None:
                result[name] = None
            else:
                result[name] = value * (1.0 + 0.10 * rng.gauss(0.0, 1.0))
        return result

    raise ValueError(f"Tipo de corrupción no válido: {kind}")


STRESS_KINDS: tuple[str, ...] = (
    "shuffle_modality_random",
    "shuffle_global",
    "swap_one",
    "noise_10pct",
)


def load_groundtruth_labels() -> dict[str, str]:
    """Etiquetas reales por identificador opaco, desde ``groundtruth/labels.jsonl``."""
    path = INGEST_DIR / "groundtruth" / "labels.jsonl"
    if not path.is_file():
        return {}
    labels: dict[str, str] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        if label:
            labels[_opaque_window_id(record["window_id"])] = label
    return labels


def load_subject_rows(subject: str) -> list[dict[str, Any]]:
    """Carga las features absolutas curadas de un sujeto, por ventana."""
    features_path = INGEST_DIR / "features" / f"{subject}.features.jsonl"
    windows_path = INGEST_DIR / "windows" / f"{subject}.windows.jsonl"
    windows = {
        json.loads(line)["window_id"]: json.loads(line)
        for line in windows_path.open(encoding="utf-8")
    }
    rows: list[dict[str, Any]] = []
    for line in features_path.open(encoding="utf-8"):
        record = json.loads(line)
        window = windows.get(record["window_id"])
        if window is None:
            continue
        modalities: dict[str, dict[str, Any]] = {}
        for modality in CHEST_MODALITIES:
            block = record.get("features", {}).get(modality, {})
            kept = {
                name: dict(meta)
                for name, meta in block.items()
                if name in LLM_ALLOWED_FEATURES
            }
            if kept:
                modalities[modality] = kept
        rows.append(
            {
                "window_id": record["window_id"],
                "window_id_opaque": _opaque_window_id(record["window_id"]),
                "label": window["label"]["name"],
                "duration_s": window.get("duration_s", 60.0),
                "modalities": modalities,
            }
        )
    return rows


def condition_values(
    modalities: dict[str, dict[str, Any]], condition: str
) -> tuple[dict[str, float | None], bool]:
    """Aplica el control negativo (si procede) y aplana las features."""
    payload_modalities = modalities
    if condition.startswith("NC_"):
        rep_row = {"window_id": "condition", "payload": {"modalities": modalities}}
        payload_modalities = generate_negative_control_rep_row(rep_row, condition)[
            "payload"
        ]["modalities"]
    values = {
        name: meta.get("value")
        for block in payload_modalities.values()
        for name, meta in block.items()
    }
    return values, condition.startswith("NC_")


def load_responses(path: Path, condition: str, representation: str | None = None):
    """Carga respuestas de un JSONL filtrando por condición y representación."""
    if not path.is_file():
        return []
    records = []
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if record.get("condition_id") != condition:
            continue
        if representation and record.get("representation") != representation:
            continue
        records.append(record)
    return records


def apply_gate(
    records: list[dict[str, Any]], gate_by_opaque: dict[str, bool]
) -> list[dict[str, Any]]:
    """Aplica la compuerta como veto: si se abstiene, la predicción es abstain."""
    gated = []
    for record in records:
        new_record = dict(record)
        if gate_by_opaque.get(record["window_id_opaque"], False):
            new_record["decision"] = "abstain"
            new_record["confidence"] = 0
        gated.append(new_record)
    return gated


def summarize_gate(
    reports: list[tuple[str, str, dict[str, Any], float, bool]],
) -> dict[str, Any]:
    """Resume la compuerta: violaciones por check y tasa de abstención."""
    per_condition: dict[str, Any] = {}
    for condition in sorted({c for c, *_ in reports}):
        subset = [r for r in reports if r[0] == condition]
        n = len(subset)
        gate_abstain = sum(1 for r in subset if r[4])
        check_failures: Counter = Counter()
        check_evaluated: Counter = Counter()
        for _, _, report, _, _ in subset:
            for check in report.checks:
                if check.ok is None:
                    continue
                check_evaluated[check.name] += 1
                if check.ok is False:
                    check_failures[check.name] += 1
        per_condition[condition] = {
            "n": n,
            "gate_abstention_rate": gate_abstain / n if n else 0.0,
            "hard_failures_mean": (
                sum(r[2].n_hard_failures for r in subset) / n if n else 0.0
            ),
            "soft_failures_mean": (
                sum(r[2].n_soft_failures for r in subset) / n if n else 0.0
            ),
            "consistency_score_mean": (
                sum(r[2].consistency_score for r in subset) / n if n else 0.0
            ),
            "missing_key_features_rate": (
                sum(1 for r in subset if r[2].missing_key_features) / n if n else 0.0
            ),
            "check_failure_rate": {
                name: check_failures[name] / check_evaluated[name]
                for name in check_evaluated
            },
            "check_severity": dict(CHECK_SEVERITY),
        }
    return per_condition


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="", help="Lista separada por comas.")
    parser.add_argument(
        "--conditions",
        default="C0_clean,NC_shuffled,NC_zero,NC_nan,NC_absent",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--max-hard-failures", type=int, default=0, help="Fallos hard tolerados."
    )
    parser.add_argument(
        "--max-soft-failures", type=int, default=1, help="Fallos soft tolerados."
    )
    args = parser.parse_args(argv)

    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        or ALL_SUBJECTS
    )
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"Sujetos: {subjects}")
    print(f"Condiciones: {conditions}")
    print(
        f"Compuerta: max_hard={args.max_hard_failures} "
        f"max_soft={args.max_soft_failures}"
    )

    reports: list[tuple[str, str, Any, float, bool]] = []
    label_by_opaque: dict[str, str] = load_groundtruth_labels()
    gate_by_condition: dict[str, dict[str, bool]] = defaultdict(dict)
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}

    for subject in subjects:
        rows = load_subject_rows(subject)
        if not rows:
            print(f"[consistency] {subject}: sin ventanas, se omite")
            continue
        rows_by_subject[subject] = rows
        for row in rows:
            label_by_opaque[row["window_id_opaque"]] = row["label"]
        for condition in conditions:
            for row in rows:
                values, _ = condition_values(row["modalities"], condition)
                report = check_consistency(values, row["duration_s"])
                abstain = should_abstain(
                    report,
                    max_hard_failures=args.max_hard_failures,
                    max_soft_failures=args.max_soft_failures,
                )
                reports.append(
                    (condition, row["window_id"], report, row["duration_s"], abstain)
                )
                gate_by_condition[condition][row["window_id_opaque"]] = abstain
        print(f"[consistency] {subject}: {len(rows)} ventanas procesadas")

    gate_summary = summarize_gate(reports)

    # Sistema compuesto: compuerta + respuestas LLM ya existentes.
    composed: dict[str, Any] = {}
    absolute_dir = PHASE2_DIR / "llm_responses"
    absolute_all: list[dict[str, Any]] = []
    for path in sorted(absolute_dir.glob("S*.responses.jsonl")):
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            if (
                record.get("condition_id") == "C0_clean"
                and record.get("representation") == "numeric_llm"
            ):
                absolute_all.append(record)

    variants: list[tuple[str, Path]] = [
        ("e12_centered", E12_DIR / "all.numeric_centered.responses.jsonl"),
        ("e12_centered_nc", E12_DIR / "all.numeric_centered_nc.responses.jsonl"),
    ]
    for path in sorted(E12_DIR.glob("all.numeric_centered_*.responses.jsonl")):
        if path.name in {p.name for _, p in variants}:
            continue
        variants.append((path.stem.replace("all.", ""), path))

    for name, path in variants:
        for condition in conditions:
            records = load_responses(path, condition)
            if not records:
                continue
            gate = gate_by_condition.get(condition, {})
            is_binary = any(
                record.get("decision_mode") == "stress_binary" for record in records
            )
            entry: dict[str, Any] = {
                "gate_abstention_rate": (
                    sum(1 for r in records if gate.get(r["window_id_opaque"], False))
                    / len(records)
                ),
                "n_records": len(records),
                "decision_mode": "stress_binary" if is_binary else "affective",
            }
            if is_binary:
                entry["llm_only"] = evaluate_binary_records(records, label_by_opaque)
                entry["llm_plus_gate"] = evaluate_binary_records(
                    apply_gate(records, gate), label_by_opaque
                )
            else:
                entry["llm_only"] = evaluate_records(records, label_by_opaque)
                entry["llm_plus_gate"] = evaluate_records(
                    apply_gate(records, gate), label_by_opaque
                )
            composed[f"{name}:{condition}"] = entry

    if absolute_all:
        gate = gate_by_condition.get("C0_clean", {})
        composed["absolute:C0_clean"] = {
            "decision_mode": "affective",
            "llm_only": evaluate_records(absolute_all, label_by_opaque),
            "llm_plus_gate": evaluate_records(
                apply_gate(absolute_all, gate), label_by_opaque
            ),
            "gate_abstention_rate": (
                sum(1 for r in absolute_all if gate.get(r["window_id_opaque"], False))
                / len(absolute_all)
            ),
            "n_records": len(absolute_all),
        }

    # Robustez: corrupciones alternativas (permutación aleatoria por ventana,
    # permutación global, un único intercambio y ruido coherente).
    stress: dict[str, Any] = {}
    for kind in STRESS_KINDS:
        abstain_total = 0
        total = 0
        scores = []
        for subject in subjects:
            for row in rows_by_subject.get(subject, []):
                values = stress_values(
                    row["modalities"], kind, seed=zlib.crc32(row["window_id"].encode("utf-8"))
                )
                report = check_consistency(values, row["duration_s"])
                abstain_total += int(
                    should_abstain(
                        report,
                        max_hard_failures=args.max_hard_failures,
                        max_soft_failures=args.max_soft_failures,
                    )
                )
                scores.append(report.consistency_score)
                total += 1
        stress[kind] = {
            "n": total,
            "gate_abstention_rate": abstain_total / total if total else 0.0,
            "consistency_score_mean": sum(scores) / len(scores) if scores else 0.0,
        }

    # Calibración de la laxitud: compromiso entre falsos positivos con ruido
    # coherente y detección de corrupción estructural.
    scale_sweep: dict[str, dict[str, float]] = {}
    for tolerance_scale in (1.0, 1.5, 2.0, 3.0):
        rates: dict[str, float] = {}
        for condition in ("C0_clean", "NC_shuffled"):
            total = 0
            abstain_total = 0
            for subject in subjects:
                for row in rows_by_subject.get(subject, []):
                    values, _ = condition_values(row["modalities"], condition)
                    report = check_consistency(
                        values, row["duration_s"], tolerance_scale=tolerance_scale
                    )
                    abstain_total += int(
                        should_abstain(
                            report,
                            max_hard_failures=args.max_hard_failures,
                            max_soft_failures=args.max_soft_failures,
                        )
                    )
                    total += 1
            rates[condition] = abstain_total / total if total else 0.0
        for kind in ("swap_one", "noise_10pct"):
            total = 0
            abstain_total = 0
            for subject in subjects:
                for row in rows_by_subject.get(subject, []):
                    values = stress_values(
                        row["modalities"], kind, seed=zlib.crc32(row["window_id"].encode("utf-8"))
                    )
                    report = check_consistency(
                        values, row["duration_s"], tolerance_scale=tolerance_scale
                    )
                    abstain_total += int(
                        should_abstain(
                            report,
                            max_hard_failures=args.max_hard_failures,
                            max_soft_failures=args.max_soft_failures,
                        )
                    )
                    total += 1
            rates[kind] = abstain_total / total if total else 0.0
        scale_sweep[f"scale={tolerance_scale:g}"] = rates

    # Barrido de umbrales sobre el corpus (limpio vs corrupto).
    sweep: dict[str, dict[str, float]] = {}
    for max_hard in (0, 1):
        for max_soft in (0, 1, 2):
            rates = {}
            for condition in conditions:
                subset = [r for r in reports if r[0] == condition]
                if not subset:
                    continue
                abstain = sum(
                    1
                    for _, _, report, _, _ in subset
                    if should_abstain(
                        report,
                        max_hard_failures=max_hard,
                        max_soft_failures=max_soft,
                    )
                )
                rates[condition] = abstain / len(subset)
            sweep[f"hard<={max_hard},soft<={max_soft}"] = rates

    output = {
        "subjects": subjects,
        "conditions": conditions,
        "gate": {
            "max_hard_failures": args.max_hard_failures,
            "max_soft_failures": args.max_soft_failures,
        },
        "gate_summary": gate_summary,
        "stress_corruptions": stress,
        "tolerance_scale_sweep": scale_sweep,
        "threshold_sweep": sweep,
        "composed_system": composed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print("\n=== Compuerta de consistencia ===")
    header = (
        f"{'condición':14s} {'n':>5s} {'abstiene':>9s} {'hard':>6s} "
        f"{'soft':>6s} {'score':>7s}"
    )
    print(header)
    for condition, entry in gate_summary.items():
        print(
            f"{condition:14s} {entry['n']:5d} "
            f"{entry['gate_abstention_rate']:9.3f} "
            f"{entry['hard_failures_mean']:6.2f} "
            f"{entry['soft_failures_mean']:6.2f} "
            f"{entry['consistency_score_mean']:7.3f}"
        )
        failures = {
            k: round(v, 3)
            for k, v in entry["check_failure_rate"].items()
            if v > 0
        }
        print(f"    checks con fallos: {failures}")

    print("\n=== Sistema compuesto (LLM + compuerta) ===")
    for name, entry in composed.items():
        llm = entry["llm_only"]
        gated = entry["llm_plus_gate"]
        if entry["decision_mode"] == "stress_binary":
            print(
                f"{name} [{entry['decision_mode']}]: n={entry['n_records']} "
                f"gate_abstain={entry['gate_abstention_rate']:.3f}\n"
                f"  LLM     abst={llm['abstention_rate']:.3f} "
                f"acc={llm['accuracy']:.3f} bal={llm['balanced_accuracy']:.3f} "
                f"recall={llm['recall_stress']:.3f} "
                f"auroc={llm['auroc_stress']:.3f}\n"
                f"  +gate   abst={gated['abstention_rate']:.3f} "
                f"acc={gated['accuracy']:.3f} bal={gated['balanced_accuracy']:.3f} "
                f"recall={gated['recall_stress']:.3f} "
                f"auroc={gated['auroc_stress']:.3f}"
            )
            continue
        print(
            f"{name}: n={entry['n_records']} "
            f"gate_abstain={entry['gate_abstention_rate']:.3f}\n"
            f"  LLM     abst={llm['abstention_rate']:.3f} "
            f"acc={llm['accuracy']:.3f} bal={llm['balanced_accuracy']:.3f} "
            f"bin_acc={llm['binary']['accuracy']:.3f} "
            f"bin_recall={llm['binary']['recall_stress']:.3f}\n"
            f"  +gate   abst={gated['abstention_rate']:.3f} "
            f"acc={gated['accuracy']:.3f} bal={gated['balanced_accuracy']:.3f} "
            f"bin_acc={gated['binary']['accuracy']:.3f} "
            f"bin_recall={gated['binary']['recall_stress']:.3f}"
        )

    print("\n=== Robustez de la compuerta (corrupciones alternativas) ===")
    print(f"{'corrupción':26s} {'abstiene':>9s} {'score':>7s}")
    for kind, entry in stress.items():
        print(
            f"{kind:26s} {entry['gate_abstention_rate']:9.3f} "
            f"{entry['consistency_score_mean']:7.3f}"
        )

    print("\n=== Calibración de la laxitud (tasa de abstención) ===")
    columns = ("C0_clean", "NC_shuffled", "swap_one", "noise_10pct")
    print(f"{'laxitud':12s} " + " ".join(f"{c:>11s}" for c in columns))
    for key, rates in scale_sweep.items():
        print(f"{key:12s} " + " ".join(f"{rates[c]:11.3f}" for c in columns))

    print("\n=== Barrido de umbrales (tasa de abstención) ===")
    conditions_in_sweep = list(sweep[next(iter(sweep))].keys())
    print(f"{'umbral':22s} " + " ".join(f"{c:>12s}" for c in conditions_in_sweep))
    for key, rates in sweep.items():
        print(f"{key:22s} " + " ".join(f"{rates[c]:12.3f}" for c in conditions_in_sweep))

    print(f"\nReporte: {output_path}")


if __name__ == "__main__":
    main()
