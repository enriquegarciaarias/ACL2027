#!/usr/bin/env python
"""P3 — Control de generalidad con PhysioNet/CinC 2017 (ritmo en ECG).

- 4 clases con etiqueta de experto: normal, AF, other-rhythm y **noisy**
  (evidencia naturalmente inutilizable).
- Split determinista: 30 % desarrollo (referencia poblacional + modelo de
  referencia), 70 % evaluación.
- Representación: z poblacional de 5 features de ECG; prompt sin few-shot.
- Se guardan por separado respuestas (sin etiqueta) y etiquetas.

Uso:
    python scripts/run_p3_physionet.py --per-class 10      # piloto
    python scripts/run_p3_physionet.py                     # 600 registros
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

from sources.common.provenance import utc_now
from sources.external.physionet2017 import (
    CANONICAL_FEATURES,
    RHYTHM_CLASSES,
    RhythmRecord,
    build_payload,
    extract_features,
    load_reference,
    population_reference,
    read_record,
    select_records,
    split_records,
    z_scores,
)
from sources.phase2.llm_config import build_experiment_config, create_ollama_chat_client
from sources.phase2.orchestrator import _call_ollama_chat
from sources.phase2.prompt_builder import (
    assert_no_leakage,
    build_rhythm_prompt,
    render_rhythm_examples,
)
from sources.phase2.response_schema import DECISION_MODE_RHYTHM4, validate_llm_output
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.claims import summarize_verdicts, verify_claims

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/tmp/opencode/physionet2017/training2017")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "output" / "PHYSIONET2017" / "P3"

DATASET_NOTE = (
    "Dataset note: this is a single-lead ECG recording (AliveCor device, "
    "300 Hz) lasting 9 to 61 seconds. Only ECG-derived features are available: "
    "there is no electrodermal activity, respiration or chest ECG."
)


def load_records(
    data_dir: Path,
    labels: dict[str, str],
    record_ids: list[str],
    cache_path: Path,
) -> list[RhythmRecord]:
    """Lee y extrae features de cada registro, con caché en disco."""
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text())
    records: list[RhythmRecord] = []
    for index, record_id in enumerate(record_ids, start=1):
        if record_id in cache:
            entry = cache[record_id]
        else:
            signal, fs = read_record(record_id, data_dir)
            entry = {
                "label": labels[record_id],
                "fs": fs,
                "duration_s": float(signal.size / fs),
                "values": extract_features(signal, fs),
            }
            cache[record_id] = entry
            print(
                f"[PN17] features {index}/{len(record_ids)} {record_id} "
                f"({entry['label']}, {entry['duration_s']:.0f} s)"
            )
        records.append(
            RhythmRecord(
                record_id=record_id,
                label=entry["label"],
                fs=float(entry["fs"]),
                duration_s=float(entry["duration_s"]),
                values=entry["values"],
            )
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    return records


def reference_model(
    dev: list[RhythmRecord],
    test: list[RhythmRecord],
    centers: dict[str, float],
    scales: dict[str, float],
) -> dict[str, Any]:
    """Modelo de referencia entrenado en desarrollo y evaluado en test."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def matrix(records: list[RhythmRecord]) -> np.ndarray:
        rows = []
        for record in records:
            scores = z_scores(record.values, centers, scales)
            rows.append(
                [
                    float("nan") if scores[name] is None else scores[name]
                    for name in CANONICAL_FEATURES
                ]
            )
        return np.asarray(rows, dtype=float)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(matrix(dev), [r.label for r in dev])
    predictions = model.predict(matrix(test))
    truth = [r.label for r in test]
    return {
        "n_train": len(dev),
        "n_test": len(test),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "predictions": dict(zip([r.record_id for r in test], predictions)),
    }


def class_exemplars(
    dev: list[RhythmRecord],
    centers: dict[str, float],
    scales: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Mediana de z por clase en desarrollo (para few-shot)."""
    exemplars: dict[str, dict[str, float]] = {}
    for label in RHYTHM_CLASSES:
        rows = [r for r in dev if r.label == label]
        if not rows:
            continue
        values: dict[str, float] = {}
        for name in CANONICAL_FEATURES:
            samples = [
                z_scores(r.values, centers, scales)[name] for r in rows
            ]
            clean = [v for v in samples if v is not None]
            if clean:
                values[name] = float(np.median(clean))
        exemplars[label] = values
    return exemplars


def run_llm(
    records: list[RhythmRecord],
    output_dir: Path,
    centers: dict[str, float],
    scales: dict[str, float],
    model: str,
    temperature: float,
    seed: int,
    max_records: int | None,
    few_shot: str | None = None,
    tag: str = "",
) -> list[dict[str, Any]]:
    suffix = f"_{tag}" if tag else ""
    responses_path = output_dir / f"physionet2017{suffix}.responses.jsonl"
    labels_path = output_dir / f"physionet2017{suffix}.labels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if responses_path.is_file():
        for line in responses_path.open(encoding="utf-8"):
            record = json.loads(line)
            existing[record["record_id"]] = record

    labels = [
        {
            "record_id": record.record_id,
            "label": record.label,
            "duration_s": record.duration_s,
        }
        for record in records
    ]
    with labels_path.open("w", encoding="utf-8") as fh:
        for entry in labels:
            fh.write(json.dumps(entry) + "\n")

    config = build_experiment_config(
        model=model, temperature=temperature, seed=seed,
        max_retries=3, cascade_enabled=False,
    )
    client = create_ollama_chat_client(config)

    pending = [r for r in records if r.record_id not in existing]
    if max_records is not None:
        pending = pending[:max_records]
    if not pending:
        print(f"[PN17] Todas las respuestas ya existen ({len(existing)}).")

    for index, record in enumerate(pending, start=1):
        scores = z_scores(record.values, centers, scales)
        payload = build_payload(record.record_id, scores, record.duration_s)
        prompt = build_rhythm_prompt(
            payload=payload,
            window_id_opaque=payload["window"]["window_id"],
            duration_s=record.duration_s,
            num_ctx=config.num_ctx,
            dataset_note=DATASET_NOTE,
            few_shot_examples=few_shot,
        )
        assert_no_leakage(prompt)
        try:
            raw_output, ollama_meta = _call_ollama_chat(client, prompt, config, seed=seed)
            validated = validate_llm_output(raw_output, decision_mode=DECISION_MODE_RHYTHM4)
            status, decision = validated.status, validated.decision
            confidence = validated.confidence
            claims = [c.to_dict() for c in validated.claims]
            reasoning, error = validated.reasoning, validated.error_message
        except Exception as exc:  # noqa: BLE001
            raw_output, ollama_meta = "", {}
            status, decision, confidence = "timeout", None, None
            claims, reasoning, error = [], "", str(exc)

        result = {
            "record_id": record.record_id,
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
        with responses_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        existing[record.record_id] = result
        print(
            f"[PN17] {index}/{len(pending)} {record.record_id}: {status} "
            f"decision={decision} conf={confidence}"
        )

    return [existing[r.record_id] for r in records if r.record_id in existing]


def evaluate(
    responses: list[dict[str, Any]],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Corrección, abstención, claims y certeza para el subconjunto evaluado."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    valid = [r for r in responses if r["status"] in ("valid_response", "valid_abstention")]
    invalid = len(responses) - len(valid)
    abstain = [r for r in valid if r["decision"] == "abstain"]
    decided = [r for r in valid if r["decision"] != "abstain"]

    truth = [labels[r["record_id"]] for r in decided]
    predicted = [r["decision"] for r in decided]
    accuracy = float(accuracy_score(truth, predicted)) if decided else float("nan")
    balanced = float(balanced_accuracy_score(truth, predicted)) if decided else float("nan")

    verdicts = []
    certainties = []
    for record in valid:
        if record.get("claims"):
            verdicts.extend(verify_claims(record["claims"], record["prompt"]))
        certainty = linguistic_certainty(record.get("reasoning", "") or "")
        if certainty["n_words"]:
            certainties.append(certainty)

    per_class = {}
    for label in RHYTHM_CLASSES:
        rows = [r for r in decided if labels[r["record_id"]] == label]
        hits = sum(1 for r in rows if r["decision"] == label)
        per_class[label] = {
            "n": len(rows),
            "recall": hits / len(rows) if rows else float("nan"),
            "abstentions": sum(
                1
                for r in abstain
                if labels[r["record_id"]] == label
            ),
        }

    confusion = {
        label: {
            predicted_label: sum(
                1
                for r in valid
                if labels[r["record_id"]] == label and r["decision"] == predicted_label
            )
            for predicted_label in list(RHYTHM_CLASSES) + ["abstain"]
        }
        for label in RHYTHM_CLASSES
    }

    return {
        "n": len(responses),
        "n_valid": len(valid),
        "n_invalid": invalid,
        "n_abstain": len(abstain),
        "abstention_rate": len(abstain) / len(valid) if valid else float("nan"),
        "n_decided": len(decided),
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "decisions": dict(Counter(r["decision"] for r in valid)),
        "per_class": per_class,
        "confusion": confusion,
        "claims": summarize_verdicts(verdicts),
        "certainty": {
            "certainty_index_mean": float(
                np.mean([c["certainty_index"] for c in certainties])
            )
            if certainties
            else float("nan"),
            "hedges_mean": float(np.mean([c["hedges"] for c in certainties]))
            if certainties
            else float("nan"),
            "boosters_mean": float(np.mean([c["boosters"] for c in certainties]))
            if certainties
            else float("nan"),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--dev-fraction", type=float, default=0.3)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--eval-split",
        choices=["test", "dev"],
        default="test",
        help="Conjunto a evaluar con el LLM (dev solo para ablaciones).",
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Añade 4 ejemplos (uno por clase) construidos con registros de desarrollo.",
    )
    parser.add_argument("--tag", default="", help="Sufijo de los ficheros de salida.")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    reference_path = data_dir / "REFERENCE.csv"
    if not reference_path.is_file():
        raise SystemExit(
            f"No existe {reference_path}. Descarga training2017.zip de PhysioNet "
            "y descomprímelo en --data-dir."
        )

    labels = load_reference(reference_path)
    selected = select_records(labels, per_class=args.per_class, seed=args.seed)
    dev_ids, test_ids = split_records(selected, dev_fraction=args.dev_fraction)
    print(
        f"Registros seleccionados: {len(selected)} "
        f"(dev={len(dev_ids)}, test={len(test_ids)}) | "
        f"clases: {dict(Counter(labels[r] for r in selected))}"
    )

    output_dir = Path(args.output_dir)
    cache_path = output_dir / "physionet2017.features.json"
    dev = load_records(data_dir, labels, dev_ids, cache_path)
    test = load_records(data_dir, labels, test_ids, cache_path)

    centers, scales = population_reference(dev)
    report: dict[str, Any] = {
        "dataset": "PHYSIONET2017",
        "task": "4-class rhythm (normal/af/other-rhythm/noisy)",
        "n_selected": len(selected),
        "n_dev": len(dev),
        "n_test": len(test),
        "classes": dict(Counter(labels[r] for r in selected)),
        "dataset_note": DATASET_NOTE,
        "population_reference": {"centers": centers, "scales": scales},
    }

    reference = reference_model(dev, test, centers, scales)
    report["reference"] = {
        "n_train": reference["n_train"],
        "n_test": reference["n_test"],
        "accuracy": reference["accuracy"],
        "balanced_accuracy": reference["balanced_accuracy"],
    }
    print(
        f"[PN17] Referencia features-only (train dev): "
        f"acc={reference['accuracy']:.3f} bal={reference['balanced_accuracy']:.3f}"
    )

    few_shot_text = None
    if args.few_shot:
        few_shot_text = render_rhythm_examples(
            class_exemplars(dev, centers, scales)
        )
        print("[PN17] few-shot construido con medianas de desarrollo")

    eval_records = dev if args.eval_split == "dev" else test
    file_tag = args.tag or (
        "dev_fs" if (args.eval_split == "dev" and args.few_shot)
        else "dev_nofs" if args.eval_split == "dev"
        else "test_fs" if args.few_shot
        else ""
    )
    responses = run_llm(
        eval_records,
        output_dir=output_dir,
        centers=centers,
        scales=scales,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_records=args.max_records,
        few_shot=few_shot_text,
        tag=file_tag,
    )
    metrics = evaluate(responses, labels)
    report["llm"] = metrics

    print("\n=== PhysioNet 2017 (evaluación) ===")
    print(
        f"n={metrics['n']} válidas={metrics['n_valid']} inválidas={metrics['n_invalid']} "
        f"abstain={metrics['n_abstain']} decididas={metrics['n_decided']}"
    )
    print(
        f"  accuracy={metrics['accuracy']:.3f} balanced={metrics['balanced_accuracy']:.3f} "
        f"abstención={metrics['abstention_rate']:.3f}"
    )
    print(f"  decisiones: {metrics['decisions']}")
    print(f"  por clase: {metrics['per_class']}")
    print(
        f"  claims: respaldadas={metrics['claims']['supported_rate']:.3f} "
        f"contradichas={metrics['claims']['contradicted_rate']:.3f} "
        f"no respaldadas={metrics['claims']['unsupported_rate']:.3f}"
    )
    print(f"  certeza: {metrics['certainty']['certainty_index_mean']:.2f}")

    summary_path = output_dir / f"physionet2017{('_' + file_tag) if file_tag else ''}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nResumen: {summary_path}")


if __name__ == "__main__":
    main()
