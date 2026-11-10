#!/usr/bin/env python
"""P3 — Validación externa con K-EmoCon: arousal y valencia (alta vs baja).

K-EmoCon **no** tiene etiqueta de estrés: sus constructos afectivos son
valencia y arousal (autoinforme, pareja y observadores externos), disponibles
solo durante el periodo ``debate``. Este script valida el LLM contra esas
etiquetas, no contra el protocolo:

- Se agregan las ventanas de 5 s a ventanas de 60 s (``sources/external/kemocon``).
- Se binariza la etiqueta afectiva por mediana (por participante por defecto).
- Se estandarizan las features intra-participante (E12) y se consulta al LLM
  con un prompt de arousal (``high-arousal`` / ``low-arousal``) o de valencia.
- Las etiquetas se guardan aparte; las respuestas no las contienen.

Uso:
    python scripts/run_p3_kemocon.py --subjects 1,5,8,10 --reference
    python scripts/run_p3_kemocon.py --task valence --source self_valence
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
from sources.external.kemocon import (
    KEMOCON_FEATURES,
    KEMOCON_REFERENCE_EXTRA,
    KemoconWindow,
    aggregate_windows,
    assign_binary_affect_label,
    build_payload,
    load_feature_rows,
    load_period_assignments,
    opaque_window_id,
    subject_centered,
)
from sources.phase2.llm_config import build_experiment_config, create_ollama_chat_client
from sources.phase2.orchestrator import _call_ollama_chat
from sources.phase2.prompt_builder import assert_no_leakage, build_numeric_prompt
from sources.phase2.response_schema import (
    DECISION_MODE_AROUSAL_BINARY,
    DECISION_MODE_VALENCE_BINARY,
    validate_llm_output,
)
from sources.phase3.records import evaluate_binary_records

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PSRI = Path("/home/vant/Documentos/proyectos/LSI/PSRI/results/output/KEMOCON")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "output" / "KEMOCON" / "P3"

DATASET_NOTE = (
    "Dataset note: this recording provides wrist electrodermal activity, heart "
    "rate, inter-beat interval and skin temperature. Chest ECG, respiration and "
    "skin-conductance response counts or amplitudes are not available; base the "
    "decision only on the channels listed above."
)

TASKS: dict[str, dict[str, Any]] = {
    "arousal": {
        "decision_mode": DECISION_MODE_AROUSAL_BINARY,
        "default_source": "self_arousal",
        "positive": "high-arousal",
        "negative": "low-arousal",
    },
    "valence": {
        "decision_mode": DECISION_MODE_VALENCE_BINARY,
        "default_source": "self_valence",
        "positive": "high-valence",
        "negative": "low-valence",
    },
}


def load_windows(
    psri_dir: Path,
    subjects: list[str] | None,
    min_coverage: float,
    scaling: str,
) -> list[KemoconWindow]:
    rows = load_feature_rows(psri_dir / "kemocon_feature_table.csv")
    assignments = load_period_assignments(psri_dir / "kemocon_period_assignments.csv")
    windows = aggregate_windows(
        rows, assignments, min_coverage=min_coverage, subjects=subjects
    )
    subject_centered(windows, features=list(KEMOCON_FEATURES), method=scaling)
    return windows


def reference_model(
    windows: list[KemoconWindow],
    feature_names: list[str],
    positive_label: str,
    negative_label: str,
) -> dict[str, Any]:
    """Regresión logística de referencia con CV agrupada por participante."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labelled = [w for w in windows if w.affect_label in (positive_label, negative_label)]
    matrix = np.array(
        [
            [
                window.zscores.get(name) if window.zscores.get(name) is not None else np.nan
                for name in feature_names
            ]
            for window in labelled
        ],
        dtype=float,
    )
    y = np.array([1 if w.affect_label == positive_label else 0 for w in labelled])
    groups = np.array([w.subject for w in labelled])

    n_splits = min(5, len(set(groups))) if len(set(groups)) > 1 else 1
    if n_splits < 2:
        return {"n": int(len(labelled)), "n_splits": n_splits, "error": "un solo participante"}

    splitter = GroupKFold(n_splits=n_splits)
    balanced, aurocs = [], []
    for train_idx, test_idx in splitter.split(matrix, y, groups):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(matrix[train_idx], y[train_idx])
        balanced.append(balanced_accuracy_score(y[test_idx], model.predict(matrix[test_idx])))
        aurocs.append(roc_auc_score(y[test_idx], model.decision_function(matrix[test_idx])))

    return {
        "n": int(len(labelled)),
        "n_splits": int(n_splits),
        "balanced_accuracy_mean": float(np.mean(balanced)) if balanced else float("nan"),
        "balanced_accuracy_std": float(np.std(balanced)) if balanced else float("nan"),
        "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
        "features": list(feature_names),
        "features_kind": "within_subject_zscore",
    }


def run_llm(
    windows: list[KemoconWindow],
    output_dir: Path,
    decision_mode: str,
    model: str,
    temperature: float,
    seed: int,
    graded_confidence: bool,
    max_windows: int | None,
    tag: str,
) -> list[dict[str, Any]]:
    suffix = f"_{tag}" if tag else ""
    responses_path = output_dir / f"kemocon{suffix}.responses.jsonl"
    labels_path = output_dir / f"kemocon{suffix}.labels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if responses_path.is_file():
        for line in responses_path.open(encoding="utf-8"):
            record = json.loads(line)
            existing[record["window_id"]] = record

    config = build_experiment_config(
        model=model, temperature=temperature, seed=seed,
        max_retries=3, cascade_enabled=False,
    )
    client = create_ollama_chat_client(config)

    labels = [
        {
            "window_id": window.window_id,
            "window_id_opaque": opaque_window_id(window.window_id),
            "subject": window.subject,
            "period": window.period,
            "label": window.affect_label,
            "affect_score": window.affect_score,
            "duration_s": window.duration_s,
            "n_subwindows": window.n_subwindows,
        }
        for window in windows
    ]

    pending = [w for w in windows if w.window_id not in existing]
    if max_windows is not None:
        pending = pending[:max_windows]
    if not pending:
        print(f"[P3] Todas las ventanas ya tienen respuesta ({len(existing)}).")

    for index, window in enumerate(pending, start=1):
        payload = build_payload(window, features=KEMOCON_FEATURES, include_motion=False)
        prompt = build_numeric_prompt(
            payload=payload,
            window_id_opaque=opaque_window_id(window.window_id),
            duration_s=window.duration_s,
            num_ctx=config.num_ctx,
            use_few_shot=False,
            include_additional=True,
            standardized=True,
            decision_mode=decision_mode,
            graded_confidence=graded_confidence,
            dataset_note=DATASET_NOTE,
        )
        assert_no_leakage(prompt)
        try:
            raw_output, ollama_meta = _call_ollama_chat(client, prompt, config, seed=seed)
            validated = validate_llm_output(raw_output, decision_mode=decision_mode)
            status, decision = validated.status, validated.decision
            confidence = validated.confidence
            claims = [c.to_dict() for c in validated.claims]
            reasoning, error = validated.reasoning, validated.error_message
        except Exception as exc:  # noqa: BLE001
            raw_output, ollama_meta = "", {}
            status, decision, confidence = "timeout", None, None
            claims, reasoning, error = [], "", str(exc)

        record = {
            "subject": window.subject,
            "period": window.period,
            "window_id": window.window_id,
            "window_id_opaque": opaque_window_id(window.window_id),
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
        with responses_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing[window.window_id] = record
        print(
            f"[P3] {index}/{len(pending)} {window.window_id}: {status} "
            f"decision={decision} conf={confidence}"
        )

    with labels_path.open("w", encoding="utf-8") as fh:
        for label in labels:
            fh.write(json.dumps(label, ensure_ascii=False) + "\n")

    return [existing[w.window_id] for w in windows if w.window_id in existing]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psri-dir", default=str(DEFAULT_PSRI))
    parser.add_argument("--subjects", default="", help="IDs de participantes separados por comas.")
    parser.add_argument("--task", choices=sorted(TASKS), default="arousal")
    parser.add_argument(
        "--source",
        default="",
        help="Etiqueta afectiva: self_arousal, external_arousal, self_valence, "
        "external_valence, partner_arousal, partner_valence.",
    )
    parser.add_argument(
        "--split",
        choices=["within_subject", "global"],
        default="within_subject",
        help="Mediana por participante (default) o global.",
    )
    parser.add_argument("--scaling", choices=["robust", "zscore"], default="robust")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--graded-confidence", action="store_true")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--tag", default="", help="Sufijo para los ficheros de salida.")
    args = parser.parse_args(argv)

    task = TASKS[args.task]
    source = args.source or task["default_source"]
    positive, negative = task["positive"], task["negative"]
    decision_mode = task["decision_mode"]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] or None
    output_dir = Path(args.output_dir)
    psri_dir = Path(args.psri_dir)
    if not psri_dir.is_dir():
        raise SystemExit(f"No existe el directorio de PSRI: {psri_dir}")

    windows = load_windows(
        psri_dir,
        subjects=subjects,
        min_coverage=args.min_coverage,
        scaling=args.scaling,
    )
    labelled_count = assign_binary_affect_label(
        windows,
        source=source,
        split=args.split,
        positive_label=positive,
        negative_label=negative,
        period="debate",
    )
    usable = [w for w in windows if w.affect_label in (positive, negative)]
    print(
        f"Ventanas de 60 s: {len(windows)} totales; {labelled_count} con etiqueta "
        f"{source} ({args.split}); {len(usable)} usables "
        f"({positive}={sum(1 for w in usable if w.affect_label == positive)}, "
        f"{negative}={sum(1 for w in usable if w.affect_label == negative)}) "
        f"de {len({w.subject for w in usable})} participantes"
    )
    if not usable:
        raise SystemExit("No hay ventanas etiquetadas; revisa la fuente afectiva.")

    file_tag = args.tag or args.task
    report: dict[str, Any] = {
        "dataset": "KEMOCON",
        "task": args.task,
        "source": source,
        "split": args.split,
        "positive_label": positive,
        "negative_label": negative,
        "scaling": args.scaling,
        "n_windows_total": len(windows),
        "n_windows_labelled": len(usable),
        "n_subjects": len({w.subject for w in usable}),
        "model": args.model,
        "graded_confidence": args.graded_confidence,
        "note": "K-EmoCon no tiene etiqueta de estrés; se validan arousal/valencia.",
    }

    if args.reference:
        report["reference"] = reference_model(
            windows, list(KEMOCON_FEATURES), positive, negative
        )
        ref = report["reference"]
        if "error" not in ref:
            print(
                f"[P3] Referencia (z, CV por participante): "
                f"balanced={ref['balanced_accuracy_mean']:.3f} ± "
                f"{ref['balanced_accuracy_std']:.3f}, AUROC={ref['auroc_mean']:.3f}"
            )

    if args.reference_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"kemocon_{file_tag}_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nResumen (solo referencia): {summary_path}")
        return

    records = run_llm(
        usable,
        output_dir=output_dir,
        decision_mode=decision_mode,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        graded_confidence=args.graded_confidence,
        max_windows=args.max_windows,
        tag=file_tag,
    )

    labels = {
        json.loads(line)["window_id_opaque"]: json.loads(line)["label"]
        for line in (output_dir / f"kemocon_{file_tag}.labels.jsonl").open(encoding="utf-8")
    }
    metrics = evaluate_binary_records(
        records, labels, positive_label=positive, negative_label=negative
    )
    report["llm"] = metrics
    print(f"\n=== P3 K-EmoCon — {args.task} ({source}, split={args.split}) ===")
    print(
        f"n={metrics['n']} válidas={metrics['n_valid']} inválidas={metrics['n_invalid']} "
        f"abstain={metrics['n_abstain']} decididas={metrics['n_decided']} "
        f"cobertura={metrics['coverage']:.3f}"
    )
    print(
        f"  accuracy={metrics['accuracy']:.3f} balanced={metrics['balanced_accuracy']:.3f} "
        f"precision={metrics['precision_stress']:.3f} recall={metrics['recall_stress']:.3f} "
        f"F1={metrics['f1_stress']:.3f}"
    )
    print(
        f"  AUROC={metrics['auroc_stress']:.3f} ECE={metrics['ece']:.3f} "
        f"Brier={metrics['brier']:.3f} decisiones={metrics['decisions']}"
    )
    print(f"  confusión real->predicho: {metrics['confusion']}")

    summary_path = output_dir / f"kemocon_{file_tag}_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\nResumen: {summary_path}")


if __name__ == "__main__":
    main()
