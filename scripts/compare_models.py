#!/usr/bin/env python
"""Comparación de modelos LLM sobre los mismos artefactos experimentales.

Lee las respuestas de cada modelo (aisladas en su subdirectorio) y produce una
tabla comparable por condición, con **IC bootstrap agrupado por sujeto** para la
diferencia de accuracy respecto al primer modelo.

Modos:

- ``degradation`` (por defecto): barrido C0–C5, directorios
  ``degradation/{modelo}/`` y ``degradation_text/{modelo}/``.
- ``phase2``: respuestas de ``PHASE2`` (``llm_responses/{modelo}/``), incluidas
  las condiciones negativas ``NC_*``.

Las ejecuciones anteriores al aislamiento por modelo (sin subdirectorio) se
leen de la carpeta histórica con un aviso.

Uso:
    python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b
    python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b \\
        --representation text --conditions C0_clean,C4_total_eda
    python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b --mode phase2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sources.common.provenance import utc_now  # noqa: E402
from sources.phase2.llm_config import model_slug  # noqa: E402

DEFAULT_CONDITIONS = ("C0_clean", "C1_partial_resp", "C2_partial_eda",
                      "C4_total_eda", "C5_total_both")
DEFAULT_OUTPUT = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / (
    "model_comparison.json"
)


def _model_dir(base: Path, model: str, quiet: bool = False) -> Path:
    """Subdirectorio del modelo, con caída a la carpeta histórica."""
    candidate = base / model_slug(model)
    if candidate.is_dir():
        return candidate
    if base.is_dir() and any(base.glob("*.responses.jsonl")):
        if not quiet:
            print(
                f"[AVISO] No existe {candidate}; se usa la carpeta histórica {base}."
            )
        return base
    raise SystemExit(
        f"No hay respuestas para {model} en {candidate}. "
        f"Ejecuta primero el experimento con --model {model}."
    )


def _base_dir(mode: str, representation: str) -> Path:
    """Directorio raíz de los datos del modo pedido."""
    if mode == "degradation":
        name = "degradation" if representation == "numeric" else "degradation_text"
    else:
        name = "llm_responses"
    return REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / name


def check_distinct_models(
    models: Sequence[str], mode: str, representation: str
) -> None:
    """Falla si dos modelos apuntan al mismo directorio de datos.

    Es la red de seguridad para el caso del job 2703 (2026-11-13): el nombre
    ``qwen3:14b`` llegó truncado como ``qwen3`` y ambos modelos cayeron en la
    carpeta histórica, así que se comparaba un corpus consigo mismo.
    """
    base = _base_dir(mode, representation)
    seen: dict[Path, str] = {}
    for model in models:
        directory = _model_dir(base, model, quiet=True)
        if directory in seen:
            raise SystemExit(
                f"Los modelos '{seen[directory]}' y '{model}' apuntan al mismo "
                f"directorio ({directory}). Suele ser un nombre truncado o mal "
                "copiado en --models: revisa la lista."
            )
        seen[directory] = model


def _models_found(records: Iterable[Mapping[str, Any]]) -> set[str]:
    found = {
        record.get("model") or (record.get("ollama_meta") or {}).get("model")
        for record in records
    }
    found.discard(None)
    return {str(m) for m in found}


# ---------------------------------------------------------------------------
# Carga por modo
# ---------------------------------------------------------------------------
def load_degradation(
    model: str,
    representation: str,
    conditions: set[str],
) -> tuple[list[dict[str, Any]], Any]:
    """Registros del barrido de degradación y la función de métricas."""
    from analyze_degradation import analyze_group, load_labels, load_records

    base = _base_dir("degradation", representation)
    directory = _model_dir(base, model)
    labels = load_labels()
    records = load_records(directory, conditions, split="evaluation")
    from sources.phase3.recovery import apply_recovery

    records, _ = apply_recovery(records)
    return records, (lambda rows: analyze_group(rows, labels))


def load_phase2(
    model: str,
    conditions: set[str],
) -> tuple[list[dict[str, Any]], Any]:
    """Registros de PHASE2 (IDs opacos resueltos) y la función de métricas."""
    from analyze_degradation import analyze_group, load_labels
    from sources.phase3.evaluator import _opaque_window_id, load_responses

    base = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "llm_responses"
    directory = _model_dir(base, model)
    labels = load_labels()
    opaque_to_window = {_opaque_window_id(w): w for w in labels}

    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.responses.jsonl")):
        for response in load_responses(path):
            if response.get("condition_id") not in conditions:
                continue
            if response.get("split") not in (None, "evaluation"):
                continue
            window_id = opaque_to_window.get(response.get("window_id_opaque", ""))
            if window_id is None:
                continue
            records.append({**response, "window_id": window_id})
    return records, (lambda rows: analyze_group(rows, labels))


# ---------------------------------------------------------------------------
# Métricas y bootstrap
# ---------------------------------------------------------------------------
def _truth_pred(record: Mapping[str, Any]) -> tuple[str, str] | None:
    """(verdad, predicción) del registro ya normalizado."""
    truth = record.get("_truth")
    decision = record.get("decision")
    if truth is None or decision is None:
        return None
    return str(truth), str(decision)


def per_subject_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Aciertos y decididas por sujeto (para bootstrap agrupado)."""
    counts: dict[str, list[int]] = {}
    for record in records:
        pair = _truth_pred(record)
        if pair is None:
            continue
        truth, predicted = pair
        if predicted == "abstain":
            continue
        entry = counts.setdefault(str(record.get("subject")), [0, 0])
        entry[1] += 1
        entry[0] += int(predicted == truth)
    return {k: (v[0], v[1]) for k, v in counts.items()}


def paired_bootstrap_delta(
    counts_a: Mapping[str, tuple[int, int]],
    counts_b: Mapping[str, tuple[int, int]],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Diferencia de accuracy (B − A) con IC bootstrap por sujeto.

    Se remuestrean **sujetos** con reemplazo y se recalcula la accuracy pooled
    de cada modelo sobre la misma muestra (diseño pareado por sujeto).
    """
    subjects = sorted(set(counts_a) & set(counts_b))
    if not subjects:
        return {"delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_subjects": 0}

    def accuracy(counts: Mapping[str, tuple[int, int]], sample: Sequence[str]) -> float:
        hits = sum(counts[s][0] for s in sample)
        total = sum(counts[s][1] for s in sample)
        return hits / total if total else float("nan")

    delta = accuracy(counts_b, subjects) - accuracy(counts_a, subjects)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        sample = [subjects[i] for i in rng.integers(0, len(subjects), len(subjects))]
        value = accuracy(counts_b, sample) - accuracy(counts_a, sample)
        if not np.isnan(value):
            samples.append(value)
    if not samples:
        return {"delta": delta, "ci_low": float("nan"),
                "ci_high": float("nan"), "n_subjects": len(subjects)}
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "delta": float(delta),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_subjects": len(subjects),
    }


def summarize(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Extrae las métricas comparables de un bloque de métricas."""
    claims = metrics.get("claims") or {}
    certainty = metrics.get("certainty") or {}
    n_claims = claims.get("n_claims", claims.get("n_claims_total", 0)) or 0

    def _rate(key: str) -> float:
        value = claims.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0
        return float(value)

    evaluable = round(
        n_claims * (1.0 - _rate("indeterminate_rate") - _rate("unsupported_rate"))
    )
    return {
        "n": metrics.get("n", 0),
        "n_valid": metrics.get("n_valid", metrics.get("n_decided", 0)),
        "n_invalid": metrics.get("n_invalid", 0),
        "n_decided": metrics.get("n_decided", 0),
        "abstention_rate": metrics.get("abstention_rate", float("nan")),
        "accuracy": metrics.get("accuracy", float("nan")),
        "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
        "ece": metrics.get("ece", float("nan")),
        "claims_n": n_claims,
        "claims_evaluable": evaluable,
        "claims_supported": claims.get("supported_rate", float("nan")),
        "claims_contradicted": claims.get("contradicted_rate", float("nan")),
        "claims_indeterminate_rate": _rate("indeterminate_rate"),
        "certainty_index": certainty.get("certainty_index_mean", float("nan")),
        "confidence_mean": metrics.get("confidence_mean", float("nan")),
    }


def compare(
    models: Sequence[str],
    mode: str,
    representation: str,
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Ejecuta la comparación y devuelve el informe."""
    from analyze_degradation import load_labels

    labels = load_labels()
    condition_set = set(conditions)
    tables: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    counts: dict[tuple[str, str], dict[str, tuple[int, int]]] = {}
    models_in_data: dict[str, list[str]] = {}

    for model in models:
        if mode == "phase2":
            records, evaluator = load_phase2(model, condition_set)
        else:
            records, evaluator = load_degradation(model, representation, condition_set)

        found = sorted(_models_found(records))
        models_in_data[model] = found
        if found and model not in found:
            raise SystemExit(
                f"Los datos de {model} contienen modelos {found}. "
                "Revisa los directorios: puede ser una carpeta histórica compartida."
            )

        for condition in conditions:
            rows = [r for r in records if r.get("condition_id") == condition]
            labels = load_labels()
            for row in rows:
                row["_truth"] = labels.get(row.get("window_id"))
            if not rows:
                continue
            entry = {"model": model, **summarize(evaluator(rows))}
            tables[condition].append(entry)
            counts[(model, condition)] = per_subject_counts(rows)

    deltas: dict[str, dict[str, Any]] = {}
    baseline = models[0]
    for condition in conditions:
        if not tables[condition]:
            continue
        deltas[condition] = {}
        for model in models[1:]:
            key_a = (baseline, condition)
            key_b = (model, condition)
            if key_a not in counts or key_b not in counts:
                continue
            deltas[condition][f"{model} - {baseline}"] = paired_bootstrap_delta(
                counts[key_a], counts[key_b]
            )

    return {
        "generated_at": utc_now(),
        "mode": mode,
        "representation": representation if mode != "phase2" else "llm_views",
        "baseline_model": baseline,
        "models": list(models),
        "models_in_data": models_in_data,
        "conditions": list(conditions),
        "tables": tables,
        "deltas": deltas,
    }


def print_report(report: Mapping[str, Any]) -> None:
    """Tabla legible por consola."""
    print(
        f"\n=== Comparación de modelos ({report['mode']}, "
        f"{report['representation']}) ==="
    )
    print(f"Modelos: {', '.join(report['models'])}")
    for model, found in report["models_in_data"].items():
        print(f"  {model}: datos de {found}")
    print(
        f"\n{'condición':16s} {'modelo':16s} {'n':>5s} {'dec':>5s} {'abst':>6s} {'acc*':>6s} "
        f"{'bal*':>6s} {'ECE*':>6s} {'sup':>6s} {'contra':>7s} {'certeza':>8s}"
    )
    for condition in report["conditions"]:
        for entry in report["tables"].get(condition, []):
            if entry.get("claims_evaluable"):
                support = f"{entry['claims_supported']:6.3f}"
                contra = f"{entry['claims_contradicted']:7.3f}"
            else:
                support, contra = "   n/a", "    n/a"
            decided = int(round(entry["n"] * (1.0 - entry["abstention_rate"])))
            print(
                f"{condition:16s} {entry['model']:16s} {entry['n']:5d} "
                f"{decided:5d} "
                f"{entry['abstention_rate']:6.3f} {entry['accuracy']:6.3f} "
                f"{entry['balanced_accuracy']:6.3f} {entry['ece']:6.3f} "
                f"{support} {contra} "
                f"{entry['certainty_index']:8.2f}  ({entry['claims_n']} claims)"
            )
        print()

    print(
        "  * acc, bal y ECE se calculan solo sobre las DECIDIDAS. Con abstención "
        "alta (mirar 'dec') no son comparables:\n"
        "    un modelo que se abstiene en el 99 % puede mostrar acc 1.000 sobre "
        "una sola ventana."
    )

    if report["deltas"]:
        print("\nDiferencias de accuracy (bootstrap por sujeto, IC 95 %):")
        for condition, entries in report["deltas"].items():
            for label, stats in entries.items():
                print(
                    f"  {condition:16s} {label:34s} "
                    f"{stats['delta']:+.3f} "
                    f"[{stats['ci_low']:+.3f}; {stats['ci_high']:+.3f}] "
                    f"({stats['n_subjects']} sujetos)"
                )
        print(
            "  Solo se remuestrean los sujetos con ventanas decididas en ambos "
            "modelos: con 1 sujeto (o 0) el intervalo no es inferencia."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        required=True,
        help="Lista de modelos separada por comas; el primero es la referencia.",
    )
    parser.add_argument(
        "--mode",
        choices=["degradation", "phase2"],
        default="degradation",
    )
    parser.add_argument(
        "--representation",
        choices=["numeric", "text"],
        default="numeric",
        help="Brazo de representación (solo modo degradation).",
    )
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) < 2:
        raise SystemExit("Se necesitan al menos dos modelos para comparar.")
    check_distinct_models(models, args.mode, args.representation)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    report = compare(models, args.mode, args.representation, conditions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"\nInforme: {output}")


if __name__ == "__main__":
    main()
