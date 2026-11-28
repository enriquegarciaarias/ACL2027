#!/usr/bin/env python
"""Análisis de repeticiones de prompt para DREAMT y PhysioNet 2017.

Responde a la misma pregunta que ``analyze_repetitions.py`` (WESAD) pero con
otro vocabulario de etiquetas: si se envía **el mismo prompt** ``R`` veces
cambiando solo la semilla, ¿cuánto cambia la respuesta y cuánto hace falta para
que la métrica agregada sea estable?

La estadística es la misma (``sources.phase3.repetition_stats``); lo que cambia
es de dónde se leen las respuestas, las etiquetas y la tarea:

- ``dreamt``: ``responses_rep{R}/{modelo}/{sujeto}.{vista}.jsonl`` con
  condiciones ``C0``…``NC_absent`` y clases ``wake``/``n1``/``n2``/``rem``.
- ``physionet``: ``physionet2017[_tag]_rep{R}.responses.jsonl`` con una sola
  condición y clases ``normal``/``af``/``other-rhythm``/``noisy``.

Uso:
    python scripts/analyze_repetitions_external.py --experiment dreamt \\
        --cohort scale --model qwen2.5:7b --repetitions 3
    python scripts/analyze_repetitions_external.py --experiment physionet \\
        --tag srv --repetitions 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Los nombres de fichero de cada corrida viven en su runner (una sola fuente):
# se importan para que el analizador no pueda leer un corpus distinto.
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sources.common.paths import atomic_write_json
from sources.common.provenance import (
    legacy_repetition_provenance_name,
    read_provenance,
    repetition_provenance_name,
    utc_now,
)
from sources.phase2.llm_config import model_slug
from sources.phase3.records import evaluate_records
from sources.phase3.repetition_stats import (
    DEFAULT_TARGETS,
    USABLE_STATUSES,
    aggregate_window,
    extrapolate,
    format_metric,
    summarize_stability,
    window_stability,
)
from sources.phase3.repetition_stats import convergence as _convergence
from sources.phase3.repetition_stats import distribution as _distribution

DREAMT_OUTPUT = REPO_ROOT / "results" / "output" / "DREAMT" / "PHASE2"
PHYSIONET_OUTPUT = REPO_ROOT / "results" / "output" / "PHYSIONET2017" / "P3"
PHYSIONET_CONDITION = "C0_eval"


# ---------------------------------------------------------------------------
# Carga de registros y etiquetas
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee un JSONL ignorando líneas vacías o truncadas (corrida interrumpida)."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def dreamt_paths(output_dir: Path, model: str, repetitions: int) -> list[Path]:
    """Ficheros de respuesta del corpus de repeticiones de DREAMT."""
    folder = "responses" if repetitions <= 1 else f"responses_rep{repetitions}"
    return sorted((output_dir / folder / model_slug(model)).glob("*.jsonl"))


def dreamt_labels(output_dir: Path) -> dict[str, str]:
    """``window_id_opaque → clase`` desde ``labels/{sujeto}.labels.jsonl``."""
    from sources.ingestion.represent import _opaque_window_id

    labels: dict[str, str] = {}
    for path in sorted((output_dir / "labels").glob("*.labels.jsonl")):
        for row in read_jsonl(path):
            window_id = row.get("window_id")
            klass = row.get("klass")
            if window_id and klass:
                labels[_opaque_window_id(window_id)] = str(klass)
    return labels


def physionet_file_tag(tag: str | None, repetitions: int) -> str:
    """Sufijo de la corrida, calculado por el **runner** (una sola fuente)."""
    from run_p3_physionet import output_file_tag

    return output_file_tag(tag or "", "test", False, repetitions)


def physionet_paths(
    output_dir: Path,
    tag: str | None,
    repetitions: int,
) -> tuple[Path, Path]:
    """``(respuestas, etiquetas)`` del corpus de PhysioNet.

    El sufijo replica el del runner (``output_file_tag``): ``_rep{R}`` solo
    aparece cuando ``R > 1``, para no confundir el corpus de una pasada con el
    de repeticiones.
    """
    file_tag = physionet_file_tag(tag, repetitions)
    suffix = f"_{file_tag}" if file_tag else ""
    return (
        output_dir / f"physionet2017{suffix}.responses.jsonl",
        output_dir / f"physionet2017{suffix}.labels.jsonl",
    )


def physionet_provenance_path(
    output_dir: Path,
    tag: str | None,
    repetitions: int,
) -> Path:
    """``run_provenance[_sufijo].json`` de la corrida analizada."""
    file_tag = physionet_file_tag(tag, repetitions)
    return output_dir / f"run_provenance{('_' + file_tag) if file_tag else ''}.json"


def physionet_labels(path: Path) -> dict[str, str]:
    """``record_id → clase`` desde el fichero de etiquetas."""
    return {
        str(row["record_id"]): str(row["label"])
        for row in read_jsonl(path)
        if row.get("record_id") and row.get("label")
    }


def normalize_records(
    records: Sequence[Mapping[str, Any]],
    condition: str | None = None,
) -> list[dict[str, Any]]:
    """Homogeneiza los campos que comparten los dos runners.

    - ``window_id_opaque``: PhysioNet no lo escribe; su identificador de registro
      es público y estable, así que se usa tal cual.
    - ``condition_id``: PhysioNet tiene una sola condición.
    - ``representation``: PhysioNet tiene una sola vista.
    """
    normalized: list[dict[str, Any]] = []
    for row in records:
        item = dict(row)
        item.setdefault("window_id_opaque", item.get("record_id", ""))
        item.setdefault("condition_id", condition or PHYSIONET_CONDITION)
        item.setdefault("representation", "numeric")
        item["repetition"] = int(item.get("repetition", 0) or 0)
        normalized.append(item)
    return normalized


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metrics_for(
    voted: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    label_order: Sequence[str],
) -> dict[str, float]:
    """Accuracy, balanced, cobertura y ECE de una lista de registros votados."""
    metrics = evaluate_records(list(voted), labels, label_order=label_order)
    total = metrics["n"] or 1
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "coverage": float(metrics["n_decided"]) / total,
        "abstention_rate": float(metrics["abstention_rate"]),
        "ece": float(metrics["ece"]),
    }


def analyze_group(
    windows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
    label_order: Sequence[str],
    repetitions: int,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    """Estadísticos de varianza de muestreo de un grupo (tarea × condición)."""
    reps = list(range(repetitions))
    stabilities = [window_stability(records) for records in windows.values()]
    stability = summarize_stability(stabilities)

    # Una pasada por repetición: variabilidad del experimento completo.
    per_rep: dict[str, list[float]] = defaultdict(list)
    for rep in reps:
        voted = [
            aggregated
            for records in windows.values()
            if (aggregated := aggregate_window(records, [rep])) is not None
        ]
        if not voted:
            continue
        metrics = metrics_for(voted, labels, label_order)
        for key, value in metrics.items():
            per_rep[key].append(value)

    convergence = _convergence(
        windows,
        lambda voted, _labels: metrics_for(voted, labels, label_order),
        reps,
        rng,
        max_combos=max(draws, 1),
    )
    single = {key: _distribution(values) for key, values in per_rep.items()}
    sd_accuracy = single.get("accuracy", {}).get("sd")
    implied = extrapolate(
        float(sd_accuracy) if sd_accuracy is not None else float("nan")
    )

    # Confianza: una pasada vs promedio de repeticiones (resolución).
    conf_single = [
        float(record["confidence"])
        for entries in windows.values()
        for record in entries
        if record.get("status") in USABLE_STATUSES
        and record.get("confidence") is not None
    ]
    conf_aggregated = [
        float(aggregated["confidence"])
        for records in windows.values()
        if (aggregated := aggregate_window(records, reps)) is not None
    ]
    usable = [
        record
        for entries in windows.values()
        for record in entries
        if record.get("status") in USABLE_STATUSES
    ]
    agreement = {
        "n_windows": len(windows),
        "n_records": sum(len(entries) for entries in windows.values()),
        "n_usable": len(usable),
        "n_invalid": sum(len(entries) for entries in windows.values()) - len(usable),
        "n_abstain_records": sum(
            1 for record in usable if record.get("decision") == "abstain"
        ),
        "n_repetitions": repetitions,
    }
    return {
        **agreement,
        "stability": stability,
        "single_run": single,
        "implied_k": implied,
        "convergence": convergence,
        "confidence": {
            "single": {
                **_distribution(conf_single),
                "n_distinct": len(set(conf_single)),
            },
            "aggregated": {
                **_distribution(conf_aggregated),
                "n_distinct": len(set(round(v, 2) for v in conf_aggregated)),
            },
        },
    }


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
def analyze_dreamt(
    output_dir: Path,
    model: str,
    repetitions: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Análisis de repeticiones del corpus de sueño."""
    from run_p4_dreamt import CLASS_ORDER

    paths = dreamt_paths(output_dir, model, repetitions)
    if not paths:
        raise SystemExit(
            f"[REP-EXT] No hay respuestas en {output_dir} "
            f"(responses_rep{repetitions}/{model_slug(model)}). "
            "Lanza antes: GENERATE --experiment dreamt --repetitions R."
        )
    labels = dreamt_labels(output_dir)
    records = normalize_records(
        [row for path in paths for row in read_jsonl(path)]
    )
    return _report(
        experiment="dreamt",
        output_dir=output_dir,
        records=records,
        labels=labels,
        label_order=CLASS_ORDER,
        repetitions=repetitions,
        draws=draws,
        seed=seed,
        provenance=read_provenance(
            dreamt_provenance_path(output_dir, repetitions, model)
        ),
    )


def dreamt_provenance_path(
    output_dir: Path, repetitions: int, model: str | None
) -> Path:
    """Procedencia del corpus de DREAMT analizado (por modelo, con caída).

    La procedencia es por modelo desde 2026-11-28; los corpus anteriores
    escribieron un fichero compartido por cohorte. Se acepta ese fichero como
    respaldo **solo si su ``model`` coincide** con el que se analiza: si no,
    describe otro corpus y publicar su motor sería un dato falso.
    """
    candidates = [output_dir / repetition_provenance_name(repetitions, model_slug(model) if model else None)]
    legacy = output_dir / legacy_repetition_provenance_name(repetitions)
    if legacy not in candidates:
        candidates.append(legacy)
    for path in candidates:
        stored = read_provenance(path)
        if stored is None:
            continue
        stored_model = stored.get("model")
        if model and stored_model and str(stored_model) != str(model):
            continue
        return path
    return candidates[0]


def analyze_physionet(
    output_dir: Path,
    model: str | None,
    tag: str | None,
    repetitions: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Análisis de repeticiones del corpus de ritmo en ECG."""
    from run_p3_physionet import RHYTHM_CLASSES

    responses_path, labels_path = physionet_paths(output_dir, tag, repetitions)
    if not responses_path.is_file():
        raise SystemExit(
            f"[REP-EXT] No existe {responses_path}. Lanza antes: "
            "GENERATE --experiment physionet --repetitions R."
        )
    records = normalize_records(read_jsonl(responses_path))
    labels = physionet_labels(labels_path) if labels_path.is_file() else {}
    return _report(
        experiment="physionet",
        output_dir=output_dir,
        records=records,
        labels=labels,
        label_order=RHYTHM_CLASSES,
        repetitions=repetitions,
        draws=draws,
        seed=seed,
        provenance=read_provenance(
            physionet_provenance_path(output_dir, tag, repetitions)
        ),
        model=model,
    )


def _report(
    *,
    experiment: str,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    label_order: Sequence[str],
    repetitions: int,
    draws: int,
    seed: int,
    provenance: Mapping[str, Any] | None,
    model: str | None = None,
) -> dict[str, Any]:
    """Agrupa por ``(representación, condición)`` y calcula la varianza."""
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = f"{record.get('representation')}|{record.get('condition_id')}"
        grouped[key][str(record.get("window_id_opaque"))].append(record)

    rng = np.random.default_rng(seed)
    groups: dict[str, Any] = {}
    for key in sorted(grouped):
        groups[key] = analyze_group(
            grouped[key], labels, label_order, repetitions, rng, draws
        )

    return {
        "task": "repetitions_external",
        "experiment": experiment,
        "version": "1.0",
        "model": model or (provenance or {}).get("model"),
        "engine": {
            "ollama_version": (provenance or {}).get("ollama_version"),
            "model_digest": (provenance or {}).get("model_digest"),
        },
        "dir": str(output_dir),
        "repetitions": repetitions,
        "seeds": [int(seed) + r for r in range(repetitions)],
        "draws": draws,
        "targets": list(DEFAULT_TARGETS),
        "classes": list(label_order),
        "generated_at": utc_now(),
        "n_classes_labelled": len(set(labels.values())),
        "groups": groups,
    }


def print_report(report: Mapping[str, Any]) -> None:
    """Resumen legible: acuerdo modal, error de una pasada y ``k`` implícito."""
    engine = report.get("engine") or {}
    print(
        f"[REP-EXT] {report['experiment']} · modelo {report.get('model')} · "
        f"R={report['repetitions']} · semillas {report['seeds']}"
    )
    print(
        f"[REP-EXT] motor: Ollama {engine.get('ollama_version')} · "
        f"digest {engine.get('model_digest')} · {report['dir']}"
    )
    print(
        f"[REP-EXT] {report['n_classes_labelled']} clases etiquetadas · "
        f"{report['draws']} remuestreos"
    )
    header = (
        f"{'grupo':<28}{'n':>5}{'acuerdo':>9}{'acc1':>8}{'sd1':>7}"
        f"{'cov1':>7}{'inv':>6}{'k(2pp)':>8}"
    )
    print(header)
    print("-" * len(header))
    for key, group in report["groups"].items():
        single = group["single_run"]
        accuracy = single.get("accuracy", {})
        coverage = single.get("coverage", {}).get("mean")
        implied = group.get("implied_k", {}).get("implied_k", {})
        # Sin ventanas decididas la accuracy no está definida: ``evaluate_records``
        # devuelve 0.0 por comodidad numérica, pero publicar ese 0.0 como
        # «accuracy 0 %» describiría un clasificador que falla cuando en realidad
        # no clasificó nada.
        decided = coverage if coverage else None
        print(
            f"{key:<28}{group['n_windows']:>5}"
            f"{format_metric(group['stability'].get('mean_modal_share'), 9)}"
            f"{format_metric(accuracy.get('mean') if decided else None, 8)}"
            f"{format_metric(accuracy.get('sd') if decided else None, 7)}"
            f"{format_metric(coverage, 7)}"
            f"{group['n_invalid']:>6}"
            f"{implied.get('2pp') or 0:>8}"
        )
    print(
        "[REP-EXT] 'acuerdo' = proporción modal media entre repeticiones; "
        "'acc1'/'sd1' = accuracy y su desviación con una sola pasada; "
        "'cov1' = fracción decidida (1 − abstención) en una pasada; "
        "'—' = sin registros decididos (no hay accuracy que promediar); "
        "'k(2pp)' = repeticiones que pediría la ley sd/sqrt(k) para ±2 puntos."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("dreamt", "physionet"), required=True)
    parser.add_argument("--dir", default=None, help="directorio del corpus")
    parser.add_argument("--model", default=None, help="modelo de Ollama")
    parser.add_argument(
        "--cohort", choices=("pilot", "scale"), default="scale", help="(DREAMT)"
    )
    parser.add_argument("--tag", default=None, help="(PhysioNet) sufijo de la corrida")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.repetitions < 2:
        raise SystemExit(
            "[REP-EXT] --repetitions debe ser >= 2: con una pasada no hay "
            "varianza de muestreo que medir."
        )

    if args.experiment == "dreamt":
        output_dir = Path(args.dir) if args.dir else DREAMT_OUTPUT / args.cohort
        if not args.model:
            raise SystemExit("[REP-EXT] DREAMT exige --model.")
        report = analyze_dreamt(
            output_dir, args.model, args.repetitions, args.draws, args.seed
        )
        default_output = output_dir / (
            f"repetitions_rep{args.repetitions}_{model_slug(args.model)}.json"
        )
    else:
        output_dir = Path(args.dir) if args.dir else PHYSIONET_OUTPUT
        report = analyze_physionet(
            output_dir,
            args.model,
            args.tag,
            args.repetitions,
            args.draws,
            args.seed,
        )
        stem = f"_{args.tag}" if args.tag else ""
        default_output = output_dir / (
            f"repetitions{stem}_rep{args.repetitions}.json"
        )

    output = Path(args.output) if args.output else default_output
    atomic_write_json(output, report)
    print_report(report)
    print(f"[REP-EXT] informe: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
