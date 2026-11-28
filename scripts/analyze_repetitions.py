#!/usr/bin/env python
"""Análisis de repeticiones de prompt: ¿cuánta varianza introduce el muestreo?

Pregunta que responde: si se envía **el mismo prompt** R veces cambiando solo la
semilla, ¿cuánto cambia la respuesta y cuánto hace falta para que la métrica
agregada sea estable?

Es el paso previo a lanzar un corpus de repeticiones completo: con una muestra
reducida (pocos sujetos, condiciones extremas) se estima si el efecto existe y
qué ``R`` lo estabiliza, antes de gastar horas de LLM.

Qué mide, por condición de degradación:

1. **Estabilidad de la decisión por ventana** — proporción modal (``1.0`` =
   respuesta determinista), entropía y fracción de ventanas con una sola
   decisión distinta.
2. **Dispersión de la confianza** — desviación típica intra-ventana y número de
   valores distintos. Un modelo que solo emite dos valores de confianza
   (``qwen2.5:7b``: 85 y 75) no puede ordenar ventanas; promediar R repeticiones
   crea resolución aunque la decisión no cambie.
3. **Variabilidad de la métrica con una pasada** — accuracy, balanced,
   cobertura y abstención calculadas con cada repetición por separado; su
   desviación típica es el error de muestreo de un experimento de una pasada.
4. **Convergencia por votación mayoritaria** — métricas con ``k = 1..R``
   repeticiones agregadas por voto (empate → ``abstain``).
5. **Extrapolación** — ``k`` necesario para que el error de muestreo baje de 1,
   2 o 3 puntos, con la ley ``sd_k ≈ sd_1 / sqrt(k)``.

Uso:
    python scripts/analyze_repetitions.py \\
        --dir results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp_rep10/qwen2_5_7b \\
        --decision-mode binary --nonstress-variant schmidt \\
        --output results/output/WESAD/PHASE2/repetitions_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.provenance import opaque_window_id, utc_now
from sources.phase3.repetition_stats import (
    DEFAULT_TARGETS,
    USABLE_STATUSES,
    aggregate_window,
    extrapolate,
    format_metric,
    subsets,
    summarize_stability,
    vote,
    window_stability,
)
from sources.phase3.repetition_stats import convergence as _generic_convergence
from sources.phase3.repetition_stats import distribution as _distribution
from sources.phase2.binary_variants import (
    DEFAULT_NONSTRESS_VARIANT,
    NONSTRESS_VARIANTS,
    variant_classes,
)
from sources.phase2.response_schema import (
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_STRESS_BINARY,
)
from sources.phase3.records import evaluate_binary_records, evaluate_records

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_labels() -> dict[str, str]:
    """Etiqueta real por identificador opaco de ventana (nunca al prompt)."""
    labels: dict[str, str] = {}
    path = INGEST_DIR / "groundtruth" / "labels.jsonl"
    if not path.is_file():
        return labels
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        labels[opaque_window_id(record["window_id"])] = label
    return labels


def load_records(dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Registros de respuesta de todos los ``*.responses.jsonl`` de los directorios."""
    records: list[dict[str, Any]] = []
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.responses.jsonl")):
            for line in path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:      # línea a medio escribir
                    continue
    return records


_DECISION_MODE_ALIASES = {"binary": DECISION_MODE_STRESS_BINARY,
                          "affective": DECISION_MODE_AFFECTIVE}


def infer_metadata(
    records: Sequence[Mapping[str, Any]],
    decision_mode: str | None,
    nonstress_variant: str | None,
) -> tuple[str, str]:
    """Deduce tarea y variante de la procedencia de las respuestas.

    Acepta el vocabulario del CLI (``binary``) además del canónico
    (``stress_binary``): el resto de la cadena —``--decision-mode`` del barrido,
    del comparador y del analizador de degradación— usa ``binary``, y un corpus
    escrito a mano o por una herramienta externa no debe abortar por el alias.
    """
    if decision_mode:
        decision_mode = _DECISION_MODE_ALIASES.get(str(decision_mode), str(decision_mode))
    if not decision_mode:
        counts = Counter(str(r.get("decision_mode") or "") for r in records)
        counts.pop("", None)
        inferred = counts.most_common(1)[0][0] if counts else DECISION_MODE_AFFECTIVE
        decision_mode = _DECISION_MODE_ALIASES.get(inferred, inferred)
    if decision_mode not in (DECISION_MODE_AFFECTIVE, DECISION_MODE_STRESS_BINARY):
        raise SystemExit(f"decision_mode desconocido: {decision_mode!r}")
    if not nonstress_variant:
        counts = Counter(str(r.get("nonstress_variant") or "") for r in records)
        counts.pop("", None)
        nonstress_variant = (
            counts.most_common(1)[0][0] if counts else DEFAULT_NONSTRESS_VARIANT
        )
    return decision_mode, nonstress_variant


def allowed_labels(decision_mode: str, nonstress_variant: str) -> set[str]:
    """Etiquetas reales que pertenecen a la tarea.

    En binario se descartan las ventanas fuera de la variante (``meditation`` en
    ``schmidt``): no forman parte del endpoint y solo añadirían ruido a las
    métricas.
    """
    classes = set(variant_classes(nonstress_variant))
    if decision_mode == DECISION_MODE_STRESS_BINARY:
        return classes | {"stress"}
    return classes | {"stress", "meditation", "amusement", "baseline"}


# ---------------------------------------------------------------------------
# Delegación en los estadísticos compartidos
# ---------------------------------------------------------------------------
# ``vote``, ``window_stability``, ``summarize_stability``, ``subsets``,
# ``aggregate_window``, ``extrapolate`` y ``_distribution`` viven en
# ``sources.phase3.repetition_stats`` porque DREAMT y PhysioNet calculan la
# misma varianza de muestreo con otro vocabulario de etiquetas.


def metrics_for(
    voted: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    decision_mode: str,
) -> dict[str, float]:
    """Accuracy, balanced, cobertura, abstención, AUROC y ECE de una pasada."""
    if decision_mode == DECISION_MODE_STRESS_BINARY:
        metrics = evaluate_binary_records(list(voted), labels)
        auroc = float(metrics.get("auroc_stress", float("nan")))
    else:
        metrics = evaluate_records(list(voted), labels)
        auroc = float(metrics.get("auroc", {}).get("stress", float("nan")))
    return {
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "balanced_accuracy": float(metrics.get("balanced_accuracy", float("nan"))),
        "coverage": float(metrics.get("coverage", float("nan"))),
        "abstention_rate": float(metrics.get("abstention_rate", float("nan"))),
        "auroc_stress": auroc,
        "ece": float(metrics.get("ece", float("nan"))),
    }


def convergence(
    windows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
    decision_mode: str,
    reps: Sequence[int],
    rng: np.random.Generator,
    max_combos: int,
) -> list[dict[str, Any]]:
    """Convergencia del voto mayoritario con la métrica de WESAD."""
    return _generic_convergence(
        windows,
        lambda voted, _labels: metrics_for(voted, labels, decision_mode),
        reps,
        rng,
        max_combos,
    )


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def analyze(
    dirs: Sequence[Path],
    decision_mode: str | None = None,
    nonstress_variant: str | None = None,
    draws: int = 400,
    seed: int = 12345,
    labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Informe completo de varianza de muestreo.

    Args:
        dirs: directorios con ``*.responses.jsonl``.
        decision_mode: ``None`` = deducir de la procedencia.
        nonstress_variant: ``None`` = deducir de la procedencia.
        draws: subconjuntos evaluados por ``k``.
        seed: semilla de los subconjuntos aleatorios.
        labels: etiquetas por identificador opaco (``None`` = leer del corpus).
    """
    records = load_records(dirs)
    if not records:
        raise SystemExit(
            f"No hay respuestas en {[str(d) for d in dirs]}. "
            "Lanza antes el barrido con --repetitions."
        )
    decision_mode, nonstress_variant = infer_metadata(
        records, decision_mode, nonstress_variant
    )
    labels = dict(labels) if labels is not None else load_labels()
    allowed = allowed_labels(decision_mode, nonstress_variant)

    kept = [
        r
        for r in records
        if labels.get(str(r.get("window_id_opaque")), "UNKNOWN") in allowed
    ]
    n_dropped_variant = len(records) - len(kept)
    if not kept:
        missing_opaque = sum(1 for r in records if not r.get("window_id_opaque"))
        hint = (
            f"{missing_opaque} de {len(records)} registros no traen "
            "`window_id_opaque`, el campo con el que se emparejan las etiquetas "
            "(lo escribe el barrido)."
            if missing_opaque
            else f"los {len(records)} registros quedan fuera de la variante "
            f"{nonstress_variant!r} (¿condiciones de otra tarea?)."
        )
        raise SystemExit(
            f"[REP] No quedan ventanas evaluables en {[str(d) for d in dirs]}.\n"
            f"      Motivo: {hint}"
        )

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reps_seen: dict[str, set[int]] = defaultdict(set)
    for record in kept:
        condition = str(record.get("condition_id"))
        window = str(record.get("window_id_opaque"))
        grouped[condition][window].append(record)
        reps_seen[condition].add(int(record.get("repetition", 0) or 0))

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "dirs": [str(d) for d in dirs],
        "model": kept[0].get("model"),
        "decision_mode": decision_mode,
        "nonstress_variant": nonstress_variant,
        "n_records": len(records),
        "n_dropped_outside_variant": n_dropped_variant,
        "targets": list(DEFAULT_TARGETS),
        "conditions": {},
    }

    for condition in sorted(grouped):
        windows = dict(grouped[condition])
        reps = sorted(reps_seen[condition])
        stabilities = [window_stability(records) for records in windows.values()]
        complete = {
            window: records
            for window, records in windows.items()
            if len({int(r.get("repetition", 0) or 0) for r in records}) == len(reps)
        }

        # Una pasada por repetición (variabilidad del experimento completo).
        per_rep: dict[str, list[float]] = defaultdict(list)
        for rep in reps:
            voted = [
                aggregated
                for records in windows.values()
                if (aggregated := aggregate_window(records, [rep])) is not None
            ]
            if not voted:
                continue
            metrics = metrics_for(voted, labels, decision_mode)
            for key, value in metrics.items():
                per_rep[key].append(value)

        single = {key: _distribution(values) for key, values in per_rep.items()}

        rng = np.random.default_rng(seed)
        curve = convergence(
            complete, labels, decision_mode, reps, rng, max_combos=max(draws, 50)
        )

        # Confianza: resolución de una pasada frente a promediada en R repeticiones.
        # "Una pasada" = cada respuesta individual (lo que tiene el corpus
        # publicado); "agregada" = media por ventana de las R repeticiones que
        # votaron la decisión ganadora.
        conf_single = [
            float(record["confidence"])
            for records in windows.values()
            for record in records
            if record.get("status") in USABLE_STATUSES
            and record.get("confidence") is not None
        ]
        conf_aggregated = [
            aggregated["confidence"]
            for records in windows.values()
            if (aggregated := aggregate_window(records, reps)) is not None
            and aggregated["status"] == "valid_response"
        ]
        aggregated_voted = [
            aggregated
            for records in windows.values()
            if (aggregated := aggregate_window(records, reps)) is not None
        ]
        auroc_aggregated = (
            metrics_for(aggregated_voted, labels, decision_mode)["auroc_stress"]
            if aggregated_voted
            else float("nan")
        )

        report["conditions"][condition] = {
            "n_windows": len(windows),
            "n_complete_windows": len(complete),
            "n_reps": len(reps),
            "stability": summarize_stability(stabilities),
            "windows": stabilities,
            "single_run": single,
            "convergence": curve,
            "extrapolation": {
                "accuracy": extrapolate(single.get("accuracy", {}).get("sd") or 0.0),
                "balanced_accuracy": extrapolate(
                    single.get("balanced_accuracy", {}).get("sd") or 0.0
                ),
            },
            "confidence": {
                "single": {
                    **_distribution(conf_single),
                    "n_distinct": len({round(v, 6) for v in conf_single}),
                },
                "aggregated": {
                    **_distribution(conf_aggregated),
                    "n_distinct": len({round(v, 6) for v in conf_aggregated}),
                },
                "auroc_single_mean": single.get("auroc_stress", {}).get("mean"),
                "auroc_aggregated": float(auroc_aggregated),
            },
        }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_report(report: Mapping[str, Any]) -> None:
    """Resumen en consola: estabilidad, convergencia y extrapolación."""
    print(
        f"Repeticiones | modelo={report['model']} | tarea={report['decision_mode']} | "
        f"variante={report['nonstress_variant']} | registros={report['n_records']} | "
        f"fuera de variante={report['n_dropped_outside_variant']}"
    )
    for condition, data in report["conditions"].items():
        stability = data["stability"]
        print(
            f"\n[{condition}] ventanas={data['n_windows']} "
            f"completas={data['n_complete_windows']} reps={data['n_reps']}"
        )
        print(
            f"  estabilidad: acuerdo modal={stability['mean_modal_share']:.3f} | "
            f"deterministas={stability['frac_deterministic']:.1%} | "
            f"entropía={stability['mean_entropy_bits']:.3f} bits | "
            f"sd(conf) intra={stability['mean_confidence_std']:.2f} | "
            f"conf constante={stability['frac_confidence_constant']:.1%}"
        )
        single = data["single_run"]
        print(
            f"  una pasada: acc={format_metric(single['accuracy']['mean'])} "
            f"±{format_metric(single['accuracy']['sd'])} | "
            f"bal={format_metric(single['balanced_accuracy']['mean'])} "
            f"±{format_metric(single['balanced_accuracy']['sd'])} | "
            f"cov={format_metric(single['coverage']['mean'])}"
        )
        confidence = data["confidence"]
        print(
            f"  confianza: valores distintos 1 pasada={confidence['single']['n_distinct']} "
            f"→ {data['n_reps']} reps={confidence['aggregated']['n_distinct']} | "
            f"AUROC {confidence['auroc_single_mean']} → "
            f"{confidence['auroc_aggregated']}"
        )
        print("    k | accuracy ± sd | balanced | cobertura | abstención")
        for row in data["convergence"]:
            print(
                f"  {row['k']:3d} | {row['accuracy_mean']:.3f} ±{row['accuracy_sd']:.3f}"
                f" | {row['balanced_mean']:.3f}   | {row['coverage_mean']:.3f}"
                f"     | {row['abstention_mean']:.3f}"
            )
        extrapolation = data["extrapolation"]["accuracy"]
        accuracy_sd = extrapolation.get("sd_single")
        if accuracy_sd:
            implied = extrapolation["implied_k"]
            print(
                f"  extrapolación (acc): k≈{implied['1pp']} para ±1pp · "
                f"k≈{implied['2pp']} para ±2pp · k≈{implied['3pp']} para ±3pp"
            )
        else:
            print(f"  extrapolación: {extrapolation.get('note', 'sin varianza')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        required=True,
        help="Directorio con *.responses.jsonl (repetible).",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["affective", "binary"],
        default=None,
        help="Tarea; vacío = deducir de la procedencia de las respuestas.",
    )
    parser.add_argument(
        "--nonstress-variant",
        choices=sorted(NONSTRESS_VARIANTS),
        default=None,
        help="Variante binaria; vacío = deducir del corpus.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=400,
        help="Subconjuntos evaluados por k (por defecto 400).",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    decision_mode = (
        DECISION_MODE_STRESS_BINARY
        if args.decision_mode == "binary"
        else DECISION_MODE_AFFECTIVE
        if args.decision_mode == "affective"
        else None
    )
    report = analyze(
        [Path(d) for d in args.dirs],
        decision_mode=decision_mode,
        nonstress_variant=args.nonstress_variant,
        draws=args.draws,
        seed=args.seed,
    )
    print_report(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nInforme: {output}")


if __name__ == "__main__":
    main()
