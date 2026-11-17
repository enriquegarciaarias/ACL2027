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
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.provenance import utc_now
from sources.ingestion.represent import _opaque_window_id
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

USABLE_STATUSES = ("valid_response", "valid_abstention")
DEFAULT_TARGETS = (0.01, 0.02, 0.03)


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
        labels[_opaque_window_id(record["window_id"])] = label
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


def infer_metadata(
    records: Sequence[Mapping[str, Any]],
    decision_mode: str | None,
    nonstress_variant: str | None,
) -> tuple[str, str]:
    """Deduce tarea y variante de la procedencia de las respuestas."""
    if not decision_mode:
        counts = Counter(str(r.get("decision_mode") or "") for r in records)
        counts.pop("", None)
        decision_mode = counts.most_common(1)[0][0] if counts else DECISION_MODE_AFFECTIVE
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
# Estabilidad por ventana
# ---------------------------------------------------------------------------
def vote(decisions: Sequence[str]) -> str:
    """Voto mayoritario; empate → ``abstain`` (criterio conservador)."""
    counts = Counter(decisions)
    top = max(counts.values())
    winners = [decision for decision, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else "abstain"


def window_stability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Estadísticos de una ventana a lo largo de sus repeticiones."""
    usable = [r for r in records if r.get("status") in USABLE_STATUSES]
    decisions = [str(r.get("decision")) for r in usable]
    confidences = [
        float(r["confidence"]) for r in usable if r.get("confidence") is not None
    ]
    counts = Counter(decisions)
    n = len(decisions)
    modal_share = max(counts.values()) / n if n else 0.0
    entropy = -sum(
        (count / n) * math.log2(count / n) for count in counts.values()
    ) if n else 0.0
    return {
        "window_id_opaque": records[0].get("window_id_opaque", ""),
        "n": len(records),
        "n_usable": n,
        "n_invalid": len(records) - n,
        "decisions": dict(counts),
        "modal_decision": vote(decisions) if decisions else "",
        "modal_share": modal_share,
        "entropy_bits": entropy,
        "n_distinct_decisions": len(counts),
        "confidence_mean": float(np.mean(confidences)) if confidences else None,
        "confidence_std": float(np.std(confidences)) if confidences else None,
        "confidence_min": float(np.min(confidences)) if confidences else None,
        "confidence_max": float(np.max(confidences)) if confidences else None,
        "confidence_n_distinct": len(set(confidences)),
        "abstain_share": (
            sum(1 for d in decisions if d == "abstain") / n if n else 0.0
        ),
    }


def _summarize_stability(stabilities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not stabilities:
        return {}
    conf_stds = [
        s["confidence_std"] for s in stabilities if s.get("confidence_std") is not None
    ]
    return {
        "n_windows": len(stabilities),
        "mean_modal_share": float(np.mean([s["modal_share"] for s in stabilities])),
        "frac_deterministic": float(
            np.mean([1.0 if s["modal_share"] == 1.0 else 0.0 for s in stabilities])
        ),
        "frac_single_decision": float(
            np.mean([1.0 if s["n_distinct_decisions"] == 1 else 0.0 for s in stabilities])
        ),
        "mean_entropy_bits": float(
            np.mean([s["entropy_bits"] for s in stabilities])
        ),
        "mean_confidence_std": float(np.mean(conf_stds)) if conf_stds else 0.0,
        "frac_confidence_constant": float(
            np.mean([1.0 if s["confidence_std"] == 0.0 else 0.0 for s in stabilities])
        ),
        "mean_abstain_share": float(
            np.mean([s["abstain_share"] for s in stabilities])
        ),
    }


# ---------------------------------------------------------------------------
# Métricas agregadas
# ---------------------------------------------------------------------------
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


def aggregate_window(
    records: Sequence[Mapping[str, Any]],
    reps: Sequence[int],
) -> dict[str, Any] | None:
    """Voto mayoritario de las repeticiones indicadas y confianza media.

    La confianza agregada es la media de la confianza de las repeticiones que
    votaron la decisión ganadora: promediar réplicas da resolución a un modelo
    que solo emite dos valores distintos.
    """
    by_rep = {int(r.get("repetition", 0) or 0): r for r in records}
    selected = [by_rep[rep] for rep in reps if rep in by_rep]
    usable = [r for r in selected if r.get("status") in USABLE_STATUSES]
    if not usable:
        return None
    decisions = [str(r.get("decision")) for r in usable]
    voted = vote(decisions)
    if voted == "abstain":
        confidence = 0.0
        status = "valid_abstention"
    else:
        agreeing = [
            float(r["confidence"])
            for r in usable
            if r.get("decision") == voted and r.get("confidence") is not None
        ]
        confidence = float(np.mean(agreeing)) if agreeing else 0.0
        status = "valid_response"
    return {
        "window_id_opaque": usable[0].get("window_id_opaque", ""),
        "decision": voted,
        "confidence": confidence,
        "status": status,
    }


def _subsets(n: int, k: int, rng: np.random.Generator, max_combos: int) -> list[tuple[int, ...]]:
    """Combinaciones de tamaño ``k`` (todas si son pocas; si no, aleatorias)."""
    if k < 1 or k > n:
        return []
    total = math.comb(n, k)
    if total <= max_combos:
        return list(combinations(range(n), k))
    seen: set[tuple[int, ...]] = set()
    while len(seen) < max_combos:
        draw = tuple(sorted(int(i) for i in rng.choice(n, size=k, replace=False)))
        seen.add(draw)
    return sorted(seen)


def convergence(
    windows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
    decision_mode: str,
    reps: Sequence[int],
    rng: np.random.Generator,
    max_combos: int,
) -> list[dict[str, Any]]:
    """Métricas por número de repeticiones agregadas (``k = 1..R``)."""
    rows: list[dict[str, Any]] = []
    for k in range(1, len(reps) + 1):
        acc: list[float] = []
        bal: list[float] = []
        cov: list[float] = []
        abst: list[float] = []
        for subset in _subsets(len(reps), k, rng, max_combos):
            chosen = [reps[i] for i in subset]
            voted = [
                aggregated
                for records in windows.values()
                if (aggregated := aggregate_window(records, chosen)) is not None
            ]
            if not voted:
                continue
            metrics = metrics_for(voted, labels, decision_mode)
            acc.append(metrics["accuracy"])
            bal.append(metrics["balanced_accuracy"])
            cov.append(metrics["coverage"])
            abst.append(metrics["abstention_rate"])
        if not acc:
            continue
        rows.append(
            {
                "k": k,
                "n_combinations": len(acc),
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_sd": float(np.std(acc)),
                "balanced_mean": float(np.nanmean(bal)),
                "balanced_sd": float(np.nanstd(bal)),
                "coverage_mean": float(np.mean(cov)),
                "abstention_mean": float(np.mean(abst)),
            }
        )
    return rows


def extrapolate(sd1: float, targets: Sequence[float] = DEFAULT_TARGETS) -> dict[str, Any]:
    """``k`` para bajar el error de muestreo de ``sd1`` a cada umbral.

    Supone independencia entre repeticiones y ``sd_k ≈ sd_1 / sqrt(k)``; es una
    cota optimista (si el modelo es determinista, ``sd1 = 0`` y no hace falta
    ninguna repetición).
    """
    if not math.isfinite(sd1) or sd1 <= 0:
        return {
            "sd_single": float(sd1) if math.isfinite(sd1) else None,
            "implied_k": {f"{pct}pp": 1 for pct in (1, 2, 3)},
            "note": "sin varianza de muestreo: las repeticiones no cambiarían la métrica",
        }
    return {
        "sd_single": float(sd1),
        "implied_k": {
            f"{int(round(target * 100))}pp": int(math.ceil((sd1 / target) ** 2))
            for target in targets
        },
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"mean": None, "sd": None, "values": []}
    return {
        "mean": float(np.mean(clean)),
        "sd": float(np.std(clean)),
        "values": [float(v) for v in clean],
    }


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
            "stability": _summarize_stability(stabilities),
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
            f"  una pasada: acc={single['accuracy']['mean']:.3f} "
            f"±{single['accuracy']['sd']:.3f} | "
            f"bal={single['balanced_accuracy']['mean']:.3f} "
            f"±{single['balanced_accuracy']['sd']:.3f} | "
            f"cov={single['coverage']['mean']:.3f}"
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
