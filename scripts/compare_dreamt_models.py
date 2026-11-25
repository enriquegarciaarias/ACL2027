"""Comparación de modelos en el piloto DREAMT (estadios de sueño).

El piloto DREAMT es el segundo eje de tarea de la tesis: la misma pregunta
(respuesta a la evidencia bajo degradación controlada) con otra etiqueta y otro
sensor. Este script responde a una pregunta concreta: **¿se comporta igual un
segundo modelo?** Compara dos o más modelos sobre las mismas ventanas,
condiciones y representaciones, recalculando las métricas por sujeto con el
mismo evaluador que usa el piloto (``run_p4_dreamt._condition_block``), de modo
que la tabla no depende de qué ``evaluation_{slug}.json`` se haya escrito.

Diseño de la comparación
------------------------
* La unidad de generalización es el **sujeto**, no la ventana: se calculan las
  métricas por sujeto y se agregan con media y rango ``[mín–máx]``.
* Con 4 sujetos de piloto **no se publican intervalos de confianza**: un
  bootstrap sobre cuatro sujetos daría cifras con apariencia de precisión. Se
  informa la dispersión real (rango) y se declara la limitación.
* La accuracy se acompaña siempre de cobertura y de accuracy con la abstención
  contada como error: en esta tarea la abstención cambia mucho entre modelos y
  la accuracy bruta premia al que decide menos.

Uso
---
    python scripts/compare_dreamt_models.py --models qwen2.5:7b,medgemma:4b
    python scripts/compare_dreamt_models.py --models qwen2.5:7b,llama3.1:8b \
        --representation numeric --recovery --output /tmp/comparison.json

Con ``--cohort scale`` compara la cohorte de escala (12 sujetos) y publica
intervalos bootstrap sobre sujetos para abstención, accuracy y balanced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sources.external import dreamt  # noqa: E402
from sources.phase2.llm_config import model_slug  # noqa: E402
from sources.phase3.recovery import apply_recovery  # noqa: E402

from scripts.run_p4_dreamt import (  # noqa: E402
    CLASS_ORDER,
    COHORTS,
    DEFAULT_OUTPUTS,
    _condition_block,
    _load_labels,
    read_jsonl,
)

#: Métricas que se agregan por sujeto (nombres del bloque ``_condition_block``).
AGGREGATED: tuple[str, ...] = (
    "n",
    "n_decided",
    "accuracy",
    "accuracy_incl_abstain",
    "balanced_accuracy",
    "coverage",
    "abstain_rate",
    "supported_rate",
    "contradicted_rate",
    "n_claims",
    "unrecognized_metric_names",
    "coherence_rate",
    "certainty_index",
    "confidence_auroc",
    "overconfidence",
)

#: Recalls por clase que se agregan como ``recall_<clase>``.
RECALL_CLASSES: tuple[str, ...] = tuple(CLASS_ORDER)


def _finite(values: Iterable[Any]) -> list[float]:
    """Valores numéricos y finitos (descarta ``None`` y ``NaN``)."""
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def aggregate(values: Iterable[Any]) -> dict[str, Any]:
    """Media y rango de una métrica entre sujetos (sin intervalos, ver módulo)."""
    numbers = _finite(values)
    if not numbers:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


#: Métricas con intervalo bootstrap sobre sujetos (se re-muestrean sujetos, no
#: ventanas: la ventana no es la unidad de generalización).
CI_METRICS: tuple[str, ...] = (
    "abstain_rate",
    "accuracy_incl_abstain",
    "balanced_accuracy",
)

#: Mínimo de sujetos para publicar un intervalo (con 4 no significa nada).
MIN_SUBJECTS_FOR_CI: int = 8


def bootstrap_subject_ci(
    values: Iterable[Any],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float] | None:
    """IC 95 % de la media re-muestreando **sujetos** (percentil).

    Devuelve ``None`` si hay menos de ``MIN_SUBJECTS_FOR_CI`` valores: con pocos
    sujetos el intervalo sería más estrecho que la incertidumbre real.
    """
    numbers = _finite(values)
    if len(numbers) < MIN_SUBJECTS_FOR_CI:
        return None
    rng = np.random.default_rng(seed)
    data = np.asarray(numbers, dtype=float)
    means = np.array(
        [rng.choice(data, size=data.size, replace=True).mean() for _ in range(n_boot)]
    )
    return {
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_subjects": int(data.size),
        "n_boot": int(n_boot),
    }


def _finite_pairs(
    reference_values: Iterable[Any], other_values: Iterable[Any]
) -> list[tuple[float, float]]:
    """Pares numéricos y finitos, manteniendo la alineación por sujeto."""
    out: list[tuple[float, float]] = []
    for left, right in zip(reference_values, other_values):
        try:
            a, b = float(left), float(right)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            out.append((a, b))
    return out


def paired_bootstrap_delta(
    reference_values: Iterable[Any],
    other_values: Iterable[Any],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Diferencia pareada ``otro − referencia`` con IC bootstrap sobre sujetos.

    Los pares se forman **por sujeto**: se re-muestrean sujetos completos, no
    ventanas. Si hay menos de ``MIN_SUBJECTS_FOR_CI`` pares se devuelve el delta
    con ``ci = None`` (misma regla que la agregación).
    """
    pairs = _finite_pairs(reference_values, other_values)
    if not pairs:
        return None
    left = np.asarray([a for a, _ in pairs], dtype=float)
    right = np.asarray([b for _, b in pairs], dtype=float)
    delta = float(right.mean() - left.mean())
    if left.size < MIN_SUBJECTS_FOR_CI:
        return {"delta": delta, "ci": None, "n_subjects": int(left.size)}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        rows = rng.integers(0, left.size, left.size)
        diffs[index] = right[rows].mean() - left[rows].mean()
    return {
        "delta": delta,
        "ci": {
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "n_subjects": int(left.size),
            "n_boot": int(n_boot),
        },
        "n_subjects": int(left.size),
    }


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce un bloque de condición a las métricas que se comparan.

    ``_condition_block`` devuelve bloques anidados (claims, coherencia, certeza,
    recall por clase); aquí se extraen los valores escalares con el criterio de
    la tabla publicada.
    """
    claims = metrics.get("claims") or {}
    coherence = metrics.get("coherence") or {}
    certainty = metrics.get("certainty") or {}
    recall = metrics.get("recall_by_class") or {}
    flat: dict[str, Any] = {
        "n": metrics.get("n"),
        "n_decided": metrics.get("n_decided"),
        "accuracy": metrics.get("accuracy"),
        "accuracy_incl_abstain": metrics.get("accuracy_incl_abstain"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "coverage": metrics.get("coverage"),
        "abstain_rate": (
            (metrics.get("n_abstain") or 0) / metrics["n"] if metrics.get("n") else None
        ),
        "supported_rate": claims.get("supported_rate"),
        "contradicted_rate": claims.get("contradicted_rate"),
        "n_claims": claims.get("n_claims"),
        "unrecognized_metric_names": claims.get("unrecognized_metric_names"),
        "coherence_rate": coherence.get("coherence_rate"),
        "certainty_index": certainty.get("median_index"),
        "confidence_auroc": metrics.get("confidence_auroc"),
        "overconfidence": metrics.get("overconfidence"),
    }
    for label in RECALL_CLASSES:
        flat[f"recall_{label}"] = recall.get(label)
    return flat


def compute_metrics(
    records_by_subject: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{sujeto: {condición: métricas}}`` con el evaluador del piloto."""
    per_subject: dict[str, dict[str, dict[str, Any]]] = {}
    for subject, records in records_by_subject.items():
        by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            by_condition[record["condition_id"]].append(record)
        per_subject[subject] = {
            condition: _condition_block(rows, labels)
            for condition, rows in by_condition.items()
        }
    return per_subject


def aggregate_by_condition(
    per_subject: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Agrega por condición las métricas de todos los sujetos."""
    conditions: set[str] = set()
    for blocks in per_subject.values():
        conditions.update(blocks)
    out: dict[str, dict[str, Any]] = {}
    for condition in dreamt.PILOT_CONDITIONS:
        if condition not in conditions:
            continue
        rows = [
            flatten_metrics(blocks[condition])
            for blocks in per_subject.values()
            if condition in blocks
        ]
        if not rows:
            continue
        keys = list(AGGREGATED) + [f"recall_{label}" for label in RECALL_CLASSES]
        out[condition] = {key: aggregate(row.get(key) for row in rows) for key in keys}
        out[condition]["subjects"] = sorted(per_subject.keys())
        out[condition]["ci95"] = {
            key: ci
            for key in CI_METRICS
            if (ci := bootstrap_subject_ci(row.get(key) for row in rows)) is not None
        }
    # Condiciones de una matriz distinta (p. ej. corpus antiguos).
    for condition in sorted(conditions - set(out)):
        rows = [
            flatten_metrics(blocks[condition])
            for blocks in per_subject.values()
            if condition in blocks
        ]
        keys = list(AGGREGATED) + [f"recall_{label}" for label in RECALL_CLASSES]
        out[condition] = {key: aggregate(row.get(key) for row in rows) for key in keys}
        out[condition]["subjects"] = sorted(per_subject.keys())
        out[condition]["ci95"] = {
            key: ci
            for key in CI_METRICS
            if (ci := bootstrap_subject_ci(row.get(key) for row in rows)) is not None
        }
    return out


def load_model_records(
    output_dir: Path,
    slug: str,
    representation: str,
    subjects: Sequence[str],
    recovery: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Respuestas por sujeto de un modelo (con recuperación opcional)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for subject in subjects:
        path = output_dir / "responses" / slug / f"{subject}.{representation}.jsonl"
        records = list(read_jsonl(path))
        if not records:
            continue
        if recovery:
            records, _ = apply_recovery(records, decision_mode="sleep4")
        out[subject] = records
    return out


def compare_models(
    models: Sequence[str],
    output_dir: Path,
    representations: Sequence[str] = ("numeric", "text"),
    subjects: Sequence[str] | None = None,
    recovery: bool = False,
    cohort: str = "pilot",
) -> dict[str, Any]:
    """Informe comparativo de ``models`` sobre una cohorte congelada."""
    if cohort not in COHORTS:
        raise ValueError(f"cohorte desconocida {cohort!r}; disponibles: {sorted(COHORTS)}")
    wanted = list(subjects) if subjects else list(COHORTS[cohort])
    labels = _load_labels(output_dir, wanted)
    report: dict[str, Any] = {
        "task": "sleep4",
        "cohort": cohort,
        "output_dir": str(output_dir),
        "models": list(models),
        "representations": list(representations),
        "recovery": bool(recovery),
        "subjects_requested": wanted,
        "by_representation": {},
        "note": (
            f"cohorte {cohort} ({len(wanted)} sujetos): media y rango [mín-máx] "
            "entre sujetos"
            + (
                "; se añade IC 95 % bootstrap sobre sujetos para abstención, "
                "accuracy con la abstención como error y balanced."
                if len(wanted) >= MIN_SUBJECTS_FOR_CI
                else "; sin intervalos de confianza: pocos sujetos no los sostienen."
            )
        ),
    }
    subject_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for representation in representations:
        block: dict[str, Any] = {}
        subject_metrics[representation] = {}
        for model in models:
            slug = model_slug(model)
            records = load_model_records(output_dir, slug, representation, wanted, recovery)
            if not records:
                block[model] = {"status": "sin respuestas", "slug": slug}
                continue
            per_subject = compute_metrics(records, labels)
            subject_metrics[representation][model] = per_subject
            block[model] = {
                "slug": slug,
                "subjects": sorted(per_subject.keys()),
                "n_records": sum(len(rows) for rows in records.values()),
                "conditions": aggregate_by_condition(per_subject),
            }
        report["by_representation"][representation] = block
    report["deltas"] = _deltas(report)
    pairs = _paired_deltas(subject_metrics)
    for representation, block in pairs.items():
        if representation in report["deltas"]:
            report["deltas"][representation]["pairs"] = block
        else:
            report["deltas"][representation] = {"pairs": block}
    return report


#: Métricas con delta pareado (se re-muestrean sujetos, no ventanas).
PAIRED_METRICS: tuple[str, ...] = (
    "abstain_rate",
    "accuracy",
    "accuracy_incl_abstain",
    "balanced_accuracy",
    "coverage",
)


def _metric_value(metrics: Mapping[str, Any], key: str) -> Any:
    """Valor de una métrica en un bloque de condición (abstención con dos nombres)."""
    if key == "abstain_rate":
        value = metrics.get("abstain_rate")
        return metrics.get("abstention_rate") if value is None else value
    return metrics.get(key)


def _paired_deltas(
    subject_metrics: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]],
) -> dict[str, dict[str, Any]]:
    """Delta pareado ``otro − referencia`` para **todos** los pares de modelos.

    Sin esto solo se podía leer el par (primero, último) y las comparaciones de
    tres modelos se quedaban sin la mitad de las filas.
    """
    out: dict[str, dict[str, Any]] = {}
    for representation, by_model in subject_metrics.items():
        models = [model for model in by_model if by_model[model]]
        pairs: dict[str, dict[str, Any]] = {}
        for index, reference in enumerate(models):
            for other in models[index + 1 :]:
                left_all = by_model[reference]
                right_all = by_model[other]
                subjects = sorted(set(left_all) & set(right_all))
                if not subjects:
                    continue
                conditions = sorted(
                    set.intersection(*(set(left_all[s]) for s in subjects))
                    & set.intersection(*(set(right_all[s]) for s in subjects))
                )
                rows: dict[str, Any] = {}
                for condition in conditions:
                    row: dict[str, Any] = {}
                    for metric in PAIRED_METRICS:
                        result = paired_bootstrap_delta(
                            [_metric_value(left_all[s][condition], metric) for s in subjects],
                            [_metric_value(right_all[s][condition], metric) for s in subjects],
                        )
                        if result is not None:
                            row[metric] = result
                    rows[condition] = row
                pairs[f"{other} - {reference}"] = {
                    "reference": reference,
                    "other": other,
                    "n_subjects": len(subjects),
                    "conditions": rows,
                }
        if pairs:
            out[representation] = pairs
    return out


def _deltas(report: Mapping[str, Any]) -> dict[str, Any]:
    """Diferencia (último modelo − primero) por condición y representación."""
    models = report.get("models") or []
    if len(models) < 2:
        return {}
    reference, other = models[0], models[-1]
    out: dict[str, Any] = {}
    for representation, block in report.get("by_representation", {}).items():
        left = block.get(reference, {}).get("conditions") or {}
        right = block.get(other, {}).get("conditions") or {}
        rows: dict[str, Any] = {}
        for condition, metrics in right.items():
            if condition not in left:
                continue
            row = {}
            for key in ("accuracy", "accuracy_incl_abstain", "balanced_accuracy", "coverage"):
                a = (left[condition].get(key) or {}).get("mean")
                b = (metrics.get(key) or {}).get("mean")
                row[key] = None if a is None or b is None else float(b) - float(a)
            rows[condition] = row
        if rows:
            out[representation] = {"reference": reference, "other": other, "conditions": rows}
    return out


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "  —  "
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "  —  "
    if not math.isfinite(number):
        return "  —  "
    return f"{number:.{digits}f}"


def print_report(report: Mapping[str, Any]) -> None:
    """Tabla larga por (representación, condición, modelo) y bloque de deltas."""
    models = report.get("models") or []
    print("\n[DREAMT] Comparación de modelos")
    header = (
        f"{'vista':7} {'condición':20} {'modelo':16} {'n':>3} {'acc':>6} {'acc*':>6} "
        f"{'bal':>6} {'cov':>6} {'abs':>6} {'sop':>6} {'coh':>6} {'cert':>7}"
    )
    print(header)
    print("-" * len(header))
    for representation, block in report.get("by_representation", {}).items():
        for model in models:
            entry = block.get(model) or {}
            conditions = entry.get("conditions") or {}
            if not conditions:
                print(
                    f"{representation:7} {'—':20} {model:16} "
                    f"{'sin respuestas':>40}"
                )
                continue
            for condition, metrics in conditions.items():
                n = (metrics.get("n") or {}).get("mean")
                print(
                    f"{representation:7} {condition:20} {model:16} "
                    f"{(f'{n:.0f}' if n is not None else '—'):>3} "
                    f"{_num((metrics.get('accuracy') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('accuracy_incl_abstain') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('balanced_accuracy') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('coverage') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('abstain_rate') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('supported_rate') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('coherence_rate') or {}).get('mean')):>6} "
                    f"{_num((metrics.get('certainty_index') or {}).get('mean'), 2):>7}"
                )
        # Rangos entre sujetos: la dispersión es el dato, no un adorno.
        for model in models:
            entry = block.get(model) or {}
            for condition, metrics in (entry.get("conditions") or {}).items():
                acc = metrics.get("accuracy") or {}
                if acc.get("n", 0) > 1:
                    print(
                        f"{'':7} {condition:20} {model:16} acc rango "
                        f"[{_num(acc.get('min'))}; {_num(acc.get('max'))}] "
                    f"({acc['n']} sujetos)"
                    )
                ci = metrics.get("ci95") or {}
                if ci:
                    parts = []
                    for key, label in (
                        ("abstain_rate", "abs"),
                        ("accuracy_incl_abstain", "acc*"),
                        ("balanced_accuracy", "bal"),
                    ):
                        block_ci = ci.get(key)
                        if block_ci:
                            parts.append(
                                f"{label} [{_num(block_ci.get('lo'))}; "
                                f"{_num(block_ci.get('hi'))}]"
                            )
                    if parts:
                        print(
                            f"{'':7} {condition:20} {'':16} IC95 sujetos: "
                            + " · ".join(parts)
                        )

    deltas = report.get("deltas") or {}
    if deltas:
        print("\n[DREAMT] Diferencias por condición (último modelo − primero)")
        for representation, block in deltas.items():
            print(f"  {representation}: {block['other']} − {block['reference']}")
            for condition, row in block["conditions"].items():
                print(
                    f"    {condition:20} Δacc {_num(row.get('accuracy')):>7} "
                    f"Δacc* {_num(row.get('accuracy_incl_abstain')):>7} "
                    f"Δbal {_num(row.get('balanced_accuracy')):>7} "
                    f"Δcov {_num(row.get('coverage')):>7}"
                )

    paired = {
        representation: block["pairs"]
        for representation, block in deltas.items()
        if block.get("pairs")
    }
    if paired:
        print(
            "\n[DREAMT] Todos los pares de modelos "
            "(delta pareado; positivo = el segundo valor es mayor)"
        )
        for representation, pairs in paired.items():
            print(f"  {representation}")
            for name, pair in pairs.items():
                print(f"    {name}  (n={pair.get('n_subjects')} sujetos)")
                for condition, row in pair["conditions"].items():
                    cells: list[str] = []
                    for metric, label in (
                        ("abstain_rate", "Δabs"),
                        ("accuracy", "Δacc"),
                        ("accuracy_incl_abstain", "Δacc*"),
                        ("balanced_accuracy", "Δbal"),
                        ("coverage", "Δcov"),
                    ):
                        entry = row.get(metric)
                        if not entry:
                            cells.append(f"{label}   —  ")
                            continue
                        cell = f"{label} {_num(entry.get('delta')):>7}"
                        ci = entry.get("ci")
                        if ci:
                            cell += (
                                f" [{_num(ci.get('lo'))}; {_num(ci.get('hi'))}]"
                            )
                        cells.append(cell)
                    print(f"      {condition:20} " + " · ".join(cells))
    print(f"\n{report.get('note', '')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara modelos en el piloto DREAMT (estadios de sueño)."
    )
    parser.add_argument("--models", required=True, help="modelos separados por comas")
    parser.add_argument(
        "--cohort",
        choices=tuple(COHORTS),
        default="pilot",
        help="cohorte congelada: pilot (4 sujetos) o scale (12 sujetos)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="carpeta de trabajo (por defecto, la de la cohorte)",
    )
    parser.add_argument(
        "--representation",
        default="both",
        choices=("numeric", "text", "both"),
    )
    parser.add_argument("--subjects", default=None, help="subconjunto de la cohorte")
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="recuperar decisión de respuestas con claims mal formados",
    )
    parser.add_argument("--output", default=None, help="informe JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    if len(models) < 2:
        raise SystemExit("--models necesita al menos dos modelos para comparar.")
    representations = (
        ("numeric", "text") if args.representation == "both" else (args.representation,)
    )
    subjects = None
    if args.subjects:
        subjects = [
            part.strip().upper()
            for part in args.subjects.split(",")
            if part.strip()
        ]
        members = COHORTS[args.cohort]
        outside = [name for name in subjects if name not in members]
        if outside:
            raise SystemExit(
                f"sujetos fuera de la cohorte {args.cohort} {members}: {outside}"
            )
    report = compare_models(
        models,
        Path(args.dir or DEFAULT_OUTPUTS[args.cohort]),
        representations=representations,
        subjects=subjects,
        recovery=args.recovery,
        cohort=args.cohort,
    )
    print_report(report)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[DREAMT] informe escrito en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
