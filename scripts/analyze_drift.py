#!/usr/bin/env python
"""Deriva de la decisión bajo degradación: hacia dónde se mueve el modelo.

Responde, sin llamadas al LLM, a tres preguntas que los informes por condición
no pueden responder porque miran cada condición por separado:

1. **¿Hacia qué clase se desplaza?** Distribución predicha frente al prior real
   y **tasa de reclamación** (``P(predicha)/P(real)``) por condición.
2. **¿Cuánto del error lo induce la degradación?** Matriz pareada
   ``C0 → Cx`` por ventana y descomposición 2×2: error inducido
   (``acierta → falla``) frente a error preexistente (``falla → falla``).
3. **¿La deriva es de la información o de la política de decisión?** Las mismas
   tablas sobre la **referencia supervisada** features-only, que ve la misma
   evidencia. Si la referencia no deriva y el LLM sí, el desplazamiento no se
   explica por falta de información.

Añade, opcionalmente, el cruce con **claims** (¿las ventanas que derivan citan
evidencia a favor o en contra de la nueva clase?) y un **bootstrap por sujeto**
para el error inducido.

Uso:
    python scripts/analyze_drift.py \\
        --model qwen2.5:7b --representation numeric \\
        --decision-mode binary --nonstress-variant schmidt --temperature exclude \\
        --output results/output/WESAD/PHASE2/drift_qwen2_5_7b.json

El corpus debe existir en ``results/output/WESAD/PHASE2/degradation_*``; el
script solo lee y no modifica nada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.provenance import opaque_window_id, utc_now
from sources.phase2.binary_variants import (
    DEFAULT_NONSTRESS_VARIANT,
    variant_classes,
)
from sources.phase2.response_schema import DECISION_MODE_STRESS_BINARY
from sources.phase3.calibration import group_kfold_indices
from sources.phase3.claims import (
    parse_prompt_features,
    response_coherence,
    verify_claims,
)
from sources.phase3.degradation import (
    EFFECTIVE_CONDITIONS,
    condition_matrix,
    load_features,
    resolve_degradation_dir,
)
from sources.phase3.drift import (
    build_decision_table,
    coherence_summary,
    normalize_decision,
    prediction_marginal,
    resolve_decision_mode,
    table_from_predictions,
    transition_analysis,
    truth_in_space,
    window_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2"
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"

USABLE_STATUSES = ("valid_response", "valid_abstention")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_labels() -> dict[str, str]:
    """Etiqueta real por **identificador completo** de ventana.

    Las features y la referencia usan el identificador completo
    (``WESAD:S5:w00001``); las respuestas guardan el opaco
    (``win_…``). Las tablas se construyen sobre el opaco y la referencia
    convierte al guardar sus predicciones, para que ambas coincidan.
    """
    labels: dict[str, str] = {}
    path = INGEST_DIR / "groundtruth" / "labels.jsonl"
    if not path.is_file():
        return labels
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        labels[record["window_id"]] = label
    return labels


def labels_by_opaque(labels: Mapping[str, str]) -> dict[str, str]:
    """Reindexa por identificador opaco (lo que guardan las respuestas)."""
    return {opaque_window_id(window_id): label for window_id, label in labels.items()}


def load_records(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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


def provenance_of(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Procedencia declarada por el corpus (la primera respuesta que la traiga)."""
    for record in records:
        flags = {
            "decision_mode": record.get("decision_mode"),
            "nonstress_variant": record.get("nonstress_variant"),
            "include_temperature": record.get("include_temperature"),
            "representation": record.get("representation"),
            "model": record.get("model"),
        }
        if any(value is not None for value in flags.values()):
            return flags
    return {}


# ---------------------------------------------------------------------------
# Referencia supervisada (mismas features, misma estandarización)
# ---------------------------------------------------------------------------
def llm_feature_names(
    subjects: Sequence[str], include_temperature: bool
) -> list[str]:
    names: set[str] = set()
    for subject in subjects:
        rows = load_features(
            INGEST_DIR,
            "C0_clean",
            subject,
            include_temperature=include_temperature,
        )
        for values in rows.values():
            names.update(values)
    return sorted(names)


def reference_predictions(
    conditions: Sequence[str],
    subjects: Sequence[str],
    labels: Mapping[str, str],
    *,
    decision_mode: str,
    nonstress_classes: Sequence[str] | None,
    include_temperature: bool,
) -> dict[str, dict[str, str]]:
    """Predicciones fuera de pliegue (CV por sujeto) de la referencia logística."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_names = llm_feature_names(subjects, include_temperature)
    predictions: dict[str, dict[str, str]] = {}
    for condition in conditions:
        matrix, groups, window_ids = condition_matrix(
            INGEST_DIR,
            condition,
            list(subjects),
            feature_names,
            include_temperature=include_temperature,
        )
        truth = [
            truth_in_space(labels.get(window_id), decision_mode, nonstress_classes)
            for window_id in window_ids
        ]
        keep = [i for i, value in enumerate(truth) if value is not None]
        if not keep:
            continue
        matrix = matrix[keep]
        groups = [groups[i] for i in keep]
        kept_ids = [window_ids[i] for i in keep]
        y = np.array([truth[i] for i in keep], dtype=object)
        if len(set(groups)) < 2 or len(set(y)) < 2:
            continue
        folds = group_kfold_indices(groups, n_splits=min(5, len(set(groups))))
        fold_predictions: dict[str, str] = {}
        for train_idx, test_idx in folds:
            if len(set(y[train_idx])) < 2:
                continue
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
            model.fit(matrix[train_idx], y[train_idx])
            for index, predicted in zip(test_idx, model.predict(matrix[test_idx])):
                # La tabla de decisiones está indexada por identificador opaco.
                fold_predictions[opaque_window_id(kept_ids[index])] = str(predicted)
        predictions[condition] = fold_predictions
    return predictions


# ---------------------------------------------------------------------------
# Bootstrap por sujeto
# ---------------------------------------------------------------------------
def bootstrap_ratio_by_subject(
    by_subject: Mapping[str, Mapping[str, int]],
    numerator: str,
    denominator_parts: Sequence[str],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> list[float]:
    """IC 95 % del cociente, remuestreando **sujetos** (no ventanas)."""
    subjects = [s for s in sorted(by_subject) if by_subject[s].get(denominator_parts[0])
                or sum(by_subject[s].get(part, 0) for part in denominator_parts)]
    if len(subjects) < 2:
        return []
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        num = sum(int(by_subject[s].get(numerator, 0)) for s in sample)
        den = sum(
            int(by_subject[s].get(part, 0)) for s in sample for part in denominator_parts
        )
        if den:
            values.append(num / den)
    if not values:
        return []
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


# ---------------------------------------------------------------------------
# Claims por grupo de ventanas
# ---------------------------------------------------------------------------
def claims_by_group(
    records: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Sequence[str]],
    condition: str,
    *,
    min_responses: int = 5,
) -> dict[str, Any]:
    """Coherencia de claims en las ventanas que derivan frente a las estables."""
    by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (record.get("window_id_opaque"), record.get("condition_id"))
        if key[1] != condition or key[0] is None:
            continue
        if record.get("status") not in USABLE_STATUSES:
            continue
        by_key.setdefault(key, record)

    report: dict[str, Any] = {}
    for group in ("drifted", "stable_correct", "stable_wrong"):
        blocks: list[dict[str, Any]] = []
        for window_id in groups.get(group, ()):
            record = by_key.get((window_id, condition))
            if not record:
                continue
            claims = record.get("claims") or []
            prompt = record.get("prompt") or ""
            if not claims or not prompt:
                continue
            features = set(parse_prompt_features(prompt))
            verdicts = verify_claims(claims, prompt)
            block = response_coherence(
                claims,
                verdicts,
                normalize_decision(record.get("decision")),
                features=features,
            )
            block["n_supported"] = sum(
                1 for verdict in verdicts if verdict.category == "supported"
            )
            blocks.append(block)
        if blocks:
            report[group] = coherence_summary(blocks, min_responses=min_responses)
    return report


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
_PREFERRED_ORDER = EFFECTIVE_CONDITIONS


def _ordered_conditions(available: set[str] | Sequence[str]) -> list[str]:
    """Orden fijo para las condiciones conocidas; el resto, alfabético."""
    known = [c for c in _PREFERRED_ORDER if c in available]
    return known + sorted(c for c in available if c not in _PREFERRED_ORDER)


def analyze_corpus(
    directory: Path,
    *,
    decision_mode: str,
    nonstress_variant: str,
    include_temperature: bool,
    conditions: Sequence[str] | None,
    base_condition: str,
    with_reference: bool,
    with_claims: bool,
    n_boot: int,
) -> dict[str, Any]:
    records = load_records(directory)
    labels_full = load_labels()
    labels = labels_by_opaque(labels_full)
    classes = variant_classes(nonstress_variant)
    table = build_decision_table(
        records,
        labels,
        decision_mode=decision_mode,
        nonstress_classes=classes,
        conditions=conditions,
    )
    available = _ordered_conditions(
        set().union(*(window["decisions"] for window in table["windows"].values()))
        if table["windows"]
        else set()
    )
    subjects = sorted({w.get("subject") for w in table["windows"].values() if w.get("subject")})
    marginal = prediction_marginal(table, available)
    transitions = transition_analysis(
        table, base=base_condition, conditions=available
    )

    report: dict[str, Any] = {
        "dir": str(directory),
        "model": next((r.get("model") for r in records if r.get("model")), ""),
        "representation": next(
            (r.get("representation") for r in records if r.get("representation")), ""
        ),
        "n_records": len(records),
        "n_windows": table["n_windows"],
        "dropped": table["dropped"],
        "subjects": subjects,
        "conditions": available,
        "marginal": marginal,
        "transitions": transitions,
        "induced_error_ci": {},
    }

    # Espacio de etiquetas realmente usado: si el modelo nunca emite una clase,
    # su deriva solo puede ir hacia las que usa. Comparar su matriz con la de
    # una referencia que sí usa todas exige declararlo.
    used: set[str] = set()
    for entry in marginal["conditions"].values():
        used.update(entry["predicted_decided"])
    report["predicted_classes_ever"] = sorted(used)
    missing_classes = sorted(set(table["classes"]) - used)
    report["label_space_note"] = (
        "el modelo nunca predice "
        + ", ".join(missing_classes)
        + "; su deriva solo puede ir hacia "
        + ", ".join(sorted(used))
        + " o abstención"
        if missing_classes and used
        else None
    )

    for condition, entry in transitions["conditions"].items():
        decomposition = entry.get("decomposition") or {}
        if decomposition.get("n_with_truth"):
            interval = bootstrap_ratio_by_subject(
                entry.get("by_subject") or {},
                "n_correct_wrong",
                ("n_correct_wrong", "n_wrong_wrong"),
                n_boot=n_boot,
            )
            if interval:
                report["induced_error_ci"][condition] = interval

    if with_claims and decision_mode == DECISION_MODE_STRESS_BINARY:
        report["claims"] = {}
        for condition in transitions["conditions"]:
            groups = window_groups(table, base=base_condition, condition=condition)
            report["claims"][condition] = claims_by_group(
                records, groups, condition
            )
    elif with_claims:
        report["claims"] = {
            "skipped": "el cruce de claims solo está definido para la tarea binaria"
        }

    if with_reference:
        ref_conditions = [
            condition
            for condition in transitions["conditions"]
            if condition in EFFECTIVE_CONDITIONS
        ]
        predictions = reference_predictions(
            ref_conditions,
            subjects,
            labels_full,
            decision_mode=decision_mode,
            nonstress_classes=classes,
            include_temperature=include_temperature,
        )
        truth_by_window = {
            window_id: window["truth"]
            for window_id, window in table["windows"].items()
        }
        subject_by_window = {
            window_id: window.get("subject", "")
            for window_id, window in table["windows"].items()
        }
        reference_table = table_from_predictions(
            predictions,
            truth_by_window,
            decision_mode=decision_mode,
            subject_by_window=subject_by_window,
        )
        reference_transitions = transition_analysis(
            reference_table, base=base_condition, conditions=ref_conditions
        )
        report["reference"] = {
            "n_windows": reference_table["n_windows"],
            "marginal": prediction_marginal(reference_table, ref_conditions),
            "transitions": reference_transitions,
            "induced_error_ci": {},
        }
        for condition, entry in reference_transitions["conditions"].items():
            decomposition = entry.get("decomposition") or {}
            if decomposition.get("n_with_truth"):
                interval = bootstrap_ratio_by_subject(
                    entry.get("by_subject") or {},
                    "n_correct_wrong",
                    ("n_correct_wrong", "n_wrong_wrong"),
                    n_boot=n_boot,
                )
                if interval:
                    report["reference"]["induced_error_ci"][condition] = interval
    else:
        report["reference"] = None

    return report


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, float):
        if np.isnan(value):
            return "  —  "
        return f"{value:.{digits}f}"
    return str(value)


def print_report(report: Mapping[str, Any], base_condition: str) -> None:
    print(f"\n=== {report['dir']}")
    print(
        f"modelo={report.get('model') or '?'} "
        f"representación={report.get('representation') or '?'} "
        f"ventanas={report['n_windows']} registros={report['n_records']}"
    )
    if report.get("dropped"):
        print(f"descartado: {report['dropped']}")
    if report.get("label_space_note"):
        print(f"AVISO: {report['label_space_note']}")

    print("\n-- Marginal predicha (decididas) frente al prior")
    marginal = report["marginal"]["conditions"]
    classes = report["marginal"]["decision_space"]
    header = f"{'condición':16s} {'n':>4s} {'abst':>6s} " + " ".join(
        f"{klass:>11s}" for klass in classes
    )
    print(header)
    for condition, entry in marginal.items():
        cells = " ".join(
            f"{_fmt(entry['predicted_decided'].get(klass, 0.0)):>11s}" for klass in classes
        )
        print(
            f"{condition:16s} {entry['n']:4d} {_fmt(entry['abstain_share']):>6s} {cells}"
        )
    truth = next(iter(marginal.values()), {}).get("truth") or {}
    print("prior real:      " + " ".join(f"{k}={_fmt(v)}" for k, v in truth.items()))
    print("tasa de reclamación (predicha/real):")
    for condition, entry in marginal.items():
        cells = " ".join(
            f"{klass}={_fmt(entry['claim_rate'].get(klass))}" for klass in classes
        )
        print(f"  {condition:16s} {cells}")

    print("\n-- Transiciones pareadas (base = %s)" % base_condition)
    for condition, entry in report["transitions"]["conditions"].items():
        decomposition = entry["decomposition"]
        interval = (report.get("induced_error_ci") or {}).get(condition)
        print(
            f"\n  {condition}: pares={entry['n_decided_pairs']} "
            f"cambian={entry['n_changed']} ({_fmt(entry['changed_share'])})"
        )
        print(
            "    bien→bien={correct_correct} bien→mal={correct_wrong} "
            "mal→bien={wrong_correct} mal→mal={wrong_wrong}".format(**decomposition)
        )
        print(
            "    error inducido={induced} error preexistente={preexisting} "
            "reparado={repaired}".format(
                induced=_fmt(decomposition["induced_error_share"]),
                preexisting=_fmt(decomposition["preexisting_error_share"]),
                repaired=_fmt(decomposition["repaired_share"]),
            )
            + (f"  IC95 inducido=[{interval[0]:.3f}; {interval[1]:.3f}]" if interval else "")
        )
        if entry.get("abstain_flow"):
            print(f"    flujos de abstención: {entry['abstain_flow']}")
        direction = ", ".join(
            f"{item['from']}→{item['to']}={item['n']}" for item in entry["direction"][:4]
        )
        if direction:
            print(f"    dirección: {direction}")
        for klass, bucket in entry["by_truth"].items():
            if bucket.get("drifted_from_correct"):
                print(
                    f"    {klass}: aciertan en limpio y fallan → "
                    f"{bucket['drifted_destination']}"
                )

    if report.get("reference"):
        print("\n-- Referencia supervisada (mismas features, CV por sujeto)")
        for condition, entry in report["reference"]["transitions"]["conditions"].items():
            decomposition = entry["decomposition"]
            interval = (report["reference"].get("induced_error_ci") or {}).get(condition)
            print(
                f"  {condition:16s} acc {_fmt(decomposition['accuracy_before'])} → "
                f"{_fmt(decomposition['accuracy_after'])}  "
                f"inducido {_fmt(decomposition['induced_error_share'])}"
                + (f" IC95=[{interval[0]:.3f}; {interval[1]:.3f}]" if interval else "")
            )

    if report.get("claims"):
        print("\n-- Claims en las ventanas que derivan")
        if isinstance(report["claims"], dict) and report["claims"].get("skipped"):
            print(f"  {report['claims']['skipped']}")
        else:
            for condition, groups in report["claims"].items():
                for group, block in groups.items():
                    print(
                        f"  {condition:16s} {group:14s} n={block['n_responses']:3d} "
                        f"coherencia={_fmt(block.get('coherence_rate'))} "
                        f"contra_respaldados={_fmt(block.get('against_supported_share'))} "
                        f"sin_confirmación={_fmt(block.get('zero_confirmation_share'))}"
                    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", action="append", default=[],
                        help="directorio de respuestas (repetible); si se omite, se resuelve")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--representation", default="numeric",
                        choices=["numeric", "text"])
    parser.add_argument("--decision-mode", default="",
                        help="affective | binary (vacío = deducir del corpus)")
    parser.add_argument("--nonstress-variant", default="")
    parser.add_argument("--temperature", default="", choices=["", "include", "exclude"])
    parser.add_argument("--conditions", default="",
                        help="lista separada por comas (por defecto las efectivas)")
    parser.add_argument("--base", default="C0_clean")
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-claims", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    conditions = (
        tuple(item.strip() for item in args.conditions.split(",") if item.strip())
        if args.conditions
        else None
    )

    directories: list[Path] = [Path(item) for item in args.dir]
    meta_mode = args.decision_mode
    meta_variant = args.nonstress_variant
    meta_temperature = args.temperature or None
    if not directories:
        directories = [
            resolve_degradation_dir(
                PHASE2_DIR,
                model=args.model,
                representation=args.representation,
                decision_mode=args.decision_mode or None,
                nonstress_variant=args.nonstress_variant or None,
                temperature_mode=args.temperature or None,
            )
        ]
    if not directories:
        print("No hay corpus que analizar.", file=sys.stderr)
        return 1

    reports: list[dict[str, Any]] = []
    for directory in directories:
        records = load_records(directory)
        flags = provenance_of(records)
        mode = resolve_decision_mode(meta_mode or flags.get("decision_mode") or None)
        variant = meta_variant or flags.get("nonstress_variant") or DEFAULT_NONSTRESS_VARIANT
        include_temperature = (
            (meta_temperature == "include")
            if meta_temperature
            else bool(flags.get("include_temperature", True))
        )
        report = analyze_corpus(
            directory,
            decision_mode=mode,
            nonstress_variant=variant,
            include_temperature=include_temperature,
            conditions=conditions,
            base_condition=args.base,
            with_reference=not args.no_reference,
            with_claims=not args.no_claims,
            n_boot=args.bootstrap,
        )
        report.update(
            {
                "decision_mode": mode,
                "nonstress_variant": variant,
                "include_temperature": include_temperature,
                "base_condition": args.base,
            }
        )
        reports.append(report)
        print_report(report, args.base)

    payload = {
        "schema": "drift_report/1.0",
        "generated_at": utc_now(),
        "bootstrap": args.bootstrap,
        "corpora": reports,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nInforme escrito en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
