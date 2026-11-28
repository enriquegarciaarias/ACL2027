"""Contraste pareado entre dos corridas de PhysioNet 2017 (motor A frente a B).

Motivación: el corpus histórico se generó con Ollama 0.30.7 y no registró el
motor; la corrida ``--tag srv`` se genera con 0.40.6. Comparar dos tablas de
métricas por separado invita a leer diferencias que son de composición del
conjunto, no del motor. Este script empareja **registro a registro** (mismos
``record_id``, mismas features, mismo prompt) y reporta:

- validez (``valid_response``/``valid_abstention``/otras) por corrida;
- cambios de decisión (matriz de transiciones);
- corrección con **McNemar exacto** sobre los pares discordantes;
- abstención por clase real con intervalo de Wilson;
- respaldo de claims (recalculado desde el prompt guardado, no desde el informe);
- certeza lingüística;
- diferencias pareadas con bootstrap sobre registros.

No llama al LLM y no reescribe nada del corpus: solo lee y escribe un JSON.

Uso::

    python scripts/compare_physionet_runs.py --tag-b srv
    python scripts/compare_physionet_runs.py --tag-a hist --tag-b srv --no-claims

Salida por defecto: ``<dir>/physionet2017_comparison_<tag_b>.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sources.common.provenance import engine_from_provenance
from sources.phase3.calibration import wilson_interval
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.claims import summarize_verdicts, verify_claims

DEFAULT_DIR = REPO_ROOT / "results" / "output" / "PHYSIONET2017" / "P3"

RHYTHM_CLASSES = ("normal", "af", "other-rhythm", "noisy")
VALID_STATUSES = ("valid_response", "valid_abstention")


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def run_paths(base_dir: Path, tag: str = "") -> dict[str, Path]:
    """Rutas de una corrida, **delegando** en el analizador.

    Se carga ``scripts/analyze_p3_physionet.py`` en lugar de duplicar el
    nombrado: una divergencia entre el analizador y este comparador produciría
    exactamente el fallo silencioso que ``--tag`` vino a corregir.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_analyze_p3_physionet", Path(__file__).with_name("analyze_p3_physionet.py")
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensivo
        raise SystemExit("No se pudo cargar scripts/analyze_p3_physionet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_paths(base_dir, tag)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Carga un JSONL completo (ignora líneas vacías o truncadas)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dedupe_responses(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Una respuesta por registro (gana la última, como en la reanudación)."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["record_id"]] = dict(row)
    return latest


def load_labels(path: Path) -> dict[str, str]:
    """``record_id -> etiqueta de experto`` de una corrida."""
    if not path.is_file():
        return {}
    return {row["record_id"]: row["label"] for row in load_jsonl(path)}


def load_run(base_dir: Path, tag: str) -> dict[str, Any]:
    """Respuestas, etiquetas y motor de una corrida."""
    paths = run_paths(base_dir, tag)
    if not paths["responses"].is_file():
        suffix = f" --tag {tag}" if tag else ""
        raise SystemExit(
            f"No existe {paths['responses']}. Genera esa corrida{suffix} o "
            "revisa --dir."
        )
    return {
        "tag": tag,
        "engine": engine_from_provenance(paths["provenance"]),
        "responses": dedupe_responses(load_jsonl(paths["responses"])),
        "labels": load_labels(paths["labels"]),
    }


# ---------------------------------------------------------------------------
# Métricas por corrida
# ---------------------------------------------------------------------------
def is_valid(row: Mapping[str, Any]) -> bool:
    """Respuesta parseable (decisión o abstención válida)."""
    return row.get("status") in VALID_STATUSES


def is_abstain(row: Mapping[str, Any]) -> bool:
    """Abstención explícita (estado o decisión)."""
    return row.get("status") == "valid_abstention" or row.get("decision") == "abstain"


def decision_label(row: Mapping[str, Any]) -> str:
    """Etiqueta de decisión normalizada (la abstención no depende del campo)."""
    return "abstain" if is_abstain(row) else str(row.get("decision"))


def correctness_table(
    responses: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str]
) -> dict[str, Any]:
    """Corrección, abstención por clase y cobertura de una corrida."""
    status = Counter(str(row.get("status")) for row in responses.values())
    valid = [row for row in responses.values() if is_valid(row)]
    labelled = [row for row in valid if row.get("record_id") in labels]
    decided = [row for row in labelled if not is_abstain(row)]
    hits = sum(1 for row in decided if row.get("decision") == labels[row["record_id"]])
    abstain = [row for row in labelled if is_abstain(row)]

    per_class: dict[str, Any] = {}
    for label in RHYTHM_CLASSES:
        rows = [row for row in labelled if labels[row["record_id"]] == label]
        rows_decided = [row for row in rows if not is_abstain(row)]
        class_hits = sum(
            1 for row in rows_decided if row.get("decision") == label
        )
        n_abstain = len(rows) - len(rows_decided)
        low, high = wilson_interval(n_abstain, len(rows))
        per_class[label] = {
            "n": len(rows),
            "decided": len(rows_decided),
            "abstentions": n_abstain,
            "abstention_rate": n_abstain / len(rows) if rows else float("nan"),
            "abstention_ci95": [low, high],
            "recall": class_hits / len(rows_decided) if rows_decided else float("nan"),
        }

    recalls = [v["recall"] for v in per_class.values() if v["decided"]]
    return {
        "n": len(responses),
        "status": dict(status),
        "n_valid": len(valid),
        "n_invalid": len(responses) - len(valid),
        "n_labelled": len(labelled),
        "n_decided": len(decided),
        "n_abstain": len(abstain),
        "coverage": len(decided) / len(labelled) if labelled else float("nan"),
        "accuracy": hits / len(decided) if decided else float("nan"),
        "accuracy_incl_abstain": hits / len(labelled) if labelled else float("nan"),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else float("nan"),
        "decisions": dict(Counter(str(row.get("decision")) for row in valid)),
        "per_class": per_class,
    }


def claims_table(
    responses: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Respaldo de claims recalculado contra el prompt de cada respuesta."""
    verdicts: list[Any] = []
    per_class: dict[str, list[Any]] = {label: [] for label in RHYTHM_CLASSES}
    by_record: dict[str, float] = {}
    for row in responses.values():
        if not is_valid(row):
            continue
        claims = row.get("claims") or []
        if not claims:
            continue
        record_verdicts = verify_claims(claims, row.get("prompt", "") or "")
        verdicts.extend(record_verdicts)
        share = (
            sum(1 for v in record_verdicts if v.category == "supported")
            / len(record_verdicts)
            if record_verdicts
            else float("nan")
        )
        by_record[row["record_id"]] = share
    summary = summarize_verdicts(verdicts)
    summary["per_record_supported"] = by_record
    return summary


def certainty_table(
    responses: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Certeza lingüística global y por clase real."""
    per_class: dict[str, list[float]] = {label: [] for label in RHYTHM_CLASSES}
    values: list[float] = []
    hedges: list[float] = []
    boosters: list[float] = []
    by_record: dict[str, float] = {}
    for row in responses.values():
        if not is_valid(row):
            continue
        certainty = linguistic_certainty(row.get("reasoning") or "")
        if not certainty["n_words"]:
            continue
        index = float(certainty["certainty_index"])
        values.append(index)
        hedges.append(float(certainty["hedges"]))
        boosters.append(float(certainty["boosters"]))
        by_record[row["record_id"]] = index
        label = labels.get(row["record_id"])
        if label in per_class:
            per_class[label].append(index)

    def _mean(items: Sequence[float]) -> float:
        return sum(items) / len(items) if items else float("nan")

    return {
        "n": len(values),
        "certainty_index_mean": _mean(values),
        "hedges_mean": _mean(hedges),
        "boosters_mean": _mean(boosters),
        "per_class": {label: _mean(items) for label, items in per_class.items()},
        "per_record": by_record,
    }


# ---------------------------------------------------------------------------
# Contraste pareado
# ---------------------------------------------------------------------------
def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, Any]:
    """McNemar exacto sobre pares dicotómicos (corrección A frente a B)."""
    a_right_b_wrong = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    a_wrong_b_right = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n_discordant = a_right_b_wrong + a_wrong_b_right
    if n_discordant:
        p_value = float(
            binomtest(n_discordant - a_right_b_wrong, n_discordant, 0.5).pvalue
        )
    else:
        p_value = 1.0
    return {
        "n_pairs": len(a_correct),
        "a_only_correct": a_right_b_wrong,
        "b_only_correct": a_wrong_b_right,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]], n_boot: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """Diferencia media B−A con IC 95 % por bootstrap sobre registros."""
    usable = [(a, b) for a, b in pairs if a == a and b == b]  # descarta NaN
    if not usable:
        return {"n": 0, "delta": float("nan"), "ci95": [float("nan")] * 2}
    deltas = [b - a for a, b in usable]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"n": len(usable), "delta": observed, "ci95": [low, high]}


def compare(
    base_dir: Path, tag_a: str, tag_b: str, n_boot: int = 2000, claims: bool = True
) -> dict[str, Any]:
    """Informe completo del contraste A→B."""
    run_a = load_run(base_dir, tag_a)
    run_b = load_run(base_dir, tag_b)
    rows_a, rows_b = run_a["responses"], run_b["responses"]
    labels = {**run_a["labels"], **run_b["labels"]}

    shared = [rid for rid in rows_a if rid in rows_b]
    shared.sort()
    # Todo se calcula sobre el conjunto **compartido**: si una corrida tiene
    # registros extra (p. ej. tres filas de desarrollo que se colaron en el
    # histórico), promediarlas dentro de una sola tabla rompería la simetría.
    paired_a = {rid: rows_a[rid] for rid in shared}
    paired_b = {rid: rows_b[rid] for rid in shared}

    correctness_a, correctness_b = [], []
    transitions: Counter[tuple[str, str]] = Counter()
    paired_decided: list[tuple[float, float]] = []
    paired_accuracy: list[tuple[float, float]] = []
    for record_id in shared:
        row_a, row_b = rows_a[record_id], rows_b[record_id]
        if not (is_valid(row_a) and is_valid(row_b)):
            continue
        transitions[(decision_label(row_a), decision_label(row_b))] += 1
        label = labels.get(record_id)
        if label is None:
            continue
        decided_a, decided_b = not is_abstain(row_a), not is_abstain(row_b)
        hit_a = decided_a and row_a.get("decision") == label
        hit_b = decided_b and row_b.get("decision") == label
        correctness_a.append(hit_a)
        correctness_b.append(hit_b)
        if decided_a and decided_b:
            paired_decided.append((1.0 if hit_a else 0.0, 1.0 if hit_b else 0.0))
        paired_accuracy.append((1.0 if hit_a else 0.0, 1.0 if hit_b else 0.0))

    claims_a = claims_table(paired_a) if claims else {}
    claims_b = claims_table(paired_b) if claims else {}
    certainty_a = certainty_table(paired_a, labels)
    certainty_b = certainty_table(paired_b, labels)

    paired_claims = [
        (
            claims_a.get("per_record_supported", {}).get(rid, float("nan")),
            claims_b.get("per_record_supported", {}).get(rid, float("nan")),
        )
        for rid in shared
    ]
    paired_certainty = [
        (
            certainty_a["per_record"].get(rid, float("nan")),
            certainty_b["per_record"].get(rid, float("nan")),
        )
        for rid in shared
    ]

    return {
        "version": "physionet_runs_comparison/1.0",
        "dir": str(base_dir),
        "runs": {
            "a": {"tag": tag_a, "engine": run_a["engine"]},
            "b": {"tag": tag_b, "engine": run_b["engine"]},
        },
        "engine_changed": bool(
            run_a["engine"] and run_b["engine"] and run_a["engine"] != run_b["engine"]
        ),
        "paired": {
            "n_shared": len(shared),
            "only_a": sorted(set(rows_a) - set(rows_b)),
            "only_b": sorted(set(rows_b) - set(rows_a)),
            "n_both_valid_labelled": len(correctness_a),
        },
        "correctness": {
            "a": correctness_table(paired_a, labels),
            "b": correctness_table(paired_b, labels),
        },
        "mcnemar": mcnemar_exact(correctness_a, correctness_b),
        "paired_deltas": {
            "accuracy_decided": paired_bootstrap(paired_decided, n_boot),
            "accuracy_incl_abstain": paired_bootstrap(paired_accuracy, n_boot),
            "claims_supported": paired_bootstrap(paired_claims, n_boot) if claims else {},
            "certainty_index": paired_bootstrap(paired_certainty, n_boot),
        },
        "transitions": {
            f"{before}->{after}": count
            for (before, after), count in sorted(
                transitions.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "claims": {"a": claims_a, "b": claims_b} if claims else {},
        "certainty": {"a": certainty_a, "b": certainty_b},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_report(report: Mapping[str, Any]) -> None:
    """Resumen legible del contraste."""
    runs = report["runs"]
    engine_a = runs["a"]["engine"] or "no registrado"
    engine_b = runs["b"]["engine"] or "no registrado"
    print("=== PhysioNet 2017 — contraste de corridas ===")
    print(f"A: tag={runs['a']['tag'] or 'histórica'} | motor={engine_a}")
    print(f"B: tag={runs['b']['tag'] or 'histórica'} | motor={engine_b}")
    if report["engine_changed"]:
        print("  AVISO: los dos corpus se generaron con motores distintos.")
    paired = report["paired"]
    print(
        f"Pares: {paired['n_shared']} compartidos | ambos válidos y etiquetados: "
        f"{paired['n_both_valid_labelled']} | solo en A: {len(paired['only_a'])} | "
        f"solo en B: {len(paired['only_b'])}"
    )
    print("\n-- Corrección --")
    for side in ("a", "b"):
        table = report["correctness"][side]
        print(
            f"{side.upper()}: acc={table['accuracy']:.3f} "
            f"acc*(abst=error)={table['accuracy_incl_abstain']:.3f} "
            f"bal={table['balanced_accuracy']:.3f} cobertura={table['coverage']:.3f} "
            f"abstención={table['n_abstain']}/{table['n_labelled']} "
            f"inválidas={table['n_invalid']}"
        )
        print(f"   decisiones: {table['decisions']}")
    mcnemar = report["mcnemar"]
    print(
        f"McNemar exacto: A solo acierta {mcnemar['a_only_correct']} | "
        f"B solo acierta {mcnemar['b_only_correct']} | p={mcnemar['p_value']:.4f}"
    )
    print("\n-- Abstención por clase real (A / B) --")
    per_class_a = report["correctness"]["a"]["per_class"]
    per_class_b = report["correctness"]["b"]["per_class"]
    for label in RHYTHM_CLASSES:
        a, b = per_class_a[label], per_class_b[label]
        print(
            f"  {label:<13} A {a['abstentions']}/{a['n']} ({a['abstention_rate']:.3f})"
            f" | B {b['abstentions']}/{b['n']} ({b['abstention_rate']:.3f})"
            f" | recall {a['recall']:.3f} -> {b['recall']:.3f}"
        )
    print("\n-- Transiciones de decisión --")
    for transition, count in report["transitions"].items():
        print(f"  {transition}: {count}")
    deltas = report["paired_deltas"]
    print("\n-- Diferencias pareadas B−A (IC 95 % bootstrap) --")
    for name in ("accuracy_decided", "accuracy_incl_abstain", "claims_supported", "certainty_index"):
        entry = deltas.get(name) or {}
        if not entry.get("n"):
            continue
        low, high = entry["ci95"]
        print(
            f"  {name:<22} Δ={entry['delta']:+.3f} [{low:+.3f}; {high:+.3f}] "
            f"(n={entry['n']})"
        )
    if report["claims"]:
        print("\n-- Claims --")
        for side in ("a", "b"):
            entry = report["claims"][side]
            print(
                f"{side.upper()}: n={entry['n_claims']} respaldados={entry['supported_rate']:.3f} "
                f"contradichos={entry['contradicted_rate']:.3f} "
                f"no respaldados={entry['unsupported_rate']:.3f} "
                f"dirección={entry.get('direction_accuracy', float('nan')):.3f}"
            )
    print("\n-- Certeza lingüística --")
    for side in ("a", "b"):
        entry = report["certainty"][side]
        print(
            f"{side.upper()}: índice={entry['certainty_index_mean']:+.3f} "
            f"hedges={entry['hedges_mean']:.3f} boosters={entry['boosters_mean']:.3f}"
        )
        print(
            "   por clase: "
            + " ".join(f"{k}={v:+.2f}" for k, v in entry["per_class"].items())
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="Directorio P3.")
    parser.add_argument("--tag-a", default="", help="Corrida de referencia (vacío = histórica).")
    parser.add_argument("--tag-b", required=True, help="Corrida nueva (p. ej. srv).")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--no-claims", action="store_true", help="Omite el recálculo de claims.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    base_dir = Path(args.dir)
    report = compare(
        base_dir,
        tag_a=args.tag_a,
        tag_b=args.tag_b,
        n_boot=args.bootstrap,
        claims=not args.no_claims,
    )
    default_name = f"physionet2017_comparison_{args.tag_b}.json"
    output = Path(args.output) if args.output else base_dir / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print_report(report)
    print(f"\nInforme: {output}")


if __name__ == "__main__":
    main()
