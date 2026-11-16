#!/usr/bin/env python
"""Analiza la jerarquía binaria ``stress`` vs ``non-stress`` (Fase 1).

Lee las respuestas ya generadas (sin llamar al LLM) y calcula, para cada
condición y cada variante declarada:

- ``schmidt`` (primario): ``stress`` vs ``baseline`` + ``amusement``.
- ``baseline``: ``stress`` vs ``baseline``.
- ``no_amusement``: ``stress`` vs ``baseline`` + ``meditation``.
- ``all``: ``stress`` vs todo ``non-stress`` (corpus histórico).

Reporta ``n``, prevalencia de ``stress``, tasa base de "siempre non-stress",
accuracy, balanced accuracy, precisión/recall/F1, AUROC, ECE, Brier y los IC
95 % por **bootstrap agrupado por sujeto** (no por ventana).

Uso:
    python scripts/analyze_binary_hierarchy.py --model qwen3:14b
    python scripts/analyze_binary_hierarchy.py --dir results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp/qwen3_14b
    python scripts/analyze_binary_hierarchy.py --from-affective  # corpus de 4 clases
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.phase2.binary_variants import (  # noqa: E402
    DEFAULT_NONSTRESS_VARIANT,
    NONSTRESS_VARIANTS,
    VARIANT_DESCRIPTIONS,
)
from sources.phase3.binary_hierarchy import (  # noqa: E402
    condition_hierarchy_report,
    provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE2_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2"
LABELS_PATH = (
    REPO_ROOT / "results" / "output" / "WESAD" / "INGEST" / "groundtruth" / "labels.jsonl"
)
DEFAULT_CONDITIONS = (
    "C0_clean",
    "C1_partial_resp",
    "C2_partial_eda",
    "C4_total_eda",
    "C5_total_both",
)
DEFAULT_OUTPUT = PHASE2_DIR / "evaluation_binary_hierarchy.json"


def model_slug(model: str) -> str:
    """Slug del modelo para localizar su directorio de artefactos."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(model or "")).strip("_")
    return slug or "model"


def load_labels(path: Path = LABELS_PATH) -> dict[str, str]:
    """``window_id -> etiqueta real`` desde el ground truth."""
    labels: dict[str, str] = {}
    if not path.is_file():
        return labels
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        label = record.get("label")
        if isinstance(label, dict):
            label = label.get("name")
        labels[record["window_id"]] = label
    return labels


def resolve_directory(directory: str, model: str) -> Path:
    """Localiza el directorio de respuestas binarias.

    Orden de preferencia: primario sin temperatura → primario con temperatura →
    corpus histórico ``degradation_binary`` (variante ``all`` con temperatura).
    """
    if directory:
        return Path(directory).expanduser()
    candidates: list[Path] = []
    names = (
        "degradation_binary_schmidt_notemp",
        "degradation_binary_schmidt",
        "degradation_binary",
    )
    if model:
        slug = model_slug(model)
        candidates.extend(PHASE2_DIR / name / slug for name in names)
    else:
        for name in names:
            root = PHASE2_DIR / name
            if not root.is_dir():
                continue
            candidates.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    for candidate in candidates:
        if any(candidate.glob("*.responses.jsonl")):
            return candidate
    raise SystemExit(
        "No se encontraron respuestas binarias. Indica --dir o --model.\n"
        f"Rutas probadas: {', '.join(str(c) for c in candidates) or '(ninguna)'}"
    )


def load_records(
    directory: Path,
    conditions: set[str],
    split: str,
) -> list[dict[str, Any]]:
    """Carga las respuestas JSONL, tolerando líneas incompletas."""
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.responses.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("condition_id") not in conditions:
                continue
            if split != "all" and record.get("split") != split:
                continue
            records.append(record)
    return records


def _fmt(value: Any, digits: int = 3) -> str:
    """Formatea un número; ``nan``/``None`` → ``—``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number):
        return "—"
    return f"{number:.{digits}f}"


def _ci(block: dict[str, Any], metric: str) -> str:
    """Formatea un IC 95 % ``[lo; hi]``."""
    entry = (block or {}).get(metric) or {}
    low, high = entry.get("lo"), entry.get("hi")
    if low is None or high is None:
        return "—"
    return f"[{_fmt(low)}; {_fmt(high)}]"


def print_table(condition: str, block: dict[str, Any]) -> None:
    """Imprime una tabla de variantes para una condición."""
    print(f"\n== {condition} ==")
    header = (
        f"{'variante':<13} {'n':>5} {'prev':>6} {'base':>6} {'acc':>6} "
        f"{'bal':>6} {'prec':>6} {'rec':>6} {'F1':>6} {'AUROC':>6} "
        f"{'ECE':>6} {'Brier':>6}  {'acc 95% CI':<17} {'AUROC 95% CI':<17}"
    )
    print(header)
    print("-" * len(header))
    for variant in NONSTRESS_VARIANTS:
        entry = block.get(variant)
        if not entry:
            continue
        metrics = entry["metrics"]
        boot = entry.get("bootstrap") or {}
        print(
            f"{variant:<13} {metrics.get('n', 0):>5} "
            f"{_fmt(metrics.get('base_rate')):>6} "
            f"{_fmt(1.0 - (metrics.get('base_rate') or 0.0)):>6} "
            f"{_fmt(metrics.get('accuracy')):>6} "
            f"{_fmt(metrics.get('balanced_accuracy')):>6} "
            f"{_fmt(metrics.get('precision_stress')):>6} "
            f"{_fmt(metrics.get('recall_stress')):>6} "
            f"{_fmt(metrics.get('f1_stress')):>6} "
            f"{_fmt(metrics.get('auroc_stress')):>6} "
            f"{_fmt(metrics.get('ece')):>6} "
            f"{_fmt(metrics.get('brier')):>6}  "
            f"{_ci(boot, 'accuracy'):<17} {_ci(boot, 'auroc_stress'):<17}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="", help="Directorio de *.responses.jsonl.")
    parser.add_argument("--model", default="", help="Modelo (para resolver el directorio).")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--variants",
        default=",".join(NONSTRESS_VARIANTS),
        help="Variantes separadas por comas (default: todas).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Remuestras por IC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        default="evaluation",
        choices=["evaluation", "development", "all"],
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--from-affective",
        action="store_true",
        help="Reinterpreta decisiones de 4 clases (baseline/amusement/meditation "
        "→ non-stress) para analizar el corpus afectivo.",
    )
    args = parser.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in NONSTRESS_VARIANTS]
    if unknown:
        raise SystemExit(f"Variantes no válidas: {unknown}")

    directory = resolve_directory(args.dir, args.model)
    records = load_records(directory, set(conditions), args.split)
    if not records:
        raise SystemExit(f"No hay registros en {directory} para las condiciones pedidas.")
    if args.from_affective:
        mapping = {
            "baseline": "non-stress",
            "amusement": "non-stress",
            "meditation": "non-stress",
        }
        records = [
            {**record, "decision": mapping.get(record.get("decision"), record.get("decision"))}
            for record in records
        ]

    labels = load_labels()
    label_by_opaque: dict[str, str] = {}
    for record in records:
        opaque = record.get("window_id_opaque") or record.get("window_id")
        label_by_opaque[opaque] = labels.get(record.get("window_id"), "UNKNOWN")

    info = provenance(records)
    print(f"Directorio: {directory}")
    print(
        f"Registros: {len(records)} | modelo={info.get('model')} | "
        f"representación={info.get('representation')} | "
        f"tarea={info.get('decision_mode')} | "
        f"variante_prompt={info.get('nonstress_variant')} | "
        f"temperatura={info.get('include_temperature')}"
    )
    if info.get("decision_mode") not in (None, "stress_binary"):
        print("AVISO: el corpus no es binario; usa --from-affective si procede.")
    if info.get("include_temperature") is True:
        print("AVISO: el corpus incluye temperatura; el primario debería ser no-temp.")
    if info.get("nonstress_variant") is None:
        print("AVISO: el corpus no registra la variante del prompt (corpus antiguo).")
    elif info.get("nonstress_variant") != DEFAULT_NONSTRESS_VARIANT:
        print(
            f"AVISO: el prompt usó la variante {info.get('nonstress_variant')!r}; "
            "el filtrado por clase real es válido, pero el prompt no es el primario."
        )

    report = condition_hierarchy_report(
        records,
        label_by_opaque,
        conditions=conditions,
        variants=variants,
        n_boot=args.bootstrap,
        seed=args.seed,
    )
    report["provenance"] = info
    report["directory"] = str(directory)

    print("\nVariantes:")
    for variant in variants:
        print(f"  {variant:<13} {VARIANT_DESCRIPTIONS[variant]}")
    for condition in conditions:
        print_table(condition, report.get(condition, {}))
    print_table("pooled (descriptivo)", report.get("pooled", {}))

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nInforme: {output}")


if __name__ == "__main__":
    main()
