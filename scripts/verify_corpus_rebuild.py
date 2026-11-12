#!/usr/bin/env python
"""Verifica que el corpus actual reproduce las entradas que vio el LLM.

Motivación
==========
El ``git commit`` forma parte de todos los fingerprints de ``INGEST`` (§43 de
``AGENTS.md``). Como consecuencia, cualquier commit invalida el corpus y el
pipeline lo reconstruye entero. Este script comprueba, **sin llamar al LLM**,
si esa reconstrucción conservó el contenido: reconstruye el prompt de cada
respuesta guardada a partir de las features del corpus actual y lo compara con
el prompt almacenado.

Qué se comprueba por registro
-----------------------------
1. ``duration_s``: la ventana actual conserva su duración.
2. ``prompt``: el prompt reconstruido es **idéntico byte a byte** al guardado.
   Es la comprobación fuerte: cubre features, referencia intra-sujeto,
   redondeo, few-shot y plantilla.
3. ``features``: las features parseadas de ambos prompts coinciden. Sirve de
   diagnóstico cuando (2) falla, para distinguir *deriva de datos* de *cambio
   de plantilla*.

Uso
===
    python scripts/verify_corpus_rebuild.py                      # todo
    python scripts/verify_corpus_rebuild.py --max-records 50     # humo
    python scripts/verify_corpus_rebuild.py --representation text
    python scripts/verify_corpus_rebuild.py --ingest /ruta/INGEST \
        --responses /ruta/PHASE2/degradation

Códigos de salida
-----------------
0 · todo coincide · 1 · hay divergencias · 2 · no hay artefactos que verificar
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.provenance import sha256_file, utc_now
from sources.ingestion.represent import _opaque_window_id
from sources.phase2.prompt_builder import build_numeric_prompt, build_text_prompt
from sources.phase2.response_schema import DECISION_MODE_AFFECTIVE
from sources.phase3.claims import parse_prompt_features
from sources.phase3.degradation import (
    build_payload,
    features_path,
    load_features,
    render_text_summary,
    subject_reference,
    z_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INGEST = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
DEFAULT_PHASE2 = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2"
DEFAULT_OUTPUT = DEFAULT_PHASE2 / "verify_rebuild_report.json"

#: ``num_ctx`` holgado: aquí solo evita el guard de longitud, no afecta al texto.
NUM_CTX = 1 << 20


def response_files(root: Path) -> list[Path]:
    """Respuestas del barrido, con o sin subdirectorio por modelo."""
    if not root.is_dir():
        return []
    files = list(root.glob("*.responses.jsonl"))
    files += list(root.glob("*/*.responses.jsonl"))
    return sorted(files)


def representation_of(path: Path) -> str:
    """``text`` si la ruta pertenece al barrido textual; ``numeric`` si no."""
    for part in path.parts:
        if part.startswith("degradation_text"):
            return "text"
    return "numeric"


def load_windows(ingest_dir: Path, subject: str) -> dict[str, dict[str, Any]]:
    path = ingest_dir / "windows" / f"{subject}.windows.jsonl"
    if not path.is_file():
        return {}
    windows: dict[str, dict[str, Any]] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        windows[record["window_id"]] = record
    return windows


def reference(ingest_dir: Path, subject: str) -> tuple[dict, dict, dict]:
    """Referencia intra-sujeto (centros y escalas) igual que el barrido."""
    clean_rows = load_features(ingest_dir, "C0_clean", subject)
    feature_names = sorted({name for values in clean_rows.values() for name in values})
    centers, scales = subject_reference(clean_rows, feature_names)
    return centers, scales, clean_rows


def rebuild_prompt(
    representation: str,
    window_id: str,
    values: dict[str, float | None],
    centers: dict[str, float],
    scales: dict[str, float],
    duration_s: float,
) -> str:
    """Prompt que produciría hoy el pipeline con estas features."""
    opaque = _opaque_window_id(window_id)
    if representation == "text":
        scores = z_scores(values, centers, scales)
        return build_text_prompt(
            text_representation=render_text_summary(scores),
            window_id_opaque=opaque,
            duration_s=duration_s,
            num_ctx=NUM_CTX,
            use_few_shot=True,
            standardized=True,
            decision_mode=DECISION_MODE_AFFECTIVE,
        )
    payload = build_payload(window_id, values, centers, scales, duration_s)
    return build_numeric_prompt(
        payload=payload,
        window_id_opaque=opaque,
        duration_s=duration_s,
        num_ctx=NUM_CTX,
        use_few_shot=True,
        include_additional=True,
        standardized=True,
        decision_mode=DECISION_MODE_AFFECTIVE,
    )


def features_match(stored: str, rebuilt: str) -> bool:
    """Compara las features visibles en dos prompts (nombre y valor)."""
    left = parse_prompt_features(stored)
    right = parse_prompt_features(rebuilt)
    if set(left) != set(right):
        return False
    return all(left[name].value == right[name].value for name in left)


def _diff_hint(stored: str, rebuilt: str) -> list[str]:
    """Primeras líneas distintas entre dos prompts (para el informe)."""
    left = stored.splitlines()
    right = rebuilt.splitlines()
    hints: list[str] = []
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else "<falta línea>"
        b = right[index] if index < len(right) else "<falta línea>"
        if a != b:
            hints.append(f"L{index + 1}: guardado={a!r} · reconstruido={b!r}")
        if len(hints) >= 3:
            break
    return hints


def verify_source(
    path: Path,
    ingest_dir: Path,
    *,
    representation: str | None = None,
    max_records: int | None = None,
    samples: int = 3,
) -> dict[str, Any]:
    """Verifica un fichero de respuestas contra el corpus actual."""
    representation = representation or representation_of(path)
    stats: dict[str, Any] = {
        "file": str(path),
        "representation": representation,
        "records": 0,
        "duration_equal": 0,
        "prompt_equal": 0,
        "prompt_diff": 0,
        "features_equal": 0,
        "features_diff": 0,
        "hash_equal": 0,
        "hash_diff": 0,
        "missing_window": 0,
        "missing_features": 0,
        "samples": [],
    }

    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if max_records is not None:
        records = records[:max_records]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["subject"], record["condition_id"])].append(record)

    cache: dict[str, tuple[dict, dict]] = {}
    windows_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for (subject, condition), group in sorted(grouped.items()):
        if subject not in cache:
            try:
                centers, scales, _ = reference(ingest_dir, subject)
                cache[subject] = (centers, scales)
                windows_cache[subject] = load_windows(ingest_dir, subject)
            except FileNotFoundError:
                stats["missing_features"] += len(group)
                continue
        centers, scales = cache[subject]
        windows = windows_cache[subject]

        try:
            condition_rows = load_features(ingest_dir, condition, subject)
        except FileNotFoundError:
            stats["missing_features"] += len(group)
            continue

        # Atajo O(1): las respuestas nuevas guardan el hash del fichero de
        # features; si coincide, no hace falta recomponer el prompt.
        recorded_sha = next(
            (item["features_sha256"] for item in group if item.get("features_sha256")),
            None,
        )
        if recorded_sha is not None:
            current_sha = sha256_file(features_path(ingest_dir, condition, subject))
            same_hash = recorded_sha == current_sha
            stats["hash_equal" if same_hash else "hash_diff"] += len(group)
            for record in group:
                stats["records"] += 1
                if same_hash:
                    stats["prompt_equal"] += 1
                    stats["features_equal"] += 1
                else:
                    stats["prompt_diff"] += 1
                    stats["features_diff"] += 1
                    if len(stats["samples"]) < samples:
                        stats["samples"].append(
                            {
                                "subject": subject,
                                "condition_id": condition,
                                "window_id": record["window_id"],
                                "features_equal": False,
                                "diff": ["hash del fichero de features distinto"],
                            }
                        )
            continue

        for record in group:
            stats["records"] += 1
            window_id = record["window_id"]
            window = windows.get(window_id)
            values = condition_rows.get(window_id)
            if window is None or values is None:
                stats["missing_window"] += 1
                continue

            duration_s = float(record.get("duration_s") or window["duration_s"])
            if float(window["duration_s"]) == duration_s:
                stats["duration_equal"] += 1

            rebuilt = rebuild_prompt(
                representation, window_id, values, centers, scales, duration_s
            )
            stored = record.get("prompt") or ""
            same_prompt = stored == rebuilt
            same_features = features_match(stored, rebuilt)
            stats["prompt_equal" if same_prompt else "prompt_diff"] += 1
            stats["features_equal" if same_features else "features_diff"] += 1
            if not same_prompt and len(stats["samples"]) < samples:
                stats["samples"].append(
                    {
                        "subject": subject,
                        "condition_id": condition,
                        "window_id": window_id,
                        "features_equal": same_features,
                        "diff": _diff_hint(stored, rebuilt),
                    }
                )

    stats["identical"] = stats["prompt_diff"] == 0 and stats["missing_window"] == 0
    return stats


def _fmt(stats: dict[str, Any]) -> str:
    total = stats["records"] or 1
    pct = 100.0 * stats["prompt_equal"] / total
    flag = "OK " if stats["identical"] else "DRIFT"
    mode = "hash" if stats["hash_equal"] or stats["hash_diff"] else "prompt"
    return (
        f"[{flag}] {stats['file']}  ({mode})\n"
        f"       {stats['records']} registros · idéntico "
        f"{stats['prompt_equal']} ({pct:.1f} %) · distinto {stats['prompt_diff']} · "
        f"features iguales {stats['features_equal']} · ventana ausente "
        f"{stats['missing_window']} · features ausentes {stats['missing_features']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest", default=str(DEFAULT_INGEST))
    parser.add_argument(
        "--responses",
        action="append",
        default=None,
        help="Directorio de respuestas (repetible). Por defecto el barrido numérico y textual.",
    )
    parser.add_argument(
        "--representation",
        choices=("numeric", "text"),
        default=None,
        help="Forzar la representación en vez de inferirla de la ruta.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    ingest_dir = Path(args.ingest).expanduser().resolve()
    roots = (
        [Path(item).expanduser().resolve() for item in args.responses]
        if args.responses
        else [DEFAULT_PHASE2 / "degradation", DEFAULT_PHASE2 / "degradation_text"]
    )

    files: list[Path] = []
    for root in roots:
        files.extend(response_files(root))
    if not files:
        print(f"[ERROR] No hay ficheros *.responses.jsonl en: "
              + ", ".join(str(root) for root in roots))
        print("[ERROR] Ejecuta este script donde esté el corpus (p. ej. el servidor).")
        return 2

    sources: list[dict[str, Any]] = []
    for path in files:
        sources.append(
            verify_source(
                path,
                ingest_dir,
                representation=args.representation,
                max_records=args.max_records,
            )
        )

    totals = defaultdict(int)
    for stats in sources:
        for key in (
            "records",
            "duration_equal",
            "prompt_equal",
            "prompt_diff",
            "features_equal",
            "features_diff",
            "hash_equal",
            "hash_diff",
            "missing_window",
            "missing_features",
        ):
            totals[key] += stats[key]
    verdict = (
        "identical"
        if totals["prompt_diff"] == 0 and totals["missing_window"] == 0
        else "drift"
    )
    report = {
        "generated_at": utc_now(),
        "ingest_dir": str(ingest_dir),
        "sources": sources,
        "totals": dict(totals),
        "verdict": verdict,
    }

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for stats in sources:
        print(_fmt(stats))
        for sample in stats["samples"]:
            print(
                f"       muestra {sample['subject']} {sample['condition_id']} "
                f"{sample['window_id']} (features iguales: {sample['features_equal']})"
            )
            for line in sample["diff"]:
                print(f"         {line}")
    print(
        f"\nTOTAL: {totals['records']} registros · prompt idéntico "
        f"{totals['prompt_equal']} · distinto {totals['prompt_diff']} · "
        f"veredicto: {verdict.upper()}"
    )
    print(f"Informe: {output}")
    return 0 if verdict == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
