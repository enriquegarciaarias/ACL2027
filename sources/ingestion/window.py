"""
Segmentación temporal (H3): ventanas de 60 s con pureza de etiqueta.

Procedimiento:

1. Se detectan los tramos contiguos de etiqueta (``label_runs``).
2. Solo se consideran etiquetas de ``labels.keep`` (1–4).
3. Cada tramo se recorta con una guarda de ``guard_s`` en ambos extremos para
   no tocar transiciones.
4. Se generan ventanas de ``length_s`` con paso ``stride_s`` **dentro** del
   tramo recortado.
5. Se comprueba pureza y cobertura por modalidad.

Las ventanas se guardan como JSONL (una por ventana) y un informe por sujeto
con los descartes y sus motivos. La etiqueta se conserva en el índice para
poder segmentar y evaluar, pero no se copiará a las representaciones (H5).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from sources.common.common import logger, writeLog
from sources.common.paths import (
    atomic_write_json,
    atomic_write_jsonl,
    corpus_dir,
    ensure_dir,
    read_json,
    read_jsonl,
)
from sources.common.provenance import build_fingerprint, git_info, sha256_json, utc_now
from sources.common.validation import ValidationError
from sources.ingestion import loader
from sources.ingestion.schemas import SCHEMA_VERSION

CHEST_CHANNELS = ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp")
WRIST_RATES = (
    ("wrist_acc", "wrist_acc", "wrist_ACC"),
    ("wrist_bvp", "wrist_bvp", "wrist_BVP"),
    ("wrist_eda", "wrist_eda", "wrist_EDA"),
    ("wrist_temp", "wrist_temp", "wrist_TEMP"),
)


def windows_dir() -> Path:
    return corpus_dir() / "windows"


def windows_path(subject: str) -> Path:
    return windows_dir() / f"{subject}.windows.jsonl"


def report_path(subject: str) -> Path:
    return windows_dir() / f"{subject}.windows.report.json"


def global_index_path() -> Path:
    return windows_dir() / "windows_index.jsonl"


def global_report_path() -> Path:
    return windows_dir() / "windows.report.json"


def window_fingerprint(meta: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    """Fingerprint de la segmentación de un sujeto."""
    return build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "canonical_sha256": meta.get("canonical_sha256"),
            "windows": cfg.get("windows"),
            "labels_keep": cfg.get("labels", {}).get("keep"),
            "sampling_rates": cfg.get("sampling_rates"),
            "git": git_info().get("commit"),
        }
    )


def label_runs(label: np.ndarray) -> list[dict]:
    """Devuelve los tramos contiguos de etiqueta como ``{label,start,end}``."""
    label = np.asarray(label)
    if label.size == 0:
        return []
    if label.size == 1:
        return [{"label": int(label[0]), "start": 0, "end": 1}]
    changes = np.flatnonzero(label[1:] != label[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [label.size]))
    return [
        {"label": int(label[start]), "start": int(start), "end": int(end)}
        for start, end in zip(starts, ends)
    ]


def _ranges(ws: int, we: int, cfg: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> dict:
    """Rangos de muestras por modalidad a su frecuencia nativa."""
    fs_label = float(cfg["sampling_rates"]["label"])
    ranges = {"label": [int(ws), int(we)], "chest": [int(ws), int(we)]}

    for key, rate_key, array_key in WRIST_RATES:
        fs = float(cfg["sampling_rates"][rate_key])
        n = int(arrays[array_key].shape[0])
        start = int(round(ws / fs_label * fs))
        end = int(round(we / fs_label * fs))
        start = max(0, min(start, n))
        end = max(start, min(end, n))
        ranges[key] = [start, end]

    return ranges


def _finite_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(values)) / values.size)


def _coverage(arrays: Mapping[str, np.ndarray], ranges: Mapping[str, list[int]]) -> dict:
    """Fracción de muestras finitas por modalidad (mínimo entre canales)."""
    chest_start, chest_end = ranges["chest"]
    chest_fractions = [
        _finite_fraction(arrays[f"chest_{channel}"][chest_start:chest_end])
        for channel in CHEST_CHANNELS
    ]
    coverage = {"chest": min(chest_fractions) if chest_fractions else 0.0}

    for key, _, array_key in WRIST_RATES:
        start, end = ranges[key]
        coverage[key] = _finite_fraction(arrays[array_key][start:end])

    return coverage


def _generate(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    cfg: Mapping[str, Any],
    fingerprint: str,
) -> tuple[list[dict], dict, dict]:
    """Genera las ventanas válidas y contabiliza los descartes."""
    label = np.asarray(arrays["label"])
    fs_label = float(cfg["sampling_rates"]["label"])
    windows_cfg = cfg["windows"]

    length = int(round(float(windows_cfg["length_s"]) * fs_label))
    stride = int(round(float(windows_cfg["stride_s"]) * fs_label))
    guard = int(round(float(windows_cfg["guard_s"]) * fs_label))
    purity_min = float(windows_cfg["purity_min"])
    min_coverage = float(windows_cfg["min_valid_fraction"])
    drop_low_coverage = bool(windows_cfg.get("drop_low_coverage", True))

    keep = set(int(v) for v in cfg["labels"]["keep"])
    names = cfg["labels"].get("names", {})
    dataset = str(cfg.get("dataset", "WESAD"))

    rows: list[dict] = []
    drops: Counter = Counter()
    by_label: Counter = Counter()
    index = 0

    for run in label_runs(label):
        label_id = run["label"]
        if label_id not in keep:
            drops["label_not_kept"] += 1
            continue

        start = run["start"] + guard
        end = run["end"] - guard
        if end - start < length:
            drops["short_run"] += 1
            continue

        for ws in range(start, end - length + 1, stride):
            we = ws + length
            segment = label[ws:we]
            purity = float(np.count_nonzero(segment == label_id) / segment.size)
            if purity < purity_min:
                drops["low_purity"] += 1
                continue

            ranges = _ranges(ws, we, cfg, arrays)
            coverage = _coverage(arrays, ranges)
            flags = [
                f"low_coverage:{modality}"
                for modality, fraction in coverage.items()
                if fraction < min_coverage
            ]
            if flags and drop_low_coverage:
                drops["low_coverage"] += 1
                continue

            index += 1
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "window_id": f"{dataset}:{meta['subject']}:w{index:05d}",
                    "subject": meta["subject"],
                    "label": {
                        "id": label_id,
                        "name": str(names.get(str(label_id), "unknown")),
                    },
                    "t_start_s": ws / fs_label,
                    "t_end_s": we / fs_label,
                    "duration_s": length / fs_label,
                    "fs": {
                        "label": fs_label,
                        "chest": float(cfg["sampling_rates"]["chest"]),
                        "wrist_acc": float(cfg["sampling_rates"]["wrist_acc"]),
                        "wrist_bvp": float(cfg["sampling_rates"]["wrist_bvp"]),
                        "wrist_eda": float(cfg["sampling_rates"]["wrist_eda"]),
                        "wrist_temp": float(cfg["sampling_rates"]["wrist_temp"]),
                    },
                    "ranges": ranges,
                    "purity": purity,
                    "guard_applied_s": float(windows_cfg["guard_s"]),
                    "coverage": coverage,
                    "flags": flags,
                    "source": {
                        "file": meta.get("source_file"),
                        "sha256": meta.get("source_sha256"),
                    },
                    "config_fingerprint": fingerprint,
                }
            )
            by_label[str(label_id)] += 1

    return rows, dict(drops), dict(by_label)


def run_window(cfg: Mapping[str, Any], subject: str, *, force: bool = False) -> dict:
    """Genera el índice de ventanas de un sujeto."""
    _, meta_path = loader.canonical_paths(subject)
    meta = read_json(meta_path)
    if meta is None:
        raise ValidationError(
            f"no existe el canónico de {subject}; ejecute primero '--step ingest'",
            where="window",
        )

    fingerprint = window_fingerprint(meta, cfg)
    target = windows_path(subject)
    report_target = report_path(subject)
    previous = read_json(report_target)
    if (
        not force
        and previous is not None
        and previous.get("fingerprint") == fingerprint
        and target.is_file()
    ):
        writeLog(
            "info",
            logger,
            f"[window] {subject}: {previous.get('n_windows', 0)} ventanas reutilizadas "
            f"(fingerprint {fingerprint[:12]}…)",
        )
        return {
            "subject": subject,
            "status": "reused",
            "fingerprint": fingerprint,
            "n_windows": int(previous.get("n_windows", 0)),
        }

    arrays = loader.load_canonical(subject)
    rows, drops, by_label = _generate(arrays, meta, cfg, fingerprint)

    ensure_dir(windows_dir())
    atomic_write_jsonl(target, rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "config_fingerprint": sha256_json(
            {"windows": cfg.get("windows"), "labels": cfg.get("labels")}
        ),
        "duration_s": float(meta.get("duration_s", 0.0)),
        "n_windows": len(rows),
        "n_by_label": by_label,
        "drops": drops,
        "windows_file": str(target),
    }
    atomic_write_json(report_target, report)

    writeLog(
        "info",
        logger,
        f"[window] {subject}: {len(rows)} ventanas {by_label}; descartes {drops or '{}'}",
    )
    return {
        "subject": subject,
        "status": "ok",
        "fingerprint": fingerprint,
        "n_windows": len(rows),
        "n_by_label": by_label,
        "drops": drops,
    }


def build_global_index(cfg: Mapping[str, Any], subjects: Iterable[str]) -> dict:
    """Concatena los índices por sujeto en ``windows_index.jsonl``."""
    subjects = list(subjects)
    all_rows: list[dict] = []
    per_subject: dict[str, int] = {}
    by_label: Counter = Counter()

    for subject in subjects:
        if not windows_path(subject).is_file():
            raise ValidationError(
                f"falta el índice de ventanas de {subject}; ejecute 'window' antes de construir el índice global",
                where="window",
            )
        rows = read_jsonl(windows_path(subject))
        per_subject[subject] = len(rows)
        all_rows.extend(rows)
        for row in rows:
            by_label[str(row["label"]["id"])] += 1

    ensure_dir(windows_dir())
    atomic_write_jsonl(global_index_path(), all_rows)

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "n_subjects": len(subjects),
        "n_windows": len(all_rows),
        "n_by_label": dict(by_label),
        "per_subject": per_subject,
        "index_file": str(global_index_path()),
        "config_fingerprint": sha256_json({"windows": cfg.get("windows"), "labels": cfg.get("labels")}),
    }
    atomic_write_json(global_report_path(), report)
    writeLog(
        "info",
        logger,
        f"[window] índice global: {len(all_rows)} ventanas de {len(subjects)} sujetos en {global_index_path()}",
    )
    return report
