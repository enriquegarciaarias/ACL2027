"""
Extracción de características por ventana (H4).

Salidas:

- ``features/{SX}.features.jsonl`` — una línea por ventana;
- ``features/{SX}.features.report.json`` — resumen del sujeto;
- ``features/features_index.jsonl`` — índice global;
- ``features/features.report.json`` — resumen global;
- ``features/feature_spec.json`` — especificación de modalidades y SQI.

Invariante anti-leakage: el registro de features **no contiene la etiqueta**.
La unión con ``windows/{SX}.windows.jsonl`` se hace por ``window_id`` en la
fase de evaluación.
"""
from __future__ import annotations

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
from sources.ingestion import degrade, loader, window
from sources.ingestion.features import registry
from sources.ingestion.features.common import FEATURE_VERSION
from sources.ingestion.schemas import FeatureRecord, SCHEMA_VERSION


def features_dir() -> Path:
    return corpus_dir() / "features"


def features_path(subject: str) -> Path:
    return features_dir() / f"{subject}.features.jsonl"


def report_path(subject: str) -> Path:
    return features_dir() / f"{subject}.features.report.json"


def global_index_path() -> Path:
    return features_dir() / "features_index.jsonl"


def global_report_path() -> Path:
    return features_dir() / "features.report.json"


def spec_path() -> Path:
    return features_dir() / "feature_spec.json"


def degraded_features_dir() -> Path:
    return corpus_dir() / "degraded_features"


def degraded_features_path(condition_id: str, subject: str) -> Path:
    return degraded_features_dir() / condition_id / f"{subject}.features.jsonl"


def degraded_features_report_path(condition_id: str, subject: str) -> Path:
    return degraded_features_dir() / condition_id / f"{subject}.features.report.json"


def feature_fingerprint(
    meta: Mapping[str, Any],
    windows_report: Mapping[str, Any],
    cfg: Mapping[str, Any],
    condition_id: str | None = None,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "canonical_sha256": meta.get("canonical_sha256"),
        "windows_fingerprint": windows_report.get("fingerprint"),
        "features": cfg.get("features"),
        "sampling_rates": cfg.get("sampling_rates"),
        "git": git_info().get("commit"),
    }
    if condition_id:
        payload["condition_id"] = condition_id
    return build_fingerprint(payload)
    return build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "canonical_sha256": meta.get("canonical_sha256"),
            "windows_fingerprint": windows_report.get("fingerprint"),
            "features": cfg.get("features"),
            "sampling_rates": cfg.get("sampling_rates"),
            "git": git_info().get("commit"),
        }
    )


def _summary(rows: list[dict], cfg: Mapping[str, Any]) -> dict:
    sqi_values = [
        float(row["quality"]["sqi_overall"])
        for row in rows
        if row.get("quality", {}).get("sqi_overall") is not None
    ]
    labels: dict[str, int] = {}
    for row in rows:
        category = row.get("quality", {}).get("category", "unknown")
        labels[category] = labels.get(category, 0) + 1

    return {
        "n_windows": len(rows),
        "n_feature_values": sum(
            len(values) for row in rows for values in row.get("features", {}).values()
        ),
        "n_features_per_window": (
            sum(len(values) for values in rows[0].get("features", {}).values())
            if rows else 0
        ),
        "sqi_mean": float(np.mean(sqi_values)) if sqi_values else None,
        "sqi_min": float(np.min(sqi_values)) if sqi_values else None,
        "sqi_max": float(np.max(sqi_values)) if sqi_values else None,
        "sqi_categories": labels,
        "modalities": sorted(rows[0]["features"]) if rows else [],
    }


def _write_spec(cfg: Mapping[str, Any]) -> None:
    spec = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "generated_at": utc_now(),
        "modalities": registry.modality_specs(cfg),
        "sqi": cfg.get("features", {}).get("sqi", {}),
        "notes": {
            "lf_hf": "no se calcula: una ventana de 60 s no permite estimar LF/HF de forma fiable",
            "labels": "este artefacto no contiene etiquetas (anti-leakage)",
        },
    }
    ensure_dir(features_dir())
    atomic_write_json(spec_path(), spec)


def run_features(
    cfg: Mapping[str, Any],
    subject: str,
    *,
    force: bool = False,
    condition_id: str | None = None,
) -> dict:
    """Extrae características de todas las ventanas de un sujeto.

    Cuando ``condition_id`` se proporciona, carga los arrays degradados de esa
    condición (generados por el paso ``degrade``) en lugar del canónico, y
    guarda las features en ``degraded_features/{condition_id}/{subject}.features.jsonl``.
    Las ventanas (rangos de tiempo, etiquetas) son las mismas que las del canónico.
    """
    _, meta_path = loader.canonical_paths(subject)
    meta = read_json(meta_path)
    if meta is None:
        raise ValidationError(
            f"no existe el canónico de {subject}; ejecute primero '--step ingest'",
            where="features",
        )
    windows_report = read_json(window.report_path(subject))
    if windows_report is None:
        raise ValidationError(
            f"no existen ventanas de {subject}; ejecute primero '--step window'",
            where="features",
        )

    # Cargar arrays: degradados si condition_id, canónico si no
    if condition_id:
        arrays = degrade.load_degraded(subject, condition_id)
        condition_label = f" (condición {condition_id})"
    else:
        arrays = loader.load_canonical(subject)
        condition_label = ""

    fingerprint = feature_fingerprint(meta, windows_report, cfg, condition_id)

    if condition_id:
        target = degraded_features_path(condition_id, subject)
        report_target = degraded_features_report_path(condition_id, subject)
    else:
        target = features_path(subject)
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
            f"[features] {subject}{condition_label}: {previous.get('n_windows', 0)} ventanas reutilizadas "
            f"(fingerprint {fingerprint[:12]}…)",
        )
        return {
            "subject": subject,
            "status": "reused",
            "fingerprint": fingerprint,
            "condition_id": condition_id,
            "n_windows": int(previous.get("n_windows", 0)),
            "n_feature_values": int(previous.get("n_feature_values", 0)),
            "sqi_mean": previous.get("sqi_mean"),
        }

    windows = read_jsonl(window.windows_path(subject))
    rows: list[dict] = []

    for item in windows:
        features, quality = registry.extract_window(arrays, item, cfg)
        record = FeatureRecord(
            window_id=item["window_id"],
            subject=subject,
            features=features,
            quality=quality,
            config_fingerprint=fingerprint,
            feature_version=FEATURE_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        rows.append(record.to_dict())

    ensure_dir(target.parent)
    atomic_write_jsonl(target, rows)
    summary = _summary(rows, cfg)
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "subject": subject,
        "condition_id": condition_id,
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "config_fingerprint": sha256_json({"features": cfg.get("features")}),
        "duration_s": float(meta.get("duration_s", 0.0)),
        "features_file": str(target),
        **summary,
    }
    atomic_write_json(report_target, report)
    _write_spec(cfg)

    writeLog(
        "info",
        logger,
        f"[features] {subject}{condition_label}: {summary['n_windows']} ventanas, "
        f"{summary['n_feature_values']} valores de feature, SQI medio={summary['sqi_mean']}",
    )
    return {
        "subject": subject,
        "status": "ok",
        "fingerprint": fingerprint,
        "condition_id": condition_id,
        "n_windows": summary["n_windows"],
        "n_feature_values": summary["n_feature_values"],
        "sqi_mean": summary["sqi_mean"],
    }


def build_global_index(cfg: Mapping[str, Any], subjects: Iterable[str]) -> dict:
    """Concatena los features por sujeto en ``features_index.jsonl``."""
    subjects = list(subjects)
    all_rows: list[dict] = []
    per_subject: dict[str, int] = {}

    for subject in subjects:
        if not features_path(subject).is_file():
            raise ValidationError(
                f"faltan features de {subject}; ejecute 'features' antes de construir el índice global",
                where="features",
            )
        rows = read_jsonl(features_path(subject))
        per_subject[subject] = len(rows)
        all_rows.extend(rows)

    ensure_dir(features_dir())
    atomic_write_jsonl(global_index_path(), all_rows)
    summary = _summary(all_rows, cfg)
    report = {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at": utc_now(),
        "n_subjects": len(subjects),
        "per_subject": per_subject,
        "index_file": str(global_index_path()),
        "config_fingerprint": sha256_json({"features": cfg.get("features")}),
        **summary,
    }
    atomic_write_json(global_report_path(), report)
    _write_spec(cfg)

    writeLog(
        "info",
        logger,
        f"[features] índice global: {summary['n_windows']} ventanas, "
        f"{summary['n_feature_values']} valores de feature de {len(subjects)} sujetos",
    )
    return report
