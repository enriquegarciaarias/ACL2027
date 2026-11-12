"""
Empaquetado final del corpus (H6).

Produce:

- ``corpus_index.jsonl`` — índice por ventana **sin etiqueta** (metadatos,
  calidad, representaciones disponibles y fingerprints);
- ``groundtruth/labels.jsonl`` — etiquetas separadas (anti-leakage);
- ``corpus_manifest.json`` — versión del corpus, versiones de artefactos,
  recuentos, integridad y lista de artefactos con SHA-256;
- ``corpus_checksums.sha256`` — checksums de todos los artefactos;
- ``corpus_errors.json`` — solo si la integridad falla.

La integridad se comprueba antes de publicar: presencia de artefactos, hash
del canónico, coherencia ventana↔features↔representación, ámbito de etiquetas,
anti-leakage y consistencia de los índices globales.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from sources.common.common import logger, writeLog
from sources.common.paths import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    corpus_dir,
    ensure_dir,
    read_json,
    read_jsonl,
)
from sources.common.provenance import (
    build_fingerprint,
    code_fingerprint,
    git_info,
    library_versions,
    sha256_file,
    sha256_json,
    utc_now,
)
from sources.common.validation import ValidationError
from sources.ingestion import discover, features, loader, qc, represent, window
from sources.ingestion.config import representations_config
from sources.ingestion.features.common import FEATURE_VERSION
from sources.ingestion.represent import REPRESENTATION_VERSION
from sources.ingestion.schemas import SCHEMA_VERSION

#: Código que produce el empaquetado (índice, manifiesto y checksums).
_CODE_SCOPE: tuple[str, ...] = ("sources/ingestion/package.py",)

DATA_KINDS = {
    "canonical",
    "windows",
    "features",
    "representation_numeric",
    "representation_text",
}

CHECK_NAMES = (
    "artifact_presence",
    "canonical_hash",
    "source_hash",
    "window_alignment",
    "label_scope",
    "anti_leakage",
    "feature_schema",
    "report_counts",
    "global_index",
)


def manifest_path() -> Path:
    return corpus_dir() / "corpus_manifest.json"


def index_path() -> Path:
    return corpus_dir() / "corpus_index.jsonl"


def labels_path() -> Path:
    return corpus_dir() / "groundtruth" / "labels.jsonl"


def checksums_path() -> Path:
    return corpus_dir() / "corpus_checksums.sha256"


def errors_path() -> Path:
    return corpus_dir() / "corpus_errors.json"


def _rel(path: Path) -> str:
    return str(Path(path).relative_to(corpus_dir()))


def _add_error(errors: list[dict], check: str, message: str) -> None:
    errors.append({"check": check, "message": message})


def _contains_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(value, key) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(value, key) for value in obj)
    return False


def _artifact(path: Path, kind: str, rows: int | None = None) -> dict:
    path = Path(path)
    return {
        "path": _rel(path),
        "kind": kind,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": rows,
    }


def _require(path: Path, errors: list[dict], label: str) -> bool:
    if not path.is_file():
        _add_error(errors, "artifact_presence", f"falta {label}: {_rel(path) if path.exists() else path}")
        return False
    return True


def _checks(errors: list[dict]) -> list[dict]:
    return [
        {"name": name, "ok": not any(error["check"] == name for error in errors)}
        for name in CHECK_NAMES
    ]


def _collect_subject(
    cfg: Mapping[str, Any],
    subject: str,
    entry: Mapping[str, Any] | None,
    enabled: list[str],
    errors: list[dict],
    artifacts: list[dict],
) -> dict | None:
    npz_path, meta_path = loader.canonical_paths(subject)
    paths = {
        "npz": npz_path,
        "meta": meta_path,
        "qc": qc.qc_path(subject),
        "windows": window.windows_path(subject),
        "windows_report": window.report_path(subject),
        "features": features.features_path(subject),
        "features_report": features.report_path(subject),
    }
    for kind in enabled:
        paths[f"representation_{kind}"] = represent.representation_path(kind, subject)
        paths[f"representation_{kind}_report"] = represent.report_path(kind, subject)

    present = True
    for name, path in paths.items():
        if not _require(path, errors, f"{name} de {subject}"):
            present = False
    if not present:
        return None

    meta = read_json(meta_path) or {}
    qc_report = read_json(paths["qc"]) or {}
    windows_report = read_json(paths["windows_report"]) or {}
    features_report = read_json(paths["features_report"]) or {}
    windows = read_jsonl(paths["windows"])
    feature_rows = read_jsonl(paths["features"])
    representations = {
        kind: read_jsonl(paths[f"representation_{kind}"]) for kind in enabled
    }
    representation_reports = {
        kind: read_json(paths[f"representation_{kind}_report"]) or {} for kind in enabled
    }

    actual_sha = sha256_file(npz_path)
    if actual_sha != meta.get("canonical_sha256"):
        _add_error(errors, "canonical_hash", f"{subject}: hash del .npz no coincide con meta")

    if entry is not None and meta.get("source_sha256") != entry.get("sha256"):
        _add_error(errors, "source_hash", f"{subject}: hash de origen no coincide con el manifiesto")

    window_ids = [row["window_id"] for row in windows]
    if len(window_ids) != len(set(window_ids)):
        _add_error(errors, "window_alignment", f"{subject}: window_id duplicados en windows")

    feature_ids = [row["window_id"] for row in feature_rows]
    if set(feature_ids) != set(window_ids):
        _add_error(errors, "window_alignment", f"{subject}: features no alineadas con windows")

    for kind, rows in representations.items():
        ids = [row["window_id"] for row in rows]
        if set(ids) != set(window_ids):
            _add_error(
                errors,
                "window_alignment",
                f"{subject}: representación '{kind}' no alineada con windows",
            )

    keep = set(int(value) for value in cfg["labels"]["keep"])
    for row in windows:
        label_id = int(row.get("label", {}).get("id", -1))
        if label_id not in keep:
            _add_error(
                errors,
                "label_scope",
                f"{subject}: etiqueta {label_id} fuera de labels.keep en {row['window_id']}",
            )

    for row in feature_rows:
        if _contains_key(row, "label"):
            _add_error(
                errors,
                "anti_leakage",
                f"{subject}: features contienen clave 'label' en {row['window_id']}",
            )
    for kind, rows in representations.items():
        for row in rows:
            if _contains_key(row, "label"):
                _add_error(
                    errors,
                    "anti_leakage",
                    f"{subject}: representación '{kind}' contiene clave 'label' en {row['window_id']}",
                )

    feature_sizes = {
        sum(len(values) for values in row.get("features", {}).values())
        for row in feature_rows
    }
    if len(feature_sizes) > 1:
        _add_error(errors, "feature_schema", f"{subject}: número de features variable por ventana")

    if int(windows_report.get("n_windows", -1)) != len(windows):
        _add_error(errors, "report_counts", f"{subject}: windows.report no coincide con windows")
    if int(features_report.get("n_windows", -1)) != len(feature_rows):
        _add_error(errors, "report_counts", f"{subject}: features.report no coincide con features")
    for kind, rows in representations.items():
        if int(representation_reports[kind].get("n_windows", -1)) != len(rows):
            _add_error(
                errors,
                "report_counts",
                f"{subject}: report de representación '{kind}' no coincide",
            )

    artifacts.extend(
        [
            _artifact(npz_path, "canonical"),
            _artifact(meta_path, "canonical_meta"),
            _artifact(paths["qc"], "qc"),
            _artifact(paths["windows"], "windows", len(windows)),
            _artifact(paths["windows_report"], "windows_report"),
            _artifact(paths["features"], "features", len(feature_rows)),
            _artifact(paths["features_report"], "features_report"),
        ]
    )
    for kind, rows in representations.items():
        artifacts.append(
            _artifact(paths[f"representation_{kind}"], f"representation_{kind}", len(rows))
        )
        artifacts.append(
            _artifact(
                paths[f"representation_{kind}_report"],
                f"representation_{kind}_report",
            )
        )

    return {
        "subject": subject,
        "meta": meta,
        "qc": qc_report,
        "windows": windows,
        "windows_report": windows_report,
        "features": feature_rows,
        "features_report": features_report,
        "representations": representations,
        "representation_reports": representation_reports,
    }


def _collect(
    cfg: Mapping[str, Any],
    subjects: list[str],
    entries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    enabled = list(representations_config(cfg).get("enabled", []))
    errors: list[dict] = []
    artifacts: list[dict] = []
    states: dict[str, dict] = {}

    for subject in subjects:
        state = _collect_subject(cfg, subject, entries.get(subject), enabled, errors, artifacts)
        if state is not None:
            states[subject] = state

    if not states:
        _add_error(errors, "artifact_presence", "no hay sujetos con artefactos completos")

    # --- índices globales ---
    dataset_manifest = read_json(discover.manifest_path())
    if dataset_manifest is None:
        _add_error(errors, "global_index", "falta dataset_manifest.json")
    else:
        artifacts.append(_artifact(discover.manifest_path(), "dataset_manifest"))
        known = {entry["subject"] for entry in dataset_manifest.get("subjects", [])}
        missing = sorted(set(subjects) - known)
        if missing:
            _add_error(errors, "global_index", f"sujetos fuera del manifiesto: {missing}")

    union_ids = {row["window_id"] for state in states.values() for row in state["windows"]}
    total_windows = sum(len(state["windows"]) for state in states.values())
    total_features = sum(len(state["features"]) for state in states.values())

    global_specs = [
        (window.global_index_path(), "windows_index", total_windows, union_ids),
        (window.global_report_path(), "windows_report_global", None, None),
        (features.global_index_path(), "features_index", total_features, union_ids),
        (features.global_report_path(), "features_report_global", None, None),
        (features.spec_path(), "feature_spec", None, None),
        (represent.spec_path(), "representation_spec", None, None),
    ]
    for kind in enabled:
        count = sum(len(state["representations"].get(kind, [])) for state in states.values())
        global_specs.append((represent.index_path(kind), f"{kind}_index", count, union_ids))
        global_specs.append((represent.global_report_path(kind), f"{kind}_report_global", None, None))

    for path, kind, expected_rows, expected_ids in global_specs:
        if not _require(path, errors, kind):
            continue
        rows = None
        if kind.endswith("_index") or kind == "windows_index":
            rows = read_jsonl(path)
            if expected_rows is not None and len(rows) != expected_rows:
                _add_error(
                    errors,
                    "global_index",
                    f"{kind}: {len(rows)} filas, se esperaban {expected_rows}",
                )
            if expected_ids is not None:
                ids = {row["window_id"] for row in rows}
                if ids != expected_ids:
                    _add_error(errors, "global_index", f"{kind}: window_id no coinciden")
        artifacts.append(_artifact(path, kind, len(rows) if rows is not None else None))

    return states, artifacts, errors


def _build_outputs(cfg: Mapping[str, Any], subjects: list[str], states: Mapping[str, dict]) -> tuple[list[dict], list[dict]]:
    enabled = list(representations_config(cfg).get("enabled", []))
    index_rows: list[dict] = []
    label_rows: list[dict] = []

    for subject in subjects:
        state = states.get(subject)
        if state is None:
            continue
        features_by_id = {row["window_id"]: row for row in state["features"]}
        reps_by_kind = {
            kind: {row["window_id"]: row for row in state["representations"].get(kind, [])}
            for kind in enabled
        }
        for row in state["windows"]:
            window_id = row["window_id"]
            feature_row = features_by_id[window_id]
            representations = {
                kind: {
                    "version": reps_by_kind[kind][window_id].get("representation_version"),
                    "file": _rel(represent.representation_path(kind, subject)),
                }
                for kind in enabled
            }
            fingerprints = {
                "window": row.get("config_fingerprint"),
                "feature": feature_row.get("config_fingerprint"),
                "representations": {
                    kind: reps_by_kind[kind][window_id].get("config_fingerprint")
                    for kind in enabled
                },
            }
            index_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "window_id": window_id,
                    "subject": subject,
                    "t_start_s": row.get("t_start_s"),
                    "t_end_s": row.get("t_end_s"),
                    "duration_s": row.get("duration_s"),
                    "n_features": sum(
                        len(values) for values in feature_row.get("features", {}).values()
                    ),
                    "quality": {
                        "sqi_overall": feature_row.get("quality", {}).get("sqi_overall"),
                        "category": feature_row.get("quality", {}).get("category"),
                        "flags": feature_row.get("quality", {}).get("flags", []),
                    },
                    "representations": representations,
                    "source": row.get("source"),
                    "fingerprints": fingerprints,
                }
            )
            label_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "window_id": window_id,
                    "subject": subject,
                    "label": row.get("label"),
                    "t_start_s": row.get("t_start_s"),
                    "t_end_s": row.get("t_end_s"),
                }
            )

    return index_rows, label_rows


def _counts(cfg: Mapping[str, Any], subjects: list[str], states: Mapping[str, dict]) -> dict:
    enabled = list(representations_config(cfg).get("enabled", []))
    by_label: Counter = Counter()
    by_subject: dict[str, int] = {}
    modalities: set[str] = set()
    feature_values = 0
    features_per_window = None
    representation_counts = {kind: 0 for kind in enabled}

    for subject in subjects:
        state = states.get(subject)
        if state is None:
            continue
        by_subject[subject] = len(state["windows"])
        for row in state["windows"]:
            by_label[str(int(row["label"]["id"]))] += 1
        for row in state["features"]:
            size = sum(len(values) for values in row.get("features", {}).values())
            feature_values += size
            features_per_window = size if features_per_window is None else features_per_window
            modalities.update(row.get("features", {}).keys())
        for kind, rows in state["representations"].items():
            representation_counts[kind] = representation_counts.get(kind, 0) + len(rows)

    return {
        "subjects": len([s for s in subjects if s in states]),
        "windows": sum(by_subject.values()),
        "windows_by_label": dict(sorted(by_label.items())),
        "windows_by_subject": by_subject,
        "features_per_window": features_per_window or 0,
        "feature_values": feature_values,
        "representations": representation_counts,
        "modalities": sorted(modalities),
        "empty_subjects": [s for s in subjects if not states.get(s, {}).get("windows")],
    }


def _verify_previous(previous: Mapping[str, Any]) -> bool:
    """Comprueba que los datos y salidas previos siguen intactos."""
    wanted = DATA_KINDS | {"corpus_index", "groundtruth"}
    verifying = False
    for artifact in previous.get("artifacts", []):
        kind = artifact.get("kind")
        if kind not in wanted:
            continue
        verifying = True
        path = corpus_dir() / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            return False
    return verifying


def run_package(cfg: Mapping[str, Any], subjects: Iterable[str], *, force: bool = False) -> dict:
    """Valida el corpus y escribe índice, etiquetas, manifiesto y checksums."""
    subjects = list(subjects)
    if not subjects:
        raise ValidationError("no hay sujetos que empaquetar", where="package")

    dataset_manifest = read_json(discover.manifest_path())
    entries = {
        entry["subject"]: entry
        for entry in (dataset_manifest or {}).get("subjects", [])
    }

    states, input_artifacts, errors = _collect(cfg, subjects, entries)
    checks = _checks(errors)
    if errors:
        ensure_dir(corpus_dir())
        atomic_write_json(
            errors_path(),
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "checks": checks,
                "errors": errors,
            },
        )
        raise ValidationError(
            f"el corpus no supera la validación ({len(errors)} errores); "
            f"ver {_rel(errors_path())}",
            where="package",
        )

    fingerprint = build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "corpus_version": cfg.get("package", {}).get("corpus_version"),
            "feature_version": FEATURE_VERSION,
            "representation_version": REPRESENTATION_VERSION,
            "artifacts": sorted(
                (artifact["path"], artifact["sha256"])
                for artifact in input_artifacts
                if artifact["kind"] in DATA_KINDS
            ),
            "sources": sorted(
                (subject, entries.get(subject, {}).get("sha256")) for subject in subjects
            ),
            "config": sha256_json(cfg),
            # Los checksums de los artefactos ya capturan el contenido: el
            # commit de git no debe invalidar el corpus.
            "code": code_fingerprint(_CODE_SCOPE),
        }
    )

    previous = read_json(manifest_path())
    reused = (
        not force
        and isinstance(previous, dict)
        and previous.get("fingerprint") == fingerprint
        and _verify_previous(previous)
        and index_path().is_file()
        and labels_path().is_file()
    )

    if not reused:
        index_rows, label_rows = _build_outputs(cfg, subjects, states)
        atomic_write_jsonl(index_path(), index_rows)
        atomic_write_jsonl(labels_path(), label_rows)
    else:
        index_rows = read_jsonl(index_path())
        label_rows = read_jsonl(labels_path())

    output_artifacts = [
        _artifact(index_path(), "corpus_index", len(index_rows)),
        _artifact(labels_path(), "groundtruth", len(label_rows)),
    ]
    all_artifacts = sorted(
        input_artifacts + output_artifacts, key=lambda artifact: artifact["path"]
    )
    counts = _counts(cfg, subjects, states)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "corpus_version": cfg.get("package", {}).get("corpus_version"),
        "created_at": utc_now(),
        "fingerprint": fingerprint,
        "dataset": cfg.get("dataset"),
        "subject_scope": subjects,
        "versions": {
            "feature": FEATURE_VERSION,
            "representation": REPRESENTATION_VERSION,
            **library_versions(),
        },
        "git": git_info(),
        "config_fingerprint": sha256_json(cfg),
        "counts": counts,
        "integrity": {
            "ok": True,
            "checks": checks,
            "errors": [],
        },
        "artifacts": all_artifacts,
        "checksums_file": "corpus_checksums.sha256",
    }
    atomic_write_json(manifest_path(), manifest)

    lines = [f"{artifact['sha256']}  {artifact['path']}" for artifact in all_artifacts]
    lines.append(f"{sha256_file(manifest_path())}  corpus_manifest.json")
    atomic_write_text(checksums_path(), "\n".join(sorted(lines)) + "\n")

    if errors_path().exists():
        errors_path().unlink(missing_ok=True)

    status = "reused" if reused else "ok"
    writeLog(
        "info",
        logger,
        f"[package] {status}: corpus '{manifest['corpus_version']}' con "
        f"{counts['windows']} ventanas de {counts['subjects']} sujetos, "
        f"{len(all_artifacts) + 1} artefactos con checksum",
    )
    return {
        "status": status,
        "fingerprint": fingerprint,
        "corpus_version": manifest["corpus_version"],
        "n_windows": counts["windows"],
        "n_artifacts": len(all_artifacts) + 1,
        "manifest": str(manifest_path()),
        "checksums": str(checksums_path()),
    }
