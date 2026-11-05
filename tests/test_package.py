"""Tests del empaquetado final del corpus (H6)."""
from __future__ import annotations

import pytest

from sources.common.paths import read_json, read_jsonl
from sources.common.validation import ValidationError
from sources.ingestion import discover, features, loader, package, qc, represent, window


def _prepare_all(cfg):
    manifest = discover.run_discover(cfg)
    subjects = []
    for entry in discover.selected_entries(manifest):
        loader.run_ingest(cfg, entry)
        qc.run_qc(cfg, entry.subject)
        window.run_window(cfg, entry.subject)
        features.run_features(cfg, entry.subject)
        represent.run_represent(cfg, entry.subject)
        subjects.append(entry.subject)
    window.build_global_index(cfg, subjects)
    features.build_global_index(cfg, subjects)
    represent.build_global_index(cfg, subjects)
    return subjects


def _contains_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(value, key) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(value, key) for value in obj)
    return False


def test_package_creates_manifest_index_labels_and_checksums(synthetic_dataset, ingestion_cfg):
    subjects = _prepare_all(ingestion_cfg)
    result = package.run_package(ingestion_cfg, subjects)

    assert result["status"] == "ok"
    assert result["n_windows"] == 4
    assert package.manifest_path().is_file()
    assert package.index_path().is_file()
    assert package.labels_path().is_file()
    assert package.checksums_path().is_file()

    manifest = read_json(package.manifest_path())
    assert manifest["integrity"]["ok"] is True
    assert all(check["ok"] for check in manifest["integrity"]["checks"])
    assert manifest["counts"]["subjects"] == 2
    assert manifest["counts"]["windows"] == 4
    assert manifest["counts"]["windows_by_label"] == {"1": 2, "2": 2}
    assert manifest["counts"]["windows_by_subject"] == {"S2": 2, "S3": 2}
    assert manifest["counts"]["features_per_window"] > 0
    assert manifest["counts"]["representations"] == {"numeric": 4, "text": 4}
    assert manifest["corpus_version"] == "0.1.0"
    assert len(manifest["artifacts"]) > 0

    index_rows = read_jsonl(package.index_path())
    assert len(index_rows) == 4
    assert all(not _contains_key(row, "label") for row in index_rows)
    assert all(row["representations"]["numeric"]["version"] == "1.0" for row in index_rows)
    assert all(row["representations"]["text"]["version"] == "1.0" for row in index_rows)

    label_rows = read_jsonl(package.labels_path())
    assert len(label_rows) == 4
    assert {row["label"]["id"] for row in label_rows} == {1, 2}

    lines = package.checksums_path().read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(manifest["artifacts"]) + 1
    assert any(line.endswith("  corpus_manifest.json") for line in lines)


def test_package_is_idempotent(synthetic_dataset, ingestion_cfg):
    subjects = _prepare_all(ingestion_cfg)

    first = package.run_package(ingestion_cfg, subjects)
    second = package.run_package(ingestion_cfg, subjects)
    forced = package.run_package(ingestion_cfg, subjects, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "reused"
    assert forced["status"] == "ok"
    assert first["fingerprint"] == second["fingerprint"] == forced["fingerprint"]


def test_package_detects_corruption(synthetic_dataset, ingestion_cfg):
    subjects = _prepare_all(ingestion_cfg)
    package.run_package(ingestion_cfg, subjects)

    npz_path, _ = loader.canonical_paths("S2")
    with npz_path.open("ab") as fh:
        fh.write(b"corruption")

    with pytest.raises(ValidationError, match="no supera la validación"):
        package.run_package(ingestion_cfg, subjects)

    errors = read_json(package.errors_path())
    assert any(error["check"] == "canonical_hash" for error in errors["errors"])
