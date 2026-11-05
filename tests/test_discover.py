"""Tests del descubrimiento y del manifiesto del dataset."""
from __future__ import annotations

from pathlib import Path

from sources.ingestion import discover


def test_manifest_lists_subjects_and_ignores_duplicates(synthetic_dataset, ingestion_cfg):
    manifest = discover.run_discover(ingestion_cfg)

    assert manifest["dataset"] == "WESAD"
    assert [entry["subject"] for entry in manifest["subjects"]] == ["S2", "S3"]
    assert all(entry["selected"] for entry in manifest["subjects"])
    assert all(len(entry["sha256"]) == 64 for entry in manifest["subjects"])
    assert any("WESAD.zip" in item for item in manifest["ignored"])
    assert any("wesad_readme.pdf" in item for item in manifest["ignored"])
    assert manifest["subjects"][0]["notes"] == ["Test fixture"]

    assert discover.manifest_path().is_file()
    reloaded = discover.read_json(discover.manifest_path())
    assert reloaded["config_fingerprint"] == manifest["config_fingerprint"]


def test_selected_filter_marks_only_requested_subjects(synthetic_dataset, ingestion_cfg):
    manifest = discover.run_discover(ingestion_cfg, requested=["S3"])

    selected = {entry["subject"] for entry in manifest["subjects"] if entry["selected"]}
    assert selected == {"S3"}
    assert [entry.subject for entry in discover.selected_entries(manifest)] == ["S3"]


def test_hash_cache_avoids_rehashing(synthetic_dataset, ingestion_cfg, monkeypatch):
    discover.run_discover(ingestion_cfg)

    def _boom(*args, **kwargs):
        raise AssertionError("no debería recalcular el hash con la caché válida")

    monkeypatch.setattr(discover, "sha256_file", _boom)
    manifest = discover.run_discover(ingestion_cfg)
    assert len(manifest["subjects"]) == 2


def test_missing_pickle_is_reported_as_ignored(synthetic_dataset, ingestion_cfg):
    (synthetic_dataset["dataset"] / "S4").mkdir()
    manifest = discover.run_discover(ingestion_cfg)

    assert [entry["subject"] for entry in manifest["subjects"]] == ["S2", "S3"]
    assert any("S4" in item and "falta" in item for item in manifest["ignored"])


def test_missing_dataset_root_raises(workspace, ingestion_cfg):
    import pytest

    cfg = dict(ingestion_cfg)
    cfg["source_dir"] = "does-not-exist"
    with pytest.raises(FileNotFoundError):
        discover.run_discover(cfg)
