"""Tests de representaciones numérica y textual (H5)."""
from __future__ import annotations

import copy

import pytest

from sources.common.paths import read_json, read_jsonl
from sources.common.validation import ValidationError
from sources.ingestion import discover, features, loader, qc, represent, window


def _prepare(cfg):
    manifest = discover.run_discover(cfg)
    entry = discover.selected_entries(manifest)[0]
    loader.run_ingest(cfg, entry)
    qc.run_qc(cfg, entry.subject)
    window.run_window(cfg, entry.subject)
    features.run_features(cfg, entry.subject)
    return entry.subject


def _contains_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(value, key) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(value, key) for value in obj)
    return False


def _cfg_no_embeddings(ingestion_cfg):
    """Devuelve una copia del cfg con representaciones solo numeric+text."""
    import copy
    cfg = copy.deepcopy(ingestion_cfg)
    cfg["representations"] = {"enabled": ["numeric", "text"]}
    return cfg


def test_numeric_and_text_representations(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    cfg = _cfg_no_embeddings(ingestion_cfg)
    result = represent.run_represent(cfg, subject)

    assert result["status"] == {"numeric": "ok", "text": "ok"}
    assert result["n_windows"] == 2

    numeric_rows = read_jsonl(represent.representation_path("numeric", subject))
    assert len(numeric_rows) == 2
    payload = numeric_rows[0]["payload"]
    assert payload["window"]["window_id"] == numeric_rows[0]["window_id"]
    assert "duration_s" in payload["window"]
    assert "ecg" in payload["modalities"]
    assert payload["modalities"]["ecg"]["ecg_hr_bpm"]["unit"] == "bpm"
    assert payload["quality"]["sqi_overall"] is not None
    assert not _contains_key(numeric_rows[0], "label")

    text_rows = read_jsonl(represent.representation_path("text", subject))
    assert len(text_rows) == 2
    text = text_rows[0]["payload"]
    assert isinstance(text, str)
    assert "PHYSIOLOGICAL SIGNAL SUMMARY" in text
    assert "Heart rate" in text
    assert "SIGNAL QUALITY" in text
    assert "baseline" not in text.lower()
    assert "stress" not in text.lower()

    assert represent.spec_path().is_file()
    spec = read_json(represent.spec_path())
    assert spec["representation_version"] == "1.0"
    assert set(spec["enabled"]) == {"numeric", "text"}


def test_representations_are_idempotent(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    cfg = _cfg_no_embeddings(ingestion_cfg)

    first = represent.run_represent(cfg, subject)
    second = represent.run_represent(cfg, subject)
    forced = represent.run_represent(cfg, subject, force=True)

    assert first["status"] == {"numeric": "ok", "text": "ok"}
    assert second["status"] == {"numeric": "reused", "text": "reused"}
    assert forced["status"] == {"numeric": "ok", "text": "ok"}


def test_global_representation_indexes(synthetic_dataset, ingestion_cfg):
    manifest = discover.run_discover(ingestion_cfg)
    subjects = []
    cfg = _cfg_no_embeddings(ingestion_cfg)
    for entry in discover.selected_entries(manifest):
        loader.run_ingest(ingestion_cfg, entry)
        qc.run_qc(ingestion_cfg, entry.subject)
        window.run_window(ingestion_cfg, entry.subject)
        features.run_features(ingestion_cfg, entry.subject)
        represent.run_represent(cfg, entry.subject)
        subjects.append(entry.subject)

    summary = represent.build_global_index(cfg, subjects)
    assert summary["numeric"]["n_windows"] == 4
    assert summary["text"]["n_windows"] == 4
    assert len(read_jsonl(represent.index_path("numeric"))) == 4
    assert len(read_jsonl(represent.index_path("text"))) == 4
    assert read_json(represent.global_report_path("numeric"))["per_subject"] == {"S2": 2, "S3": 2}


def test_list_form_is_backward_compatible(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    cfg = copy.deepcopy(ingestion_cfg)
    cfg["representations"] = ["numeric"]

    result = represent.run_represent(cfg, subject, force=True)
    assert result["status"] == {"numeric": "ok"}
    assert represent.representation_path("numeric", subject).is_file()


def test_represent_without_features_fails(workspace, ingestion_cfg):
    with pytest.raises(ValidationError, match="no existe el canónico"):
        represent.run_represent(ingestion_cfg, "S2")
