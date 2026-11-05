"""Tests de carga, validación y canonicalización de un sujeto."""
from __future__ import annotations

import numpy as np
import pytest

from sources.common.validation import ValidationError
from sources.ingestion import discover, loader


def _first_entry(cfg):
    manifest = discover.run_discover(cfg)
    return discover.selected_entries(manifest)[0]


def test_validate_pickle_accepts_synthetic_subject(synthetic_dataset, ingestion_cfg):
    entry = _first_entry(ingestion_cfg)
    raw = loader.load_pickle(entry.file)
    info = loader.validate_pickle(raw, entry.subject, ingestion_cfg)

    assert info["n_samples"] == 700 * 300
    assert info["duration_s"] == pytest.approx(300.0)
    assert info["wrist_lengths"] == info["expected_wrist_lengths"]


def test_validate_pickle_rejects_wrong_label_length(synthetic_dataset, ingestion_cfg):
    entry = _first_entry(ingestion_cfg)
    raw = loader.load_pickle(entry.file)
    raw["label"] = raw["label"][:-1]

    with pytest.raises(ValidationError, match="label tiene"):
        loader.validate_pickle(raw, entry.subject, ingestion_cfg)


def test_validate_pickle_rejects_subject_mismatch(synthetic_dataset, ingestion_cfg):
    entry = _first_entry(ingestion_cfg)
    raw = loader.load_pickle(entry.file)

    with pytest.raises(ValidationError, match="subject del pkl"):
        loader.validate_pickle(raw, "S99", ingestion_cfg)


def test_ingest_writes_canonical_npz_and_meta(synthetic_dataset, ingestion_cfg):
    entry = _first_entry(ingestion_cfg)
    result = loader.run_ingest(ingestion_cfg, entry)

    assert result["status"] == "ok"
    npz_path, meta_path = loader.canonical_paths(entry.subject)
    assert npz_path.is_file() and meta_path.is_file()

    arrays = loader.load_canonical(entry.subject)
    assert set(arrays) == {
        "chest_ACC", "chest_ECG", "chest_EMG", "chest_EDA", "chest_Temp", "chest_Resp",
        "wrist_ACC", "wrist_BVP", "wrist_EDA", "wrist_TEMP", "label",
    }
    assert arrays["chest_ECG"].shape == (700 * 300,)
    assert arrays["chest_ACC"].shape == (700 * 300, 3)
    assert arrays["wrist_ACC"].shape == (32 * 300, 3)
    assert arrays["label"].dtype == np.int32
    assert arrays["chest_ECG"].dtype == np.float32
    assert np.isfinite(arrays["wrist_ACC"]).all()

    meta = loader.read_json(meta_path)
    assert meta["schema_version"] == "1.0"
    assert meta["source_sha256"] == entry.sha256
    assert meta["label_counts"] == {"0": 42000, "1": 84000, "2": 84000}
    assert meta["units"]["wrist.ACC"] == "g"
    assert meta["shapes"]["wrist_ACC"] == [32 * 300, 3]
    assert len(meta["canonical_sha256"]) == 64
    assert all(report["violations"] == 0 for report in meta["unit_reports"])


def test_ingest_is_idempotent_and_force_recomputes(synthetic_dataset, ingestion_cfg):
    entry = _first_entry(ingestion_cfg)

    first = loader.run_ingest(ingestion_cfg, entry)
    second = loader.run_ingest(ingestion_cfg, entry)
    forced = loader.run_ingest(ingestion_cfg, entry, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "reused"
    assert forced["status"] == "ok"
    assert first["fingerprint"] == second["fingerprint"] == forced["fingerprint"]
