"""Tests de extracción de características y SQI (H4)."""
from __future__ import annotations

import numpy as np

from sources.common.paths import read_json, read_jsonl
from sources.ingestion import discover, features, loader, qc, window
from sources.ingestion.features import registry


def _prepare(cfg):
    manifest = discover.run_discover(cfg)
    entry = discover.selected_entries(manifest)[0]
    loader.run_ingest(cfg, entry)
    qc.run_qc(cfg, entry.subject)
    window.run_window(cfg, entry.subject)
    return entry.subject


def _contains_key(obj, key: str) -> bool:
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(value, key) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(value, key) for value in obj)
    return False


def test_features_have_provenance_and_no_label(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    result = features.run_features(ingestion_cfg, subject)

    assert result["status"] == "ok"
    assert result["n_windows"] == 2

    rows = read_jsonl(features.features_path(subject))
    assert len(rows) == 2
    for row in rows:
        assert row["feature_version"] == "1.0"
        assert not _contains_key(row, "label")
        assert "ecg" in row["features"]

        hr = row["features"]["ecg"]["ecg_hr_bpm"]
        assert hr["value"] is not None
        assert hr["unit"] == "bpm"
        assert hr["method"]
        assert hr["method_version"]
        assert hr["kind"] == "derived"

        quality = row["quality"]
        assert 0.0 <= quality["sqi_overall"] <= 1.0
        assert quality["category"] in {"good", "fair", "poor"}
        assert "ecg" in quality["per_modality"]
        assert "sqi" in quality["per_modality"]["ecg"]

    assert features.spec_path().is_file()
    spec = read_json(features.spec_path())
    assert spec["feature_version"] == "1.0"
    assert any(item["name"] == "ecg" for item in spec["modalities"])


def test_time_domain_only_hrv(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    features.run_features(ingestion_cfg, subject)

    rows = read_jsonl(features.features_path(subject))
    ecg_keys = set(rows[0]["features"]["ecg"])
    assert "ecg_rmssd_ms" in ecg_keys
    assert "ecg_sdnn_ms" in ecg_keys
    assert not any("lf" in key or "hf" in key for key in ecg_keys)


def test_features_are_idempotent(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)

    first = features.run_features(ingestion_cfg, subject)
    second = features.run_features(ingestion_cfg, subject)
    forced = features.run_features(ingestion_cfg, subject, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "reused"
    assert forced["status"] == "ok"
    assert first["fingerprint"] == second["fingerprint"] == forced["fingerprint"]


def test_global_features_index(synthetic_dataset, ingestion_cfg):
    manifest = discover.run_discover(ingestion_cfg)
    subjects = []
    for entry in discover.selected_entries(manifest):
        loader.run_ingest(ingestion_cfg, entry)
        qc.run_qc(ingestion_cfg, entry.subject)
        window.run_window(ingestion_cfg, entry.subject)
        features.run_features(ingestion_cfg, entry.subject)
        subjects.append(entry.subject)

    report = features.build_global_index(ingestion_cfg, subjects)
    assert report["n_subjects"] == 2
    assert report["n_windows"] == 4
    assert report["n_feature_values"] > 0
    assert report["sqi_mean"] is not None
    assert len(read_jsonl(features.global_index_path())) == 4


def test_flat_signal_does_not_crash(ingestion_cfg):
    n = 700 * 60
    arrays = {
        "chest_ACC": np.zeros((n, 3)),
        "chest_ECG": np.zeros(n),
        "chest_EMG": np.zeros(n),
        "chest_EDA": np.zeros(n),
        "chest_Temp": np.zeros(n),
        "chest_Resp": np.zeros(n),
        "wrist_ACC": np.zeros((32 * 60, 3)),
        "wrist_BVP": np.zeros(64 * 60),
        "wrist_EDA": np.zeros(4 * 60),
        "wrist_TEMP": np.zeros(4 * 60),
    }
    w = {
        "window_id": "test:w00001",
        "duration_s": 60.0,
        "ranges": {
            "chest": [0, n],
            "wrist_acc": [0, 32 * 60],
            "wrist_bvp": [0, 64 * 60],
            "wrist_eda": [0, 4 * 60],
            "wrist_temp": [0, 4 * 60],
        },
        "coverage": {
            "chest": 1.0,
            "wrist_acc": 1.0,
            "wrist_bvp": 1.0,
            "wrist_eda": 1.0,
            "wrist_temp": 1.0,
        },
    }
    modality_features, quality = registry.extract_window(arrays, w, ingestion_cfg)

    assert isinstance(modality_features, dict)
    assert 0.0 <= quality["sqi_overall"] <= 1.0
    assert quality["sqi_min"] <= quality["sqi_max"]
