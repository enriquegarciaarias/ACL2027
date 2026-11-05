"""Tests del control de calidad (H2)."""
from __future__ import annotations

import numpy as np
import pytest

from sources.common.paths import read_json
from sources.common.validation import ValidationError
from sources.ingestion import discover, loader, qc


def _prepare(cfg):
    manifest = discover.run_discover(cfg)
    entry = discover.selected_entries(manifest)[0]
    loader.run_ingest(cfg, entry)
    return entry


def test_qc_report_has_all_channels_and_labels(synthetic_dataset, ingestion_cfg):
    entry = _prepare(ingestion_cfg)
    result = qc.run_qc(ingestion_cfg, entry.subject)

    assert result["status"] == "ok"
    report = read_json(qc.qc_path(entry.subject))
    assert report["subject"] == "S2"
    assert list(report["channels"]) == list(qc.CHANNEL_ORDER)
    assert report["labels"]["counts"] == {"0": 42000, "1": 84000, "2": 84000}
    assert report["labels"]["transitions"] == 2
    assert report["summary"]["n_channels"] == 10
    assert report["quality_ok"] is True
    assert all(not channel["flags"] for channel in report["channels"].values())


def test_qc_is_idempotent_and_force_recomputes(synthetic_dataset, ingestion_cfg):
    entry = _prepare(ingestion_cfg)

    first = qc.run_qc(ingestion_cfg, entry.subject)
    second = qc.run_qc(ingestion_cfg, entry.subject)
    forced = qc.run_qc(ingestion_cfg, entry.subject, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "reused"
    assert forced["status"] == "ok"
    assert first["fingerprint"] == second["fingerprint"] == forced["fingerprint"]


def test_flat_channel_is_flagged():
    cfg = {"qc": {"flatline_std_eps": 1e-9, "max_zero_diff_fraction": 0.99,
                  "max_nan_fraction": 0.01, "saturation_fraction_threshold": 0.05}}
    report = qc.compute_channel_qc("chest.EDA", np.zeros(1000), cfg)

    assert report["std"] == 0.0
    assert "constant" in report["flags"]
    assert "flatline" in report["flags"]
    assert "saturated" in report["flags"]


def test_nan_channel_is_flagged():
    cfg = {"qc": {"max_nan_fraction": 0.01, "flatline_std_eps": 1e-9}}
    values = np.ones(100)
    values[:10] = np.nan
    report = qc.compute_channel_qc("chest.ECG", values, cfg)

    assert report["nan_count"] == 10
    assert "high_nan" in report["flags"]


def test_qc_without_canonical_fails(workspace, ingestion_cfg):
    with pytest.raises(ValidationError, match="no existe el canónico"):
        qc.run_qc(ingestion_cfg, "S2")
