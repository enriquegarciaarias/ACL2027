"""Tests de segmentación temporal (H3)."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sources.common.paths import read_json, read_jsonl
from sources.ingestion import discover, loader, qc, window


def _prepare(cfg):
    manifest = discover.run_discover(cfg)
    entry = discover.selected_entries(manifest)[0]
    loader.run_ingest(cfg, entry)
    qc.run_qc(cfg, entry.subject)
    return entry.subject


def test_label_runs_detects_contiguous_segments():
    runs = window.label_runs(np.array([1, 1, 1, 2, 2, 0]))
    assert runs == [
        {"label": 1, "start": 0, "end": 3},
        {"label": 2, "start": 3, "end": 5},
        {"label": 0, "start": 5, "end": 6},
    ]


def test_window_generates_two_pure_windows(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    result = window.run_window(ingestion_cfg, subject)

    assert result["status"] == "ok"
    assert result["n_windows"] == 2
    assert result["n_by_label"] == {"1": 1, "2": 1}

    rows = read_jsonl(window.windows_path(subject))
    assert [row["label"]["id"] for row in rows] == [1, 2]
    assert [row["label"]["name"] for row in rows] == ["baseline", "stress"]
    assert all(row["purity"] == 1.0 for row in rows)
    assert all(row["guard_applied_s"] == 5.0 for row in rows)
    assert all(row["ranges"]["chest"][1] - row["ranges"]["chest"][0] == 42000 for row in rows)
    assert all(row["coverage"]["chest"] == 1.0 for row in rows)
    assert all(not row["flags"] for row in rows)

    # La primera ventana empieza tras la guarda de 5 s del tramo baseline.
    assert rows[0]["t_start_s"] == pytest.approx(5.0)
    assert rows[0]["t_end_s"] == pytest.approx(65.0)
    assert rows[0]["ranges"]["wrist_eda"] == [20, 260]


def test_window_never_crosses_a_transition(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    window.run_window(ingestion_cfg, subject)

    arrays = loader.load_canonical(subject)
    for row in read_jsonl(window.windows_path(subject)):
        start, end = row["ranges"]["label"]
        segment = arrays["label"][start:end]
        assert len(np.unique(segment)) == 1
        assert int(segment[0]) == row["label"]["id"]


def test_guard_can_discard_short_runs(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)
    cfg = copy.deepcopy(ingestion_cfg)
    cfg["windows"]["guard_s"] = 60.0

    result = window.run_window(cfg, subject, force=True)
    assert result["n_windows"] == 0
    assert result["drops"]["short_run"] == 2


def test_window_is_idempotent(synthetic_dataset, ingestion_cfg):
    subject = _prepare(ingestion_cfg)

    first = window.run_window(ingestion_cfg, subject)
    second = window.run_window(ingestion_cfg, subject)
    forced = window.run_window(ingestion_cfg, subject, force=True)

    assert first["status"] == "ok"
    assert second["status"] == "reused"
    assert forced["status"] == "ok"
    assert first["fingerprint"] == second["fingerprint"] == forced["fingerprint"]


def test_global_index_concatenates_subjects(synthetic_dataset, ingestion_cfg):
    manifest = discover.run_discover(ingestion_cfg)
    subjects = []
    for entry in discover.selected_entries(manifest):
        loader.run_ingest(ingestion_cfg, entry)
        qc.run_qc(ingestion_cfg, entry.subject)
        window.run_window(ingestion_cfg, entry.subject)
        subjects.append(entry.subject)

    report = window.build_global_index(ingestion_cfg, subjects)
    assert report["n_subjects"] == 2
    assert report["n_windows"] == 4
    assert report["n_by_label"] == {"1": 2, "2": 2}
    assert len(read_jsonl(window.global_index_path())) == 4

    stored = read_json(window.global_report_path())
    assert stored["per_subject"] == {"S2": 2, "S3": 2}
