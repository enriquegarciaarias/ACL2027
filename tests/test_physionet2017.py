"""Tests del adaptador de PhysioNet/CinC 2017."""

from __future__ import annotations

import pytest

from sources.external.physionet2017 import (
    CANONICAL_FEATURES,
    LABEL_MAP,
    RHYTHM_CLASSES,
    build_payload,
    population_reference,
    select_records,
    split_records,
    z_scores,
)
from sources.external.physionet2017 import RhythmRecord


def test_label_mapping():
    assert LABEL_MAP == {"N": "normal", "A": "af", "O": "other-rhythm", "~": "noisy"}
    assert RHYTHM_CLASSES == ("normal", "af", "other-rhythm", "noisy")


def test_select_records_is_stratified_and_deterministic():
    labels = {
        **{f"N{i}": "normal" for i in range(20)},
        **{f"A{i}": "af" for i in range(10)},
        **{f"O{i}": "other-rhythm" for i in range(5)},
        **{f"X{i}": "noisy" for i in range(3)},
    }
    first = select_records(labels, per_class=4, seed=42)
    second = select_records(labels, per_class=4, seed=42)
    assert first == second
    counts = {label: 0 for label in RHYTHM_CLASSES}
    for record_id in first:
        counts[labels[record_id]] += 1
    assert counts == {"normal": 4, "af": 4, "other-rhythm": 4, "noisy": 3}


def test_split_is_disjoint_and_deterministic():
    ids = [f"ID{i:03d}" for i in range(100)]
    dev1, test1 = split_records(ids, dev_fraction=0.3)
    dev2, test2 = split_records(ids, dev_fraction=0.3)
    assert dev1 == dev2 and test1 == test2
    assert set(dev1).isdisjoint(test1)
    assert set(dev1) | set(test1) == set(ids)


def _record(name: str, values: dict) -> RhythmRecord:
    return RhythmRecord(record_id=name, label="normal", fs=300.0, duration_s=30.0, values=values)


def test_population_reference_and_z_scores():
    rows = [
        _record("a", {"ecg_hr_bpm": 60.0, "ecg_rr_mean_ms": 1000.0}),
        _record("b", {"ecg_hr_bpm": 70.0, "ecg_rr_mean_ms": 900.0}),
        _record("c", {"ecg_hr_bpm": 80.0, "ecg_rr_mean_ms": 800.0}),
    ]
    centers, scales = population_reference(rows)
    assert centers["ecg_hr_bpm"] == pytest.approx(70.0)
    assert scales["ecg_hr_bpm"] > 0
    scores = z_scores({"ecg_hr_bpm": 80.0, "ecg_rr_mean_ms": None}, centers, scales)
    assert scores["ecg_hr_bpm"] > 0
    assert scores["ecg_rr_mean_ms"] is None


def test_build_payload_has_only_ecg_and_all_features():
    payload = build_payload("A00001", {name: 0.5 for name in CANONICAL_FEATURES}, 30.0)
    assert set(payload["modalities"]) == {"ecg"}
    assert set(payload["modalities"]["ecg"]) == set(CANONICAL_FEATURES)
    assert payload["window"]["duration_s"] == 30.0
    assert payload["window"]["window_id"].startswith("win_")
