"""Tests del adaptador de PhysioNet/CinC 2017."""

from __future__ import annotations

import json

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


# ---------------------------------------------------------------------------
# Procedencia del motor y aborto si el modelo no está descargado
# ---------------------------------------------------------------------------


_VALID_RHYTHM_OUTPUT = (
    '{"decision": "normal", "confidence": 80, '
    '"claims": [{"metric": "ecg_hr_bpm", "channel": "ecg", '
    '"direction": "stable", "magnitude": "low"}], '
    '"reasoning": "Heart rate is near the population median."}'
)


def _one_record() -> RhythmRecord:
    values = {name: 0.0 for name in CANONICAL_FEATURES}
    return RhythmRecord(
        record_id="A00001", label="normal", fs=300.0, duration_s=30.0, values=values
    )


def test_run_llm_writes_engine_provenance(tmp_path, monkeypatch):
    from scripts import run_p3_physionet as pn

    monkeypatch.setattr(pn, "check_model_available", lambda _model: True)
    monkeypatch.setattr(
        pn,
        "ollama_run_provenance",
        lambda model: {"model": model, "ollama_version": "0.40.6", "model_digest": "abc"},
    )
    monkeypatch.setattr(
        pn,
        "_call_ollama_chat",
        lambda *a, **k: (_VALID_RHYTHM_OUTPUT, {"model": "qwen2.5:7b"}),
    )

    centers = {name: 0.0 for name in CANONICAL_FEATURES}
    scales = {name: 1.0 for name in CANONICAL_FEATURES}
    pn.run_llm(
        [_one_record()],
        output_dir=tmp_path,
        centers=centers,
        scales=scales,
        model="qwen2.5:7b",
        temperature=0.0,
        seed=42,
        max_records=None,
        tag="fs",
    )

    prov = json.loads((tmp_path / "run_provenance_fs.json").read_text(encoding="utf-8"))
    assert prov["ollama_version"] == "0.40.6"
    assert prov["model"] == "qwen2.5:7b"
    assert prov["task"] == "physionet2017_rhythm4"
    assert prov["tag"] == "fs"
    assert prov["config"]["model"] == "qwen2.5:7b"
    # La respuesta se escribe aparte y sin etiqueta.
    assert (tmp_path / "physionet2017_fs.responses.jsonl").is_file()


def test_run_llm_aborts_when_model_is_not_available(tmp_path, monkeypatch):
    from scripts import run_p3_physionet as pn

    monkeypatch.setattr(pn, "check_model_available", lambda _model: False)

    centers = {name: 0.0 for name in CANONICAL_FEATURES}
    scales = {name: 1.0 for name in CANONICAL_FEATURES}
    with pytest.raises(SystemExit) as exc:
        pn.run_llm(
            [_one_record()],
            output_dir=tmp_path,
            centers=centers,
            scales=scales,
            model="medgemma:27b",
            temperature=0.0,
            seed=42,
            max_records=None,
        )
    assert "ollama pull" in str(exc.value)
    # No se escribe ni una respuesta: no se cachean timeouts.
    assert not list(tmp_path.glob("*.responses.jsonl"))
    assert not (tmp_path / "run_provenance.json").exists()
