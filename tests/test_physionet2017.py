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


def test_certainty_superiority_and_cliffs_delta_are_consistent():
    """Cliff's δ es 2A − 1; publicar A como δ triplica la magnitud leída."""
    from scripts.analyze_p3_physionet import _certainty_significance

    # A = (2 victorias + 2 empates/2) / 4 = 0.75  →  δ = +0.5
    grouped = {
        "noisy": [{"certainty_index": 1.0}, {"certainty_index": 1.0}],
        "normal": [{"certainty_index": -1.0}, {"certainty_index": 1.0}],
    }
    significance = _certainty_significance(grouped)
    assert significance["noisy_vs_rest_superiority"] == pytest.approx(0.75)
    assert significance["noisy_vs_rest_cliffs_delta"] == pytest.approx(0.5)

    # Dominancia total: A = 1 → δ = +1 (y nunca 2).
    extreme = {
        "noisy": [{"certainty_index": 2.0}],
        "normal": [{"certainty_index": 0.0}, {"certainty_index": 0.5}],
    }
    dominance = _certainty_significance(extreme)
    assert dominance["noisy_vs_rest_superiority"] == pytest.approx(1.0)
    assert dominance["noisy_vs_rest_cliffs_delta"] == pytest.approx(1.0)


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


def test_run_llm_does_not_relabel_provenance_when_nothing_pending(tmp_path, monkeypatch):
    """Reanudar sin llamadas no puede etiquetar con el motor actual.

    El corpus local de PhysioNet se generó con Ollama 0.30.7; al copiarlo al
    servidor (0.40.6), una reanudación sin `--tag` reescribía la procedencia y
    afirmaba un motor que no lo produjo (mismo riesgo que se corrigió en el
    barrido de degradación).
    """
    from scripts import run_p3_physionet as pn

    record = _one_record()
    (tmp_path / "physionet2017.responses.jsonl").write_text(
        json.dumps({"record_id": record.record_id, "status": "valid_response"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run_provenance.json").write_text(
        json.dumps({"ollama_version": "0.30.7", "model": "qwen2.5:7b"}),
        encoding="utf-8",
    )

    calls: list[str] = []
    monkeypatch.setattr(pn, "check_model_available", lambda m: calls.append(m) or True)
    monkeypatch.setattr(
        pn, "ollama_run_provenance", lambda m: pytest.fail("no debe consultarse")
    )
    monkeypatch.setattr(pn, "_call_ollama_chat", lambda *a, **k: pytest.fail("sin LLM"))

    centers = {name: 0.0 for name in CANONICAL_FEATURES}
    scales = {name: 1.0 for name in CANONICAL_FEATURES}
    responses = pn.run_llm(
        [record],
        output_dir=tmp_path,
        centers=centers,
        scales=scales,
        model="qwen2.5:7b",
        temperature=0.0,
        seed=42,
        max_records=None,
    )

    assert len(responses) == 1
    assert calls == []  # ni se comprueba el modelo: no hay nada que generar
    prov = json.loads((tmp_path / "run_provenance.json").read_text(encoding="utf-8"))
    assert prov["ollama_version"] == "0.30.7"


def test_run_llm_aborts_when_engine_changes_mid_corpus(tmp_path, monkeypatch):
    """Con llamadas pendientes y otro motor registrado, aborta por defecto.

    Mezclar motores dentro del mismo fichero no se detectaría después (la caché
    de reanudación no indexa el motor) y dejaría un corpus con dos motores.
    """
    from scripts import run_p3_physionet as pn

    (tmp_path / "physionet2017.responses.jsonl").write_text(
        json.dumps({"record_id": "B99999", "status": "valid_response"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run_provenance.json").write_text(
        json.dumps({"ollama_version": "0.30.7"}), encoding="utf-8"
    )

    monkeypatch.setattr(pn, "check_model_available", lambda _model: True)
    monkeypatch.setattr(
        pn,
        "ollama_run_provenance",
        lambda model: {"model": model, "ollama_version": "0.40.6"},
    )

    centers = {name: 0.0 for name in CANONICAL_FEATURES}
    scales = {name: 1.0 for name in CANONICAL_FEATURES}
    with pytest.raises(SystemExit) as excinfo:
        pn.run_llm(
            [_one_record()],
            output_dir=tmp_path,
            centers=centers,
            scales=scales,
            model="qwen2.5:7b",
            temperature=0.0,
            seed=42,
            max_records=None,
        )

    message = str(excinfo.value)
    assert "0.30.7" in message and "0.40.6" in message
    stored = json.loads((tmp_path / "run_provenance.json").read_text(encoding="utf-8"))
    assert stored["ollama_version"] == "0.30.7"


def test_run_llm_allows_engine_change_with_flag(tmp_path, monkeypatch, capsys):
    """``--allow-engine-change`` degrada el aborto a aviso y reescribe."""
    from scripts import run_p3_physionet as pn

    (tmp_path / "run_provenance.json").write_text(
        json.dumps({"ollama_version": "0.30.7"}), encoding="utf-8"
    )
    monkeypatch.setattr(pn, "check_model_available", lambda _model: True)
    monkeypatch.setattr(
        pn,
        "ollama_run_provenance",
        lambda model: {"model": model, "ollama_version": "0.40.6"},
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
        allow_engine_change=True,
    )

    out = capsys.readouterr().out
    assert "AVISO" in out and "0.30.7" in out and "0.40.6" in out
    prov = json.loads((tmp_path / "run_provenance.json").read_text(encoding="utf-8"))
    assert prov["ollama_version"] == "0.40.6"


# ---------------------------------------------------------------------------
# Análisis de una corrida con --tag (re-ejecución en el servidor)
# ---------------------------------------------------------------------------


def test_analysis_run_paths_selects_tagged_files(tmp_path):
    from scripts import analyze_p3_physionet as an

    historical = an.run_paths(tmp_path)
    assert historical["responses"].name == "physionet2017.responses.jsonl"
    assert historical["labels"].name == "physionet2017.labels.jsonl"
    assert historical["summary"].name == "physionet2017_summary.json"
    assert historical["provenance"].name == "run_provenance.json"

    tagged = an.run_paths(tmp_path, "srv")
    assert tagged["responses"].name == "physionet2017_srv.responses.jsonl"
    assert tagged["labels"].name == "physionet2017_srv.labels.jsonl"
    assert tagged["summary"].name == "physionet2017_srv_summary.json"
    assert tagged["provenance"].name == "run_provenance_srv.json"


def test_development_labels_rebuilds_split_from_cache(tmp_path):
    """Sin el fichero de la ablación, el split se reconstruye igual (CRC32)."""
    from scripts import analyze_p3_physionet as an

    labels = {
        **{f"N{i:03d}": "normal" for i in range(40)},
        **{f"A{i:03d}": "af" for i in range(40)},
        **{f"O{i:03d}": "other-rhythm" for i in range(40)},
        **{f"X{i:03d}": "noisy" for i in range(40)},
    }
    cache = {rid: {"label": label} for rid, label in labels.items()}

    dev = an.development_labels(tmp_path, cache)
    selected = select_records(labels, per_class=150, seed=42)
    expected_dev, _ = split_records(selected, dev_fraction=0.3)
    assert set(dev) == set(expected_dev)
    assert all(dev[rid] == labels[rid] for rid in dev)

    # Si existe el fichero histórico, manda él.
    (tmp_path / "physionet2017_dev_nofs.labels.jsonl").write_text(
        json.dumps({"record_id": "N000", "label": "normal"}) + "\n", encoding="utf-8"
    )
    assert an.development_labels(tmp_path, cache) == {"N000": "normal"}


def test_few_shot_ablation_reports_missing_files(tmp_path):
    from scripts import analyze_p3_physionet as an

    result = an.few_shot_ablation(tmp_path, {"A00001": "normal"})
    assert result["available"] is False
    assert "physionet2017_dev_nofs" in result["missing"]
