"""Tests del orquestador ``process_ingest``."""
from __future__ import annotations

import json

import pytest

from sources.common.common import processControl
from sources.ingestion import loader, pipeline


def test_full_run_executes_all_implemented_steps(synthetic_dataset):
    pipeline.process_ingest()

    state = pipeline.load_steps()
    assert state["steps"]["discover"]["status"] == "done"
    assert state["steps"]["ingest"]["status"] == "done"
    assert state["steps"]["qc"]["status"] == "done"
    assert state["steps"]["window"]["status"] == "done"
    assert state["steps"]["features"]["status"] == "done"
    assert state["steps"]["represent"]["status"] == "done"
    assert state["steps"]["package"]["status"] == "done"
    assert state["steps"]["ingest"]["results"] == {"ok": 2}
    assert state["steps"]["window"]["n_windows"] == 4
    assert state["steps"]["features"]["n_feature_values"] > 0
    assert state["steps"]["represent"]["n_windows"] == {"numeric": 4, "text": 4}
    assert state["steps"]["package"]["n_windows"] == 4

    for subject in ("S2", "S3"):
        npz_path, meta_path = loader.canonical_paths(subject)
        assert npz_path.is_file()
        assert meta_path.is_file()
        assert pipeline.qc.qc_path(subject).is_file()
        assert pipeline.window.windows_path(subject).is_file()
        assert pipeline.features.features_path(subject).is_file()
        assert pipeline.represent.representation_path("numeric", subject).is_file()
        assert pipeline.represent.representation_path("text", subject).is_file()
    assert pipeline.discover.manifest_path().is_file()
    assert pipeline.window.global_index_path().is_file()
    assert pipeline.features.global_index_path().is_file()
    assert pipeline.represent.index_path("numeric").is_file()
    assert pipeline.represent.index_path("text").is_file()
    assert pipeline.package.manifest_path().is_file()
    assert pipeline.package.index_path().is_file()
    assert pipeline.package.labels_path().is_file()
    assert pipeline.package.checksums_path().is_file()


def test_step_discover_stops_before_ingest(synthetic_dataset):
    processControl.args.step = "discover"
    pipeline.process_ingest()

    assert pipeline.discover.manifest_path().is_file()
    assert not loader.canonical_paths("S2")[0].exists()
    assert "ingest" not in pipeline.load_steps()["steps"]
    assert "qc" not in pipeline.load_steps()["steps"]


def test_step_window_runs_through_qc_and_window(synthetic_dataset):
    processControl.args.step = "window"
    pipeline.process_ingest()

    state = pipeline.load_steps()
    assert state["steps"]["ingest"]["status"] == "done"
    assert state["steps"]["qc"]["status"] == "done"
    assert state["steps"]["window"]["status"] == "done"
    assert state["steps"]["window"]["n_windows"] == 4


def test_all_steps_are_implemented():
    assert set(pipeline.STEPS) == set(pipeline.IMPLEMENTED)
    assert pipeline.PENDING == ()


def test_unknown_step_fails(synthetic_dataset):
    from sources.common.validation import ValidationError

    processControl.args.step = "nope"
    with pytest.raises(ValidationError, match="paso desconocido"):
        pipeline.process_ingest()


def test_dry_run_does_not_touch_output(synthetic_dataset, capsys):
    processControl.args.dry_run = True
    pipeline.process_ingest()

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["selected_subjects"] == ["S2", "S3"]
    assert payload["config"]["windows"]["length_s"] == 60.0
    assert not pipeline.discover.manifest_path().exists()


def test_strip_embeddings_quita_solo_esa_representacion():
    cfg = {"representations": {"enabled": ["numeric", "text", "embeddings"]}}
    assert pipeline._strip_embeddings(cfg) is True
    assert cfg["representations"]["enabled"] == ["numeric", "text"]


def test_strip_embeddings_sin_embeddings_activos():
    cfg = {"representations": {"enabled": ["numeric", "text"]}}
    assert pipeline._strip_embeddings(cfg) is False
    assert cfg["representations"]["enabled"] == ["numeric", "text"]


def test_no_embeddings_no_calcula_vectores(synthetic_dataset, monkeypatch):
    """``--noEmbeddings`` quita los vectores densos de la configuración de INGEST."""
    args = dict(
        processControl.args
        if isinstance(processControl.args, dict)
        else vars(processControl.args)
    )
    args["noEmbeddings"] = True
    args["step"] = "represent"  # detiene el pipeline antes de empaquetar
    monkeypatch.setattr(processControl, "args", args)

    captured: list[list[str]] = []

    def _fake_run_represent(cfg, subjects, force):
        captured.append(list(cfg["representations"]["enabled"]))
        return {"subjects": subjects}

    monkeypatch.setattr(pipeline, "_run_represent", _fake_run_represent)
    pipeline.process_ingest()

    base = pipeline.get_ingestion_config()["representations"]["enabled"]
    assert captured, "el paso represent debe ejecutarse"
    assert "embeddings" not in captured[0]
    assert captured[0] == [kind for kind in base if kind != "embeddings"]


def test_sin_flag_los_embeddings_siguen_activos(synthetic_dataset, monkeypatch):
    args = dict(
        processControl.args
        if isinstance(processControl.args, dict)
        else vars(processControl.args)
    )
    args.pop("noEmbeddings", None)
    args["step"] = "represent"
    monkeypatch.setattr(processControl, "args", args)

    captured: list[list[str]] = []

    def _fake_run_represent(cfg, subjects, force):
        captured.append(list(cfg["representations"]["enabled"]))
        return {"subjects": subjects}

    monkeypatch.setattr(pipeline, "_run_represent", _fake_run_represent)
    pipeline.process_ingest()

    base = pipeline.get_ingestion_config()["representations"]["enabled"]
    assert captured[0] == base


# --------------------------------------------------------------------------
# Arranque limpio (--fresh): borrar derivados sin tocar results/input
# --------------------------------------------------------------------------


def _fake_corpus() -> Path:
    corpus = pipeline.corpus_dir()
    (corpus / "canonical").mkdir(parents=True, exist_ok=True)
    (corpus / "canonical" / "S2.npz").write_bytes(b"x" * (2 * 1024 * 1024))
    (corpus / "steps.json").write_text("{}", encoding="utf-8")
    return corpus


def test_fresh_sin_yes_no_borra(workspace):
    corpus = _fake_corpus()
    with pytest.raises(SystemExit) as excinfo:
        pipeline.reset_derived(confirm=False, dry_run=False)
    assert excinfo.value.code == 2
    assert (corpus / "canonical" / "S2.npz").is_file()


def test_fresh_dry_run_lista_sin_borrar(workspace):
    corpus = _fake_corpus()
    report = pipeline.reset_derived(confirm=False, dry_run=True)
    assert report["dry_run"] is True
    assert report["total_mb"] > 0
    assert {entry["name"] for entry in report["entries"]} == {"canonical", "steps.json"}
    assert (corpus / "canonical" / "S2.npz").is_file()


def test_fresh_borra_derivados_y_conserva_la_entrada(workspace):
    corpus = _fake_corpus()
    dataset = workspace["dataset"]
    raw = dataset / "S2"
    raw.mkdir()
    (raw / "S2.pkl").write_bytes(b"crudo")

    report = pipeline.reset_derived(confirm=True, dry_run=False)

    assert report["exists"] is True
    assert corpus.is_dir()
    assert list(corpus.iterdir()) == []
    # Los datos crudos quedan intactos: empezar de cero no toca results/input.
    assert (raw / "S2.pkl").read_bytes() == b"crudo"


def test_fresh_avisa_de_artefactos_de_otras_fases(workspace):
    _fake_corpus()
    (workspace["output"] / "WESAD" / "PHASE2").mkdir(parents=True)
    report = pipeline.reset_derived(confirm=False, dry_run=True)
    assert any("PHASE2" in warning for warning in report["warnings"])


def test_fresh_recalcula_el_corpus(synthetic_dataset, monkeypatch):
    """Tras --fresh el pipeline vuelve a producir el corpus desde el .pkl."""
    args = dict(
        processControl.args
        if isinstance(processControl.args, dict)
        else vars(processControl.args)
    )
    args.update({"fresh": True, "yes": True})
    monkeypatch.setattr(processControl, "args", args)

    pipeline.process_ingest()

    state = pipeline.load_steps()
    assert state["steps"]["ingest"]["status"] == "done"
    assert state["steps"]["package"]["n_windows"] == 4
    assert pipeline.loader.canonical_paths("S2")[0].is_file()
