"""Tests de las repeticiones de prompt en DREAMT y PhysioNet 2017.

Cubren las tres piezas que pueden fallar en silencio:

1. **Aislamiento del corpus**: con ``R > 1`` la salida va a ``*_rep{R}`` y el
   corpus de una pasada no se toca (sostiene las cifras publicadas).
2. **Reanudación por repetición**: la clave es ``(ventana/registro, condición,
   repetición)``; si no, una segunda ejecución no haría nada.
3. **Análisis**: la varianza de muestreo se mide con la estadística compartida
   y el vocabulario de etiquetas de cada tarea.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_repetitions_external as reps  # noqa: E402
import run_p3_physionet as pn  # noqa: E402
import run_p4_dreamt as dr  # noqa: E402
from sources.ingestion.represent import _opaque_window_id  # noqa: E402

_VALID_RHYTHM_OUTPUT = (
    '{"decision": "normal", "confidence": 80, '
    '"claims": [{"metric": "ecg_hr_bpm", "channel": "ecg", '
    '"direction": "stable", "magnitude": "low"}], '
    '"reasoning": "Heart rate is near the population median."}'
)


# ---------------------------------------------------------------------------
# Nombres de salida (aislamiento del corpus)
# ---------------------------------------------------------------------------
def test_physionet_file_tag_adds_repetition_suffix_only_when_needed():
    """``R = 1`` conserva el nombre histórico; ``R > 1`` va a ``_rep{R}``."""
    assert pn.output_file_tag("", "test", False, 1) == ""
    assert pn.output_file_tag("srv", "test", False, 1) == "srv"
    assert pn.output_file_tag("", "test", False, 3) == "rep3"
    assert pn.output_file_tag("srv", "test", False, 3) == "srv_rep3"
    assert pn.output_file_tag("", "dev", True, 3) == "dev_fs_rep3"


def test_dreamt_repetition_paths_are_isolated():
    assert dr.responses_directory(1) == "responses"
    assert dr.responses_directory(3) == "responses_rep3"
    assert dr.provenance_filename(1) == "run_provenance.json"
    assert dr.provenance_filename(3) == "run_provenance_rep3.json"
    assert dr.repetition_seeds(42, 3) == [42, 43, 44]


def test_repetition_seeds_rejects_zero():
    with pytest.raises(ValueError):
        dr.repetition_seeds(42, 0)


# ---------------------------------------------------------------------------
# PhysioNet: generación con repeticiones
# ---------------------------------------------------------------------------
def _one_record(name: str = "A00001"):
    values = {feature: 0.0 for feature in pn.CANONICAL_FEATURES}
    return pn.RhythmRecord(
        record_id=name, label="normal", fs=300.0, duration_s=30.0, values=values
    )


def _monkeypatch_llm(monkeypatch, calls: list[int]):
    monkeypatch.setattr(pn, "check_model_available", lambda _model: True)
    monkeypatch.setattr(
        pn,
        "ollama_run_provenance",
        lambda model: {"model": model, "ollama_version": "0.40.6"},
    )

    def fake_call(client, prompt, config, seed=None):
        calls.append(int(seed))
        return _VALID_RHYTHM_OUTPUT, {"model": "qwen2.5:7b"}

    monkeypatch.setattr(pn, "_call_ollama_chat", fake_call)


def test_physionet_writes_one_record_per_seed(tmp_path, monkeypatch):
    calls: list[int] = []
    _monkeypatch_llm(monkeypatch, calls)
    centers = {f: 0.0 for f in pn.CANONICAL_FEATURES}
    scales = {f: 1.0 for f in pn.CANONICAL_FEATURES}

    responses = pn.run_llm(
        [_one_record()],
        output_dir=tmp_path,
        centers=centers,
        scales=scales,
        model="qwen2.5:7b",
        temperature=0.0,
        seed=42,
        max_records=None,
        tag=pn.output_file_tag("", "test", False, 3),
        repetitions=3,
    )

    assert calls == [42, 43, 44]
    assert len(responses) == 3
    assert [row["repetition"] for row in responses] == [0, 1, 2]
    assert [row["seed"] for row in responses] == [42, 43, 44]
    assert all(row["model"] == "qwen2.5:7b" for row in responses)
    # El corpus de una pasada no se crea.
    assert (tmp_path / "physionet2017_rep3.responses.jsonl").is_file()
    assert not (tmp_path / "physionet2017.responses.jsonl").exists()
    provenance = json.loads(
        (tmp_path / "run_provenance_rep3.json").read_text(encoding="utf-8")
    )
    assert provenance["repetitions"] == 3
    assert provenance["seeds"] == [42, 43, 44]


def test_physionet_resumes_by_record_and_repetition(tmp_path, monkeypatch):
    """Con dos repeticiones hechas, solo se llama a la tercera."""
    path = tmp_path / "physionet2017_rep3.responses.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rep in (0, 1):
            fh.write(
                json.dumps(
                    {
                        "record_id": "A00001",
                        "repetition": rep,
                        "status": "valid_response",
                        "decision": "normal",
                        "confidence": 80,
                    }
                )
                + "\n"
            )
    calls: list[int] = []
    _monkeypatch_llm(monkeypatch, calls)
    centers = {f: 0.0 for f in pn.CANONICAL_FEATURES}
    scales = {f: 1.0 for f in pn.CANONICAL_FEATURES}

    pn.run_llm(
        [_one_record()],
        output_dir=tmp_path,
        centers=centers,
        scales=scales,
        model="qwen2.5:7b",
        temperature=0.0,
        seed=42,
        max_records=None,
        tag="rep3",
        repetitions=3,
    )

    assert calls == [44]


# ---------------------------------------------------------------------------
# Analizador: PhysioNet
# ---------------------------------------------------------------------------
def _physionet_corpus(directory: Path, repetitions: int = 2) -> None:
    labels = {"A00001": "normal", "A00002": "af", "A00003": "noisy", "A00004": "normal"}
    with (directory / "physionet2017_rep2.labels.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for record_id, label in labels.items():
            fh.write(
                json.dumps(
                    {"record_id": record_id, "label": label, "duration_s": 30.0}
                )
                + "\n"
            )
    # A00001 y A00002 estables; A00003 siempre `normal` (una `noisy` mal);
    # A00004 cambia entre normal y af en la segunda repetición.
    decisions = {
        "A00001": ["normal", "normal"],
        "A00002": ["af", "af"],
        "A00003": ["normal", "normal"],
        "A00004": ["normal", "af"],
    }
    with (directory / "physionet2017_rep2.responses.jsonl").open(
        "w", encoding="utf-8"
    ) as fh:
        for record_id, rows in decisions.items():
            for rep, decision in enumerate(rows):
                fh.write(
                    json.dumps(
                        {
                            "record_id": record_id,
                            "repetition": rep,
                            "status": "valid_response",
                            "decision": decision,
                            "confidence": 80,
                        }
                    )
                    + "\n"
                )
    (directory / "run_provenance_rep2.json").write_text(
        json.dumps(
            {
                "model": "qwen2.5:7b",
                "ollama_version": "0.40.6",
                "model_digest": "abc",
            }
        ),
        encoding="utf-8",
    )


def test_physionet_analyzer_reports_stability_and_engine(tmp_path, capsys):
    _physionet_corpus(tmp_path)
    report = reps.analyze_physionet(tmp_path, "qwen2.5:7b", None, 2, 20, 12345)

    assert report["engine"]["ollama_version"] == "0.40.6"
    assert report["seeds"] == [12345, 12346]
    group = report["groups"]["numeric|C0_eval"]
    assert group["n_windows"] == 4
    assert group["n_repetitions"] == 2
    # Tres de cuatro ventanas son deterministas.
    assert group["stability"]["frac_deterministic"] == pytest.approx(0.75)
    assert group["stability"]["mean_modal_share"] == pytest.approx(0.875)
    assert group["convergence"][0]["k"] == 1
    assert group["convergence"][-1]["k"] == 2


def test_physionet_analyzer_rejects_a_single_pass(tmp_path):
    with pytest.raises(SystemExit):
        reps.main(
            [
                "--experiment",
                "physionet",
                "--dir",
                str(tmp_path),
                "--repetitions",
                "1",
            ]
        )


# ---------------------------------------------------------------------------
# Analizador: DREAMT
# ---------------------------------------------------------------------------
def test_dreamt_analyzer_groups_by_representation_and_condition(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    windows = {
        "s1:w1": "wake",
        "s1:w2": "n1",
        "s1:w3": "n2",
        "s1:w4": "rem",
    }
    with (labels_dir / "S001.labels.jsonl").open("w", encoding="utf-8") as fh:
        for window_id, klass in windows.items():
            fh.write(
                json.dumps({"window_id": window_id, "klass": klass}) + "\n"
            )
    responses = tmp_path / "responses_rep2" / "qwen2_5_7b"
    responses.mkdir(parents=True)
    with (responses / "S001.numeric.jsonl").open("w", encoding="utf-8") as fh:
        for window_id, klass in windows.items():
            for condition in ("C0_clean", "C4_absent_core"):
                for rep in (0, 1):
                    fh.write(
                        json.dumps(
                            {
                                "window_id": window_id,
                                "window_id_opaque": _opaque_window_id(window_id),
                                "condition_id": condition,
                                "representation": "numeric",
                                "repetition": rep,
                                "status": "valid_response",
                                "decision": klass,
                                "confidence": 80,
                            }
                        )
                        + "\n"
                    )
    (tmp_path / "run_provenance_rep2.json").write_text(
        json.dumps({"model": "qwen2.5:7b", "ollama_version": "0.40.6"}),
        encoding="utf-8",
    )

    report = reps.analyze_dreamt(tmp_path, "qwen2.5:7b", 2, 20, 12345)

    assert set(report["groups"]) == {"numeric|C0_clean", "numeric|C4_absent_core"}
    for group in report["groups"].values():
        assert group["n_windows"] == 4
        assert group["stability"]["mean_modal_share"] == pytest.approx(1.0)
        assert group["single_run"]["accuracy"]["mean"] == pytest.approx(1.0)
        assert group["implied_k"]["implied_k"]["2pp"] == 1
    assert report["n_classes_labelled"] == 4


# ---------------------------------------------------------------------------
# Presentación: un 0.0 legítimo no es «indefinido»
# ---------------------------------------------------------------------------
def test_format_metric_keeps_zero_and_marks_undefined():
    from sources.phase3.repetition_stats import format_metric

    assert format_metric(0.0, 7) == "  0.000"
    assert format_metric(0.1154) == "0.115"
    assert format_metric(None, 5) == "    —"
    assert format_metric(float("nan"), 5) == "    —"


def test_print_report_shows_zero_coverage_instead_of_nan(capsys):
    """Un brazo que se abstiene al 100 % tiene cobertura 0, no indefinida.

    Regresión del 2026-11-28: el idioma ``valor or nan`` imprimía ``nan`` para
    ``cov1`` cuando valía exactamente 0 (texto `C1` de DREAMT), ocultando el
    número que sostiene la conclusión.
    """
    report = {
        "experiment": "dreamt",
        "model": "qwen2.5:7b",
        "repetitions": 3,
        "seeds": [42, 43, 44],
        "dir": "/tmp/x",
        "engine": {"ollama_version": "0.40.6", "model_digest": "abc"},
        "n_classes_labelled": 4,
        "draws": 10,
        "groups": {
            "text|C1_absent_eda": {
                "n_windows": 480,
                "n_invalid": 0,
                "stability": {"mean_modal_share": 1.0},
                "single_run": {
                    "accuracy": {"mean": 0.0, "sd": 0.0},
                    "coverage": {"mean": 0.0},
                },
                "implied_k": {"implied_k": {"2pp": 1}},
            }
        },
    }

    reps.print_report(report)

    row = [line for line in capsys.readouterr().out.splitlines() if line.startswith("text|")][0]
    cells = row.split()
    assert cells[0] == "text|C1_absent_eda"
    assert cells[1] == "480"
    assert cells[2] == "1.000"  # acuerdo modal
    assert cells[3] == "—"      # accuracy sin registros decididos
    assert cells[5] == "0.000"  # cobertura: 0, no nan


# ---------------------------------------------------------------------------
# Procedencia por modelo (el corpus es por modelo; la procedencia también)
# ---------------------------------------------------------------------------
def test_dreamt_provenance_name_includes_the_model():
    """Dos modelos en la misma cohorte no comparten fichero de procedencia.

    Regresión del 2026-11-28 (job 16719): el guard abortaba ``medgemma:4b``
    porque ``run_provenance_rep3.json`` decía ``qwen2.5:7b``. La procedencia
    compartida impedía correr un segundo modelo en la cohorte y, si hubiera
    pasado, habría borrado el registro del primero.
    """
    assert dr.provenance_filename(3, "qwen2.5:7b") == "run_provenance_rep3_qwen2_5_7b.json"
    assert dr.provenance_filename(3, "medgemma:4b") == "run_provenance_rep3_medgemma_4b.json"
    assert dr.provenance_filename(1, "medgemma:4b") == "run_provenance_medgemma_4b.json"
    # Sin modelo: nombre histórico, para leer corpus anteriores.
    assert dr.provenance_filename(3) == "run_provenance_rep3.json"


def test_guard_corpus_engine_ignores_another_models_provenance(tmp_path):
    from sources.common.provenance import guard_corpus_engine

    legacy = tmp_path / "run_provenance_rep3.json"
    legacy.write_text(
        json.dumps({"model": "qwen2.5:7b", "model_digest": "aaa", "ollama_version": "0.40.6"}),
        encoding="utf-8",
    )
    current = {
        "model": "medgemma:4b",
        "model_digest": "bbb",
        "ollama_version": "0.40.6",
    }

    conflicts = guard_corpus_engine(tmp_path / "run_provenance_rep3_medgemma_4b.json", legacy, current)

    assert conflicts == []


def test_guard_corpus_engine_aborts_on_a_change_within_the_same_model(tmp_path):
    from sources.common.provenance import guard_corpus_engine

    legacy = tmp_path / "run_provenance_rep3.json"
    legacy.write_text(
        json.dumps({"model": "qwen2.5:7b", "model_digest": "aaa", "ollama_version": "0.30.7"}),
        encoding="utf-8",
    )
    current = {
        "model": "qwen2.5:7b",
        "model_digest": "bbb",
        "ollama_version": "0.40.6",
    }

    with pytest.raises(SystemExit):
        guard_corpus_engine(tmp_path / "run_provenance_rep3_qwen2_5_7b.json", legacy, current)


def test_dreamt_provenance_path_prefers_the_models_own_file(tmp_path):
    (tmp_path / "run_provenance_rep3.json").write_text(
        json.dumps({"model": "qwen2.5:7b", "ollama_version": "0.40.6"}), encoding="utf-8"
    )
    (tmp_path / "run_provenance_rep3_medgemma_4b.json").write_text(
        json.dumps({"model": "medgemma:4b", "ollama_version": "0.40.6"}), encoding="utf-8"
    )

    medgemma = reps.dreamt_provenance_path(tmp_path, 3, "medgemma:4b")
    qwen = reps.dreamt_provenance_path(tmp_path, 3, "qwen2.5:7b")

    assert medgemma.name == "run_provenance_rep3_medgemma_4b.json"
    # ``qwen`` no tiene fichero propio: se lee la procedencia antigua, que sí es suya.
    assert qwen.name == "run_provenance_rep3.json"


def test_dreamt_provenance_path_ignores_a_foreign_legacy_file(tmp_path):
    """Sin fichero propio, la procedencia antigua solo vale si es del mismo modelo."""
    (tmp_path / "run_provenance_rep3.json").write_text(
        json.dumps({"model": "qwen2.5:7b", "ollama_version": "0.40.6"}), encoding="utf-8"
    )

    path = reps.dreamt_provenance_path(tmp_path, 3, "medgemma:4b")

    assert path.name == "run_provenance_rep3_medgemma_4b.json"
    assert not path.exists()
