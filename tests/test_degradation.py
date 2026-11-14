"""Tests de las utilidades del experimento de degradación."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sources.phase3.degradation import (
    EFFECTIVE_CONDITIONS,
    build_payload,
    features_path,
    subject_reference,
)


def test_features_path_clean_vs_degraded():
    ingest = Path("/tmp/ingest")
    clean = features_path(ingest, "C0_clean", "S5")
    degraded = features_path(ingest, "C4_total_eda", "S5")
    assert clean == ingest / "features" / "S5.features.jsonl"
    assert degraded == ingest / "degraded_features" / "C4_total_eda" / "S5.features.jsonl"


def test_effective_conditions_exclude_c3():
    assert "C3_total_resp" not in EFFECTIVE_CONDITIONS
    assert len(EFFECTIVE_CONDITIONS) == 5


def test_subject_reference_uses_provided_rows_only():
    rows = {
        "w1": {"ecg_hr_bpm": 60.0, "eda_mean_us": 2.0},
        "w2": {"ecg_hr_bpm": 70.0, "eda_mean_us": 4.0},
        "w3": {"ecg_hr_bpm": 80.0, "eda_mean_us": 6.0},
    }
    centers, scales = subject_reference(rows, ["ecg_hr_bpm", "eda_mean_us"])
    assert centers["ecg_hr_bpm"] == pytest.approx(70.0)
    assert centers["eda_mean_us"] == pytest.approx(4.0)
    assert scales["ecg_hr_bpm"] > 0
    assert scales["eda_mean_us"] > 0


def test_subject_reference_handles_missing_feature():
    rows = {"w1": {"ecg_hr_bpm": 60.0}}
    centers, scales = subject_reference(rows, ["ecg_hr_bpm", "eda_mean_us"])
    assert centers["eda_mean_us"] == 0.0
    assert scales["eda_mean_us"] == 1.0


def test_build_payload_clips_and_marks_missing():
    payload = build_payload(
        window_id="WESAD:S5:w00001",
        values={"ecg_hr_bpm": 300.0, "eda_mean_us": None},
        centers={"ecg_hr_bpm": 70.0, "eda_mean_us": 3.0},
        scales={"ecg_hr_bpm": 10.0, "eda_mean_us": 1.0},
        duration_s=60.0,
    )
    assert payload["modalities"]["ecg"]["ecg_hr_bpm"]["value"] == 3.0  # recortado
    assert payload["modalities"]["eda"]["eda_mean_us"]["value"] is None
    assert payload["feature_version"] == "degradation-e12"
    assert payload["window"]["duration_s"] == 60.0


def test_build_payload_reference_is_condition_independent():
    """Con la misma referencia, degradar EDA baja su z (no se re-normaliza)."""
    centers = {"eda_mean_us": 4.0}
    scales = {"eda_mean_us": 2.0}
    clean = build_payload("w", {"eda_mean_us": 4.0}, centers, scales, 60.0)
    degraded = build_payload("w", {"eda_mean_us": 0.0}, centers, scales, 60.0)
    assert clean["modalities"]["eda"]["eda_mean_us"]["value"] == pytest.approx(0.0)
    assert degraded["modalities"]["eda"]["eda_mean_us"]["value"] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Desglose por clase y endpoint binario mapeado (analyze_degradation)
# ---------------------------------------------------------------------------
def _analysis_records() -> list[dict]:
    """Cuatro ventanas, una por clase; el modelo nunca predice amusement ni
    meditation (patrón real del 7b) y se abstiene en una."""
    return [
        {
            "window_id": "WESAD:S5:w00001",
            "window_id_opaque": "win_00000001",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "baseline",
            "confidence": 80,
            "claims": [],
            "reasoning": "Stable.",
            "prompt": "",
        },
        {
            "window_id": "WESAD:S5:w00002",
            "window_id_opaque": "win_00000002",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "stress",
            "confidence": 90,
            "claims": [],
            "reasoning": "Elevated.",
            "prompt": "",
        },
        {
            # amusement real, predicho baseline: fallo de 4 clases y acierto binario
            "window_id": "WESAD:S5:w00003",
            "window_id_opaque": "win_00000003",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_response",
            "decision": "baseline",
            "confidence": 70,
            "claims": [],
            "reasoning": "Calm.",
            "prompt": "",
        },
        {
            # meditation real, abstención
            "window_id": "WESAD:S5:w00004",
            "window_id_opaque": "win_00000004",
            "condition_id": "C0_clean",
            "subject": "S5",
            "status": "valid_abstention",
            "decision": "abstain",
            "confidence": 30,
            "claims": [],
            "reasoning": "Insufficient.",
            "prompt": "",
        },
    ]


def _analysis_labels() -> dict[str, str]:
    return {
        "WESAD:S5:w00001": "baseline",
        "WESAD:S5:w00002": "stress",
        "WESAD:S5:w00003": "amusement",
        "WESAD:S5:w00004": "meditation",
    }


def test_analyze_group_per_class_recall():
    """El recall por clase distingue 'no predice la clase' de 'la confunde'."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    per_class = metrics["per_class"]

    assert per_class["baseline"]["recall"] == 1.0
    assert per_class["stress"]["recall"] == 1.0
    assert per_class["amusement"]["recall"] == 0.0
    assert per_class["amusement"]["predictions"] == {"baseline": 1}
    assert per_class["meditation"]["decided"] == 0
    assert per_class["meditation"]["recall_decided"] == 0.0


def test_analyze_group_binary_mapped_endpoint():
    """Con el mapeo 4→2, amusement/baseline cuentan como acierto binario."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    binary = metrics["binary_mapped"]

    assert binary["positive_label"] == "stress"
    assert binary["n_decided"] == 3          # 3 decididas, 1 abstención
    assert binary["abstention_rate"] == pytest.approx(0.25)
    assert binary["coverage"] == pytest.approx(0.75)
    assert binary["accuracy"] == pytest.approx(1.0)
    assert binary["recall_stress"] == pytest.approx(1.0)


def test_analyze_group_splits_certainty_by_decision():
    """Una abstención habla de la evidencia, no del afecto: se separa (H2b)."""
    from analyze_degradation import analyze_group

    metrics = analyze_group(_analysis_records(), _analysis_labels())
    certainty = metrics["certainty"]

    assert certainty["n"] == 4
    assert certainty["decided"]["n"] == 3
    assert certainty["abstain"]["n"] == 1
    assert certainty["decided"]["n"] + certainty["abstain"]["n"] == certainty["n"]


def test_certainty_stats_empty_is_nan():
    from analyze_degradation import _certainty_stats

    stats = _certainty_stats([])
    assert stats["n"] == 0
    assert stats["certainty_index_mean"] != stats["certainty_index_mean"]  # NaN


# ---------------------------------------------------------------------------
# Tarea binaria (stress / non-stress) en el barrido y en el análisis
# ---------------------------------------------------------------------------
def test_output_directory_separates_binary_from_affective(tmp_path, monkeypatch):
    """La tarea va en el nombre del directorio: no se mezclan corpus."""
    import run_degradation_sweep as sweep

    monkeypatch.setattr(sweep, "OUTPUT_DIR", tmp_path)

    affective = sweep.output_directory("numeric", "affective", "qwen3:14b")
    binary = sweep.output_directory("numeric", "stress_binary", "qwen3:14b")

    assert affective == tmp_path.parent / "degradation" / "qwen3_14b"
    assert binary == tmp_path.parent / "degradation_binary" / "qwen3_14b"
    assert affective != binary


def _binary_records() -> list[dict]:
    base = {
        "condition_id": "C0_clean",
        "subject": "S5",
        "claims": [],
        "reasoning": "Steady.",
        "prompt": "",
    }
    return [
        {**base, "window_id": "WESAD:S5:w1", "window_id_opaque": "o1",
         "status": "valid_response", "decision": "stress", "confidence": 90},
        {**base, "window_id": "WESAD:S5:w2", "window_id_opaque": "o2",
         "status": "valid_response", "decision": "non-stress", "confidence": 80},
        {**base, "window_id": "WESAD:S5:w3", "window_id_opaque": "o3",
         "status": "valid_response", "decision": "stress", "confidence": 70},
        {**base, "window_id": "WESAD:S5:w4", "window_id_opaque": "o4",
         "status": "valid_abstention", "decision": "abstain", "confidence": 30},
    ]


def _binary_labels() -> dict[str, str]:
    return {
        "WESAD:S5:w1": "stress",
        "WESAD:S5:w2": "baseline",   # non-stress
        "WESAD:S5:w3": "baseline",   # falso positivo
        "WESAD:S5:w4": "meditation",  # non-stress, abstención
    }


def test_analyze_group_binary_uses_binary_metrics():
    """Con prompt binario no hay mapeo 4→2: se evalúa la decisión tal cual."""
    from analyze_degradation import analyze_group_binary

    metrics = analyze_group_binary(_binary_records(), _binary_labels())

    assert metrics["n"] == 4
    assert metrics["n_decided"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision_stress"] == pytest.approx(0.5)
    assert metrics["recall_stress"] == pytest.approx(1.0)
    assert metrics["per_class"] == {}


def test_analyze_group_binary_empty():
    from analyze_degradation import analyze_group_binary

    assert analyze_group_binary([], {}) == {"n": 0}


def test_print_per_class_table_silent_without_per_class(capsys):
    """Con prompt binario no hay desglose por clase: no se imprime la tabla."""
    from analyze_degradation import print_per_class_table

    report = {"conditions": {"C0_clean": {"n": 567, "per_class": {}}}}
    print_per_class_table(report, ["C0_clean"])
    out = capsys.readouterr().out
    assert out == ""   # nada: sin per_class no hay tabla que imprimir


def test_existing_pairs_retries_timeouts(tmp_path):
    """Un timeout no es una respuesta: se reintenta al relanzar."""
    import json

    import run_degradation_sweep as sweep

    path = tmp_path / "S5.responses.jsonl"
    rows = [
        {"window_id": "w1", "condition_id": "C0_clean", "status": "valid_response"},
        {"window_id": "w2", "condition_id": "C0_clean", "status": "timeout"},
        {"window_id": "w3", "condition_id": "C0_clean", "status": "valid_abstention"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    existing, n_retry = sweep.existing_pairs(path)

    assert existing == {("w1", "C0_clean"), ("w3", "C0_clean")}
    assert n_retry == 1


def test_ensure_model_available_aborts_when_missing(monkeypatch):
    import run_degradation_sweep as sweep

    monkeypatch.setattr(sweep, "check_model_available", lambda model: False)
    with pytest.raises(SystemExit) as exc:
        sweep.ensure_model_available("qwen2.5:7b")
    assert "ollama pull qwen2.5:7b" in str(exc.value)


def test_ensure_model_available_passes_when_present(monkeypatch):
    import run_degradation_sweep as sweep

    monkeypatch.setattr(sweep, "check_model_available", lambda model: True)
    sweep.ensure_model_available("qwen2.5:7b")      # no lanza
