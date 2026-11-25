"""Pruebas del comparador de modelos del piloto DREAMT.

Lo que se prueba aquí es que la comparación no pueda mentir por agregación: las
métricas se calculan por sujeto con el evaluador del piloto, la abstención entra
en ``accuracy_incl_abstain`` y un modelo sin respuestas se declara como tal en
vez de desaparecer de la tabla.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sources.ingestion.represent import _opaque_window_id
from scripts import compare_dreamt_models as cmp


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------


def test_aggregate_ignores_nan_and_none():
    block = cmp.aggregate([0.5, None, float("nan"), "0.75", float("inf")])
    assert block == {"n": 2, "mean": pytest.approx(0.625), "min": 0.5, "max": 0.75}


def test_aggregate_empty_is_explicit():
    assert cmp.aggregate([]) == {"n": 0, "mean": None, "min": None, "max": None}


def test_flatten_metrics_reads_nested_blocks():
    metrics = {
        "n": 10,
        "n_decided": 8,
        "n_abstain": 2,
        "accuracy": 0.5,
        "accuracy_incl_abstain": 0.4,
        "balanced_accuracy": 0.45,
        "coverage": 0.8,
        "claims": {"supported_rate": 0.7, "contradicted_rate": 0.3},
        "coherence": {"coherence_rate": 0.6},
        "certainty": {"median_index": -0.2},
        "recall_by_class": {"wake": 0.9},
        "confidence_auroc": 0.55,
        "overconfidence": 0.1,
    }
    flat = cmp.flatten_metrics(metrics)
    assert flat["abstain_rate"] == pytest.approx(0.2)
    assert flat["supported_rate"] == 0.7
    assert flat["coherence_rate"] == 0.6
    assert flat["certainty_index"] == -0.2
    assert flat["recall_wake"] == 0.9
    assert flat["recall_n2"] is None


def _record(window: str, condition: str, decision: str, klass: str, status="valid_response"):
    return {
        "window_id": window,
        "window_id_opaque": _opaque_window_id(window),
        "condition_id": condition,
        "status": status,
        "decision": decision,
        "confidence": 90,
        "claims": [],
        "prompt": "",
        "reasoning": "",
        "_klass": klass,
    }


def test_metrics_are_computed_per_subject_and_aggregated():
    """Dos sujetos con aciertos distintos deben dar media y rango, no un total."""
    labels, records_a, records_b = {}, {}, {}
    for subject, hits in (("S001", 2), ("S002", 0)):
        labels_rows = [
            _record(f"{subject}:e{i:04d}", "C0_clean", "n2", "n2") for i in range(2)
        ]
        for row in labels_rows:
            labels[row["window_id_opaque"]] = row["_klass"]
        rows = []
        for index, row in enumerate(labels_rows):
            decision = "n2" if index < hits else "wake"
            rows.append(
                _record(row["window_id"], "C0_clean", decision, row["_klass"])
            )
        records_a[subject] = rows
        records_b[subject] = [
            _record(row["window_id"], "C0_clean", "wake", row["_klass"]) for row in rows
        ]

    aggregated_a = cmp.aggregate_by_condition(cmp.compute_metrics(records_a, labels))
    aggregated_b = cmp.aggregate_by_condition(cmp.compute_metrics(records_b, labels))
    assert aggregated_a["C0_clean"]["accuracy"]["mean"] == pytest.approx(0.5)
    assert aggregated_a["C0_clean"]["accuracy"]["min"] == 0.0
    assert aggregated_a["C0_clean"]["accuracy"]["max"] == 1.0
    assert aggregated_b["C0_clean"]["accuracy"]["mean"] == 0.0
    assert aggregated_a["C0_clean"]["coverage"]["mean"] == pytest.approx(1.0)


def test_abstention_costs_accuracy_in_the_reported_metric():
    """Abstenerse no puede salir gratis: ``acc*`` cuenta la abstención como error."""
    labels = {}
    rows = []
    for index in range(4):
        record = _record(f"S003:e{index:04d}", "C0_clean", "n2", "n2")
        labels[record["window_id_opaque"]] = "n2"
        record["decision"] = "n2" if index == 0 else "abstain"
        record["status"] = "valid_response" if index == 0 else "valid_abstention"
        rows.append(record)
    aggregated = cmp.aggregate_by_condition(
        cmp.compute_metrics({"S003": rows}, labels)
    )["C0_clean"]
    assert aggregated["accuracy"]["mean"] == pytest.approx(1.0)
    assert aggregated["accuracy_incl_abstain"]["mean"] == pytest.approx(0.25)
    assert aggregated["coverage"]["mean"] == pytest.approx(0.25)
    assert aggregated["abstain_rate"]["mean"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Lectura de disco y deltas
# ---------------------------------------------------------------------------


def _write_pilot(tmp_path: Path, model: str, hits: int) -> None:
    slug = cmp.model_slug(model)
    responses = tmp_path / "responses" / slug
    responses.mkdir(parents=True, exist_ok=True)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_rows = []
    with (responses / "S036.numeric.jsonl").open("w", encoding="utf-8") as handle:
        for index, klass in enumerate(("wake", "n1", "n2", "rem")):
            window = f"S036:e{index:04d}"
            label_rows.append({"window_id": window, "klass": klass})
            decision = klass if index < hits else ("rem" if klass != "rem" else "wake")
            handle.write(
                json.dumps(
                    {
                        "window_id": window,
                        "window_id_opaque": _opaque_window_id(window),
                        "condition_id": "C0_clean",
                        "status": "valid_response",
                        "decision": decision,
                        "confidence": 90,
                        "claims": [],
                        "prompt": "",
                        "reasoning": "",
                    }
                )
                + "\n"
            )
    with (labels_dir / "S036.labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in label_rows:
            handle.write(json.dumps(row) + "\n")


def test_comparison_reports_deltas_and_flags_missing_model(tmp_path):
    _write_pilot(tmp_path, "qwen2.5:7b", hits=3)
    _write_pilot(tmp_path, "medgemma:4b", hits=1)
    report = cmp.compare_models(
        ["qwen2.5:7b", "medgemma:4b"],
        tmp_path,
        representations=("numeric",),
        subjects=("S036",),
    )
    numeric = report["by_representation"]["numeric"]
    assert numeric["qwen2.5:7b"]["conditions"]["C0_clean"]["accuracy"]["mean"] == pytest.approx(
        0.75
    )
    assert numeric["medgemma:4b"]["conditions"]["C0_clean"]["accuracy"]["mean"] == pytest.approx(
        0.25
    )
    delta = report["deltas"]["numeric"]["conditions"]["C0_clean"]["accuracy"]
    assert delta < 0  # el segundo modelo acierta menos
    assert report["deltas"]["numeric"]["other"] == "medgemma:4b"

    missing = cmp.compare_models(
        ["qwen2.5:7b", "llama3.1:8b"],
        tmp_path,
        representations=("numeric",),
        subjects=("S036",),
    )
    assert missing["by_representation"]["numeric"]["llama3.1:8b"]["status"] == "sin respuestas"


def test_recovery_rescues_decisions_of_malformed_claims(tmp_path):
    """Una respuesta inválida por vocabulario no puede desaparecer de la cobertura."""
    slug = cmp.model_slug("medgemma:4b")
    responses = tmp_path / "responses" / slug
    responses.mkdir(parents=True)
    (tmp_path / "labels").mkdir()
    window = "S036:e0001"
    raw = json.dumps(
        {
            "decision": "wake",
            "confidence": 95,
            "claims": [{"metric": "acc_std_g", "channel": "acc", "direction": "high", "magnitude": "high"}],
            "reasoning": "movement is high",
        }
    )
    with (responses / "S036.numeric.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "window_id": window,
                    "window_id_opaque": _opaque_window_id(window),
                    "condition_id": "C0_clean",
                    "status": "invalid_output",
                    "decision": None,
                    "confidence": None,
                    "claims": [],
                    "prompt": "- acc_std_g: 0.4 dimensionless",
                    "reasoning": "",
                    "raw_output": raw,
                }
            )
            + "\n"
        )
    with (tmp_path / "labels" / "S036.labels.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"window_id": window, "klass": "wake"}) + "\n")

    without = cmp.compare_models(
        ["qwen2.5:7b", "medgemma:4b"], tmp_path, representations=("numeric",), subjects=("S036",)
    )
    assert without["by_representation"]["numeric"]["medgemma:4b"]["conditions"]["C0_clean"][
        "coverage"
    ]["mean"] == pytest.approx(0.0)
    with_recovery = cmp.compare_models(
        ["qwen2.5:7b", "medgemma:4b"],
        tmp_path,
        representations=("numeric",),
        subjects=("S036",),
        recovery=True,
    )
    assert with_recovery["by_representation"]["numeric"]["medgemma:4b"]["conditions"][
        "C0_clean"
    ]["coverage"]["mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_requires_two_models(tmp_path):
    with pytest.raises(SystemExit):
        cmp.main(["--models", "qwen2.5:7b", "--dir", str(tmp_path)])


def test_cli_rejects_subjects_outside_the_pilot(tmp_path):
    with pytest.raises(SystemExit):
        cmp.main(
            [
                "--models",
                "qwen2.5:7b,medgemma:4b",
                "--dir",
                str(tmp_path),
                "--subjects",
                "S002",
            ]
        )


# ---------------------------------------------------------------------------
# Cohortes e intervalos por sujeto (escala)
# ---------------------------------------------------------------------------


def test_bootstrap_ci_needs_enough_subjects():
    """Con pocos sujetos no se publica intervalo (sería más estrecho que la realidad)."""
    assert cmp.bootstrap_subject_ci([0.1, 0.2, 0.3, 0.4]) is None
    assert cmp.bootstrap_subject_ci([0.0] * 3) is None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    first = cmp.bootstrap_subject_ci(values, n_boot=500, seed=7)
    second = cmp.bootstrap_subject_ci(values, n_boot=500, seed=7)
    assert first == second
    assert first["n_subjects"] == 10
    assert first["lo"] <= 0.45 <= first["hi"]
    assert 0.0 <= first["lo"] < first["hi"] <= 1.0


def test_bootstrap_ci_is_narrower_with_more_subjects():
    base = [0.0, 1.0] * 5
    wide = cmp.bootstrap_subject_ci(base, n_boot=2000, seed=1)
    narrow = cmp.bootstrap_subject_ci(base * 3, n_boot=2000, seed=1)
    assert wide is not None and narrow is not None
    assert (narrow["hi"] - narrow["lo"]) < (wide["hi"] - wide["lo"])


def test_aggregate_by_condition_adds_ci_only_when_supported():
    """``ci95`` aparece con 8+ sujetos y desaparece con menos."""
    def block(i: int) -> dict[str, object]:
        return {
            "C0_clean": {
                "n": 10,
                "n_decided": 10 - i % 5,
                "n_abstain": i % 5,
                "accuracy": 0.5,
                "accuracy_incl_abstain": 0.5 - 0.01 * (i % 5),
                "balanced_accuracy": 0.4,
                "coverage": 1.0 - 0.01 * (i % 5),
            }
        }

    rows = {f"S{i:03d}": block(i) for i in range(10)}
    out = cmp.aggregate_by_condition(rows)
    assert "abstain_rate" in out["C0_clean"]["ci95"]

    few = {key: rows[key] for key in list(rows)[:4]}
    assert cmp.aggregate_by_condition(few)["C0_clean"]["ci95"] == {}


def test_compare_models_rejects_unknown_cohort(tmp_path):
    with pytest.raises(ValueError):
        cmp.compare_models(["qwen2.5:7b"], tmp_path, cohort="nope")


def test_comparator_cli_defaults_to_the_cohort_directory():
    """``--cohort scale`` sin ``--dir`` apunta a la carpeta de escala."""
    from scripts.run_p4_dreamt import DEFAULT_OUTPUTS

    args = cmp.build_parser().parse_args(["--models", "a,b", "--cohort", "scale"])
    assert args.cohort == "scale"
    assert args.dir is None
    assert DEFAULT_OUTPUTS["scale"].name == "scale"


# ---------------------------------------------------------------------------
# Deltas pareados para todos los pares (tres modelos)
# ---------------------------------------------------------------------------


def test_paired_delta_keeps_subject_alignment_and_needs_enough_subjects():
    """El delta pareado se forma por sujeto: un ``None`` descarta el par, no la serie."""
    result = cmp.paired_bootstrap_delta([0.2, 0.4, None, 0.6], [0.3, 0.5, 0.9, 0.7])
    assert result["delta"] == pytest.approx(0.1)
    assert result["n_subjects"] == 3
    assert result["ci"] is None  # menos de MIN_SUBJECTS_FOR_CI

    assert cmp.paired_bootstrap_delta([], []) is None


def test_paired_delta_is_deterministic_and_brackets_zero_when_identical():
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    first = cmp.paired_bootstrap_delta(values, values, n_boot=500, seed=11)
    second = cmp.paired_bootstrap_delta(values, values, n_boot=500, seed=11)
    assert first == second
    assert first["delta"] == pytest.approx(0.0)
    assert first["ci"]["lo"] <= 0.0 <= first["ci"]["hi"]


def test_comparison_reports_deltas_for_every_pair(tmp_path):
    """Con tres modelos hay tres pares; antes solo se publicaba (primero, último)."""
    _write_pilot(tmp_path, "qwen2.5:7b", hits=3)
    _write_pilot(tmp_path, "medgemma:4b", hits=1)
    _write_pilot(tmp_path, "llama3.1:8b", hits=0)
    report = cmp.compare_models(
        ["qwen2.5:7b", "medgemma:4b", "llama3.1:8b"],
        tmp_path,
        representations=("numeric",),
        subjects=("S036",),
    )
    pairs = report["deltas"]["numeric"]["pairs"]
    assert set(pairs) == {
        "medgemma:4b - qwen2.5:7b",
        "llama3.1:8b - qwen2.5:7b",
        "llama3.1:8b - medgemma:4b",
    }
    # sentidos: qwen 3/4 aciertos, medgemma 1/4, llama 0/4
    assert pairs["medgemma:4b - qwen2.5:7b"]["conditions"]["C0_clean"]["accuracy"][
        "delta"
    ] == pytest.approx(-0.5)
    assert pairs["llama3.1:8b - medgemma:4b"]["conditions"]["C0_clean"]["accuracy"][
        "delta"
    ] == pytest.approx(-0.25)
    # y se mantiene el bloque histórico (último − primero)
    assert report["deltas"]["numeric"]["other"] == "llama3.1:8b"
