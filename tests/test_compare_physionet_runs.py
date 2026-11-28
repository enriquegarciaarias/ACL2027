"""Tests del contraste pareado entre corridas de PhysioNet 2017."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_physionet_runs import (
    compare,
    load_run,
    main,
    print_report,
    run_paths,
)


def _row(
    record_id: str,
    decision: str,
    status: str = "valid_response",
    claims: list[dict] | None = None,
    reasoning: str = "The rhythm looks regular and the values are within range.",
    hr: float = 1.5,
) -> dict:
    prompt = (
        "Input features (single-lead ECG):\n\n"
        "Modality: ECG\n"
        f"  - ecg_hr_bpm: {hr} SD\n"
        "  - ecg_rmssd_ms: -0.70 SD\n"
    )
    return {
        "record_id": record_id,
        "status": status,
        "decision": None if status == "valid_abstention" else decision,
        "confidence": 85,
        "claims": claims or [],
        "reasoning": reasoning,
        "prompt": prompt,
    }


def _write_run(
    base: Path,
    tag: str,
    rows: list[dict],
    labels: dict[str, str],
    engine: str | None = "0.40.6",
) -> None:
    base.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    with (base / f"physionet2017{suffix}.responses.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with (base / f"physionet2017{suffix}.labels.jsonl").open("w", encoding="utf-8") as fh:
        for record_id, label in labels.items():
            fh.write(json.dumps({"record_id": record_id, "label": label}) + "\n")
    if engine:
        (base / f"run_provenance{suffix}.json").write_text(
            json.dumps({"ollama_version": engine, "model": "qwen2.5:7b"}),
            encoding="utf-8",
        )


LABELS = {"A00001": "normal", "A00002": "af", "A00003": "other-rhythm", "A00004": "noisy"}


def test_run_paths_uses_the_analyzer_naming():
    """El comparador no duplica el nombrado de ficheros del analizador."""
    paths = run_paths(Path("/tmp/x"), "srv")
    assert paths["responses"].name == "physionet2017_srv.responses.jsonl"
    assert paths["labels"].name == "physionet2017_srv.labels.jsonl"
    assert paths["provenance"].name == "run_provenance_srv.json"
    assert run_paths(Path("/tmp/x"))["responses"].name == "physionet2017.responses.jsonl"


def test_compare_detects_abstention_change_on_noisy(tmp_path):
    """A decide `normal` en todo; B se abstiene en la única ventana `noisy`."""
    rows_a = [_row(rid, "normal") for rid in LABELS]
    rows_b = [
        _row(rid, "normal", status="valid_abstention") if rid == "A00004" else _row(rid, "normal")
        for rid in LABELS
    ]
    _write_run(tmp_path, "", rows_a, LABELS, engine="0.30.7")
    _write_run(tmp_path, "srv", rows_b, LABELS, engine="0.40.6")

    report = compare(tmp_path, tag_a="", tag_b="srv", n_boot=50, claims=False)

    assert report["paired"]["n_shared"] == 4
    assert report["paired"]["n_both_valid_labelled"] == 4
    assert report["transitions"] == {"normal->normal": 3, "normal->abstain": 1}
    # A: 4 decididas, 1 acierto. B: 3 decididas, 1 acierto.
    assert report["correctness"]["a"]["accuracy"] == pytest.approx(0.25)
    assert report["correctness"]["b"]["accuracy"] == pytest.approx(1 / 3)
    assert report["correctness"]["b"]["accuracy_incl_abstain"] == pytest.approx(0.25)
    in_noisy = report["correctness"]["b"]["per_class"]["noisy"]
    assert in_noisy["abstentions"] == 1 and in_noisy["abstention_rate"] == 1.0
    assert report["correctness"]["a"]["per_class"]["noisy"]["abstentions"] == 0
    # Sin pares discordantes, McNemar no puede rechazar.
    assert report["mcnemar"]["n_discordant"] == 0
    assert report["mcnemar"]["p_value"] == 1.0
    assert report["engine_changed"] is True


def test_engine_change_requires_both_engines_known(tmp_path):
    """Sin procedencia no se puede afirmar que hubo cambio de motor."""
    _write_run(tmp_path, "", [_row("A00001", "normal")], {"A00001": "normal"}, engine=None)
    _write_run(tmp_path, "srv", [_row("A00001", "normal")], {"A00001": "normal"}, engine="0.40.6")
    report = compare(tmp_path, tag_a="", tag_b="srv", n_boot=10, claims=False)
    assert report["engine_changed"] is False


def test_compare_reports_missing_records(tmp_path):
    """Los registros no compartidos se declaran, no se promedian a ciegas."""
    rows_a = [_row(rid, "normal") for rid in LABELS]
    rows_b = [_row(rid, "normal") for rid in list(LABELS)[:3]]
    _write_run(tmp_path, "", rows_a, LABELS)
    _write_run(tmp_path, "srv", rows_b, LABELS)

    report = compare(tmp_path, tag_a="", tag_b="srv", n_boot=20, claims=False)
    assert report["paired"]["only_a"] == ["A00004"]
    assert report["paired"]["n_shared"] == 3


def test_compare_claims_and_certainty_are_paired(tmp_path):
    """El claim pasa de respaldado a contradicho y la certeza cae."""
    claim_ok = [{"metric": "ecg_hr_bpm", "channel": "ecg", "direction": "increase", "magnitude": "medium"}]
    claim_bad = [{"metric": "ecg_hr_bpm", "channel": "ecg", "direction": "decrease", "magnitude": "medium"}]
    rows_a = [_row("A00001", "normal", claims=claim_ok, reasoning="The rate is clearly high, definitely.")]
    rows_b = [_row("A00001", "normal", claims=claim_bad, reasoning="It might possibly be slightly high.")]
    labels = {"A00001": "normal"}
    _write_run(tmp_path, "", rows_a, labels)
    _write_run(tmp_path, "srv", rows_b, labels)

    report = compare(tmp_path, tag_a="", tag_b="srv", n_boot=50)

    assert report["claims"]["a"]["supported_rate"] == 1.0
    assert report["claims"]["b"]["contradicted_rate"] == 1.0
    delta_claims = report["paired_deltas"]["claims_supported"]
    assert delta_claims["delta"] == pytest.approx(-1.0)
    delta_certainty = report["paired_deltas"]["certainty_index"]
    assert delta_certainty["delta"] < 0
    print_report(report)  # el resumen no debe romperse con un caso mínimo


def test_missing_run_raises_with_actionable_message(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        load_run(tmp_path, "srv")
    assert "--tag srv" in str(excinfo.value)


def test_main_writes_report(tmp_path):
    _write_run(tmp_path, "", [_row("A00001", "normal")], {"A00001": "normal"})
    _write_run(tmp_path, "srv", [_row("A00001", "normal")], {"A00001": "normal"})
    output = tmp_path / "comparison.json"
    main(["--dir", str(tmp_path), "--tag-b", "srv", "--bootstrap", "10", "--output", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["version"] == "physionet_runs_comparison/1.0"
    assert report["paired"]["n_shared"] == 1
