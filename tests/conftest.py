"""
Fixtures compartidas por los tests de ingesta.

Los tests no usan los 15 GB de WESAD: construyen un dataset sintético mínimo
con la misma estructura de ``.pkl`` (``signal``/``label``/``subject``), tasas
de muestreo y unidades.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pytest

from sources.common.common import processControl


CHEST_FS = 700
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
DURATION_S = 300
LABEL_RUNS = ((1, 120), (2, 120), (0, 60))


def _make_subject(subject: str, *, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = CHEST_FS * DURATION_S
    t = np.arange(n, dtype=np.float64) / CHEST_FS

    # ECG sintético con morfología QRS-like y RR ligeramente variable.
    ecg = rng.normal(0, 0.01, size=n)
    beat = 0.5
    while beat < DURATION_S:
        i = int(beat * CHEST_FS)
        if i + 40 < n:
            ecg[i:i + 10] += np.hanning(10)
            ecg[i + 10:i + 30] -= 0.3 * np.hanning(20)
        beat += 0.83 + rng.normal(0, 0.02)

    bvp_t = np.arange(int(DURATION_S * WRIST_FS["BVP"]), dtype=np.float64) / WRIST_FS["BVP"]
    bvp = np.sin(2 * np.pi * 1.2 * bvp_t) + rng.normal(0, 0.02, size=bvp_t.size)

    chest = {
        "ACC": rng.normal(0, 0.1, size=(n, 3)),
        "ECG": ecg.reshape(-1, 1),
        "EMG": rng.normal(0, 0.01, size=(n, 1)),
        "EDA": (2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)).reshape(-1, 1),
        "Temp": (33.0 + 0.1 * t / DURATION_S).reshape(-1, 1),
        "Resp": (10.0 * np.sin(2 * np.pi * 0.25 * t)).reshape(-1, 1),
    }
    wrist = {
        "ACC": rng.integers(-64, 64, size=(int(DURATION_S * WRIST_FS["ACC"]), 3)).astype(np.float64),
        "BVP": bvp.reshape(-1, 1),
        "EDA": (0.5 + 0.1 * np.sin(2 * np.pi * 0.05 * t[:: int(CHEST_FS / WRIST_FS["EDA"])])).reshape(-1, 1),
        "TEMP": (34.0 + 0.1 * np.sin(2 * np.pi * 0.01 * t[:: int(CHEST_FS / WRIST_FS["EDA"])])).reshape(-1, 1),
    }
    label = np.concatenate(
        [
            np.full(duration * CHEST_FS, label_id, dtype=np.int32)
            for label_id, duration in LABEL_RUNS
        ]
    )
    return {"subject": subject, "signal": {"chest": chest, "wrist": wrist}, "label": label}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Entorno aislado con ``results/input`` y ``results/output`` temporales."""
    results = tmp_path / "results"
    input_dir = results / "input"
    output_dir = results / "output"
    dataset_dir = input_dir / "WESAD"
    dataset_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(processControl, "env", {"input": str(input_dir), "output": str(output_dir)})
    monkeypatch.setattr(
        processControl,
        "args",
        argparse.Namespace(
            subject="WESAD",
            proc="INGEST",
            step=None,
            serverGPU=None,
            subjects=None,
            force=False,
            dry_run=False,
        ),
    )
    monkeypatch.setattr(processControl, "defaults", {})
    monkeypatch.setattr(processControl, "parms", {})
    monkeypatch.setattr(processControl, "config", {})

    return {
        "tmp": tmp_path,
        "input": input_dir,
        "output": output_dir,
        "dataset": dataset_dir,
    }


@pytest.fixture
def synthetic_dataset(workspace):
    """Crea dos sujetos sintéticos y ficheros que deben ignorarse."""
    for index, subject in enumerate(("S2", "S3")):
        folder = workspace["dataset"] / subject
        folder.mkdir()
        with (folder / f"{subject}.pkl").open("wb") as fh:
            pickle.dump(_make_subject(subject, seed=index), fh)
        (folder / f"{subject}_readme.txt").write_text(
            "### Additional notes ###\nTest fixture\n", encoding="utf-8"
        )
        (folder / f"{subject}_quest.csv").write_text("dummy\n", encoding="utf-8")

    (workspace["dataset"] / "WESAD.zip").write_bytes(b"not a real zip")
    (workspace["dataset"] / "wesad_readme.pdf").write_bytes(b"%PDF-1.4")
    return workspace


@pytest.fixture
def ingestion_cfg():
    from sources.ingestion.config import get_ingestion_config

    return get_ingestion_config()
