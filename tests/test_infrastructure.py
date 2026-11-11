"""Tests de la infraestructura común (rutas, escritura atómica, procedencia)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sources.common import paths
from sources.common.provenance import build_fingerprint, canonical_json, sha256_file, sha256_json


def test_corpus_dir_resolves_subject_and_proc(workspace):
    expected = workspace["output"] / "WESAD" / "INGEST"
    assert paths.corpus_dir() == expected
    assert paths.subject_input_dir() == workspace["input"] / "WESAD"


def test_atomic_write_json_roundtrip(workspace):
    target = workspace["tmp"] / "nested" / "artifact.json"
    payload = {"a": 1, "b": [1, 2, 3], "numpy": np.int64(7)}
    paths.atomic_write_json(target, payload)

    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3], "numpy": 7}
    assert not list(target.parent.glob("*.tmp-*"))


def test_atomic_savez_roundtrip(workspace):
    target = workspace["tmp"] / "arrays.npz"
    arrays = {"x": np.arange(5, dtype=np.float32), "y": np.ones((3, 2), dtype=np.float32)}
    paths.atomic_savez(target, arrays)

    with np.load(target) as data:
        assert set(data.files) == {"x", "y"}
        np.testing.assert_array_equal(data["x"], arrays["x"])
        np.testing.assert_array_equal(data["y"], arrays["y"])


def test_sha256_file_matches_hashlib(workspace, tmp_path):
    import hashlib

    file_path = workspace["tmp"] / "blob.bin"
    file_path.write_bytes(b"acl2027" * 1000)
    expected = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert sha256_file(file_path) == expected


def test_fingerprint_is_deterministic_and_order_independent():
    a = build_fingerprint({"b": 2, "a": 1})
    b = build_fingerprint({"a": 1, "b": 2})
    assert a == b
    assert a == sha256_json({"a": 1, "b": 2})
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_manage_env_creates_input_and_output_dirs(tmp_path, monkeypatch):
    """`results/input` y `results/output` se crean si no existen."""
    from sources.common.paramsManager import manageEnv

    monkeypatch.chdir(tmp_path)
    env = manageEnv(
        {
            "environment": {
                "input": "results/input",
                "output": "results/output",
                "cache": ".pycache",
            }
        }
    )
    assert (tmp_path / "results" / "input").is_dir()
    assert (tmp_path / "results" / "output").is_dir()
    assert env["input"].endswith("results/input")
