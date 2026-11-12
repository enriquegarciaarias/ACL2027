"""Tests del fingerprint por contenido de código (sin historial de git).

Regresión que motivó el cambio: el commit de git formaba parte de todos los
fingerprints, así que **cualquier commit** (incluida documentación) invalidaba
el corpus y obligaba a reprocesar 19 GB. Ahora el fingerprint depende del
código que produce el artefacto, no del historial del repositorio.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import sources.ingestion.features as features_module
from sources.common.provenance import code_fingerprint, repo_root, sha256_json
from sources.ingestion import loader, qc, represent, window
from sources.ingestion.features import feature_fingerprint

META = {"canonical_sha256": "abc123"}
WINDOWS_REPORT = {"fingerprint": "win123"}
FEATURES_REPORT = {"fingerprint": "feat123"}


def _entry():
    class _Entry:
        subject = "SX"
        sha256 = "src123"

    return _Entry()


INGEST_CFG = {
    "sampling_rates": {"chest": 700.0},
    "units": {"chest.ECG": "mV"},
    "labels": {"keep": [1, 2, 3, 4]},
    "hashing": {"chunk_size": 1024},
}


# --------------------------------------------------------------------------
# code_fingerprint
# --------------------------------------------------------------------------


def test_code_fingerprint_es_estable():
    scope = ("sources/ingestion/qc.py",)
    assert code_fingerprint(scope) == code_fingerprint(scope)


def test_code_fingerprint_distingue_ficheros():
    assert code_fingerprint(("sources/ingestion/qc.py",)) != code_fingerprint(
        ("sources/ingestion/window.py",)
    )


def test_code_fingerprint_ruta_inexistente_no_lanza():
    digest = code_fingerprint(("sources/ingestion/no_existe.py",))
    assert digest == sha256_json({"sources/ingestion/no_existe.py": "missing"})


def test_code_fingerprint_expande_directorios():
    """Un directorio se expande a todos sus ``*.py`` (sin ``__pycache__``)."""
    scope = ("sources/ingestion/features",)
    root = repo_root()
    digests = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / scope[0]).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    assert len(digests) > 5
    assert code_fingerprint(scope) == sha256_json(digests)


# --------------------------------------------------------------------------
# Invariante: el commit no entra en ningún fingerprint
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "sources/ingestion/loader.py",
        "sources/ingestion/qc.py",
        "sources/ingestion/window.py",
        "sources/ingestion/features/__init__.py",
        "sources/ingestion/represent.py",
        "sources/ingestion/package.py",
    ],
)
def test_ningun_fingerprint_incluye_git(path):
    text = (repo_root() / path).read_text(encoding="utf-8")
    # El commit puede seguir como *metadata* (meta.json, manifiesto), pero
    # nunca dentro de un payload de fingerprint.
    assert 'git_info().get("commit")' not in text, (
        f"{path} vuelve a meter el commit de git en un fingerprint"
    )


def test_ingest_fingerprint_no_cambia_con_el_commit(monkeypatch):
    monkeypatch.setattr(loader, "git_info", lambda: {"commit": "aaa"})
    first = loader.ingest_fingerprint(_entry(), INGEST_CFG)
    monkeypatch.setattr(loader, "git_info", lambda: {"commit": "bbb"})
    second = loader.ingest_fingerprint(_entry(), INGEST_CFG)
    assert first == second


# --------------------------------------------------------------------------
# Invariante: sí depende del código que lo produce
# --------------------------------------------------------------------------


def test_ingest_fingerprint_cambia_con_el_codigo(monkeypatch):
    monkeypatch.setattr(loader, "code_fingerprint", lambda _scope: "code-a")
    first = loader.ingest_fingerprint(_entry(), INGEST_CFG)
    monkeypatch.setattr(loader, "code_fingerprint", lambda _scope: "code-b")
    assert loader.ingest_fingerprint(_entry(), INGEST_CFG) != first


def test_qc_fingerprint_cambia_con_el_codigo(monkeypatch):
    cfg = {"qc": {"channels": []}, "sampling_rates": {"chest": 700.0}}
    base = qc.qc_fingerprint(META, cfg)
    monkeypatch.setattr(qc, "code_fingerprint", lambda _scope: "otro")
    assert qc.qc_fingerprint(META, cfg) != base


def test_window_fingerprint_cambia_con_el_codigo(monkeypatch):
    cfg = {"windows": {"length_s": 60}, "labels": {"keep": [1]},
           "sampling_rates": {"chest": 700.0}}
    base = window.window_fingerprint(META, cfg)
    monkeypatch.setattr(window, "code_fingerprint", lambda _scope: "otro")
    assert window.window_fingerprint(META, cfg) != base


def test_feature_fingerprint_cambia_con_el_codigo(monkeypatch):
    cfg = {"features": {"sqi": {}}, "sampling_rates": {"chest": 700.0}}
    base = feature_fingerprint(META, WINDOWS_REPORT, cfg)
    monkeypatch.setattr(features_module, "code_fingerprint", lambda _scope: "otro")
    assert feature_fingerprint(META, WINDOWS_REPORT, cfg) != base


def test_representation_fingerprint_cambia_con_el_codigo(monkeypatch):
    cfg = {"representations": {"numeric": {}, "text": {}, "enabled": ["numeric"]}}
    base = represent.representation_fingerprint(
        META, WINDOWS_REPORT, FEATURES_REPORT, cfg, "numeric"
    )
    monkeypatch.setattr(represent, "code_fingerprint", lambda _scope: "otro")
    assert (
        represent.representation_fingerprint(
            META, WINDOWS_REPORT, FEATURES_REPORT, cfg, "numeric"
        )
        != base
    )


def test_feature_fingerprint_separa_condiciones():
    """La condición degradada sigue formando parte de la clave del fingerprint."""
    cfg = {"features": {}, "sampling_rates": {"chest": 700.0}}
    clean = feature_fingerprint(META, WINDOWS_REPORT, cfg)
    degraded = feature_fingerprint(META, WINDOWS_REPORT, cfg, condition_id="C4_total_eda")
    assert clean != degraded
