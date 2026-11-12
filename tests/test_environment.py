"""Tests de la comprobación de entorno (venv y dependencias)."""

from __future__ import annotations

import pytest

from sources.common import environment


def test_missing_modules_detects_absent_module():
    missing = environment.missing_modules(("numpy", "modulo_que_no_existe"))
    assert missing == ["modulo_que_no_existe"]


def test_missing_modules_empty_when_all_present():
    assert environment.missing_modules(("numpy", "scipy", "sklearn")) == []


def test_in_virtualenv_returns_bool():
    assert isinstance(environment.in_virtualenv(), bool)


def test_snapshot_structure():
    snap = environment.snapshot()
    assert set(snap) == {"python", "version", "in_venv", "missing_modules"}
    assert snap["python"]


def test_check_environment_ok():
    ok, errors, warnings = environment.check_environment()
    assert ok is True
    assert errors == []


def test_check_environment_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(environment, "missing_modules", lambda *a, **k: ["neurokit2"])
    ok, errors, warnings = environment.check_environment()
    assert ok is False
    assert any("neurokit2" in message for message in errors)
    assert any("pip install -r requirements.txt" in message for message in errors)


def test_check_environment_suggests_venv_when_outside(monkeypatch):
    monkeypatch.setattr(environment, "missing_modules", lambda *a, **k: ["neurokit2"])
    monkeypatch.setattr(environment, "in_virtualenv", lambda: False)
    ok, errors, _ = environment.check_environment()
    assert ok is False
    assert any("source .venv/bin/activate" in message for message in errors)


def test_check_environment_hints_other_venv(monkeypatch, tmp_path):
    """Si se usa otro venv (p. ej. en Slurm), sugiere el del proyecto."""
    other = tmp_path / "otro" / "bin" / "python"
    project = tmp_path / "ACL2027" / ".venv" / "bin" / "python"
    monkeypatch.setattr(environment, "missing_modules", lambda *a, **k: ["neurokit2"])
    monkeypatch.setattr(environment, "python_executable", lambda: str(other))
    monkeypatch.setattr(environment, "project_venv_python", lambda: project)

    ok, errors, _ = environment.check_environment()

    assert ok is False
    assert any("no es el del proyecto" in message for message in errors)
    assert any(str(project) in message for message in errors)


def test_check_environment_warns_outside_venv_without_blocking(monkeypatch):
    monkeypatch.setattr(environment, "missing_modules", lambda *a, **k: [])
    monkeypatch.setattr(environment, "in_virtualenv", lambda: False)
    ok, errors, warnings = environment.check_environment()
    assert ok is True
    assert errors == []
    assert any(".venv" in message for message in warnings)


def test_main_exits_3_when_environment_incomplete(monkeypatch):
    """`main` aborta con código 3 si faltan dependencias."""
    import main as entrypoint

    monkeypatch.setattr(
        entrypoint.environment,
        "check_environment",
        lambda: (False, ["Faltan dependencias: neurokit2"], []),
    )
    with pytest.raises(SystemExit) as excinfo:
        entrypoint._check_environment_or_exit()
    assert excinfo.value.code == 3
