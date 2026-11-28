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


# ---------------------------------------------------------------------------
# Estado del código en el log de arranque
#
# Motivo: el 2026-11-28 un job se lanzó en el servidor sin `git pull`. El
# despachador antiguo ignoró `--repetitions` en silencio, hizo 0 llamadas y el
# log no lo delataba. La línea `código: <commit> (<rama>, <fecha>)` hace visible
# qué código corrió.
# ---------------------------------------------------------------------------
def test_code_state_in_repository():
    from sources.common import provenance

    state = provenance.code_state()
    assert state["available"] is True
    assert len(str(state["commit"])) == 40
    assert isinstance(state["dirty"], bool)
    assert state["date"]


def test_code_state_unavailable_without_git(monkeypatch):
    from sources.common import provenance

    monkeypatch.setattr(provenance, "git_info", lambda *a, **k: {"available": False})
    state = provenance.code_state()
    assert state["available"] is False
    assert "desconocido" in provenance.format_code_state(state)


def test_format_code_state_marks_uncommitted_changes():
    from sources.common import provenance

    line = provenance.format_code_state(
        {
            "available": True,
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "branch": "main",
            "date": "2026-11-28T12:00:00+01:00",
            "dirty": True,
        }
    )
    assert "código: 0123456" in line
    assert "main" in line
    assert "sin commitear" in line


def test_format_code_state_clean_tree():
    from sources.common import provenance

    line = provenance.format_code_state(
        {
            "available": True,
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "branch": "main",
            "date": "2026-11-28T12:00:00+01:00",
            "dirty": False,
        }
    )
    assert "sin commitear" not in line


def test_main_logs_code_state(monkeypatch):
    """`main._run` registra el commit en ejecución antes de delegar."""
    from types import SimpleNamespace

    import main as entrypoint

    messages: list[str] = []
    monkeypatch.setattr(entrypoint, "_check_environment_or_exit", lambda: None)
    monkeypatch.setattr(entrypoint, "getConfigs", lambda: None)
    monkeypatch.setattr(
        entrypoint,
        "code_state",
        lambda: {
            "available": True,
            "commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "branch": "main",
            "date": "2026-11-28T12:00:00+01:00",
            "dirty": False,
        },
    )
    monkeypatch.setattr(
        entrypoint,
        "writeLog",
        lambda level, logger_, message: messages.append(str(message)),
    )
    monkeypatch.setattr(
        entrypoint,
        "processControl",
        SimpleNamespace(args=SimpleNamespace(proc="EVALUATE", step=None)),
    )
    monkeypatch.setattr(entrypoint, "_run_registered_process", lambda proc: None)

    assert entrypoint._run() == 0
    assert any("código: deadbee" in message for message in messages)
