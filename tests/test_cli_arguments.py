"""Pruebas de los argumentos de CLI de ``main.py``.

Cubren el caso real del job 2592 (2026-11-13): al copiar el comando de
``launch_lsi.sh`` se coló un ``--`` de más y ``argparse`` abortaba con
``unrecognized arguments: --``. El separador de la fase va una sola vez.
"""

from __future__ import annotations

import sys

import pytest

from sources.common.paramsManager import _strip_stray_separator, manageArgs


def test_strip_removes_single_trailing_separator():
    cleaned, dropped = _strip_stray_separator(["--proc", "GENERATE", "--"])
    assert cleaned == ["--proc", "GENERATE"]
    assert dropped == 1


def test_strip_keeps_argv_without_separator():
    argv = ["--proc", "INGEST", "--subject", "WESAD", "--dry-run"]
    cleaned, dropped = _strip_stray_separator(argv)
    assert cleaned == argv
    assert dropped == 0


def test_strip_removes_several_separators():
    cleaned, dropped = _strip_stray_separator(["--", "--proc", "EVALUATE", "--", "--"])
    assert cleaned == ["--proc", "EVALUATE"]
    assert dropped == 3


def test_strip_keeps_plain_option_values():
    argv = ["--model", "qwen3:14b", "--experiment", "degradation"]
    cleaned, dropped = _strip_stray_separator(argv)
    assert cleaned == argv
    assert dropped == 0


def test_manage_args_tolerates_trailing_separator(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--proc", "GENERATE",
            "--experiment", "degradation",
            "--subject", "WESAD",
            "--model", "qwen3:14b",
            "--",
        ],
    )
    args = manageArgs()
    assert args.proc == "GENERATE"
    assert args.experiment == "degradation"
    assert args.subject == "WESAD"
    assert args.model == "qwen3:14b"


def test_manage_args_still_fails_on_unknown_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--proc", "GENERATE", "--nope"])
    with pytest.raises(SystemExit) as exc:
        manageArgs()
    assert exc.value.code == 2
