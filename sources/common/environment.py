"""Comprobación del entorno de ejecución (venv y dependencias).

Un fallo muy fácil de cometer en un servidor es lanzar el pipeline con el
**intérprete del sistema** en lugar del ``.venv`` del proyecto: entonces faltan
paquetes (`neurokit2`, `scipy`, …) y el error aparece en mitad de un import,
lejos del origen. Este módulo lo detecta al arrancar y produce mensajes
accionables.

No sustituye a `requirements.txt`: solo verifica lo mínimo imprescindible para
que el pipeline pueda importarse.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Sequence

#: Raíz del repositorio (para localizar su ``.venv``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Módulos que deben poder importarse para ejecutar el pipeline.
REQUIRED_MODULES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "sklearn",
    "requests",
    "neurokit2",
)


def in_virtualenv() -> bool:
    """¿El intérprete actual es un entorno virtual?"""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def python_executable() -> str:
    """Ruta del intérprete en uso."""
    return sys.executable


def missing_modules(
    modules: Sequence[str] = REQUIRED_MODULES,
) -> list[str]:
    """Módulos de ``modules`` que no se pueden importar."""
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - cualquier fallo de import cuenta
            missing.append(name)
    return missing


def project_venv_python() -> Path | None:
    """Intérprete del ``.venv`` del proyecto, si existe."""
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def snapshot() -> dict:
    """Estado del entorno (para el plan de ``--dry-run``)."""
    return {
        "python": python_executable(),
        "version": sys.version.split()[0],
        "in_venv": in_virtualenv(),
        "missing_modules": missing_modules(),
    }


def check_environment() -> tuple[bool, list[str], list[str]]:
    """Evalúa el entorno y devuelve ``(ok, errores, avisos)``.

    - ``ok`` es False si falta algún módulo obligatorio.
    - ``avisos`` incluye ejecutar fuera de un entorno virtual (no bloqueante si
      las dependencias están presentes).
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = missing_modules()
    if missing:
        errors.append(
            f"Faltan dependencias en {python_executable()}: {', '.join(missing)}"
        )
        # Pista habitual en servidores: otro venv (p. ej. el del job de Slurm)
        # en lugar del que vive dentro del proyecto.
        project_python = project_venv_python()
        if project_python and str(project_python) != python_executable():
            errors.append(
                f"El intérprete en uso no es el del proyecto. Usa "
                f"{project_python} (o 'source {REPO_ROOT / '.venv'}/bin/activate')."
            )
        if not in_virtualenv():
            errors.append(
                "No se está ejecutando dentro de un entorno virtual. "
                "Usa el del proyecto: source .venv/bin/activate"
            )
        errors.append(
            "Instala las dependencias con: "
            "python -m pip install -r requirements.txt"
        )
    elif not in_virtualenv():
        warnings.append(
            f"Ejecutando con el intérprete del sistema ({python_executable()}); "
            "las dependencias están disponibles, pero se recomienda .venv "
            "para reproducir el entorno."
        )

    return not errors, errors, warnings
