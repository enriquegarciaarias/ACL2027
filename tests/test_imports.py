"""Smoke test de importación de todos los módulos del proyecto.

Python 3.14 evalúa las anotaciones de forma perezosa (PEP 649), así que un
nombre sin importar (`Mapping`, `Any`, …) solo falla al usar la función. En la
versión de referencia (**3.12**) la anotación se evalúa al definir la función y
el módulo ni siquiera importa. Este test recorre `sources/` para detectarlo.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

SOURCES_DIR = Path(__file__).resolve().parents[1] / "sources"


def _module_name(path: Path) -> str:
    parts = list(path.with_suffix("").relative_to(SOURCES_DIR.parent).parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


MODULES = sorted(
    _module_name(path)
    for path in SOURCES_DIR.rglob("*.py")
    if "__pycache__" not in path.parts
)


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str):
    """Cada módulo de `sources/` se importa sin errores."""
    assert importlib.import_module(module) is not None


def test_all_sources_modules_discovered():
    """El barrido cubre el paquete (evita un test vacío por ruta mal calculada)."""
    assert len(MODULES) >= 30
    assert "sources.common.llm_client" in MODULES
    assert "sources.pipelines" in MODULES
