"""
Procedencia, hashing y versionado de artefactos.

Toda decisión del pipeline debe ser reconstruible a partir de:

- el hash del fichero fuente,
- la configuración resuelta,
- el commit del repositorio,
- las versiones del intérprete y de las librerías,
- y una semilla explícita.

Con esos ingredientes se calcula un ``fingerprint`` reproducible que permite
decidir si un paso ya está hecho (idempotencia) y auditar un corpus antiguo.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from sources.common.common import logger, writeLog


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calcula el SHA-256 de un fichero por bloques (sin cargarlo en memoria)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serializa a JSON canónico (claves ordenadas, sin espacios)."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_json(value: Any) -> str:
    """SHA-256 del JSON canónico de ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Marca de tiempo ISO-8601 en UTC con sufijo ``Z``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@lru_cache(maxsize=4)
def git_info(repo_dir: Optional[Path] = None) -> dict:
    """Devuelve commit, rama y estado *dirty* del repositorio.

    Nunca lanza: si no hay git o el directorio no es un repositorio, devuelve
    ``{"available": false}``.
    """
    cwd = str(repo_dir or Path.cwd())
    try:
        commit = _git(["rev-parse", "HEAD"], cwd)
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
        dirty = bool(_git(["status", "--porcelain", "--untracked-files=no"], cwd).strip())
        return {
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception as exc:  # pragma: no cover - entorno sin git
        writeLog("debug", logger, f"No se pudo obtener información de git: {exc}")
        return {"available": False}


def _git(args: list[str], cwd: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@lru_cache(maxsize=1)
def library_versions() -> dict:
    """Versiones relevantes del entorno de ejecución."""
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "scipy", "neurokit2", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    return versions


def build_fingerprint(parts: Mapping[str, Any]) -> str:
    """Construye un fingerprint reproducible a partir de sus componentes."""
    return sha256_json(dict(parts))


def repo_root() -> Path:
    """Raíz del repositorio (``sources/common/provenance.py`` → tres niveles)."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=32)
def code_fingerprint(scope: tuple[str, ...]) -> str:
    """SHA-256 del código que produce un artefacto.

    Sustituye al commit de git dentro de los fingerprints. La diferencia es
    deliberada: un commit de documentación **no** invalida el corpus, pero un
    cambio en el código que calcula el artefacto **sí**. Así el fingerprint
    mide el contenido del pipeline, no el historial del repositorio.

    Args:
        scope: rutas relativas a la raíz del repositorio. Un directorio se
            expande a todos sus ``*.py`` (recursivo, sin ``__pycache__``).

    Returns:
        SHA-256 de ``{ruta: sha256}`` ordenado por ruta.
    """
    root = repo_root()
    digests: dict[str, str] = {}
    for entry in sorted(scope):
        path = root / entry
        if path.is_dir():
            files = sorted(
                item
                for item in path.rglob("*.py")
                if "__pycache__" not in item.parts
            )
        elif path.is_file():
            files = [path]
        else:
            digests[entry] = "missing"
            continue
        for file in files:
            try:
                relative = str(file.relative_to(root))
            except ValueError:  # pragma: no cover - ruta fuera del repo
                relative = str(file)
            digests[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    return sha256_json(digests)



def environment_snapshot() -> dict:
    """Instantánea de procedencia para incluir en el manifiesto del corpus."""
    return {
        "created_at": utc_now(),
        "git": git_info(),
        "versions": library_versions(),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    raise TypeError(f"Objeto no serializable: {type(value)!r}")
