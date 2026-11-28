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


def code_state(repo_dir: Optional[Path] = None) -> dict:
    """Estado del código en ejecución: commit, rama, fecha y cambios locales.

    Sirve para saber, leyendo el log de un job, **qué código corrió**. Un
    despliegue sin actualizar produce un log indistinguible del correcto salvo
    por esta línea: el 2026-11-28 un job se lanzó en el servidor sin haber
    hecho `git pull`, el despachador antiguo ignoró `--repetitions` en silencio
    y el log no lo delataba (0 llamadas nuevas, sin error).

    Nunca lanza: si no hay git o el directorio no es un repositorio, devuelve
    ``{"available": False}``.
    """
    info = git_info(repo_dir)
    state = {
        "available": bool(info.get("available")),
        "commit": info.get("commit"),
        "branch": info.get("branch"),
        "date": None,
        "dirty": bool(info.get("dirty")),
    }
    if state["available"]:
        try:
            state["date"] = _git(["log", "-1", "--format=%cI"], str(repo_dir or Path.cwd()))
        except Exception:  # pragma: no cover - entorno sin git
            state["date"] = None
    return state


def format_code_state(state: Mapping[str, Any]) -> str:
    """Línea de log con el estado del código (ver :func:`code_state`)."""
    if not state.get("available"):
        return "código: desconocido (sin git en el directorio de trabajo)"
    commit = str(state.get("commit") or "")[:7] or "?"
    branch = state.get("branch") or "?"
    date = state.get("date") or "fecha desconocida"
    dirty = " · con cambios sin commitear" if state.get("dirty") else ""
    return f"código: {commit} ({branch}, {date}){dirty}"


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


def opaque_window_id(window_id: str) -> str:
    """Identificador opaco de 8 caracteres para la vista del LLM.

    Evita fuga indirecta de etiqueta mediante IDs reales (``w00001``) o nombres
    de sujeto que permiten memorización o correlación con el protocolo fijo de
    WESAD.

    Vive en ``provenance`` (solo stdlib) y no en ``ingestion.represent`` porque
    también lo usan los analizadores: importarlo desde ``represent`` arrastra
    ``features`` → ``neurokit2``, que en nodos de login con otra CPU puede
    abortar con ``Illegal instruction``.
    """
    return "win_" + hashlib.sha256(window_id.encode()).hexdigest()[:8]


def engine_from_provenance(path: Path) -> Optional[str]:
    """Versión de Ollama registrada en un ``run_provenance*.json``.

    Devuelve ``None`` si el fichero no existe, no es JSON válido o no trae
    versión. Se usa para no reescribir una procedencia ajena: si un corpus se
    generó con otro motor, etiquetarlo con el actual sería una falsificación de
    la procedencia (pasó al copiar el corpus local 0.30.7 al servidor).
    """
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, Mapping):
        return None
    version = payload.get("ollama_version")
    return str(version) if version else None


def read_provenance(path: Path) -> dict[str, Any] | None:
    """Contenido de un ``run_provenance*.json`` (``None`` si no es legible)."""
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def engine_conflicts(
    stored: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
) -> list[str]:
    """Diferencias de identidad del motor entre el corpus y la ejecución actual.

    El motor forma parte del experimento: dos builds del mismo nombre no dan el
    mismo corpus (2026-11-18: 0 frente a 28 abstenciones con Ollama 0.40.6 y
    0.30.7). La caché de reanudación se indexa por ``(ventana, condición,
    repetición)``, **sin** identidad de motor, así que reanudar tras un cambio de
    versión mezclaría corpora en silencio: en un experimento de varianza de
    muestreo eso es indistinguible del ruido que se quiere medir.

    Args:
        stored: bloque de procedencia de un ``run_provenance*.json`` previo.
        current: bloque de procedencia de la ejecución actual.

    Returns:
        Lista de descripciones ``campo: antes -> ahora``. Vacía si no hay
        diferencias comprobables (también si falta el campo en alguno de los dos,
        porque entonces no se puede afirmar que haya cambiado).
    """
    if not stored or not current:
        return []
    conflicts: list[str] = []
    for field in ("model", "model_digest", "ollama_version"):
        before = stored.get(field)
        after = current.get(field)
        if before and after and str(before) != str(after):
            conflicts.append(f"{field}: {before} -> {after}")
    return conflicts


def guard_engine(
    path: Path,
    current: Mapping[str, Any],
    allow_change: bool = False,
    *,
    label: str = "RUN",
) -> list[str]:
    """Aborta si el corpus existente se generó con otro motor.

    Sin esta guardia, reanudar un corpus con otra versión de Ollama mezcla
    motores en el mismo fichero y la varianza de muestreo deja de ser medible.

    Returns:
        Conflictos detectados (vacíos si no hay procedencia previa o no hay
        diferencias). Lanza ``SystemExit`` si hay conflictos y no se permite el
        cambio.
    """
    stored = read_provenance(path)
    conflicts = engine_conflicts(stored, current)
    if not conflicts:
        return []
    message = "; ".join(conflicts)
    if not allow_change:
        raise SystemExit(
            f"[{label}] El corpus de {Path(path).name} se generó con otro motor "
            f"({message}).\n"
            "        Mezclar motores invalida la comparación: usa otro --tag/"
            "--output, borra ese corpus, o pasa --allow-engine-change si aceptas "
            "la advertencia."
        )
    print(f"[{label}] AVISO: cambio de motor en {Path(path).name} ({message})")
    return conflicts


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
