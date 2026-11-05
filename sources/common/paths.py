"""
Resolución centralizada de rutas y escritura atómica de artefactos.

Regla del proyecto: ninguna ruta bajo ``results/`` se construye a mano. Todas
se derivan de ``processControl.env`` (poblado por
``paramsManager.manageEnv()``) y del par ``(subject, proc)``.

Este módulo añade la noción de *corpus*:

    results/output/{subject}/{proc}/

que es el directorio raíz de todos los artefactos de un proceso (por ejemplo,
``results/output/WESAD/INGEST``).

Las escrituras son atómicas (fichero temporal + ``os.replace``) para que un
fallo a mitad de escritura nunca deje un JSON o un ``.npz`` corrupto.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from sources.common.common import processControl


class PathResolutionError(RuntimeError):
    """Error al resolver una ruta del entorno del proyecto."""


def _args_value(name: str) -> Optional[str]:
    """Lee un atributo de ``processControl.args`` sea Namespace o dict."""
    args = processControl.args
    if args is None:
        return None
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def _env_path(key: str) -> Path:
    """Devuelve una ruta obligatoria de ``processControl.env``."""
    env = processControl.env or {}
    value = env.get(key)
    if not value:
        raise PathResolutionError(
            f"La ruta '{key}' no está definida en processControl.env. "
            f"¿Se ejecutó paramsManager.getConfigs()?"
        )
    return Path(value)


def subject_name() -> str:
    """Devuelve el subject activo (``--subject``)."""
    subject = _args_value("subject")
    if not subject:
        raise PathResolutionError(
            "No hay subject activo. Use --subject (p. ej. --subject WESAD)."
        )
    return str(subject)


def proc_name() -> str:
    """Devuelve el proceso activo (``--proc``)."""
    proc = _args_value("proc")
    if not proc:
        raise PathResolutionError(
            "No hay proceso activo. Use --proc (p. ej. --proc INGEST)."
        )
    return str(proc)


def input_root() -> Path:
    """``results/input`` resuelto vía ``processControl.env``."""
    return _env_path("input")


def output_root() -> Path:
    """``results/output`` resuelto vía ``processControl.env``."""
    return _env_path("output")


def subject_input_dir() -> Path:
    """Directorio de entrada del subject: ``results/input/{subject}``."""
    return input_root() / subject_name()


def subject_output_dir() -> Path:
    """Directorio de salida del subject: ``results/output/{subject}``."""
    return output_root() / subject_name()


def corpus_dir() -> Path:
    """Raíz de artefactos del proceso: ``results/output/{subject}/{proc}``."""
    return subject_output_dir() / proc_name()


def ensure_dir(path: Path) -> Path:
    """Crea ``path`` (y sus padres) si no existe y lo devuelve."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tmp_path(path: Path) -> Path:
    """Ruta temporal única en el mismo directorio que ``path``."""
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Escribe texto de forma atómica."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = _tmp_path(path)
    try:
        with tmp.open("w", encoding=encoding, newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Escribe JSON de forma atómica y determinista."""
    payload = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=False,
        default=_json_default,
    )
    return atomic_write_text(path, payload + "\n")


def atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    """Escribe un JSONL de forma atómica (una línea por registro)."""
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, default=_json_default) for row in rows
    )
    return atomic_write_text(path, payload + ("\n" if payload else ""))


def atomic_savez(path: Path, arrays: Mapping[str, np.ndarray], *, compress: bool = False) -> Path:
    """Guarda un ``.npz`` de forma atómica.

    ``np.savez`` añade ``.npz`` si no está presente, por lo que el temporal se
    crea ya con esa extensión y después se renombra al destino final.
    """
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        if compress:
            np.savez_compressed(tmp, **arrays)
        else:
            np.savez(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path


def read_json(path: Path) -> Any:
    """Lee un JSON, devolviendo ``None`` si no existe."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    """Lee un JSONL, devolviendo una lista vacía si no existe."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido en {path}:{line_number}: {exc}") from exc
    return rows


def _json_default(value: Any) -> Any:
    """Serializador para tipos numpy y Path."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Objeto no serializable a JSON: {type(value)!r}")
