"""
Descubrimiento y manifiesto reproducible del dataset WESAD.

Reglas:

- Solo se consideran directorios ``S<dígitos>`` que contengan
  ``S<id>/S<id>.pkl``. Los ZIP de duplicados, PDFs y ficheros sueltos se
  registran como ignorados.
- El SHA-256 se cachea por ``(tamaño, mtime_ns)`` en el manifiesto anterior
  para no rehashear 15 GB en cada ejecución. ``--force`` invalida la caché.
- El manifiesto se escribe de forma atómica en
  ``results/output/{subject}/{proc}/dataset_manifest.json``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

from sources.common.common import logger, writeLog
from sources.common.paths import atomic_write_json, corpus_dir, ensure_dir, read_json, subject_input_dir
from sources.common.provenance import build_fingerprint, sha256_file, utc_now
from sources.common.validation import require
from sources.ingestion.config import filter_available, requested_subjects
from sources.ingestion.schemas import DatasetManifest, SCHEMA_VERSION, SubjectEntry


def manifest_path() -> Path:
    return corpus_dir() / "dataset_manifest.json"


def dataset_root(cfg: Mapping[str, Any]) -> Path:
    """Directorio que contiene las carpetas ``SX``."""
    source_dir = str(cfg.get("source_dir", ".")).strip()
    return (subject_input_dir() / source_dir).resolve()


def discover_subject_dirs(root: Path, pattern: str) -> list[tuple[str, Path]]:
    """Devuelve ``(subject, ruta)`` ordenados numéricamente."""
    regex = re.compile(pattern)
    found: list[tuple[str, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"no existe el directorio del dataset: {root}")

    for child in root.iterdir():
        if not child.is_dir():
            continue
        if regex.match(child.name):
            found.append((child.name, child))

    found.sort(key=lambda item: int(re.sub(r"\D", "", item[0]) or 0))
    return found


def _read_readme_notes(path: Path) -> list[str]:
    """Extrae las notas adicionales del ``SX_readme.txt`` si existen."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    marker = "Additional notes"
    if marker not in text:
        return []
    tail = text.split(marker, 1)[1]
    lines = [line.strip(" #\t") for line in tail.splitlines()]
    return [line for line in lines if line]


def _cached_hash(previous: Optional[dict], size: int, mtime_ns: int, algorithm: str) -> Optional[str]:
    if not previous:
        return None
    if previous.get("size_bytes") != size or previous.get("mtime_ns") != mtime_ns:
        return None
    digest = previous.get("sha256")
    return digest if digest else None


def build_manifest(
    cfg: Mapping[str, Any],
    *,
    force_hash: bool = False,
    requested: Optional[list[str]] = None,
) -> dict:
    """Construye (sin escribir) el manifiesto del dataset."""
    root = dataset_root(cfg)
    algorithm = cfg.get("hashing", {}).get("algorithm", "sha256")
    chunk_size = int(cfg.get("hashing", {}).get("chunk_size", 1024 * 1024))
    require(algorithm == "sha256", f"algoritmo de hash no soportado: {algorithm}", where="discover")

    previous_doc = read_json(manifest_path()) or {}
    previous_by_subject = {
        entry.get("subject"): entry for entry in previous_doc.get("subjects", [])
    }

    candidates = discover_subject_dirs(root, str(cfg.get("subject_glob", r"^S\d+$")))
    available = [name for name, _ in candidates]
    selected = filter_available(requested, available)
    selected_set = set(selected)

    subjects: list[SubjectEntry] = []
    ignored: list[str] = []

    for name, folder in candidates:
        pkl_path = folder / str(cfg.get("source_file", "{subject}.pkl")).format(subject=name)
        if not pkl_path.is_file():
            ignored.append(f"{folder.name}: falta {pkl_path.name}")
            continue

        stat = pkl_path.stat()
        cached = None if force_hash else _cached_hash(
            previous_by_subject.get(name), stat.st_size, stat.st_mtime_ns, algorithm
        )
        if cached:
            digest = cached
            writeLog("debug", logger, f"[discover] {name}: hash reutilizado ({digest[:12]}…)")
        else:
            writeLog("info", logger, f"[discover] {name}: calculando sha256 de {pkl_path.name}…")
            digest = sha256_file(pkl_path, chunk_size=chunk_size)

        readme = folder / f"{name}_readme.txt"
        quest = folder / f"{name}_quest.csv"
        entry = SubjectEntry(
            subject=name,
            file=str(pkl_path),
            sha256=digest,
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            selected=name in selected_set,
            readme=str(readme) if readme.is_file() else None,
            quest=str(quest) if quest.is_file() else None,
            notes=_read_readme_notes(readme),
            issues=[],
        )
        subjects.append(entry)

    # Ficheros y directorios de primer nivel no reconocidos.
    known = {folder.name for _, folder in candidates}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name in known:
            continue
        if child.name.startswith("."):
            continue
        ignored.append(child.name)

    fingerprint = build_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": cfg.get("dataset"),
            "root": str(root),
            "subjects": [(entry.subject, entry.sha256) for entry in subjects],
        }
    )

    manifest = DatasetManifest(
        dataset=str(cfg.get("dataset", "WESAD")),
        root=str(root),
        algorithm=algorithm,
        generated_at=utc_now(),
        config_fingerprint=fingerprint,
        subjects=subjects,
        ignored=ignored,
        schema_version=SCHEMA_VERSION,
    )
    return manifest.to_dict()


def run_discover(
    cfg: Mapping[str, Any],
    *,
    force_hash: bool = False,
    requested: Optional[list[str]] = None,
) -> dict:
    """Construye y persiste el manifiesto del dataset."""
    manifest = build_manifest(cfg, force_hash=force_hash, requested=requested)
    ensure_dir(corpus_dir())
    path = manifest_path()
    atomic_write_json(path, manifest)

    n_total = len(manifest["subjects"])
    n_selected = sum(1 for entry in manifest["subjects"] if entry["selected"])
    total_gb = sum(entry["size_bytes"] for entry in manifest["subjects"]) / (1024 ** 3)
    writeLog(
        "info",
        logger,
        f"[discover] {n_total} sujetos ({total_gb:.1f} GB), {n_selected} seleccionados; "
        f"manifiesto en {path}",
    )
    return manifest


def selected_entries(manifest: Mapping[str, Any]) -> list[SubjectEntry]:
    """Devuelve las entradas seleccionadas del manifiesto."""
    return [
        SubjectEntry.from_dict(entry)
        for entry in manifest.get("subjects", [])
        if entry.get("selected", False)
    ]
