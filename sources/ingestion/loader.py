"""
Carga, validación y canonicalización de un sujeto WESAD.

La salida canónica es un ``.npz`` con arrays ``float32`` (``int32`` para las
etiquetas) y nombres planos, más un ``.meta.json`` con formas, dtypes,
unidades, contadores de etiquetas y fingerprint. El ``.npz`` es un *caché
reproducible*: puede regenerarse desde el ``.pkl`` original.

Decisiones:

- Se procesa un sujeto a la vez para acotar memoria (~1 GB por ``.pkl``).
- Las etiquetas se guardan en el canónico (son necesarias para segmentar),
  pero **no** se copian a las representaciones.
- La conversión de unidades se registra; no se aplican filtros aquí.
"""
from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sources.common.common import logger, writeLog
from sources.common.paths import atomic_savez, atomic_write_json, ensure_dir, read_json
from sources.common.provenance import (
    build_fingerprint,
    code_fingerprint,
    git_info,
    sha256_file,
    utc_now,
)
from sources.common.validation import ValidationError, check_array, check_same_length, require, require_keys
from sources.ingestion import units
from sources.ingestion.schemas import CanonicalMeta, SCHEMA_VERSION, SubjectEntry

REQUIRED_TOP_KEYS = ("signal", "label", "subject")
CHEST_CHANNELS = ("ACC", "ECG", "EMG", "EDA", "Temp", "Resp")
WRIST_CHANNELS = ("ACC", "BVP", "EDA", "TEMP")

#: Código que produce el canónico (no incluye el historial de git).
_CODE_SCOPE: tuple[str, ...] = (
    "sources/ingestion/loader.py",
    "sources/ingestion/units.py",
)


def canonical_dir() -> Path:
    from sources.common.paths import corpus_dir

    return corpus_dir() / "canonical"


def canonical_paths(subject: str) -> tuple[Path, Path]:
    """Devuelve ``(npz, meta.json)`` del sujeto."""
    base = canonical_dir()
    return base / f"{subject}.npz", base / f"{subject}.meta.json"


def load_pickle(path: Path) -> dict:
    """Carga un ``SX.pkl`` de WESAD con la codificación original."""
    path = Path(path)
    require(path.is_file(), f"no existe el fichero: {path}", where="loader")
    try:
        with path.open("rb") as fh:
            return pickle.load(fh, encoding="latin1")
    except Exception as exc:
        raise ValidationError(f"no se pudo cargar '{path}': {exc}", where="loader") from exc


def validate_pickle(raw: Mapping[str, Any], subject: str, cfg: Mapping[str, Any]) -> dict:
    """Valida el esquema del ``.pkl`` y la coherencia temporal.

    Returns:
        dict con ``n_samples``, ``duration_s``, ``wrist_lengths`` y
        ``expected_wrist_lengths``.
    """
    require_keys(raw, REQUIRED_TOP_KEYS, where=f"WESAD/{subject}")
    require(str(raw["subject"]) == subject, f"subject del pkl={raw['subject']!r} != carpeta={subject!r}", where="loader")

    signal = raw["signal"]
    require_keys(signal, ("chest", "wrist"), where=f"WESAD/{subject}.signal")
    require_keys(signal["chest"], CHEST_CHANNELS, where=f"WESAD/{subject}.signal.chest")
    require_keys(signal["wrist"], WRIST_CHANNELS, where=f"WESAD/{subject}.signal.wrist")

    chest = {name: check_array(f"chest.{name}", signal["chest"][name], min_len=1) for name in CHEST_CHANNELS}
    n_samples = check_same_length(chest, where=f"WESAD/{subject}.chest")

    label = check_array("label", raw["label"], ndim=1, finite=True)
    require(len(label) == n_samples, f"label tiene {len(label)} muestras, chest {n_samples}", where=f"WESAD/{subject}")

    rates = cfg["sampling_rates"]
    fs_chest = float(rates["chest"])
    fs_label = float(rates["label"])
    require(fs_chest == fs_label, "chest y label deben compartir frecuencia de muestreo", where="ingestion")

    duration_s = n_samples / fs_chest
    wrist = signal["wrist"]
    wrist_lengths: dict[str, int] = {}
    expected: dict[str, int] = {}
    for channel, rate_key in (
        ("ACC", "wrist_acc"),
        ("BVP", "wrist_bvp"),
        ("EDA", "wrist_eda"),
        ("TEMP", "wrist_temp"),
    ):
        arr = check_array(f"wrist.{channel}", wrist[channel], min_len=1)
        wrist_lengths[channel] = int(arr.shape[0])
        expected[channel] = int(round(duration_s * float(rates[rate_key])))
        tolerance = max(2, int(round(0.005 * expected[channel])))
        if abs(wrist_lengths[channel] - expected[channel]) > tolerance:
            raise ValidationError(
                f"wrist.{channel} tiene {wrist_lengths[channel]} muestras; "
                f"se esperaban {expected[channel]} ± {tolerance} para {duration_s:.2f}s "
                f"a {rates[rate_key]} Hz",
                where=f"WESAD/{subject}",
            )

    return {
        "n_samples": int(n_samples),
        "duration_s": float(duration_s),
        "wrist_lengths": wrist_lengths,
        "expected_wrist_lengths": expected,
    }


def canonicalize(
    normalized: Mapping[str, Mapping[str, np.ndarray]],
    label: np.ndarray,
) -> dict[str, np.ndarray]:
    """Aplana las modalidades a claves compatibles con ``np.savez``."""
    arrays: dict[str, np.ndarray] = {}
    for modality in ("chest", "wrist"):
        for channel, arr in normalized[modality].items():
            arrays[f"{modality}_{channel}"] = np.ascontiguousarray(arr)
    arrays["label"] = np.ascontiguousarray(np.asarray(label, dtype=np.int32))
    return arrays


def ingest_fingerprint(entry: SubjectEntry, cfg: Mapping[str, Any]) -> str:
    """Fingerprint del canónico de un sujeto."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": entry.sha256,
        "sampling_rates": cfg["sampling_rates"],
        "units": cfg["units"],
        "labels": cfg["labels"],
        "hashing": cfg["hashing"],
        # El commit de git queda fuera a propósito: solo invalida el corpus un
        # cambio en el código que lo produce (ver ``code_fingerprint``).
        "code": code_fingerprint(_CODE_SCOPE),
    }
    return build_fingerprint(payload)


def run_ingest(
    cfg: Mapping[str, Any],
    entry: SubjectEntry,
    *,
    force: bool = False,
) -> dict:
    """Canonicaliza un sujeto y escribe ``.npz`` + ``.meta.json``.

    Returns:
        dict con ``status`` (``ok``/``reused``), ``subject``, ``fingerprint``
        y la ruta del canónico.
    """
    npz_path, meta_path = canonical_paths(entry.subject)
    fingerprint = ingest_fingerprint(entry, cfg)

    previous = read_json(meta_path)
    if (
        not force
        and previous is not None
        and previous.get("fingerprint") == fingerprint
        and npz_path.is_file()
    ):
        writeLog("info", logger, f"[ingest] {entry.subject}: canónico reutilizado (fingerprint {fingerprint[:12]}…)")
        return {
            "subject": entry.subject,
            "status": "reused",
            "fingerprint": fingerprint,
            "canonical": str(npz_path),
        }

    writeLog("info", logger, f"[ingest] {entry.subject}: cargando {entry.file}")
    raw = load_pickle(Path(entry.file))
    try:
        info = validate_pickle(raw, entry.subject, cfg)
        normalized, unit_reports = units.normalize_units(raw["signal"], cfg)
        arrays = canonicalize(normalized, raw["label"])
    finally:
        del raw
        gc.collect()

    compress = bool(cfg.get("persist", {}).get("compression", False))
    persist = bool(cfg.get("persist", {}).get("canonical_arrays", True))
    if not persist:
        writeLog("warning", logger, f"[ingest] {entry.subject}: persist.canonical_arrays=false; no se guarda .npz")
        return {"subject": entry.subject, "status": "skipped", "fingerprint": fingerprint}

    shapes = {key: list(value.shape) for key, value in arrays.items()}
    dtypes = {key: str(value.dtype) for key, value in arrays.items()}
    values, counts = np.unique(arrays["label"], return_counts=True)
    label_counts_dict = {str(int(v)): int(c) for v, c in zip(values, counts)}

    ensure_dir(npz_path.parent)
    atomic_savez(npz_path, arrays, compress=compress)
    canonical_sha = sha256_file(npz_path)
    del arrays, normalized
    gc.collect()

    meta = CanonicalMeta(
        subject=entry.subject,
        source_file=str(entry.file),
        source_sha256=entry.sha256,
        canonical_file=str(npz_path),
        canonical_sha256=canonical_sha,
        schema_version=SCHEMA_VERSION,
        created_at=utc_now(),
        fingerprint=fingerprint,
        shapes=shapes,
        dtypes=dtypes,
        units={
            f"{modality}.{channel}": unit
            for (modality, channel), unit in units.CHANNEL_UNITS.items()
        },
        sampling_rates=dict(cfg["sampling_rates"]),
        n_samples=info["n_samples"],
        duration_s=info["duration_s"],
        label_counts=label_counts_dict,
        unit_reports=unit_reports,
    )
    atomic_write_json(meta_path, {**meta.to_dict(), "git": git_info()})

    writeLog(
        "info",
        logger,
        f"[ingest] {entry.subject}: ok ({info['duration_s']:.1f}s, "
        f"{len(meta.label_counts)} etiquetas, sha256 {canonical_sha[:12]}…)",
    )
    return {
        "subject": entry.subject,
        "status": "ok",
        "fingerprint": fingerprint,
        "canonical": str(npz_path),
        "meta": str(meta_path),
    }


def load_canonical(subject: str) -> dict[str, np.ndarray]:
    """Carga el ``.npz`` canónico de un sujeto."""
    npz_path, _ = canonical_paths(subject)
    require(npz_path.is_file(), f"no existe el canónico: {npz_path}", where="loader")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
