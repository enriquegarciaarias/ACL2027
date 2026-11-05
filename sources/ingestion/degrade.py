"""
Paso de degradación de canales (experimento de resolubilidad).

Genera versiones degradadas del array canónico de un sujeto aplicando
degradación controlada a los canales especificados. Cada condición de
degradación produce su propio ``.npz`` en
``results/output/{subject}/INGEST/degraded/{condition_id}/{subject}.npz``.

Tipos de degradación soportados:

- ``severity == 1.0``: canal plano (todos los valores = 0). Simula sensor
  desconectado o señal inutilizable.
- ``severity == 0.5``: degradación parcial — 50 % de las muestras sustituidas
  por NaN (dropout aleatorio).
- ``severity == 0.0``: sin degradación (copia del canónico original).

Invariante: la etiqueta (label) nunca se degrada; solo los canales de señal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sources.common.common import logger, writeLog
from sources.common.paths import atomic_savez, atomic_write_json, ensure_dir, read_json
from sources.common.provenance import build_fingerprint, sha256_file, utc_now
from sources.ingestion import loader
from sources.ingestion.config import get_ingestion_config
from sources.ingestion.schemas import SCHEMA_VERSION


def degraded_dir() -> Path:
    from sources.common.paths import corpus_dir

    return corpus_dir() / "degraded"


def degraded_paths(subject: str, condition_id: str) -> tuple[Path, Path]:
    """Devuelve ``(npz, meta.json)`` de una condición degradada de un sujeto."""
    base = degraded_dir() / condition_id
    return base / f"{subject}.npz", base / f"{subject}.meta.json"


def _degrade_array(arr: np.ndarray, severity: float) -> np.ndarray:
    """Aplica degradación a un array 1D o 2D.

    Args:
        arr: array de señal original.
        severity: 0.0 (sin degradación), 0.5 (parcial: 50 % NaN),
                  1.0 (total: canal plano en ceros).

    Returns:
        array degradado (misma forma, float32).
    """
    out = arr.astype(np.float32).copy()
    if severity <= 0.0:
        return out
    if severity >= 1.0:
        return np.zeros_like(out)

    # Degradación parcial: sustituir `severity` fracción de muestras por NaN
    flat = out.reshape(-1)
    n = flat.size
    n_nan = int(round(severity * n))
    if n_nan > 0:
        idx = np.random.default_rng(42).choice(n, size=n_nan, replace=False)
        flat[idx] = np.nan
    return out.reshape(out.shape)


def _condition_fingerprint(
    subject: str,
    condition_id: str,
    channels: list[str],
    severity: float,
    canonical_sha256: str,
) -> str:
    """Calcula el fingerprint de una condición degradada."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "subject": subject,
        "condition_id": condition_id,
        "channels": sorted(channels),
        "severity": float(severity),
        "canonical_sha256": canonical_sha256,
    }
    return build_fingerprint(payload)


def _condition_id_hash(channels: list[str], severity: float) -> str:
    """Genera un identificador corto y determinístico para una condición."""
    payload = f"{','.join(sorted(channels))}:{severity:.2f}"
    return "deg_" + hashlib.sha256(payload.encode()).hexdigest()[:8]


def run_degrade(
    cfg: Mapping[str, Any],
    subject: str,
    *,
    force: bool = False,
) -> dict:
    """Aplica todas las condiciones de degradación configuradas a un sujeto.

    Lee el canónico ``.npz`` del sujeto y genera una versión degradada por
    cada condición en ``experiment.degradation_conditions``.

    Args:
        cfg: configuración de ingesta resuelta.
        subject: identificador del sujeto.
        force: recalcular aunque el fingerprint coincida.

    Returns:
        dict con ``subject``, ``status``, ``conditions`` (lista de resultados
        por condición) y ``n_conditions``.
    """
    exp_cfg = cfg.get("experiment", {})
    if not bool(exp_cfg.get("degradation_enabled", False)):
        return {
            "subject": subject,
            "status": "skipped",
            "reason": "experiment.degradation_enabled=false",
            "conditions": [],
            "n_conditions": 0,
        }

    conditions = exp_cfg.get("degradation_conditions", [])
    if not conditions:
        return {
            "subject": subject,
            "status": "skipped",
            "reason": "experiment.degradation_conditions vacía",
            "conditions": [],
            "n_conditions": 0,
        }

    # Cargar canónico
    npz_path, meta_path = loader.canonical_paths(subject)
    if not npz_path.is_file():
        from sources.common.validation import ValidationError

        raise ValidationError(
            f"no existe el canónico de {subject}; ejecute primero '--step ingest'",
            where="degrade",
        )
    meta = read_json(meta_path)
    canonical_sha256 = meta.get("canonical_sha256", "")
    arrays = loader.load_canonical(subject)

    results: list[dict] = []
    for cond in conditions:
        cond_id = str(cond.get("id", ""))
        channels = list(cond.get("channels", []))
        severity = float(cond.get("severity", 0.0))

        if not cond_id:
            cond_id = _condition_id_hash(channels, severity)

        fingerprint = _condition_fingerprint(
            subject, cond_id, channels, severity, canonical_sha256
        )

        out_npz, out_meta = degraded_paths(subject, cond_id)
        previous = read_json(out_meta)

        if (
            not force
            and previous is not None
            and previous.get("fingerprint") == fingerprint
            and out_npz.is_file()
        ):
            writeLog(
                "info",
                logger,
                f"[degrade] {subject}/{cond_id}: reutilizado (fingerprint {fingerprint[:12]}…)",
            )
            results.append(
                {
                    "condition_id": cond_id,
                    "channels": channels,
                    "severity": severity,
                    "status": "reused",
                    "fingerprint": fingerprint,
                    "npz": str(out_npz),
                }
            )
            continue

        # Aplicar degradación a los canales especificados
        degraded = dict(arrays)
        for ch in channels:
            key = f"chest_{ch}" if f"chest_{ch}" in degraded else ch
            if key not in degraded:
                # Probar también con prefijo wrist_
                alt_key = f"wrist_{ch}"
                if alt_key in degraded:
                    key = alt_key
                else:
                    writeLog(
                        "warning",
                        logger,
                        f"[degrade] {subject}/{cond_id}: canal '{ch}' no encontrado en canónico; se omite",
                    )
                    continue
            degraded[key] = _degrade_array(degraded[key], severity)

        # Guardar .npz degradado
        ensure_dir(out_npz.parent)
        compress = bool(cfg.get("persist", {}).get("compression", False))
        atomic_savez(out_npz, degraded, compress=compress)
        degraded_sha = sha256_file(out_npz)
        del degraded
        import gc

        gc.collect()

        out_meta_dict = {
            "schema_version": SCHEMA_VERSION,
            "subject": subject,
            "condition_id": cond_id,
            "channels": channels,
            "severity": severity,
            "source_canonical": str(npz_path),
            "source_canonical_sha256": canonical_sha256,
            "degraded_file": str(out_npz),
            "degraded_sha256": degraded_sha,
            "fingerprint": fingerprint,
            "created_at": utc_now(),
            "shapes": {k: list(v.shape) for k, v in arrays.items()},
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        }
        atomic_write_json(out_meta, out_meta_dict)

        writeLog(
            "info",
            logger,
            f"[degrade] {subject}/{cond_id}: ok (canales={channels}, severidad={severity}, sha256={degraded_sha[:12]}…)",
        )
        results.append(
            {
                "condition_id": cond_id,
                "channels": channels,
                "severity": severity,
                "status": "ok",
                "fingerprint": fingerprint,
                "npz": str(out_npz),
            }
        )

    return {
        "subject": subject,
        "status": "ok",
        "conditions": results,
        "n_conditions": len(results),
    }


def load_degraded(subject: str, condition_id: str) -> dict[str, np.ndarray]:
    """Carga los arrays degradados de una condición de un sujeto."""
    npz_path, _ = degraded_paths(subject, condition_id)
    if not npz_path.is_file():
        from sources.common.validation import ValidationError

        raise ValidationError(
            f"no existen datos degradados de {subject} para '{condition_id}'; ejecute primero '--step degrade'",
            where="degrade",
        )
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def available_conditions(cfg: Mapping[str, Any]) -> list[dict]:
    """Devuelve la lista de condiciones de degradación configuradas."""
    exp_cfg = cfg.get("experiment", {})
    return list(exp_cfg.get("degradation_conditions", []))
