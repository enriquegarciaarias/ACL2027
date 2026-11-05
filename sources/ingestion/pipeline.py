"""
Orquestador del pipeline de ingesta (proceso ``INGEST``).

Pasos:

    discover → ingest → qc → window → degrade → features → represent → package

Implementados (H0–H6): ``discover``, ``ingest``, ``qc``, ``window``,
``degrade``, ``features``, ``represent``, ``package``.

El paso ``degrade`` genera arrays degradados por condición (entre ``window``
y ``features``). Las features de cada condición degradada se guardan en
``degraded_features/{condition_id}/{subject}.features.jsonl``.

``--step X`` ejecuta el pipeline **hasta X inclusive** (los pasos anteriores se
reutilizan por fingerprint, así que el coste es bajo). Los pasos pendientes
fallan con un mensaje claro si se solicitan explícitamente.

El estado de cada paso se persiste en ``steps.json`` dentro del corpus.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from sources.common.common import logger, processControl, writeLog
from sources.common.paths import atomic_write_json, corpus_dir, read_json
from sources.common.provenance import environment_snapshot, utc_now
from sources.common.validation import ValidationError
from sources.ingestion import (
    degrade,
    discover,
    features,
    loader,
    package,
    qc,
    represent,
    window,
)
from sources.ingestion.config import filter_available, get_ingestion_config, requested_subjects
from sources.ingestion.schemas import SCHEMA_VERSION

STEPS: tuple[str, ...] = (
    "discover",
    "ingest",
    "qc",
    "window",
    "degrade",
    "features",
    "represent",
    "package",
)

IMPLEMENTED: tuple[str, ...] = (
    "discover",
    "ingest",
    "qc",
    "window",
    "degrade",
    "features",
    "represent",
    "package",
)
PENDING: tuple[str, ...] = tuple(step for step in STEPS if step not in IMPLEMENTED)


def steps_path():
    return corpus_dir() / "steps.json"


def load_steps() -> dict:
    state = read_json(steps_path())
    if not isinstance(state, dict):
        state = {}
    state.setdefault("schema_version", SCHEMA_VERSION)
    state.setdefault("steps", {})
    return state


def update_step(step: str, **fields: Any) -> dict:
    state = load_steps()
    current = state["steps"].get(step, {})
    current.update(fields)
    state["steps"][step] = current
    atomic_write_json(steps_path(), state)
    return state


def _arg(name: str, default: Any = None) -> Any:
    args = processControl.args
    if args is None:
        return default
    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _resolve_requested(cfg: Mapping[str, Any]) -> Optional[list[str]]:
    return requested_subjects(_arg("subjects"), cfg)


def _dry_run(cfg: Mapping[str, Any], requested: Optional[list[str]]) -> None:
    root = discover.dataset_root(cfg)
    candidates: list[str] = []
    selected: list[str] = []
    try:
        candidates = [name for name, _ in discover.discover_subject_dirs(root, str(cfg["subject_glob"]))]
        selected = filter_available(requested, candidates)
    except Exception as exc:
        writeLog("warning", logger, f"[dry-run] no se pudo enumerar sujetos: {exc}")

    payload = {
        "proc": _arg("proc"),
        "subject": _arg("subject"),
        "dataset_root": str(root),
        "corpus_dir": str(corpus_dir()),
        "available_subjects": candidates,
        "selected_subjects": selected,
        "config": cfg,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _check_step(step: Optional[str]) -> Optional[str]:
    if not step:
        return None
    step = step.lower()
    if step not in STEPS:
        raise ValidationError(
            f"paso desconocido '{step}'. Pasos válidos: {', '.join(STEPS)}", where="pipeline"
        )
    if step not in IMPLEMENTED:
        raise NotImplementedError(
            f"el paso '{step}' todavía no está implementado (H2–H6). "
            f"Disponibles ahora: {', '.join(IMPLEMENTED)}"
        )
    return step


def _count_statuses(results: Iterable[Mapping[str, Any]]) -> dict:
    counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _run_ingest(cfg: Mapping[str, Any], entries: list, force: bool) -> dict:
    update_step("ingest", status="running", started_at=utc_now())
    results = [loader.run_ingest(cfg, entry, force=force) for entry in entries]
    counts = _count_statuses(results)
    update_step(
        "ingest",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=[result["subject"] for result in results],
    )
    writeLog("info", logger, f"[INGEST] canónicos: {counts}")
    return {"status": "done", "results": counts}


def _run_qc(cfg: Mapping[str, Any], subjects: list[str], force: bool) -> dict:
    update_step("qc", status="running", started_at=utc_now())
    results = [qc.run_qc(cfg, subject, force=force) for subject in subjects]
    counts = _count_statuses(results)
    update_step(
        "qc",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=subjects,
        n_flags=sum(int(result.get("n_flags", 0)) for result in results),
    )
    writeLog("info", logger, f"[QC] informes: {counts}")
    return {"status": "done", "results": counts}


def _run_window(cfg: Mapping[str, Any], subjects: list[str], force: bool) -> dict:
    update_step("window", status="running", started_at=utc_now())
    results = [window.run_window(cfg, subject, force=force) for subject in subjects]
    counts = _count_statuses(results)
    global_report = window.build_global_index(cfg, subjects)
    update_step(
        "window",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=subjects,
        n_windows=global_report["n_windows"],
        n_by_label=global_report["n_by_label"],
    )
    writeLog("info", logger, f"[WINDOW] ventanas: {counts}, total={global_report['n_windows']}")
    return {"status": "done", "results": counts, "n_windows": global_report["n_windows"]}


def _run_degrade(
    cfg: Mapping[str, Any],
    subjects: list[str],
    force: bool,
) -> dict:
    update_step("degrade", status="running", started_at=utc_now())
    per_subject_results = [
        degrade.run_degrade(cfg, subject, force=force) for subject in subjects
    ]
    counts = _count_statuses(per_subject_results)
    update_step(
        "degrade",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=subjects,
        n_conditions=sum(int(r.get("n_conditions", 0)) for r in per_subject_results),
    )
    writeLog("info", logger, f"[DEGRADE] condiciones: {counts}")
    return {
        "status": "done",
        "results": counts,
        "per_subject": per_subject_results,
    }


def _run_features(
    cfg: Mapping[str, Any],
    subjects: list[str],
    force: bool,
    condition_id: str | None = None,
) -> dict:
    condition_label = condition_id or "canónico"
    update_step(
        "features",
        status="running",
        started_at=utc_now(),
        condition_id=condition_id,
    )
    results = [
        features.run_features(cfg, subject, force=force, condition_id=condition_id)
        for subject in subjects
    ]
    counts = _count_statuses(results)
    # Índice global solo para el canónico (condition_id None)
    if condition_id is None:
        global_report = features.build_global_index(cfg, subjects)
        n_windows = global_report["n_windows"]
        n_feature_values = global_report["n_feature_values"]
        n_features_per_window = global_report["n_features_per_window"]
        sqi_mean = global_report["sqi_mean"]
    else:
        # Para condiciones degradadas, agregar resultados individuales
        n_windows = sum(int(r.get("n_windows", 0)) for r in results)
        n_feature_values = sum(int(r.get("n_feature_values", 0)) for r in results)
        n_features_per_window = (
            results[0].get("n_feature_values", 0) // max(1, results[0].get("n_windows", 1))
            if results
            else 0
        )
        sqi_values = [
            float(r.get("sqi_mean", 0)) for r in results if r.get("sqi_mean") is not None
        ]
        sqi_mean = float(np.mean(sqi_values)) if sqi_values else None
        global_report = None

    update_step(
        "features",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=subjects,
        condition_id=condition_id,
        n_windows=n_windows,
        n_feature_values=n_feature_values,
        n_features_per_window=n_features_per_window,
        sqi_mean=sqi_mean,
    )
    writeLog(
        "info",
        logger,
        f"[FEATURES] extracción{condition_label}: {counts}, "
        f"{n_feature_values} valores de feature",
    )
    return {
        "status": "done",
        "results": counts,
        "n_feature_values": n_feature_values,
        "condition_id": condition_id,
    }


def _run_represent(cfg: Mapping[str, Any], subjects: list[str], force: bool) -> dict:
    update_step("represent", status="running", started_at=utc_now())
    results = [represent.run_represent(cfg, subject, force=force) for subject in subjects]
    global_report = represent.build_global_index(cfg, subjects)
    counts: dict[str, int] = {}
    for result in results:
        for kind, status in result.get("status", {}).items():
            key = f"{kind}:{status}"
            counts[key] = counts.get(key, 0) + 1
    update_step(
        "represent",
        status="done",
        finished_at=utc_now(),
        results=counts,
        subjects=subjects,
        n_windows={kind: report["n_windows"] for kind, report in global_report.items()},
    )
    writeLog("info", logger, f"[REPRESENT] representaciones: {counts}")
    return {
        "status": "done",
        "results": counts,
        "n_windows": {kind: report["n_windows"] for kind, report in global_report.items()},
    }


def _run_package(cfg: Mapping[str, Any], subjects: list[str], force: bool) -> dict:
    update_step("package", status="running", started_at=utc_now())
    result = package.run_package(cfg, subjects, force=force)
    update_step(
        "package",
        status="done",
        finished_at=utc_now(),
        result=result["status"],
        fingerprint=result["fingerprint"],
        corpus_version=result["corpus_version"],
        n_windows=result["n_windows"],
        n_artifacts=result["n_artifacts"],
    )
    writeLog("info", logger, f"[PACKAGE] corpus '{result['corpus_version']}' listo: {result['manifest']}")
    return result


def process_ingest() -> None:
    """Punto de entrada del proceso ``INGEST``."""
    cfg = get_ingestion_config()
    requested = _resolve_requested(cfg)

    writeLog("info", logger, "-" * 60)
    writeLog(
        "info",
        logger,
        f"🚀 [INGEST] dataset={cfg['dataset']} subject={_arg('subject')} "
        f"step={_arg('step') or 'auto'} force={bool(_arg('force'))}",
    )

    if bool(_arg("dry_run")):
        _dry_run(cfg, requested)
        return

    requested_step = _check_step(_arg("step"))
    target_index = STEPS.index(requested_step) if requested_step else len(IMPLEMENTED) - 1
    force = bool(_arg("force"))
    snapshot = environment_snapshot()

    # `discover` es prerequisito de todo el pipeline y barato gracias a la
    # caché de hashes.
    update_step("discover", status="running", started_at=utc_now())
    manifest = discover.run_discover(cfg, force_hash=force, requested=requested)
    update_step(
        "discover",
        status="done",
        finished_at=utc_now(),
        fingerprint=manifest["config_fingerprint"],
        n_subjects=len(manifest["subjects"]),
        n_selected=sum(1 for entry in manifest["subjects"] if entry["selected"]),
    )

    entries = discover.selected_entries(manifest)
    if not entries:
        raise ValidationError("no hay sujetos seleccionados para procesar", where="pipeline")
    subjects = [entry.subject for entry in entries]

    if target_index >= 1:
        _run_ingest(cfg, entries, force)
    if target_index >= 2:
        _run_qc(cfg, subjects, force)
    if target_index >= 3:
        _run_window(cfg, subjects, force)
    if target_index >= 4:
        degrade_result = _run_degrade(cfg, subjects, force)
        # Ejecutar features para cada condición degradada (solo si se pide
        # features o más, es decir, target_index >= 5)
        if target_index >= 5:
            exp_cfg = cfg.get("experiment", {})
            if bool(exp_cfg.get("degradation_enabled", False)):
                for subject_result in degrade_result.get("per_subject", []):
                    for cond in subject_result.get("conditions", []):
                        cond_id = cond.get("condition_id")
                        if cond_id:
                            _run_features(
                                cfg, subjects, force, condition_id=cond_id
                            )
    if target_index >= 5:
        _run_features(cfg, subjects, force)  # canónico (condition_id=None)
    if target_index >= 6:
        _run_represent(cfg, subjects, force)
    if target_index >= 7:
        _run_package(cfg, subjects, force)

    if requested_step is None:
        writeLog(
            "info",
            logger,
            "[INGEST] Fase 1 completa: canónico, QC, ventanas, features, "
            "representaciones y corpus empaquetado.",
        )

    writeLog("info", logger, f"✅ [INGEST] manifest: {discover.manifest_path()}")
    writeLog("debug", logger, f"entorno: {snapshot['versions']}")
