"""Piloto DREAMT: estadios de sueño bajo degradación controlada (segunda tarea).

Tres fases, separables para poder reanudar y auditar:

``prepare``
    Carga las noches, construye la rejilla de épocas de 30 s, selecciona las
    ventanas puras de 60 s, calcula las features absolutas de **todas** las
    ventanas puras (referencia intra-sujeto), muestrea las ventanas del piloto,
    aplica las condiciones de degradación y renderiza los prompts numérico y
    textual. Incluye el pre-chequeo **P0** de la tabla de firma del sueño sobre
    los sujetos de desarrollo.

``generate``
    Envía los prompts a Ollama y guarda la respuesta con procedencia completa.
    Reanudable por ``(ventana, condición, representación)``.

``evaluate``
    Corrección, cobertura, coherencia de claims, respaldo y certeza por
    condición y representación, más una referencia ``features-only`` por
    validación cruzada agrupada por sujeto.

Todo el detalle del diseño está congelado en ``docs/piloto_dreamt.md``.

Ejemplos::

    python scripts/run_p4_dreamt.py --phase prepare
    python scripts/run_p4_dreamt.py --phase generate --representation numeric
    python scripts/run_p4_dreamt.py --phase evaluate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sources.common.llm_client import check_model_available, ollama_run_provenance
from sources.external import dreamt
from sources.external.paths import DREAMT_DIR
from sources.ingestion.config import get_ingestion_config
from sources.phase2.llm_config import (
    build_experiment_config,
    create_ollama_chat_client,
    model_slug,
)
from sources.phase2.orchestrator import _call_ollama_chat
from sources.phase2.prompt_builder import (
    SLEEP_FEATURES,
    SLEEP_FEW_SHOT_REASONING,
    assert_no_leakage,
    build_numeric_prompt,
    build_text_prompt,
    render_sleep_few_shot_examples,
)
from sources.phase2.response_schema import (
    DECISION_MODE_SLEEP4,
    validate_llm_output,
)
from sources.phase3.certainty import linguistic_certainty
from sources.phase3.claims import (
    response_coherence,
    summarize_coherence,
    summarize_verdicts,
    verify_claims,
)
from sources.phase3.records import evaluate_records
from sources.phase3.signatures import (
    SLEEP_CLASSES_WITH_SIGNATURE,
    SLEEP_SIGNATURE_RATIONALE,
    SLEEP_SIGNATURES,
)

#: Sujetos del piloto (preregistro §7): reservados como evaluación, nunca
#: usados para elegir features, umbrales ni ejemplares.
PILOT_SUBJECTS: tuple[str, ...] = ("S034", "S051", "S076", "S084")

#: Sujetos de desarrollo de la línea DREAMT: solo para few-shot y P0.
#:
#: No se reutiliza ``development_split.DEV_SUBJECTS`` porque ese split es de
#: WESAD (``S2``/``S3``/``S4``) y no existe en DREAMT. Los tres sujetos se
#: reservaron en el preregistro §7 y no se usan para evaluar.
DREAMT_DEV_SUBJECTS: tuple[str, ...] = ("S002", "S010", "S026")

#: Orden de clases del espacio de decisión (N3 excluido).
CLASS_ORDER: tuple[str, ...] = ("wake", "n1", "n2", "rem")

#: Canales admitidos en los claims de esta tarea (el acelerómetro es evidencia
#: legítima aquí; en WESAD era un confusor de movimiento).
SLEEP_CLAIM_CHANNELS: frozenset[str] = frozenset({"ecg", "eda", "temp", "acc"})

DATASET_NOTE: str = (
    "Dataset note: signals come from one wrist-worn device over a single "
    "night. Respiration and muscle activity are not available. Values are "
    "relative to this subject's own night, not to other people."
)

#: Features que se muestran en los ejemplos few-shot. Se limitan a los tres
#: mecanismos con firma más el movimiento fino, para que el ejemplo ilustre un
#: patrón y no una tabla.
FEW_SHOT_FEATURES: tuple[str, ...] = (
    "acc_std_g",
    "ecg_hr_bpm",
    "eda_std_us",
    "acc_motion_fraction",
)

DEFAULT_OUTPUT = REPO_ROOT / "results" / "output" / "DREAMT" / "PHASE2" / "pilot"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Marca temporal en UTC (segundos)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee un JSONL tolerando la última línea incompleta (proceso cortado)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Añade una línea y la vuelca a disco (reanudable sin perder trabajo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    """Escribe JSON con indentación (informes y manifiestos)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def feature_names() -> list[str]:
    """Vocabulario de features de la tarea (las claves del payload)."""
    keys = dreamt._canonical_keys()
    return [name for names in keys.values() for name in names]


def available_subjects(data_dir: Path) -> list[str]:
    """Sujetos cuyo CSV existe en el directorio de datos."""
    present = []
    for subject in dreamt.list_subjects(data_dir):
        if dreamt.subject_path(subject, data_dir).is_file():
            present.append(subject)
    return present


# ---------------------------------------------------------------------------
# Fase 1: prepare
# ---------------------------------------------------------------------------


def night_rows(
    subject: str,
    data_dir: Path,
    cache_dir: Path,
    cfg: Mapping[str, Any],
    force: bool = False,
) -> list[dict[str, Any]]:
    """Features absolutas de **todas** las ventanas puras de una noche.

    Se cachean: la referencia intra-sujeto y el pre-chequeo P0 las necesitan y
    recalcularlas cuesta ~15 ms por ventana.
    """
    path = cache_dir / f"{subject}.night.jsonl"
    if path.is_file() and not force:
        rows = read_jsonl(path)
        if rows:
            return rows

    frame = dreamt.load_subject(subject, data_dir)
    grid = dreamt.epoch_grid(dreamt.stage_array(frame))
    windows = dreamt.pure_windows(subject, grid)
    rows = []
    started = time.time()
    for window in windows:
        arrays = dreamt.window_arrays(frame, window)
        features, notes = dreamt.extract_features(arrays, cfg)
        rows.append(
            {
                "window_id": window.window_id,
                "stage": window.stage,
                "epoch_index": window.epoch_index,
                "night_frac": round(float(window.night_frac), 4),
                "flat": dreamt.flatten_features(features),
                "notes": notes,
            }
        )
    write_jsonl_atomic(path, rows)
    print(
        f"[DREAMT] {subject}: {len(rows)} ventanas puras "
        f"({time.time() - started:.1f}s) {dict(Counter(r['stage'] for r in rows))}"
    )
    return rows


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Escritura atómica de un JSONL completo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare_subject(
    subject: str,
    data_dir: Path,
    output_dir: Path,
    cfg: Mapping[str, Any],
    reference: tuple[dict[str, float], dict[str, float]],
    few_shot_text: str,
    windows_per_class: int,
    max_windows: int | None,
    conditions: Sequence[str],
    force: bool,
) -> dict[str, Any]:
    """Muestrea ventanas y renderiza los prompts de un sujeto del piloto."""
    rows = read_jsonl(output_dir / "features" / f"{subject}.night.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"faltan las features de la noche de {subject}; ejecuta --phase prepare"
        )

    frame = dreamt.load_subject(subject, data_dir)
    grid = dreamt.epoch_grid(dreamt.stage_array(frame))
    windows = dreamt.pure_windows(subject, grid)
    sampled = dreamt.sample_windows(windows, per_class=windows_per_class)
    if max_windows is not None:
        sampled = sampled[:max_windows]

    by_id = {row["window_id"]: row for row in rows}
    centers, scales = reference
    prompts_path = output_dir / "prompts" / f"{subject}.prompts.jsonl"
    labels_path = output_dir / "labels" / f"{subject}.labels.jsonl"
    if prompts_path.is_file() and not force:
        print(f"[DREAMT] {subject}: prompts ya generados, se reutilizan")
        return {"subject": subject, "reused": True}

    prompt_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    started = time.time()

    for window in sampled:
        night_row = by_id.get(window.window_id)
        if night_row is None:
            continue
        arrays = dreamt.window_arrays(frame, window)
        base_features, _ = dreamt.extract_features(arrays, cfg)
        base_flat = night_row["flat"]

        labels = {
            "window_id": window.window_id,
            "stage": window.stage,
            "klass": dreamt.CLASS_OF_STAGE.get(window.stage, window.stage.lower()),
            "epoch_index": window.epoch_index,
            "night_frac": round(float(window.night_frac), 4),
        }
        label_rows.append(labels)

        for condition in conditions:
            if condition == dreamt.NC_ABSENT:
                flat = dreamt.flatten_features(
                    dreamt.apply_feature_condition(base_features, dreamt.NC_ABSENT)
                )
            elif condition == dreamt.C0:
                flat = base_flat
            else:
                degraded = dreamt.apply_array_condition(arrays, condition)
                features, _notes = dreamt.extract_features(degraded, cfg)
                flat = dreamt.flatten_features(features)

            zvalues = dreamt.z_from_reference(flat, centers, scales)
            payload = dreamt.build_sleep_payload(window.window_id, zvalues)
            opaque = payload["window"]["window_id"]
            numeric_prompt = build_numeric_prompt(
                payload,
                opaque,
                dreamt.WINDOW_S,
                use_few_shot=True,
                standardized=True,
                decision_mode=DECISION_MODE_SLEEP4,
                dataset_note=DATASET_NOTE,
                extra_examples=few_shot_text,
            )
            text_prompt = build_text_prompt(
                dreamt.render_sleep_text(zvalues),
                opaque,
                dreamt.WINDOW_S,
                use_few_shot=True,
                standardized=True,
                decision_mode=DECISION_MODE_SLEEP4,
                dataset_note=DATASET_NOTE,
                extra_examples=few_shot_text,
            )
            assert_no_leakage(numeric_prompt)
            assert_no_leakage(text_prompt)
            prompt_rows.append(
                {
                    "window_id": window.window_id,
                    "window_id_opaque": opaque,
                    "condition_id": condition,
                    "stage": window.stage,
                    "night_frac": round(float(window.night_frac), 4),
                    "prompts": {"numeric": numeric_prompt, "text": text_prompt},
                }
            )

    write_jsonl_atomic(prompts_path, prompt_rows)
    write_jsonl_atomic(labels_path, label_rows)
    print(
        f"[DREAMT] {subject}: {len(sampled)} ventanas × {len(conditions)} "
        f"condiciones = {len(prompt_rows)} prompts ({time.time() - started:.1f}s)"
    )
    return {
        "subject": subject,
        "n_windows": len(sampled),
        "n_prompts": len(prompt_rows),
        "stage_counts": dict(Counter(r["stage"] for r in label_rows)),
        "reused": False,
    }


def reference_from_dev(
    dev_subjects: Sequence[str],
    output_dir: Path,
    names: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Referencia intra-sujeto (mediana/IQR) de cada noche.

    Se calcula con **todas** las ventanas puras de la noche y no con la muestra
    del piloto: si la muestra definiera su propia referencia, la degradación
    quedaría re-normalizada y se perdería el efecto.
    """
    centers: dict[str, dict[str, float]] = {}
    scales: dict[str, dict[str, float]] = {}
    for subject in dev_subjects:
        rows = read_jsonl(output_dir / "features" / f"{subject}.night.jsonl")
        if not rows:
            continue
        subject_centers, subject_scales = dreamt.night_reference(rows, names)
        centers[subject] = subject_centers
        scales[subject] = subject_scales
        write_json(
            output_dir / "reference" / f"{subject}.json",
            {
                "centers": subject_centers,
                "scales": subject_scales,
                "n_windows": len(rows),
            },
        )
    return centers, scales


def few_shot_from_dev(
    dev_subjects: Sequence[str],
    output_dir: Path,
    centers: Mapping[str, Mapping[str, float]],
    scales: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Mediana por clase de las z de los sujetos de desarrollo.

    Es la traducción honesta de los ``FEW_SHOT_EXEMPLARS`` de WESAD: los
    ejemplares salen **solo** de sujetos de desarrollo, nunca de las ventanas
    que se van a evaluar.
    """
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for subject in dev_subjects:
        rows = read_jsonl(output_dir / "features" / f"{subject}.night.jsonl")
        subject_centers = centers.get(subject)
        subject_scales = scales.get(subject)
        if not rows or subject_centers is None or subject_scales is None:
            continue
        for row in rows:
            klass = dreamt.CLASS_OF_STAGE.get(row["stage"], row["stage"].lower())
            zvalues = dreamt.z_from_reference(row["flat"], subject_centers, subject_scales)
            for name in FEW_SHOT_FEATURES:
                value = zvalues.get(name)
                if value is not None:
                    collected[klass][name].append(float(value))

    exemplars: dict[str, dict[str, float]] = {}
    for klass in CLASS_ORDER:
        values = collected.get(klass, {})
        exemplars[klass] = {
            name: float(np.median(values[name]))
            for name in FEW_SHOT_FEATURES
            if values.get(name)
        }
    provenance = {
        "subjects": list(dev_subjects),
        "statistic": "median of within-subject z over all pure windows",
        "features": list(FEW_SHOT_FEATURES),
        "exemplars": exemplars,
    }
    write_json(output_dir / "few_shot.json", provenance)
    return exemplars


def precheck_p0(
    dev_subjects: Sequence[str],
    output_dir: Path,
    centers: Mapping[str, Mapping[str, float]],
    scales: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """P0: ¿las medianas por clase respetan el signo de la tabla de firma?

    Se ejecuta **antes** de gastar una sola llamada al LLM. Es el falsificador
    F1 del preregistro: si la tabla no describe los datos limpios en los
    sujetos de desarrollo, la tasa de coherencia no sería interpretable.
    """
    collected: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for subject in dev_subjects:
        rows = read_jsonl(output_dir / "features" / f"{subject}.night.jsonl")
        subject_centers = centers.get(subject)
        subject_scales = scales.get(subject)
        if not rows or subject_centers is None or subject_scales is None:
            continue
        for row in rows:
            klass = dreamt.CLASS_OF_STAGE.get(row["stage"], row["stage"].lower())
            zvalues = dreamt.z_from_reference(row["flat"], subject_centers, subject_scales)
            for name in SLEEP_SIGNATURES.get(klass, {}):
                value = zvalues.get(name)
                if value is not None:
                    collected[klass][name].append(float(value))

    checks = []
    failures = 0
    for klass, table in SLEEP_SIGNATURES.items():
        for metric, expected in table.items():
            values = collected.get(klass, {}).get(metric, [])
            median = float(np.median(values)) if values else None
            passed = median is not None and np.sign(median) == np.sign(expected)
            if not passed:
                failures += 1
            checks.append(
                {
                    "class": klass,
                    "metric": metric,
                    "expected_sign": int(np.sign(expected)),
                    "median_z": None if median is None else round(median, 3),
                    "n_windows": len(values),
                    "passed": bool(passed),
                    "rationale": SLEEP_SIGNATURE_RATIONALE.get(metric, ""),
                }
            )

    total = len(checks)
    report = {
        "n_checks": total,
        "n_failures": failures,
        "pass_rate": (total - failures) / total if total else float("nan"),
        "dev_subjects": list(dev_subjects),
        "checks": checks,
        "verdict": "pass" if failures < 2 else "fail",
    }
    write_json(output_dir / "p0.json", report)
    print(f"[DREAMT] P0: {total - failures}/{total} comprobaciones OK ({report['verdict']})")
    for check in checks:
        mark = "ok " if check["passed"] else "FALLA"
        print(
            f"         {mark} {check['class']:5} {check['metric']:18} "
            f"esperado {check['expected_sign']:+d} mediana {check['median_z']}"
        )
    return report


def phase_prepare(args: argparse.Namespace) -> int:
    """Fase 1: rejilla, features de la noche, muestra, degradación y prompts."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    cfg = get_ingestion_config()

    present = set(available_subjects(data_dir))
    dev = [s for s in DREAMT_DEV_SUBJECTS if s in present]
    pilot = [s for s in PILOT_SUBJECTS if s in present]
    missing = [s for s in PILOT_SUBJECTS if s not in present]
    if missing:
        print(f"[DREAMT] Aviso: sujetos del piloto sin datos: {missing}")
    if not pilot:
        raise SystemExit(
            f"[DREAMT] Ningún sujeto del piloto está en {data_dir}. "
            "Comprueba --data-dir (results/input/dreamt/data_64Hz)."
        )
    if not dev:
        raise SystemExit(
            f"[DREAMT] Faltan los sujetos de desarrollo {DREAMT_DEV_SUBJECTS} en "
            f"{data_dir}: sin ellos no hay ejemplos few-shot ni pre-chequeo P0."
        )
    print(f"[DREAMT] desarrollo {dev} | piloto {pilot}")

    names = feature_names()
    for subject in dev + pilot:
        night_rows(subject, data_dir, output_dir / "features", cfg, force=args.force)

    centers: dict[str, dict[str, float]] = {}
    scales: dict[str, dict[str, float]] = {}
    for subject in dev + pilot:
        rows = read_jsonl(output_dir / "features" / f"{subject}.night.jsonl")
        subject_centers, subject_scales = dreamt.night_reference(rows, names)
        centers[subject] = subject_centers
        scales[subject] = subject_scales
        write_json(
            output_dir / "reference" / f"{subject}.json",
            {
                "centers": subject_centers,
                "scales": subject_scales,
                "n_windows": len(rows),
            },
        )

    exemplars = few_shot_from_dev(dev, output_dir, centers, scales)
    p0 = precheck_p0(dev, output_dir, centers, scales)
    if p0["verdict"] == "fail" and not args.allow_failed_p0:
        print(
            "[DREAMT] P0 falla: la tabla de firma no describe los datos limpios de "
            "desarrollo. No se generan prompts (falsificador F1).\n"
            "         Revisa docs/piloto_dreamt.md §9 antes de continuar; para "
            "forzar de todos modos usa --allow-failed-p0."
        )
        return 4

    few_shot_text = render_sleep_few_shot_examples(exemplars, SLEEP_FEW_SHOT_REASONING)
    conditions = dreamt.PILOT_CONDITIONS if not args.conditions else tuple(
        part.strip() for part in args.conditions.split(",") if part.strip()
    )

    summary = []
    for subject in pilot:
        summary.append(
            prepare_subject(
                subject,
                data_dir,
                output_dir,
                cfg,
                (centers[subject], scales[subject]),
                few_shot_text,
                windows_per_class=args.windows_per_class,
                max_windows=args.max_windows,
                conditions=conditions,
                force=args.force,
            )
        )

    manifest = {
        "task": "sleep4",
        "dataset": "DREAMT (wearable branch, data_64Hz)",
        "generated_at": utc_now(),
        "dev_subjects": dev,
        "pilot_subjects": pilot,
        "conditions": list(conditions),
        "classes": list(CLASS_ORDER),
        "n3_excluded": True,
        "window_s": dreamt.WINDOW_S,
        "windows_per_class": args.windows_per_class,
        "features_extracted": names,
        "features_llm_view": list(SLEEP_FEATURES),
        "few_shot_provenance": {
            "subjects": dev,
            "features": list(FEW_SHOT_FEATURES),
        },
        "p0_verdict": p0["verdict"],
        "p0_failures": p0["n_failures"],
        "subjects": summary,
        "revisions": {
            "cardiac_block": (
                "device HR (1 Hz); sin HRV porque IBI está cuantizada con dither y "
                "los picos de BVP son inestables (medido; desviación D7)"
            ),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    total_prompts = sum(item.get("n_prompts", 0) for item in summary)
    print(
        f"[DREAMT] prepare listo: {total_prompts} prompts por representación "
        f"({total_prompts * 2} llamadas si se ejecutan las dos)"
    )
    return 0


# ---------------------------------------------------------------------------
# Fase 2: generate
# ---------------------------------------------------------------------------


def phase_generate(args: argparse.Namespace) -> int:
    """Fase 2: llamadas al LLM con reanudación por (ventana, condición, vista)."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    representations = (
        ["numeric", "text"] if args.representation == "both" else [args.representation]
    )

    config = build_experiment_config(
        model=args.model,
        seed=args.seed,
        max_retries=args.max_retries,
        think=args.think,
    )
    slug = model_slug(config.model)
    responses_dir = output_dir / "responses" / slug
    responses_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and check_model_available(config.model) is False:
        raise SystemExit(
            f"[DREAMT] El modelo {config.model!r} no está instalado en Ollama.\n"
            f"         Instálalo con: ollama pull {config.model}"
        )

    provenance = ollama_run_provenance(config.model)
    provenance["config"] = config.to_dict()
    provenance["started_at"] = utc_now()
    write_json(output_dir / "run_provenance.json", provenance)

    present = set(available_subjects(data_dir))
    subjects = [s for s in PILOT_SUBJECTS if s in present]
    client = create_ollama_chat_client(config) if not args.dry_run else None

    planned = 0
    done = 0
    errors = 0
    for subject in subjects:
        prompt_rows = read_jsonl(output_dir / "prompts" / f"{subject}.prompts.jsonl")
        if not prompt_rows:
            print(f"[DREAMT] {subject}: sin prompts; se omite")
            continue
        for representation in representations:
            path = responses_dir / f"{subject}.{representation}.jsonl"
            existing = {
                (row["window_id"], row["condition_id"])
                for row in read_jsonl(path)
                if row.get("status") not in ("timeout",)
            }
            pending = [
                row for row in prompt_rows if (row["window_id"], row["condition_id"]) not in existing
            ]
            planned += len(pending)
            if args.dry_run:
                print(
                    f"[DREAMT] dry-run {subject} {representation}: "
                    f"{len(pending)} llamadas pendientes de {len(prompt_rows)}"
                )
                continue
            print(
                f"[DREAMT] {subject} {representation}: {len(pending)} pendientes "
                f"({len(existing)} ya hechas)"
            )
            for index, row in enumerate(pending, start=1):
                prompt = row["prompts"][representation]
                try:
                    raw_output, meta = _call_ollama_chat(
                        client, prompt, config, seed=config.seed
                    )
                    validated = validate_llm_output(
                        raw_output,
                        decision_mode=DECISION_MODE_SLEEP4,
                        allowed_channels=SLEEP_CLAIM_CHANNELS,
                    )
                    status = validated.status
                    decision = validated.decision
                    confidence = validated.confidence
                    claims = [claim.to_dict() for claim in validated.claims]
                    reasoning = validated.reasoning
                    error = validated.error_message
                    if status == "invalid_output":
                        errors += 1
                except Exception as exc:  # noqa: BLE001 - una llamada no corta el barrido
                    raw_output, meta = "", {}
                    status, decision, confidence = "timeout", None, None
                    claims, reasoning, error = [], "", str(exc)
                    errors += 1

                record = {
                    "window_id": row["window_id"],
                    "window_id_opaque": row["window_id_opaque"],
                    "condition_id": row["condition_id"],
                    "representation": representation,
                    "decision_mode": DECISION_MODE_SLEEP4,
                    "model": config.model,
                    "model_slug": slug,
                    "status": status,
                    "decision": decision,
                    "confidence": confidence,
                    "claims": claims,
                    "reasoning": reasoning,
                    "error_message": error,
                    "raw_output": raw_output,
                    "prompt": prompt,
                    "ollama_meta": meta,
                    "generated_at": utc_now(),
                }
                append_jsonl(path, record)
                done += 1
                print(
                    f"[DREAMT] {subject} {representation} {index}/{len(pending)} "
                    f"{row['condition_id']} → {status} {decision} ({confidence})"
                )

    if args.dry_run:
        print(f"[DREAMT] dry-run: {planned} llamadas pendientes en total")
        return 0

    summary = {
        "phase": "generate",
        "model": config.model,
        "model_slug": slug,
        "representations": representations,
        "subjects": subjects,
        "n_calls_new": done,
        "n_errors": errors,
        "config": config.to_dict(),
        "finished_at": utc_now(),
    }
    write_json(output_dir / "generate_summary.json", summary)
    print(f"[DREAMT] generate: {done} llamadas nuevas, {errors} errores/inválidas")
    return 0


# ---------------------------------------------------------------------------
# Fase 3: evaluate
# ---------------------------------------------------------------------------


def _load_labels(output_dir: Path, subjects: Sequence[str]) -> dict[str, str]:
    """``window_id_opaque → clase`` construido desde el payload publicado.

    El identificador opaco se recalcula igual que en ``prepare`` (hash estable
    de ``window_id``), así que la unión con la verdad de campo es exacta sin
    guardar la etiqueta junto al prompt.
    """
    from sources.ingestion.represent import _opaque_window_id

    labels: dict[str, str] = {}
    for subject in subjects:
        for row in read_jsonl(output_dir / "labels" / f"{subject}.labels.jsonl"):
            labels[_opaque_window_id(row["window_id"])] = row["klass"]
    return labels


def _condition_block(
    records: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    """Métricas de corrección, abstención, claims y certeza de una condición."""
    metrics = evaluate_records(list(records), labels, label_order=CLASS_ORDER)
    metrics["coverage"] = (
        metrics["n_valid"] - metrics["n_abstain"]
    ) / metrics["n"] if metrics["n"] else 0.0
    metrics["accuracy_incl_abstain"] = (
        metrics["accuracy"] * metrics["n_decided"] / metrics["n"]
        if metrics["n"]
        else 0.0
    )

    # Recall por clase real (el evaluador devuelve la matriz, no el recall):
    # es lo único que revela si el modelo colapsa a `n2` (la clase mayoritaria).
    recall_by_class: dict[str, float] = {}
    confusion = metrics.get("confusion", {})
    for label in CLASS_ORDER:
        row = confusion.get(label, {})
        decided = sum(row.get(predicted, 0) for predicted in CLASS_ORDER)
        if decided:
            recall_by_class[label] = row.get(label, 0) / decided
    metrics["recall_by_class"] = recall_by_class

    # H2a: ¿la confianza separa aciertos de errores? Con una confianza casi
    # constante el AUROC depende solo de la dirección y hay que decirlo.
    from sources.phase3.calibration import auroc_from_scores

    decided = [
        record
        for record in records
        if record.get("status") in ("valid_response", "valid_abstention")
        and record.get("decision") not in (None, "", "abstain")
    ]
    conf_scores = [float(record.get("confidence") or 0) / 100.0 for record in decided]
    conf_hits = [
        1.0 if record.get("decision") == labels.get(record.get("window_id_opaque", "")) else 0.0
        for record in decided
    ]
    metrics["confidence_auroc"] = auroc_from_scores(conf_hits, conf_scores)
    confidence_counts: Counter[str] = Counter(
        f"{round(score * 100, 1):g}" for score in conf_scores
    )
    metrics["confidence_values"] = dict(sorted(confidence_counts.items(), key=lambda kv: float(kv[0])))
    metrics["overconfidence"] = (
        (sum(conf_scores) / len(conf_scores) - metrics["accuracy"])
        if conf_scores
        else 0.0
    )

    verdicts = []
    claims_blocks = []
    certainties = []
    truth_counts: Counter[str] = Counter()
    for record in records:
        truth_counts[labels.get(record.get("window_id_opaque", ""), "UNKNOWN")] += 1
        if record.get("status") not in ("valid_response", "valid_abstention"):
            continue
        record_verdicts = []
        if record.get("claims"):
            record_verdicts = verify_claims(
                record["claims"], record["prompt"], extra_aliases=dreamt.SLEEP_METRIC_ALIASES
            )
            verdicts.extend(record_verdicts)
            claims_blocks.append(
                response_coherence(
                    record["claims"],
                    record_verdicts,
                    record.get("decision"),
                    signatures=SLEEP_SIGNATURES,
                    extra_aliases=dreamt.SLEEP_METRIC_ALIASES,
                    subject=str(record.get("window_id", "")).split(":")[0],
                )
            )
        certainty = linguistic_certainty(record.get("reasoning", "") or "")
        if certainty["n_words"]:
            certainties.append(certainty)

    metrics["claims"] = summarize_verdicts(verdicts) if verdicts else {"n_claims": 0}
    metrics["coherence"] = (
        summarize_coherence(
            [pair for block in claims_blocks for pair in block.get("pairs", [])],
            responses=claims_blocks,
            classes=SLEEP_CLASSES_WITH_SIGNATURE,
        )
        if claims_blocks
        else {}
    )
    metrics["certainty"] = {
        "n": len(certainties),
        "median_index": (
            float(np.median([c["certainty_index"] for c in certainties]))
            if certainties
            else float("nan")
        ),
        "median_hedges": (
            float(np.median([c["hedges"] for c in certainties])) if certainties else float("nan")
        ),
        "median_boosters": (
            float(np.median([c["boosters"] for c in certainties])) if certainties else float("nan")
        ),
    }
    metrics["truth_counts"] = dict(truth_counts)
    return metrics


def _reference_classifier(
    output_dir: Path,
    subjects: Sequence[str],
    condition: str = dreamt.C0,
) -> dict[str, Any]:
    """Referencia ``features-only`` con validación cruzada agrupada por sujeto.

    Usa las mismas features que ve el LLM en la condición limpia. No es un
    resultado del LLM: sirve para saber si la tarea es resoluble con esas
    features y cuánto pierde el modelo respecto del techo supervisado.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}

    if condition != dreamt.C0:
        return {
            "available": False,
            "error": "la referencia solo se calcula sobre C0 (features sin degradar)",
        }

    names = list(SLEEP_FEATURES)
    X, y, groups = [], [], []
    for subject in subjects:
        reference_path = output_dir / "reference" / f"{subject}.json"
        if not reference_path.is_file():
            continue
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        centers, scales = reference["centers"], reference["scales"]
        for row in read_jsonl(output_dir / "features" / f"{subject}.night.jsonl"):
            klass = dreamt.CLASS_OF_STAGE.get(row["stage"], row["stage"].lower())
            if klass not in CLASS_ORDER:
                continue
            # Misma estandarización intra-sujeto que recibe el LLM: si la
            # referencia usara valores absolutos, el clasificador podría
            # apoyarse en el nivel individual del sujeto y no en la ventana.
            zvalues = dreamt.z_from_reference(row["flat"], centers, scales)
            values = [zvalues.get(name) for name in names]
            if any(value is None for value in values):
                continue
            X.append(values)
            y.append(klass)
            groups.append(subject)
    if len(set(y)) < 2 or len(set(groups)) < 2:
        return {"available": False, "error": "muestra insuficiente"}

    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    groups_array = np.asarray(groups)
    predictions = np.empty(len(y_array), dtype=object)
    folds = []
    for held_out in sorted(set(groups)):
        train = groups_array != held_out
        test = ~train
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        )
        model.fit(X_array[train], y_array[train])
        pred = model.predict(X_array[test])
        predictions[test] = pred
        folds.append(
            {
                "subject": held_out,
                "n": int(test.sum()),
                "accuracy": float(accuracy_score(y_array[test], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_array[test], pred)),
            }
        )
    return {
        "available": True,
        "condition": condition,
        "n_windows": int(len(y_array)),
        "n_features": len(names),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_array, list(predictions))
        ),
        "folds": folds,
    }


def phase_evaluate(args: argparse.Namespace) -> int:
    """Fase 3: métricas por condición y representación, y reporte."""
    output_dir = Path(args.output_dir)
    representations = (
        ["numeric", "text"] if args.representation == "both" else [args.representation]
    )
    slug = model_slug(args.model or "")
    responses_dir = output_dir / "responses" / slug

    present = set(available_subjects(Path(args.data_dir)))
    subjects = [s for s in PILOT_SUBJECTS if s in present]
    labels = _load_labels(output_dir, subjects)

    report: dict[str, Any] = {
        "task": "sleep4",
        "model": args.model,
        "model_slug": slug,
        "generated_at": utc_now(),
        "subjects": subjects,
        "classes": list(CLASS_ORDER),
        "by_representation": {},
    }

    for representation in representations:
        records = []
        for subject in subjects:
            records.extend(read_jsonl(responses_dir / f"{subject}.{representation}.jsonl"))
        if not records:
            print(f"[DREAMT] {representation}: sin respuestas en {responses_dir}")
            continue
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_condition[record["condition_id"]].append(record)
        blocks = {}
        for condition in dreamt.PILOT_CONDITIONS:
            if condition not in by_condition:
                continue
            blocks[condition] = _condition_block(by_condition[condition], labels)
        report["by_representation"][representation] = {
            "n_records": len(records),
            "conditions": blocks,
        }

    if args.reference:
        report["reference"] = _reference_classifier(output_dir, subjects)

    write_json(output_dir / f"evaluation_{slug}.json", report)
    _print_report(report)
    return 0


def _print_report(report: Mapping[str, Any]) -> None:
    """Tabla compacta por representación y condición."""
    print("\n[DREAMT] Resultados del piloto")
    print(
        f"{'vista':8} {'condición':20} {'n':>4} {'inv':>4} {'abs%':>6} "
        f"{'cov':>6} {'acc':>6} {'bal':>6} {'sop':>6} {'cert':>7}"
    )
    for representation, block in report.get("by_representation", {}).items():
        for condition, metrics in block["conditions"].items():
            claims = metrics.get("claims", {})
            certainty = metrics.get("certainty", {})
            print(
                f"{representation:8} {condition:20} {metrics['n']:4d} "
                f"{metrics['n_invalid']:4d} "
                f"{metrics['abstention_rate'] * 100:6.1f} "
                f"{metrics['coverage'] * 100:6.1f} "
                f"{metrics['accuracy']:6.3f} {metrics['balanced_accuracy']:6.3f} "
                f"{claims.get('supported_rate', float('nan')):6.3f} "
                f"{certainty.get('median_index', float('nan')):7.2f}"
            )
    for representation, block in report.get("by_representation", {}).items():
        clean = block["conditions"].get(dreamt.C0)
        if not clean:
            continue
        recall = clean.get("recall_by_class", {})
        if not recall:
            continue
        pretty = "  ".join(
            f"{label} {value:.2f}" for label, value in recall.items()
        )
        print(f"{representation:8} recall C0: {pretty}")

    reference = report.get("reference")
    if reference and reference.get("available"):
        print(
            f"\nReferencia features-only (LOSO, {reference['n_windows']} ventanas, "
            f"{reference['n_features']} features): "
            f"acc {reference['accuracy']:.3f} · bal {reference['balanced_accuracy']:.3f}"
        )
    elif reference:
        print(f"\nReferencia no disponible: {reference.get('error')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Argumentos del piloto DREAMT."""
    parser = argparse.ArgumentParser(
        description="Piloto DREAMT: estadios de sueño bajo degradación controlada."
    )
    parser.add_argument("--phase", choices=("prepare", "generate", "evaluate"), required=True)
    parser.add_argument("--data-dir", default=str(DREAMT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--representation", choices=("numeric", "text", "both"), default="both")
    parser.add_argument("--model", default=None, help="modelo de Ollama (por defecto, config.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--think", action="store_true", help="activar modo *thinking* (solo qwen3)")
    parser.add_argument("--windows-per-class", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=None, help="tope total de ventanas por sujeto")
    parser.add_argument("--conditions", default=None, help="lista separada por comas")
    parser.add_argument("--reference", action="store_true", help="incluir referencia features-only")
    parser.add_argument("--allow-failed-p0", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del piloto."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == "prepare":
        return phase_prepare(args)
    if args.phase == "generate":
        return phase_generate(args)
    return phase_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
