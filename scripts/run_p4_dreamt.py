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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sources.common.llm_client import check_model_available, ollama_run_provenance
from sources.common.provenance import guard_engine
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
    normalize_metric,
    response_coherence,
    summarize_coherence,
    summarize_verdicts,
    verify_claims,
)
from sources.phase3.records import evaluate_records
from sources.phase3.recovery import apply_recovery
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

#: Cohorte de **escala** (declarada y congelada el 2026-11-24, antes de generar
#: una sola llamada de esta cohorte).
#:
#: Regla de selección, ciega a los resultados: (1) sujetos con **≥ 12 ventanas
#: puras en cada una de las cuatro clases** (criterio de disponibilidad de
#: datos, medido solo con las etiquetas de sueño); (2) se excluyen los sujetos
#: de desarrollo (``DREAMT_DEV_SUBJECTS``) y los del piloto
#: (``PILOT_SUBJECTS``), y (3) de la lista ordenada resultante —50 sujetos— se
#: toman **12 repartidos uniformemente** (``round(i*(n-1)/11)``), para no
#: concentrar la cohorte en un tramo de identificadores (orden de reclutamiento).
#:
#: Nunca se usaron resultados del LLM para elegirla: la lista se calculó con
#: recuentos de ventanas puras, que solo dependen del etiquetado de sueño.
SCALE_SUBJECTS: tuple[str, ...] = (
    "S008",
    "S015",
    "S021",
    "S035",
    "S039",
    "S045",
    "S061",
    "S067",
    "S071",
    "S086",
    "S093",
    "S103",
)

#: Cohortes disponibles: la del piloto (preregistro §7) y la de escala.
COHORTS: dict[str, tuple[str, ...]] = {
    "pilot": PILOT_SUBJECTS,
    "scale": SCALE_SUBJECTS,
}

#: Directorio de salida por defecto de cada cohorte: nunca se mezclan.
DEFAULT_OUTPUTS: dict[str, Path] = {
    "pilot": REPO_ROOT / "results" / "output" / "DREAMT" / "PHASE2" / "pilot",
    "scale": REPO_ROOT / "results" / "output" / "DREAMT" / "PHASE2" / "scale",
}

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

#: Compatibilidad: el piloto sigue siendo el destino por defecto de la cohorte
#: ``pilot`` (ver ``DEFAULT_OUTPUTS``).
DEFAULT_OUTPUT = DEFAULT_OUTPUTS["pilot"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def repetition_seeds(base_seed: int, repetitions: int) -> list[int]:
    """Semillas de las repeticiones: ``base_seed + r`` para ``r = 0..R-1``.

    La repetición 0 usa ``--seed`` (42 por defecto), de modo que una corrida con
    ``R = 1`` y otra con ``R > 1`` comparten la primera muestra.
    """
    if int(repetitions) < 1:
        raise ValueError(f"repetitions debe ser >= 1 (recibido {repetitions!r})")
    return [int(base_seed) + r for r in range(int(repetitions))]


def responses_directory(repetitions: int) -> str:
    """Nombre del directorio de respuestas, aislado por número de repeticiones.

    El corpus de una sola pasada (``responses/``) sostiene las cifras
    publicadas; el de repeticiones vive en ``responses_rep{R}/`` y nunca se
    mezcla con él.
    """
    return "responses" if int(repetitions) <= 1 else f"responses_rep{int(repetitions)}"


def provenance_filename(repetitions: int) -> str:
    """Nombre del ``run_provenance`` de la corrida (por número de repeticiones)."""
    return (
        "run_provenance.json"
        if int(repetitions) <= 1
        else f"run_provenance_rep{int(repetitions)}.json"
    )


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


def parse_subject_filter(
    raw: str | None,
    cohort: str = "pilot",
) -> tuple[str, ...] | None:
    """Subconjunto de la cohorte pedido con ``--subjects`` (``S034,S051``).

    Las cohortes están congeladas: ``pilot`` en el preregistro
    ([`docs/piloto_dreamt.md`](../docs/piloto_dreamt.md) §7) y ``scale`` en
    [`docs/escala_dreamt.md`](../docs/escala_dreamt.md) §2. Aquí **solo** se
    permite elegir un subconjunto (para humos y reanudaciones parciales): pedir
    un sujeto de desarrollo o cualquiera fuera de la cohorte es un error, porque
    añadiría datos de evaluación no preregistrados o mezclaría desarrollo con
    evaluación. Devuelve ``None`` cuando no se filtra.

    Raises:
        ValueError: si la cohorte no existe o algún sujeto no pertenece a ella.
    """
    if raw is None:
        return None
    names = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
    if not names:
        return None
    if cohort not in COHORTS:
        raise ValueError(f"cohorte desconocida {cohort!r}; disponibles: {sorted(COHORTS)}")
    members = COHORTS[cohort]
    outside = [name for name in names if name not in set(members)]
    if outside:
        raise ValueError(
            f"sujetos fuera de la cohorte {cohort!r} {members}: {outside}. "
            "Las cohortes están congeladas; solo se puede elegir un subconjunto."
        )
    return tuple(name for name in members if name in set(names))


def select_cohort_subjects(
    present: Iterable[str],
    requested: tuple[str, ...] | None = None,
    cohort: str = "pilot",
) -> list[str]:
    """Sujetos disponibles de la cohorte, en el orden declarado."""
    members = COHORTS[cohort]
    present_set = set(present)
    wanted = set(requested) if requested else None
    return [
        subject
        for subject in members
        if subject in present_set and (wanted is None or subject in wanted)
    ]


def select_pilot_subjects(
    present: Iterable[str],
    requested: tuple[str, ...] | None = None,
) -> list[str]:
    """Compatibilidad: cohorte ``pilot`` (ver ``select_cohort_subjects``)."""
    return select_cohort_subjects(present, requested, cohort="pilot")


def requested_subjects(args: argparse.Namespace) -> tuple[str, ...] | None:
    """Filtro ``--subjects`` ya validado contra la cohorte, con error legible."""
    cohort = getattr(args, "cohort", "pilot") or "pilot"
    try:
        return parse_subject_filter(getattr(args, "subjects", None), cohort=cohort)
    except ValueError as exc:
        raise SystemExit(f"[DREAMT] {exc}") from exc


def cohort_of(args: argparse.Namespace) -> str:
    """Cohorte pedida (``pilot`` por defecto)."""
    return getattr(args, "cohort", "pilot") or "pilot"


def filter_prompt_rows(
    prompt_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str] | None,
) -> list[Mapping[str, Any]]:
    """Restringe los prompts a las condiciones pedidas.

    Se aplica en la fase de generación porque los prompts ya contienen todas
    las condiciones del muestreo: sin este filtro, un humo de dos condiciones
    gastaba las seis (fallo real detectado el 2026-11-23).

    Raises:
        SystemExit: si se pide una condición que no está en los prompts.
    """
    if not conditions:
        return list(prompt_rows)
    available = {row["condition_id"] for row in prompt_rows}
    absent = [condition for condition in conditions if condition not in available]
    if absent:
        raise SystemExit(
            f"[DREAMT] Condiciones sin prompts: {absent}. "
            f"Disponibles: {sorted(available)}"
        )
    wanted = set(conditions)
    return [row for row in prompt_rows if row["condition_id"] in wanted]


def prompts_sampling(
    output_dir: Path, subject: str
) -> tuple[int | None, set[str]]:
    """Muestreo **efectivo** de los prompts de un sujeto, leído del disco.

    Devuelve ``(ventanas por clase, condiciones)``. Se calcula desde el propio
    fichero porque el manifiesto guarda lo que se *pidió*, no lo que quedó
    escrito: tras un ``prepare`` que reutilizó prompts, el manifiesto quedaba
    con el muestreo nuevo y la comparación no detectaba nada (fallo real del
    2026-11-23).
    """
    rows = read_jsonl(output_dir / "prompts" / f"{subject}.prompts.jsonl")
    if not rows:
        return None, set()
    windows_by_stage: dict[str, set[str]] = defaultdict(set)
    conditions: set[str] = set()
    for row in rows:
        windows_by_stage[str(row.get("stage", ""))].add(str(row["window_id"]))
        conditions.add(str(row["condition_id"]))
    per_class = max((len(ids) for ids in windows_by_stage.values()), default=0)
    return per_class, conditions


def check_sampling_consistency(
    output_dir: Path,
    subjects: Sequence[str],
    windows_per_class: int,
    conditions: Sequence[str],
) -> list[str]:
    """Diferencias entre el muestreo pedido y el que tienen los prompts.

    Los prompts se reutilizan mientras existan (``prepare`` no los regenera sin
    ``--force``), así que ``--windows-per-class 1`` o una condición que no se
    generó devolvían **en silencio** otro muestreo del pedido. Pedir **menos**
    condiciones es legítimo (se filtra al generar); el resto se avisa.
    """
    issues: list[str] = []
    for subject in subjects:
        per_class, available = prompts_sampling(output_dir, subject)
        if per_class is None:
            continue
        if int(per_class) != int(windows_per_class):
            issues.append(
                f"{subject}: windows_per_class: prompts con {per_class}, "
                f"petición pide {windows_per_class}"
            )
        missing = [condition for condition in conditions if condition not in available]
        if missing:
            issues.append(
                f"{subject}: condiciones sin prompts: {missing} "
                f"(disponibles: {sorted(available)})"
            )
    return issues


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
    requested = requested_subjects(args)
    cohort = cohort_of(args)
    members = select_cohort_subjects(present, requested, cohort=cohort)
    missing = [s for s in (requested or COHORTS[cohort]) if s not in present]
    if missing:
        print(f"[DREAMT] Aviso: sujetos de la cohorte sin datos: {missing}")
    if not members:
        raise SystemExit(
            f"[DREAMT] Ningún sujeto de la cohorte {cohort!r} está en {data_dir}. "
            "Comprueba --data-dir (results/input/dreamt/data_64Hz)."
        )
    if not dev:
        raise SystemExit(
            f"[DREAMT] Faltan los sujetos de desarrollo {DREAMT_DEV_SUBJECTS} en "
            f"{data_dir}: sin ellos no hay ejemplos few-shot ni pre-chequeo P0."
        )
    print(f"[DREAMT] cohorte {cohort} | desarrollo {dev} | evaluación {members}")

    names = feature_names()
    for subject in dev + members:
        night_rows(subject, data_dir, output_dir / "features", cfg, force=args.force)

    centers: dict[str, dict[str, float]] = {}
    scales: dict[str, dict[str, float]] = {}
    for subject in dev + members:
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

    drift = check_sampling_consistency(
        output_dir, members, args.windows_per_class, conditions
    )
    if drift and not args.force:
        raise SystemExit(
            "[DREAMT] Los prompts existentes no corresponden a lo pedido:\n  - "
            + "\n  - ".join(drift)
            + "\n  Los prompts se reutilizan tal cual, así que la ejecución no "
            "mediría lo que pides. Opciones:\n"
            "    a) humo aislado: añade --output-dir <carpeta nueva>\n"
            "    b) regenerar prompts: repite con --force (las respuestas ya "
            "escritas de ese modelo no se borran: revísalas antes de evaluar)."
        )
    if drift:
        print(
            "[DREAMT] Aviso: se regeneran los prompts por --force ("
            + "; ".join(drift)
            + ")"
        )

    summary = []
    for subject in members:
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
        "cohort": cohort,
        "dev_subjects": dev,
        "cohort_subjects": list(members),
        "pilot_subjects": list(members),
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
    repetitions = max(int(getattr(args, "repetitions", None) or 1), 1)
    seeds = repetition_seeds(config.seed, repetitions)
    responses_dir = output_dir / responses_directory(repetitions) / slug
    responses_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and check_model_available(config.model) is False:
        raise SystemExit(
            f"[DREAMT] El modelo {config.model!r} no está instalado en Ollama.\n"
            f"         Instálalo con: ollama pull {config.model}"
        )

    provenance_path = output_dir / provenance_filename(repetitions)
    provenance = ollama_run_provenance(config.model)
    provenance["config"] = config.to_dict()
    provenance["started_at"] = utc_now()
    provenance["repetitions"] = repetitions
    provenance["seeds"] = seeds
    if not args.dry_run:
        # El motor forma parte del experimento: mezclar versiones dentro del
        # mismo fichero haría indistinguible del ruido el efecto que se mide.
        guard_engine(
            provenance_path,
            provenance,
            bool(getattr(args, "allow_engine_change", False)),
            label="DREAMT",
        )
        write_json(provenance_path, provenance)

    present = set(available_subjects(data_dir))
    cohort = cohort_of(args)
    subjects = select_cohort_subjects(present, requested_subjects(args), cohort=cohort)
    if not subjects:
        raise SystemExit(
            f"[DREAMT] Ningún sujeto de la cohorte {cohort!r} está en {data_dir}: "
            "no hay nada que generar."
    )
    print(f"[DREAMT] cohorte {cohort} | {len(subjects)} sujetos | {output_dir}")
    client = create_ollama_chat_client(config) if not args.dry_run else None

    # ``--conditions`` también filtra en la generación: los prompts ya
    # contienen todas las condiciones del muestreo, y sin este filtro un humo
    # de dos condiciones gastaría las seis (fallo real, 2026-11-23).
    requested_conditions = (
        [part.strip() for part in str(args.conditions).split(",") if part.strip()]
        if getattr(args, "conditions", None)
        else None
    )

    planned = 0
    done = 0
    errors = 0
    for subject in subjects:
        prompt_rows = read_jsonl(output_dir / "prompts" / f"{subject}.prompts.jsonl")
        if not prompt_rows:
            print(f"[DREAMT] {subject}: sin prompts; se omite")
            continue
        if requested_conditions:
            prompt_rows = filter_prompt_rows(prompt_rows, requested_conditions)
            print(
                f"[DREAMT] {subject}: filtrado a {len(requested_conditions)} condiciones "
                f"({len(prompt_rows)} prompts)"
            )
        for representation in representations:
            path = responses_dir / f"{subject}.{representation}.jsonl"
            existing = {
                (
                    row["window_id"],
                    row["condition_id"],
                    int(row.get("repetition", 0) or 0),
                )
                for row in read_jsonl(path)
                if row.get("status") not in ("timeout",)
            }
            pending = [
                (row, rep)
                for row in prompt_rows
                for rep in range(repetitions)
                if (row["window_id"], row["condition_id"], rep) not in existing
            ]
            planned += len(pending)
            if args.dry_run:
                print(
                    f"[DREAMT] dry-run {subject} {representation}: "
                    f"{len(pending)} llamadas pendientes de "
                    f"{len(prompt_rows) * repetitions}"
                )
                continue
            print(
                f"[DREAMT] {subject} {representation}: {len(pending)} pendientes "
                f"({len(existing)} ya hechas, {repetitions} repeticiones)"
            )
            for index, (row, rep) in enumerate(pending, start=1):
                prompt = row["prompts"][representation]
                try:
                    raw_output, meta = _call_ollama_chat(
                        client, prompt, config, seed=seeds[rep]
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
                    "repetition": rep,
                    "seed": seeds[rep],
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
        "repetitions": repetitions,
        "seeds": seeds,
        "n_calls_new": done,
        "n_errors": errors,
        "config": config.to_dict(),
        "finished_at": utc_now(),
    }
    summary_name = (
        "generate_summary.json"
        if repetitions <= 1
        else f"generate_summary_rep{repetitions}.json"
    )
    write_json(output_dir / summary_name, summary)
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
    unrecognized = 0
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
            # Diagnóstico del vocabulario: un ``unsupported`` puede significar
            # «la métrica no está en la entrada» (evidencia ausente) o «el
            # verificador no conoce ese nombre». Sin separarlos, la tasa de
            # respaldo mide el alias más que la conducta del modelo.
            unrecognized += sum(
                1
                for claim in record["claims"]
                if normalize_metric(
                    str(claim.get("metric", "")), dreamt.SLEEP_METRIC_ALIASES
                )
                is None
            )
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
    metrics["claims"]["unrecognized_metric_names"] = unrecognized
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
    repetitions = max(int(getattr(args, "repetitions", None) or 1), 1)
    if repetitions > 1:
        raise SystemExit(
            "[DREAMT] La varianza de muestreo se analiza con "
            "scripts/analyze_repetitions_external.py: usa "
            "`--proc EVALUATE --experiment dreamt --repetitions R`."
        )
    responses_dir = output_dir / responses_directory(repetitions) / slug

    present = set(available_subjects(Path(args.data_dir)))
    cohort = cohort_of(args)
    subjects = select_cohort_subjects(present, requested_subjects(args), cohort=cohort)
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
    recovery_counts: dict[str, int] = {}

    for representation in representations:
        records = []
        for subject in subjects:
            records.extend(read_jsonl(responses_dir / f"{subject}.{representation}.jsonl"))
        if not records:
            print(f"[DREAMT] {representation}: sin respuestas en {responses_dir}")
            continue
        if getattr(args, "recovery", False):
            # Un modelo que mezcla magnitud y dirección (``direction: "high"``)
            # invalida la respuesta entera; la decisión y la confianza siguen
            # siendo legibles. Se recuperan para el análisis —nunca se tocan los
            # datos originales— y se cuentan aparte (medgemma:4b, 9 % inválidas).
            records, recovered = apply_recovery(records, decision_mode="sleep4")
            recovery_counts[representation] = recovered
            print(
                f"[DREAMT] {representation}: recuperadas {recovered} respuestas "
                "inválidas (decisión conservada, claims mal formados descartados)"
            )
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

    if recovery_counts:
        report["recovery"] = {"enabled": True, "recovered": recovery_counts}

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
    parser.add_argument(
        "--cohort",
        choices=tuple(COHORTS),
        default="pilot",
        help="cohorte de evaluación congelada (pilot o scale)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="carpeta de trabajo (por defecto, la de la cohorte)",
    )
    parser.add_argument("--representation", choices=("numeric", "text", "both"), default="both")
    parser.add_argument("--model", default=None, help="modelo de Ollama (por defecto, config.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeticiones del mismo prompt con semillas distintas (--seed+r). "
        "Con R>1 la salida va a responses_rep{R}/ (nunca se mezcla con el "
        "corpus de una pasada).",
    )
    parser.add_argument(
        "--allow-engine-change",
        action="store_true",
        help="Permite reanudar un corpus generado con otro motor de Ollama.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--think", action="store_true", help="activar modo *thinking* (solo qwen3)")
    parser.add_argument("--windows-per-class", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=None, help="tope total de ventanas por sujeto")
    parser.add_argument("--conditions", default=None, help="lista separada por comas")
    parser.add_argument(
        "--subjects",
        default=None,
        help="subconjunto del piloto (p. ej. S034,S051); fuera del piloto es error",
    )
    parser.add_argument("--reference", action="store_true", help="incluir referencia features-only")
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="recuperar decisión/confianza de respuestas inválidas por claims mal formados",
    )
    parser.add_argument("--allow-failed-p0", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del piloto."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "output_dir", None) is None:
        args.output_dir = str(DEFAULT_OUTPUTS[args.cohort])
    if args.phase == "prepare":
        return phase_prepare(args)
    if args.phase == "generate":
        return phase_generate(args)
    return phase_evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
