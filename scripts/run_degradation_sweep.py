#!/usr/bin/env python
"""Barrido de degradación controlada (H1/H3): C0, C1, C2, C4, C5.

Envía al LLM la representación numérica estandarizada por sujeto (E12) de cada
ventana bajo cada condición de degradación. Diseño congelado:

- Sujetos de evaluación por defecto (S5–S17 salvo S12); los de desarrollo se
  bloquean salvo ``--allow-dev``.
- La referencia de estandarización (mediana/IQR) se calcula **solo con las
  ventanas limpias C0 del sujeto** y se aplica a todas las condiciones. Si se
  recalculara por condición, la degradación se re-normalizaría y se perdería el
  efecto experimental.
- Tarea: clasificación afectiva de 4 clases (tarea preregistrada). El endpoint
  binario es una desviación declarada y queda como análisis secundario.
- Few-shot con medianas de S2–S4 (sin fuga del test).
- Las etiquetas no se incluyen en las respuestas.

Nota de diseño: ``C3_total_resp`` produce exactamente las mismas features que
``C1_partial_resp`` (con ≥50 % de datos ausentes, el extractor devuelve ``null``
para toda la modalidad). Por eso el barrido ejecuta las 5 condiciones
efectivamente distintas y omite C3 (idéntica a C1). Ver
``docs/degradacion_llm.md``.

Repeticiones (varianza de muestreo): ``--repetitions R`` repite el **mismo
prompt** con R semillas (``--seed + r``) y guarda ``repetition`` y ``seed`` en
cada registro. El corpus de una pasada queda intacto: con ``R > 1`` y sin
``--output-dir`` la salida va a ``{nombre}_rep{R}/{modelo}/``. Si el modelo es
determinista para el prompt, las R respuestas serán idénticas y las
repeticiones no aportarán nada; eso lo mide ``scripts/analyze_repetitions.py``.

Pilotos: ``--sample-windows K`` toma K ventanas repartidas por todo el registro
del sujeto (muestra sistemática y ciega a las etiquetas). ``--max-windows``
corta por el principio y en WESAD eso son casi todas ``baseline``.

Uso:
    python scripts/run_degradation_sweep.py --subjects S5,S6 --max-windows 3
    python scripts/run_degradation_sweep.py            # evaluación completa
    python scripts/run_degradation_sweep.py --subjects S5,S6,S7 \
        --sample-windows 12 --conditions C0_clean,C4_total_eda,C5_total_both \
        --repetitions 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.llm_client import check_model_available, ollama_run_provenance
from sources.common.provenance import engine_conflicts, sha256_file, utc_now
from sources.ingestion.represent import LLM_ALLOWED_FEATURES, _opaque_window_id
from sources.phase2.development_split import (
    DEV_SUBJECTS,
    TEST_SUBJECTS,
    split_warning,
)
from sources.phase2.binary_variants import (
    DEFAULT_NONSTRESS_VARIANT,
    NONSTRESS_VARIANTS,
    variant_classes,
)
from sources.phase2.llm_config import (
    build_experiment_config,
    create_ollama_chat_client,
)
from sources.phase3.degradation import degradation_output_dir
from sources.phase2.orchestrator import _call_ollama_chat
from sources.phase2.prompt_builder import (
    assert_no_leakage,
    build_numeric_prompt,
    build_text_prompt,
)
from sources.phase2.response_schema import (
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_STRESS_BINARY,
    validate_llm_output,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INGEST_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "INGEST"
OUTPUT_DIR = REPO_ROOT / "results" / "output" / "WESAD" / "PHASE2" / "degradation"

from sources.phase3.degradation import (  # noqa: E402
    ALL_CONDITIONS,
    EFFECTIVE_CONDITIONS,
    build_payload,
    features_path,
    load_features,
    render_text_summary,
    subject_reference,
    z_scores,
)


def repetition_seeds(base_seed: int, repetitions: int) -> list[int]:
    """Semillas de las repeticiones: ``base_seed + r`` para ``r = 0..R-1``.

    La repetición 0 usa ``--seed`` (42 por defecto), de modo que un barrido con
    ``--repetitions 1`` y uno con ``--repetitions R`` comparten la primera
    muestra y son comparables sin volver a llamar al modelo.

    Raises:
        ValueError: si ``repetitions`` no es un entero positivo.
    """
    if int(repetitions) < 1:
        raise ValueError(f"repetitions debe ser >= 1 (recibido {repetitions!r})")
    return [int(base_seed) + r for r in range(int(repetitions))]


def existing_pairs(path: Path) -> tuple[set[tuple[str, str, int]], int]:
    """Ternas ``(ventana, condición, repetición)`` ya respondidas.

    Devuelve también cuántas respuestas con ``timeout`` se reintentarán. Un
    ``timeout`` es un fallo de transporte, no una respuesta: si se contara como
    cacheado, un barrido lanzado con el modelo sin instalar quedaría marcado
    como completo y no se reintentaría nunca.

    Los corpus anteriores a las repeticiones no tienen el campo
    ``repetition``: se interpretan como repetición 0 (la del ``--seed`` base).
    """
    existing: set[tuple[str, str, int]] = set()
    n_retry = 0
    if not path.is_file():
        return existing, n_retry
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:      # línea a medio escribir
            continue
        if record.get("status") == "timeout":
            n_retry += 1
            continue
        existing.add(
            (
                record["window_id"],
                record["condition_id"],
                int(record.get("repetition", 0) or 0),
            )
        )
    return existing, n_retry


def sample_window_ids(window_ids: list[str], k: int | None) -> list[str]:
    """Muestra sistemática de ``k`` ventanas repartidas por todo el registro.

    Se usa en pilotos: ``--max-windows`` corta por el principio y en WESAD el
    protocolo es secuencial (las primeras ventanas son todas ``baseline``), así
    que un piloto con las N primeras mediría una sola clase. Esta muestra
    recorre el registro completo con paso constante y **no** consulta las
    etiquetas: la selección es ciega respecto de la clase real.

    Args:
        window_ids: identificadores ordenados del sujeto.
        k: número de ventanas; ``None`` o ``>= len`` devuelve todas.
    """
    ordered = sorted(window_ids)
    if not k or int(k) >= len(ordered):
        return ordered
    k = int(k)
    step = len(ordered) / k
    return [ordered[int(i * step)] for i in range(k)]


def ensure_model_available(model: str) -> None:
    """Aborta si el modelo no está en Ollama.

    Antes solo se avisaba y el barrido gastaba horas escribiendo timeouts, que
    además quedaban cacheados (job del 2026-11-14 con el 7b sin instalar).
    """
    if check_model_available(model) is False:
        raise SystemExit(
            f"[DEG] El modelo {model!r} no está instalado en Ollama.\n"
            f"      Instálalo con: ollama pull {model}\n"
            f"      (comprueba también OLLAMA_MODELS: el job usa $HOME/data/.ollama/models)"
        )


def output_directory(
    representation: str,
    decision_mode: str,
    model: str,
    nonstress_variant: str = DEFAULT_NONSTRESS_VARIANT,
    temperature_mode: str = "include",
    repetitions: int = 1,
) -> Path:
    """Directorio de salida: representación × tarea × variante × modelo × R.

    Delega en :func:`sources.phase3.degradation.degradation_output_dir` para que
    el barrido y los analizadores resuelvan **exactamente** las mismas rutas
    (evita que un cambio de nombre en un lado deje huérfano al otro).

    La tarea y la variante forman parte del nombre: mezclar respuestas de 4
    clases, de binario con distintas clases ``non-stress`` o con/sin
    temperatura invalidaría cualquier comparación. Las repeticiones añaden
    ``_rep{R}``: el corpus de una sola pasada es el que sostiene las cifras
    publicadas y no debe mezclarse con el de repeticiones.
    """
    return degradation_output_dir(
        OUTPUT_DIR.parent,
        model,
        representation=representation,
        decision_mode=decision_mode,
        nonstress_variant=nonstress_variant,
        temperature_mode=temperature_mode,
        repetitions=repetitions,
    )


def load_windows(subject: str) -> dict[str, dict[str, Any]]:
    path = INGEST_DIR / "windows" / f"{subject}.windows.jsonl"
    return {
        json.loads(line)["window_id"]: json.loads(line)
        for line in path.open(encoding="utf-8")
    }


def run_subject(
    subject: str,
    conditions: list[str],
    output_dir: Path,
    config: Any,
    client: dict[str, Any],
    seeds: list[int],
    max_windows: int | None,
    use_few_shot: bool,
    representation: str,
    decision_mode: str = DECISION_MODE_AFFECTIVE,
    nonstress_variant: str = DEFAULT_NONSTRESS_VARIANT,
    include_temperature: bool = True,
    sample_windows: int | None = None,
) -> dict[str, Any]:
    responses_path = output_dir / f"{subject}.responses.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = variant_classes(nonstress_variant)

    existing, n_retry = existing_pairs(responses_path)
    if n_retry:
        print(
            f"[DEG] {subject}: {n_retry} respuestas con timeout se reintentarán "
            "(no se reutilizan)."
        )

    windows = load_windows(subject)
    if not windows:
        return {"subject": subject, "n_calls": 0, "n_cached": 0, "n_errors": 0}

    clean_rows = load_features(
        INGEST_DIR,
        "C0_clean",
        subject,
        include_temperature=include_temperature,
    )
    feature_names = sorted({name for values in clean_rows.values() for name in values})
    centers, scales = subject_reference(clean_rows, feature_names)

    condition_rows = {
        condition: load_features(
            INGEST_DIR,
            condition,
            subject,
            include_temperature=include_temperature,
        )
        for condition in conditions
    }
    # Ancla de procedencia: hash del fichero exacto de features que se usa para
    # construir el prompt. Permite verificar después (en O(1)) que el corpus
    # reconstruido es el mismo, sin recomponer prompts.
    condition_sha = {
        condition: sha256_file(features_path(INGEST_DIR, condition, subject))
        for condition in conditions
    }
    window_ids = sorted(clean_rows)
    if sample_windows is not None:
        window_ids = sample_window_ids(window_ids, sample_windows)
    if max_windows is not None:
        window_ids = window_ids[:max_windows]

    n_calls = n_cached = n_errors = 0
    for window_id in window_ids:
        window = windows.get(window_id, {})
        duration_s = window.get("duration_s", 60.0)
        for condition in conditions:
            pending = [
                (repetition, seed)
                for repetition, seed in enumerate(seeds)
                if (window_id, condition, repetition) not in existing
            ]
            n_cached += len(seeds) - len(pending)
            if not pending:
                continue

            # El prompt se construye una sola vez por (ventana, condición): las
            # repeticiones deben recibir exactamente el mismo texto y variar
            # solo la semilla. Si se reconstruyera por repetición, cualquier
            # no determinismo de la construcción se confundiría con muestreo.
            values = condition_rows[condition].get(window_id, {})
            if representation == "text":
                scores = z_scores(values, centers, scales)
                prompt = build_text_prompt(
                    text_representation=render_text_summary(scores),
                    window_id_opaque=_opaque_window_id(window_id),
                    duration_s=duration_s,
                    num_ctx=config.num_ctx,
                    use_few_shot=use_few_shot,
                    standardized=True,
                    decision_mode=decision_mode,
                    nonstress_classes=classes,
                )
            else:
                payload = build_payload(
                    window_id,
                    values,
                    centers,
                    scales,
                    duration_s,
                    include_temperature=include_temperature,
                )
                prompt = build_numeric_prompt(
                    payload=payload,
                    window_id_opaque=_opaque_window_id(window_id),
                    duration_s=duration_s,
                    num_ctx=config.num_ctx,
                    use_few_shot=use_few_shot,
                    include_additional=True,
                    standardized=True,
                    decision_mode=decision_mode,
                    nonstress_classes=classes,
                )
            assert_no_leakage(prompt)

            for repetition, seed in pending:
                try:
                    raw_output, ollama_meta = _call_ollama_chat(
                        client, prompt, config, seed=seed
                    )
                    validated = validate_llm_output(
                        raw_output, decision_mode=decision_mode
                    )
                    status, decision = validated.status, validated.decision
                    confidence = validated.confidence
                    claims = [c.to_dict() for c in validated.claims]
                    reasoning, error = validated.reasoning, validated.error_message
                except Exception as exc:  # noqa: BLE001
                    raw_output, ollama_meta = "", {}
                    status, decision, confidence = "timeout", None, None
                    claims, reasoning, error = [], "", str(exc)

                record = {
                    "subject": subject,
                    "split": "development" if subject in DEV_SUBJECTS else "evaluation",
                    "model": config.model,
                    "condition_id": condition,
                    "representation": representation,
                    "decision_mode": decision_mode,
                    "nonstress_variant": nonstress_variant,
                    "nonstress_classes": list(classes),
                    "include_temperature": include_temperature,
                    "window_id": window_id,
                    "window_id_opaque": _opaque_window_id(window_id),
                    "duration_s": duration_s,
                    "features_sha256": condition_sha[condition],
                    "repetition": repetition,
                    "seed": seed,
                    "status": status,
                    "decision": decision,
                    "confidence": confidence,
                    "claims": claims,
                    "reasoning": reasoning,
                    "error_message": error,
                    "raw_output": raw_output,
                    "prompt": prompt,
                    "ollama_meta": ollama_meta,
                    "generated_at": utc_now(),
                }
                with responses_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_calls += 1
                if status in ("timeout", "invalid_output"):
                    n_errors += 1
                print(
                    f"[DEG] {subject} {window_id} {condition} rep={repetition} "
                    f"seed={seed}: {status} decision={decision} "
                    f"conf={confidence} claims={len(claims)}"
                )

    return {
        "subject": subject,
        "n_calls": n_calls,
        "n_cached": n_cached,
        "n_errors": n_errors,
        "conditions": conditions,
        "n_repetitions": len(seeds),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="", help="Lista separada por comas.")
    parser.add_argument(
        "--conditions",
        default=",".join(EFFECTIVE_CONDITIONS),
        help="Condiciones a ejecutar (C3 es idéntica a C1 y se omite por defecto).",
    )
    parser.add_argument("--allow-dev", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="Modelo LLM (default: defaults.llm.primary_model de config.json).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.3,
        help="Temperatura de muestreo del LLM (no confundir con --temperature, "
        "que incluye o excluye la modalidad de temperatura de la evidencia).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeticiones del mismo prompt con semillas distintas (varianza de "
        "muestreo). La repetición 0 usa --seed y la r usa --seed+r.",
    )
    parser.add_argument(
        "--sample-windows",
        type=int,
        default=None,
        help="Pilotos: K ventanas por sujeto repartidas por todo el registro "
        "(muestra sistemática, ciega a las etiquetas). Sin esta bandera se "
        "usan todas.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Activar el modo *thinking* (solo qwen3). Por defecto, desactivado.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument(
        "--representation",
        choices=["numeric", "text"],
        default="numeric",
        help="Representación de la señal (H3).",
    )
    parser.add_argument(
        "--decision-mode",
        choices=["affective", "binary"],
        default="affective",
        help="Tarea del prompt: 4 clases afectivas (preregistrada) o binario "
        "stress/non-stress (endpoint primario).",
    )
    parser.add_argument(
        "--nonstress-variant",
        choices=sorted(NONSTRESS_VARIANTS),
        default=DEFAULT_NONSTRESS_VARIANT,
        help="Clases de WESAD que forman non-stress en el prompt binario "
        "(default: schmidt = baseline + amusement). Las demás son sensibilidades.",
    )
    parser.add_argument(
        "--temperature",
        choices=["include", "exclude"],
        default="include",
        help="Incluir o excluir la temperatura de la evidencia del LLM. El "
        "experimento primario usa 'exclude' (confusor de orden y única "
        "modalidad de pecho que no se degrada).",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--allow-engine-change",
        action="store_true",
        help="Reanudar aunque el motor (versión de Ollama o digest del modelo) "
        "haya cambiado desde el corpus existente. Por defecto se aborta: mezclar "
        "motores es indistinguible de la varianza de muestreo.",
    )
    args = parser.parse_args(argv)

    subjects = [
        s.strip().upper() for s in args.subjects.split(",") if s.strip()
    ] or list(TEST_SUBJECTS)
    warning = split_warning(subjects)
    if warning and not args.allow_dev:
        raise SystemExit(
            f"[DEG] {warning}.\nUsa --allow-dev o elige sujetos de evaluación."
        )
    if warning:
        print(f"[DEG] AVISO: {warning}")

    decision_mode = (
        DECISION_MODE_STRESS_BINARY
        if args.decision_mode == "binary"
        else DECISION_MODE_AFFECTIVE
    )
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise SystemExit(f"Condiciones no válidas: {unknown}")

    config = build_experiment_config(
        model=args.model,
        temperature=args.llm_temperature,
        seed=args.seed,
        max_retries=3,
        think=True if args.think else None,
        cascade_enabled=False,
    )
    client = create_ollama_chat_client(config)
    seeds = repetition_seeds(args.seed, args.repetitions)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Un subdirectorio por modelo: comparar modelos no sobrescribe
        # ni reutiliza (por error) las respuestas de otro modelo. Las
        # repeticiones llevan sufijo _rep{R} (ver degradation_dir_name).
        output_dir = output_directory(
            args.representation,
            decision_mode,
            config.model,
            args.nonstress_variant,
            args.temperature,
            repetitions=len(seeds),
        )

    ensure_model_available(config.model)
    # Identidad del motor y del modelo: dos builds del mismo nombre no
    # producen el mismo corpus (2026-11-18: 0 vs 28 abstenciones con Ollama
    # 0.40.6 y 0.30.7). Se escribe al arrancar, no al final, para que quede
    # registrada aunque el job se corte.
    provenance = ollama_run_provenance(config.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "run_provenance.json"
    if provenance_path.is_file():
        try:
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:          # archivo a medio escribir
            previous = {}
        conflicts = engine_conflicts(previous.get("ollama"), provenance)
        if conflicts:
            detail = "\n".join(f"  - {c}" for c in conflicts)
            if not args.allow_engine_change:
                raise SystemExit(
                    f"[DEG] El corpus de {output_dir} se generó con otro motor o "
                    f"modelo:\n{detail}\n"
                    f"      Reanudar mezclaría corpora y, en un experimento de "
                    f"varianza de muestreo, el cambio de motor sería\n"
                    f"      indistinguible del ruido que se mide. Mueve o renombra "
                    f"el corpus existente, o pasa --allow-engine-change."
                )
            print(f"[DEG] AVISO: cambio de motor con --allow-engine-change:\n{detail}")
    with provenance_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": utc_now(),
                "model": config.model,
                "model_slug": config.model_slug,
                "representation": args.representation,
                "decision_mode": args.decision_mode,
                "nonstress_variant": args.nonstress_variant,
                "temperature_mode": args.temperature,
                "repetitions": len(seeds),
                "seeds": seeds,
                "sample_windows": args.sample_windows,
                "config": config.to_dict(),
                "ollama": provenance,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"[DEG] motor: ollama={provenance.get('ollama_version')} "
        f"digest={provenance.get('model_digest')}"
    )
    print(
        f"Barrido de degradación | sujetos={subjects} | condiciones={conditions} | "
        f"representación={args.representation} | tarea={args.decision_mode} | "
        f"variante={args.nonstress_variant} | temperatura={args.temperature} | "
        f"modelo={config.model} | repeticiones={len(seeds)} (semillas={seeds}) | "
        f"think={config.think} | few_shot={not args.no_few_shot} | "
        f"salida={output_dir}"
    )
    total = {"n_calls": 0, "n_cached": 0, "n_errors": 0}
    summary = {
        "subjects": {},
        "generated_at": utc_now(),
        "conditions": conditions,
        "model": config.model,
        "model_slug": config.model_slug,
        "representation": args.representation,
        "decision_mode": args.decision_mode,
        "nonstress_variant": args.nonstress_variant,
        "nonstress_classes": list(variant_classes(args.nonstress_variant)),
        "temperature_mode": args.temperature,
        "include_temperature": args.temperature == "include",
        "repetitions": len(seeds),
        "seeds": seeds,
        "sample_windows": args.sample_windows,
        "config": config.to_dict(),
        "ollama": provenance,
    }
    for subject in subjects:
        result = run_subject(
            subject=subject,
            conditions=conditions,
            output_dir=output_dir,
            config=config,
            client=client,
            seeds=seeds,
            max_windows=args.max_windows,
            use_few_shot=not args.no_few_shot,
            representation=args.representation,
            decision_mode=decision_mode,
            nonstress_variant=args.nonstress_variant,
            include_temperature=args.temperature == "include",
            sample_windows=args.sample_windows,
        )
        summary["subjects"][subject] = result
        for key in total:
            total[key] += result.get(key, 0)
        print(f"[DEG] {subject}: {result}")

    summary["totals"] = total
    summary_path = output_dir / "degradation_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nTotal: {total}")
    print(f"Resumen: {summary_path}")


if __name__ == "__main__":
    main()
