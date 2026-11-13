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

Uso:
    python scripts/run_degradation_sweep.py --subjects S5,S6 --max-windows 3
    python scripts/run_degradation_sweep.py            # evaluación completa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.common.llm_client import check_model_available
from sources.common.provenance import sha256_file, utc_now
from sources.ingestion.represent import LLM_ALLOWED_FEATURES, _opaque_window_id
from sources.phase2.development_split import (
    DEV_SUBJECTS,
    TEST_SUBJECTS,
    split_warning,
)
from sources.phase2.llm_config import (
    build_experiment_config,
    create_ollama_chat_client,
    model_slug,
)
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


def output_directory(
    representation: str, decision_mode: str, model: str
) -> Path:
    """Directorio de salida: representación × tarea × modelo.

    La tarea forma parte del nombre: mezclar respuestas de 4 clases y binarias
    invalidaría cualquier comparación.
    """
    base_name = "degradation" if representation == "numeric" else "degradation_text"
    if decision_mode == DECISION_MODE_STRESS_BINARY:
        base_name += "_binary"
    return OUTPUT_DIR.parent / base_name / model_slug(model)


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
    seed: int,
    max_windows: int | None,
    use_few_shot: bool,
    representation: str,
    decision_mode: str = DECISION_MODE_AFFECTIVE,
) -> dict[str, Any]:
    responses_path = output_dir / f"{subject}.responses.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing: set[tuple[str, str]] = set()
    if responses_path.is_file():
        for line in responses_path.open(encoding="utf-8"):
            record = json.loads(line)
            existing.add((record["window_id"], record["condition_id"]))

    windows = load_windows(subject)
    if not windows:
        return {"subject": subject, "n_calls": 0, "n_cached": 0, "n_errors": 0}

    clean_rows = load_features(INGEST_DIR, "C0_clean", subject)
    feature_names = sorted({name for values in clean_rows.values() for name in values})
    centers, scales = subject_reference(clean_rows, feature_names)

    condition_rows = {
        condition: load_features(INGEST_DIR, condition, subject)
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
    if max_windows is not None:
        window_ids = window_ids[:max_windows]

    n_calls = n_cached = n_errors = 0
    for window_id in window_ids:
        window = windows.get(window_id, {})
        duration_s = window.get("duration_s", 60.0)
        for condition in conditions:
            if (window_id, condition) in existing:
                n_cached += 1
                continue
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
                )
            else:
                payload = build_payload(
                    window_id, values, centers, scales, duration_s
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
                )
            assert_no_leakage(prompt)
            try:
                raw_output, ollama_meta = _call_ollama_chat(client, prompt, config, seed=seed)
                validated = validate_llm_output(raw_output, decision_mode=decision_mode)
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
                "window_id": window_id,
                "window_id_opaque": _opaque_window_id(window_id),
                "duration_s": duration_s,
                "features_sha256": condition_sha[condition],
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
                f"[DEG] {subject} {window_id} {condition}: {status} "
                f"decision={decision} conf={confidence} claims={len(claims)}"
            )

    return {
        "subject": subject,
        "n_calls": n_calls,
        "n_cached": n_cached,
        "n_errors": n_errors,
        "conditions": conditions,
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
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--output-dir", default="")
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
        temperature=args.temperature,
        seed=args.seed,
        max_retries=3,
        think=True if args.think else None,
        cascade_enabled=False,
    )
    client = create_ollama_chat_client(config)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Un subdirectorio por modelo: comparar modelos no sobrescribe
        # ni reutiliza (por error) las respuestas de otro modelo.
        output_dir = output_directory(
            args.representation, decision_mode, config.model
        )

    if check_model_available(config.model) is False:
        print(
            f"[DEG] AVISO: {config.model} no está instalado en Ollama "
            f"(ollama pull {config.model})."
        )
    print(
        f"Barrido de degradación | sujetos={subjects} | condiciones={conditions} | "
        f"representación={args.representation} | tarea={args.decision_mode} | "
        f"modelo={config.model} | "
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
        "config": config.to_dict(),
    }
    for subject in subjects:
        result = run_subject(
            subject=subject,
            conditions=conditions,
            output_dir=output_dir,
            config=config,
            client=client,
            seed=args.seed,
            max_windows=args.max_windows,
            use_few_shot=not args.no_few_shot,
            representation=args.representation,
            decision_mode=decision_mode,
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
