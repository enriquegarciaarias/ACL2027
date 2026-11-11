"""Despacho homogéneo de fases: ``INGEST`` → ``GENERATE`` → ``EVALUATE``.

El CLI se organiza por **verbos** (qué hace el sistema) y deja el **experimento**
como parámetro:

- ``--proc INGEST``: señal → corpus (features, SQI, representaciones). No
  depende del modelo; se ejecuta una vez por corpus.
- ``--proc GENERATE --experiment {controls,degradation,centering,physionet,
  kemocon}``: construye prompts y genera respuestas del LLM.
- ``--proc EVALUATE --experiment {...}``: mide las respuestas
  (corrección, confianza, claims, certeza, abstención) y, con
  ``--compare-models a,b``, compara modelos.

Este módulo no reimplementa ningún experimento: traduce las banderas de
``processControl.args`` al ``argv`` de los runners y analizadores existentes.
``PHASE2``/``PHASE3`` se mantienen como alias deprecados de ``GENERATE`` y
``EVALUATE`` con el experimento ``controls``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from sources.common.common import logger, processControl, writeLog

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Experimentos disponibles en GENERATE y EVALUATE.
EXPERIMENTS: tuple[str, ...] = (
    "controls",
    "degradation",
    "centering",
    "physionet",
    "kemocon",
)
DEFAULT_EXPERIMENT = "controls"

#: Condiciones del experimento ``controls`` (C0 + controles negativos).
CONTROL_CONDITIONS: int = 5
#: Condiciones efectivas del barrido de degradación.
DEGRADATION_CONDITIONS: int = 5
#: Representaciones LLM por defecto en ``controls``.
CONTROLS_REPRESENTATIONS: int = 2

_SCRIPT_CACHE: dict[str, Any] = {}


def _arg(name: str, default: Any = None) -> Any:
    """Lee un argumento de CLI del singleton ``processControl``."""
    args = processControl.args
    if args is None:
        return default
    return getattr(args, name, default)


def experiment_of(args: Mapping[str, Any] | None = None) -> str:
    """Experimento solicitado (``controls`` si no se indica)."""
    if args is None:
        value = _arg("experiment")
    elif isinstance(args, Mapping):
        value = args.get("experiment")
    else:
        value = getattr(args, "experiment", None)
    return str(value or DEFAULT_EXPERIMENT)


def _load_script(name: str):
    """Carga un script de ``scripts/`` como módulo (con caché)."""
    if name in _SCRIPT_CACHE:
        return _SCRIPT_CACHE[name]
    path = REPO_ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"No existe el script {path}")
    spec = importlib.util.spec_from_file_location(f"acl2027_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensivo
        raise ImportError(f"No se pudo cargar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SCRIPT_CACHE[name] = module
    return module


def _run_script(name: str, argv: Sequence[str]) -> None:
    """Ejecuta ``main(argv)`` de un script del repositorio."""
    module = _load_script(name)
    module.main(list(argv))


def _append_flag(argv: list[str], flag: str, value: Any) -> None:
    """Añade ``flag value`` si el valor está definido."""
    if value in (None, "", 0, False):
        return
    argv.extend([flag, str(value)])


# ---------------------------------------------------------------------------
# GENERATE
# ---------------------------------------------------------------------------
def build_generate_argv(
    experiment: str,
    args: Any,
) -> list[str] | None:
    """``argv`` del script de generación, o ``None`` si corre en proceso.

    ``controls`` se ejecuta en proceso (``orchestrator.run_phase2``); el resto
    se delega a su script.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"experimento desconocido: {experiment!r}. "
            f"Disponibles: {', '.join(EXPERIMENTS)}"
        )
    if experiment == "controls":
        return None

    argv: list[str] = []
    _append_flag(argv, "--model", getattr(args, "model", None))
    _append_flag(argv, "--max-windows", getattr(args, "max_windows", None))

    if experiment == "degradation":
        _append_flag(argv, "--representation", getattr(args, "representation", None))
        _append_flag(argv, "--conditions", getattr(args, "conditions", None))
        _append_flag(argv, "--subjects", getattr(args, "subjects", None))
        if getattr(args, "allowDev", False):
            argv.append("--allow-dev")
    elif experiment == "centering":
        _append_flag(argv, "--conditions", getattr(args, "conditions", None))
        _append_flag(argv, "--subjects", getattr(args, "subjects", None))
        _append_flag(argv, "--tag", getattr(args, "tag", None))
        if getattr(args, "allowDev", False):
            argv.append("--allow-dev")
    elif experiment == "physionet":
        _append_flag(argv, "--per-class", getattr(args, "per_class", None))
        _append_flag(argv, "--max-records", getattr(args, "max_records", None))
        _append_flag(argv, "--tag", getattr(args, "tag", None))
        if getattr(args, "few_shot", False):
            argv.append("--few-shot")
    elif experiment == "kemocon":
        _append_flag(argv, "--task", getattr(args, "task", None))
        _append_flag(argv, "--subjects", getattr(args, "subjects", None))
        _append_flag(argv, "--max-windows", getattr(args, "max_windows", None))
        _append_flag(argv, "--tag", getattr(args, "tag", None))
    return argv


def _script_for_generate(experiment: str) -> str | None:
    return {
        "degradation": "run_degradation_sweep",
        "centering": "run_e12_pilot",
        "physionet": "run_p3_physionet",
        "kemocon": "run_p3_kemocon",
    }.get(experiment)


# ---------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------
def build_evaluate_argv(experiment: str, args: Any) -> list[str]:
    """``argv`` del analizador correspondiente al experimento."""
    if experiment not in EXPERIMENTS:
        raise ValueError(
            f"experimento desconocido: {experiment!r}. "
            f"Disponibles: {', '.join(EXPERIMENTS)}"
        )
    argv: list[str] = []
    output = getattr(args, "output", None)

    if experiment == "controls":
        _append_flag(argv, "--model", getattr(args, "model", None))
        _append_flag(argv, "--subjects", getattr(args, "subjects", None))
        _append_flag(argv, "--output", output)
        if getattr(args, "raw_decisions", False):
            argv.append("--raw-decisions")
    elif experiment == "degradation":
        _append_flag(argv, "--model", getattr(args, "model", None))
        _append_flag(argv, "--representation", getattr(args, "representation", None))
        _append_flag(argv, "--conditions", getattr(args, "conditions", None))
        _append_flag(argv, "--split", getattr(args, "split", None))
        _append_flag(argv, "--output", output)
    elif experiment == "centering":
        # Calibración y predicción selectiva sobre las respuestas E12.
        _append_flag(argv, "--output", output)
    elif experiment == "physionet":
        _append_flag(argv, "--output", output)
    elif experiment == "kemocon":
        # Sin LLM: referencia features-only y evaluación del piloto existente.
        argv.extend(["--reference", "--reference-only"])
        _append_flag(argv, "--task", getattr(args, "task", None))
        _append_flag(argv, "--subjects", getattr(args, "subjects", None))
    return argv


def _script_for_evaluate(experiment: str) -> str:
    return {
        "controls": "evaluate_phase3",
        "degradation": "analyze_degradation",
        "centering": "analyze_calibration",
        "physionet": "analyze_p3_physionet",
        "kemocon": "run_p3_kemocon",
    }[experiment]


def build_compare_argv(args: Any) -> list[str] | None:
    """``argv`` del comparador de modelos, o ``None`` si no se pidió."""
    models = str(getattr(args, "compare_models", "") or "").strip()
    if not models:
        return None
    names = [m.strip() for m in models.split(",") if m.strip()]
    if len(names) < 2:
        raise ValueError(
            "--compare-models necesita al menos dos modelos separados por comas"
        )
    experiment = _arg("experiment") or DEFAULT_EXPERIMENT
    if experiment not in ("controls", "degradation"):
        raise ValueError(
            "--compare-models solo está disponible para los experimentos "
            "'controls' y 'degradation'"
        )
    argv = ["--models", ",".join(names)]
    argv.extend(
        [
            "--mode",
            "phase2" if experiment == "controls" else "degradation",
        ]
    )
    _append_flag(argv, "--representation", getattr(args, "representation", None))
    _append_flag(argv, "--conditions", getattr(args, "conditions", None))
    _append_flag(argv, "--output", getattr(args, "output", None))
    return argv


# ---------------------------------------------------------------------------
# Planes para --dry-run
# ---------------------------------------------------------------------------
def _n_windows(subject: str) -> int | None:
    """Número de ventanas del sujeto (``None`` si no se puede leer).

    Las ventanas las produce la Fase 1 (``INGEST``), con independencia de la
    fase que se esté planificando.
    """
    path: Path | None = None
    try:
        from sources.common.paths import subject_output_dir

        candidate = (
            subject_output_dir() / "INGEST" / "windows"
            / f"{subject}.windows.jsonl"
        )
        if candidate.is_file():
            path = candidate
    except Exception:  # noqa: BLE001 - el plan es best-effort
        path = None
    if path is None:
        fallback = (
            REPO_ROOT / "results" / "output" / "WESAD" / "INGEST" / "windows"
            / f"{subject}.windows.jsonl"
        )
        path = fallback if fallback.is_file() else None
    if path is None:
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return None


def _subjects_for_plan(args: Any) -> list[str]:
    """Sujetos del plan: ``--subjects`` > ``--subject SX`` > split de evaluación."""
    raw = str(getattr(args, "subjects", None) or "")
    subjects = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if subjects:
        return subjects

    single = str(getattr(args, "subject", None) or "").strip().upper()
    if single.startswith("S") and single[1:].isdigit():
        return [single]

    try:
        from sources.phase2.development_split import TEST_SUBJECTS

        return list(TEST_SUBJECTS)
    except Exception:  # noqa: BLE001 - plan best-effort
        return [single or "default"]


def _estimate_calls(experiment: str, subjects: Sequence[str], args: Any) -> int | None:
    """Estimación de llamadas al LLM (``None`` si no es calculable)."""
    counts = [n for n in (_n_windows(s) for s in subjects) if n is not None]
    if not counts and experiment != "physionet":
        return None
    total_windows = sum(counts)
    cap = getattr(args, "max_windows", None)
    if cap:
        total_windows = min(total_windows, cap * len(subjects))

    conditions_arg = str(getattr(args, "conditions", "") or "")
    n_conditions = (
        len([c for c in conditions_arg.split(",") if c.strip()])
        if conditions_arg
        else None
    )
    if experiment == "controls":
        n_reps = int(
            ((processControl.defaults or {}).get("llm", {}) or {}).get(
                "repetitions", 1
            )
        )
        return (
            total_windows
            * (n_conditions or CONTROL_CONDITIONS)
            * CONTROLS_REPRESENTATIONS
            * max(n_reps, 1)
        )
    if experiment == "degradation":
        return total_windows * (n_conditions or DEGRADATION_CONDITIONS)
    if experiment == "centering":
        return total_windows * (n_conditions or 1)
    if experiment == "physionet":
        per_class = int(getattr(args, "per_class", None) or 150)
        n_records = 4 * per_class
        max_records = getattr(args, "max_records", None)
        if max_records:
            return min(n_records, int(max_records))
        return int(round(n_records * 0.7))  # split test por defecto
    return None


def _plan(phase: str) -> dict[str, Any]:
    """Plan legible para ``--dry-run``."""
    args = processControl.args
    experiment = experiment_of(args)
    model = getattr(args, "model", None) or (
        ((processControl.defaults or {}).get("llm", {}) or {}).get("primary_model")
    )
    plan: dict[str, Any] = {
        "proc": getattr(args, "proc", None),
        "phase": phase,
        "experiment": experiment,
        "subject": getattr(args, "subject", None),
        "subjects": _subjects_for_plan(args),
        "model": model,
        "representation": getattr(args, "representation", None),
        "conditions": getattr(args, "conditions", None) or None,
        "split": getattr(args, "split", None),
        "force": bool(getattr(args, "force", False)),
        "allowDev": bool(getattr(args, "allowDev", False)),
    }
    if phase == "generate":
        plan["think"] = bool(getattr(args, "think", False))
        plan["estimated_llm_calls"] = _estimate_calls(
            experiment, plan["subjects"], args
        )
        if experiment == "controls":
            plan["runner"] = "sources.phase2.orchestrator.run_phase2"
        else:
            plan["runner"] = f"scripts/{_script_for_generate(experiment)}.py"
    else:
        compare = build_compare_argv(args)
        plan["runner"] = (
            "scripts/compare_models.py"
            if compare
            else f"scripts/{_script_for_evaluate(experiment)}.py"
        )
        plan["compare_models"] = getattr(args, "compare_models", "") or None
        if compare:
            # La evaluación de un modelo suelto no aplica: se comparan varios.
            plan["model"] = None
        plan["output"] = getattr(args, "output", None)
    return plan


# ---------------------------------------------------------------------------
# Puntos de entrada
# ---------------------------------------------------------------------------
def _warn_alias(proc: str, replacement: str) -> None:
    """Avisa si se usa un nombre de fase deprecado."""
    if str(proc or "").upper() in ("PHASE2", "PHASE3"):
        writeLog(
            "warning",
            logger,
            f"[CLI] '--proc {proc}' está deprecado: usa '--proc {replacement}' "
            f"(con --experiment cuando corresponda).",
        )


def run_generate() -> None:
    """Fase 2: genera respuestas del LLM para el experimento indicado."""
    args = processControl.args
    _warn_alias(getattr(args, "proc", ""), "GENERATE")
    experiment = experiment_of(args)

    if getattr(args, "dry_run", False):
        print(json.dumps(_plan("generate"), indent=2, ensure_ascii=False))
        return

    if experiment == "controls":
        from sources.phase2.orchestrator import run_phase2

        run_phase2()
        return

    argv = build_generate_argv(experiment, args) or []
    writeLog(
        "info",
        logger,
        f"[GENERATE] experimento={experiment} → "
        f"scripts/{_script_for_generate(experiment)}.py {' '.join(argv)}",
    )
    _run_script(_script_for_generate(experiment), argv)


def run_evaluate() -> None:
    """Fase 3: mide las respuestas y, opcionalmente, compara modelos."""
    args = processControl.args
    _warn_alias(getattr(args, "proc", ""), "EVALUATE")
    experiment = experiment_of(args)

    if getattr(args, "dry_run", False):
        print(json.dumps(_plan("evaluate"), indent=2, ensure_ascii=False))
        return

    compare = build_compare_argv(args)
    if compare is not None:
        writeLog(
            "info",
            logger,
            f"[EVALUATE] comparación de modelos → "
            f"scripts/compare_models.py {' '.join(compare)}",
        )
        _run_script("compare_models", compare)
        return

    script = _script_for_evaluate(experiment)
    argv = build_evaluate_argv(experiment, args)
    writeLog(
        "info",
        logger,
        f"[EVALUATE] experimento={experiment} → scripts/{script}.py "
        f"{' '.join(argv)}",
    )
    _run_script(script, argv)
