"""
Punto de entrada principal del proyecto.

Este módulo inicializa el entorno, carga la configuración global y lanza el
proceso correspondiente al argumento `--proc`. Cada pipeline se importa y
ejecuta de forma diferida dentro de su propia función para evitar
dependencias circulares y cargar únicamente lo necesario.

Soporta:
  - --proc: tipo de proceso (despacha al pipeline correspondiente).
  - --step: iniciar el pipeline directamente desde un paso con nombre
    (case-insensitive) y continuar hasta el final (mejora propuesta).
  - --serverGPU: ejecutar solo la parte interactiva local ("start") o solo
    la parte no interactiva en servidor GPU ("server") (mejora propuesta).

Códigos de salida: 0 correcto · 1 fallo de ejecución · 2 falta config.json ·
3 entorno incompleto (venv o dependencias) · 130 interrumpido.

Esqueleto reutilizable (patrón común heredado de PSRI y SLIP).
"""
from sources.common import environment
from sources.common.common import processControl, logger, writeLog
from sources.common.paramsManager import getConfigs

# ---------------------------------------------------------------------------
# Registro de procesos disponibles.
# Añade aquí las entradas de cada pipeline del proyecto:
#   "NOMBREPROC": "modulo.funcion"
# La función se importa y ejecuta de forma diferida.
# ---------------------------------------------------------------------------
PROCESS_REGISTRY: dict[str, str] = {
    # Fase 1: señal -> corpus (no depende del modelo).
    "INGEST": "sources.ingestion.pipeline.process_ingest",
    # Fase 2: prompts + respuestas del LLM (experimento en --experiment).
    "GENERATE": "sources.pipelines.run_generate",
    # Fase 3: evaluación y comparación de modelos.
    "EVALUATE": "sources.pipelines.run_evaluate",
    # Alias deprecados (misma maquinaria, experimento ``controls``).
    "PHASE2": "sources.pipelines.run_generate",
    "PHASE3": "sources.pipelines.run_evaluate",
}


def _run_registered_process(proc_name: str):
    """Ejecuta el proceso registrado para `proc_name` de forma diferida.

    Args:
        proc_name (str): Nombre del proceso (clave de PROCESS_REGISTRY).

    Raises:
        ValueError: Si el proceso no está registrado.
    """
    if proc_name not in PROCESS_REGISTRY:
        available = ", ".join(PROCESS_REGISTRY.keys()) or "(ninguno)"
        writeLog("error", logger,
                 f"❌ Proceso '{proc_name}' no registrado. Disponibles: {available}")
        raise ValueError(f"Proceso '{proc_name}' no válido. Disponibles: {available}")

    import_module_path, funcion = PROCESS_REGISTRY[proc_name].rsplit(".", 1)
    from importlib import import_module
    module = import_module(import_module_path)
    fn = getattr(module, funcion)
    fn()


def customProcess():
    """Proceso por defecto cuando `--proc` no es un valor conocido.

    Punto de extensión vacío. El proyecto concreto debe sobrescribir este
    comportamiento si necesita procesos personalizados.

    Returns:
        None
    """
    writeLog("info", logger,
             f"customProcess() para proc='{processControl.args.proc}' no implementado.")
    return


def _check_environment_or_exit() -> None:
    """Verifica venv y dependencias antes de cargar la configuración.

    Evita el error típico de lanzar el pipeline con el Python del sistema:
    faltaría `neurokit2`/`scipy` y el fallo aparecería dentro de un import.
    """
    ok, errors, warnings = environment.check_environment()
    for message in warnings:
        writeLog("warning", logger, message)
    if not ok:
        for message in errors:
            writeLog("error", logger, message)
        raise SystemExit(3)


def _run() -> int:
    """Ejecuta el proceso y devuelve un código de salida explícito."""
    _check_environment_or_exit()
    getConfigs()
    writeLog("info", logger,
             f"********** STARTING {processControl.args.proc.upper() if processControl.args and processControl.args.proc else 'PROJECT'} Process **********")

    proc = getattr(processControl.args, "proc", None)
    step = getattr(processControl.args, "step", None)

    # El módulo concreto gestiona --step internamente (p. ej. INGEST).
    if step is not None:
        writeLog("debug", logger,
                 f"--step '{step}' delegado al proceso '{proc}'.")

    if proc in PROCESS_REGISTRY:
        _run_registered_process(proc)
    else:
        customProcess()

    writeLog("info", logger, "********** PROCESS COMPLETED **********")
    return 0


def _shape_hint(exc: BaseException) -> str:
    """Sugerencia accionable para los fallos de preparación más frecuentes."""
    text = str(exc)
    if isinstance(exc, FileNotFoundError):
        if "config.json" in text:
            return (
                "Falta config.json en el directorio de trabajo. Créalo con: "
                "cp config_example.json config.json"
            )
        if "results/input" in text or "directorio del dataset" in text:
            return (
                "Falta el dataset. Estructura esperada: "
                "results/input/<dataset>/<sujeto>/<sujeto>.pkl "
                "(revisa --subject y la sección environment de config.json)."
            )
    return ""


if __name__ == '__main__':
    import sys
    import traceback

    try:
        code = _run()
    except KeyboardInterrupt:
        writeLog("warning", logger, "Ejecución interrumpida por el usuario.")
        code = 130
    except SystemExit as exc:  # salidas controladas (p. ej. preflight)
        code = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:  # noqa: BLE001 - se registra y se propaga el fallo
        writeLog(
            "error",
            logger,
            f"Fallo no controlado: {type(exc).__name__}: {exc}",
        )
        hint = _shape_hint(exc)
        if hint:
            writeLog("error", logger, hint)
        # Traza completa en el log (el wrapper del servidor solo ve stdout).
        writeLog("error", logger, traceback.format_exc())
        code = 1
    sys.exit(code)
