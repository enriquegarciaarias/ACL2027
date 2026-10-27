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

Esqueleto reutilizable (patrón común heredado de PSRI y SLIP).
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.paramsManager import getConfigs

# ---------------------------------------------------------------------------
# Registro de procesos disponibles.
# Añade aquí las entradas de cada pipeline del proyecto:
#   "NOMBREPROC": "modulo.funcion"
# La función se importa y ejecuta de forma diferida.
# ---------------------------------------------------------------------------
PROCESS_REGISTRY: dict[str, str] = {
    # Ejemplo (descomentar y adaptar al proyecto concreto):
    # "ACL2027": "sources.acl2027.pipeline.process_acl2027",
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


if __name__ == '__main__':
    writeLog("info", logger,
             f"********** STARTING {processControl.args.proc.upper() if processControl.args and processControl.args.proc else 'PROJECT'} Process **********")
    getConfigs()

    proc = getattr(processControl.args, "proc", None)
    step = getattr(processControl.args, "step", None)

    # Si se indica --step, iniciar desde ese paso (solo aplica a procesos con
    # registro de pasos; el esqueleto base lo registra como aviso).
    if step is not None:
        writeLog("warning", logger,
                 f"--step '{step}' indicado pero el proceso '{proc}' no define "
                 f"registro de pasos. Ejecutando proceso completo.")
        # Se ejecuta el proceso completo; el módulo concreto puede leer
        # processControl.args.step internamente.

    if proc in PROCESS_REGISTRY:
        _run_registered_process(proc)
    else:
        customProcess()

    writeLog("info", logger, "********** PROCESS COMPLETED **********")
