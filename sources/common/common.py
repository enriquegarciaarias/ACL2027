"""
Utilidades globales de logging y control de proceso.

Este módulo configura el sistema de logging del proyecto (consola con color y
archivo rotatorio), define el objeto global de control de proceso
(`processControl`, un singleton) y expone las funciones auxiliares para
escribir mensajes de log. Es el punto común del que dependen todos los demás
módulos del proyecto.

Esqueleto reutilizable (patrón común heredado de PSRI y SLIP, con mejoras
propuestas incorporadas).
"""
from sources.common import global_vars

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from colorama import just_fix_windows_console

# ----------------------------------------------------------------------
# Soporte ANSI (Windows/Linux/macOS)
# ----------------------------------------------------------------------

just_fix_windows_console()

USE_COLORS = (
    sys.stderr.isatty()
    and os.getenv("TERM") not in (None, "dumb")
)

# ----------------------------------------------------------------------
# Global process control (singleton)
# ----------------------------------------------------------------------


class controlProcess:
    """Contenedor global de la configuración y estado del proceso.

    Instancia única accesible vía `processControl`. Almacena los argumentos de
    línea de comandos, los valores por defecto, las variables del dataset,
    los parámetros del entorno y las rutas resueltas.

    Attributes:
        datasetVars (dict): Variables específicas del dataset/proyecto.
        args (argparse.Namespace): Argumentos parseados de línea de comandos.
        defaults (dict): Valores por defecto cargados de la configuración.
        parms (dict): Parámetros adicionales del proceso.
        env (dict): Rutas y variables del entorno resueltas por paramsManager.
    """

    def __init__(self, datasetVars=None, args=None, defaults=None, parms=None, env=None):
        self.datasetVars = datasetVars or {}
        self.args = args or {}
        self.defaults = defaults or {}
        self.parms = parms or {}
        self.env = env or {}

    def to_dict(self):
        """Serializa el estado del control de proceso en un diccionario."""
        return {
            "datasetVars": self.datasetVars,
            "args": self.args,
            "defaults": self.defaults,
            "parms": self.parms,
            "env": self.env,
        }


global_vars.procCtrl = controlProcess()
processControl = global_vars.procCtrl

# ----------------------------------------------------------------------
# Colores
# ----------------------------------------------------------------------

COLORS = {
    "DEBUG": "\033[36m",       # Cyan
    "INFO": "\033[92m",        # Green
    "WARNING": "\033[93m",     # Yellow
    "ERROR": "\033[91m",       # Red
    "CRITICAL": "\033[95m",    # Magenta
}

RESET = "\033[0m"

# Iconos para logs sin color (útil en ficheros sbatch out)
ICONS = {
    "warning": "⚠️  ",
    "error": "❌  ",
    "info": "ℹ️  ",
    "debug": "🔍  ",
    "critical": "🚨  ",
}

# ----------------------------------------------------------------------
# Formatter
# ----------------------------------------------------------------------


class ColoredFormatter(logging.Formatter):
    """Formateador de logs que colorea el nombre del nivel en consola.

    Aplica el código ANSI del color correspondiente al nivel solo al nombre
    del nivel, y lo restaura tras formatear para que los handlers de archivo
    no reciban secuencias ANSI.
    """

    def format(self, record):
        original_level = record.levelname

        if USE_COLORS and original_level in COLORS:
            record.levelname = f"{COLORS[original_level]}{original_level:<8}{RESET}"
        else:
            record.levelname = f"{original_level:<8}"

        message = super().format(record)

        record.levelname = original_level
        return message


# ----------------------------------------------------------------------
# Logger configuration
# ----------------------------------------------------------------------


def configureLogger(log_type="log", logger_name="projectLog"):
    """Configura y devuelve un logger con handler de archivo y consola.

    Args:
        log_type (str): "log" escribe también en consola (ProcessLog.txt);
            cualquier otro valor solo escribe en archivo (Process.txt).
        logger_name (str): Nombre del logger.

    Returns:
        logging.Logger: Logger configurado. Idempotente si ya tiene handlers.
    """
    log_file = "./ProcessLog.txt" if log_type == "log" else "./Process.txt"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (solo para log_type == "log")
    if log_type == "log":
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# ----------------------------------------------------------------------
# Logging helper
# ----------------------------------------------------------------------


def writeLog(level: str, logger: logging.Logger, message: str):
    """Escribe un mensaje en el logger en el nivel indicado.

    Añade iconos visuales a los niveles warning/error para facilitar la
    lectura en logs sin color (ficheros de salida de servidores).

    Args:
        level (str): Nivel de log ("debug", "info", "warning", "error",
            "critical").
        logger (logging.Logger): Logger sobre el que escribir.
        message (str): Mensaje a registrar.
    """
    level = level.lower()
    if not hasattr(logger, level):
        logger.error(f"Invalid log level '{level}'")
        return

    # Añadir icono visual para niveles clave en salida sin color
    icon = ICONS.get(level, "")
    if icon:
        message = f"{icon}{message}"

    getattr(logger, level)(message)


# ----------------------------------------------------------------------
# Global loggers
# ----------------------------------------------------------------------

logger = configureLogger("log", "projectLog")
logProc = configureLogger("proc", "projectProc")
