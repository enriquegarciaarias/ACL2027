"""
Gestión de parámetros y configuración global del proyecto.

Este módulo centraliza el arranque de la configuración: parsea los argumentos
de línea de comandos, carga la configuración JSON, detecta la GPU disponible,
configura el entorno (variables de entorno, caché de Python) y rellena el
singleton `processControl` con todos los valores necesarios para cualquier
pipeline.

Soporta carga de configuración global (`config.json`) con opción de merge
profundo de configuración por subject/asunto de investigación.

Patrón común heredado de PSRI y SLIP, con mejoras propuestas incorporadas.
"""
from sources.common.common import processControl, logger, writeLog
from sources.common.utils import configLoader, dbTimestamp

import argparse
import os
import sys
import socket

# Intentar detección de GPU (opcional: torch puede no estar instalado en todos
# los entornos; se degrada graceful a CPU si no está disponible).
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch opcional
    _TORCH_AVAILABLE = False

# Constants for parameter files
JSON_PARMS = "config.json"


def manageArgs():
    """Parsea los argumentos de línea de comandos del proceso principal.

    Returns:
        argparse.Namespace: Argumentos parseados con los atributos
            `subject`, `proc`, `step` y `serverGPU`.
    """
    parser = argparse.ArgumentParser(
        description="Punto de entrada del proyecto. Gestiona modos de proceso y ejecución por pasos."
    )
    parser.add_argument('--subject', type=str, help="Subject/investigación (carpeta en results/input/)", default="default")
    parser.add_argument('--proc', type=str, help="Tipo de proceso (despacha al pipeline correspondiente)", default="default")
    parser.add_argument('--step', type=str, help="Iniciar el pipeline directamente desde este paso (nombre case-insensitive)", default=None)
    parser.add_argument('--serverGPU', type=str, help="Modo dividido local↔servidor GPU: 'start' (parte interactiva local) o 'server' (parte GPU no interactiva)", default=None)

    args = parser.parse_args()
    return args


def check_gpu(min_memory_gb=8.0):
    """Detecta GPUs CUDA disponibles con memoria suficiente.

    Filtra GPUs con memoria total >= `min_memory_gb`, fija la variable de
    entorno `CUDA_VISIBLE_DEVICES` y actualiza
    `processControl.defaults['device']` a 'cuda' o 'cpu'.

    Returns:
        None
    """
    suitable_gpus = []
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        writeLog("info", logger, f"{num_gpus} CUDA devices available")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            if memory_gb >= min_memory_gb:
                suitable_gpus.append(i)
                writeLog("info", logger, f"GPU {i} suitable: {props.name} ({memory_gb:.1f} GB)")
            else:
                writeLog("info", logger, f"GPU {i} skipped: {props.name} ({memory_gb:.1f} GB < {min_memory_gb} GB)")
    else:
        writeLog("info", logger, "No CUDA devices available (torch not installed or no CUDA)")

    if suitable_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, suitable_gpus))
        processControl.defaults['device'] = 'cuda'
        writeLog("info", logger, f"Selected GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        processControl.defaults['device'] = 'cpu'
        writeLog("info", logger, "No suitable GPUs found; falling back to CPU")


def huggingface_login():
    """Autentica contra Hugging Face usando el token de configuración.

    Usa el token almacenado en `processControl.defaults['huggingFaceToken']`.
    Se emplea para descargar modelos privados del hub.

    Returns:
        None

    Raises:
        Exception: Si la autenticación falla, se registra el error y se relanza.
    """
    try:
        token = (
            processControl.defaults['huggingFaceToken']
            if 'huggingFaceToken' in processControl.defaults
            else ''
        )
        from huggingface_hub import login
        login(token)
        writeLog("info", logger, "Successfully logged in to Hugging Face.")
    except Exception as e:
        writeLog("error", logger, f"Error logging into Hugging Face {str(e)}")
        raise


def setEnvironment():
    """Configura variables de entorno y caché de Python para el proceso.

    Fija `TOKENIZERS_PARALLELISM=false`, reubica la caché de bytecode de
    Python (`PYTHONPYCACHEPREFIX`) y detecta la GPU disponible.

    Returns:
        None
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cache = processControl.env.get('cache', os.path.expanduser('~/.cache'))
    os.environ['PYTHONPYCACHEPREFIX'] = cache
    os.makedirs(cache, exist_ok=True)
    try:
        sys.pycache_prefix = cache
    except Exception:
        pass

    min_memory = float(processControl.defaults.get('min_gpu_memory_gb', 8.0))
    check_gpu(min_memory_gb=min_memory)


def deep_merge(base, override):
    """Realiza un merge profundo de diccionarios.

    Las claves de `override` sobrescriben las de `base`. Los diccionarios
    anidados se fusionan recursivamente en lugar de reemplazarse por completo.

    Returns:
        dict: Diccionario fusionado.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def manageEnv():
    """Construye el diccionario de rutas y variables del entorno.

    Lee la sección `environment` de la configuración, resuelve las rutas
    relativas contra `realPath`, soporta `results_root` (ruta alternativa
    para results/), crea el directorio de caché y añade el nombre del
    sistema vía `socket.getfqdn()`.

    Returns:
        dict: Diccionario con las rutas de entorno y el nombre del sistema.
    """
    config = configLoader()
    environment = config.get_environment()

    env_data = {}
    results_root = environment.get("results_root", "") or ""

    for key, value in environment.items():
        if key in ("realPath", "results_root"):
            env_data[key] = value
            continue

        if results_root:
            if os.path.isabs(results_root):
                env_data[key] = os.path.join(results_root, value)
            else:
                env_data[key] = os.path.join(environment.get("realPath", ""), results_root, value)
        else:
            env_data[key] = os.path.join(environment["realPath"], value)

    env_data['cache'] = os.path.join(env_data.get('cache', '.pycache'))
    os.makedirs(env_data['cache'], exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['cache']
    try:
        sys.pycache_prefix = env_data['cache']
    except Exception:
        pass
    env_data['systemName'] = socket.getfqdn()
    return env_data


def manageDefaults():
    """Carga la sección de valores por defecto de la configuración.

    Returns:
        dict: Valores por defecto definidos en `config.json`.
    """
    config = configLoader()
    environment = config.get_defaults()
    return environment


def manageDatasetVars():
    """Carga las variables del dataset y añade un timestamp.

    Returns:
        dict: Variables del dataset con el campo `timestamp` añadido.
    """
    config = configLoader()
    datasetVars = config.get_datasetVars()
    datasetVars['timestamp'] = dbTimestamp()
    return datasetVars


def load_subject_config(subject):
    """Carga la configuración específica de un subject si existe.

    Busca `results/input/{subject}/config.json` y, si existe, hace un merge
    profundo sobre los valores por defecto globales (las claves del subject
    sobrescriben las globales).

    Returns:
        dict: Configuración fusionada (global + subject).
    """
    from pathlib import Path

    config = configLoader()
    defaults = config.get_defaults()

    env = processControl.env
    input_root = Path(env.get("input", "results/input"))
    subject_cfg_path = input_root / subject / "config.json"

    if subject_cfg_path.exists():
        try:
            import json
            with subject_cfg_path.open("r", encoding="utf-8") as f:
                subject_cfg = json.load(f)
            subject_defaults = subject_cfg.get("defaults", {})
            merged = deep_merge(defaults, subject_defaults)
            writeLog("info", logger, f"Configuración de subject '{subject}' fusionada (deep merge).")
            return merged
        except Exception as exc:
            writeLog("warning", logger, f"No se pudo cargar config de subject '{subject}': {exc}")
    return defaults


def getConfigs():
    """Carga la configuración completa en el singleton `processControl`.

    Puebla `processControl.env`, `processControl.args`,
    `processControl.defaults` y `processControl.datasetVars`, y aplica la
    configuración del entorno (GPU, variables de entorno).

    Returns:
        None
    """
    processControl.env = manageEnv()
    processControl.args = manageArgs()

    processControl.defaults = manageDefaults()
    processControl.datasetVars = manageDatasetVars()

    setEnvironment()
    writeLog("info", logger, "Configuration loaded.")
