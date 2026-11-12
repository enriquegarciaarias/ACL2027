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
import subprocess
import sys
import socket
import warnings

# ``torch`` se importa de forma perezosa (ver ``_torch_module``): importarlo en
# el arranque dispara la inicialización de CUDA y su warning aunque no se use.
_TORCH = None
_TORCH_CHECKED = False

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
    parser.add_argument('--subjects', type=str, help="Subconjunto de sujetos del dataset (p. ej. 'S2,S3'); vacío = todos", default=None)
    parser.add_argument('--force', action='store_true', help="Recalcular artefactos aunque existan con el mismo fingerprint")
    parser.add_argument('--dry-run', action='store_true', help="Resolver configuración y rutas sin ejecutar el pipeline")
    parser.add_argument(
        '--noGate',
        action='store_true',
        help="Desactivar la compuerta de consistencia fisiológica en PHASE2 (ablación)",
    )
    parser.add_argument(
        '--allowDev',
        action='store_true',
        help="Permitir sujetos de desarrollo (S2-S4) en PHASE2; por defecto solo se advierte",
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Modelo LLM para PHASE2 (p. ej. 'qwen3:14b'). Si se omite, se usa "
             "defaults.llm.primary_model de config.json. Cada modelo escribe en "
             "llm_responses/{modelo}/ para no sobrescribir resultados.",
    )
    parser.add_argument(
        '--think',
        action='store_true',
        help="Activar el modo *thinking* del modelo (solo qwen3). Por defecto "
             "está desactivado para mantener comparable el presupuesto de tokens.",
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        choices=[
            'controls', 'degradation', 'centering', 'physionet', 'kemocon',
        ],
        help="Experimento para GENERATE/EVALUATE: controls (C0 + controles "
             "negativos), degradation (C0–C5), centering (E12), physionet o "
             "kemocon. Por defecto: controls.",
    )
    parser.add_argument(
        '--compare-models',
        type=str,
        default='',
        help="EVALUATE: lista de modelos separada por comas; el primero es la "
             "referencia (p. ej. 'qwen2.5:7b,qwen3:14b').",
    )
    parser.add_argument(
        '--representation',
        type=str,
        default=None,
        choices=['numeric', 'text'],
        help="Brazo de representación (experimento degradation).",
    )
    parser.add_argument(
        '--conditions',
        type=str,
        default='',
        help="Condiciones separadas por comas (p. ej. 'C0_clean,C4_total_eda').",
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['evaluation', 'development', 'all'],
        help="Conjunto a evaluar (EVALUATE).",
    )
    parser.add_argument(
        '--max-windows',
        type=int,
        default=None,
        help="Límite de ventanas por sujeto (pilotos).",
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help="Límite de registros (experimento physionet).",
    )
    parser.add_argument(
        '--per-class',
        type=int,
        default=None,
        help="Registros por clase en el muestreo de physionet.",
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help="Tarea del experimento kemocon (p. ej. arousal).",
    )
    parser.add_argument(
        '--tag',
        type=str,
        default=None,
        help="Sufijo de los ficheros de salida (experimentos externos).",
    )
    parser.add_argument(
        '--raw-decisions',
        action='store_true',
        help="EVALUATE controls: usar decision_llm (modelo crudo, sin compuerta).",
    )
    parser.add_argument(
        '--few-shot',
        action='store_true',
        help="Activar few-shot en el experimento physionet.",
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Ruta alternativa del informe de EVALUATE.",
    )
    parser.add_argument(
        '--noEmbeddings',
        action='store_true',
        help="INGEST: no calcular representaciones densas (no las usa la Fase 2).",
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help="INGEST: borrar los artefactos derivados y empezar de cero (no toca results/input).",
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help="Confirmar operaciones destructivas (--fresh).",
    )

    args = parser.parse_args()
    return args


def _torch_module():
    """Importa ``torch`` de forma perezosa.

    Importarlo en el arranque inicializa CUDA y emite su warning aunque el
    proyecto no vaya a usar torch. Aquí solo se importa cuando se consulta el
    hardware.

    Returns:
        El módulo ``torch`` o ``None`` si no está instalado.
    """
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import importlib

            _TORCH = importlib.import_module("torch")
        except Exception as exc:  # pragma: no cover - torch opcional
            writeLog("debug", logger, f"[gpu] torch no disponible: {exc}")
            _TORCH = None
    return _TORCH


def _nvidia_smi(query: str) -> str:
    """Ejecuta ``nvidia-smi`` con una consulta y devuelve la salida cruda.

    Raises:
        RuntimeError: si ``nvidia-smi`` no existe o falla.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"nvidia-smi no disponible: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi falló: {result.stderr.strip()}")
    return result.stdout.strip()


def _detect_all_gpus_nvidia_smi() -> list[dict]:
    """GPUs físicas según ``nvidia-smi`` (no requiere torch).

    Returns:
        Lista de ``{"index", "name", "memory_gb", "driver"}``. Vacía si no hay
        ``nvidia-smi`` o no hay GPUs.
    """
    try:
        output = _nvidia_smi("index,name,memory.total,driver_version")
    except RuntimeError as exc:
        writeLog("debug", logger, f"[gpu] {exc}")
        return []
    gpus: list[dict] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_gb": float(parts[2]) / 1024.0,
                    "driver": parts[3] if len(parts) > 3 else None,
                }
            )
        except ValueError:  # pragma: no cover - salida inesperada
            continue
    return gpus


def _detect_gpus_nvidia_smi(min_memory_gb: float) -> list[dict]:
    """GPUs físicas que superan ``min_memory_gb`` según ``nvidia-smi``."""
    return [
        gpu for gpu in _detect_all_gpus_nvidia_smi()
        if gpu["memory_gb"] >= min_memory_gb
    ]


def _torch_gpus() -> list[dict]:
    """GPUs usables por torch, o lista vacía si no puede inicializar CUDA.

    Silencia el warning de torch cuando el wheel está compilado para una CUDA
    más nueva que el driver: no es un fallo del proyecto y se informa aparte.
    """
    torch = _torch_module()
    if torch is None:
        return []
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
            if not torch.cuda.is_available():
                return []
        return [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "memory_gb": torch.cuda.get_device_properties(index).total_memory
                / (1024 ** 3),
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception as exc:  # pragma: no cover - entorno sin CUDA
        writeLog("debug", logger, f"[gpu] torch no puede usar CUDA: {exc}")
        return []


def gpu_report() -> dict:
    """Informe estructurado del hardware y de lo que el proyecto puede usar.

    Es la fuente única para el log de arranque, ``--dry-run`` y el manifiesto
    del corpus. Distingue tres situaciones que antes se confundían:

    - ``torch`` inicializa CUDA: la GPU es usable desde Python.
    - ``torch`` no puede (wheel/driver) pero ``nvidia-smi`` ve la GPU: el
      hardware existe y Ollama puede usarlo; ``mismatch`` lo señala.
    - No hay GPU: se trabaja en CPU.

    Returns:
        dict con ``torch``, ``nvidia_smi``, ``usable``, ``device``,
        ``mismatch`` y ``reason``.
    """
    torch = _torch_module()
    torch_version = getattr(torch, "version", None)
    build_cuda = getattr(torch_version, "cuda", None) if torch else None
    torch_gpus = _torch_gpus()
    smi_gpus = _detect_all_gpus_nvidia_smi()
    driver = next((gpu.get("driver") for gpu in smi_gpus if gpu.get("driver")), None)

    if torch_gpus:
        usable, device, mismatch = "torch", "cuda", False
        reason = None
        devices = torch_gpus
    elif smi_gpus:
        usable, device, mismatch = "nvidia-smi", "cpu", True
        reason = (
            "la GPU existe pero torch no puede usar CUDA "
            f"(wheel CUDA {build_cuda or 'n/d'}, driver {driver or 'desconocido'}); "
            "la inferencia la hace Ollama"
        )
        devices = smi_gpus
    else:
        usable, device, mismatch = None, "cpu", False
        reason = "sin GPU visible (torch y nvidia-smi)"
        devices = []

    return {
        "torch": {
            "available": torch is not None,
            "version": getattr(torch, "__version__", None) if torch else None,
            "build_cuda": build_cuda,
        },
        "nvidia_smi": {
            "available": bool(smi_gpus),
            "driver": driver,
            "gpus": smi_gpus,
        },
        "usable": usable,
        "device": device,
        "mismatch": mismatch,
        "reason": reason,
        "devices": devices,
    }


def check_gpu() -> None:
    """Detecta GPUs, registra el informe y fija el dispositivo del proceso.

    Fija ``CUDA_VISIBLE_DEVICES`` y actualiza ``processControl.defaults['device']``
    a ``cuda`` o ``cpu``. Cuando la GPU existe pero torch no puede usarla, **no**
    se pone ``CUDA_VISIBLE_DEVICES=-1``: eso la ocultaría a procesos hijos que sí
    puedan usarla (Ollama entre ellos).

    Returns:
        None
    """
    report = gpu_report()

    for gpu in report["devices"]:
        writeLog(
            "info",
            logger,
            f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)",
        )

    processControl.defaults["device"] = report["device"]
    processControl.defaults["gpu"] = {
        key: report[key] for key in ("usable", "device", "mismatch", "reason")
    }

    if report["mismatch"]:
        writeLog(
            "warning",
            logger,
            "[gpu] " + report["reason"] + ". Para que torch la use, instala el "
            "wheel de la CUDA del driver: pip install -r requirements.txt "
            "(el índice de PyTorch está fijado en ese fichero; ver README).",
        )
        return

    if report["usable"] == "torch":
        best = max(report["devices"], key=lambda gpu: gpu["memory_gb"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
        writeLog(
            "info",
            logger,
            f"Selected GPU {best['index']} ({best['memory_gb']:.1f} GB)",
        )
        return

    writeLog("info", logger, "Sin GPU visible (torch y nvidia-smi): se usa CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    check_gpu()


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


def _config_document(config=None):
    """Normaliza un ``configLoader`` o un diccionario a documento de config."""
    if config is None:
        return configLoader().config
    if isinstance(config, configLoader):
        return config.config
    return config


def _environment(config, base_path):
    """Copia la sección ``environment`` y le añade ``realPath``."""
    environment = dict(config.get("environment", {}))
    environment["realPath"] = base_path
    return environment


def manageEnv(config=None):
    """Construye el diccionario de rutas y variables del entorno.

    Lee la sección `environment` de la configuración, resuelve las rutas
    relativas contra `realPath`, soporta `results_root` (ruta alternativa
    para results/), crea el directorio de caché y añade el nombre del
    sistema vía `socket.getfqdn()`.

    Args:
        config: documento de configuración ya fusionado (opcional). Si no se
            indica, se carga `config.json` con `configLoader`.

    Returns:
        dict: Diccionario con las rutas de entorno y el nombre del sistema.
    """
    # Suprimir warnings ruidosos de neurokit2 (EDA a baja frecuencia de
    # muestreo). Son informativos y no bloqueantes; se documentan en
    # SESSION_STATE.md. Se filtran aquí para evitar que ensucien el log
    # una vez por ventana y sujeto.
    warnings.filterwarnings(
        "ignore",
        message=".*very low frequency.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*missing data points in your signal.*",
    )

    config = _config_document(config)
    base_path = os.path.realpath(os.getcwd())
    environment = _environment(config, base_path)

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
    # Los directorios de entrada/salida pueden no existir en un clon reciente
    # (results/ está gitignorado): se crean para que el fallo sea el dataset
    # ausente, con un mensaje claro, y no un FileNotFoundError genérico.
    for key in ('input', 'output'):
        path = env_data.get(key)
        if path:
            os.makedirs(path, exist_ok=True)
    os.environ['PYTHONPYCACHEPREFIX'] = env_data['cache']
    try:
        sys.pycache_prefix = env_data['cache']
    except Exception:
        pass
    env_data['systemName'] = socket.getfqdn()
    return env_data


def manageDefaults(config=None):
    """Carga la sección de valores por defecto de la configuración.

    Returns:
        dict: Valores por defecto definidos en `config.json`.
    """
    config = _config_document(config)
    return config.get("defaults", {})


def manageDatasetVars(config=None):
    """Carga las variables del dataset y añade un timestamp.

    Returns:
        dict: Variables del dataset con el campo `timestamp` añadido.
    """
    config = _config_document(config)
    datasetVars = dict(config.get("datasetVars", {}))
    datasetVars['timestamp'] = dbTimestamp()
    return datasetVars


def load_subject_config(subject, base_config=None):
    """Fusiona la configuración del subject sobre la configuración global.

    Busca `results/input/{subject}/config.json` y, si existe, hace un merge
    profundo del documento completo (``environment``, ``defaults``, ``params``
    y ``datasetVars``); las claves del subject sobrescriben las globales.

    Args:
        subject (str): nombre de la carpeta en ``results/input``.
        base_config: documento global (``configLoader`` o dict). Si no se
            indica, se carga ``config.json``.

    Returns:
        dict: Documento de configuración fusionado (global + subject).
    """
    from pathlib import Path

    if base_config is None:
        base_config = configLoader()
    base_doc = _config_document(base_config)

    # Resolver la ruta de entrada con la configuración global (results_root
    # incluido) antes de que processControl.env esté poblado.
    global_env = manageEnv(base_doc)
    input_root = Path(global_env.get("input", "results/input"))
    subject_cfg_path = input_root / str(subject) / "config.json"

    if not subject_cfg_path.exists():
        return base_doc

    try:
        import json
        with subject_cfg_path.open("r", encoding="utf-8") as f:
            subject_cfg = json.load(f)
        merged = deep_merge(base_doc, subject_cfg)
        writeLog("info", logger, f"Configuración de subject '{subject}' fusionada (deep merge).")
        return merged
    except Exception as exc:
        writeLog("warning", logger, f"No se pudo cargar config de subject '{subject}': {exc}")
        return base_doc


def _require_config_file(base_path: str | None = None) -> str:
    """Comprueba que exista ``config.json`` en el directorio de trabajo.

    Falla de forma explícita (código 2) en lugar de dejar un
    ``FileNotFoundError`` sin mensaje en el log: es el error más habitual en un
    clon reciente del repositorio, donde ``config.json`` está gitignorado.

    Returns:
        Ruta absoluta de ``config.json``.
    """
    base = base_path or os.path.realpath(os.getcwd())
    path = os.path.join(base, "config.json")
    if not os.path.isfile(path):
        writeLog(
            "error",
            logger,
            f"No existe {path}. Créalo con: cp config_example.json config.json",
        )
        raise SystemExit(2)
    return path


def getConfigs():
    """Carga la configuración completa en el singleton `processControl`.

    Puebla `processControl.config` (global + subject), `processControl.env`,
    `processControl.args`, `processControl.defaults`,
    `processControl.parms` y `processControl.datasetVars`, y aplica la
    configuración del entorno (GPU, variables de entorno).

    Returns:
        None
    """
    processControl.args = manageArgs()

    _require_config_file()

    base_config = configLoader()
    merged = load_subject_config(processControl.args.subject, base_config)

    processControl.config = merged
    processControl.env = manageEnv(merged)
    processControl.defaults = manageDefaults(merged)
    processControl.parms = dict(merged.get("params", {}))
    processControl.datasetVars = manageDatasetVars(merged)

    setEnvironment()
    writeLog("info", logger, "Configuration loaded.")
