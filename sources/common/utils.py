"""
Utilidades genéricas compartidas por todos los pipelines del proyecto.

Contiene funciones auxiliares de propósito general: hashing y normalización de
texto, gestión de directorios, marcas de tiempo, carga/serialización de JSON,
la clase `configLoader` para acceder a la configuración y la utilidad
`inicioModulo` que construye las rutas de entrada/salida de cada proceso.

Esqueleto reutilizable (patrón común heredado de PSRI y SLIP, con mejoras
propuestas incorporadas: resolución de rutas vía `processControl.env` sin
hardcodear "results", y soporte a `results_root`).
"""
from sources.common.common import logger, processControl, writeLog

import json
import time
import os
from os.path import isdir

import unicodedata
import re
import hashlib
from pathlib import Path
from json import JSONDecodeError


def sha1(text: str) -> str:
    """Genera un hash SHA1 a partir de un texto."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256(text: str) -> str:
    """Genera un hash SHA-256 a partir de un texto (útil para checksums de corpus)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def mkdir(dir_path):
    """Crea un directorio si no existe."""
    if not isdir(dir_path):
        os.makedirs(dir_path)


def safe_filename(name: str) -> str:
    """Convierte una cadena en un nombre de archivo seguro.

    Elimina tildes y acentos, sustituye espacios por guiones bajos, elimina
    cualquier carácter no alfanumérico (salvo guión y subrayado) y pasa el
    resultado a minúsculas.
    """
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name.lower()


def dbTimestamp():
    """Genera una marca de tiempo con formato YYYYMMDDHHMMSS (UTC)."""
    timestamp = int(time.time())
    formatted_timestamp = str(time.strftime("%Y%m%d%H%M%S", time.gmtime(timestamp)))
    return formatted_timestamp


def normalized_title(text: str) -> str:
    """Normaliza un título a minúsculas y sin signos diacríticos."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    """Normaliza un texto a una única línea sin espacios redundantes."""
    if not text:
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_doi(doi):
    """Normaliza un DOI a minúsculas y sin espacios."""
    if not doi:
        return ""
    return doi.strip().lower()


def inicioModulo(modulo):
    """Construye las rutas de entrada y salida de un proceso.

    Las rutas se derivan de `processControl.env` (`input`/`output`) y del
    argumento `processControl.args.proc`, según el patrón
    `{input|output}/{PROC}`. Todas las rutas se resuelven vía
    `processControl.env` (poblado por `paramsManager.manageEnv()`); está
    prohibido hardcodear "results" o "results/input" en las rutas de archivos.

    Args:
        modulo (str): Nombre del módulo/proceso, usado solo en el log.

    Returns:
        tuple: (Path de entrada, Path de salida).
    """
    writeLog("info", logger, "-" * 60)
    writeLog("info", logger, f"🚀 [START] Processing {modulo}")
    base_input_dir = Path(processControl.env.get("input", "")) / processControl.args.proc
    base_output_dir = Path(processControl.env.get("output", "")) / processControl.args.proc

    return base_input_dir, base_output_dir


def read_json(filepath):
    """Lee y parsea un fichero JSON.

    Raises:
        FileNotFoundError, IsADirectoryError, ValueError, OSError.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Expected a file, got: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            writeLog("info", logger, f"📄 JSON loaded from {filepath}")
            return json.load(f)
    except JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in '{path}': {e}") from e


def write_json(filepath, data, *, indent=4, ensure_ascii=False):
    """Escribe datos en un fichero JSON, creando directorios padre si faltan."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.write("\n")
        writeLog("info", logger, f"💾 JSON written to {path}")
    except TypeError as e:
        raise TypeError(f"Object is not JSON serializable: {path}") from e
    except OSError as e:
        raise OSError(f"Could not write JSON file: {path}") from e


class configLoader:
    """Carga y proporciona acceso a la configuración JSON del proyecto.

    Attributes:
        base_path (str): Ruta absoluta del directorio de trabajo.
        config (dict): Contenido del fichero JSON de configuración.
    """

    def __init__(self, config_path='config.json'):
        self.base_path = os.path.realpath(os.getcwd())
        realConfigPath = os.path.join(self.base_path, config_path)
        self.config = self.load_config(realConfigPath)

    def load_config(self, realConfigPath):
        """Lee y parsea el fichero JSON de configuración."""
        with open(realConfigPath, 'r', encoding='utf-8') as config_file:
            return json.load(config_file)

    def get_environment(self):
        """Devuelve la sección `environment` con `realPath` añadido."""
        environment = self.config.get("environment", {})
        environment["realPath"] = self.base_path
        return environment

    def get_defaults(self):
        """Devuelve la sección `defaults` de la configuración."""
        return self.config.get("defaults", {})

    def get_search(self):
        """Devuelve la sección `search` de la configuración."""
        return self.config.get("search", {})

    def get_datasetVars(self):
        """Devuelve la sección `datasetVars` de la configuración."""
        return self.config.get("datasetVars", {})

    def get_params(self):
        """Devuelve la sección `params` de la configuración."""
        return self.config.get("params", {})

    def get_llm(self):
        """Devuelve la sección `llm` dentro de `defaults`."""
        return self.config.get("defaults", {}).get("llm", {})
