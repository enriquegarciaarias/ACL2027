"""Rutas por defecto de los datos de los experimentos externos.

Los datos viven **dentro del proyecto** (``results/input/EXTERNAL/``), de modo
que la réplica en otro equipo no dependa de rutas de otros repositorios ni de
directorios temporales. Se centralizan aquí para que los runners, el despachador
y el preflight de ``--dry-run`` no puedan divergir.

Cada ruta puede sobreescribirse con ``--data-dir`` o con
``config.json > params.external.<experimento>.data_dir``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Raíz de datos de entrada externos.
EXTERNAL_ROOT: Path = REPO_ROOT / "results" / "input" / "EXTERNAL"

#: PhysioNet/CinC 2017: ``.mat``/``.hea`` + ``REFERENCE.csv``.
PHYSIONET2017_DIR: Path = EXTERNAL_ROOT / "physionet2017" / "training2017"

#: K-EmoCon: tablas derivadas (features y asignación de periodos).
KEMOCON_DIR: Path = EXTERNAL_ROOT / "KEMOCON"

#: DREAMT: dataset completo (descargado aparte por su licencia restringida).
#: No vive bajo ``EXTERNAL_ROOT`` porque se distribuye como un ZIP con su
#: propia estructura y licencia PhysioNet Restricted 1.5.0.
DREAMT_DIR: Path = REPO_ROOT / "results" / "input" / "dreamt"

#: Fichero o directorio que debe existir para considerar utilizable cada
#: dataset externo.
REQUIRED_FILES: dict[str, str] = {
    "physionet": "REFERENCE.csv",
    "kemocon": "kemocon_feature_table.csv",
    "dreamt": "data_64Hz",
}

#: Directorio por defecto de cada experimento externo.
DEFAULT_DIRS: dict[str, Path] = {
    "physionet": PHYSIONET2017_DIR,
    "kemocon": KEMOCON_DIR,
    "dreamt": DREAMT_DIR,
}
