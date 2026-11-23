"""Adaptador de DREAMT para el piloto de la segunda tarea (estadios de sueño).

DREAMT no comparte instrumentación con WESAD ni comparte tarea: aporta la
Empatica E4 (BVP, ACC, TEMP, EDA, HR, IBI) sincronizada con una puntuación de
polisomnografía (PSG) de estadios de sueño y eventos de apnea. **No tiene
ninguna etiqueta de estrés ni afectiva**; la tarea es de estadios de sueño.

Este módulo es la única pieza que conoce el formato del dataset. Hace cinco
cosas y ninguna más:

1. Lee el CSV de 64 Hz del sujeto con las columnas que hacen falta
   (``usecols``), porque el fichero completo tiene 2 M de filas y tipos mixtos.
2. Ancla la rejilla de épocas de 30 s **en la primera muestra con etiqueta
   distinta de ``P``**: las épocas están puntuadas desde el inicio de la noche
   y no desde la primera fila del CSV. Anclar en la fila 0 produce épocas
   «mixtas» que no existen (hasta un 27 % de falsos positivos).
3. Construye ventanas de 60 s con pureza total (dos épocas consecutivas del
   mismo estadio) y muestrea por clase repartiendo por tercios de la noche.
4. Extrae las features canónicas del proyecto reutilizando los extractores de
   Fase 1. La E4 se mapea a las ranuras de *pecho* (``chest_EDA``,
   ``chest_Temp``, ``chest_ACC``) para que la vista del LLM no la descarte como
   modalidad de muñeca, y el bloque ``ecg`` se **deriva de la columna ``HR``**
   del dispositivo (1 Hz) porque el dataset no trae onda de ECG en la rama
   *wearable* y ni ``IBI`` ni ``BVP`` permiten HRV (desviación D7).
5. Aplica la matriz de degradación del preregistro: ausencia (NaN) frente a
   degeneración (valores fisiológicamente imposibles).

Convenciones del dataset que **no** son las del ``Metadata.txt``:

- ``IBI`` está en **segundos** (0,469–1,375), no en milisegundos.
- ``ACC`` viene en cuentas de **1/64 g**; se divide por 64.
- ``EDA`` y ``TEMP`` son señales de 4 Hz retenidas a 64 Hz (16 muestras por
  valor); ``ACC`` es de 32 Hz retenida (2 muestras por valor). Se deciman para
  que los extractores reciban la frecuencia real.

Documentación y preregistro: ``docs/piloto_dreamt.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sources.external.paths import DREAMT_DIR

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------

#: Raíz por defecto del dataset (``results/input/dreamt``).
DREAMT_ROOT: Path = DREAMT_DIR

#: Rama de datos usada por el piloto: solo E4 **y las etiquetas PSG**, 14 GB.
WEARABLE_DIRNAME = "data_64Hz"

#: Plantilla del fichero por sujeto.
SUBJECT_TEMPLATE = "{subject}_whole_df.csv"

#: Frecuencia de la rejilla del CSV (Hz).
FS = 64.0

#: Duración de una época puntuada (s) y filas que ocupa.
EPOCH_S = 30.0
EPOCH_ROWS = int(EPOCH_S * FS)  # 1920

#: Duración de la ventana del piloto (s) y épocas que ocupa.
WINDOW_S = 60.0
WINDOW_EPOCHS = int(WINDOW_S / EPOCH_S)  # 2

#: Columnas leídas. Nunca se carga el CSV completo.
USECOLS: tuple[str, ...] = (
    "TIMESTAMP",
    "BVP",
    "EDA",
    "TEMP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "HR",
    "IBI",
    "Sleep_Stage",
)

#: Estadios posibles en la columna ``Sleep_Stage``.
STAGES: tuple[str, ...] = ("P", "W", "N1", "N2", "N3", "R")

#: Estadios que forman el espacio de decisión (N3 se excluye, ver §6 del
#: preregistro: 5,3 % de las épocas y presente en 6 de 13 sujetos).
TASK_STAGES: tuple[str, ...] = ("W", "N1", "N2", "R")

#: Estadios descartados siempre.
EXCLUDED_STAGES: tuple[str, ...] = ("P", "N3", "Missing", "", "nan")

#: Etiqueta de decisión asociada a cada estadio.
CLASS_OF_STAGE: dict[str, str] = {
    "W": "wake",
    "N1": "n1",
    "N2": "n2",
    "R": "rem",
}

#: Clases de decisión del piloto (incluye ``abstain`` en el esquema).
SLEEP_CLASSES: tuple[str, ...] = ("wake", "n1", "n2", "rem")

#: Escala del acelerómetro de la E4 (cuentas -> g).
ACC_SCALE = 1.0 / 64.0

#: Frecuencias nativas tras decimar las señales retenidas.
FS_EDA = 4.0
FS_TEMP = 4.0
FS_ACC = 32.0

#: Factores de decimación desde la rejilla de 64 Hz.
DECIMATE_EDA = int(FS / FS_EDA)  # 16
DECIMATE_TEMP = int(FS / FS_TEMP)  # 16
DECIMATE_ACC = int(FS / FS_ACC)  # 2

#: Pureza mínima de cada época (proporción de muestras con la etiqueta).
PURITY_MIN = 0.999

# ---------------------------------------------------------------------------
# Matriz de degradación (preregistro §8)
# ---------------------------------------------------------------------------

#: Condición limpia.
C0 = "C0_clean"
#: EDA ausente (100 % NaN).
C1_ABSENT_EDA = "C1_absent_eda"
#: EDA degenerada a ceros: valor presente pero fisiológicamente imposible.
C2_DEGENERATE_EDA = "C2_degenerate_eda"
#: Ausencia de la evidencia autonómica (IBI y HR).
C3_ABSENT_HRV = "C3_absent_hrv"
#: Ausencia del núcleo (EDA + HRV + acelerometría).
C4_ABSENT_CORE = "C4_absent_core"
#: Control negativo: todas las features a ``not available``.
NC_ABSENT = "NC_absent"

#: Condiciones del piloto, en orden de ejecución.
PILOT_CONDITIONS: tuple[str, ...] = (
    C0,
    C1_ABSENT_EDA,
    C2_DEGENERATE_EDA,
    C3_ABSENT_HRV,
    C4_ABSENT_CORE,
    NC_ABSENT,
)

#: Condiciones que actúan sobre los arrays antes de extraer features.
ARRAY_CONDITIONS: tuple[str, ...] = (
    C0,
    C1_ABSENT_EDA,
    C2_DEGENERATE_EDA,
    C3_ABSENT_HRV,
    C4_ABSENT_CORE,
)

#: Descripción legible de cada condición (se guarda en la procedencia).
CONDITION_NOTES: dict[str, str] = {
    C0: "sin manipulación",
    C1_ABSENT_EDA: "EDA sustituida por NaN (ausencia honesta)",
    C2_DEGENERATE_EDA: "EDA a ceros (valor imposible: el mínimo medido es 0,10 µS)",
    C3_ABSENT_HRV: "IBI y HR sustituidos por NaN (sin evidencia autonómica cardiaca)",
    C4_ABSENT_CORE: "EDA + IBI/HR + ACC sustituidos por NaN (ausencia del núcleo)",
    NC_ABSENT: "todas las features a not available (control negativo)",
}


# ---------------------------------------------------------------------------
# Sujetos y lectura
# ---------------------------------------------------------------------------


def data_dir(root: Path | str | None = None) -> Path:
    """Directorio de la rama *wearable* (``data_64Hz``)."""
    base = Path(root) if root is not None else DREAMT_ROOT
    candidate = base / WEARABLE_DIRNAME
    return candidate if candidate.is_dir() else base


def subject_path(subject: str, root: Path | str | None = None) -> Path:
    """Ruta del CSV de 64 Hz de un sujeto."""
    return data_dir(root) / SUBJECT_TEMPLATE.format(subject=subject)


def list_subjects(root: Path | str | None = None) -> list[str]:
    """Sujetos disponibles (``S002``…), ordenados."""
    directory = data_dir(root)
    if not directory.is_dir():
        return []
    return sorted(
        path.name.split("_")[0]
        for path in directory.glob("S*_whole_df.csv")
    )


def load_subject(
    subject: str,
    root: Path | str | None = None,
    usecols: Sequence[str] = USECOLS,
):
    """Carga las columnas necesarias del sujeto con ``usecols``.

    Se usa ``usecols`` porque el CSV completo (14 columnas) tiene tipos mixtos
    en columnas que no se necesitan y pandas las infiere como ``object``,
    provocando ``ValueError`` al convertir.
    """
    import pandas as pd  # import local: el módulo debe poder importarse sin pandas

    path = subject_path(subject, root)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el fichero de DREAMT: {path}")
    return pd.read_csv(path, usecols=list(usecols))


def stage_array(frame) -> np.ndarray:
    """Etiquetas de estadio como array de cadenas (sin tocar el resto)."""
    return frame["Sleep_Stage"].astype(str).to_numpy()


# ---------------------------------------------------------------------------
# Rejilla de épocas y ventanas
# ---------------------------------------------------------------------------


def anchor_index(stage_values: Sequence[str]) -> int:
    """Índice de la primera muestra con etiqueta distinta de ``P``.

    Las épocas de 30 s están puntuadas desde el inicio de la noche, no desde la
    primera fila del CSV. Anclar en la fila 0 mezcla dos épocas reales casi en
    cada bloque.
    """
    values = np.asarray(stage_values)
    non_prep = np.flatnonzero(values != "P")
    if non_prep.size == 0:
        return 0
    return int(non_prep[0])


@dataclass(frozen=True)
class EpochGrid:
    """Rejilla de épocas puntuadas de una noche."""

    anchor: int
    labels: tuple[str, ...]
    purity: tuple[float, ...]

    @property
    def n_epochs(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        """Número de épocas por estadio."""
        out: dict[str, int] = {}
        for label in self.labels:
            out[label] = out.get(label, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def epoch_grid(
    stage_values: Sequence[str],
    anchor: int | None = None,
) -> EpochGrid:
    """Corta las épocas de 30 s desde el ancla y calcula etiqueta y pureza.

    La etiqueta de la época es el estadio de su muestra central y la pureza es
    la proporción de muestras que lo comparten. Se descarta la última época si
    está incompleta.
    """
    values = np.asarray(stage_values)
    start = anchor_index(values) if anchor is None else int(anchor)
    tail = values[start:]
    n_epochs = tail.size // EPOCH_ROWS
    if n_epochs <= 0:
        return EpochGrid(start, (), ())
    blocks = tail[: n_epochs * EPOCH_ROWS].reshape(n_epochs, EPOCH_ROWS)
    labels = blocks[:, EPOCH_ROWS // 2]
    purity = (blocks == labels[:, None]).mean(axis=1)
    return EpochGrid(start, tuple(labels.tolist()), tuple(purity.tolist()))


@dataclass(frozen=True)
class SleepWindow:
    """Ventana de 60 s pura (dos épocas consecutivas del mismo estadio)."""

    subject: str
    stage: str
    epoch_index: int
    start_row: int
    night_frac: float
    n_epochs: int = WINDOW_EPOCHS

    @property
    def window_id(self) -> str:
        """Identificador interno (nunca va al prompt; se hashea antes)."""
        return f"{self.subject}:e{self.epoch_index:04d}"

    @property
    def klass(self) -> str | None:
        """Clase de decisión asociada al estadio."""
        return CLASS_OF_STAGE.get(self.stage)

    @property
    def end_row(self) -> int:
        return self.start_row + self.n_epochs * EPOCH_ROWS


def pure_windows(
    subject: str,
    grid: EpochGrid,
    stages: Iterable[str] = TASK_STAGES,
    purity_min: float = PURITY_MIN,
) -> list[SleepWindow]:
    """Ventanas de 60 s con pureza 1,0, sin solape, en estadios de la tarea."""
    allowed = set(stages)
    out: list[SleepWindow] = []
    n = grid.n_epochs
    index = 0
    while index + WINDOW_EPOCHS <= n:
        block = grid.labels[index : index + WINDOW_EPOCHS]
        block_purity = grid.purity[index : index + WINDOW_EPOCHS]
        ok = (
            block[0] == block[1]
            and block[0] in allowed
            and min(block_purity) >= purity_min
        )
        if ok:
            out.append(
                SleepWindow(
                    subject=subject,
                    stage=block[0],
                    epoch_index=index,
                    start_row=grid.anchor + index * EPOCH_ROWS,
                    night_frac=(index / n) if n else 0.0,
                )
            )
            index += WINDOW_EPOCHS
        else:
            index += 1
    return out


def stage_counts(windows: Sequence[SleepWindow]) -> dict[str, int]:
    """Número de ventanas puras por estadio (solo estadios de la tarea)."""
    counts: dict[str, int] = {stage: 0 for stage in TASK_STAGES}
    for window in windows:
        counts[window.stage] = counts.get(window.stage, 0) + 1
    return counts


def _tertile(night_frac: float) -> int:
    """Tercio de la noche al que pertenece una posición relativa."""
    if night_frac < 1.0 / 3.0:
        return 0
    if night_frac < 2.0 / 3.0:
        return 1
    return 2


def _even_positions(n_items: int, k: int) -> list[int]:
    """Posiciones repartidas uniformemente en ``[0, n_items)``."""
    if n_items <= 0 or k <= 0:
        return []
    if k >= n_items:
        return list(range(n_items))
    if k == 1:
        return [n_items // 2]
    return sorted(
        {int(round(x)) for x in np.linspace(0, n_items - 1, k)}
    )[:k]


def sample_windows(
    windows: Sequence[SleepWindow],
    per_class: int = 5,
    seed: int = 42,
) -> list[SleepWindow]:
    """Muestreo sistemático por clase, repartido por tercios de la noche.

    Para cada clase se reparten las ``per_class`` ventanas entre los tercios de
    la noche (cuotas 2/2/1 con 5) y dentro de cada tercio se toman posiciones
    uniformes. Es determinista: ``seed`` solo rota el tercio por el que empieza
    cada clase, de modo que dos clases no eligen siempre el mismo.

    El reparto por tercios evita que una clase quede concentrada en un tramo y
    que la firma se confunda con el orden de la noche.
    """
    by_class: dict[str, list[SleepWindow]] = {}
    for window in windows:
        if window.klass is None:
            continue
        by_class.setdefault(window.klass, []).append(window)

    quotas = [per_class // 3] * 3
    for slot in range(per_class - sum(quotas)):
        quotas[slot % 3] += 1

    selected: list[SleepWindow] = []
    for offset, klass in enumerate(SLEEP_CLASSES):
        candidates = sorted(by_class.get(klass, []), key=lambda w: w.epoch_index)
        if not candidates:
            continue
        tertiles: dict[int, list[SleepWindow]] = {0: [], 1: [], 2: []}
        for window in candidates:
            tertiles[_tertile(window.night_frac)].append(window)

        # Rotación determinista por clase (seed) para no sesgar el orden.
        rotation = (seed + offset) % 3
        order = [(rotation + i) % 3 for i in range(3)]
        picked: list[SleepWindow] = []
        for slot, tertile in enumerate(order):
            quota = quotas[slot]
            if quota <= 0:
                continue
            bucket = tertiles[tertile]
            if not bucket:
                continue
            for position in _even_positions(len(bucket), quota):
                picked.append(bucket[position])

        # Si algún tercio no tenía ventanas, se completa con el resto.
        if len(picked) < per_class:
            taken = {id(w) for w in picked}
            for window in candidates:
                if len(picked) >= per_class:
                    break
                if id(window) not in taken:
                    picked.append(window)
        selected.extend(sorted(picked[:per_class], key=lambda w: w.epoch_index))

    return sorted(selected, key=lambda w: (SLEEP_CLASSES.index(w.klass), w.epoch_index))


# ---------------------------------------------------------------------------
# Arrays de una ventana
# ---------------------------------------------------------------------------


def window_arrays(
    frame,
    window: SleepWindow,
    acc_scale: float = ACC_SCALE,
) -> dict[str, np.ndarray]:
    """Extrae los arrays nativos de una ventana (ya decimados y en unidades).

    Returns:
        ``{"eda": (240,), "temp": (240,), "acc": (1920, 3), "bvp": (3840,),
        "ibi": (3840,), "hr": (3840,)}`` con EDA/TEMP a 4 Hz, ACC a 32 Hz y
        BVP/IBI/HR a 64 Hz. Solo ``hr`` genera el bloque cardiaco (ver
        ``cardiac_features``); ``bvp`` e ``ibi`` se conservan para aplicar las
        condiciones de degradación y para la procedencia.
    """
    start, end = window.start_row, window.end_row
    eda = frame["EDA"].to_numpy(dtype=float)[start:end]
    temp = frame["TEMP"].to_numpy(dtype=float)[start:end]
    acc = np.column_stack(
        [
            frame["ACC_X"].to_numpy(dtype=float)[start:end],
            frame["ACC_Y"].to_numpy(dtype=float)[start:end],
            frame["ACC_Z"].to_numpy(dtype=float)[start:end],
        ]
    )
    bvp = frame["BVP"].to_numpy(dtype=float)[start:end]
    ibi = frame["IBI"].to_numpy(dtype=float)[start:end]
    hr = frame["HR"].to_numpy(dtype=float)[start:end]
    return {
        "eda": eda[::DECIMATE_EDA],
        "temp": temp[::DECIMATE_TEMP],
        "acc": acc[::DECIMATE_ACC] * float(acc_scale),
        "bvp": bvp,
        "ibi": ibi,
        "hr": hr,
    }


# ---------------------------------------------------------------------------
# Bloque cardiaco: columna ``HR`` del dispositivo
# ---------------------------------------------------------------------------
#
# Se midió sobre S002 antes de fijar la fuente:
#
# * ``IBI`` está cuantizada a 1/64 s (15,6 ms) y con *dither* de una muestra.
#   Tras deduplicar los cambios reales, la mitad de las ventanas no llega a 20
#   intervalos y donde los hay el SDNN sale de 190 a 280 ms (imposible para un
#   adulto sano). Usarla daría un HRV artificial.
# * ``BVP`` (64 Hz real) con ``nk.ppg_peaks``: la HR coincide con la del
#   dispositivo en algunas ventanas (76,4 vs 75,2 bpm), pero la localización de
#   picos fluctúa ±100 ms y aparecen RR de 328 ms y de 5.641 ms. La HRV
#   derivada no es utilizable.
#
# Por eso el bloque ``ecg`` usa la columna ``HR`` que calcula el propio E4
# (1 Hz, sin huecos en la noche puntuada). Consecuencia declarada: esta tarea
# **no tiene HRV** (ni SDNN ni RMSSD), y la tabla de firma del sueño se fija con
# tres features. Es la desviación D7 del preregistro, que preveía
# ``ibi_derived``.

#: Rango fisiológico aceptado para la serie ``HR`` del dispositivo (bpm).
HR_MIN_BPM: float = 25.0
HR_MAX_BPM: float = 220.0


def cardiac_features(
    hr: np.ndarray,
    duration_s: float = WINDOW_S,
    min_samples: int = 10,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Bloque ``ecg_*`` a partir de la serie ``HR`` del dispositivo.

    ``ecg_hr_bpm`` es la **mediana** de la serie de 1 Hz de la ventana (robusta
    a los picos de artefacto de pérdida de contacto) y ``ecg_rr_mean_ms`` su
    equivalente ``60000 / HR``, para que los dos valores del prompt sean
    coherentes entre sí. Las claves de HRV se devuelven a ``None``: la rama
    *wearable* de DREAMT no permite calcularlas (ver comentario del bloque).

    Returns:
        ``(features, meta)``; ``meta`` anota el método, el número de muestras
        usadas y el motivo cuando no se publica ninguna feature.
    """
    keys = (
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
        "ecg_sdnn_ms",
        "ecg_rmssd_ms",
        "ecg_n_beats",
    )
    meta: dict[str, Any] = {"method": "device_hr_1hz", "n_samples": 0}
    series = np.asarray(hr, dtype=float).reshape(-1)
    finite = series[np.isfinite(series)]
    finite = finite[(finite >= HR_MIN_BPM) & (finite <= HR_MAX_BPM)]
    meta["n_samples"] = int(finite.size)
    if finite.size < min_samples:
        meta["reason"] = "hr_missing"
        return {key: None for key in keys}, meta

    hr_bpm = float(np.median(finite))
    features: dict[str, float | None] = {
        "ecg_hr_bpm": hr_bpm,
        "ecg_rr_mean_ms": float(60000.0 / hr_bpm) if hr_bpm > 0 else None,
        "ecg_sdnn_ms": None,
        "ecg_rmssd_ms": None,
        "ecg_n_beats": None,
    }
    return features, meta


# ---------------------------------------------------------------------------
# Degradación
# ---------------------------------------------------------------------------


def apply_array_condition(
    arrays: Mapping[str, np.ndarray],
    condition: str,
) -> dict[str, np.ndarray]:
    """Aplica la degradación sobre los arrays **antes** de extraer features.

    Se manipula la señal y no las features para que el extractor decida qué es
    representable: así «ausencia» produce ``None`` por la vía normal y
    «degeneración» produce ceros reales con su propia coherencia interna.
    """
    if condition not in ARRAY_CONDITIONS:
        raise ValueError(
            f"condición desconocida: {condition!r}. "
            f"Disponibles: {list(ARRAY_CONDITIONS) + [NC_ABSENT]}"
        )
    out = {
        name: np.array(values, dtype=float, copy=True)
        for name, values in arrays.items()
    }
    if condition in (C1_ABSENT_EDA, C4_ABSENT_CORE):
        out["eda"] = np.full_like(out["eda"], np.nan)
    if condition == C2_DEGENERATE_EDA:
        out["eda"] = np.zeros_like(out["eda"])
    if condition in (C3_ABSENT_HRV, C4_ABSENT_CORE):
        # Quita el bloque cardiaco completo. El identificador conserva el nombre
        # del preregistro (``C3_absent_hrv``), pero en la rama *wearable* lo que
        # se retira es HR + RR: HRV no existe (desviación D7).
        for name in ("hr", "bvp", "ibi"):
            out[name] = np.full_like(out[name], np.nan)
    if condition == C4_ABSENT_CORE:
        out["acc"] = np.full_like(out["acc"], np.nan)
    return out


def apply_feature_condition(
    features: Mapping[str, Mapping[str, float | None]],
    condition: str,
) -> dict[str, dict[str, float | None]]:
    """Aplica ``NC_absent`` sobre las features ya extraídas.

    Las claves se conservan con valor ``None``: el prompt debe decir «not
    available» para que el modelo pueda abstenerse. Borrar la clave ocultaría
    que el canal existe y no se ha medido.
    """
    if condition != NC_ABSENT:
        return {modality: dict(values) for modality, values in features.items()}
    return {
        modality: {name: None for name in values}
        for modality, values in features.items()
    }


# ---------------------------------------------------------------------------
# Extracción de features
# ---------------------------------------------------------------------------


def _canonical_keys() -> dict[str, tuple[str, ...]]:
    """Features canónicas por modalidad (el vocabulario que ve el LLM)."""
    from sources.ingestion.represent import LLM_ALLOWED_FEATURES

    prefix = {"ecg": "ecg_", "eda": "eda_", "temp": "temp_", "acc": "acc_"}
    return {
        modality: tuple(
            sorted(
                name
                for name in LLM_ALLOWED_FEATURES
                if name.startswith(prefix[modality])
            )
        )
        for modality in prefix
    }


#: Modalidades de la evidencia en la rama *wearable* (sin ``resp`` ni ``emg``).
SLEEP_MODALITIES: tuple[str, ...] = ("ecg", "eda", "temp", "acc")


def extract_features(
    arrays: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
    duration_s: float = WINDOW_S,
) -> tuple[dict[str, dict[str, float | None]], dict[str, Any]]:
    """Extrae las cuatro modalidades disponibles y anota el método usado.

    Un fallo de una modalidad **no** invalida la ventana: se devuelven sus
    claves con ``None`` y un motivo en el segundo valor de retorno, que se
    guarda en la procedencia.

    Returns:
        ``(features, notes)``. ``notes`` incluye, por modalidad, el motivo de
        los valores ausentes y, en ``ecg``, los metadatos del detector de picos
        (número de latidos y fracción de RR descartados).
    """
    from sources.ingestion.features import acc as acc_features
    from sources.ingestion.features import eda as eda_features
    from sources.ingestion.features import temp as temp_features

    keys = _canonical_keys()
    features: dict[str, dict[str, float | None]] = {
        modality: {name: None for name in names} for modality, names in keys.items()
    }
    notes: dict[str, Any] = {}

    # --- Cardiaco: serie HR del dispositivo (sin HRV; ver bloque anterior) ---
    ecg_values, ecg_meta = cardiac_features(
        arrays["hr"], duration_s=duration_s
    )
    for name, value in ecg_values.items():
        if name in features["ecg"]:
            features["ecg"][name] = value
    notes["ecg"] = ecg_meta

    # --- EDA y TEMP a 4 Hz, ACC a 32 Hz ---
    calls = (
        ("eda", eda_features.extract, arrays["eda"], FS_EDA),
        ("temp", temp_features.extract, arrays["temp"], FS_TEMP),
        ("acc", acc_features.extract, arrays["acc"], FS_ACC),
    )
    for modality, extractor, values, fs in calls:
        try:
            extracted, _quality = extractor(
                np.asarray(values, dtype=float),
                fs=float(fs),
                cfg=cfg,
                coverage=1.0,
                duration_s=duration_s,
            )
        except Exception as exc:  # noqa: BLE001 - una modalidad no rompe la ventana
            notes[modality] = {"reason": f"{type(exc).__name__}: {exc}"}
            continue
        for name, meta in extracted.items():
            if name not in features[modality]:
                continue
            # Los extractores devuelven metadatos (``{"value": ..., "unit": ...}``);
            # aquí solo interesa el valor escalar.
            features[modality][name] = (
                meta.get("value") if isinstance(meta, Mapping) else meta
            )

    return features, notes


def flatten_features(
    features: Mapping[str, Mapping[str, float | None]],
) -> dict[str, float | None]:
    """Aplana ``{modalidad: {feature: valor}}`` a ``{feature: valor}``."""
    return {
        name: value
        for modality_values in features.values()
        for name, value in modality_values.items()
    }


def full_night_features(
    subject: str,
    frame,
    windows: Sequence[SleepWindow],
    cfg: Mapping[str, Any],
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Extrae las features de **todas** las ventanas puras de una noche.

    Se usan para dos cosas y ninguna de ellas es el LLM: construir la
    referencia intra-sujeto (mediana/IQR) y ejecutar el pre-chequeo P0 de la
    tabla de firma sobre la noche completa.
    """
    rows: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        arrays = window_arrays(frame, window)
        features, notes = extract_features(arrays, cfg)
        rows.append(
            {
                "subject": subject,
                "window_id": window.window_id,
                "stage": window.stage,
                "klass": window.klass,
                "epoch_index": window.epoch_index,
                "night_frac": round(window.night_frac, 6),
                "duration_s": WINDOW_S,
                "features": features,
                "notes": notes,
            }
        )
        if progress and index % 100 == 0:
            print(f"[DREAMT] {subject}: {index}/{len(windows)} ventanas puras")
    return rows


# ---------------------------------------------------------------------------
# Representación para el LLM
# ---------------------------------------------------------------------------

#: Etiquetas legibles de las features de movimiento (no existían en WESAD,
#: porque allí la modalidad estaba excluida de la vista del LLM).
MOVEMENT_LABELS: dict[str, str] = {
    "acc_mean_g": "Mean movement",
    "acc_std_g": "Movement variability",
    "acc_p95_g": "Movement 95th percentile",
    "acc_energy_g2": "Movement energy",
    "acc_motion_fraction": "Movement fraction",
}

#: Secciones de la representación textual (mismo estilo que Fase 1).
#: Mismas features que la vista numérica (``prompt_builder.SLEEP_FEATURES``): si
#: el texto mostrara otras, el contraste H3 mediría contenidos y no formatos.
#: La rama *wearable* no permite HRV, así que ``ecg_n_beats``, ``ecg_sdnn_ms`` y
#: ``ecg_rmssd_ms`` no aparecen (desviación D7).
SLEEP_TEXT_SECTIONS: dict[str, tuple[str, ...]] = {
    "CARDIOVASCULAR (ECG)": (
        "ecg_hr_bpm",
        "ecg_rr_mean_ms",
    ),
    "ELECTRODERMAL ACTIVITY (EDA)": (
        "eda_mean_us",
        "eda_std_us",
        "eda_slope_us_per_s",
        "eda_tonic_mean_us",
        "eda_phasic_std_us",
        "eda_scr_count",
        "eda_scr_rate_per_min",
    ),
    "TEMPERATURE": (
        "temp_mean_degc",
        "temp_std_degc",
        "temp_slope_degc_per_s",
        "temp_range_degc",
    ),
    "MOVEMENT (ACCELEROMETER)": (
        "acc_mean_g",
        "acc_std_g",
        "acc_p95_g",
        "acc_energy_g2",
        "acc_motion_fraction",
    ),
}

SLEEP_TEXT_LABELS: dict[str, str] = {
    "ecg_hr_bpm": "Heart rate",
    "ecg_rr_mean_ms": "Mean RR interval",
    "eda_mean_us": "Mean conductance",
    "eda_std_us": "Conductance variability",
    "eda_slope_us_per_s": "Conductance trend",
    "eda_tonic_mean_us": "Tonic level",
    "eda_phasic_std_us": "Phasic variability",
    "eda_scr_count": "SCR count",
    "eda_scr_rate_per_min": "SCR rate",
    "eda_scr_amplitude_mean_us": "Mean SCR amplitude",
    "eda_scr_amplitude_max_us": "Maximum SCR amplitude",
    "temp_mean_degc": "Mean temperature",
    "temp_std_degc": "Temperature variability",
    "temp_slope_degc_per_s": "Temperature trend",
    "temp_range_degc": "Temperature range",
    **MOVEMENT_LABELS,
}

#: Alias de métrica del *brazo textual*: el modelo cita la etiqueta que leyó
#: («Conductance variability»), y el verificador necesita el nombre canónico.
#: Se construye desde ``SLEEP_TEXT_LABELS`` para que no pueda quedar incompleto
#: (una etiqueta sin alias se contaría como ``unsupported`` por error).
SLEEP_METRIC_ALIASES: dict[str, str] = {
    label.lower(): name for name, label in SLEEP_TEXT_LABELS.items()
}

#: Variantes que el modelo escribe sin ser la etiqueta exacta del texto.
SLEEP_METRIC_ALIASES.update(
    {
        "acc std": "acc_std_g",
        "acc mean": "acc_mean_g",
        "acc p95": "acc_p95_g",
        "acc energy": "acc_energy_g2",
        "acceleration variability": "acc_std_g",
        "mean acceleration": "acc_mean_g",
        "movement": "acc_std_g",
        "movement intensity": "acc_std_g",
    }
)


def build_sleep_payload(
    window_id: str,
    zvalues: Mapping[str, float | None],
    duration_s: float = WINDOW_S,
    modalities: Sequence[str] = SLEEP_MODALITIES,
) -> dict[str, Any]:
    """Payload ``numeric_llm`` de una ventana de sueño (z intra-sujeto).

    ``build_payload`` de Fase 3 no sirve porque solo admite las modalidades de
    pecho de WESAD y **excluye el acelerómetro**, que aquí es evidencia
    legítima (en WESAD era un confusor: ACC y EMG discriminaban estrés con
    AUC 0,928 por el movimiento de hablar en público).
    """
    from sources.ingestion.represent import _opaque_window_id

    allowed = set(modalities)
    blocks: dict[str, dict[str, Any]] = {}
    for name, value in zvalues.items():
        modality = name.split("_")[0]
        if modality not in allowed:
            continue
        blocks.setdefault(modality, {})[name] = {
            # Tres decimales: el recorte es ±3, así que la precisión sobrante
            # solo alarga el prompt (el verificador de claims tolera ±0,2).
            "value": None if value is None else round(float(value), 3),
            "unit": "SD",
            "kind": "derived",
            "method": "within_subject_zscore_from_night_reference",
        }
    return {
        "modalities": blocks,
        "feature_version": "dreamt-sleep-pilot",
        "window": {
            "window_id": _opaque_window_id(window_id),
            "duration_s": duration_s,
        },
    }


def render_sleep_text(zvalues: Mapping[str, float | None]) -> str:
    """Representación textual de los mismos valores z (brazo H3)."""
    lines = ["PHYSIOLOGICAL SIGNAL SUMMARY", "=" * 50, ""]
    for section, names in SLEEP_TEXT_SECTIONS.items():
        present = [name for name in names if name in zvalues]
        if not present:
            continue
        lines.append(section)
        for name in present:
            value = zvalues[name]
            label = SLEEP_TEXT_LABELS.get(name, name)
            if value is None:
                lines.append(f"  {label}: not available")
            else:
                lines.append(f"  {label}: {value:+.2f} SD")
        lines.append("")
    return "\n".join(lines).rstrip()


def night_reference(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    clip_values: bool = True,
) -> tuple[dict[str, float], dict[str, float]]:
    """Mediana e IQR por feature sobre **todas** las ventanas puras de la noche.

    La referencia es intra-sujeto: absorbe el nivel individual y hace
    comparables las ventanas entre sí. Se calcula con la noche completa y no
    con las ventanas muestreadas para que la muestra no defina su propia
    referencia.
    """
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = np.asarray(
            [row["flat"].get(name) for row in rows if row["flat"].get(name) is not None],
            dtype=float,
        )
        if values.size == 0:
            centers[name], scales[name] = 0.0, 1.0
            continue
        if clip_values:
            low, high = np.percentile(values, [1, 99])
            values = np.clip(values, low, high)
        centers[name] = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        scale = float(q3 - q1) or float(np.std(values)) or 1.0
        scales[name] = scale
    return centers, scales


def z_from_reference(
    flat: Mapping[str, float | None],
    centers: Mapping[str, float],
    scales: Mapping[str, float],
    clip: float = 3.0,
) -> dict[str, float | None]:
    """Convierte valores absolutos a z robustos y recorta a ``±clip``."""
    out: dict[str, float | None] = {}
    for name, value in flat.items():
        if value is None:
            out[name] = None
        else:
            out[name] = float(
                np.clip(
                    (value - centers.get(name, 0.0)) / scales.get(name, 1.0),
                    -clip,
                    clip,
                )
            )
    return out


def stage_of_class(klass: str) -> str:
    """Estadio asociado a una clase de decisión (``rem`` -> ``R``)."""
    for stage, name in CLASS_OF_STAGE.items():
        if name == klass:
            return stage
    raise ValueError(f"clase desconocida: {klass!r}")


def is_finite(value: float | None) -> bool:
    """``True`` si el valor existe y es finito."""
    return value is not None and math.isfinite(float(value))
