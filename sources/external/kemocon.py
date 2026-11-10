"""Adaptador de K-EmoCon para validación externa (P3).

K-EmoCon (Park et al., 2020) no comparte la instrumentación de WESAD: aporta
EDA de muñeca, BVP, temperatura, IBI, acelerómetro y frecuencia cardiaca
(Polar / derivada), pero **no** tiene ECG de pecho, respiración ni conteo de
SCR. Este módulo:

1. Une la tabla de features de PSRI (`kemocon_feature_table.csv`, ventanas de
   5 s) con la asignación de periodos (`kemocon_period_assignments.csv`).
2. Agrega ventanas de 5 s a ventanas de 60 s dentro de cada periodo.
3. Traduce las columnas de K-EmoCon a los nombres canónicos del proyecto,
   dejando como ``None`` lo que no existe (respiración, SCR, latidos).
4. Aplica la estandarización robusta intra-sujeto de E12.

La tarea externa es ``pre`` (no estrés) vs ``debate`` (estrés social). Ambas
etiquetas son de protocolo, no autoinformes: ``pre`` siempre ocurre antes que
``debate`` (confusor temporal documentado).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Duración de la subventana original de PSRI (s).
SUBWINDOW_S = 5.0
#: Duración objetivo de la ventana agregada (s).
WINDOW_S = 60.0
#: Subventanas necesarias para formar una ventana agregada.
SUBWINDOWS_PER_WINDOW = int(WINDOW_S / SUBWINDOW_S)

#: Periodos del protocolo. K-EmoCon **no** tiene etiqueta de estrés: ``pre`` y
#: ``debate`` son periodos, no niveles de estrés. La validación externa usa las
#: etiquetas afectivas (arousal/valencia) del periodo ``debate``.
PERIODS: tuple[str, ...] = ("pre", "debate")

#: Etiquetas afectivas agregables a la ventana (columnas de la tabla de PSRI).
AFFECT_SOURCES: dict[str, str] = {
    "self_arousal": "self_arousal",
    "self_valence": "self_valence",
    "partner_arousal": "partner_arousal",
    "partner_valence": "partner_valence",
    "external_arousal": "external_arousal_mean",
    "external_valence": "external_valence_mean",
}

#: Features canónicas que K-EmoCon puede aportar (subconjunto de WESAD).
KEMOCON_FEATURES: tuple[str, ...] = (
    "ecg_hr_bpm",
    "ecg_rr_mean_ms",
    "ecg_sdnn_ms",
    "eda_mean_us",
    "eda_std_us",
    "temp_mean_degc",
    "temp_std_degc",
)

#: Features adicionales solo para el modelo de referencia (no van al LLM).
KEMOCON_REFERENCE_EXTRA: tuple[str, ...] = ("acc_std_g",)

#: Canales que K-EmoCon NO aporta y que el prompt de WESAD sí menciona.
KEMOCON_MISSING_CHANNELS: tuple[str, ...] = (
    "chest ECG",
    "respiration",
    "SCR counts and amplitudes",
    "skin conductance responses",
)


@dataclass
class KemoconWindow:
    """Ventana de 60 s agregada de K-EmoCon."""

    subject: str
    period: str
    window_index: int
    n_subwindows: int
    values: dict[str, float | None] = field(default_factory=dict)
    zscores: dict[str, float | None] = field(default_factory=dict)
    ratings: dict[str, float | None] = field(default_factory=dict)
    affect_label: str | None = None
    affect_score: float | None = None

    @property
    def window_id(self) -> str:
        return f"KEMOCON:{self.subject}:{self.period}:w{self.window_index:03d}"

    @property
    def duration_s(self) -> float:
        return self.n_subwindows * SUBWINDOW_S


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def load_period_assignments(path: Path) -> dict[tuple[str, str], tuple[str, int]]:
    """Carga ``(subject_id, win) -> (period, win_in_period)``."""
    assignments: dict[tuple[str, str], tuple[str, int]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assignments[(row["subject_id"], row["win"])] = (
                row["period"],
                int(row["win_in_period"]),
            )
    return assignments


def load_feature_rows(path: Path) -> dict[tuple[str, str], dict[str, float | None]]:
    """Carga la tabla de features indexada por ``(subject_id, win)``."""
    rows: dict[tuple[str, str], dict[str, float | None]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["subject_id"], row["win"])
            rows[key] = {
                name: _to_float(value)
                for name, value in row.items()
                if name not in ("subject_id", "win")
            }
    return rows


def _mean(values: Iterable[float | None]) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return None
    return float(array.mean())


def _rms(values: Iterable[float | None]) -> float | None:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return None
    return float(np.sqrt(np.mean(array**2)))


def _first_available(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def aggregate_windows(
    rows: Mapping[tuple[str, str], Mapping[str, float | None]],
    assignments: Mapping[tuple[str, str], tuple[str, int]],
    min_coverage: float = 0.5,
    subjects: Sequence[str] | None = None,
) -> list[KemoconWindow]:
    """Agrega ventanas de 5 s en ventanas de 60 s por sujeto y periodo.

    Args:
        rows: features por ``(subject_id, win)``.
        assignments: periodo y orden dentro del periodo por ``(subject_id, win)``.
        min_coverage: fracción mínima de subventanas con datos de EDA para
            aceptar la ventana agregada.
        subjects: subconjunto de sujetos (``None`` = todos).

    Returns:
        lista de ``KemoconWindow`` ordenada por sujeto y periodo.
    """
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for (subject, win), (period, win_in_period) in assignments.items():
        if subjects is not None and subject not in set(subjects):
            continue
        if (subject, win) not in rows:
            continue
        grouped.setdefault((subject, period), []).append((win_in_period, win))

    windows: list[KemoconWindow] = []
    period_order = {"pre": 0, "debate": 1}
    for (subject, period), entries in sorted(
        grouped.items(), key=lambda item: (item[0][0], period_order.get(item[0][1], 9))
    ):
        entries.sort()
        ordered = [win for _, win in entries]
        for index in range(0, len(ordered) // SUBWINDOWS_PER_WINDOW):
            chunk = ordered[index * SUBWINDOWS_PER_WINDOW : (index + 1) * SUBWINDOWS_PER_WINDOW]
            subrows = [rows[(subject, win)] for win in chunk]

            hr_values = [
                _first_available(row.get("hr_mean"), row.get("polar_hr_mean"))
                for row in subrows
            ]
            eda_values = [row.get("eda_mean") for row in subrows]
            coverage = sum(1 for value in eda_values if value is not None) / len(chunk)
            if coverage < min_coverage:
                continue

            values = {
                "ecg_hr_bpm": _mean(hr_values),
                "ecg_rr_mean_ms": _mean([row.get("ibi_mean") for row in subrows]),
                "ecg_sdnn_ms": _rms([row.get("ibi_std") for row in subrows]),
                "eda_mean_us": _mean(eda_values),
                "eda_std_us": _rms([row.get("eda_std") for row in subrows]),
                "temp_mean_degc": _mean([row.get("temp_mean") for row in subrows]),
                "temp_std_degc": _rms([row.get("temp_std") for row in subrows]),
                "acc_std_g": _rms([row.get("acc_std") for row in subrows]),
            }
            ratings = {
                name: _mean([row.get(column) for row in subrows])
                for name, column in AFFECT_SOURCES.items()
            }
            windows.append(
                KemoconWindow(
                    subject=subject,
                    period=period,
                    window_index=index,
                    n_subwindows=len(chunk),
                    values=values,
                    ratings=ratings,
                )
            )
    return windows


def subject_centered(
    windows: Sequence[KemoconWindow],
    features: Sequence[str] = KEMOCON_FEATURES,
    method: str = "robust",
    clip: float = 3.0,
) -> None:
    """Estandariza cada feature dentro de cada sujeto (E12) in situ.

    ``robust``: ``(x - mediana) / IQR`` recortado a ``[-clip, clip]``.
    ``zscore``: ``(x - media) / DE``.
    """
    by_subject: dict[str, list[KemoconWindow]] = {}
    for window in windows:
        by_subject.setdefault(window.subject, []).append(window)

    for subject, subject_windows in by_subject.items():
        for name in features:
            values = np.asarray(
                [
                    window.values[name]
                    for window in subject_windows
                    if window.values.get(name) is not None
                ],
                dtype=float,
            )
            if values.size == 0:
                center, scale = 0.0, 1.0
            elif method == "robust":
                center = float(np.median(values))
                q1, q3 = np.percentile(values, [25, 75])
                scale = float(q3 - q1) or float(np.std(values)) or 1.0
            else:
                center = float(values.mean())
                scale = float(values.std()) or 1.0
            for window in subject_windows:
                value = window.values.get(name)
                if value is None:
                    window.zscores[name] = None
                else:
                    z = (value - center) / scale
                    if method == "robust":
                        z = float(np.clip(z, -clip, clip))
                    window.zscores[name] = float(z)


#: Modalidad canónica de cada feature (la vista LLM excluye ``acc``).
FEATURE_MODALITY: dict[str, str] = {
    "ecg_hr_bpm": "ecg",
    "ecg_rr_mean_ms": "ecg",
    "ecg_sdnn_ms": "ecg",
    "eda_mean_us": "eda",
    "eda_std_us": "eda",
    "temp_mean_degc": "temp",
    "temp_std_degc": "temp",
    "acc_std_g": "acc",
}


def build_payload(
    window: KemoconWindow,
    features: Sequence[str] = KEMOCON_FEATURES,
    include_motion: bool = False,
) -> dict[str, Any]:
    """Construye el payload ``numeric_llm`` de una ventana externa."""
    modalities: dict[str, dict[str, Any]] = {}
    for name in features:
        modality = FEATURE_MODALITY[name]
        if modality == "acc" and not include_motion:
            continue
        modalities.setdefault(modality, {})[name] = {
            "value": window.zscores.get(name),
            "unit": "SD",
            "kind": "derived",
            "method": "within_subject_zscore",
        }
    return {
        "modalities": modalities,
        "feature_version": "kemocon-e12",
        "window": {
            "window_id": opaque_window_id(window.window_id),
            "duration_s": window.duration_s,
        },
    }


def opaque_window_id(window_id: str) -> str:
    """Identificador opaco estable (SHA-256) para K-EmoCon."""
    import hashlib

    return "win_" + hashlib.sha256(window_id.encode()).hexdigest()[:8]


def assign_binary_affect_label(
    windows: Sequence[KemoconWindow],
    source: str = "self_arousal",
    split: str = "within_subject",
    positive_label: str = "high-arousal",
    negative_label: str = "low-arousal",
    period: str = "debate",
) -> int:
    """Asigna etiqueta binaria de arousal/valencia por mediana.

    Solo se etiquetan ventanas del periodo indicado (por defecto ``debate``,
    único periodo con autoinformes). Las ventanas sin valor se dejan con
    ``affect_label = None``.

    Args:
        windows: ventanas agregadas (se modifican in situ).
        source: clave de ``AFFECT_SOURCES`` (``self_arousal``, ``external_arousal``,
            ``self_valence``...).
        split: ``within_subject`` (mediana del propio participante) o ``global``
            (mediana de todos los participantes).
        positive_label: etiqueta para valores por encima de la mediana.
        negative_label: etiqueta para valores en o por debajo de la mediana.
        period: periodo al que se restringe el etiquetado.

    Returns:
        número de ventanas etiquetadas.
    """
    if source not in AFFECT_SOURCES:
        raise ValueError(
            f"Fuente afectiva no válida: {source}. Valores: {sorted(AFFECT_SOURCES)}"
        )
    if split not in ("within_subject", "global"):
        raise ValueError(f"split no válido: {split}. Valores: within_subject, global")

    candidates = [
        window
        for window in windows
        if window.period == period and window.ratings.get(source) is not None
    ]

    if split == "global":
        scores = [window.ratings[source] for window in candidates]
        median = float(np.median(scores)) if scores else 0.0
        for window in candidates:
            score = window.ratings[source]
            window.affect_score = float(score)
            window.affect_label = (
                positive_label if score > median else negative_label
            )
        return len(candidates)

    by_subject: dict[str, list[KemoconWindow]] = {}
    for window in candidates:
        by_subject.setdefault(window.subject, []).append(window)
    labelled = 0
    for subject_windows in by_subject.values():
        scores = [window.ratings[source] for window in subject_windows]
        median = float(np.median(scores))
        for window in subject_windows:
            score = window.ratings[source]
            window.affect_score = float(score)
            window.affect_label = (
                positive_label if score > median else negative_label
            )
            labelled += 1
    return labelled
