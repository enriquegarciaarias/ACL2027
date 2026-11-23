"""Firma fisiológica por clase para el eje de relevancia de H5.

Este módulo **materializa el preregistro** de
[`docs/h5_relevancia.md`](../../docs/h5_relevancia.md): la dirección esperada
de cada feature bajo ``stress``, fijada el 2026-11-21 **antes** de medir la
coherencia y sin posibilidad de ajuste posterior.

La tabla no se obtiene del rendimiento en estos datos: sale de mecanismo
fisiológico y literatura (Boucsein 2012; Dawson et al. 2007; Grossman 1983;
Kim et al. 2018; Task Force 1996). El documento declara qué tres direcciones
ya eran conocidas de sesiones anteriores.

Uso principal: ``sign_of(metric, FEATURE_SETS["primary"])`` da ``+1`` / ``-1``
para las features con firma y ``0`` para las que no la tienen (fuera de
numerator y denominador).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

#: Etiquetas de lado de decisión (propias del endpoint binario).
STRESS = "stress"
NON_STRESS = "non-stress"


@dataclass(frozen=True)
class FeatureSignature:
    """Firma preregistrada de una feature.

    ``sign`` es la dirección esperada bajo ``stress`` respecto de la
    referencia intra-sujeto (``+1`` sube, ``-1`` baja). ``level`` es la
    confianza de la firma; el conjunto primario es ``strong ∪ moderate``.
    """

    metric: str
    channel: str
    sign: int
    level: str
    primary: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "channel": self.channel,
            "sign": self.sign,
            "level": self.level,
            "primary": self.primary,
            "rationale": self.rationale,
        }


#: Las 12 features del conjunto primario (documento §3.2).
STRESS_SIGNATURE: dict[str, FeatureSignature] = {
    # --- EDA: activación simpática (Boucsein, 2012; Dawson et al., 2007) ---
    "eda_mean_us": FeatureSignature(
        "eda_mean_us", "eda", +1, "strong", True,
        "nivel tónico de conductancia sube con la activación simpática",
    ),
    "eda_tonic_mean_us": FeatureSignature(
        "eda_tonic_mean_us", "eda", +1, "strong", True,
        "mismo mecanismo que la media; componente tónico",
    ),
    "eda_phasic_std_us": FeatureSignature(
        "eda_phasic_std_us", "eda", +1, "strong", True,
        "más actividad fásica / respuestas no específicas",
    ),
    "eda_scr_count": FeatureSignature(
        "eda_scr_count", "eda", +1, "strong", True,
        "conteo en ventana fija de 60 s: no confundido con la duración",
    ),
    "eda_scr_rate_per_min": FeatureSignature(
        "eda_scr_rate_per_min", "eda", +1, "strong", True,
        "normalizado por duración; redundante con el conteo",
    ),
    "eda_std_us": FeatureSignature(
        "eda_std_us", "eda", +1, "moderate", True,
        "variabilidad de la señal; sube con la actividad fásica",
    ),
    "eda_scr_amplitude_mean_us": FeatureSignature(
        "eda_scr_amplitude_mean_us", "eda", +1, "moderate", True,
        "amplitud de respuesta tiende a crecer con la activación",
    ),
    "eda_scr_amplitude_max_us": FeatureSignature(
        "eda_scr_amplitude_max_us", "eda", +1, "moderate", True,
        "igual que la amplitud media, sensible a un único evento",
    ),
    # --- ECG: taquicardia y retirada vagal (Task Force, 1996; Kim et al., 2018) ---
    "ecg_hr_bpm": FeatureSignature(
        "ecg_hr_bpm", "ecg", +1, "strong", True,
        "taquicardia de estrés agudo",
    ),
    "ecg_sdnn_ms": FeatureSignature(
        "ecg_sdnn_ms", "ecg", -1, "strong", True,
        "retirada vagal; SDNN global baja",
    ),
    "ecg_rmssd_ms": FeatureSignature(
        "ecg_rmssd_ms", "ecg", -1, "strong", True,
        "componente vagal; RMSSD baja",
    ),
    # --- Respiración (Grossman, 1983) ---
    "resp_rate_bpm": FeatureSignature(
        "resp_rate_bpm", "resp", +1, "moderate", True,
        "aumento de frecuencia respiratoria bajo demanda/estrés",
    ),
}

#: Features con firma **débil**, solo en el conjunto de sensibilidad (§3.4).
SENSITIVITY_SIGNATURE: dict[str, FeatureSignature] = {
    "resp_interval_cv": FeatureSignature(
        "resp_interval_cv", "resp", +1, "weak", False,
        "irregularidad respiratoria; podría reflejar el habla",
    ),
}

#: Features **sin firma** y el motivo (documento §3.3).
EXCLUDED_FEATURES: dict[str, str] = {
    "eda_slope_us_per_s": (
        "deriva/adaptación dentro de 60 s; el signo no es identificable "
        "sin modelo de recuperación"
    ),
    "ecg_rr_mean_ms": (
        "inversa aritmética de ecg_hr_bpm; incluirla duplicaría el peso de HR"
    ),
    "ecg_n_beats": "conteo = HR × duración y nº de detecciones; no es un estado",
    "resp_amplitude_pct": (
        "profundidad: sube con el habla, baja con respiración superficial; ambiguo"
    ),
    "resp_std_pct": "variabilidad de amplitud; dirección no fijada por la literatura",
    "resp_n_cycles": "conteo = frecuencia × duración; redundante con resp_rate_bpm",
    "temp_mean_degc": (
        "vasoconstricción periférica mal identificable: confundida con el orden "
        "del protocolo (r = 0,801) y fuera del conjunto primario"
    ),
    "temp_std_degc": "sin mecanismo direccional claro; temperatura fuera del primario",
    "temp_slope_degc_per_s": "idem",
    "temp_range_degc": "idem",
}

#: Conjunto primario: las 12 firmas ``strong``/``moderate``.
PRIMARY_FEATURES: frozenset[str] = frozenset(STRESS_SIGNATURE)

#: Conjunto reducido: una feature por mecanismo (neutraliza que EDA sea 8 de 12).
COMPACT_FEATURES: frozenset[str] = frozenset(
    {
        "eda_mean_us",
        "eda_phasic_std_us",
        "ecg_hr_bpm",
        "ecg_rmssd_ms",
        "resp_rate_bpm",
    }
)

#: Sensibilidad: primario + la única firma débil.
SENSITIVITY_FEATURES: frozenset[str] = PRIMARY_FEATURES | frozenset(
    SENSITIVITY_SIGNATURE
)

#: Conjuntos con nombre, tal como los expone la CLI (``--coherence-set``).
FEATURE_SETS: dict[str, frozenset[str]] = {
    "primary": PRIMARY_FEATURES,
    "compact": COMPACT_FEATURES,
    "sensitivity": SENSITIVITY_FEATURES,
}

#: Niveles que entran en el conjunto primario.
PRIMARY_LEVELS: tuple[str, ...] = ("strong", "moderate")


def feature_set(name: str) -> frozenset[str]:
    """Devuelve un conjunto con nombre; error explícito si no existe."""
    try:
        return FEATURE_SETS[name]
    except KeyError:  # pragma: no cover - la CLI restringe las opciones
        raise ValueError(
            f"Conjunto de firma desconocido: {name!r}. "
            f"Opciones: {sorted(FEATURE_SETS)}"
        ) from None


def sign_of(metric: str, features: Iterable[str] | None = None) -> int:
    """Dirección esperada bajo ``stress`` (``0`` = sin firma).

    ``features`` restringe el cálculo a un conjunto (``primary``, ``compact``
    o ``sensitivity``). Si se omite, cualquier firma registrada vale.
    """
    signature = STRESS_SIGNATURE.get(metric) or SENSITIVITY_SIGNATURE.get(metric)
    if signature is None:
        return 0
    if features is not None and metric not in set(features):
        return 0
    return signature.sign


def has_signature(metric: str) -> bool:
    """¿La métrica tiene firma registrada (en cualquier nivel)?"""
    return metric in STRESS_SIGNATURE or metric in SENSITIVITY_SIGNATURE


def signature_table(features: Iterable[str] | None = None) -> list[dict[str, object]]:
    """Tabla serializable de firmas (para informes y auditoría)."""
    rows: list[dict[str, object]] = []
    for table in (STRESS_SIGNATURE, SENSITIVITY_SIGNATURE):
        for metric, signature in table.items():
            if features is not None and metric not in set(features):
                continue
            rows.append(signature.to_dict())
    return rows


def excluded_table() -> list[dict[str, str]]:
    """Tabla serializable de features sin firma y su motivo."""
    return [{"metric": metric, "reason": reason} for metric, reason in EXCLUDED_FEATURES.items()]


# ===========================================================================
# Segunda tarea: estadios de sueño (DREAMT)
# ===========================================================================
#
# Congelada en ``docs/piloto_dreamt.md`` §9 el 2026-11-22, **antes** de
# generar ninguna respuesta. La dirección es relativa a la referencia
# intra-sujeto de la noche (z sobre todas las ventanas puras del sujeto), no
# respecto de otra clase: por eso ``n2`` aparece con signos negativos en
# ``ecg_hr_bpm`` (la noche está dominada por N2, así que N2 queda cerca de su
# propia mediana y lo informativo es la desviación de las demás clases).

#: Una feature por mecanismo: movimiento, cronotropismo y actividad
#: sudomotora. Se evita repetir features del mismo mecanismo para que ninguna
#: modalidad pese más que las demás en la tasa de coherencia.
#:
#: La tabla se fijó con **tres** features, no cuatro: la rama *wearable* de
#: DREAMT no permite HRV (``IBI`` cuantizada con *dither* y ``BVP`` con picos
#: inestables; ver ``sources/external/dreamt.py`` y la desviación D7 del
#: preregistro), así que ``ecg_rmssd_ms`` y ``ecg_sdnn_ms`` no existen en esta
#: tarea. ``rem`` se queda sin ``ecg_hr_bpm`` porque su frecuencia cardiaca cae
#: prácticamente en la mediana de la noche y el signo no sería identificable.
SLEEP_SIGNATURES: dict[str, dict[str, int]] = {
    "wake": {
        "acc_std_g": +1,
        "ecg_hr_bpm": +1,
        "eda_std_us": +1,
    },
    "n2": {
        "acc_std_g": -1,
        "ecg_hr_bpm": -1,
        "eda_std_us": -1,
    },
    "rem": {
        "acc_std_g": -1,
        "eda_std_us": +1,
    },
    # N1 no tiene firma: sus direcciones coinciden con N2 en estas features, así
    # que no sería discriminativa (ver §9 del preregistro).
    "n1": {},
}

#: Motivo fisiológico de cada dirección (para el informe P0 y la auditoría).
SLEEP_SIGNATURE_RATIONALE: dict[str, str] = {
    "acc_std_g": (
        "movimiento en vigilia y atonía muscular en REM son los hechos mejor "
        "establecidos de la polisomnografía"
    ),
    "ecg_hr_bpm": (
        "la frecuencia cardiaca baja al dormir y sube en vigilia; N2 queda por "
        "debajo de la mediana de la noche"
    ),
    "eda_std_us": (
        "la actividad fásica de la piel es mayor en vigilia y en REM que en "
        "NREM estable"
    ),
}

#: Clases que se declaran **sin firma** y el motivo.
SLEEP_EXCLUDED_CLASSES: dict[str, str] = {
    "n1": (
        "sus direcciones coinciden con n2 en las features con firma; no permite "
        "distinguir N1 de N2, así que se excluye del cálculo de coherencia"
    ),
}

#: Features sin firma en la tarea de sueño y el motivo.
SLEEP_EXCLUDED_FEATURES: dict[str, str] = {
    "temp_mean_degc": (
        "confusor de orden de la noche: la temperatura cutánea sigue el ciclo "
        "y el orden de los estadios (equivalente al confusor térmico de WESAD)"
    ),
    "temp_std_degc": "idem: sin mecanismo direccional propio del estadio",
    "temp_slope_degc_per_s": "idem",
    "temp_range_degc": "idem",
    "eda_slope_us_per_s": "deriva/adaptación dentro de la ventana; signo no identificable",
    "ecg_rr_mean_ms": "inversa aritmética de ecg_hr_bpm; duplicaría el peso de HR",
    "ecg_n_beats": "conteo = HR × duración; no es un estado",
    "ecg_sdnn_ms": (
        "no disponible en la rama wearable: sin intervalos RR fiables (D7). "
        "Se conserva la clave en el vocabulario para no silenciar la ausencia"
    ),
    "ecg_rmssd_ms": "idem: sin intervalos RR fiables en esta tarea",
    "acc_mean_g": "incluye la gravedad; no es direccional",
    "acc_p95_g": "redundante con acc_std_g",
    "acc_energy_g2": "redundante con acc_std_g",
    "eda_tonic_mean_us": "redundante con eda_std_us en esta tarea",
    "eda_phasic_std_us": "redundante con eda_std_us",
    "eda_scr_amplitude_mean_us": "depende del umbral SCR (calibrado para EDA de pecho)",
    "eda_scr_amplitude_max_us": "idem",
}

#: Sensibilidad: la tabla primaria más dos features dependientes de umbral o
#: redundantes. ``eda_scr_rate_per_min`` se queda fuera del primario porque el
#: umbral SCR (0,05 µS) se calibró para EDA de pecho en WESAD y satura en la
#: EDA de muñeca de la E4.
SLEEP_SENSITIVITY_SIGNATURES: dict[str, dict[str, int]] = {
    "wake": {"eda_scr_rate_per_min": +1, "acc_motion_fraction": +1},
    "n2": {"eda_scr_rate_per_min": -1, "acc_motion_fraction": -1},
    "rem": {"eda_scr_rate_per_min": +1, "acc_motion_fraction": -1},
    "n1": {},
}

#: Conjunto primario (unión de las features con firma en alguna clase).
SLEEP_PRIMARY_FEATURES: frozenset[str] = frozenset(
    metric for table in SLEEP_SIGNATURES.values() for metric in table
)

#: Conjunto de sensibilidad (primario + firmas dependientes de umbral).
SLEEP_SENSITIVITY_FEATURES: frozenset[str] = SLEEP_PRIMARY_FEATURES | frozenset(
    metric for table in SLEEP_SENSITIVITY_SIGNATURES.values() for metric in table
)

#: Conjuntos con nombre para la tarea de sueño.
SLEEP_FEATURE_SETS: dict[str, frozenset[str]] = {
    "primary": SLEEP_PRIMARY_FEATURES,
    "sensitivity": SLEEP_SENSITIVITY_FEATURES,
}

#: Clases con firma (las que entran en la tasa de coherencia).
SLEEP_CLASSES_WITH_SIGNATURE: tuple[str, ...] = tuple(
    klass for klass, table in SLEEP_SIGNATURES.items() if table
)


def sleep_sign(
    klass: str,
    metric: str,
    features: Iterable[str] | None = None,
) -> int:
    """Dirección esperada de ``metric`` bajo la clase ``klass`` (``0`` = sin firma).

    Args:
        klass: clase de decisión (``wake``, ``n1``, ``n2``, ``rem``).
        metric: nombre canónico de la feature.
        features: restringe a un conjunto (``primary``/``sensitivity``).

    Returns:
        ``+1``, ``-1`` o ``0`` si la clase no tiene firma para esa feature.
    """
    if features is not None and metric not in set(features):
        return 0
    for table in (SLEEP_SIGNATURES, SLEEP_SENSITIVITY_SIGNATURES):
        value = table.get(klass, {}).get(metric)
        if value:
            return int(value)
    return 0


def has_sleep_signature(klass: str, metric: str | None = None) -> bool:
    """¿La clase tiene firma (y, si se indica, la métrica dentro de ella)?"""
    if metric is None:
        return bool(SLEEP_SIGNATURES.get(klass))
    return sleep_sign(klass, metric) != 0


def sleep_feature_set(name: str) -> frozenset[str]:
    """Conjunto de firma de sueño con nombre; error explícito si no existe."""
    try:
        return SLEEP_FEATURE_SETS[name]
    except KeyError:
        raise ValueError(
            f"Conjunto de firma de sueño desconocido: {name!r}. "
            f"Opciones: {sorted(SLEEP_FEATURE_SETS)}"
        ) from None


def sleep_signature_table() -> list[dict[str, object]]:
    """Tabla serializable ``clase × feature`` de la firma de sueño."""
    metrics = sorted(SLEEP_PRIMARY_FEATURES | SLEEP_SENSITIVITY_FEATURES)
    rows: list[dict[str, object]] = []
    for klass in SLEEP_SIGNATURES:
        row: dict[str, object] = {"class": klass}
        for metric in metrics:
            row[metric] = sleep_sign(klass, metric)
        rows.append(row)
    return rows
