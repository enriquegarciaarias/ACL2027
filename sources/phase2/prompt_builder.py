"""Constructor de prompts para el LLM (Fase 2).

Construye prompts en inglés a partir de las representaciones LLM filtradas
(vistas ``numeric_llm`` y ``text_llm`` generadas en Fase 1).

Invariantes anti-leakage:
- Nunca se incluyen etiquetas (label), tiempos absolutos, IDs reales de
  ventana o sujeto, SQI, cobertura, flags ni procedencia.
- Solo se incluyen los canales del conjunto mínimo experimental (EDA + HR).
- Nunca se trunca el prompt: si excede ``num_ctx``, se registra como
  exclusión y no se genera el prompt.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from sources.ingestion.represent import LLM_ALLOWED_FEATURES
from sources.phase2.response_schema import (
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_STRESS_BINARY,
    DECISION_MODES,
)

# Conjunto mínimo de canales experimental: EDA + frecuencia cardíaca.
# Features de chest_EDA y chest_ECG que se envían al LLM.
EDA_HR_FEATURES: tuple[str, ...] = (
    # EDA (chest_EDA)
    "eda_mean_us",
    "eda_std_us",
    "eda_slope_us_per_s",
    "eda_tonic_mean_us",
    "eda_phasic_std_us",
    "eda_scr_count",
    "eda_scr_rate_per_min",
    "eda_scr_amplitude_mean_us",
    "eda_scr_amplitude_max_us",
    # HR / cardiovascular (chest_ECG)
    "ecg_hr_bpm",
    "ecg_rr_mean_ms",
    "ecg_sdnn_ms",
    "ecg_rmssd_ms",
    "ecg_n_beats",
)

# Features adicionales para ampliar el conjunto experimental.
# Estas modalidades de pecho pueden ayudar a diferenciar estados que EDA+HR
# no puede distinguir (ej. baseline vs meditation vs amusement).
ADDITIONAL_FEATURES: tuple[str, ...] = (
    # Respiración (chest_Resp)
    "resp_rate_bpm",
    "resp_interval_cv",
    "resp_amplitude_pct",
    "resp_std_pct",
    "resp_n_cycles",
    # Temperatura (chest_Temp)
    "temp_mean_degc",
    "temp_std_degc",
    "temp_slope_degc_per_s",
    "temp_range_degc",
    # Pulsómetro (chest_BVP) — NO incluido: ``bvp`` se extrae de ``wrist_BVP``
    # y pertenece a la vista de muñeca. Se reserva para el experimento de
    # sobre-confianza en medidas derivadas (HR de BVP vs ECG de referencia).
)

# Modalidades de pecho permitidas (excluye muñeca, incluida bvp).
CHEST_MODALITIES: frozenset[str] = frozenset({"ecg", "eda", "resp", "temp"})

# Las features del prompt deben ser un subconjunto de las que contiene la
# representación LLM curada; si no, numeric y text volverían a divergir.
_CURATED_FEATURES = set(EDA_HR_FEATURES) | set(ADDITIONAL_FEATURES)
if not _CURATED_FEATURES <= LLM_ALLOWED_FEATURES:  # pragma: no cover
    raise RuntimeError(
        "features del prompt no incluidas en LLM_ALLOWED_FEATURES: "
        f"{sorted(_CURATED_FEATURES - LLM_ALLOWED_FEATURES)}"
    )

# Expresiones regulares para detectar fugas de información prohibida en el prompt.
LEAKAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    # Fuga de etiqueta real: asignación explícita de una etiqueta a la ventana
    # (ej. "label: stress", "this window is baseline", "true label: meditation").
    # La mera mención de las opciones en el esquema JSON de salida no es fuga.
    "labels": re.compile(
        r"\b(label|state|condition|true label|ground truth)\s*[:=]\s*"
        r"(baseline|stress|amusement|meditation)\b"
        r"|\b(window|this|the subject|the patient)\s+(is|was|are)\s+"
        r"(in\s+)?(a\s+)?(an\s+)?(baseline|stress|amusement|meditation)\b",
        re.IGNORECASE,
    ),
    # Timestamps absolutos y fechas (hora:minuto:segundo, o YYYY-MM-DD).
    # Las duraciones relativas (ej. "60 s" de ventana) son contexto legítimo,
    # no timestamps absolutos.
    "timestamps": re.compile(r"\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}"),
    "subject_ids": re.compile(r"\bS\d+\b"),
    "window_ids": re.compile(r"\bwindow_\d+\b|w\d{5,}"),
    "quality_terms": re.compile(
        r"\b(sqi|quality|coverage|flag)\b", re.IGNORECASE
    ),
}

# Ejemplos few-shot para mitigar el sesgo del modelo hacia "stress".
#
# PROVENANCE (corregido 2026-11-09): los valores son **medianas por clase
# calculadas solo con los sujetos de desarrollo** (S2, S3, S4; 138 ventanas),
# nunca con el conjunto de evaluación. Antes usaban las medias de las 705
# ventanas evaluadas, lo que constituía una fuga de estadísticos del conjunto
# de test al prompt. Ver ``docs/split_dev_test.md``.
#
# No se incluyen temperatura ni BVP: la temperatura deriva con el orden del
# protocolo WESAD (confusor) y BVP es una modalidad de muñeca reservada para el
# experimento de medidas derivadas.
from sources.phase2.development_split import (  # noqa: E402
    DEV_CLASS_COUNTS,
    DEV_N_WINDOWS,
    DEV_SUBJECTS,
)

FEW_SHOT_PROVENANCE = {
    "subjects": list(DEV_SUBJECTS),
    "n_windows": DEV_N_WINDOWS,
    "class_counts": dict(DEV_CLASS_COUNTS),
    "statistic": "median per class (rounded)",
    "split_version": "1.0",
}

#: Medianas por clase en unidades absolutas (solo sujetos de desarrollo).
FEW_SHOT_EXEMPLARS: dict[str, dict[str, float]] = {
    "baseline": {
        "ecg_hr_bpm": 58.7,
        "ecg_rmssd_ms": 91.0,
        "eda_mean_us": 1.6,
        "eda_scr_count": 0,
        "resp_rate_bpm": 17.6,
    },
    "stress": {
        "ecg_hr_bpm": 79.5,
        "ecg_rmssd_ms": 40.9,
        "eda_mean_us": 4.3,
        "eda_scr_count": 4,
        "resp_rate_bpm": 13.7,
    },
    "amusement": {
        "ecg_hr_bpm": 61.8,
        "ecg_rmssd_ms": 89.3,
        "eda_mean_us": 1.4,
        "eda_scr_count": 0,
        "resp_rate_bpm": 17.7,
    },
    "meditation": {
        "ecg_hr_bpm": 62.2,
        "ecg_rmssd_ms": 77.4,
        "eda_mean_us": 2.0,
        "eda_scr_count": 0,
        "resp_rate_bpm": 12.8,
    },
}

#: Medianas por clase en z intra-sujeto (solo sujetos de desarrollo).
FEW_SHOT_EXEMPLARS_STANDARDIZED: dict[str, dict[str, float]] = {
    "baseline": {
        "ecg_hr_bpm": -0.1,
        "ecg_rmssd_ms": 0.0,
        "eda_mean_us": -0.1,
        "eda_scr_count": 0.0,
        "resp_rate_bpm": 0.2,
    },
    "stress": {
        "ecg_hr_bpm": 2.1,
        "ecg_rmssd_ms": -1.0,
        "eda_mean_us": 1.3,
        "eda_scr_count": 2.0,
        "resp_rate_bpm": -0.3,
    },
    "amusement": {
        "ecg_hr_bpm": -0.2,
        "ecg_rmssd_ms": 0.5,
        "eda_mean_us": 0.0,
        "eda_scr_count": 0.0,
        "resp_rate_bpm": 0.5,
    },
    "meditation": {
        "ecg_hr_bpm": -0.3,
        "ecg_rmssd_ms": 0.5,
        "eda_mean_us": -0.1,
        "eda_scr_count": 0.0,
        "resp_rate_bpm": -0.8,
    },
}

# Variante para la vista estandarizada por sujeto (E12). Los valores son
# z-scores intra-sujeto: 0 = media de la sesión del sujeto, +1 = 1 DE por
# encima. Los ejemplos son ilustrativos y usan la misma escala.
STANDARDIZED_NOTE = (
    "All feature values are within-subject standardized scores: 0 is the "
    "subject's session reference and +1 is one scale unit above it. Small "
    "deviations (within about ±1) are normal. Use the dominant pattern across "
    "features; abstain only when key features are missing or clearly contradictory."
)

STANDARDIZED_RUBRIC = """Physiological decision guidance (use patterns, not fixed thresholds):
- stress: heart rate and EDA/SCR clearly above the subject's reference.
- meditation: respiratory rate clearly below the subject's reference, with heart rate at or below the reference and HRV at or above it.
- amusement: features close to the subject's reference with a slightly elevated respiratory rate.
- baseline: features close to the subject's reference without a consistent activation pattern.
If one pattern clearly applies, classify; if the evidence is contradictory or missing, abstain."""

#: Razonamientos por clase. Deben ser coherentes con los valores de desarrollo.
FEW_SHOT_REASONING: dict[str, str] = {
    "baseline": (
        "Resting heart rate with high HRV and a low SCR count, plus a normal "
        "waking respiratory rate, is consistent with baseline."
    ),
    "stress": (
        "Elevated heart rate and EDA with reduced HRV and several SCR bursts "
        "indicates an acute stress response."
    ),
    "amusement": (
        "Heart rate, EDA and SCR count are close to baseline, with a slightly "
        "higher respiratory rate; these channels do not separate amusement strongly."
    ),
    "meditation": (
        "High HRV together with a markedly lower respiratory rate and minimal "
        "SCR indicates a relaxed, meditative state."
    ),
}

INTER_SUBJECT_NOTE = (
    "Note: feature values vary substantially across subjects; rely on the "
    "joint pattern and relative levels rather than fixed population thresholds."
)


def format_example_value(value: float, standardized: bool) -> str:
    """Formatea un valor de ejemplo (con signo explícito en z-scores)."""
    if not standardized:
        return str(value)
    if value == 0:
        return "0.0"
    return f"{value:+.1f}"


def render_few_shot_examples(
    exemplars: dict[str, dict[str, float]],
    standardized: bool = False,
) -> str:
    """Renderiza los ejemplos few-shot a partir de los valores de desarrollo."""
    unit = "SD" if standardized else ""
    title = (
        "Examples (illustrative within-subject standardized patterns, not fixed thresholds):"
        if standardized
        else "Examples (illustrative patterns, not fixed thresholds):"
    )
    lines = [title, ""]
    for index, (label, values) in enumerate(exemplars.items(), start=1):
        heading = label.capitalize()
        if label == "baseline":
            heading = "Baseline"
        features = ", ".join(
            f"{name}={format_example_value(value, standardized)} {unit}".strip()
            for name, value in values.items()
        )
        lines.append(f"Example {index} - {heading}:")
        lines.append(f"Features: {features}")
        lines.append(f"Decision: {label}")
        lines.append(f"Reasoning: {FEW_SHOT_REASONING[label]}")
        lines.append("")
    return "\n".join(lines).rstrip()


FEW_SHOT_EXAMPLES: str = render_few_shot_examples(FEW_SHOT_EXEMPLARS)
FEW_SHOT_EXAMPLES_STANDARDIZED: str = render_few_shot_examples(
    FEW_SHOT_EXEMPLARS_STANDARDIZED, standardized=True
)



# Comprobación de consistencia explícita para prompts sobre la vista centrada.
# En z-scores intra-sujeto las relaciones aritméticas exactas no se pueden
# recalcular, pero las features ligadas por una misma señal deben estar en
# niveles estandarizados parecidos. Esa comprobación sí es verificable.
CONSISTENCY_RULE_STANDARDIZED: str = """Consistency check (perform before deciding):
The values are within-subject standardized scores, so features driven by the same underlying events must be at similar levels:
- ecg_n_beats and ecg_hr_bpm should be at similar standardized levels.
- eda_scr_count and eda_scr_rate_per_min should be at similar standardized levels.
- eda_mean_us and eda_tonic_mean_us should be at similar standardized levels.
- resp_n_cycles and resp_rate_bpm should be at similar standardized levels.
- eda_scr_amplitude_mean_us should not be at a higher standardized level than eda_scr_amplitude_max_us.
If any of these pairs differs by more than about 1.5 scale units, the window is internally inconsistent: set decision to "abstain" and name the contradicting pair in the reasoning."""


# Variante binaria (P2): endpoint primario ``stress`` vs ``non-stress``.
# ``non-stress`` agrupa baseline, amusement y meditation del protocolo WESAD.
BINARY_TASK_NOTE: str = (
    "Task: decide whether this 60-second window shows an acute stress response "
    "(decision \"stress\") or not (decision \"non-stress\"). The \"non-stress\" "
    "class covers calm baseline, amusement and meditative states."
)

#: Razonamientos para el prompt binario (coherentes con los valores de dev).
BINARY_FEW_SHOT_REASONING: dict[str, str] = {
    "stress": (
        "Elevated heart rate and EDA with reduced HRV and several SCR bursts "
        "indicates an acute stress response."
    ),
    "baseline": (
        "Resting heart rate with high HRV and a low SCR count, plus a normal "
        "waking respiratory rate, indicates no stress response."
    ),
    "meditation": (
        "High HRV together with a markedly lower respiratory rate and minimal "
        "SCR reflects a relaxed state, not stress."
    ),
}


def render_binary_few_shot_examples(
    exemplars: dict[str, dict[str, float]],
    standardized: bool = False,
) -> str:
    """Renderiza los ejemplos few-shot del prompt binario (solo dev)."""
    unit = "SD" if standardized else ""
    title = (
        "Examples (illustrative within-subject standardized patterns, not fixed thresholds):"
        if standardized
        else "Examples (illustrative patterns, not fixed thresholds):"
    )
    layout = (
        ("stress", "Stress"),
        ("baseline", "Non-stress (calm baseline)"),
        ("meditation", "Non-stress (meditation)"),
    )
    lines = [title, ""]
    for index, (label, heading) in enumerate(layout, start=1):
        values = exemplars[label]
        features = ", ".join(
            f"{name}={format_example_value(value, standardized)} {unit}".strip()
            for name, value in values.items()
        )
        decision = "stress" if label == "stress" else "non-stress"
        lines.append(f"Example {index} - {heading}:")
        lines.append(f"Features: {features}")
        lines.append(f"Decision: {decision}")
        lines.append(f"Reasoning: {BINARY_FEW_SHOT_REASONING[label]}")
        lines.append("")
    return "\n".join(lines).rstrip()


BINARY_FEW_SHOT_EXAMPLES: str = render_binary_few_shot_examples(FEW_SHOT_EXEMPLARS)
BINARY_FEW_SHOT_EXAMPLES_STANDARDIZED: str = render_binary_few_shot_examples(
    FEW_SHOT_EXEMPLARS_STANDARDIZED, standardized=True
)

BINARY_SCHEMA_DECISION: str = '"stress" | "non-stress" | "abstain"'

BINARY_RULES: tuple[str, ...] = (
    '- Use "stress" only when the pattern indicates an acute stress response.',
    '- Use "non-stress" for calm baseline, amusement or meditative states.',
)

# Validación externa (P3) con K-EmoCon: arousal y valencia alta vs baja.
AROUSAL_TASK_NOTE: str = (
    "Task: decide whether this 60-second window shows higher (decision "
    "\"high-arousal\") or lower (decision \"low-arousal\") physiological arousal "
    "than the participant's own typical level."
)
AROUSAL_SCHEMA_DECISION: str = '"high-arousal" | "low-arousal" | "abstain"'
AROUSAL_RULES: tuple[str, ...] = (
    '- Use "high-arousal" when activation is above the participant\'s typical level.',
    '- Use "low-arousal" when activation is at or below that level.',
)

VALENCE_TASK_NOTE: str = (
    "Task: decide whether this 60-second window reflects a more positive "
    "(decision \"high-valence\") or more negative (decision \"low-valence\") "
    "affective state than the participant's own typical level."
)
VALENCE_SCHEMA_DECISION: str = '"high-valence" | "low-valence" | "abstain"'
VALENCE_RULES: tuple[str, ...] = (
    '- Use "high-valence" when the pattern is more positive than typical.',
    '- Use "low-valence" when the pattern is more negative than typical.',
)

#: Tareas binarias disponibles fuera del endpoint de estrés.
BINARY_TASKS: dict[str, str] = {
    "stress_binary": BINARY_TASK_NOTE,
    "arousal_binary": AROUSAL_TASK_NOTE,
    "valence_binary": VALENCE_TASK_NOTE,
}
BINARY_SCHEMAS: dict[str, str] = {
    "stress_binary": BINARY_SCHEMA_DECISION,
    "arousal_binary": AROUSAL_SCHEMA_DECISION,
    "valence_binary": VALENCE_SCHEMA_DECISION,
}
BINARY_RULE_SETS: dict[str, tuple[str, ...]] = {
    "stress_binary": BINARY_RULES,
    "arousal_binary": AROUSAL_RULES,
    "valence_binary": VALENCE_RULES,
}

# Rúbrica de confianza graduada (P4). El modelo tiende a saturar la confianza
# en un único valor (85 en 703/705 respuestas del corpus), lo que inutiliza la
# calibración y el umbral selectivo. Esta rúbrica ancla la escala a la fuerza
# de la evidencia y prohíbe explícitamente el valor por defecto.
GRADED_CONFIDENCE_RUBRIC: str = """Confidence guidance (use the whole range; do not default to one fixed value):
- 50-60: the evidence is ambiguous or mixed; the pattern is weak.
- 60-75: moderate; some features support the decision and others do not.
- 75-90: strong; most features agree with the decision.
- 90-100: very strong; nearly all features agree and none contradicts the decision.
Set confidence to the value that best reflects how strongly the features support your decision."""


RHYTHM_TASK_NOTE: str = (
    "Task: classify a single-lead ECG recording into one of four rhythm "
    "classes: \"normal\" (normal sinus rhythm), \"af\" (atrial fibrillation), "
    "\"other-rhythm\" (any other rhythm) or \"noisy\" (the recording is too "
    "noisy to be interpreted). Use \"abstain\" only when the available "
    "features do not allow a decision."
)
RHYTHM_SCHEMA_DECISION: str = (
    '"normal" | "af" | "other-rhythm" | "noisy" | "abstain"'
)
RHYTHM_RULES: tuple[str, ...] = (
    '- Use "noisy" when the tracing cannot support any rhythm interpretation.',
    '- Use "other-rhythm" for rhythms that are neither normal sinus nor AF.',
    "- Base the decision only on the ECG features provided.",
)


RHYTHM_REASONING: dict[str, str] = {
    "normal": "Regular rhythm with heart rate and HRV close to the population reference.",
    "af": "Markedly irregular RR intervals with elevated SDNN/RMSSD suggest atrial fibrillation.",
    "other-rhythm": "Rhythm is neither clearly regular nor typical of atrial fibrillation.",
    "noisy": "The features are inconsistent or dominated by artifact, so the tracing cannot be interpreted.",
}


def render_rhythm_examples(exemplars: Mapping[str, Mapping[str, float]]) -> str:
    """Renderiza ejemplos few-shot de ritmo a partir de valores de desarrollo."""
    lines = ["Examples (illustrative patterns from development recordings, not fixed thresholds):", ""]
    for index, (label, values) in enumerate(exemplars.items(), start=1):
        features = ", ".join(f"{name}={value:+.2f} SD" for name, value in values.items())
        lines.append(f"Example {index} - {label.capitalize()}:")
        lines.append(f"Features: {features}")
        lines.append(f"Decision: {label}")
        lines.append(f"Reasoning: {RHYTHM_REASONING.get(label, '')}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_rhythm_prompt(
    payload: Mapping[str, Any],
    window_id_opaque: str,
    duration_s: float | None = None,
    num_ctx: int = 32768,
    dataset_note: str | None = None,
    few_shot_examples: str | None = None,
) -> str:
    """Construye el prompt de clasificación de ritmo (PhysioNet 2017).

    Sin few-shot: no hay ejemplos de este dataset y fabricarlos introduciría
    sesgo. Los valores son z poblacionales calculados con registros de
    desarrollo.
    """
    lines: list[str] = []
    lines.append("You are a clinical ECG analysis assistant.")
    lines.append("")
    lines.append(RHYTHM_TASK_NOTE)
    lines.append("")
    if dataset_note:
        lines.append(dataset_note)
        lines.append("")
    lines.append(
        "All feature values are population-standardized scores computed from "
        "development recordings: 0 is the population reference and +1 is one "
        "scale unit above it."
    )
    lines.append("")
    if few_shot_examples:
        lines.append(few_shot_examples)
        lines.append("")
        lines.append("Now classify the following recording based only on its features:")
        lines.append("")
    lines.append("Input features (single-lead ECG):")
    lines.append("")
    for modality, feature_block in payload.get("modalities", {}).items():
        lines.append(f"Modality: {modality.upper()}")
        for name, meta in feature_block.items():
            lines.append(f"  - {_format_numeric_feature(name, meta)}")
        lines.append("")
    lines.append(f"Window ID (opaque): {window_id_opaque}")
    if duration_s is not None:
        lines.append(f"Window duration: {duration_s} s")
    lines.append("")
    lines.append("Respond ONLY with a JSON object containing exactly these keys:")
    lines.append("")
    lines.append("{")
    lines.append(f'  "decision": {RHYTHM_SCHEMA_DECISION},')
    lines.append('  "confidence": <integer 0-100>,')
    lines.append('  "claims": [')
    lines.append('    {"metric": "<feature name>", "channel": "<modality>", "direction": "increase|decrease|stable", "magnitude": "low|medium|high"}')
    lines.append("  ],")
    lines.append('  "reasoning": "<brief explanation>"')
    lines.append("}")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use only the features provided above. Do not invent data.")
    lines.append('- If the evidence is insufficient, set decision to "abstain".')
    lines.append("- confidence must be an integer between 0 and 100.")
    lines.append("- Each claim must reference a real feature from the input.")
    for rule in RHYTHM_RULES:
        lines.append(rule)
    lines.append("- Respond with valid JSON only. No markdown, no extra text.")

    prompt = "\n".join(lines)
    estimated = estimate_tokens(prompt)
    if estimated > num_ctx:
        raise ValueError(
            f"Prompt excede num_ctx: {estimated} tokens estimados > {num_ctx}. "
            "Exclusión registrada, no se trunca."
        )
    return prompt


def estimate_tokens(text: str) -> int:
    """Estima el número de tokens de un texto (aproximación conservadora).

    Usa ~4 caracteres por token, redondeando hacia arriba. No requiere
    tokenizer externo y sirve para rechazar prompts demasiado largos antes
    de enviar al LLM.
    """
    if not text:
        return 0
    return -(-len(text) // 4)  # ceil division


def filter_eda_hr_modalities(
    modalities: Mapping[str, dict[str, Any]],
    include_additional: bool = False,
) -> dict[str, dict[str, Any]]:
    """Filtra un bloque de modalidades para incluir solo EDA + HR.

    Devuelve una copia con solo las features listadas en ``EDA_HR_FEATURES``
    y solo de las modalidades de pecho (``ecg`` y ``eda``). Las modalidades
    de muñeca (``wrist_*``) u otras (``bvp``, ``resp``, ``temp``) se excluyen
    porque no forman parte del conjunto mínimo experimental.

    Si ``include_additional`` es True, también se incluyen las modalidades
    de pecho adicionales (``resp``, ``temp``, ``bvp``) con sus features
    definidas en ``ADDITIONAL_FEATURES``. Esto puede ayudar a diferenciar
    estados que EDA+HR sola no distingue (ej. baseline vs meditation).

    Las modalidades que no contienen ninguna feature permitida se omiten.
    """
    allowed = set(EDA_HR_FEATURES)
    if include_additional:
        allowed = allowed | set(ADDITIONAL_FEATURES)
        allowed_modalities = CHEST_MODALITIES
    else:
        allowed_modalities = {"ecg", "eda"}
    filtered: dict[str, dict[str, Any]] = {}
    for modality, feature_block in modalities.items():
        # Excluir modalidades de muñeca y otras no permitidas.
        if modality not in allowed_modalities:
            continue
        kept = {
            name: meta
            for name, meta in feature_block.items()
            if name in allowed
        }
        if kept:
            filtered[modality] = kept
    return filtered


def _format_numeric_feature(name: str, meta: dict[str, Any]) -> str:
    """Formatea una feature numérica como 'label: value unit'."""
    value = meta.get("value")
    unit = meta.get("unit")
    if value is None:
        return f"{name}: not available"
    suffix = f" {unit}" if unit else ""
    return f"{name}: {value}{suffix}"


def build_numeric_prompt(
    payload: Mapping[str, Any],
    window_id_opaque: str,
    duration_s: float | None = None,
    num_ctx: int = 32768,
    use_few_shot: bool = False,
    include_additional: bool = False,
    standardized: bool = False,
    rubric: bool = False,
    consistency_check: bool = False,
    decision_mode: str = "affective",
    graded_confidence: bool = False,
    dataset_note: str | None = None,
) -> str:
    """Construye un prompt en inglés a partir de una representación ``numeric_llm``.

    Incluye solo las features EDA + HR del conjunto mínimo experimental.

    Args:
        payload: payload de la representación ``numeric_llm`` (dict con
            ``modalities``, ``feature_version``, ``window``).
        window_id_opaque: identificador opaco de la ventana (ya generado
            por la vista LLM, nunca el ID real).
        duration_s: duración de la ventana en segundos (opcional, solo para
            contexto, nunca tiempos absolutos).
        num_ctx: tamaño máximo de contexto del modelo en tokens. Si el
            prompt estimado excede este valor, se lanza ``ValueError``.
        use_few_shot: si True, incluye ejemplos few-shot (uno por clase
            afectiva) para mitigar el sesgo del modelo hacia "stress".
        include_additional: si True, incluye modalidades adicionales de pecho
            (resp, temp, bvp) además de EDA+HR.
        standardized: si True, el prompt describe la escala como z-scores
            intra-sujeto (vista E12).
        rubric: si True, añade la rúbrica fisiológica explícita (solo en la
            vista estandarizada; empeoró la calibración en S3).
        consistency_check: si True, añade la comprobación de consistencia
            interna verificable en la escala estandarizada (P1 de Fase 2).
        graded_confidence: si True, añade la rúbrica de confianza graduada
            (P4) para evitar la saturación de la confianza en un valor fijo.
        dataset_note: nota opcional (sin etiquetas) sobre los canales
            disponibles; se usa en datasets externos (P3) cuando faltan
            modalidades respecto a WESAD.

    Returns:
        string del prompt listo para enviar al LLM.

    Raises:
        ValueError: si el prompt estimado excede ``num_ctx`` (exclusión
            registrada, no se trunca).
    """
    modalities = filter_eda_hr_modalities(
        payload.get("modalities", {}), include_additional=include_additional
    )

    if decision_mode not in DECISION_MODES:
        raise ValueError(
            f"decision_mode no válido: {decision_mode}. "
            f"Valores permitidos: {sorted(DECISION_MODES)}"
        )
    binary = decision_mode in BINARY_TASKS

    lines: list[str] = []
    lines.append("You are a physiological signal analysis assistant.")
    lines.append("")
    if binary:
        lines.append(BINARY_TASKS[decision_mode])
        features_header = (
            "Input features (within-subject z-scores; 0 = subject mean, "
            "±1 = 1 standard deviation):"
            if standardized
            else "Input features (chest channels only):"
        )
    elif standardized:
        lines.append(
            "Task: Classify the affective state of a 60-second window based on "
            "within-subject standardized chest physiological features (z-scores) "
            "covering EDA, heart rate, respiration and skin temperature."
        )
        features_header = (
            "Input features (within-subject z-scores; 0 = subject mean, "
            "±1 = 1 standard deviation):"
        )
    elif include_additional:
        lines.append(
            "Task: Classify the affective state of a 60-second window based on "
            "chest physiological features: electrodermal activity (EDA), heart "
            "rate (HR), respiration, skin temperature and pulse (BVP)."
        )
        features_header = "Input features (chest channels only):"
    else:
        lines.append(
            "Task: Classify the affective state of a 60-second window based on "
            "electrodermal activity (EDA) and heart rate (HR) features."
        )
        features_header = "Input features (EDA + HR only):"
    lines.append("")
    if dataset_note:
        lines.append(dataset_note)
        lines.append("")

    lines.append(STANDARDIZED_NOTE if standardized else INTER_SUBJECT_NOTE)
    lines.append("")

    if use_few_shot:
        if binary and decision_mode == DECISION_MODE_STRESS_BINARY:
            lines.append(
                BINARY_FEW_SHOT_EXAMPLES_STANDARDIZED
                if standardized
                else BINARY_FEW_SHOT_EXAMPLES
            )
        else:
            lines.append(
                FEW_SHOT_EXAMPLES_STANDARDIZED if standardized else FEW_SHOT_EXAMPLES
            )
        lines.append("")
        lines.append("Now classify this window based only on the features below:")
        lines.append("")

    lines.append(features_header)
    lines.append("")

    for modality, feature_block in modalities.items():
        lines.append(f"Modality: {modality.upper()}")
        for name, meta in feature_block.items():
            lines.append(f"  - {_format_numeric_feature(name, meta)}")
        lines.append("")

    lines.append(f"Window ID (opaque): {window_id_opaque}")
    if duration_s is not None:
        lines.append(f"Window duration: {duration_s} s")
    lines.append("")

    if standardized and rubric:
        lines.append(STANDARDIZED_RUBRIC)
        lines.append("")

    if consistency_check:
        lines.append(CONSISTENCY_RULE_STANDARDIZED)
        lines.append("")

    if graded_confidence:
        lines.append(GRADED_CONFIDENCE_RUBRIC)
        lines.append("")

    lines.append("Respond ONLY with a JSON object containing exactly these keys:")
    lines.append("")
    lines.append("{")
    if binary:
        lines.append(f'  "decision": {BINARY_SCHEMAS[decision_mode]},')
    else:
        lines.append('  "decision": "baseline" | "stress" | "amusement" | "meditation" | "abstain",')
    lines.append('  "confidence": <integer 0-100>,')
    lines.append('  "claims": [')
    lines.append('    {"metric": "<feature name>", "channel": "<modality>", "direction": "increase|decrease|stable", "magnitude": "low|medium|high"}')
    lines.append("  ],")
    lines.append('  "reasoning": "<brief explanation>"')
    lines.append("}")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use only the features provided above. Do not invent data.")
    lines.append("- If the evidence is insufficient, set decision to \"abstain\".")
    lines.append("- confidence must be an integer between 0 and 100.")
    lines.append("- Each claim must reference a real feature from the input.")
    if binary:
        for rule in BINARY_RULE_SETS[decision_mode]:
            lines.append(rule)
    else:
        lines.append("- Consider all four classes equally; do not bias toward any particular state.")
    lines.append("- Respond with valid JSON only. No markdown, no extra text.")

    prompt = "\n".join(lines)

    # Verificación de tamaño: nunca truncar, rechazar si excede.
    estimated = estimate_tokens(prompt)
    if estimated > num_ctx:
        raise ValueError(
            f"Prompt excede num_ctx: {estimated} tokens estimados > {num_ctx}. "
            "Exclusión registrada, no se trunca."
        )

    return prompt


def build_text_prompt(
    text_representation: str,
    window_id_opaque: str,
    duration_s: float | None = None,
    num_ctx: int = 32768,
    use_few_shot: bool = False,
    standardized: bool = False,
    rubric: bool = False,
    consistency_check: bool = False,
    decision_mode: str = "affective",
    graded_confidence: bool = False,
) -> str:
    """Construye un prompt en inglés a partir de una representación ``text_llm``.

    Args:
        text_representation: texto estructurado de la representación ``text_llm``.
        window_id_opaque: identificador opaco de la ventana.
        duration_s: duración de la ventana en segundos (opcional).
        num_ctx: tamaño máximo de contexto del modelo en tokens.
        use_few_shot: si True, incluye ejemplos few-shot (uno por clase
            afectiva) para mitigar el sesgo del modelo hacia "stress".
        standardized: si True, describe la escala como z-scores intra-sujeto.
        rubric: si True, añade la rúbrica fisiológica explícita.
        consistency_check: si True, añade la comprobación de consistencia.
        decision_mode: ``"affective"`` (4 clases) o ``"stress_binary"``.
        graded_confidence: si True, añade la rúbrica de confianza graduada (P4).

    Returns:
        string del prompt listo para enviar al LLM.

    Raises:
        ValueError: si el prompt estimado excede ``num_ctx`` o el modo no existe.
    """
    if decision_mode not in DECISION_MODES:
        raise ValueError(
            f"decision_mode no válido: {decision_mode}. "
            f"Valores permitidos: {sorted(DECISION_MODES)}"
        )
    binary = decision_mode in BINARY_TASKS

    lines: list[str] = []
    lines.append("You are a physiological signal analysis assistant.")
    lines.append("")
    if binary:
        lines.append(BINARY_TASKS[decision_mode])
    else:
        lines.append("Task: Classify the affective state of a 60-second window "
                     "based on the physiological signal summary below.")
    lines.append("")

    lines.append(STANDARDIZED_NOTE if standardized else INTER_SUBJECT_NOTE)
    lines.append("")

    if use_few_shot:
        if binary and decision_mode != DECISION_MODE_STRESS_BINARY:
            use_few_shot = False
        elif binary:
            lines.append(
                BINARY_FEW_SHOT_EXAMPLES_STANDARDIZED
                if standardized
                else BINARY_FEW_SHOT_EXAMPLES
            )
        else:
            lines.append(
                FEW_SHOT_EXAMPLES_STANDARDIZED if standardized else FEW_SHOT_EXAMPLES
            )
        lines.append("")
        lines.append(
            "Now classify this window based only on the features below. "
            "The input is a textual summary of the same chest features."
        )
        lines.append("")

    lines.append("Window ID (opaque): " + window_id_opaque)
    if duration_s is not None:
        lines.append(f"Window duration: {duration_s} s")
    lines.append("")
    lines.append("Signal summary:")
    lines.append(text_representation)
    lines.append("")
    if standardized and rubric:
        lines.append(STANDARDIZED_RUBRIC)
        lines.append("")

    if consistency_check:
        lines.append(
            CONSISTENCY_RULE_STANDARDIZED
            if standardized
            else "Consistency check (perform before deciding): features driven "
            "by the same underlying events must agree (SCR count with SCR rate, "
            "heart rate with beat count, mean EDA with tonic EDA). If they "
            "contradict each other, set decision to \"abstain\"."
        )
        lines.append("")

    if graded_confidence:
        lines.append(GRADED_CONFIDENCE_RUBRIC)
        lines.append("")

    lines.append("Respond ONLY with a JSON object containing exactly these keys:")
    lines.append("")
    lines.append("{")
    if binary:
        lines.append(f'  "decision": {BINARY_SCHEMAS[decision_mode]},')
    else:
        lines.append('  "decision": "baseline" | "stress" | "amusement" | "meditation" | "abstain",')
    lines.append('  "confidence": <integer 0-100>,')
    lines.append('  "claims": [')
    lines.append('    {"metric": "<feature name>", "channel": "<modality>", "direction": "increase|decrease|stable", "magnitude": "low|medium|high"}')
    lines.append("  ],")
    lines.append('  "reasoning": "<brief explanation>"')
    lines.append("}")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Use only the information provided above. Do not invent data.")
    lines.append("- If the evidence is insufficient, set decision to \"abstain\".")
    lines.append("- confidence must be an integer between 0 and 100.")
    lines.append("- Each claim must reference a real feature from the input.")
    if binary:
        for rule in BINARY_RULE_SETS[decision_mode]:
            lines.append(rule)
    else:
        lines.append("- Consider all four classes equally; do not bias toward any particular state.")
    lines.append("- Respond with valid JSON only. No markdown, no extra text.")

    prompt = "\n".join(lines)

    estimated = estimate_tokens(prompt)
    if estimated > num_ctx:
        raise ValueError(
            f"Prompt excede num_ctx: {estimated} tokens estimados > {num_ctx}. "
            "Exclusión registrada, no se trunca."
        )

    return prompt


def check_prompt_leakage(prompt: str) -> dict[str, list[str]]:
    """Busca patrones de fuga de información prohibida en un prompt.

    Args:
        prompt: string del prompt generado.

    Returns:
        dict con claves por tipo de fuga y listas de coincidencias encontradas.
        Un dict vacío significa que no se detectó fuga.
    """
    leaks: dict[str, list[str]] = {}
    for name, pattern in LEAKAGE_PATTERNS.items():
        matches = pattern.findall(prompt)
        if matches:
            leaks[name] = matches
    return leaks


def assert_no_leakage(prompt: str) -> None:
    """Lanza ``ValueError`` si se detecta cualquier fuga en el prompt."""
    leaks = check_prompt_leakage(prompt)
    if leaks:
        detail = "; ".join(f"{k}: {v[:5]}" for k, v in leaks.items())
        raise ValueError(f"Fuga de información detectada en el prompt: {detail}")
