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

# Conjunto mínimo de canales experimental: EDA + frecuencia cardíaca.
# Features de chest_EDA y chest_ECG que se envían al LLM.
EDA_HR_FEATURES: tuple[str, ...] = (
    # EDA (chest_EDA)
    "eda_mean_us",
    "eda_std_us",
    "eda_slope_us_per_s",
    "eda_tonic_mean_us",
    "eda_phasic_std_us",
    "eda_scr_candidates",
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
    # Pulsómetro (chest_BVP)
    "bvp_n_beats",
    "bvp_amplitude_std",
    "bvp_hr_bpm",
    "bvp_rr_mean_ms",
    "bvp_rmssd_ms",
)

# Modalidades de pecho permitidas (excluye muñeca y otras).
CHEST_MODALITIES: frozenset[str] = frozenset({"ecg", "eda", "resp", "temp", "bvp"})

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
# Un ejemplo por clase afectiva para equilibrar las prioridades del modelo.
FEW_SHOT_EXAMPLES: str = """Examples:

Example 1 - Baseline:
Features: ecg_hr_bpm=65.0 bpm, ecg_rmssd_ms=45.0 ms, eda_mean_us=2.1 uS, eda_scr_count=2 count
Decision: baseline
Reasoning: Heart rate and EDA are within resting ranges, consistent with baseline.

Example 2 - Stress:
Features: ecg_hr_bpm=95.0 bpm, ecg_rmssd_ms=28.0 ms, eda_mean_us=5.2 uS, eda_scr_count=8 count
Decision: stress
Reasoning: Elevated heart rate and high EDA SCR rate indicate stress.

Example 3 - Amusement:
Features: ecg_hr_bpm=85.0 bpm, ecg_rmssd_ms=35.0 ms, eda_mean_us=3.5 uS, eda_scr_count=5 count
Decision: amusement
Reasoning: Moderately elevated heart rate with positive arousal pattern.

Example 4 - Meditation:
Features: ecg_hr_bpm=58.0 bpm, ecg_rmssd_ms=55.0 ms, eda_mean_us=1.8 uS, eda_scr_count=1 count
Decision: meditation
Reasoning: Low heart rate with high HRV and minimal EDA activity indicate meditation."""


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

    Returns:
        string del prompt listo para enviar al LLM.

    Raises:
        ValueError: si el prompt estimado excede ``num_ctx`` (exclusión
            registrada, no se trunca).
    """
    modalities = filter_eda_hr_modalities(
        payload.get("modalities", {}), include_additional=include_additional
    )

    lines: list[str] = []
    lines.append("You are a physiological signal analysis assistant.")
    lines.append("")
    lines.append("Task: Classify the affective state of a 60-second window "
                 "based on electrodermal activity (EDA) and heart rate (HR) features.")
    lines.append("")

    if use_few_shot:
        lines.append(FEW_SHOT_EXAMPLES)
        lines.append("")
        lines.append("Now classify this window based only on the features below:")
        lines.append("")

    lines.append("Input features (EDA + HR only):")
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

    lines.append("Respond ONLY with a JSON object containing exactly these keys:")
    lines.append("")
    lines.append("{")
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
) -> str:
    """Construye un prompt en inglés a partir de una representación ``text_llm``.

    Args:
        text_representation: texto estructurado de la representación ``text_llm``.
        window_id_opaque: identificador opaco de la ventana.
        duration_s: duración de la ventana en segundos (opcional).
        num_ctx: tamaño máximo de contexto del modelo en tokens.
        use_few_shot: si True, incluye ejemplos few-shot (uno por clase
            afectiva) para mitigar el sesgo del modelo hacia "stress".

    Returns:
        string del prompt listo para enviar al LLM.

    Raises:
        ValueError: si el prompt estimado excede ``num_ctx``.
    """
    lines: list[str] = []
    lines.append("You are a physiological signal analysis assistant.")
    lines.append("")
    lines.append("Task: Classify the affective state of a 60-second window "
                 "based on the physiological signal summary below.")
    lines.append("")

    if use_few_shot:
        lines.append(FEW_SHOT_EXAMPLES)
        lines.append("")
        lines.append("Now classify this window based only on the features below:")
        lines.append("")

    lines.append("Window ID (opaque): " + window_id_opaque)
    if duration_s is not None:
        lines.append(f"Window duration: {duration_s} s")
    lines.append("")
    lines.append("Signal summary:")
    lines.append(text_representation)
    lines.append("")
    lines.append("Respond ONLY with a JSON object containing exactly these keys:")
    lines.append("")
    lines.append("{")
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
