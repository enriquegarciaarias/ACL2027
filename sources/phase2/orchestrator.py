"""Orquestador factorial de llamadas LLM (Fase 2, Paso 4).

Ejecuta una ventana por llamada, con contexto independiente. Construye un
manifiesto de ejecución que cubre todas las combinaciones factorial:

    (ventana, condición_degradación, representación, variante_de_prompt,
     repetición, semilla)

Cada ejecución se identifica por un hash del prompt completo + parámetros del
modelo, que se usa como caché: si el hash coincide con una ejecución previa,
se reutiliza la respuesta sin llamar al LLM.

Las respuestas se guardan con procedencia completa en
results/output/{subject}/INGEST/llm_responses/.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import requests

from sources.common.common import logger, processControl, writeLog
from sources.common.paths import (
    atomic_write_json,
    atomic_write_jsonl,
    corpus_dir,
    ensure_dir,
    read_json,
    read_jsonl,
)
from sources.common.llm_client import check_model_available
from sources.common.provenance import utc_now
from sources.common.validation import ValidationError
from sources.ingestion import discover, features, loader, qc, represent, window
from sources.ingestion.config import get_ingestion_config
from sources.ingestion.represent import _opaque_window_id
from sources.phase2.development_split import (
    DEV_SUBJECTS,
    SPLIT_VERSION,
    contains_development_subjects,
    split_warning,
)
from sources.phase2.consistency import (
    ConsistencyReport,
    DEFAULT_MAX_HARD_FAILURES,
    DEFAULT_MAX_SOFT_FAILURES,
    check_consistency,
    should_abstain,
)
from sources.phase2.llm_config import (
    LlmExperimentConfig,
    build_experiment_config,
    create_ollama_chat_client,
    model_slug,
    supports_thinking,
)
from sources.phase2.negative_controls import (
    ALL_NEGATIVE_CONTROLS,
    NC_SHUFFLED,
    NC_ZERO,
    NC_NAN,
    NC_ABSENT,
    generate_negative_control_rep_row,
    is_negative_control,
    negative_control_type,
)
from sources.phase2.prompt_builder import (
    assert_no_leakage,
    build_numeric_prompt,
    build_text_prompt,
    estimate_tokens,
)
from sources.phase2.response_schema import (
    LlmResponse,
    make_timeout_response,
    validate_llm_output,
)

RESPONSES_DIR_NAME = "llm_responses"
RESPONSES_VERSION = "1.0"

PROMPT_VARIANT_BASE = "base"
PROMPT_VARIANT_PARAPHRASE = "paraphrase"
PROMPT_VARIANT_NO_ABSTAIN = "no_abstain"

REPRESENTATION_NUMERIC = "numeric"
REPRESENTATION_TEXT = "text"

CONDITION_CLEAN = "C0_clean"

#: Estado de la respuesta cuando la compuerta de consistencia fuerza la
#: abstención (el LLM había emitido una decisión concreta).
GATE_STATUS = "gate_abstention"


def _gate_values(numeric_row: Mapping[str, Any] | None) -> dict[str, float | None] | None:
    """Extrae los valores absolutos de una fila ``numeric_llm``.

    Devuelve ``None`` si la fila no tiene payload numérico (p. ej. una
    representación textual sin homólogo numérico).
    """
    if not isinstance(numeric_row, Mapping):
        return None
    payload = numeric_row.get("payload")
    if not isinstance(payload, Mapping):
        return None
    modalities = payload.get("modalities")
    if not isinstance(modalities, Mapping):
        return None
    return {
        name: meta.get("value") if isinstance(meta, Mapping) else meta
        for block in modalities.values()
        if isinstance(block, Mapping)
        for name, meta in block.items()
    }


def build_consistency_reports(
    numeric_rows: Mapping[str, Mapping[str, Any]],
    duration_by_window: Mapping[str, float | None],
    conditions: list[str],
    tolerance_scale: float = 1.0,
) -> dict[tuple[str, str], ConsistencyReport]:
    """Calcula el veredicto de consistencia por (ventana, condición).

    La compuerta se evalúa sobre los valores **absolutos** de la vista
    numérica, que son los mismos features que ve el LLM en ambas
    representaciones (el conjunto curado está unificado). Para las condiciones
    de control negativo se aplica antes la misma degradación que recibe el LLM,
    de modo que la compuerta juzgue exactamente la evidencia enviada.

    Args:
        numeric_rows: filas ``numeric_llm`` indexadas por ``window_id``.
        duration_by_window: duración de la ventana en segundos.
        conditions: condiciones a evaluar (``C0_clean`` y/o ``NC_*``).
        tolerance_scale: laxitud de las tolerancias (ver ``check_consistency``).

    Returns:
        dict ``(window_id, condition_id) -> ConsistencyReport``.
    """
    reports: dict[tuple[str, str], ConsistencyReport] = {}
    for window_id, row in numeric_rows.items():
        duration = duration_by_window.get(window_id) or 60.0
        for condition in conditions:
            condition_row = row
            if condition.startswith("NC_"):
                condition_row = generate_negative_control_rep_row(row, condition)
            values = _gate_values(condition_row)
            if values is None:
                continue
            reports[(window_id, condition)] = check_consistency(
                values,
                duration_s=duration,
                tolerance_scale=tolerance_scale,
            )
    return reports


def apply_consistency_gate(
    result: dict[str, Any],
    report: ConsistencyReport | None,
    max_hard_failures: int = DEFAULT_MAX_HARD_FAILURES,
    max_soft_failures: int = DEFAULT_MAX_SOFT_FAILURES,
) -> bool:
    """Aplica el veredicto de la compuerta a un registro de respuesta.

    El registro conserva la decisión original del modelo en ``decision_llm`` /
    ``confidence_llm``; si la compuerta se abstiene, ``decision`` pasa a
    ``abstain``, ``confidence`` a 0 y ``status`` a ``gate_abstention``.

    Args:
        result: registro de respuesta (se modifica in situ).
        report: informe de consistencia de la ventana, o ``None`` si no se pudo
            evaluar (no hay vista numérica).
        max_hard_failures: fallos ``hard`` tolerados.
        max_soft_failures: fallos ``soft`` tolerados.

    Returns:
        True si la compuerta forzó la abstención.
    """
    if report is None:
        result["consistency"] = {"evaluated": False}
        return False

    gate_abstains = should_abstain(
        report,
        max_hard_failures=max_hard_failures,
        max_soft_failures=max_soft_failures,
    )
    payload = report.to_dict()
    payload["evaluated"] = True
    payload["gate_abstained"] = gate_abstains
    result["consistency"] = payload

    result.setdefault("decision_llm", result.get("decision"))
    result.setdefault("confidence_llm", result.get("confidence"))

    if not gate_abstains:
        return False

    if result.get("status") == "valid_response" and result.get("decision") not in (
        None,
        "abstain",
    ):
        result["status"] = GATE_STATUS
        result["decision"] = "abstain"
        result["confidence"] = 0
        return True
    return False


def _prompt_hash(prompt: str, config: LlmExperimentConfig) -> str:
    """Calcula el hash SHA-256 del prompt + parámetros del modelo.

    Se usa como clave de caché: si dos ejecuciones generan el mismo prompt
    y los mismos parámetros, se reutiliza la respuesta.
    """
    payload = {
        "prompt": prompt,
        "model": config.model,
        "temperature": config.temperature,
        "num_ctx": config.num_ctx,
        "seed": config.seed,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "think": config.think,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _config_for_hash(config: LlmExperimentConfig, seed: int) -> LlmExperimentConfig:
    """Devuelve una copia de ``config`` con la semilla de la repetición.

    La semilla por repetición participa en el hash de caché (``_prompt_hash``)
    para que dos repeticiones con semillas distintas no colisionen en caché.
    """
    return LlmExperimentConfig(
        model=config.model,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        seed=seed,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        keep_alive=config.keep_alive,
        max_retries=config.max_retries,
        cascade_enabled=config.cascade_enabled,
        think=config.think,
    )


def responses_dir(model: str | None = None) -> Path:
    """Directorio de respuestas LLM, aislado por modelo.

    - Con ``model``: ``.../llm_responses/{model_slug}/`` (una carpeta por
      modelo, de modo que comparar modelos nunca sobrescribe resultados).
    - Sin ``model``: ``.../llm_responses/`` (ubicación histórica, se mantiene
      para leer ejecuciones anteriores).
    """
    base = corpus_dir() / RESPONSES_DIR_NAME
    if model is None:
        return base
    return base / model_slug(model)


def response_path(subject: str, model: str | None = None) -> Path:
    """Ruta del JSONL de respuestas de un sujeto (por modelo)."""
    return responses_dir(model) / f"{subject}.responses.jsonl"


def manifest_path(subject: str, model: str | None = None) -> Path:
    """Ruta del manifiesto de ejecución de un sujeto (por modelo)."""
    return responses_dir(model) / f"{subject}.manifest.json"


def subject_split(subject: str) -> str:
    """Etiqueta el sujeto como ``development`` o ``evaluation`` (split congelado)."""
    return "development" if subject in DEV_SUBJECTS else "evaluation"


def _build_manifest_entry(
    subject: str,
    window_id: str,
    window_id_opaque: str,
    condition_id: str,
    representation: str,
    prompt_variant: str,
    repetition: int,
    seed: int,
    prompt_hash: str,
) -> dict[str, Any]:
    """Crea una entrada del manifiesto de ejecución."""
    return {
        "subject": subject,
        "split": subject_split(subject),
        "window_id": window_id,
        "window_id_opaque": window_id_opaque,
        "condition_id": condition_id,
        "representation": representation,
        "prompt_variant": prompt_variant,
        "repetition": repetition,
        "seed": seed,
        "prompt_hash": prompt_hash,
    }


def _load_cached_response(cache_dir: Path, prompt_hash: str) -> dict[str, Any] | None:
    """Busca una respuesta en caché por hash de prompt."""
    cache_file = cache_dir / f"{prompt_hash}.json"
    if not cache_file.is_file():
        return None
    return read_json(cache_file)


def _cache_usable(cached: dict[str, Any] | None) -> bool:
    """¿Sirve la respuesta cacheada?

    Un ``timeout`` es un fallo de transporte, no una respuesta del modelo: si se
    reutilizara, un corte de Ollama quedaría fijado en la caché para siempre.
    """
    return bool(cached) and cached.get("status") != "timeout"


def _save_cached_response(cache_dir: Path, prompt_hash: str, data: dict[str, Any]) -> None:
    """Guarda una respuesta en caché."""
    ensure_dir(cache_dir)
    atomic_write_json(cache_dir / f"{prompt_hash}.json", data)


def _call_ollama_chat(
    client: dict[str, Any],
    prompt: str,
    config: LlmExperimentConfig,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Llama a Ollama /api/chat con reintentos y semilla por repetición.

    Args:
        client: configuración de cliente Ollama.
        prompt: prompt a enviar.
        config: configuración experimental resuelta.
        seed: semilla para esta llamada (por repetición). Si es None, se usa
            ``config.seed``.

    Returns:
        (respuesta_texto, metadatos_de_llamada)

    Raises:
        ValidationError: si Ollama no responde tras ``config.max_retries``
            reintentos, o si la respuesta no es JSON válido.
    """
    host = str(client.get("host", "http://localhost:11434")).rstrip("/")
    url = f"{host}/api/chat"

    # ``think`` solo se envía a familias que lo documentan (qwen3); en otros
    # modelos el campo sobra y versiones antiguas de Ollama podrían rechazarlo.
    think_requested = bool(client.get("think", config.think)) if supports_thinking(
        str(client.get("model", config.model))
    ) else None

    payload = {
        "model": client["model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.num_ctx,
            "seed": seed if seed is not None else config.seed,
            "top_p": config.top_p,
            "num_predict": config.max_tokens,
        },
        "keep_alive": client["keep_alive"],
    }
    if think_requested is not None:
        payload["think"] = think_requested

    max_retries = int(client.get("max_retries", config.max_retries))
    start = time.perf_counter()
    last_error: str | None = None
    last_exc: Exception | None = None
    data = None

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=config.timeout)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_error = f"Ollama no respondió: {exc}"
            last_exc = exc
        except (ValueError, KeyError) as exc:
            last_error = f"Respuesta de Ollama no es JSON válido: {exc}"
            last_exc = exc
        if attempt < max_retries:
            # Backoff progresivo (2s, 8s, 18s, ...) para dar tiempo a Ollama
            # de recuperarse de un pico de carga.
            time.sleep(2.0 * (attempt + 1) ** 2)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if data is None:
        raise ValidationError(
            last_error or "Ollama no respondió", where="orchestrator"
        ) from last_exc

    message = data.get("message", {})
    content = message.get("content", "") or ""
    thinking = message.get("thinking", "") or ""
    parts = []
    if thinking.strip():
        parts.append(thinking)
    if content.strip():
        parts.append(content)
    combined = "\n".join(parts) if parts else ""

    ollama_meta = {
        "model": data.get("model", client["model"]),
        "created_at": data.get("created_at", ""),
        "done": data.get("done", True),
        "total_duration_ms": data.get("total_duration", 0) / 1000000.0,
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ms": data.get("prompt_eval_duration", 0) / 1000000.0,
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ms": data.get("eval_duration", 0) / 1000000.0,
        "latency_ms": elapsed_ms,
        "has_thinking": bool(thinking.strip()),
        "content_empty": not bool(content.strip()),
        "think_requested": think_requested,
    }
    return combined, ollama_meta


def _build_prompt_for_variant(
    payload: Mapping[str, Any],
    text_representation: str | None,
    window_id_opaque: str,
    duration_s: float | None,
    num_ctx: int,
    variant: str,
) -> str:
    """Construye el prompt según la variante solicitada.

    Variantes:
    - base: instrucción estándar con opción de abstención.
    - paraphrase: paráfrasis de la instrucción base.
    - no_abstain: sin opción de abstención (forzada).
    """
    if variant == PROMPT_VARIANT_NO_ABSTAIN:
        base_prompt = build_numeric_prompt(
            payload=payload,
            window_id_opaque=window_id_opaque,
            duration_s=duration_s,
            num_ctx=num_ctx,
            use_few_shot=True,
            include_additional=True,
        )
        base_prompt = base_prompt.replace(
            'If the evidence is insufficient, set decision to "abstain".',
            "Always provide a decision. Do not abstain.",
        )
        return base_prompt

    if variant == PROMPT_VARIANT_PARAPHRASE:
        lines = [
            "You are an expert in physiological signal interpretation.",
            "",
            "Your task: assign an affective state label to this 60-second window using only the electrodermal activity (EDA) and heart rate (HR) features below.",
            "",
            f"Window ID (opaque): {window_id_opaque}",
        ]
        if duration_s is not None:
            lines.append(f"Window duration: {duration_s} s")
        lines.append("")
        lines.append("Provided features (EDA + HR only):")
        for modality, feature_block in payload.get("modalities", {}).items():
            lines.append(f"- {modality.upper()}:")
            for name, meta in feature_block.items():
                value = meta.get("value")
                unit = meta.get("unit")
                suffix = f" {unit}" if unit else ""
                lines.append(f"  * {name}: {value}{suffix}")
        lines.append("")
        lines.append("Output a JSON object with these exact keys:")
        lines.append("")
        lines.append("{")
        lines.append('  "decision": "baseline" | "stress" | "amusement" | "meditation" | "abstain",')
        lines.append('  "confidence": <integer 0-100>,')
        lines.append('  "claims": [{"metric": "...", "channel": "...", "direction": "...", "magnitude": "..."}],')
        lines.append('  "reasoning": "..."')
        lines.append("}")
        lines.append("")
        lines.append("Constraints:")
        lines.append("- Base your answer only on the features shown.")
        lines.append('- If uncertain, choose "abstain".')
        lines.append("- Confidence is an integer from 0 to 100.")
        lines.append("- Each claim must reference a provided feature.")
        lines.append("- Return valid JSON only, no markdown.")
        prompt = "\n".join(lines)
        estimated = estimate_tokens(prompt)
        if estimated > num_ctx:
            raise ValidationError(
                f"Prompt paraphrase excede num_ctx: {estimated} > {num_ctx}",
                where="orchestrator",
            )
        return prompt

    if text_representation is not None:
        return build_text_prompt(
            text_representation=text_representation,
            window_id_opaque=window_id_opaque,
            duration_s=duration_s,
            num_ctx=num_ctx,
            use_few_shot=True,
        )

    return build_numeric_prompt(
        payload=payload,
        window_id_opaque=window_id_opaque,
        duration_s=duration_s,
        num_ctx=num_ctx,
        use_few_shot=True,
        include_additional=True,
    )


def build_execution_manifest(
    subjects: list[str],
    representations: list[str] | None = None,
    prompt_variants: list[str] | None = None,
    repetitions: int = 1,
    seeds: list[int] | None = None,
    condition_ids: list[str] | None = None,
    negative_controls: list[str] | None = None,
    config: LlmExperimentConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Construye el manifiesto de ejecución factorial para una lista de sujetos.

    Args:
        subjects: lista de identificadores de sujeto.
        representations: tipos de representación (default: ["numeric_llm", "text_llm"]).
        prompt_variants: variantes de prompt (default: ["base"]).
        repetitions: número de repeticiones por combinación.
        seeds: lista de semillas (una por repetición).
        condition_ids: condiciones de degradación (default: ["C0_clean"]).
        negative_controls: tipos de control negativo a generar
            (default: None, sin controles). Valores: NC_shuffled, NC_zero,
            NC_nan, NC_absent.
        config: configuración experimental de LLM (se usa para calcular
            el hash de caché del prompt). Si no se proporciona, se usan
            valores por defecto.

    Returns:
        dict sujeto -> lista de entradas de manifiesto.
    """
    if representations is None:
        representations = [REPRESENTATION_NUMERIC, REPRESENTATION_TEXT]
    if prompt_variants is None:
        prompt_variants = [PROMPT_VARIANT_BASE]
    if seeds is None:
        seeds = [0] * repetitions
    if condition_ids is None:
        condition_ids = [CONDITION_CLEAN]
    if negative_controls is None:
        negative_controls = []
    if config is None:
        config = LlmExperimentConfig(
            model="qwen2.5:7b",
            temperature=0.3,
            num_ctx=32768,
            seed=42,
            top_p=0.9,
            max_tokens=800,
            timeout=120.0,
            keep_alive="5m",
            max_retries=3,
            cascade_enabled=False,
        )

    for nc in negative_controls:
        if nc not in ALL_NEGATIVE_CONTROLS:
            raise ValidationError(
                f"Control negativo no válido: {nc}. Valores permitidos: {ALL_NEGATIVE_CONTROLS}",
                where="orchestrator",
            )

    manifest: dict[str, list[dict[str, Any]]] = {}

    for subject in subjects:
        windows_path = window.windows_path(subject)
        window_rows = read_jsonl(windows_path)
        if not window_rows:
            raise ValidationError(
                f"no existen ventanas para {subject}; ejecute '--step window' primero",
                where="orchestrator",
            )

        reps: dict[str, list[dict[str, Any]]] = {}
        for rep in representations:
            path = represent.llm_representation_path(rep, subject)
            if not path.is_file():
                raise ValidationError(
                    f"no existe la representación '{rep}_llm' de {subject}; "
                    "ejecute '--step represent' primero",
                    where="orchestrator",
                )
            reps[rep] = read_jsonl(path)

        window_by_id: dict[str, dict[str, Any]] = {}
        for row in window_rows:
            window_by_id[row["window_id"]] = row

        entries: list[dict[str, Any]] = []

        for rep in representations:
            rep_rows = reps[rep]
            for rep_row in rep_rows:
                window_id = rep_row["window_id"]
                window_row = window_by_id.get(window_id)
                duration_s = window_row.get("duration_s")
                payload = rep_row.get("payload")

                for condition_id in condition_ids:
                    for variant in prompt_variants:
                        for rep_idx in range(repetitions):
                            seed = seeds[rep_idx] if rep_idx < len(seeds) else 0
                            text_rep = rep_row.get("payload") if rep == REPRESENTATION_TEXT else None

                            try:
                                prompt = _build_prompt_for_variant(
                                    payload=payload,
                                    text_representation=text_rep,
                                    window_id_opaque=_opaque_window_id(window_id),
                                    duration_s=duration_s,
                                    num_ctx=32768,
                                    variant=variant,
                                )
                                prompt_hash = _prompt_hash(
                                    prompt, _config_for_hash(config, seed)
                                )
                            except ValidationError:
                                prompt = ""
                                prompt_hash = "EXCLUDED_TOO_LONG"

                            entries.append(
                                _build_manifest_entry(
                                    subject=subject,
                                    window_id=window_id,
                                    window_id_opaque=_opaque_window_id(window_id),
                                    condition_id=condition_id,
                                    representation=f"{rep}_llm",
                                    prompt_variant=variant,
                                    repetition=rep_idx,
                                    seed=seed,
                                    prompt_hash=prompt_hash,
                                )
                            )

                for nc_type in negative_controls:
                    nc_row = generate_negative_control_rep_row(rep_row, nc_type)
                    nc_payload = nc_row.get("payload")
                    nc_text_rep = nc_payload if rep == REPRESENTATION_TEXT else None
                    nc_duration = window_row.get("duration_s")

                    for variant in prompt_variants:
                        for rep_idx in range(repetitions):
                            seed = seeds[rep_idx] if rep_idx < len(seeds) else 0

                            try:
                                prompt = _build_prompt_for_variant(
                                    payload=nc_payload,
                                    text_representation=nc_text_rep,
                                    window_id_opaque=_opaque_window_id(window_id),
                                    duration_s=nc_duration,
                                    num_ctx=32768,
                                    variant=variant,
                                )
                                prompt_hash = _prompt_hash(
                                    prompt, _config_for_hash(config, seed)
                                )
                            except ValidationError:
                                prompt = ""
                                prompt_hash = "EXCLUDED_TOO_LONG"

                            entries.append(
                                _build_manifest_entry(
                                    subject=subject,
                                    window_id=window_id,
                                    window_id_opaque=_opaque_window_id(window_id),
                                    condition_id=nc_type,
                                    representation=f"{rep}_llm",
                                    prompt_variant=variant,
                                    repetition=rep_idx,
                                    seed=seed,
                                    prompt_hash=prompt_hash,
                                )
                            )

        manifest[subject] = entries

    return manifest


def run_orchestrator(
    subjects: list[str],
    config: LlmExperimentConfig,
    *,
    representations: list[str] | None = None,
    prompt_variants: list[str] | None = None,
    repetitions: int = 1,
    seeds: list[int] | None = None,
    condition_ids: list[str] | None = None,
    negative_controls: list[str] | None = None,
    force: bool = False,
    consistency_gate: bool = True,
    gate_max_hard_failures: int = DEFAULT_MAX_HARD_FAILURES,
    gate_max_soft_failures: int = DEFAULT_MAX_SOFT_FAILURES,
    gate_tolerance_scale: float = 1.0,
) -> dict[str, Any]:
    """Ejecuta el orquestador factorial para una lista de sujetos.

    Args:
        subjects: lista de sujetos a procesar.
        config: configuración experimental de LLM.
        representations: tipos de representación a usar.
        prompt_variants: variantes de prompt.
        repetitions: repeticiones por combinación.
        seeds: semillas por repetición.
        condition_ids: condiciones de degradación.
        negative_controls: tipos de control negativo a generar
            (default: None, sin controles).
        force: si True, recalcula aunque exista en caché.
        consistency_gate: si True (default), aplica la compuerta determinista de
            consistencia fisiológica antes de aceptar la decisión del LLM. La
            compuerta evalúa las features absolutas de la ventana; para los
            controles negativos evalúa la versión degradada. La decisión
            original del modelo se conserva en ``decision_llm``.
        gate_max_hard_failures: fallos ``hard`` tolerados por la compuerta.
        gate_max_soft_failures: fallos ``soft`` tolerados por la compuerta.
        gate_tolerance_scale: laxitud de las tolerancias de la compuerta.

    Returns:
        dict con resumen de ejecución por sujeto.
    """
    if representations is None:
        representations = [REPRESENTATION_NUMERIC, REPRESENTATION_TEXT]
    if prompt_variants is None:
        prompt_variants = [PROMPT_VARIANT_BASE]
    if seeds is None:
        seeds = [config.seed] * repetitions
    if condition_ids is None:
        condition_ids = [CONDITION_CLEAN]
    if negative_controls is None:
        negative_controls = []

    client = create_ollama_chat_client(config)
    # Caché y respuestas aisladas por modelo: comparar modelos no comparte
    # entradas ni sobrescribe artefactos.
    cache_dir = responses_dir(config.model) / "cache"
    ensure_dir(cache_dir)

    summary: dict[str, Any] = {
        "schema_version": RESPONSES_VERSION,
        "config": config.to_dict(),
        "consistency_gate": {
            "enabled": consistency_gate,
            "max_hard_failures": gate_max_hard_failures,
            "max_soft_failures": gate_max_soft_failures,
            "tolerance_scale": gate_tolerance_scale,
        },
        "subjects": {},
        "total_calls": 0,
        "total_cached": 0,
        "total_errors": 0,
        "total_gate_abstentions": 0,
        "generated_at": utc_now(),
        "split_version": SPLIT_VERSION,
        "development_subjects": list(DEV_SUBJECTS),
    }

    for subject in subjects:
        manifest = build_execution_manifest(
            subjects=[subject],
            representations=representations,
            prompt_variants=prompt_variants,
            repetitions=repetitions,
            seeds=seeds,
            condition_ids=condition_ids,
            negative_controls=negative_controls,
            config=config,
        )
        entries = manifest.get(subject, [])

        subject_results: list[dict[str, Any]] = []
        subject_calls = 0
        subject_cached = 0
        subject_errors = 0
        subject_gate_abstentions = 0

        reps_data: dict[str, dict[str, dict[str, Any]]] = {}
        window_by_id: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(window.windows_path(subject)):
            window_by_id[row["window_id"]] = row
        for rep in representations:
            rep_rows = read_jsonl(represent.llm_representation_path(rep, subject))
            for rep_row in rep_rows:
                reps_data.setdefault(f"{rep}_llm", {})[rep_row["window_id"]] = rep_row

        # Veredictos de consistencia: se calculan una vez por (ventana,
        # condición) sobre la vista numérica (mismas features que el LLM).
        gate_reports: dict[tuple[str, str], ConsistencyReport] = {}
        if consistency_gate:
            numeric_path = represent.llm_representation_path(
                REPRESENTATION_NUMERIC, subject
            )
            numeric_rows: dict[str, dict[str, Any]] = {}
            if numeric_path.is_file():
                for rep_row in read_jsonl(numeric_path):
                    numeric_rows[rep_row["window_id"]] = rep_row
            all_conditions = list(dict.fromkeys(list(condition_ids) + list(negative_controls)))
            gate_reports = build_consistency_reports(
                numeric_rows=numeric_rows,
                duration_by_window={
                    wid: row.get("duration_s") for wid, row in window_by_id.items()
                },
                conditions=all_conditions,
                tolerance_scale=gate_tolerance_scale,
            )
            if not numeric_rows:
                writeLog(
                    "warning",
                    logger,
                    f"[PHASE2] {subject}: sin vista numérica; la compuerta de "
                    "consistencia no se aplica",
                )

        def _apply_gate(result: dict[str, Any], window_id: str) -> None:
            """Aplica la compuerta a un registro y actualiza el contador."""
            nonlocal subject_gate_abstentions
            report = gate_reports.get((window_id, result.get("condition_id")))
            if apply_consistency_gate(
                result,
                report,
                max_hard_failures=gate_max_hard_failures,
                max_soft_failures=gate_max_soft_failures,
            ):
                subject_gate_abstentions += 1

        for entry in entries:
            window_id = entry["window_id"]
            window_id_opaque = entry["window_id_opaque"]
            condition_id = entry["condition_id"]
            representation = entry["representation"]
            variant = entry["prompt_variant"]
            repetition = entry["repetition"]
            seed = entry["seed"]
            prompt_hash = entry["prompt_hash"]

            if not force and prompt_hash != "EXCLUDED_TOO_LONG":
                cached = _load_cached_response(cache_dir, prompt_hash)
                if not _cache_usable(cached):
                    cached = None
                if cached is not None:
                    cached = dict(cached)
                    cached["subject"] = subject
                    cached["window_id_opaque"] = window_id_opaque
                    cached["condition_id"] = condition_id
                    cached["representation"] = representation
                    cached["prompt_variant"] = variant
                    cached["repetition"] = repetition
                    cached["seed"] = seed
                    _apply_gate(cached, window_id)
                    subject_results.append(cached)
                    subject_cached += 1
                    continue

            if prompt_hash == "EXCLUDED_TOO_LONG":
                result = {
                    "subject": subject,
                    "model": config.model,
                    "window_id_opaque": window_id_opaque,
                    "condition_id": condition_id,
                    "representation": representation,
                    "prompt_variant": variant,
                    "repetition": repetition,
                    "seed": seed,
                    "prompt_hash": prompt_hash,
                    "status": "excluded_too_long",
                    "error_message": "Prompt excede num_ctx",
                    "raw_output": "",
                    "generated_at": utc_now(),
                }
                _apply_gate(result, window_id)
                subject_results.append(result)
                subject_errors += 1
                continue

            rep_row = reps_data.get(representation, {}).get(window_id)
            payload = rep_row.get("payload")
            window_row = window_by_id.get(window_id)
            duration_s = window_row.get("duration_s")

            if condition_id.startswith("NC_"):
                nc_type = condition_id
                from sources.phase2.negative_controls import generate_negative_control_rep_row
                nc_row = generate_negative_control_rep_row(rep_row, nc_type)
                payload = nc_row.get("payload")
            text_rep = payload if isinstance(payload, str) else None

            try:
                prompt = _build_prompt_for_variant(
                    payload=payload,
                    text_representation=text_rep,
                    window_id_opaque=window_id_opaque,
                    duration_s=duration_s,
                    num_ctx=config.num_ctx,
                    variant=variant,
                )
            except ValidationError as exc:
                result = {
                    "subject": subject,
                    "model": config.model,
                    "window_id_opaque": window_id_opaque,
                    "condition_id": condition_id,
                    "representation": representation,
                    "prompt_variant": variant,
                    "repetition": repetition,
                    "seed": seed,
                    "prompt_hash": _prompt_hash("", _config_for_hash(config, seed)),
                    "status": "prompt_error",
                    "error_message": str(exc),
                    "raw_output": "",
                    "generated_at": utc_now(),
                }
                _apply_gate(result, window_id)
                subject_results.append(result)
                subject_errors += 1
                continue

            try:
                assert_no_leakage(prompt)
            except ValueError as exc:
                result = {
                    "subject": subject,
                    "model": config.model,
                    "window_id_opaque": window_id_opaque,
                    "condition_id": condition_id,
                    "representation": representation,
                    "prompt_variant": variant,
                    "repetition": repetition,
                    "seed": seed,
                    "prompt_hash": _prompt_hash(prompt, _config_for_hash(config, seed)),
                    "status": "leakage_detected",
                    "error_message": str(exc),
                    "raw_output": "",
                    "generated_at": utc_now(),
                }
                _apply_gate(result, window_id)
                subject_results.append(result)
                subject_errors += 1
                continue

            try:
                raw_output, ollama_meta = _call_ollama_chat(client, prompt, config, seed=seed)
            except ValidationError as exc:
                result = {
                    "subject": subject,
                    "model": config.model,
                    "window_id_opaque": window_id_opaque,
                    "condition_id": condition_id,
                    "representation": representation,
                    "prompt_variant": variant,
                    "repetition": repetition,
                    "seed": seed,
                    "prompt_hash": _prompt_hash(prompt, _config_for_hash(config, seed)),
                    "status": "timeout",
                    "error_message": str(exc),
                    "raw_output": "",
                    "generated_at": utc_now(),
                    "ollama_meta": {},
                }
                _apply_gate(result, window_id)
                subject_results.append(result)
                subject_errors += 1
                continue

            validated = validate_llm_output(raw_output)

            result = {
                "subject": subject,
                "split": subject_split(subject),
                "model": config.model,
                "window_id_opaque": window_id_opaque,
                "condition_id": condition_id,
                "representation": representation,
                "prompt_variant": variant,
                "repetition": repetition,
                "seed": seed,
                "prompt_hash": _prompt_hash(prompt, _config_for_hash(config, seed)),
                "status": validated.status,
                "decision": validated.decision,
                "confidence": validated.confidence,
                "claims": [c.to_dict() for c in validated.claims],
                "reasoning": validated.reasoning,
                "error_message": validated.error_message,
                "raw_output": raw_output,
                "prompt": prompt,
                "ollama_meta": ollama_meta,
                "generated_at": utc_now(),
            }

            _save_cached_response(
                cache_dir, _prompt_hash(prompt, _config_for_hash(config, seed)), result
            )
            _apply_gate(result, window_id)
            subject_results.append(result)
            subject_calls += 1

            writeLog(
                "info",
                logger,
                f"[PHASE2] {subject}: llamada {subject_calls}/{len(entries)} "
                f"({entry['condition_id']}, {entry['representation']}, rep={entry['repetition']}) "
                f"→ {result['status']}, decision={result.get('decision', '—')}"
                + (
                    f" (LLM: {result['decision_llm']})"
                    if result.get("status") == GATE_STATUS
                    else ""
                ),
            )

        # Etiqueta de split congelado en TODOS los registros (incluidos caché y errores).
        for record in subject_results:
            record.setdefault("split", subject_split(subject))

        responses_path = response_path(subject, config.model)
        atomic_write_jsonl(responses_path, subject_results)

        manifest_path_subject = manifest_path(subject, config.model)
        atomic_write_json(
            manifest_path_subject,
            {
                "subject": subject,
                "n_entries": len(entries),
                "n_calls": subject_calls,
                "n_cached": subject_cached,
                "n_errors": subject_errors,
                "n_gate_abstentions": subject_gate_abstentions,
                "consistency_gate": {
                    "enabled": consistency_gate,
                    "max_hard_failures": gate_max_hard_failures,
                    "max_soft_failures": gate_max_soft_failures,
                    "tolerance_scale": gate_tolerance_scale,
                },
                "config": config.to_dict(),
                "entries": entries,
            },
        )

        summary["subjects"][subject] = {
            "n_entries": len(entries),
            "n_calls": subject_calls,
            "n_cached": subject_cached,
            "n_errors": subject_errors,
            "n_gate_abstentions": subject_gate_abstentions,
        }
        summary["total_calls"] += subject_calls
        summary["total_cached"] += subject_cached
        summary["total_errors"] += subject_errors
        summary["total_gate_abstentions"] += subject_gate_abstentions

    return summary


def run_phase2() -> None:
    """Punto de entrada para la Fase 2 desde ``main.py --proc PHASE2``.

    Lee los sujetos de ``processControl.args.subject`` o
    ``processControl.args.subjects``, habilita las vistas LLM (numeric_llm y
    text_llm), ejecuta ``represent`` para los sujetos que las necesiten y
    luego ejecuta el orquestador factorial de llamadas LLM para cada sujeto.

    Uso desde CLI:

        python main.py --proc PHASE2 --subject WESAD --subjects S3
        python main.py --proc PHASE2 --subject WESAD --subjects S3,S4,S5
        python main.py --proc PHASE2 --subject WESAD   # todos los sujetos
    """
    args = processControl.args
    if args is None:
        raise ValidationError(
            "processControl.args no inicializado. Ejecute main.py con --proc PHASE2.",
            where="orchestrator",
        )

    subjects_arg = getattr(args, "subjects", None) or getattr(args, "subject", None)
    if not subjects_arg:
        raise ValidationError(
            "Debe especificar --subjects o --subject para --proc PHASE2",
            where="orchestrator",
        )

    subjects = [s.strip() for s in str(subjects_arg).split(",") if s.strip()]
    force = bool(getattr(args, "force", False))

    # Split congelado dev/test: avisar (no bloquear) si hay sujetos de desarrollo.
    warning = split_warning(subjects)
    if warning and not getattr(args, "allowDev", False):
        writeLog("warning", logger, f"[PHASE2] {warning}")

    cfg = copy.deepcopy(get_ingestion_config())
    for kind in ("numeric", "text"):
        llm_opts = cfg["representations"][kind].setdefault("llm_view", {})
        llm_opts["enabled"] = True
        llm_opts["exclude_motion_channels"] = True
        llm_opts["opaque_window_id"] = True
    # Fase 2 solo consume las vistas numérica y textual (las únicas que se
    # envían al LLM). Los embeddings son material complementario y no se usan
    # aquí: se evita recalcularlos (requiere Ollama y es costoso).
    requested_kinds = list(cfg["representations"].get("enabled", []))
    cfg["representations"]["enabled"] = [
        kind for kind in requested_kinds if kind in ("numeric", "text")
    ]
    if len(cfg["representations"]["enabled"]) != len(requested_kinds):
        writeLog(
            "info",
            logger,
            f"[PHASE2] representaciones omitidas (no usadas en Fase 2): "
            f"{[k for k in requested_kinds if k not in ('numeric', 'text')]}",
        )

    llm_defaults = (processControl.defaults or {}).get("llm", {}) or {}

    # Selección de modelo: ``--model`` en CLI > config.json > referencia.
    # Comparar modelos es una decisión experimental explícita; cada modelo
    # escribe en su propio subdirectorio (``llm_responses/{modelo}/``).
    requested_model = getattr(args, "model", None) or llm_defaults.get(
        "primary_model"
    )
    think_override = getattr(args, "think", False) or None
    config = build_experiment_config(
        model=str(requested_model) if requested_model else None,
        temperature=float(llm_defaults.get("temperature", 0.3)),
        seed=42,
        max_retries=int(llm_defaults.get("max_retries", 3)),
        think=think_override,
        cascade_enabled=False,
    )
    writeLog(
        "info",
        logger,
        f"[PHASE2] modelo={config.model} (slug={config.model_slug}) "
        f"think={config.think} max_tokens={config.max_tokens} "
        f"num_ctx={config.num_ctx}",
    )
    if check_model_available(config.model) is False:
        writeLog(
            "warning",
            logger,
            f"[PHASE2] el modelo {config.model} no está instalado en Ollama; "
            f"descárgalo con 'ollama pull {config.model}' antes de continuar.",
        )

    # Repeticiones y semillas (configurables en config.json > defaults.llm).
    # Con más de una repetición se usan semillas distintas para no colapsar
    # en la misma respuesta (temperatura > 0 pero semilla fija por repetición).
    repetitions = int(llm_defaults.get("repetitions", 1))
    seeds = [int(s) for s in llm_defaults.get("seeds", [])]
    if not seeds:
        seeds = [config.seed]
    if len(seeds) < repetitions:
        seeds = seeds + [config.seed] * (repetitions - len(seeds))
    seeds = seeds[:repetitions]

    # Compuerta determinista de consistencia fisiológica (P1). Activa por
    # defecto; se puede desactivar con --noGate o con
    # config.json > defaults.llm.consistency_gate = false (ablación).
    consistency_gate = bool(llm_defaults.get("consistency_gate", True))
    if getattr(args, "noGate", False):
        consistency_gate = False
    gate_tolerance_scale = float(llm_defaults.get("consistency_tolerance_scale", 1.0))

    representations = [REPRESENTATION_NUMERIC, REPRESENTATION_TEXT]
    prompt_variants = [PROMPT_VARIANT_BASE]
    condition_ids = [CONDITION_CLEAN]
    negative_controls = [NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT]

    writeLog(
        "info",
        logger,
        f"[PHASE2] Sujetos: {subjects}, representaciones: {representations}, "
        f"variantes: {prompt_variants}, condiciones: {condition_ids}, "
        f"controles negativos: {negative_controls}, "
        f"repeticiones: {repetitions}, semillas: {seeds}, "
        f"compuerta de consistencia: {'on' if consistency_gate else 'off'} "
        f"(tolerance_scale={gate_tolerance_scale})",
    )

    total_calls = 0
    total_errors = 0
    total_gate_abstentions = 0

    for subject in subjects:
        writeLog("info", logger, f"[PHASE2] Procesando sujeto: {subject}")

        manifest = discover.run_discover(cfg)
        entries = discover.selected_entries(manifest)
        entry = next((e for e in entries if e.subject == subject), None)
        if entry is None:
            raise ValidationError(
                f"no existe el sujeto '{subject}' en el manifiesto de descubrimiento",
                where="orchestrator",
            )

        loader.run_ingest(cfg, entry)
        qc.run_qc(cfg, subject)
        window.run_window(cfg, subject)
        features.run_features(cfg, subject)

        rep_result = represent.run_represent(cfg, subject, force=force)
        writeLog(
            "info",
            logger,
            f"[PHASE2] {subject}: representaciones {rep_result['status']} "
            f"({rep_result['n_windows']} ventanas)",
        )

        summary = run_orchestrator(
            subjects=[subject],
            config=config,
            representations=representations,
            prompt_variants=prompt_variants,
            repetitions=repetitions,
            seeds=seeds,
            condition_ids=condition_ids,
            negative_controls=negative_controls,
            force=force,
            consistency_gate=consistency_gate,
            gate_tolerance_scale=gate_tolerance_scale,
        )

        subj_summary = summary["subjects"].get(subject, {})
        total_calls += subj_summary.get("n_calls", 0)
        total_errors += subj_summary.get("n_errors", 0)
        total_gate_abstentions += subj_summary.get("n_gate_abstentions", 0)

        writeLog(
            "info",
            logger,
            f"[PHASE2] {subject}: llamadas={subj_summary.get('n_calls', 0)}, "
            f"caché={subj_summary.get('n_cached', 0)}, "
            f"errores={subj_summary.get('n_errors', 0)}, "
            f"abstenciones por compuerta={subj_summary.get('n_gate_abstentions', 0)}",
        )

    writeLog(
        "info",
        logger,
        f"[PHASE2] Completado. Total llamadas: {total_calls}, total errores: {total_errors}, "
        f"abstenciones por compuerta: {total_gate_abstentions}",
    )
    print(f"\nFase 2 completada para sujetos: {subjects}")
    print(f"  Llamadas LLM totales: {total_calls}")
    print(f"  Errores totales: {total_errors}")
    print(f"  Abstenciones forzadas por la compuerta de consistencia: {total_gate_abstentions}")
    print(f"  Modelo: {config.model}")
    print(f"  Respuestas en: {responses_dir(config.model)}/")
