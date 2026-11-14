"""Pruebas del orquestador factorial de llamadas LLM (Fase 2, Paso 4)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sources.common.common import processControl
from sources.common.paths import atomic_write_jsonl
from sources.ingestion import represent, window
from sources.phase2.consistency import should_abstain
from sources.phase2.llm_config import LlmExperimentConfig
from sources.phase2.negative_controls import (
    ALL_NEGATIVE_CONTROLS,
    NC_ABSENT,
    NC_NAN,
    NC_SHUFFLED,
    NC_ZERO,
    generate_negative_control_rep_row,
    is_negative_control,
    negative_control_type,
)
from sources.phase2.orchestrator import (
    CONDITION_CLEAN,
    PROMPT_VARIANT_BASE,
    PROMPT_VARIANT_NO_ABSTAIN,
    PROMPT_VARIANT_PARAPHRASE,
    REPRESENTATION_NUMERIC,
    REPRESENTATION_TEXT,
    _build_manifest_entry,
    _call_ollama_chat,
    _config_for_hash,
    _prompt_hash,
    build_consistency_reports,
    build_execution_manifest,
    run_orchestrator,
)


class _MockResponse:
    """Mock de respuesta de requests.post."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json


def _setup_env(monkeypatch, tmp_path):
    """Configura processControl.env y args para pruebas."""
    results = tmp_path / "results"
    input_dir = results / "input"
    output_dir = results / "output"
    dataset_dir = input_dir / "WESAD"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        processControl,
        "env",
        {"input": str(input_dir), "output": str(output_dir)},
    )
    monkeypatch.setattr(
        processControl,
        "args",
        SimpleNamespace(
            subject="WESAD",
            proc="INGEST",
            step=None,
            serverGPU=None,
            subjects=None,
            force=False,
            dry_run=False,
        ),
    )
    return dataset_dir, output_dir


def _write_windows(subject_dir, subject, windows):
    """Escribe un archivo de ventanas sintético en la ruta del corpus."""
    path = window.windows_path(subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, windows)
    return path


def _write_llm_representation(subject_dir, subject, kind, rows):
    """Escribe una representación LLM sintética en la ruta del corpus."""
    path = represent.llm_representation_path(kind, subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(path, rows)
    return path


def _make_numeric_rep_row(window_id, features=None):
    """Crea una fila de representación numérica LLM."""
    if features is None:
        features = {
            "ecg": {
                "ecg_hr_bpm": {"value": 72.0, "unit": "bpm"},
                "ecg_rr_mean_ms": {"value": 827.0, "unit": "ms"},
            },
            "eda": {
                "eda_mean_us": {"value": 12.4, "unit": "uS"},
                "eda_std_us": {"value": 3.1, "unit": "uS"},
            },
        }
    return {
        "window_id": window_id,
        "subject": "S2",
        "representation": "numeric_llm",
        "payload": {"modalities": features, "feature_version": "1.0"},
    }


def _make_text_rep_row(window_id, text="EDA mean: 12.4 uS. HR: 72 bpm."):
    """Crea una fila de representación textual LLM."""
    return {
        "window_id": window_id,
        "subject": "S2",
        "representation": "text_llm",
        "payload": text,
    }


def _make_window_row(window_id, duration_s=60.0):
    """Crea una fila de ventana sintética."""
    return {
        "window_id": window_id,
        "subject": "S2",
        "label": {"id": 1, "name": "baseline"},
        "t_start_s": 0.0,
        "t_end_s": duration_s,
        "duration_s": duration_s,
        "ranges": {},
        "purity": 1.0,
        "guard_applied_s": 5.0,
        "source": {},
        "coverage": {},
    }


def test_prompt_hash_deterministic():
    """El hash de prompt es determinista para la misma entrada."""
    config = LlmExperimentConfig(
        model="qwen3:8b",
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
    h1 = _prompt_hash("prompt de prueba", config)
    h2 = _prompt_hash("prompt de prueba", config)
    assert h1 == h2
    assert len(h1) == 16


def test_prompt_hash_changes_with_params():
    """El hash cambia si cambian los parámetros del modelo."""
    base = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=42,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )
    different_seed = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=99,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )
    assert _prompt_hash("misma prompt", base) != _prompt_hash("misma prompt", different_seed)


def test_build_manifest_entry_structure():
    """La entrada de manifiesto tiene todos los campos requeridos."""
    entry = _build_manifest_entry(
        subject="S2",
        window_id="w00001",
        window_id_opaque="win_abc123",
        condition_id="C0_clean",
        representation="numeric_llm",
        prompt_variant="base",
        repetition=0,
        seed=42,
        prompt_hash="a1b2c3d4",
    )
    assert entry["subject"] == "S2"
    assert entry["window_id"] == "w00001"
    assert entry["window_id_opaque"] == "win_abc123"
    assert entry["condition_id"] == "C0_clean"
    assert entry["representation"] == "numeric_llm"
    assert entry["prompt_variant"] == "base"
    assert entry["repetition"] == 0
    assert entry["seed"] == 42
    assert entry["prompt_hash"] == "a1b2c3d4"


def test_build_execution_manifest_basic(tmp_path, monkeypatch):
    """El manifiesto se construye con las combinaciones esperadas."""
    dataset_dir, _ = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001"), _make_window_row("w002")]
    _write_windows(subject_dir, "S2", windows)

    rep_rows = [
        _make_numeric_rep_row("w001"),
        _make_numeric_rep_row("w002"),
    ]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    manifest = build_execution_manifest(
        subjects=["S2"],
        representations=["numeric"],
        prompt_variants=["base"],
        repetitions=1,
        seeds=[42],
        condition_ids=["C0_clean"],
    )

    assert "S2" in manifest
    entries = manifest["S2"]
    # 2 ventanas × 1 representación × 1 variante × 1 repetición × 1 condición = 2
    assert len(entries) == 2

    for entry in entries:
        assert entry["subject"] == "S2"
        assert entry["condition_id"] == "C0_clean"
        assert entry["representation"] == "numeric_llm"
        assert entry["prompt_variant"] == "base"
        assert entry["repetition"] == 0
        assert entry["seed"] == 42
        assert entry["prompt_hash"] != "EXCLUDED_TOO_LONG"


def test_build_execution_manifest_repetitions(tmp_path, monkeypatch):
    """Con múltiples repeticiones, se generan múltiples entradas por ventana."""
    dataset_dir, _ = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    manifest = build_execution_manifest(
        subjects=["S2"],
        representations=["numeric"],
        prompt_variants=["base"],
        repetitions=3,
        seeds=[42, 123, 999],
        condition_ids=["C0_clean"],
    )

    entries = manifest["S2"]
    assert len(entries) == 3
    seeds_used = {e["seed"] for e in entries}
    assert seeds_used == {42, 123, 999}


def test_build_execution_manifest_missing_windows(tmp_path, monkeypatch):
    """Falta archivo de ventanas -> ValidationError."""
    dataset_dir, _ = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    subject_dir.mkdir(parents=True, exist_ok=True)
    # No escribir ventanas

    with pytest.raises(Exception):
        build_execution_manifest(
            subjects=["S2"],
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
        )


def test_build_execution_manifest_missing_representation(tmp_path, monkeypatch):
    """Falta representación LLM -> ValidationError."""
    dataset_dir, _ = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    # No escribir representación

    with pytest.raises(Exception):
        build_execution_manifest(
            subjects=["S2"],
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
        )


def test_run_orchestrator_mocked_ollama(tmp_path, monkeypatch):
    """Ejecución completa con Ollama mockeado."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    config = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=42,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )

    fake_response = {
        "model": "qwen3:8b",
        "message": {
            "content": (
                '{"decision": "stress", "confidence": 85, '
                '"claims": [{"metric": "eda_mean_us", "channel": "eda", '
                '"direction": "increase", "magnitude": "high"}], '
                '"reasoning": "EDA elevated."}'
            )
        },
        "done": True,
        "total_duration": 500000000,
        "prompt_eval_count": 100,
        "prompt_eval_duration": 200000000,
        "eval_count": 50,
        "eval_duration": 100000000,
    }

    with patch("requests.post") as mock_post:
        mock_post.return_value = _MockResponse(fake_response)

        summary = run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
        )

    assert "S2" in summary["subjects"]
    subj = summary["subjects"]["S2"]
    assert subj["n_entries"] == 1
    assert subj["n_calls"] == 1
    assert subj["n_cached"] == 0
    assert subj["n_errors"] == 0
    assert summary["total_calls"] == 1
    assert summary["total_errors"] == 0

    # Verificar que se escribió el archivo de respuestas (aislado por modelo).
    responses_path = (
        output_dir / "WESAD" / "INGEST" / "llm_responses"
        / "qwen3_8b" / "S2.responses.jsonl"
    )
    assert responses_path.is_file()
    record = json.loads(responses_path.read_text().splitlines()[0])
    assert record["model"] == "qwen3:8b"

    # Verificar que se escribió el manifiesto.
    manifest_path = (
        output_dir / "WESAD" / "INGEST" / "llm_responses"
        / "qwen3_8b" / "S2.manifest.json"
    )
    assert manifest_path.is_file()


def test_run_orchestrator_cache_reuse(tmp_path, monkeypatch):
    """La segunda ejecución reutiliza la caché (sin llamadas a Ollama)."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    config = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=42,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )

    call_count = 0

    def fake_post(url, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _MockResponse({
            "model": "qwen3:8b",
            "message": {
                "content": (
                    '{"decision": "baseline", "confidence": 90, '
                    '"claims": [], "reasoning": "ok."}'
                )
            },
            "done": True,
            "total_duration": 100000000,
            "prompt_eval_count": 50,
            "prompt_eval_duration": 10000000,
            "eval_count": 20,
            "eval_duration": 5000000,
        })

    with patch("requests.post", side_effect=fake_post):
        # Primera ejecución: 1 llamada.
        summary1 = run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
        )
        assert summary1["subjects"]["S2"]["n_calls"] == 1
        assert call_count == 1

        # Segunda ejecución (sin force): debe reusar caché, 0 llamadas nuevas.
        summary2 = run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
        )
        assert summary2["subjects"]["S2"]["n_cached"] == 1
        assert summary2["subjects"]["S2"]["n_calls"] == 0
        assert call_count == 1  # No se llamó a Ollama de nuevo


def test_run_orchestrator_force_recalculates(tmp_path, monkeypatch):
    """Con force=True, se recalcula aunque exista caché."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    config = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=42,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )

    call_count = 0

    def fake_post(url, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _MockResponse({
            "model": "qwen3:8b",
            "message": {
                "content": (
                    '{"decision": "stress", "confidence": 80, '
                    '"claims": [], "reasoning": "x."}'
                )
            },
            "done": True,
        })

    with patch("requests.post", side_effect=fake_post):
        run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
        )
        assert call_count == 1

        # Con force=True, se vuelve a llamar.
        run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            force=True,
        )
        assert call_count == 2


def test_generate_negative_control_shuffled():
    """El control shuffled baraja los valores de las features."""
    rep_row = _make_numeric_rep_row("w001")
    nc_row = generate_negative_control_rep_row(rep_row, NC_SHUFFLED)

    assert is_negative_control(nc_row)
    assert negative_control_type(nc_row) == NC_SHUFFLED
    assert nc_row["_original_window_id"] == "w001"

    # Los valores deben ser los mismos pero reordenados.
    original_values = {
        name: meta["value"]
        for modality in rep_row["payload"]["modalities"].values()
        for name, meta in modality.items()
    }
    nc_values = {
        name: meta["value"]
        for modality in nc_row["payload"]["modalities"].values()
        for name, meta in modality.items()
    }
    assert sorted(original_values.values()) == sorted(nc_values.values())


def test_generate_negative_control_zero():
    """El control zero pone todos los valores a cero."""
    rep_row = _make_numeric_rep_row("w001")
    nc_row = generate_negative_control_rep_row(rep_row, NC_ZERO)

    assert is_negative_control(nc_row)
    assert negative_control_type(nc_row) == NC_ZERO

    for modality in nc_row["payload"]["modalities"].values():
        for meta in modality.values():
            assert meta["value"] == 0.0


def test_generate_negative_control_nan():
    """El control nan pone todos los valores a None."""
    rep_row = _make_numeric_rep_row("w001")
    nc_row = generate_negative_control_rep_row(rep_row, NC_NAN)

    assert is_negative_control(nc_row)
    assert negative_control_type(nc_row) == NC_NAN

    for modality in nc_row["payload"]["modalities"].values():
        for meta in modality.values():
            assert meta["value"] is None


def test_generate_negative_control_absent():
    """El control absent pone valores y unidades a None."""
    rep_row = _make_numeric_rep_row("w001")
    nc_row = generate_negative_control_rep_row(rep_row, NC_ABSENT)

    assert is_negative_control(nc_row)
    assert negative_control_type(nc_row) == NC_ABSENT

    for modality in nc_row["payload"]["modalities"].values():
        for meta in modality.values():
            assert meta["value"] is None
            assert meta["unit"] is None


def test_generate_negative_control_invalid_type():
    """Tipo de control inválido lanza ValueError."""
    rep_row = _make_numeric_rep_row("w001")
    with pytest.raises(ValueError, match="no válido"):
        generate_negative_control_rep_row(rep_row, "invalid_type")


def test_build_execution_manifest_with_negative_controls(tmp_path, monkeypatch):
    """El manifiesto incluye entradas para controles negativos."""
    dataset_dir, _ = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    manifest = build_execution_manifest(
        subjects=["S2"],
        representations=["numeric"],
        prompt_variants=["base"],
        repetitions=1,
        seeds=[42],
        condition_ids=["C0_clean"],
        negative_controls=[NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT],
    )

    entries = manifest["S2"]
    # 1 ventana × 1 rep × 1 var × 1 cond normal = 1
    # + 1 ventana × 4 NC × 1 rep × 1 var = 4
    assert len(entries) == 5

    normal_entries = [e for e in entries if e["condition_id"] == "C0_clean"]
    nc_entries = [e for e in entries if e["condition_id"] in ALL_NEGATIVE_CONTROLS]

    assert len(normal_entries) == 1
    assert len(nc_entries) == 4

    nc_types = {e["condition_id"] for e in nc_entries}
    assert nc_types == set(ALL_NEGATIVE_CONTROLS)


def test_run_orchestrator_with_negative_controls(tmp_path, monkeypatch):
    """Ejecución con controles negativos y Ollama mockeado."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    windows = [_make_window_row("w001")]
    _write_windows(subject_dir, "S2", windows)
    rep_rows = [_make_numeric_rep_row("w001")]
    _write_llm_representation(subject_dir, "S2", "numeric", rep_rows)

    config = LlmExperimentConfig(
        model="qwen3:8b", temperature=0.3, num_ctx=32768, seed=42,
        top_p=0.9, max_tokens=800, timeout=120.0, keep_alive="5m",
        max_retries=3, cascade_enabled=False,
    )

    call_count = 0

    def fake_post(url, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _MockResponse({
            "model": "qwen3:8b",
            "message": {
                "content": (
                    '{"decision": "abstain", "confidence": 30, '
                    '"claims": [], "reasoning": "Sin evidencia."}'
                )
            },
            "done": True,
        })

    with patch("requests.post", side_effect=fake_post):
        summary = run_orchestrator(
            subjects=["S2"],
            config=config,
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            negative_controls=[NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT],
        )

    assert "S2" in summary["subjects"]
    subj = summary["subjects"]["S2"]
    assert subj["n_entries"] == 5  # 1 normal + 4 NC
    # NC_nan y NC_absent producen el mismo prompt para numeric_llm
    # (todos los valores son "not available"), así que comparten el hash
    # de caché y se reutiliza la respuesta: 4 llamadas únicas, no 5.
    assert subj["n_calls"] == 4
    assert subj["n_errors"] == 0
    assert summary["total_calls"] == 4


def test_negative_control_constants():
    """Verificar constantes de controles negativos."""
    assert NC_SHUFFLED == "NC_shuffled"
    assert NC_ZERO == "NC_zero"
    assert NC_NAN == "NC_nan"
    assert NC_ABSENT == "NC_absent"
    assert ALL_NEGATIVE_CONTROLS == (
        "NC_shuffled",
        "NC_zero",
        "NC_nan",
        "NC_absent",
    )


def test_constant_values():
    """Verificar constantes del módulo."""
    assert CONDITION_CLEAN == "C0_clean"
    assert PROMPT_VARIANT_BASE == "base"
    assert PROMPT_VARIANT_PARAPHRASE == "paraphrase"
    assert PROMPT_VARIANT_NO_ABSTAIN == "no_abstain"
    assert REPRESENTATION_NUMERIC == "numeric"
    assert REPRESENTATION_TEXT == "text"


def _make_chat_config(max_retries: int = 0) -> LlmExperimentConfig:
    return LlmExperimentConfig(
        model="qwen2.5:7b",
        temperature=0.3,
        num_ctx=32768,
        seed=42,
        top_p=0.9,
        max_tokens=800,
        timeout=120.0,
        keep_alive="5m",
        max_retries=max_retries,
        cascade_enabled=False,
    )


def _make_chat_client(max_retries: int = 0) -> dict:
    return {
        "host": "http://localhost:11434",
        "model": "qwen2.5:7b",
        "keep_alive": "5m",
        "max_retries": max_retries,
    }


def _valid_ollama_json() -> dict:
    return {
        "model": "qwen2.5:7b",
        "message": {
            "content": '{"decision": "abstain", "confidence": 30, '
            '"claims": [], "reasoning": "no evidence"}'
        },
        "done": True,
        "total_duration": 1000000000,
        "prompt_eval_count": 100,
        "prompt_eval_duration": 200000000,
        "eval_count": 50,
        "eval_duration": 100000000,
    }


def test_config_for_hash_sets_repetition_seed():
    """_config_for_hash reemplaza la semilla conservando el resto de parámetros."""
    config = _make_chat_config()
    forked = _config_for_hash(config, 123)
    assert forked.seed == 123
    assert forked.model == config.model
    assert forked.temperature == config.temperature
    assert forked.num_ctx == config.num_ctx
    assert _prompt_hash("prompt", forked) != _prompt_hash("prompt", config)


def test_call_ollama_chat_uses_per_repetition_seed():
    """_call_ollama_chat envía la semilla de la repetición (no config.seed)."""
    config = _make_chat_config()
    client = _make_chat_client()
    captured: dict = {}

    def fake_post(url, json=None, timeout=None):
        captured["payload"] = json
        return _MockResponse(_valid_ollama_json())

    with patch("requests.post", side_effect=fake_post):
        _call_ollama_chat(client, "prompt", config, seed=999)

    assert captured["payload"]["options"]["seed"] == 999


def test_call_ollama_chat_retries_on_failure():
    """_call_ollama_chat reintenta hasta max_retries veces ante error de red."""
    import requests as requests_mod

    config = _make_chat_config(max_retries=2)
    client = _make_chat_client(max_retries=2)
    attempts = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise requests_mod.RequestException("connection refused")
        return _MockResponse(_valid_ollama_json())

    with patch("requests.post", side_effect=fake_post), patch("time.sleep"):
        text, meta = _call_ollama_chat(client, "prompt", config)

    assert attempts["n"] == 3  # 1 intento + 2 reintentos
    assert "abstain" in text


def test_call_ollama_chat_raises_after_retries_exhausted():
    """Si todos los reintentos fallan, _call_ollama_chat lanza ValidationError."""
    import requests as requests_mod

    config = _make_chat_config(max_retries=1)
    client = _make_chat_client(max_retries=1)

    def fake_post(url, json=None, timeout=None):
        raise requests_mod.RequestException("connection refused")

    with patch("requests.post", side_effect=fake_post), patch("time.sleep"):
        with pytest.raises(Exception, match="Ollama no respondió"):
            _call_ollama_chat(client, "prompt", config)


# ---------------------------------------------------------------------------
# Compuerta de consistencia fisiológica (P1) integrada en el orquestador
# ---------------------------------------------------------------------------


def _make_full_numeric_rep_row(window_id):
    """Fila ``numeric_llm`` con las 23 features y valores coherentes."""
    coherent = {
        "ecg": {
            "ecg_hr_bpm": 60.0,
            "ecg_rr_mean_ms": 1000.0,
            "ecg_n_beats": 60.0,
            "ecg_sdnn_ms": 50.0,
            "ecg_rmssd_ms": 40.0,
        },
        "eda": {
            "eda_mean_us": 5.0,
            "eda_std_us": 0.5,
            "eda_slope_us_per_s": 0.01,
            "eda_tonic_mean_us": 4.99,
            "eda_phasic_std_us": 0.1,
            "eda_scr_count": 3.0,
            "eda_scr_rate_per_min": 3.0,
            "eda_scr_amplitude_mean_us": 0.2,
            "eda_scr_amplitude_max_us": 0.4,
        },
        "resp": {
            "resp_rate_bpm": 15.0,
            "resp_interval_cv": 0.3,
            "resp_amplitude_pct": 10.0,
            "resp_std_pct": 4.0,
            "resp_n_cycles": 15.0,
        },
        "temp": {
            "temp_mean_degc": 33.0,
            "temp_std_degc": 0.25,
            "temp_slope_degc_per_s": 0.0,
            "temp_range_degc": 1.0,
        },
    }
    return _make_numeric_rep_row(
        window_id,
        features={
            modality: {name: {"value": value} for name, value in block.items()}
            for modality, block in coherent.items()
        },
    )


def _stress_ollama_response() -> dict:
    return {
        "model": "qwen2.5:7b",
        "message": {
            "content": (
                '{"decision": "stress", "confidence": 85, '
                '"claims": [], "reasoning": "EDA elevated."}'
            )
        },
        "done": True,
    }


def _read_responses(output_dir, subject="S2", model="qwen2.5:7b"):
    slug = model.replace(":", "_").replace(".", "_")
    path = (
        output_dir / "WESAD" / "INGEST" / "llm_responses"
        / slug / f"{subject}.responses.jsonl"
    )
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_consistency_gate_overrides_incoherent_window(tmp_path, monkeypatch):
    """La compuerta convierte en abstención una decisión sobre datos barajados."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    _write_windows(subject_dir, "S2", [_make_window_row("w001")])
    _write_llm_representation(
        subject_dir, "S2", "numeric", [_make_full_numeric_rep_row("w001")]
    )

    with patch(
        "requests.post",
        side_effect=lambda *a, **k: _MockResponse(_stress_ollama_response()),
    ):
        summary = run_orchestrator(
            subjects=["S2"],
            config=_make_chat_config(),
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            negative_controls=[NC_SHUFFLED],
        )

    records = _read_responses(output_dir)
    clean = next(r for r in records if r["condition_id"] == "C0_clean")
    shuffled = next(r for r in records if r["condition_id"] == NC_SHUFFLED)

    # Ventana limpia: la compuerta se evalúa y no cambia la decisión.
    assert clean["consistency"]["evaluated"] is True
    assert clean["consistency"]["gate_abstained"] is False
    assert clean["decision"] == "stress"
    assert clean["decision_llm"] == "stress"

    # Ventana barajada: abstención forzada conservando la decisión del LLM.
    assert shuffled["consistency"]["evaluated"] is True
    assert shuffled["consistency"]["gate_abstained"] is True
    assert shuffled["consistency"]["n_hard_failures"] >= 1
    assert shuffled["status"] == "gate_abstention"
    assert shuffled["decision"] == "abstain"
    assert shuffled["confidence"] == 0
    assert shuffled["decision_llm"] == "stress"
    assert shuffled["confidence_llm"] == 85

    assert summary["subjects"]["S2"]["n_gate_abstentions"] == 1
    assert summary["total_gate_abstentions"] == 1
    assert summary["consistency_gate"]["enabled"] is True


def test_consistency_gate_can_be_disabled(tmp_path, monkeypatch):
    """Con consistency_gate=False la decisión del LLM no se modifica."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    _write_windows(subject_dir, "S2", [_make_window_row("w001")])
    _write_llm_representation(
        subject_dir, "S2", "numeric", [_make_full_numeric_rep_row("w001")]
    )

    with patch(
        "requests.post",
        side_effect=lambda *a, **k: _MockResponse(_stress_ollama_response()),
    ):
        summary = run_orchestrator(
            subjects=["S2"],
            config=_make_chat_config(),
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            negative_controls=[NC_SHUFFLED],
            consistency_gate=False,
        )

    records = _read_responses(output_dir)
    shuffled = next(r for r in records if r["condition_id"] == NC_SHUFFLED)
    assert shuffled["consistency"] == {"evaluated": False}
    assert shuffled["decision"] == "stress"
    assert shuffled["status"] == "valid_response"
    assert summary["total_gate_abstentions"] == 0
    assert summary["consistency_gate"]["enabled"] is False


def test_consistency_gate_applies_to_cached_responses(tmp_path, monkeypatch):
    """La compuerta se reaplica al reutilizar una respuesta de caché."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    _write_windows(subject_dir, "S2", [_make_window_row("w001")])
    _write_llm_representation(
        subject_dir, "S2", "numeric", [_make_full_numeric_rep_row("w001")]
    )

    call_count = 0

    def fake_post(url, json=None, timeout=None):
        nonlocal call_count
        call_count += 1
        return _MockResponse(_stress_ollama_response())

    with patch("requests.post", side_effect=fake_post):
        # Primera pasada sin compuerta: la caché guarda la respuesta cruda.
        run_orchestrator(
            subjects=["S2"],
            config=_make_chat_config(),
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            negative_controls=[NC_SHUFFLED],
            consistency_gate=False,
        )
        # Segunda pasada con compuerta: 0 llamadas nuevas y veto aplicado.
        summary = run_orchestrator(
            subjects=["S2"],
            config=_make_chat_config(),
            representations=["numeric"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
            negative_controls=[NC_SHUFFLED],
            consistency_gate=True,
        )

    assert call_count == 2  # 1 ventana limpia + 1 barajada
    assert summary["subjects"]["S2"]["n_calls"] == 0
    assert summary["subjects"]["S2"]["n_cached"] == 2

    records = _read_responses(output_dir)
    shuffled = next(r for r in records if r["condition_id"] == NC_SHUFFLED)
    assert shuffled["status"] == "gate_abstention"
    assert shuffled["decision"] == "abstain"
    assert shuffled["decision_llm"] == "stress"


def test_consistency_gate_without_numeric_view(tmp_path, monkeypatch):
    """Sin vista numérica la compuerta no se evalúa (no rompe la ejecución)."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    _write_windows(subject_dir, "S2", [_make_window_row("w001")])
    _write_llm_representation(subject_dir, "S2", "text", [_make_text_rep_row("w001")])

    with patch(
        "requests.post",
        side_effect=lambda *a, **k: _MockResponse(_stress_ollama_response()),
    ):
        summary = run_orchestrator(
            subjects=["S2"],
            config=_make_chat_config(),
            representations=["text"],
            prompt_variants=["base"],
            repetitions=1,
            seeds=[42],
            condition_ids=["C0_clean"],
        )

    records = _read_responses(output_dir)
    assert records[0]["consistency"] == {"evaluated": False}
    assert records[0]["decision"] == "stress"
    assert summary["total_gate_abstentions"] == 0


def test_build_consistency_reports_covers_conditions():
    """Los informes se calculan por ventana y condición, incluidos los NC."""
    rows = {"w001": _make_full_numeric_rep_row("w001")}
    reports = build_consistency_reports(
        numeric_rows=rows,
        duration_by_window={"w001": 60.0},
        conditions=["C0_clean", NC_SHUFFLED, NC_NAN],
    )
    assert set(reports) == {
        ("w001", "C0_clean"),
        ("w001", NC_SHUFFLED),
        ("w001", NC_NAN),
    }
    assert should_abstain(reports[("w001", "C0_clean")]) is False
    assert should_abstain(reports[("w001", NC_SHUFFLED)]) is True
    assert should_abstain(reports[("w001", NC_NAN)]) is True


def test_call_ollama_chat_sends_think_only_for_qwen3(monkeypatch):
    """``think`` viaja solo a familias que lo soportan (qwen3)."""
    from sources.phase2.orchestrator import _call_ollama_chat

    captured: list[dict] = []
    fake = {
        "model": "x",
        "message": {"content": '{"decision": "baseline", "confidence": 80, "claims": [], "reasoning": "ok"}'},
        "done": True,
    }

    class _Mock:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return fake

    def fake_post(url, json=None, timeout=None):
        captured.append(json)
        return _Mock()

    monkeypatch.setattr("requests.post", fake_post)

    qwen3 = LlmExperimentConfig(
        model="qwen3:14b", temperature=0.3, num_ctx=32768, seed=42, top_p=0.9,
        max_tokens=800, timeout=120.0, keep_alive="5m", max_retries=1,
        cascade_enabled=False, think=False,
    )
    qwen25 = LlmExperimentConfig(
        model="qwen2.5:7b", temperature=0.3, num_ctx=32768, seed=42, top_p=0.9,
        max_tokens=800, timeout=120.0, keep_alive="5m", max_retries=1,
        cascade_enabled=False, think=False,
    )
    _, meta3 = _call_ollama_chat(
        {"model": "qwen3:14b", "host": "http://localhost:11434", "keep_alive": "5m"},
        "prompt", qwen3, seed=42,
    )
    _, meta25 = _call_ollama_chat(
        {"model": "qwen2.5:7b", "host": "http://localhost:11434", "keep_alive": "5m"},
        "prompt", qwen25, seed=42,
    )

    assert captured[0]["think"] is False
    assert meta3["think_requested"] is False
    assert "think" not in captured[1]
    assert meta25["think_requested"] is None


def test_responses_isolated_by_model(tmp_path, monkeypatch):
    """Dos modelos escriben en carpetas distintas y no se pisan."""
    dataset_dir, output_dir = _setup_env(monkeypatch, tmp_path)

    subject_dir = dataset_dir / "S2"
    _write_windows(subject_dir, "S2", [_make_window_row("w001")])
    _write_llm_representation(
        subject_dir, "S2", "numeric", [_make_numeric_rep_row("w001")]
    )

    fake = {
        "model": "x",
        "message": {
            "content": (
                '{"decision": "baseline", "confidence": 85, '
                '"claims": [], "reasoning": "Stable."}'
            )
        },
        "done": True,
    }

    def run(model: str):
        config = LlmExperimentConfig(
            model=model, temperature=0.3, num_ctx=32768, seed=42, top_p=0.9,
            max_tokens=800, timeout=120.0, keep_alive="5m", max_retries=1,
            cascade_enabled=False,
        )
        with patch("requests.post") as mock_post:
            mock_post.return_value = _MockResponse(fake)
            return run_orchestrator(
                subjects=["S2"],
                config=config,
                representations=["numeric"],
                prompt_variants=["base"],
                repetitions=1,
                seeds=[42],
                condition_ids=["C0_clean"],
            )

    summary_a = run("qwen2.5:7b")
    summary_b = run("qwen3:14b")

    assert summary_a["total_calls"] == 1
    assert summary_b["total_calls"] == 1  # no reutiliza la caché del otro modelo

    base = output_dir / "WESAD" / "INGEST" / "llm_responses"
    path_a = base / "qwen2_5_7b" / "S2.responses.jsonl"
    path_b = base / "qwen3_14b" / "S2.responses.jsonl"
    assert path_a.is_file() and path_b.is_file()
    assert json.loads(path_a.read_text().splitlines()[0])["model"] == "qwen2.5:7b"
    assert json.loads(path_b.read_text().splitlines()[0])["model"] == "qwen3:14b"


def test_prompt_hash_includes_think():
    """El modo thinking forma parte de la clave de caché."""
    from sources.phase2.orchestrator import _prompt_hash

    base = dict(
        model="qwen3:14b", temperature=0.3, num_ctx=32768, seed=42, top_p=0.9,
        max_tokens=800, timeout=120.0, keep_alive="5m", max_retries=3,
        cascade_enabled=False,
    )
    with_think = LlmExperimentConfig(think=True, **base)
    without_think = LlmExperimentConfig(think=False, **base)

    assert _prompt_hash("prompt", with_think) != _prompt_hash("prompt", without_think)


def test_cache_usable_rejects_timeout():
    """Un timeout cacheado no debe reutilizarse (fallo de transporte)."""
    from sources.phase2.orchestrator import _cache_usable

    assert _cache_usable({"status": "valid_response"}) is True
    assert _cache_usable({"status": "valid_abstention"}) is True
    assert _cache_usable({"status": "timeout"}) is False
    assert _cache_usable(None) is False
    assert _cache_usable({}) is False
