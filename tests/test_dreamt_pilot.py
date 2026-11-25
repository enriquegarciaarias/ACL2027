"""Pruebas del adaptador de DREAMT y del piloto de sueño.

Cubren las piezas donde un error silencioso cambiaría el resultado sin romper
nada visible: el anclaje de la rejilla de épocas, la selección de ventanas
puras, la fuente del bloque cardiaco, las condiciones de degradación y la
coherencia multiclase de los claims.

Todo se construye con datos sintéticos: las noches reales (14 GB) no pueden
formar parte de la suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from sources.external import dreamt
from sources.phase2 import prompt_builder as pb
from sources.phase2.response_schema import (
    DECISION_MODE_AFFECTIVE,
    DECISION_MODE_SLEEP4,
    validate_llm_output,
)
from sources.phase3 import claims, signatures


FS = int(dreamt.FS)  # 64 Hz
EPOCH_ROWS = dreamt.EPOCH_ROWS  # 1920 muestras = 30 s


# ---------------------------------------------------------------------------
# Utilidades de fixture
# ---------------------------------------------------------------------------


def _frame(rows: int, stages: list[str] | None = None):
    """DataFrame sintético con las columnas que usa el adaptador.

    Genera señales con variabilidad suficiente para que los extractores
    publicen valores (no una constante).
    """
    import pandas as pd

    rng = np.random.default_rng(7)
    index = np.arange(rows)
    if stages is None:
        stages = ["N2"] * rows
    eda = 0.10 + 0.02 * np.sin(index / 300.0) + rng.normal(0, 0.002, rows)
    return pd.DataFrame(
        {
            "TIMESTAMP": index / FS,
            "BVP": np.sin(2 * np.pi * 1.1 * index / FS) * 100,
            "EDA": eda,
            "TEMP": 32.0 + rng.normal(0, 0.01, rows),
            "ACC_X": rng.normal(0, 2.0, rows),
            "ACC_Y": rng.normal(0, 2.0, rows),
            "ACC_Z": rng.normal(0, 2.0, rows),
            "HR": 62.0 + rng.normal(0, 0.5, rows),
            "IBI": np.full(rows, 0.96875),
            "Sleep_Stage": list(stages),
        }
    )


def _expand(epoch_labels: list[str]) -> list[str]:
    """Convierte etiquetas por época en la serie de muestras que espera la rejilla."""
    out: list[str] = []
    for label in epoch_labels:
        out.extend([label] * EPOCH_ROWS)
    return out


def _arrays(values: np.ndarray, size: int) -> dict[str, np.ndarray]:
    """Arrays mínimos para probar las condiciones de degradación."""
    return {
        "eda": values,
        "temp": np.full(size, 32.0),
        "acc": np.zeros((size, 3)),
        "bvp": np.zeros(size),
        "ibi": np.zeros(size),
        "hr": np.full(size, 60.0),
    }


# ---------------------------------------------------------------------------
# Rejilla de épocas y ventanas puras
# ---------------------------------------------------------------------------


def test_anchor_skips_preparation():
    """El ancla es la primera muestra que no es preparación ``P``."""
    stages = ["P"] * 10 + ["W"] * 5
    assert dreamt.anchor_index(stages) == 10


def test_epoch_grid_anchored_has_no_mixed_epochs():
    """Con el ancla correcta, una noche limpia no produce épocas mixtas.

    Anclar en la fila 0 (el error documentado) mezclaría prep con W casi en
    cada bloque; con el ancla real la pureza es 1.
    """
    offset = 704  # desfase real medido en S004 (§3 del preregistro)
    stages = _expand(["P", "N2", "N2", "N2"])
    stages = ["P"] * offset + stages[EPOCH_ROWS:]
    grid = dreamt.epoch_grid(stages)
    assert grid.anchor == offset
    assert len(grid.labels) == 3
    assert all(p == pytest.approx(1.0) for p in grid.purity)


def test_epoch_grid_without_anchor_produces_mixed_epochs():
    """Contraprueba: sin ancla la pureza baja, así que el test anterior muerde."""
    offset = 704
    stages = _expand(["P", "N2", "N2", "N2"])
    stages = ["P"] * offset + stages[EPOCH_ROWS:]
    grid = dreamt.epoch_grid(stages, anchor=0)
    assert min(grid.purity) < 1.0


def test_pure_windows_require_two_consecutive_epochs():
    """Una ventana de 60 s necesita dos épocas consecutivas del mismo estadio."""
    stages = _expand(["W", "W", "N2", "W", "W"])
    grid = dreamt.epoch_grid(stages, anchor=0)
    windows = dreamt.pure_windows("S999", grid, purity_min=0.99)
    labels = [w.stage for w in windows]
    # W,W (épocas 0-1) y W,W (3-4); el salto N2 en medio rompe el tramo.
    assert labels == ["W", "W"]
    assert [w.epoch_index for w in windows] == [0, 3]


def test_sample_windows_spreads_across_night():
    """El muestreo cubre el principio y el final de la noche, no un extremo."""
    stages = _expand(["W"] * 2 + ["N2"] * 20 + ["R"] * 8)
    grid = dreamt.epoch_grid(stages, anchor=0)
    windows = dreamt.pure_windows("S999", grid, purity_min=0.99)
    picked = dreamt.sample_windows(windows, per_class=2)
    assert picked
    fractions = [w.night_frac for w in picked]
    assert min(fractions) < 0.25 and max(fractions) > 0.75
    # Dos por clase como máximo.
    counts = {}
    for window in picked:
        counts[window.stage] = counts.get(window.stage, 0) + 1
    assert max(counts.values()) <= 2


# ---------------------------------------------------------------------------
# Bloque cardiaco
# ---------------------------------------------------------------------------


def test_cardiac_features_use_device_hr():
    """HR sale de la mediana de la serie del dispositivo; RR es su inversa."""
    hr = np.full(60, 60.0)
    hr[5] = 200.0  # artefacto de contacto: la mediana no lo sigue
    features, meta = dreamt.cardiac_features(hr)
    assert meta["method"] == "device_hr_1hz"
    assert features["ecg_hr_bpm"] == pytest.approx(60.0)
    assert features["ecg_rr_mean_ms"] == pytest.approx(1000.0)
    # La rama wearable no permite HRV: se publica como no disponible.
    assert features["ecg_sdnn_ms"] is None
    assert features["ecg_rmssd_ms"] is None
    assert features["ecg_n_beats"] is None


def test_cardiac_features_without_hr_report_reason():
    """Sin muestras fisiológicas no se inventa HR."""
    features, meta = dreamt.cardiac_features(np.full(60, np.nan))
    assert features["ecg_hr_bpm"] is None
    assert meta["reason"] == "hr_missing"


def test_cardiac_features_reject_impossible_values():
    """Ceros de pérdida de contacto se descartan antes de la mediana."""
    hr = np.zeros(60)
    hr[10:40] = 70.0
    features, meta = dreamt.cardiac_features(hr)
    assert features["ecg_hr_bpm"] == pytest.approx(70.0)
    assert meta["n_samples"] == 30


# ---------------------------------------------------------------------------
# Condiciones de degradación
# ---------------------------------------------------------------------------


def test_c1_makes_eda_absent_but_keeps_cardiac():
    arrays = _arrays(np.full(240, 0.2), 240)
    out = dreamt.apply_array_condition(arrays, dreamt.C1_ABSENT_EDA)
    assert np.isnan(out["eda"]).all()
    assert np.isfinite(out["hr"]).all()


def test_c2_degenerates_eda_to_zeros():
    arrays = _arrays(np.full(240, 0.2), 240)
    out = dreamt.apply_array_condition(arrays, dreamt.C2_DEGENERATE_EDA)
    assert np.all(out["eda"] == 0)
    assert np.isfinite(out["hr"]).all()


def test_c3_removes_cardiac_block():
    arrays = _arrays(np.full(240, 0.2), 240)
    out = dreamt.apply_array_condition(arrays, dreamt.C3_ABSENT_HRV)
    assert np.isnan(out["hr"]).all()
    assert np.isfinite(out["eda"]).all()


def test_c4_removes_core_and_keeps_only_auxiliary():
    arrays = _arrays(np.full(240, 0.2), 240)
    out = dreamt.apply_array_condition(arrays, dreamt.C4_ABSENT_CORE)
    assert np.isnan(out["eda"]).all()
    assert np.isnan(out["hr"]).all()
    assert np.isnan(out["acc"]).all()
    assert np.isfinite(out["temp"]).all()


def test_nc_absent_keeps_keys_with_none():
    """``NC_absent`` deja las claves: el prompt debe decir «not available»."""
    features = {"ecg": {"ecg_hr_bpm": 60.0}, "eda": {"eda_std_us": 0.1}}
    out = dreamt.apply_feature_condition(features, dreamt.NC_ABSENT)
    assert out["ecg"]["ecg_hr_bpm"] is None
    assert set(out["eda"]) == {"eda_std_us"}


def test_unknown_condition_raises():
    arrays = _arrays(np.full(240, 0.2), 240)
    with pytest.raises(ValueError):
        dreamt.apply_array_condition(arrays, "C9_invented")


# ---------------------------------------------------------------------------
# Vistas numérica y textual
# ---------------------------------------------------------------------------


def test_numeric_and_text_views_show_the_same_features():
    """H3 compara formatos: si el texto mostrara otras features, mediría otra cosa."""
    text_features = {name for names in dreamt.SLEEP_TEXT_SECTIONS.values() for name in names}
    assert text_features == set(pb.SLEEP_FEATURES)


def test_sleep_view_excludes_unavailable_hrv():
    """Las features sin datos en la rama wearable no se envían al modelo."""
    for name in ("ecg_sdnn_ms", "ecg_rmssd_ms", "ecg_n_beats"):
        assert name not in pb.SLEEP_FEATURES


def test_render_sleep_text_round_trips_through_parser():
    """Toda etiqueta del texto se resuelve al nombre canónico al verificar claims."""
    zvalues = {name: 0.5 for name in pb.SLEEP_FEATURES}
    zvalues["ecg_hr_bpm"] = None
    text = dreamt.render_sleep_text(zvalues)
    parsed = claims.parse_prompt_features(
        "Signal summary:\n" + text, dreamt.SLEEP_METRIC_ALIASES
    )
    assert set(parsed) == set(pb.SLEEP_FEATURES)
    assert parsed["ecg_hr_bpm"].value is None
    assert parsed["eda_std_us"].value == pytest.approx(0.5)


def test_sleep_payload_keeps_missing_values_visible():
    """Un valor ausente aparece como ``None`` con su unidad, no se omite."""
    payload = dreamt.build_sleep_payload("S999:e0001", {"eda_std_us": None})
    blocks = payload["modalities"]
    assert blocks["eda"]["eda_std_us"]["value"] is None


def test_metric_aliases_resolve_every_text_label():
    """Sin alias completo, los claims del texto se contarían como ``unsupported``."""
    for label in dreamt.SLEEP_TEXT_LABELS.values():
        assert claims.normalize_metric(label, dreamt.SLEEP_METRIC_ALIASES) is not None


def test_metric_aliases_accept_canonical_movement_names():
    """Caso real: ``acc_std_g`` no está en el vocabulario de WESAD."""
    assert claims.normalize_metric("acc_std_g") is None
    assert (
        claims.normalize_metric("acc_std_g", dreamt.SLEEP_METRIC_ALIASES) == "acc_std_g"
    )


def test_verify_claims_uses_dataset_vocabulary_for_claim_metrics():
    """El vocabulario del dataset tiene que llegar también al claim, no solo al prompt.

    Bug real del piloto 2026-11-23: el modelo citaba la etiqueta legible que
    había leído («Movement variability») y el verificador la contaba como
    ``unsupported`` (métrica ausente) porque normalizaba el prompt con alias y
    el claim sin ellos.
    """
    zvalues = {name: 0.5 for name in pb.SLEEP_FEATURES}
    prompt = "Signal summary:\n" + dreamt.render_sleep_text(zvalues)
    claim = {
        "metric": "Movement variability",
        "channel": "acc",
        "direction": "increase",
        "magnitude": "low",
    }
    verdicts = claims.verify_claims([claim], prompt, dreamt.SLEEP_METRIC_ALIASES)
    assert verdicts[0].metric == "acc_std_g"
    assert verdicts[0].category == "supported"


def test_metric_aliases_tolerate_channel_level_names():
    """Caso real (piloto 2026-11-23): el modelo nombra el canal, no la métrica.

    medgemma:4b escribe «EDA conductance» y qwen2.5:7b «eda_mean_conductance»;
    sin alias se contarían como ``unsupported`` (métrica ausente) y la tasa de
    respaldo mediría el vocabulario del verificador. El caso con guion bajo
    («acc_mean») se resuelve por la forma alternativa del nombre.
    """
    aliases = dreamt.SLEEP_METRIC_ALIASES
    assert claims.normalize_metric("EDA conductance", aliases) == "eda_mean_us"
    assert claims.normalize_metric("Conductance", aliases) == "eda_mean_us"
    assert claims.normalize_metric("eda_mean_conductance", aliases) == "eda_mean_us"
    assert claims.normalize_metric("acc_mean", aliases) == "acc_mean_g"
    # Fuera de un vocabulario de dataset el comportamiento no cambia.
    assert claims.normalize_metric("acc_mean") is None


# ---------------------------------------------------------------------------
# Esquema de respuesta sleep4
# ---------------------------------------------------------------------------


def _sleep_json(decision: str = "n2", channel: str = "acc") -> str:
    return (
        '{"decision": "%s", "confidence": 70, '
        '"claims": [{"metric": "acc_std_g", "channel": "%s", '
        '"direction": "decrease", "magnitude": "low"}], '
        '"reasoning": "Low movement."}' % (decision, channel)
    )


def test_sleep4_accepts_stage_and_movement_channel():
    """En esta tarea el acelerómetro es evidencia legítima."""
    response = validate_llm_output(
        _sleep_json("n2", "acc"),
        decision_mode=DECISION_MODE_SLEEP4,
        allowed_channels={"ecg", "eda", "temp", "acc"},
    )
    assert response.status == "valid_response"
    assert response.decision == "n2"


def test_sleep4_rejects_affective_label():
    """Una etiqueta afectiva no existe en el espacio de sueño."""
    response = validate_llm_output(
        _sleep_json("baseline", "eda"),
        decision_mode=DECISION_MODE_SLEEP4,
        allowed_channels={"ecg", "eda", "temp", "acc"},
    )
    assert response.status == "invalid_output"


def test_sleep4_accepts_abstention():
    response = validate_llm_output(
        '{"decision": "abstain", "confidence": 0, "claims": [], '
        '"reasoning": "No EDA available."}',
        decision_mode=DECISION_MODE_SLEEP4,
        allowed_channels={"ecg", "eda", "temp", "acc"},
    )
    assert response.status == "valid_abstention"


def test_affective_mode_rejects_sleep_label():
    response = validate_llm_output(
        _sleep_json("n2", "eda"), decision_mode=DECISION_MODE_AFFECTIVE
    )
    assert response.status == "invalid_output"


def test_wesad_default_channels_still_reject_movement():
    """El alias de canal no abre el acelerómetro en la tarea afectiva."""
    response = validate_llm_output(_sleep_json("n2", "accelerometer"))
    assert response.status == "invalid_output"


# ---------------------------------------------------------------------------
# Tabla de firma y coherencia multiclase
# ---------------------------------------------------------------------------


def test_sleep_signature_excludes_temperature_confounder():
    for table in signatures.SLEEP_SIGNATURES.values():
        assert "temp_mean_degc" not in table
        assert "temp_std_degc" not in table


def test_sleep_signature_has_no_hrv_after_deviation_d7():
    for table in signatures.SLEEP_SIGNATURES.values():
        assert "ecg_rmssd_ms" not in table
        assert "ecg_sdnn_ms" not in table


def test_n1_has_no_signature():
    assert signatures.SLEEP_SIGNATURES["n1"] == {}
    assert not signatures.has_sleep_signature("n1")
    assert signatures.has_sleep_signature("wake")


def test_sleep_signs_are_defined_for_the_three_classes():
    assert signatures.sleep_sign("wake", "acc_std_g") == +1
    assert signatures.sleep_sign("n2", "acc_std_g") == -1
    assert signatures.sleep_sign("rem", "eda_std_us") == +1
    assert signatures.sleep_sign("rem", "ecg_hr_bpm") == 0


def test_coherence_wake_with_movement_increase_is_coherent():
    claim = {"metric": "acc_std_g", "channel": "acc", "direction": "increase", "magnitude": "high"}
    pair = claims.claim_coherence(claim, "wake", signatures=signatures.SLEEP_SIGNATURES)
    assert pair.category == "coherent"


def test_coherence_n2_with_movement_increase_is_contrary():
    claim = {"metric": "acc_std_g", "channel": "acc", "direction": "increase", "magnitude": "high"}
    pair = claims.claim_coherence(claim, "n2", signatures=signatures.SLEEP_SIGNATURES)
    assert pair.category == "contrary"


def test_coherence_n1_is_not_applicable():
    """Sin firma, la coherencia no se puede juzgar: no cuenta como fallo."""
    claim = {"metric": "acc_std_g", "channel": "acc", "direction": "increase", "magnitude": "low"}
    pair = claims.claim_coherence(claim, "n1", signatures=signatures.SLEEP_SIGNATURES)
    assert pair.category == "no_signature"


def test_abstention_is_not_counted_as_incoherence():
    claim = {"metric": "acc_std_g", "channel": "acc", "direction": "increase", "magnitude": "low"}
    pair = claims.claim_coherence(claim, "abstain", signatures=signatures.SLEEP_SIGNATURES)
    assert pair.category == "not_applicable"


# ---------------------------------------------------------------------------
# Clases y etapas
# ---------------------------------------------------------------------------


def test_n3_is_excluded_from_the_class_space():
    assert "N3" not in dreamt.TASK_STAGES
    assert "N3" not in dreamt.CLASS_OF_STAGE


def test_class_of_stage_maps_the_four_classes():
    assert dreamt.CLASS_OF_STAGE["W"] == "wake"
    assert dreamt.CLASS_OF_STAGE["N1"] == "n1"
    assert dreamt.CLASS_OF_STAGE["N2"] == "n2"
    assert dreamt.CLASS_OF_STAGE["R"] == "rem"


def test_window_arrays_decimates_to_the_documented_rates():
    """EDA/TEMP 4 Hz, ACC 32 Hz, BVP/HR a 64 Hz y ACC en g (÷64)."""
    frame = _frame(2 * EPOCH_ROWS)
    grid = dreamt.epoch_grid(dreamt.stage_array(frame), anchor=0)
    window = dreamt.pure_windows("S999", grid, purity_min=0.99)[0]
    arrays = dreamt.window_arrays(frame, window)
    assert arrays["eda"].size == int(dreamt.WINDOW_S * dreamt.FS_EDA)
    assert arrays["temp"].size == int(dreamt.WINDOW_S * dreamt.FS_TEMP)
    assert arrays["acc"].shape == (int(dreamt.WINDOW_S * dreamt.FS_ACC), 3)
    assert arrays["hr"].size == int(dreamt.WINDOW_S * FS)


def test_low_level_extraction_produces_the_documented_keys():
    """Las claves del payload son las que el prompt espera mostrar."""
    from sources.ingestion.config import get_ingestion_config

    frame = _frame(2 * EPOCH_ROWS)
    grid = dreamt.epoch_grid(dreamt.stage_array(frame), anchor=0)
    window = dreamt.pure_windows("S999", grid, purity_min=0.99)[0]
    features, _notes = dreamt.extract_features(
        dreamt.window_arrays(frame, window), get_ingestion_config()
    )
    flat = dreamt.flatten_features(features)
    assert set(pb.SLEEP_FEATURES) <= set(flat)


# ---------------------------------------------------------------------------
# Diagnóstico de confianza del informe (H2a)
# ---------------------------------------------------------------------------


def test_condition_block_reports_confidence_diagnostics():
    """El informe debe separar acierto de confianza y detectar el techo.

    Con una confianza casi constante (lo que hace el 7b) el AUROC no puede
    separar aciertos de errores: sin esta cifra se publicaría una
    "sobreconfianza" sin saber si la confianza informa.
    """
    from scripts.run_p4_dreamt import _condition_block

    def record(opaque: str, decision: str, confidence: int) -> dict:
        return {
            "window_id": opaque.replace("opaque-", ""),
            "window_id_opaque": opaque,
            "status": "valid_response",
            "decision": decision,
            "confidence": confidence,
            "claims": [],
            "reasoning": "",
        }

    labels = {
        "opaque-1": "wake",
        "opaque-2": "n2",
        "opaque-3": "n2",
        "opaque-4": "rem",
    }
    records = [
        record("opaque-1", "wake", 85),   # acierto, alta confianza
        record("opaque-2", "n2", 75),     # acierto, baja confianza
        record("opaque-3", "wake", 70),   # error
        record("opaque-4", "n2", 70),     # error
    ]
    block = _condition_block(records, labels)
    assert block["confidence_auroc"] == pytest.approx(1.0)
    assert block["overconfidence"] == pytest.approx(0.75 - 0.5)
    assert block["confidence_values"] == {"70": 2, "75": 1, "85": 1}


def test_condition_block_separates_unknown_names_from_absent_metrics():
    """``unsupported`` mezcla dos cosas y el informe tiene que poder separarlas.

    Un claim puede ser ``unsupported`` porque la métrica no está en la entrada
    (evidencia ausente, lo que mide H5) o porque el verificador no conoce ese
    nombre (vocabulario). Sin el diagnóstico, un alias que falte se leería como
    alucinación. Caso real: el brazo textual del piloto de 2026-11-23.
    """
    from scripts.run_p4_dreamt import _condition_block

    zvalues = {name: 1.0 for name in pb.SLEEP_FEATURES}
    prompt = "Signal summary:\n" + dreamt.render_sleep_text(zvalues)

    def record(index: int, claim: dict) -> dict:
        return {
            "window_id": f"S999:e{index:04d}",
            "window_id_opaque": f"opaque-{index}",
            "condition_id": "C0_clean",
            "status": "valid_response",
            "decision": "n2",
            "confidence": 80,
            "claims": [claim],
            "prompt": prompt,
            "reasoning": "evidence is consistent",
        }

    unknown = {
        "metric": "banana_score",
        "channel": "eda",
        "direction": "increase",
        "magnitude": "low",
    }
    # ``ecg_sdnn_ms`` existe en el vocabulario pero no en la rama wearable.
    absent = {
        "metric": "ecg_sdnn_ms",
        "channel": "ecg",
        "direction": "increase",
        "magnitude": "low",
    }
    readable = {
        "metric": "Mean conductance",
        "channel": "eda",
        "direction": "increase",
        "magnitude": "medium",
    }
    block = _condition_block(
        [record(0, unknown), record(1, absent), record(2, readable)],
        {f"opaque-{i}": "n2" for i in range(3)},
    )
    assert block["claims"]["unrecognized_metric_names"] == 1
    assert block["claims"]["categories"]["unsupported"] == 2
    assert block["claims"]["categories"]["supported"] == 1


def test_condition_block_ignores_abstentions_for_confidence():
    """Las abstenciones no tienen acierto/error: quedan fuera del AUROC."""
    from scripts.run_p4_dreamt import _condition_block

    labels = {"opaque-1": "wake"}
    records = [
        {
            "window_id": "1",
            "window_id_opaque": "opaque-1",
            "status": "valid_abstention",
            "decision": "abstain",
            "confidence": 0,
            "claims": [],
            "reasoning": "insufficient evidence",
        }
    ]
    block = _condition_block(records, labels)
    assert np.isnan(block["confidence_auroc"])
    assert block["confidence_values"] == {}
    assert block["n_abstain"] == 1


# ---------------------------------------------------------------------------
# Filtro de sujetos (--subjects): el CLI reenvía la bandera, el runner la aplica
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [None, "", "   ", ","])
def test_subject_filter_empty_means_no_filter(raw):
    """Sin valor no se filtra: el piloto congelado se usa entero."""
    from scripts.run_p4_dreamt import parse_subject_filter

    assert parse_subject_filter(raw) is None


def test_subject_filter_accepts_subset_in_preregistered_order():
    """El subconjunto conserva el orden del preregistro, no el del CLI."""
    from scripts.run_p4_dreamt import parse_subject_filter

    assert parse_subject_filter(" s084 , S034 ") == ("S034", "S084")


@pytest.mark.parametrize("raw", ["S002", "S034,S002", "S999"])
def test_subject_filter_rejects_outside_pilot(raw):
    """Un sujeto de desarrollo o desconocido rompería el preregistro."""
    from scripts.run_p4_dreamt import parse_subject_filter

    with pytest.raises(ValueError):
        parse_subject_filter(raw)


def test_select_pilot_subjects_respects_filter_and_presence():
    """Disponibilidad y filtro se combinan sin cambiar el orden."""
    from scripts.run_p4_dreamt import PILOT_SUBJECTS, select_pilot_subjects

    available = list(PILOT_SUBJECTS)
    assert select_pilot_subjects(available) == list(PILOT_SUBJECTS)
    assert select_pilot_subjects(available, ("S076",)) == ["S076"]
    # Un sujeto del piloto sin datos no aparece aunque se pida.
    assert select_pilot_subjects(["S034"], ("S076",)) == []


def test_requested_subjects_turns_validation_into_exit_code():
    """El CLI debe salir con mensaje, no con traza, ante un sujeto inválido."""
    from argparse import Namespace

    from scripts.run_p4_dreamt import requested_subjects

    assert requested_subjects(Namespace(subjects="S034")) == ("S034",)
    with pytest.raises(SystemExit):
        requested_subjects(Namespace(subjects="S002"))


def test_filter_prompt_rows_limits_conditions():
    """Un humo de 2 condiciones no debe gastar las 6 (fallo del 2026-11-23)."""
    from scripts.run_p4_dreamt import filter_prompt_rows

    rows = [
        {"window_id": f"w{i}", "condition_id": cond}
        for i in range(3)
        for cond in ("C0_clean", "C1_absent_eda", "C2_degenerate_eda")
    ]
    assert len(filter_prompt_rows(rows, None)) == 9
    subset = filter_prompt_rows(rows, ["C0_clean", "C2_degenerate_eda"])
    assert len(subset) == 6
    assert {row["condition_id"] for row in subset} == {"C0_clean", "C2_degenerate_eda"}
    with pytest.raises(SystemExit):
        filter_prompt_rows(rows, ["C9_inexistente"])


def test_sampling_consistency_reads_effective_prompts(tmp_path):
    """El aviso se basa en los prompts escritos, no en lo que pidió el manifiesto."""
    import json as _json

    from scripts.run_p4_dreamt import check_sampling_consistency

    assert check_sampling_consistency(tmp_path, ["S034"], 1, ("C0_clean",)) == []

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    rows = [
        {"window_id": f"S034:e{i:04d}", "condition_id": cond, "stage": stage}
        for stage in ("W", "N1", "N2", "R")
        for i in range(5)
        for cond in ("C0_clean", "C4_absent_core")
    ]
    with (prompts / "S034.prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json.dumps(row) + "\n")
    # El manifiesto miente a propósito: dice 1 cuando los prompts tienen 5.
    (tmp_path / "manifest.json").write_text(
        _json.dumps({"windows_per_class": 1, "conditions": ["C0_clean"]}), encoding="utf-8"
    )

    assert check_sampling_consistency(tmp_path, ["S034"], 5, ("C0_clean",)) == []
    drift = check_sampling_consistency(tmp_path, ["S034"], 1, ("C0_clean", "NC_absent"))
    assert len(drift) == 2
    assert any("windows_per_class" in item for item in drift)
    assert any("NC_absent" in item for item in drift)


# ---------------------------------------------------------------------------
# Cohorte de escala (declarada el 2026-11-24 antes de generar)
# ---------------------------------------------------------------------------


def test_scale_cohort_is_frozen_and_disjoint():
    """12 sujetos, sin solape con desarrollo ni con el piloto."""
    from scripts.run_p4_dreamt import (
        DREAMT_DEV_SUBJECTS,
        PILOT_SUBJECTS,
        SCALE_SUBJECTS,
    )

    assert len(SCALE_SUBJECTS) == 12
    assert len(set(SCALE_SUBJECTS)) == 12
    assert list(SCALE_SUBJECTS) == sorted(SCALE_SUBJECTS)
    assert not set(SCALE_SUBJECTS) & set(PILOT_SUBJECTS)
    assert not set(SCALE_SUBJECTS) & set(DREAMT_DEV_SUBJECTS)


def test_scale_cohort_is_spread_across_the_id_range():
    """No debe ser un bloque de identificadores consecutivos (orden de reclutamiento)."""
    from scripts.run_p4_dreamt import SCALE_SUBJECTS

    numbers = [int(name[1:]) for name in SCALE_SUBJECTS]
    assert min(numbers) < 20
    assert max(numbers) >= 90
    # Ningún par consecutivo del dataset (paso mínimo observado > 1).
    assert all(b - a > 1 for a, b in zip(numbers, numbers[1:]))


def test_cohorts_and_default_outputs_are_separate():
    """Las dos cohortes existen y nunca escriben en la misma carpeta."""
    from scripts.run_p4_dreamt import (
        COHORTS,
        DEFAULT_OUTPUTS,
        PILOT_SUBJECTS,
        SCALE_SUBJECTS,
    )

    assert set(COHORTS) == {"pilot", "scale"}
    assert COHORTS["pilot"] == PILOT_SUBJECTS
    assert COHORTS["scale"] == SCALE_SUBJECTS
    assert DEFAULT_OUTPUTS["pilot"] != DEFAULT_OUTPUTS["scale"]
    assert DEFAULT_OUTPUTS["pilot"].name == "pilot"
    assert DEFAULT_OUTPUTS["scale"].name == "scale"


def test_cohort_filter_validates_against_the_requested_cohort():
    """Un sujeto del piloto no es válido en la cohorte de escala (y al revés)."""
    from scripts.run_p4_dreamt import parse_subject_filter

    assert parse_subject_filter("S008", cohort="scale") == ("S008",)
    with pytest.raises(ValueError):
        parse_subject_filter("S034", cohort="scale")
    with pytest.raises(ValueError):
        parse_subject_filter("S008", cohort="pilot")
    with pytest.raises(ValueError):
        parse_subject_filter("S008", cohort="nope")


def test_select_cohort_subjects_respects_order_and_presence():
    """El orden declarado manda y los sujetos sin datos desaparecen."""
    from scripts.run_p4_dreamt import SCALE_SUBJECTS, select_cohort_subjects

    available = list(reversed(SCALE_SUBJECTS))
    assert select_cohort_subjects(available, cohort="scale") == list(SCALE_SUBJECTS)
    first, second = SCALE_SUBJECTS[0], SCALE_SUBJECTS[1]
    assert select_cohort_subjects(available, (second, first), cohort="scale") == [
        first,
        second,
    ]
    assert select_cohort_subjects([first], (second,), cohort="scale") == []


def test_requested_subjects_accepts_cohort_namespace():
    """El CLI traduce la cohorte antes de validar el filtro."""
    from argparse import Namespace

    from scripts.run_p4_dreamt import requested_subjects

    assert requested_subjects(Namespace(subjects="S008", cohort="scale")) == ("S008",)
    with pytest.raises(SystemExit):
        requested_subjects(Namespace(subjects="S034", cohort="scale"))
    # Sin cohorte explícita sigue siendo el piloto (compatibilidad).
    assert requested_subjects(Namespace(subjects="S034")) == ("S034",)
