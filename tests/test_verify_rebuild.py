"""Tests de `scripts/verify_corpus_rebuild.py` (verificación de reconstrucción).

Comprueban que el verificador detecta de verdad una deriva del corpus: si las
features cambian, el prompt reconstruido debe dejar de coincidir con el
guardado. Sin eso, el script daría falsos "IDENTICAL".

Nota de diseño: el prompt usa z-scores respecto a la mediana/IQR del propio
sujeto, así que un desplazamiento uniforme de todas las ventanas limpias es
invisible por construcción. La deriva se prueba cambiando solo la condición
degradada, que es el caso realista (recalcular C1 sin tocar la referencia C0).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

W1 = "WESAD:SX:w00001"
W2 = "WESAD:SX:w00002"


def _load_module():
    path = REPO_ROOT / "scripts" / "verify_corpus_rebuild.py"
    spec = importlib.util.spec_from_file_location("verify_corpus_rebuild", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verify = _load_module()


def _features_row(window_id: str, hr: float) -> dict:
    return {
        "window_id": window_id,
        "features": {
            "ecg": {"ecg_hr_bpm": {"value": hr}},
            "eda": {"eda_mean_us": {"value": 2.0}},
            "resp": {"resp_rate_bpm": {"value": 14.0}},
            "temp": {"temp_mean_degc": {"value": 33.0}},
        },
    }


def _write_corpus(root: Path, degraded_hr: float = 80.0) -> None:
    """Dos ventanas limpias (referencia estable) y una condición degradada."""
    windows = root / "windows"
    features = root / "features"
    degraded = root / "degraded_features" / "C1_partial_resp"
    windows.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    degraded.mkdir(parents=True, exist_ok=True)

    with (windows / "SX.windows.jsonl").open("w", encoding="utf-8") as handle:
        for window_id in (W1, W2):
            handle.write(json.dumps({"window_id": window_id, "duration_s": 60.0}) + "\n")

    with (features / "SX.features.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_features_row(W1, 80.0)) + "\n")
        handle.write(json.dumps(_features_row(W2, 90.0)) + "\n")

    with (degraded / "SX.features.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_features_row(W1, degraded_hr)) + "\n")
        handle.write(json.dumps(_features_row(W2, 90.0)) + "\n")


def _prompt_for(corpus: Path, condition: str, window_id: str) -> str:
    centers, scales, _ = verify.reference(corpus, "SX")
    values = verify.load_features(corpus, condition, "SX")[window_id]
    return verify.rebuild_prompt("numeric", window_id, values, centers, scales, 60.0)


def _write_responses(path: Path, prompt: str, condition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "subject": "SX",
        "condition_id": condition,
        "window_id": W1,
        "duration_s": 60.0,
        "prompt": prompt,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_representation_of_por_ruta():
    assert verify.representation_of(Path("x/degradation/S5.responses.jsonl")) == "numeric"
    assert (
        verify.representation_of(Path("x/degradation_text/S5.responses.jsonl")) == "text"
    )
    assert (
        verify.representation_of(Path("x/degradation_text/qwen2.5_7b/S5.responses.jsonl"))
        == "text"
    )


def test_response_files_admite_subdirectorio_de_modelo(tmp_path):
    (tmp_path / "a.responses.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "qwen2.5_7b").mkdir()
    (tmp_path / "qwen2.5_7b" / "b.responses.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "otro.jsonl").write_text("", encoding="utf-8")
    names = [path.name for path in verify.response_files(tmp_path)]
    assert names == ["a.responses.jsonl", "b.responses.jsonl"]


def test_response_files_directorio_inexistente(tmp_path):
    assert verify.response_files(tmp_path / "no_existe") == []


def test_features_match_detecta_cambio_de_valor():
    prompt_a = "Features:\n- ecg_hr_bpm: +0.12 SD\n- eda_mean_us: -0.40 SD\n"
    prompt_b = "Features:\n- ecg_hr_bpm: +0.90 SD\n- eda_mean_us: -0.40 SD\n"
    assert verify.features_match(prompt_a, prompt_a)
    assert not verify.features_match(prompt_a, prompt_b)


def test_verifica_corpus_identico(tmp_path):
    corpus = tmp_path / "INGEST"
    _write_corpus(corpus)
    responses = tmp_path / "degradation" / "SX.responses.jsonl"
    _write_responses(responses, _prompt_for(corpus, "C1_partial_resp", W1), "C1_partial_resp")

    stats = verify.verify_source(responses, corpus)
    assert stats["records"] == 1
    assert stats["prompt_equal"] == 1
    assert stats["features_equal"] == 1
    assert stats["identical"] is True


def test_detecta_deriva_del_corpus(tmp_path):
    """Recalcular la condición degradada con otros valores debe detectarse."""
    corpus = tmp_path / "INGEST"
    _write_corpus(corpus, degraded_hr=80.0)
    prompt_guardado = _prompt_for(corpus, "C1_partial_resp", W1)
    responses = tmp_path / "degradation" / "SX.responses.jsonl"
    _write_responses(responses, prompt_guardado, "C1_partial_resp")

    _write_corpus(corpus, degraded_hr=95.0)  # la referencia C0 no cambia
    stats = verify.verify_source(responses, corpus)

    assert stats["records"] == 1
    assert stats["prompt_equal"] == 0
    assert stats["prompt_diff"] == 1
    assert stats["features_diff"] == 1
    assert stats["identical"] is False
    assert stats["samples"], "debe incluir una muestra de la diferencia"


def test_ventana_ausente_no_cuenta_como_identica(tmp_path):
    corpus = tmp_path / "INGEST"
    _write_corpus(corpus)
    responses = tmp_path / "degradation" / "SX.responses.jsonl"
    responses.parent.mkdir(parents=True)
    responses.write_text(
        json.dumps(
            {
                "subject": "SX",
                "condition_id": "C0_clean",
                "window_id": "WESAD:SX:w99999",
                "duration_s": 60.0,
                "prompt": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stats = verify.verify_source(responses, corpus)
    assert stats["missing_window"] == 1
    assert stats["identical"] is False


def test_condicion_sin_features_cuenta_como_ausente(tmp_path):
    corpus = tmp_path / "INGEST"
    _write_corpus(corpus)
    responses = tmp_path / "degradation" / "SX.responses.jsonl"
    _write_responses(responses, "x", condition="C4_total_eda")
    stats = verify.verify_source(responses, corpus)
    assert stats["missing_features"] == 1
    assert stats["records"] == 0


def test_main_devuelve_2_sin_artefactos(tmp_path):
    code = verify.main(
        ["--ingest", str(tmp_path / "INGEST"), "--responses", str(tmp_path / "vacio"),
         "--output", str(tmp_path / "out.json")]
    )
    assert code == 2
