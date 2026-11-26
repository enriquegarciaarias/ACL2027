"""Tests del despacho de fases (`sources/pipelines.py`)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sources.common.common import processControl
from sources import pipelines


def _args(**overrides) -> SimpleNamespace:
    """Argumentos de CLI mínimos con los valores por defecto del proyecto."""
    base = dict(
        proc="GENERATE",
        experiment=None,
        subject="WESAD",
        subjects=None,
        model="qwen3:14b",
        think=False,
        force=False,
        dry_run=False,
        allowDev=False,
        representation=None,
        conditions="",
        split=None,
        max_windows=None,
        max_records=None,
        per_class=None,
        task=None,
        tag=None,
        raw_decisions=False,
        few_shot=False,
        output=None,
        compare_models="",
        noGate=False,
        nonstress_variant=None,
        temperature=None,
        coherence_set=None,
        hierarchy=False,
        from_affective=False,
        bootstrap=None,
        phase=None,
        windows_per_class=None,
        reference=False,
        seed=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_process_control(monkeypatch):
    """Aísla ``processControl`` entre tests."""
    monkeypatch.setattr(processControl, "args", None, raising=False)
    monkeypatch.setattr(processControl, "defaults", {"llm": {}}, raising=False)
    monkeypatch.setattr(processControl, "parms", {}, raising=False)


def test_experiment_defaults_to_controls():
    assert pipelines.experiment_of(_args()) == "controls"
    assert pipelines.experiment_of(_args(experiment="degradation")) == "degradation"


def test_generate_controls_runs_in_process():
    """``controls`` no delega en un script."""
    assert pipelines.build_generate_argv("controls", _args()) is None


def test_generate_degradation_argv():
    argv = pipelines.build_generate_argv(
        "degradation",
        _args(
            representation="text",
            conditions="C0_clean,C4_total_eda",
            subjects="S5,S6",
            allowDev=True,
        ),
    )
    assert argv[:2] == ["--model", "qwen3:14b"]
    assert "--representation" in argv and "text" in argv
    assert "--conditions" in argv and "C0_clean,C4_total_eda" in argv
    assert "--subjects" in argv and "S5,S6" in argv
    assert "--allow-dev" in argv
    assert "--think" not in argv


def test_generate_degradation_passes_think():
    """``--think`` no se perdía en el despacho (gap detectado el 2026-11-13)."""
    argv = pipelines.build_generate_argv(
        "degradation", _args(representation="numeric", think=True)
    )
    assert "--think" in argv


def test_generate_physionet_argv():
    argv = pipelines.build_generate_argv(
        "physionet", _args(per_class=20, max_records=10, few_shot=True)
    )
    assert "--per-class" in argv and "20" in argv
    assert "--max-records" in argv and "10" in argv
    assert "--few-shot" in argv
    assert "--subjects" not in argv  # el muestreo es interno


def test_generate_kemocon_argv():
    argv = pipelines.build_generate_argv(
        "kemocon", _args(task="arousal", max_windows=5)
    )
    assert "--task" in argv and "arousal" in argv
    assert "--max-windows" in argv and "5" in argv


def test_generate_dreamt_argv():
    """El piloto de sueño es un runner con fases: el --proc decide la fase."""
    argv = pipelines.build_generate_argv(
        "dreamt",
        _args(representation="both", windows_per_class=5, seed=42, subjects="S034"),
    )
    assert argv[:2] == ["--phase", "generate"]
    assert "--representation" in argv and "both" in argv
    assert "--windows-per-class" in argv and "5" in argv
    assert "--subjects" in argv and "S034" in argv


def test_evaluate_dreamt_argv_resolves_model_and_phase():
    """Sin --model la evaluación no podría leer la carpeta del modelo."""
    processControl.defaults = {"llm": {"primary_model": "qwen2.5:7b"}}
    argv = pipelines.build_evaluate_argv(
        "dreamt", _args(model=None, reference=True, subjects="S034")
    )
    assert argv[:2] == ["--phase", "evaluate"]
    assert "--model" in argv and "qwen2.5:7b" in argv
    assert "--reference" in argv
    # La evaluación también debe poder restringirse a un subconjunto del piloto.
    assert "--subjects" in argv and "S034" in argv


def test_dreamt_call_estimate_counts_both_views():
    """80 ventanas x 6 condiciones x 2 vistas = 960, el coste del preregistro."""
    calls = pipelines._estimate_calls(
        "dreamt",
        ["S034", "S051", "S076", "S084"],
        _args(windows_per_class=5, representation="both"),
    )
    assert calls == 960
    numeric_only = pipelines._estimate_calls(
        "dreamt",
        ["S034", "S051", "S076", "S084"],
        _args(windows_per_class=5, representation="numeric"),
    )
    assert numeric_only == 480


def test_unknown_experiment_raises():
    with pytest.raises(ValueError):
        pipelines.build_generate_argv("no-existe", _args())
    with pytest.raises(ValueError):
        pipelines.build_evaluate_argv("no-existe", _args())


def test_evaluate_controls_argv():
    argv = pipelines.build_evaluate_argv(
        "controls", _args(raw_decisions=True, subjects="S5,S6", output="r.json")
    )
    assert argv[:2] == ["--model", "qwen3:14b"]
    assert "--raw-decisions" in argv
    assert "--subjects" in argv and "S5,S6" in argv
    assert "--output" in argv and "r.json" in argv


def test_evaluate_degradation_argv():
    argv = pipelines.build_evaluate_argv(
        "degradation",
        _args(
            representation="numeric",
            conditions="C0_clean",
            split="evaluation",
            nonstress_variant="schmidt",
            temperature="exclude",
        ),
    )
    assert "--representation" in argv and "numeric" in argv
    assert "--conditions" in argv and "C0_clean" in argv
    assert "--split" in argv and "evaluation" in argv
    # El analizador debe recibir el diseño primario, o leería otro corpus.
    assert "--nonstress-variant" in argv and "schmidt" in argv
    assert "--temperature" in argv and "exclude" in argv


def test_evaluate_degradation_forwards_coherence_set():
    """El eje de relevancia de H5 se elige desde el CLI, no se fija por defecto."""
    argv = pipelines.build_evaluate_argv(
        "degradation", _args(coherence_set="compact")
    )
    assert "--coherence-set" in argv and "compact" in argv

    # Sin bandera, el analizador aplica su propio valor por defecto (primary).
    argv = pipelines.build_evaluate_argv("degradation", _args())
    assert "--coherence-set" not in argv


def test_dry_run_plan_reports_coherence_set(monkeypatch):
    monkeypatch.setattr(
        processControl, "args", _args(experiment="degradation"), raising=False
    )
    plan = pipelines._plan("evaluate")
    assert plan["coherence_set"] == "primary"
    # En evaluación la tarea también se deduce del corpus, como la variante.
    assert plan["decision_mode"] == "auto"
    monkeypatch.setattr(
        processControl,
        "args",
        _args(experiment="degradation", coherence_set="sensitivity"),
        raising=False,
    )
    assert pipelines._plan("evaluate")["coherence_set"] == "sensitivity"


def test_evaluate_controls_does_not_forward_degradation_flags():
    """En ``controls`` no hay variante ni temperatura de evidencia."""
    argv = pipelines.build_evaluate_argv(
        "controls", _args(nonstress_variant="schmidt", temperature="exclude")
    )
    assert "--nonstress-variant" not in argv
    assert "--temperature" not in argv


def test_evaluate_kemocon_is_reference_only():
    """K-EmoCon no tiene LLM propio: se evalúa la referencia features-only."""
    argv = pipelines.build_evaluate_argv("kemocon", _args(task="valence"))
    assert "--reference" in argv
    assert "--reference-only" in argv
    assert "--task" in argv and "valence" in argv


def test_compare_argv_requires_two_models():
    assert pipelines.build_compare_argv(_args(compare_models="")) is None
    processControl.args = _args(compare_models="qwen2.5:7b")
    with pytest.raises(ValueError):
        pipelines.build_compare_argv(processControl.args)


def test_compare_argv_modes(monkeypatch):
    monkeypatch.setattr(
        processControl, "args", _args(compare_models="a:1,b:1"), raising=False
    )
    argv = pipelines.build_compare_argv(processControl.args)
    assert argv[:2] == ["--models", "a:1,b:1"]
    assert "--mode" in argv and "phase2" in argv  # controls -> phase2

    monkeypatch.setattr(
        processControl,
        "args",
        _args(experiment="degradation", compare_models="a:1,b:1"),
        raising=False,
    )
    argv = pipelines.build_compare_argv(processControl.args)
    assert "degradation" in argv


def test_compare_argv_forwards_design_flags(monkeypatch):
    """El comparador debe leer el mismo corpus (variante y temperatura)."""
    monkeypatch.setattr(
        processControl,
        "args",
        _args(
            experiment="degradation",
            compare_models="a:1,b:1",
            nonstress_variant="schmidt",
            temperature="exclude",
        ),
        raising=False,
    )
    argv = pipelines.build_compare_argv(processControl.args)
    assert "--nonstress-variant" in argv and "schmidt" in argv
    assert "--temperature" in argv and "exclude" in argv


def test_compare_argv_rejects_other_experiments(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "args",
        _args(experiment="physionet", compare_models="a:1,b:1"),
        raising=False,
    )
    with pytest.raises(ValueError):
        pipelines.build_compare_argv(processControl.args)


def test_dry_run_plan_estimates_calls(monkeypatch):
    """El plan de GENERATE calcula el número de llamadas."""
    monkeypatch.setattr(pipelines, "_n_windows", lambda subject: 10)
    monkeypatch.setattr(processControl, "args", _args(subjects="S5"), raising=False)
    plan = pipelines._plan("generate")
    assert plan["experiment"] == "controls"
    assert plan["estimated_llm_calls"] == 10 * 5 * 2  # ventanas × condiciones × reps
    assert plan["runner"].endswith("run_phase2")


def test_dry_run_plan_evaluate_uses_compare_runner(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "args",
        _args(
            proc="EVALUATE",
            experiment="degradation",
            compare_models="a:1,b:1",
        ),
        raising=False,
    )
    plan = pipelines._plan("evaluate")
    assert plan["runner"] == "scripts/compare_models.py"


def test_generate_degradation_passes_binary_variant_and_temperature():
    argv = pipelines.build_generate_argv(
        "degradation",
        _args(
            decision_mode="binary",
            nonstress_variant="schmidt",
            temperature="exclude",
        ),
    )
    assert "--decision-mode" in argv and "binary" in argv
    assert "--nonstress-variant" in argv and "schmidt" in argv
    assert "--temperature" in argv and "exclude" in argv


def test_evaluate_hierarchy_argv_does_not_leak_other_flags():
    args = _args(
        proc="EVALUATE",
        experiment="degradation",
        hierarchy=True,
        bootstrap=500,
        conditions="C0_clean",
        representation="text",
        decision_mode="binary",
    )
    argv = pipelines.build_evaluate_argv("degradation", args)
    assert "--bootstrap" in argv and "500" in argv
    assert "--conditions" in argv and "C0_clean" in argv
    # El analizador de jerarquía no acepta estos flags.
    assert "--representation" not in argv
    assert "--decision-mode" not in argv


def test_dry_run_plan_evaluate_hierarchy_uses_hierarchy_runner(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "args",
        _args(proc="EVALUATE", experiment="degradation", hierarchy=True),
        raising=False,
    )
    plan = pipelines._plan("evaluate")
    assert plan["runner"] == "scripts/analyze_binary_hierarchy.py"
    assert plan["hierarchy"] is True


def test_evaluate_drift_argv_carries_primary_design():
    args = _args(
        proc="EVALUATE",
        experiment="degradation",
        drift=True,
        model="qwen3:14b",
        representation="text",
        conditions="C0_clean,C4_total_eda",
        decision_mode="binary",
        nonstress_variant="schmidt",
        temperature="exclude",
        base_condition="C0_clean",
        bootstrap=1000,
    )
    argv = pipelines.build_evaluate_argv("degradation", args)
    assert "--model" in argv and "qwen3:14b" in argv
    assert "--representation" in argv and "text" in argv
    assert "--decision-mode" in argv and "binary" in argv
    assert "--nonstress-variant" in argv and "schmidt" in argv
    assert "--temperature" in argv and "exclude" in argv
    assert "--base" in argv and "C0_clean" in argv
    assert "--bootstrap" in argv and "1000" in argv
    assert "--conditions" in argv


def test_dry_run_plan_evaluate_drift_uses_drift_runner(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "args",
        _args(proc="EVALUATE", experiment="degradation", drift=True),
        raising=False,
    )
    plan = pipelines._plan("evaluate")
    assert plan["runner"] == "scripts/analyze_drift.py"
    assert plan["drift"] is True
    assert plan["base_condition"] == "C0_clean"


def test_evaluate_refuses_drift_and_hierarchy_together(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "args",
        _args(proc="EVALUATE", experiment="degradation", drift=True, hierarchy=True),
        raising=False,
    )
    with pytest.raises(SystemExit):
        pipelines.run_evaluate()


def test_process_registry_has_three_phases():
    from main import PROCESS_REGISTRY

    assert PROCESS_REGISTRY["GENERATE"] == "sources.pipelines.run_generate"
    assert PROCESS_REGISTRY["EVALUATE"] == "sources.pipelines.run_evaluate"
    assert PROCESS_REGISTRY["INGEST"] == "sources.ingestion.pipeline.process_ingest"
    # alias deprecados
    assert PROCESS_REGISTRY["PHASE2"] == "sources.pipelines.run_generate"
    assert PROCESS_REGISTRY["PHASE3"] == "sources.pipelines.run_evaluate"


def test_plan_reports_missing_dataset(monkeypatch, tmp_path):
    """El preflight avisa si falta el dataset (fallo típico en un servidor)."""
    monkeypatch.setattr(
        processControl,
        "env",
        {
            "input": str(tmp_path / "results" / "input"),
            "output": str(tmp_path / "results" / "output"),
        },
        raising=False,
    )
    monkeypatch.setattr(processControl, "args", _args(subjects="S5"), raising=False)
    info = pipelines._paths_info("controls", processControl.args)
    assert info["dataset_exists"] is False
    assert any("no existe el dataset" in w for w in info["warnings"])


def test_plan_reports_present_dataset(monkeypatch, tmp_path):
    dataset = tmp_path / "results" / "input" / "WESAD" / "S5"
    dataset.mkdir(parents=True)
    monkeypatch.setattr(
        processControl,
        "env",
        {
            "input": str(tmp_path / "results" / "input"),
            "output": str(tmp_path / "results" / "output"),
        },
        raising=False,
    )
    monkeypatch.setattr(processControl, "args", _args(subjects="S5"), raising=False)
    info = pipelines._paths_info("controls", processControl.args)
    assert info["dataset_exists"] is True
    assert info["warnings"] == []


def test_main_shape_hint_for_missing_config():
    from main import _shape_hint

    hint = _shape_hint(FileNotFoundError("/x/config.json"))
    assert "cp config_example.json config.json" in hint


def test_main_shape_hint_for_missing_dataset():
    from main import _shape_hint

    hint = _shape_hint(
        FileNotFoundError("no existe el directorio del dataset: /x/results/input/WESAD")
    )
    assert "results/input" in hint


def test_require_config_file_fails_fast(tmp_path, monkeypatch):
    """Sin `config.json` el arranque falla con mensaje claro y código 2."""
    from sources.common import paramsManager

    messages: list[str] = []
    monkeypatch.setattr(
        paramsManager, "writeLog", lambda level, logger, message: messages.append(message)
    )
    with pytest.raises(SystemExit) as excinfo:
        paramsManager._require_config_file(str(tmp_path))
    assert excinfo.value.code == 2
    assert any("cp config_example.json config.json" in m for m in messages)


def test_require_config_file_accepts_existing(tmp_path):
    from sources.common import paramsManager

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert paramsManager._require_config_file(str(tmp_path)).endswith("config.json")


def test_external_data_paths_live_inside_project():
    """Los datos externos se resuelven dentro del repositorio."""
    from sources.external import paths

    for directory in (paths.PHYSIONET2017_DIR, paths.KEMOCON_DIR):
        assert paths.REPO_ROOT in directory.parents
        assert "EXTERNAL" in directory.parts


def test_external_data_dir_from_config(monkeypatch):
    """`params.external.<exp>.data_dir` sobreescribe la ruta por defecto."""
    monkeypatch.setattr(
        processControl,
        "parms",
        {"external": {"physionet": {"data_dir": "/datos/pn17"}}},
        raising=False,
    )
    assert pipelines.external_data_dir("physionet") == "/datos/pn17"
    assert pipelines.external_data_dir("kemocon") is None


def test_generate_argv_passes_configured_data_dir(monkeypatch):
    monkeypatch.setattr(
        processControl,
        "parms",
        {"external": {"physionet": {"data_dir": "/datos/pn17"}}},
        raising=False,
    )
    argv = pipelines.build_generate_argv("physionet", _args())
    assert "--data-dir" in argv
    assert "/datos/pn17" in argv


def test_plan_warns_when_external_data_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        processControl,
        "env",
        {"input": str(tmp_path / "in"), "output": str(tmp_path / "out")},
        raising=False,
    )
    monkeypatch.setattr(
        processControl,
        "parms",
        {"external": {"physionet": {"data_dir": str(tmp_path / "nada")}}},
        raising=False,
    )
    monkeypatch.setattr(processControl, "args", _args(experiment="physionet"), raising=False)
    info = pipelines._paths_info("physionet", processControl.args)
    assert info["external_data_exists"] is False
    assert any("dato externo" in w for w in info["warnings"])


def test_plan_incluye_informe_de_gpu(monkeypatch):
    """El plan de ``--dry-run`` informa del hardware (sin confundir 'sin GPU')."""
    from sources.common import paramsManager

    monkeypatch.setattr(
        paramsManager,
        "_torch_gpus",
        lambda: [{"index": 0, "name": "RTX 4060 Ti", "memory_gb": 8.0}],
    )
    monkeypatch.setattr(paramsManager, "_detect_all_gpus_nvidia_smi", lambda: [])
    info = pipelines._paths_info("controls", processControl.args)
    assert info["gpu"]["usable"] == "torch"
    assert info["gpu"]["device"] == "cuda"
    assert info["gpu"]["mismatch"] is False
    assert info["gpu"]["devices"][0]["name"] == "RTX 4060 Ti"


def test_plan_avisa_si_la_gpu_no_es_usable_por_torch(monkeypatch):
    from sources.common import paramsManager

    monkeypatch.setattr(paramsManager, "_torch_gpus", lambda: [])
    monkeypatch.setattr(
        paramsManager,
        "_detect_all_gpus_nvidia_smi",
        lambda: [{"index": 0, "name": "A5000", "memory_gb": 24.0, "driver": "570.1"}],
    )
    info = pipelines._paths_info("controls", processControl.args)
    assert info["gpu"]["mismatch"] is True
    assert info["gpu"]["driver"] == "570.1"
    assert any("requirements.txt" in warning for warning in info["warnings"])


def test_generate_degradation_passes_decision_mode():
    """El modo binario debe llegar al barrido (si no, se mediría otra tarea)."""
    argv = pipelines.build_generate_argv(
        "degradation", _args(decision_mode="binary")
    )
    assert "--decision-mode" in argv and "binary" in argv


def test_evaluate_degradation_passes_decision_mode():
    argv = pipelines.build_evaluate_argv(
        "degradation", _args(decision_mode="binary")
    )
    assert "--decision-mode" in argv and "binary" in argv


def test_generate_argv_forwards_compare_with_for_h3():
    """El contraste H3 (texto − numérico) necesita el directorio del otro brazo."""
    from types import SimpleNamespace

    from sources.pipelines import build_evaluate_argv

    args = SimpleNamespace(
        model="qwen3:14b",
        representation=None,
        conditions=None,
        decision_mode="binary",
        nonstress_variant="schmidt",
        temperature="exclude",
        hierarchy=False,
        repetitions=None,
        compare_with="results/output/WESAD/PHASE2/degradation_text_binary_schmidt_notemp/qwen3_14b",
        output=None,
        seed=None,
    )
    argv = build_evaluate_argv("degradation", args)
    assert "--compare-with" in argv
    assert argv[argv.index("--compare-with") + 1].endswith("qwen3_14b")


def test_generate_argv_omits_compare_with_by_default():
    from types import SimpleNamespace

    from sources.pipelines import build_evaluate_argv

    args = SimpleNamespace(
        model="qwen3:14b",
        representation=None,
        conditions=None,
        decision_mode="binary",
        nonstress_variant="schmidt",
        temperature="exclude",
        hierarchy=False,
        repetitions=None,
        compare_with=None,
        output=None,
        seed=None,
    )
    assert "--compare-with" not in build_evaluate_argv("degradation", args)


def test_system_exit_with_message_keeps_it_visible(caplog):
    """`raise SystemExit("texto")` no puede quedarse en un 1 mudo.

    Todos los runners explican los errores controlados con SystemExit y una
    cadena; si el mensaje se descarta, el usuario solo ve un código.
    """
    from main import _exit_code_from_system_exit

    assert _exit_code_from_system_exit(SystemExit(0)) == 0
    assert _exit_code_from_system_exit(SystemExit(None)) == 0
    assert _exit_code_from_system_exit(SystemExit(2)) == 2
    with caplog.at_level("ERROR"):
        assert _exit_code_from_system_exit(SystemExit("motivo legible")) == 1
    assert "motivo legible" in caplog.text


# ---------------------------------------------------------------------------
# Cohorte de escala de DREAMT
# ---------------------------------------------------------------------------


def test_dreamt_argv_passes_the_cohort():
    """``--cohort`` debe llegar al runner en las dos fases (pilot/scale)."""
    gen = pipelines.build_generate_argv(
        "dreamt", _args(cohort="scale", representation="both", windows_per_class=10)
    )
    assert "--cohort" in gen and "scale" in gen

    processControl.defaults = {"llm": {"primary_model": "qwen2.5:7b"}}
    ev = pipelines.build_evaluate_argv("dreamt", _args(cohort="scale", model=None))
    assert "--cohort" in ev and "scale" in ev


def test_dreamt_plan_uses_the_requested_cohort():
    """El plan de ``--dry-run`` no debe contar los sujetos del piloto si se pide escala."""
    from scripts.run_p4_dreamt import PILOT_SUBJECTS, SCALE_SUBJECTS

    assert pipelines._subjects_for_plan(_args(experiment="dreamt")) == list(
        PILOT_SUBJECTS
    )
    assert pipelines._subjects_for_plan(
        _args(experiment="dreamt", cohort="scale")
    ) == list(SCALE_SUBJECTS)
    assert pipelines._subjects_for_plan(
        _args(experiment="dreamt", cohort="scale", subjects="S008")
    ) == ["S008"]


def test_generate_argv_forwards_engine_change_opt_in():
    """La guardia de motor aborta por defecto; el opt-in debe poder pasarse."""
    argv = pipelines.build_generate_argv("degradation", _args())
    assert "--allow-engine-change" not in argv
    argv = pipelines.build_generate_argv(
        "degradation", _args(allow_engine_change=True)
    )
    assert "--allow-engine-change" in argv
