"""Script piloto Fase 2 — Ejecución sobre S3 (sujeto piloto).

Ejecuta los pasos faltantes del pipeline para S3, genera vistas LLM y
luego ejecuta el orquestador factorial.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from sources.common.common import processControl
from sources.common.paramsManager import getConfigs
from sources.common.paths import read_jsonl
from sources.ingestion import discover, loader, qc, window, features, represent
from sources.ingestion.config import get_ingestion_config
from sources.phase2.llm_config import LlmExperimentConfig, build_experiment_config
from sources.phase2.negative_controls import (
    NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT,
)
from sources.phase2.orchestrator import (
    run_orchestrator,
    REPRESENTATION_NUMERIC,
    REPRESENTATION_TEXT,
    PROMPT_VARIANT_BASE,
)

SUBJECT_PILOT = "S3"


def setup_environment():
    """Configura processControl.env y args para usar rutas de WESAD/INGEST."""
    env = {
        "realPath": "/home/vant/Documentos/proyectos/LSI/ACL2027",
        "results_root": "",
        "models": "/home/vant/Documentos/proyectos/LSI/ACL2027/models",
        "input": "/home/vant/Documentos/proyectos/LSI/ACL2027/results/input",
        "output": "/home/vant/Documentos/proyectos/LSI/ACL2027/results/output",
        "logs": "/home/vant/Documentos/proyectos/LSI/ACL2027/",
        "cache": "/home/vant/Documentos/proyectos/LSI/ACL2027/.pycache",
    }
    processControl.env = env
    processControl.args = SimpleNamespace(
        subject="WESAD",
        proc="INGEST",
        step=None,
        serverGPU=None,
        subjects=None,
        force=False,
        dry_run=False,
    )
    getConfigs()
    processControl.args.subject = "WESAD"
    processControl.args.proc = "INGEST"


def main():
    print("=" * 70)
    print("PILOTO FASE 2 — Sujeto S3")
    print("=" * 70)

    setup_environment()

    # 1. Cargar config y habilitar vistas LLM
    cfg = get_ingestion_config()
    cfg = copy.deepcopy(cfg)
    for kind in ("numeric", "text"):
        llm_opts = cfg["representations"][kind].setdefault("llm_view", {})
        llm_opts["enabled"] = True
        llm_opts["exclude_motion_channels"] = True
        llm_opts["opaque_window_id"] = True

    # 2. Ejecutar pasos previos para S3 (idempotentes por fingerprint)
    print(f"\n[1/6] Ejecutando pipeline previo para {SUBJECT_PILOT}...")

    # Ingest (reutiliza canónico si existe)
    print("  [ingest] ", end="", flush=True)
    manifest = discover.run_discover(cfg)
    entry = discover.selected_entries(manifest)[0]
    result = loader.run_ingest(cfg, entry)
    print(f"{result['status']}")

    # QC
    print("  [qc] ", end="", flush=True)
    qc_result = qc.run_qc(cfg, SUBJECT_PILOT)
    print(f"ok")

    # Window
    print("  [window] ", end="", flush=True)
    window_result = window.run_window(cfg, SUBJECT_PILOT)
    print(f"ok ({window_result.get('n_windows', 0)} ventanas)")

    # Features
    print("  [features] ", end="", flush=True)
    features_result = features.run_features(cfg, SUBJECT_PILOT)
    print(f"ok")

    # 3. Generar vistas LLM para S3
    print(f"\n[2/6] Generando vistas LLM para {SUBJECT_PILOT}...")
    rep_result = represent.run_represent(cfg, SUBJECT_PILOT, force=True)
    print(f"   Estado: {rep_result['status']}")
    print(f"   Ventanas: {rep_result['n_windows']}")

    numeric_llm_path = represent.llm_representation_path("numeric", SUBJECT_PILOT)
    text_llm_path = represent.llm_representation_path("text", SUBJECT_PILOT)
    print(f"   numeric_llm: {'OK' if numeric_llm_path.is_file() else 'FALTA'}")
    print(f"   text_llm: {'OK' if text_llm_path.is_file() else 'FALTA'}")

    # 4. Contar ventanas para presupuesto
    window_rows = read_jsonl(window.windows_path(SUBJECT_PILOT))
    n_windows = len(window_rows)

    # 5. Configuración experimental
    print(f"\n[3/6] Configuración experimental...")
    config = build_experiment_config(
        model="qwen3:8b",
        temperature=0.3,
        seed=42,
        max_retries=3,
        cascade_enabled=False,
    )
    print(f"   Modelo: {config.model}")
    print(f"   Semilla: {config.seed}")
    print(f"   Reintentos: {config.max_retries}")
    print(f"   Cascada: {config.cascade_enabled}")
    print(f"   num_ctx: {config.num_ctx}")

    # 6. Presupuesto (solo variante base para piloto inicial)
    print(f"\n[4/6] Presupuesto estimado para {SUBJECT_PILOT}...")
    print(f"   Ventanas: {n_windows}")
    print(f"   Representaciones: numeric_llm, text_llm (2)")
    print(f"   Variantes de prompt: base (1, piloto exploratorio)")
    print(f"   Condiciones: C0_clean + 4 controles negativos (5)")
    print(f"   Repeticiones: 1")
    total_calls = n_windows * 2 * 1 * 5 * 1
    print(f"   Total de llamadas LLM estimadas: {total_calls}")
    print(f"   Tokens estimados por llamada (promedio): ~300-500")
    print(f"   Tokens totales estimados: ~{total_calls * 400:,} - {total_calls * 500:,}")

    # 7. Ejecutar orquestador (piloto exploratorio)
    print(f"\n[5/6] Ejecutando orquestador sobre {SUBJECT_PILOT}...")
    print(f"   Variante: base | Condiciones: C0_clean + 4 controles negativos")

    # Depuración: verificar args antes del orquestador
    from sources.common.paths import corpus_dir
    print(f"   DEBUG args.subject: {processControl.args.subject}")
    print(f"   DEBUG args.proc: {processControl.args.proc}")
    print(f"   DEBUG corpus_dir(): {corpus_dir()}")
    print(f"   DEBUG llm_rep path: {represent.llm_representation_path('numeric', SUBJECT_PILOT)}")
    print(f"   DEBUG llm_rep exists: {represent.llm_representation_path('numeric', SUBJECT_PILOT).is_file()}")

    # Asegurar args correctos (get_ingestion_config puede haberlos modificado)
    processControl.args.subject = "WESAD"
    processControl.args.proc = "INGEST"

    summary = run_orchestrator(
        subjects=[SUBJECT_PILOT],
        config=config,
        representations=[REPRESENTATION_NUMERIC, REPRESENTATION_TEXT],
        prompt_variants=[PROMPT_VARIANT_BASE],
        repetitions=1,
        seeds=[42],
        condition_ids=["C0_clean"],
        negative_controls=[NC_SHUFFLED, NC_ZERO, NC_NAN, NC_ABSENT],
    )

    # 8. Resultados
    print(f"\n[6/6] Resultados")
    print(f"{'=' * 70}")
    subj = summary["subjects"].get(SUBJECT_PILOT, {})
    print(f"   Sujeto: {SUBJECT_PILOT}")
    print(f"   Entradas en manifiesto: {subj.get('n_entries', 0)}")
    print(f"   Llamadas LLM realizadas: {subj.get('n_calls', 0)}")
    print(f"   Caché reutilizada: {subj.get('n_cached', 0)}")
    print(f"   Errores: {subj.get('n_errors', 0)}")
    print(f"   Total llamadas: {summary['total_calls']}")
    print(f"   Total errores: {summary['total_errors']}")

    # Mostrar respuestas de ejemplo
    responses_path = (
        Path(processControl.env["output"])
        / SUBJECT_PILOT
        / "INGEST"
        / "llm_responses"
        / f"{SUBJECT_PILOT}.responses.jsonl"
    )
    if responses_path.is_file():
        with responses_path.open() as fh:
            lines = fh.readlines()
        print(f"\n   Respuestas guardadas en: {responses_path}")
        print(f"   Número de respuestas: {len(lines)}")
        for i, line in enumerate(lines[:5]):
            row = json.loads(line)
            print(f"\n   Respuesta {i+1}:")
            print(f"     condition_id: {row.get('condition_id')}")
            print(f"     status: {row.get('status')}")
            print(f"     decision: {row.get('decision')}")
            print(f"     confidence: {row.get('confidence')}")
            print(f"     reasoning: {row.get('reasoning', '')[:120]}...")

    # Análisis de controles negativos
    print(f"\n{'=' * 70}")
    print("ANÁLISIS DE CONTROLES NEGATIVOS")
    print(f"{'=' * 70}")
    if responses_path.is_file():
        abstention_rate = 0
        nc_total = 0
        for line in lines:
            row = json.loads(line)
            if row.get("condition_id", "").startswith("NC_"):
                nc_total += 1
                if row.get("decision") == "abstain":
                    abstention_rate += 1
        if nc_total > 0:
            print(f"   Controles negativos totales: {nc_total}")
            print(f"   Abstenciones en controles negativos: {abstention_rate}")
            print(f"   Tasa de abstención: {abstention_rate / nc_total * 100:.1f}%")
            if abstention_rate / nc_total >= 0.8:
                print("   ✅ CRITERIO CUMPLIDO: abstención > 80% en controles negativos")
            else:
                print("   ⚠️  CRITERIO NO CUMPLIDO: abstención <= 80% en controles negativos")

    print(f"\n{'=' * 70}")
    print("PILOTO COMPLETADO")
    print(f"{'=' * 70}")
    print("\nCriterios de paso para experimento completo:")
    print("  - Tasa de valid_abstention en controles negativos > 80%")
    print("  - Tasa de invalid_output < 5%")
    print("  - Tasa de timeout < 5%")
    print("  - Consistencia entre repeticiones (temperatura > 0) > 70%")


if __name__ == "__main__":
    main()
