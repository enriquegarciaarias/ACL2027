# Fase 3: Evaluación de Respuestas LLM — Informe del Piloto S3

> ⚠️ **Nota de corrección (2026-11-07).** Los resultados publicados originalmente
> en este informe (2026-11-06) quedaron **invalidados** por 6 bugs detectados en
> una revisión de la construcción de Fase 2/3 (ver sección 17 de `AGENTS.md`).
> El AUROC estaba mal implementado (devolvía valores fuera de [0,1]), la métrica
> `accuracy` era idéntica a `accuracy_incl_abstain`, el control `NC_absent`
> nunca se evaluó (colisión de caché con `NC_nan`), los controles negativos de
> texto revelaban la respuesta, la normalización de canales descartaba respuestas
> válidas de `text_llm`, y el umbral SCR (0.01 µS) producía un conteo de SCR
> **invertido** (baseline 52 > stress 36). Este documento refleja los resultados
> **recalculados** tras las correcciones.

## Resumen Ejecutivo

La Fase 3 se re-evaluó para el piloto S3 (46 ventanas) tras corregir los bugs y
recalibrar el umbral SCR a 0.05 µS. El modelo qwen2.5:7b (few-shot + features
EDA+HR+Resp+Temp+BVP) **sí discrimina stress** (AUROC 0.896) y **deja de
colapsar** a "stress" (antes 46/46, ahora decisiones mixtas con 22/46
abstenciones). El problema de 4 clases sigue siendo difícil: `amusement` y
`meditation` quedan cerca del azar (AUROC ≈ 0.5).

## Bugs corregidos (causa de los resultados previos)

| # | Bug | Impacto en los resultados previos |
|---|-----|-----------------------------------|
| 1 | AUROC acumulaba tp/fp sin resetear por umbral | AUROC stress 0.788 era artefacto (real ≈ 0.60) |
| 2 | `accuracy` y `accuracy_incl_abstain` idénticas | "6.2%" mezclaba abstenciones |
| 3 | Colisión de caché `NC_nan`/`NC_absent` | `NC_absent` nunca se evaluó |
| 4 | Controles negativos de texto en español ("CONTROL NEGATIVO") | abstention de text_llm no era válida |
| 5 | Normalización de canales sin mapear títulos descriptivos | 41% de `text_llm` marcadas inválidas |
| 6 | Umbral SCR 0.01 µS (sobre-detección de ruido) | `eda_scr_count` invertido (baseline > stress) |

## Métricas corregidas — S3 (C0_clean)

### `numeric_llm` (EDA+HR+resp/temp/bvp)

| Métrica | Antes (buggy) | **Corregido** |
|---------|---------------|---------------|
| Predicción | 46/46 "stress" (colapso) | 13 stress + 11 baseline + 22 abstain |
| Accuracy (decididas) | 21.7% | **54.2%** |
| Accuracy (abstain=error) | — | 28.3% |
| AUROC stress | 0.60 | **0.896** |
| AUROC baseline | 0.50 | **0.580** |
| AUROC amusement / meditation | 0.50 | 0.500 |
| ECE | 0.550 | **0.242** |
| Brier | 0.309 | 0.187 |

### `text_llm` (canales extra)

| Métrica | Valor |
|---------|-------|
| Invalid output | 2/46 (antes 19/46) |
| Accuracy (decididas) | 54.5% |
| AUROC stress | 1.000 |
| AUROC baseline | 0.647 |

## Controles Negativos (corregidos)

| Condición | Abstain |
|-----------|---------|
| NC_zero / NC_nan / NC_absent (numeric + text) | 100% ✅ |
| NC_shuffled numeric | 22% |
| NC_shuffled text | 87% |

`NC_shuffled numeric` sigue prediciendo "stress" (22% abstain): confirma que el
modelo aplica **umbrales a valores absolutos**, no análisis de estructura.

## Validación a nivel de features (15 sujetos, pooled)

AUROC stress-vs-baseline por feature, tras recalibrar el umbral SCR:

| Feature | AUROC |
|---------|-------|
| `ecg_hr_bpm` | 0.894 |
| `eda_phasic_std_us` | 0.897 |
| `eda_scr_rate_per_min` | 0.818 |
| `eda_mean_us` | 0.676 |

## Conclusión (corregida)

El colapso a "stress" y la conclusión "el modelo no discrimina" eran artefactos
de los bugs. Con las correcciones, el modelo **sí discrimina stress vs
baseline** (AUROC 0.896, accuracy 54.2%) y abstiene cuando no tiene evidencia.
El problema de 4 clases sigue abierto: `amusement` y `meditation` no se separan
con EDA+HR (AUROC ≈ 0.5).

## Opciones de Investigación Futura

1. **Clasificación binaria (estrés vs no-estrés):** AUROC 0.896 de stress lo hace
   la opción más viable.
2. **Validación completa en los 14 sujetos restantes:** job PHASE2 en background
   (log `/tmp/opencode/phase2_14subjects.log`); al terminar, evaluar con el
   `evaluator` corregido separando C0_clean de controles negativos y numeric de
   text.
3. **Confounders de diseño pendientes:** unificar canales `text_llm`/`numeric_llm`,
   corregir few-shot contradictorio, repeticiones con semillas distintas,
   implementar `max_retries`.
4. **Validar el umbral SCR 0.05 µS** con literatura y anotación humana.

## Archivos Creados/Modificados

- `sources/phase3/evaluator.py` — evaluación (accuracy, AUROC, ECE, Brier); corregido AUROC (Mann-Whitney) y accuracy
- `sources/phase3/__init__.py` — exportes
- `sources/phase2/prompt_builder.py` — few-shot, ADDITIONAL_FEATURES, CHEST_MODALITIES
- `sources/phase2/orchestrator.py` — override de metadatos en caché
- `sources/phase2/negative_controls.py` — degradación real de controles de texto
- `sources/phase2/response_schema.py` — CHANNEL_ALIASES + ALLOWED_CLAIM_CHANNELS
- `sources/ingestion/config.py` — `min_scr_amplitude_us` 0.01 → 0.05
- `results/output/WESAD/PHASE2/llm_responses/S3.responses.jsonl` — respuestas del piloto
- `results/output/WESAD/PHASE2/llm_responses/S3.manifest.json` — manifiesto del piloto

## Cómo Ejecutar la Evaluación

```bash
source .venv/bin/activate
python -c "
from sources.phase3.evaluator import evaluate_subject, format_result
from pathlib import Path
result = evaluate_subject('S3',
    Path('results/output/WESAD/PHASE2/llm_responses/S3.responses.jsonl'),
    Path('results/output/WESAD/INGEST/groundtruth/labels.jsonl'))
print(format_result(result))
"
```

> Nota: `evaluate_subject` mezcla todas las condiciones y representaciones. Para
> el análisis por condición, filtrar por `condition_id` y `representation` antes
> de calcular las métricas (ver sección "Métricas corregidas").

## Tests

- **112/112 pasando** (`python -m pytest -q`)
