# Fase 2: Prompt builder y orquestador — correcciones de diseño

> **Fecha:** 2026-11-07. Este documento describe las correcciones de diseño de
> la Fase 2 aplicadas tras la validación LLM del corpus completo (15 sujetos).
> Cubre (1) la recuperación del módulo `orchestrator.py`, (2) la unificación de
> canales entre `text_llm` y `numeric_llm`, (3) la corrección del few-shot
> contradictorio con los datos reales, (4) las repeticiones con semillas
> distintas y (5) la implementación de `max_retries`.

---

## 1. Recuperación de `sources/phase2/orchestrator.py`

El módulo `orchestrator.py` (orquestador factorial de llamadas LLM) se perdió:
nunca se había commiteado a git y su archivo fuente fue eliminado del disco.
Solo quedaba su bytecode compilado
(`sources/phase2/__pycache__/orchestrator.cpython-314.pyc`).

Se reconstruyó el código fuente íntegro (12 funciones + constantes + imports) a
partir de:

- el bytecode (desensamblado con `dis`/`marshal`),
- el test file `tests/test_orchestrator.py` (676 líneas, con la API completa),
- las secciones 14–17 de `AGENTS.md` (comportamiento documentado).

**Verificaciones de fidelidad:**

- `_prompt_hash` reproduce **460/460** hashes de la caché real generada por el
  módulo original (misma salida, byte a byte).
- Los 19 tests del orquestador pasan (hash, manifiesto, caché, `force`,
  controles negativos).

> Lección: cualquier archivo de código nuevo debe commiteares a git **en la
> misma sesión** en que se crea; el `.pyc` en `__pycache__` no es un respaldo
> fiable y además está gitignorado.

---

## 2. Unificación de canales `text_llm` / `numeric_llm`

**Problema.** `build_text_prompt()` usaba la representación textual cruda
(`text_llm`), que incluía las modalidades de muñeca (`wrist_eda`, `wrist_temp`,
`wrist_acc`), mientras que `build_numeric_prompt()` filtraba a pecho
(`ecg, eda, resp, temp, bvp`). Así, `text_llm` exponía más canales que
`numeric_llm`, rompiendo la comparabilidad entre representaciones.

**Corrección** (`sources/ingestion/represent.py`):

- Nueva función `_llm_view_features(feature_row, exclude_motion_channels)` que
  filtra a nivel de **modalidad** (no de feature).
- Constantes `LLM_CHEST_MODALITIES = {ecg, eda, resp, temp, bvp}`,
  `LLM_MOTION_MODALITIES = {acc, emg}` y
  `WRIST_MODALITIES = {wrist_eda, wrist_temp, wrist_acc}`.
- La vista LLM (tanto `numeric_llm` como `text_llm`, y también las vistas
  `*_llm_motion`) excluye siempre las modalidades de muñeca; los canales de
  movimiento (`acc`/`emg`) se controlan con `exclude_motion_channels`.

**Resultado.** `numeric_llm` y `text_llm` comparten el mismo conjunto de
canales (pecho + movimiento opcional), diferiendo solo en el formato (numérico
vs textual).

---

## 3. Few-shot corregido con los datos reales (SCR)

**Problema.** Los ejemplos few-shot usaban valores de SCR inflados (baseline=2,
stress=8, amusement=5, meditation=1) heredados del umbral SCR antiguo (0.01 µS),
y describían un "amusement" elevado (HR=85, SCR=5) que no existe en los datos.
Esto sesgaba las prioridades del modelo.

**Medias reales del corpus** (15 sujetos, umbral SCR 0.05 µS):

| feature | baseline | stress | amusement | meditation |
|---|---|---|---|---|
| `ecg_hr_bpm` | 73.0 | 98.3 | 73.0 | 69.5 |
| `ecg_rmssd_ms` | 50.5 | 43.9 | 52.3 | 68.4 |
| `eda_mean_us` | 2.56 | 4.59 | 2.89 | 2.99 |
| `eda_scr_count` | 0.67 | 5.10 | 0.54 | 0.16 |

**Corrección** (`sources/phase2/prompt_builder.py`, `FEW_SHOT_EXAMPLES`):

- baseline: HR 73, SCR 1 · stress: HR 98, SCR 5 · amusement: HR 73, SCR 1 ·
  meditation: HR 69, SCR 0.
- El ejemplo de "amusement" ahora declara honestamente que es casi idéntico a
  baseline con EDA+HR (no hay respuesta fisiológica diferenciable).

> Dato clave: `amusement` es **fisiológicamente indistinguible de baseline**
> con EDA+HR (HR 73 vs 73, SCR 0.54 vs 0.67). Esto explica su AUROC ≈ 0.5 y
> refuerza la recomendación de clasificación binaria stress vs no-stress.

---

## 4. Repeticiones con semillas distintas

**Problema.** `_call_ollama_chat()` usaba siempre `config.seed` en
`options.seed`, y `run_orchestrator()` calculaba el hash de caché con
`config.seed`. Con varias repeticiones, todas colapsaban a la misma semilla (y
a la misma respuesta).

**Corrección** (`sources/phase2/orchestrator.py`):

- `_call_ollama_chat(client, prompt, config, seed=None)` acepta la semilla por
  repetición y la usa en `options.seed` (fallback a `config.seed`).
- `run_orchestrator()` pasa `seed=seed` (la semilla de la entrada del
  manifiesto) y usa `_config_for_hash(config, seed)` para el hash de caché y de
  resultado (antes usaba `config.seed`).
- `_config_for_hash(config, seed)` devuelve una copia de la configuración con
  la semilla de la repetición, asegurando que dos semillas distintas no
  colisionen en caché.
- `run_phase2()` lee `repetitions` y `seeds` de `config.json > defaults.llm`
  (por defecto `repetitions=1`, `seeds=[42]`).

---

## 5. `max_retries` (reintentos)

**Problema.** `_call_ollama_chat()` hacía una única llamada HTTP; ante un error
transitorio (Ollama caído, JSON mal formado) no reintentaba.

**Corrección** (`sources/phase2/orchestrator.py`): la llamada se envuelve en un
bucle de hasta `max_retries + 1` intentos con backoff progresivo
(2s, 8s, 18s, …). Si todos fallan, lanza `ValidationError` (registrada como
`status=timeout` en la respuesta). El número de reintentos se toma de
`client["max_retries"]` (o `config.max_retries`), configurable vía
`defaults.llm.max_retries` (ahora `3` en `config_example.json`).

---

## 6. Configuración y tests

- `config_example.json` y `config.json`: `defaults.llm.max_retries` 1 → 3;
  añadidos `defaults.llm.repetitions` (1) y `defaults.llm.seeds` ([42]).
- Tests nuevos (total **120/120**):
  - `test_represent.py`: exclusión de muñeca en la vista LLM.
  - `test_prompt_leakage.py`: few-shot con SCR reales y sin fugas.
  - `test_orchestrator.py`: `_config_for_hash`, semilla por repetición,
    reintentos y agotamiento de reintentos.
- Script de evaluación agregada: `scripts/evaluate_phase3.py` (reporte en
  `results/output/WESAD/PHASE2/evaluation_report.json`).
