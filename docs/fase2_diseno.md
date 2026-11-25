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
- Constantes `LLM_CHEST_MODALITIES = {ecg, eda, resp, temp}`,
  `LLM_MOTION_MODALITIES = {acc, emg}` y
  `WRIST_MODALITIES = {bvp, wrist_eda, wrist_temp, wrist_acc}`.
  **Corrección 2026-11-08:** `bvp` se extrae de `wrist_BVP` (ver
  `features/registry.py`) y es, por tanto, una modalidad de muñeca; antes se
  había clasificado erróneamente como pecho. Se reserva para el experimento
  de sobre-confianza en medidas derivadas (HR de BVP vs ECG de referencia),
  no para la vista principal.
- La vista LLM (tanto `numeric_llm` como `text_llm`, y también las vistas
  `*_llm_motion`) excluye siempre las modalidades de muñeca; los canales de
  movimiento (`acc`/`emg`) se controlan con `exclude_motion_channels`.

**Resultado.** `numeric_llm` y `text_llm` comparten el mismo conjunto de
canales (pecho + movimiento opcional) y el mismo conjunto de features
(`LLM_ALLOWED_FEATURES`, 23 features: 5 ECG, 9 EDA, 5 resp, 4 temp),
diferiendo solo en el formato (numérico vs textual). `representation_version`
se incrementó a `1.3` para invalidar las vistas antiguas. Se excluyó
`eda_scr_candidates` de la vista LLM (es un diagnóstico interno: se confunde
con el número real de SCR y no aporta al modelo).

**Few-shot (2026-11-08).** Los ejemplos usan ahora medias reales de las 705
ventanas e incluyen `resp_rate_bpm`, que es la señal que mejor separa
`meditation` (≈9 rpm) del resto (≈14–17 rpm). Se omiten deliberadamente
`temp` (deriva con el orden del protocolo WESAD y puede actuar como confusor)
y `bvp` (muñeca).

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

---

## 7. Compuerta de consistencia (P1)

La revisión de E12 mostró que la vista centrada pierde sensibilidad a la
corrupción estructural (`NC_shuffled`: 23,6 % de abstención frente a 95,3 % de
la vista absoluta). Se añadió una **compuerta determinista de coherencia
fisiológica** antes de aceptar la decisión del LLM:

- `sources/phase2/consistency.py`: 8 comprobaciones (identidades HR↔RR,
  latidos↔HR, SCR↔tasa, amplitud media ≤ máxima, EDA media↔tónica; empíricas
  ciclos↔FR, rango↔DE, DE↔DE fásica), severidad `hard`/`soft` y
  `should_abstain`.
- Integración en `run_orchestrator` (`consistency_gate=True` por defecto):
  cada respuesta guarda `consistency.*`, conserva `decision_llm` y usa
  `status="gate_abstention"` si la compuerta anula una decisión concreta.
- La caché guarda la respuesta cruda del LLM: el veredicto se recalcula siempre.
- Ablación: `--noGate` o `defaults.llm.consistency_gate=false`.
- Resultados: ver `docs/consistencia.md` (0 % de falsos positivos en limpio,
  100 % de detección en `NC_shuffled`, 128 vetos en el corpus).

---

## 8. Selección de modelo y aislamiento por modelo

Comparar modelos es una **decisión experimental explícita**, no una cascada de
fallback: el modo experimento sigue usando un solo modelo por ejecución y sin
cascada (`cascade_enabled=False`).

- **Selección**: `--model` en CLI (`python main.py --proc GENERATE
  --experiment controls ... --model qwen3:14b`; `PHASE2` sigue como alias). Sin flag se usa `defaults.llm.primary_model`
  (`qwen2.5:7b`, el modelo que generó el corpus publicado). Los scripts de
  experimento (`run_degradation_sweep.py`, `run_e12_pilot.py`,
  `run_p3_physionet.py`, `run_p3_kemocon.py`) aceptan el mismo flag.
- **Aislamiento**: cada modelo escribe en su propio subdirectorio con un slug
  (`qwen3:14b` → `qwen3_14b`):
  - `llm_responses/{modelo}/{sujeto}.responses.jsonl`
  - `degradation/{modelo}/` y `degradation_text/{modelo}/`
  - la caché de prompts también es por modelo.
  Una segunda ejecución no sobrescribe ni reutiliza por error los artefactos de
  otro modelo; las ejecuciones antiguas quedan en las carpetas históricas y los
  analizadores las leen con un aviso.
- **Procedencia**: cada registro incluye `model` a nivel superior (además de
  `ollama_meta.model`, que es el modelo que responde realmente), y los resúmenes
  incorporan `model`, `model_slug` y `config`.
- **Modo *thinking***: `defaults.llm.think` (por defecto `false`) y `--think`
  para activarlo. El parámetro `think` **solo** se envía a la familia `qwen3`
  (`supports_thinking`), porque es la que lo documenta; en otros modelos se
  omite para no romper la llamada. Desactivarlo mantiene comparable el
  presupuesto de tokens (`max_tokens`) entre modelos de razonamiento y modelos
  clásicos. El valor enviado se registra en `ollama_meta.think_requested`.
- **Comparación**: `python main.py --proc EVALUATE --experiment
  degradation --compare-models qwen2.5:7b,qwen3:14b` (o el script
  `scripts/compare_models.py` directamente) produce una tabla por condición (abstención, accuracy,
  balanced, ECE, respaldo de claims, certeza) y la **diferencia de accuracy con
  bootstrap agrupado por sujeto** (IC 95 %). El primer modelo es la referencia.

---

## 9. Compatibilidad de modelos nuevos: `cniongolo/biomistral` (verificado 2026-11-23)

Antes de añadir un modelo a la matriz hay que comprobar que **emite el JSON del
esquema** y que **reacciona a la evidencia**. Se verificó con el pipeline real
(prompt de una ventana del piloto DREAMT, `validate_llm_output` con
`decision_mode="sleep4"` y `SLEEP_CLAIM_CHANNELS`).

**Disponible**: `cniongolo/biomistral:latest` (ID `e0733a4eee80`), 7B, Q4_K_M,
4,4 GB, arquitectura `llama`, contexto 32768. El CLI ya lo acepta
(`--model cniongolo/biomistral:latest` → slug `cniongolo_biomistral_latest`,
artefactos aislados, `--dry-run` estima 960 llamadas).

**No es un sustituto directo de Qwen**, por tres motivos medidos:

| Comprobación | Resultado |
|---|---|
| `/api/chat` sin `format` (ruta actual del pipeline) | devuelve prosa (`"The input is a 60-second window…"`) → `invalid_output` |
| Plantilla cargada | ChatML genérica (`<|im_start|>`); su formato nativo es Mistral `[INST] … [/INST]` |
| `[INST]` + `format` (JSON schema) | JSON válido, pero **constante**: `rem`/80 en las 4 clases y en `C1`/`C4`; 1/4 aciertos (azar 1/4), **0 abstenciones** |
| ChatML + `format` | JSON válido, constante `wake`/`rem` a 90; 0/4 aciertos, **0 abstenciones** |
| Velocidad con `format` | 135–174 tokens en 2,5–3,4 s (≈55 tok/s) |
| `num_ctx` 8192 | 100 % GPU (4,7 GB) |
| `num_ctx` 32768 (el valor global de `config.json`) | 8,9 GB → **25 %/75 % CPU/GPU**, inutilizable en la RTX de 8 GB |

Interpretación: es un modelo de **continuación** (BioMistral sobre PMC) con
ajuste a instrucciones muy débil; no sigue el esquema en prosa y, cuando se le
fuerza con decodificación restringida, responde con una etiqueta fija sin mirar
las features. En esa situación **H1/H4/H5 no son medibles**: no hay degradación
que separe condiciones porque la respuesta no depende de la entrada.

**Qué haría falta para integrarlo** (si se decide usarlo como control):

1. `format` (JSON schema) en el cliente de Ollama: hoy `_call_ollama_chat` no lo
   envía. Es un parámetro neutro para Qwen y necesario para este modelo.
2. Plantilla correcta: `[INST] … [/INST]` o un Modelfile propio
   (`ollama create`) con `TEMPLATE` y `PARAMETER stop` de Mistral.
3. `num_ctx` por modelo (8k) — hoy es global en `defaults.llm`.
4. Y aun con todo eso, su papel razonable es **control negativo** («modelo sin
   ajuste a instrucciones»), no un brazo experimental comparable a Qwen.

### 9.1 No existe un «BioMistral instruct»

Todos los tags de BioMistral publicados en Ollama (`cniongolo/biomistral`,
`adrienbrault/biomistral-7b`, `jsk/bio-mistral`, `m/biomistral`,
`f0rodo/bio-mistral-dare`) son **reempaquetados del mismo modelo base**
(BioMistral-7B, continuación sobre PMC). Probar «BioMistral instruct» implica
descargar **otro** modelo biomédico, no otro tag:

| Modelo | Tamaño | Contexto | Nota |
|---|---:|---:|---|
| `medllama2:7b` | 3,8 GB | 4K | Llama 2 ajustado a QA médica |
| `meditron:7b` | 3,8 GB | **2K** | justo: los prompts miden ~0,93k tokens (máx 0,95k) + salida |
| `medgemma:4b` | 3,3 GB | 128K | Gemma 3 *instruct* para texto médico; cabe de sobra en 8 GB |
| `medgemma:27b` | 17 GB | 128K | no cabe en la GPU local |

### 9.2 El modelo adicional que **no** requiere descarga

`llama3.1:8b` ya estaba instalado (8B, Q4_K_M, 131K, plantilla Llama 3.1
correcta) y **pasa el pipeline tal cual**, sin `format` y sin cambios de código:
4/4 `valid_response` en `C0` (claims verificados con `SLEEP_CLAIM_CHANNELS`) y
`valid_abstention` en `C4_absent_core`. Con `format` también funciona y acierta
2/4 en la muestra de una ventana por clase (n insuficiente: el 1/4 o 2/4 de
cuatro ventanas no es un resultado). Sirve como **brazo de otra familia
instruct** —el contraste que interesa— a coste cero.

> **Corrección (2026-11-24).** El smoke de una ventana decía
> `valid_abstention` en `C4_absent_core`; con las 80 ventanas del piloto se
> abstiene el **51,2 %** en numérico y el **0 %** en texto (§9.4). Una ventana no
> describe el comportamiento de una condición: es la lección ya registrada para
> `--max-windows`.

### 9.3 Veredicto: `medgemma:4b` es un brazo experimental utilizable

No basta con que un modelo emita JSON válido (BioMistral lo hace con
decodificación restringida y aun así no es medible). La prueba es si **la
respuesta depende de la evidencia**. `medgemma:4b` se ejecutó sobre el piloto
DREAMT completo (4 sujetos × 20 ventanas × 6 condiciones × 2 representaciones =
960 llamadas, 2026-11-23) y cumple:

| Criterio | `medgemma:4b` | Umbral |
|---|---|---|
| JSON válido sin `format` | 849/960 (88,4 %) | ≥ 80 % |
| Abstención real en ausencia total (`C4`, `NC_absent`) | **100 %** en las dos vistas | > 0 |
| Decisión dependiente de la condición | sí (§19.5 del piloto) | cambios entre `C0` y `C1`/`C3` |
| Confianza con resolución | **no**: 95 constante en las 80 ventanas de `C0` (AUROC 0,500) | informativa |
| Velocidad | ≈1,1 s/llamada en la RTX 4060 Ti | utilizable |

**Veredicto**: válido como **brazo de otra familia**, con dos avisos que hay que
declarar junto a sus cifras: (1) 11,6 % de respuestas con defecto de forma
—`direction: "high"`— que exige `--recovery` para no perder decisiones, y (2) una
confianza degenerada que **no** permite usar su AUROC para H2a. Detalle de
resultados en [`piloto_dreamt.md`](piloto_dreamt.md) §19.

### 9.4 `llama3.1:8b`: tercer brazo (piloto completo, 2026-11-24)

Se ejecutó el piloto completo (960 llamadas, 0 inválidas, ≈3,5 s/llamada,
57 min) para saber si el patrón de H4 era de dos modelos o del lector en general.
Cumple los criterios de §9.3 con dos matices que **cambian la lectura del eje**:

| Criterio | `llama3.1:8b` | Umbral |
|---|---|---|
| JSON válido sin `format` | **960/960 (100 %)** | ≥ 80 % |
| Abstención en `NC_absent` (nada disponible) | **100 %** en las dos vistas | > 0 |
| Abstención en `C4` (núcleo ausente, temperatura intacta) | **51,2 % num. / 0 % txt.**: decide con la temperatura | no decide sin evidencia |
| Abstención con ausencia parcial (`C1`, `C3`) | **0 %** en ambas vistas: no la percibe | > 0 |
| Confianza con resolución | **no**: 80 en 79 de 80 ventanas (AUROC 0,518); sobreconfianza +0,45 | informativa |
| Velocidad | ≈3,5 s/llamada | utilizable |

**Veredicto**: válido como brazo de otra familia y **el que más corrige**
(`n1` recall 0,30 numérico y 0,50 textual, frente a 0,10 de `qwen` y 0,00 de
`medgemma`), pero es también **el que menos se abstiene** y el que convierte la
ausencia de evidencia en una decisión a partir del canal restante. Con él, el
único invariante de abstención que queda es `NC_absent`; ver
[`piloto_dreamt.md`](piloto_dreamt.md) §19.4–§19.5.

**`cniongolo/biomistral`** queda como control negativo (§9), no como brazo.
