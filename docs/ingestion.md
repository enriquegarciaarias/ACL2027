# Pipeline de ingesta WESAD (`INGEST`)

## Resumen ejecutivo

La Fase 1 del proyecto ACL2027 convierte los datos fisiológicos crudos de WESAD en un corpus estructurado, listo para análisis con modelos de lenguaje (LLM). El pipeline procesa 15 sujetos, extrae características fisiológicas por ventana de tiempo, verifica su calidad y empaqueta todo en un conjunto de datos reproducible. El resultado es un corpus de 705 ventanas con 63 características cada una, organizado para que las fases posteriores puedan alimentar a un LLM sin riesgo de fuga de información.

## Dataset WESAD

WESAD es un conjunto de datos fisiológicos multimodal que registra señales de 15 sujetos bajo diferentes condiciones emocionales: baseline, estrés, diversión y meditación. Cada sujeto aporta aproximadamente 47 ventanas de 60 segundos, generando un total de 705 ventanas de análisis. Las señales incluyen ECG, actividad electrodérmica, respiración, temperatura, movimiento y actividad muscular, capturadas tanto en el pecho como en la muñeca. El conjunto permite estudiar cómo varían las respuestas fisiológicas ante diferentes estados de estrés y relajación, y sirve de base para evaluar si un modelo de lenguaje puede interpretar correctamente la señal fisiológica ante degradaciones controladas.

---


## 1. Decisiones metodológicas cerradas

| Decisión | Valor adoptado | Estado |
|---|---|---|
| Segmentación primaria | 60 s **sin solape** | ⚠️ Factor experimental pendiente de sensibilidad (30/60/120 s) |
| Etiquetas del corpus | 1 (baseline), 2 (stress), 3 (amusement), 4 (meditation) | ✅ Cerrada |
| Fin de la Fase 1 | `features` + `represent` (numérico, texto y embebidos) | ✅ Cerrada |
| Features | `neurokit2` + `scipy` | ✅ Cerrada |
| Formato de artefactos | JSONL | ✅ Cerrada |
| Proceso | `--proc INGEST` (dentro del pipeline global ACL2027) | ✅ Cerrada |
| Canónico | `canonical/{SX}.npz` persistido (sin compresión por defecto) | ✅ Cerrada |

Invariantes:

- Las etiquetas **nunca** entran en una representación destinada al LLM.
- La unidad estadística es el sujeto; las ventanas son medidas repetidas.
- Ningún artefacto se sobrescribe sin que cambie su fingerprint.
- Si Ollama no responde, las representaciones **densas** (`embeddings`) se
  omiten con un aviso y la ingesta continúa: son material complementario y la
  Fase 2 solo consume `numeric` y `text`. Un Ollama caído nunca bloquea el
  corpus.
- El corpus se puede reiniciar con `--fresh --yes`: borra
  `results/output/{subject}/INGEST/` y recalcula. **Nunca** toca `results/input`.
  Ver [`ingestion.md#empezar-de-cero`](#empezar-de-cero).
- El fingerprint depende del **contenido del código** que produce el artefacto
  (`code_fingerprint`), no del commit de git: un commit de documentación no
  invalida el corpus. Ver [`verificacion_corpus.md`](verificacion_corpus.md).
- Toda magnitud declara unidad y método de cálculo.

---

## 2. Hechos verificados del dataset

Estructura real de `SX.pkl`:

```python
{
  "subject": "S2",
  "signal": {
      "chest": {"ACC": (N,3), "ECG": (N,1), "EMG": (N,1),
                "EDA": (N,1), "Temp": (N,1), "Resp": (N,1)},   # 700 Hz
      "wrist": {"ACC": (M,3) 32 Hz, "BVP": (K,1) 64 Hz,
                "EDA": (J,1) 4 Hz, "TEMP": (J,1) 4 Hz}
  },
  "label": (N,) int32                                      # 700 Hz
}
```

- 15 sujetos (`S2…S17`, sin `S1` ni `S12`), 12,9 GB de `.pkl`.
- Etiquetas: `0` transitorio, `1` baseline, `2` stress, `3` amusement,
  `4` meditation, `6`/`7` ignorar.
- Las transiciones son cortes duros separados por tramos `0`; no hay ruido de
  etiqueta.
- `chest` ya está en unidades físicas (mV, µS, °C, %, g).
- `wrist.ACC` está en cuentas de 1/64 g y `wrist.BVP` en crudo.
- `WESAD.zip`, `SX_respiban.txt` y `SX_E4_Data.zip` son duplicados: la fuente
  canónica es `SX.pkl` (ya sincronizado).

---

## 3. Pasos del pipeline

| # | Paso | Estado | Entrada | Salida |
|---|---|---|---|---|
| 0 | `discover` | ✅ H1 | `results/input/WESAD/` | `dataset_manifest.json` |
| 1 | `ingest` | ✅ H1 | `SX.pkl` | `canonical/{SX}.npz` + `.meta.json` |
| 2 | `qc` | ✅ H2 | canónico | `qc/{SX}.qc.json` |
| 3 | `window` | ✅ H3 | canónico + etiquetas | `windows/{SX}.windows.jsonl` + índice global |
| 4 | `features` | ✅ H4 | ventanas | `features/{SX}.features.jsonl` + índice global |
| 5 | `represent` | ✅ H5 | features + ventanas | `representations/{numeric,text}/{SX}.jsonl` + índices. `embeddings/` como material complementario (no entrada al LLM) |
| 4.5 | `degrade` | ✅ Nuevo | canónico + condiciones | `degraded/{condition_id}/{SX}.npz` + `.meta.json` |
| 6 | `package` | ✅ H6 | todo | `corpus_index.jsonl`, `groundtruth/labels.jsonl`, `corpus_manifest.json`, `corpus_checksums.sha256` |

`--step` es case-insensitive y ejecuta el pipeline **hasta el paso indicado
inclusive**; los pasos anteriores se reutilizan por fingerprint. Pedir un paso
pendiente falla con un mensaje explícito; no se simula éxito.

---

## 4. Artefactos

```
results/output/WESAD/INGEST/
├── dataset_manifest.json          # sujetos, sha256, tamaños, ignorados
├── steps.json                     # estado y fingerprint de cada paso
├── canonical/
│   ├── S2.npz                     # arrays float32 / label int32
│   └── S2.meta.json               # formas, dtypes, unidades, contadores
├── qc/
│   └── S2.qc.json                 # QC por canal + etiquetas + flags
├── windows/
│   ├── S2.windows.jsonl           # índice de ventanas del sujeto
│   ├── S2.windows.report.json     # ventanas y descartes por sujeto
│   ├── windows_index.jsonl        # índice global
│   └── windows.report.json        # totales globales
├── features/
│   ├── S2.features.jsonl          # 1 línea por ventana (sin etiqueta)
│   ├── S2.features.report.json    # SQI y conteos por sujeto
│   ├── features_index.jsonl       # índice global
│   ├── features.report.json       # resumen global
│   └── feature_spec.json          # modalidades, versiones y SQI
├── representations/
│   ├── numeric/
│   │   ├── S2.jsonl               # JSON autodescriptivo por ventana
│   │   ├── numeric_index.jsonl
│   │   └── numeric.report.json
│   ├── text/
│   │   ├── S2.jsonl               # texto estructurado por ventana
│   │   ├── text_index.jsonl
│   │   └── text.report.json
│   ├── embeddings/
│   │   ├── S2.jsonl               # vectores densos vía Ollama (1024 dims)
│   │   ├── embeddings_index.jsonl # índice global de vectores
│   │   └── embeddings.report.json
│   └── representation_spec.json
├── groundtruth/
│   └── labels.jsonl               # etiquetas separadas (anti-leakage)
├── corpus_index.jsonl             # índice por ventana sin etiqueta
├── corpus_manifest.json           # versión, recuentos, integridad y checksums
├── corpus_checksums.sha256        # checksums de todos los artefactos
└── corpus_errors.json             # solo si falla la integridad
```

Claves del `.npz`:

```
chest_ACC, chest_ECG, chest_EMG, chest_EDA, chest_Temp, chest_Resp,
wrist_ACC, wrist_BVP, wrist_EDA, wrist_TEMP, label
```

El `.npz` es un **caché reproducible**: se regenera desde el `.pkl` original.
El `.meta.json` contiene `fingerprint`, `source_sha256`, `canonical_sha256`,
formas, dtypes, unidades y contadores de etiquetas.

---

## 5. Configuración

La configuración vive en `params.ingestion` del documento fusionado
(global + subject). Los valores por defecto están en
`sources/ingestion/config.py` y el ejemplo en `config_example.json`.

```json
{
  "params": {
    "ingestion": {
      "dataset": "WESAD",
      "source_dir": ".",
      "subjects": "auto",
      "seed": 42,
      "labels": {"keep": [1, 2, 3, 4]},
      "windows": {
        "length_s": 60.0,
        "stride_s": 60.0,
        "purity_min": 0.999,
        "guard_s": 5.0,
        "min_valid_fraction": 0.9,
        "drop_low_coverage": true
      },
      "qc": {
        "max_nan_fraction": 0.01,
        "max_zero_diff_fraction": 0.99,
        "saturation_fraction_threshold": 0.05
      },
      "features": {
        "enabled": ["ecg", "eda", "resp", "temp", "acc", "emg",
                    "bvp", "wrist_eda", "wrist_temp", "wrist_acc"],
        "sqi": {"good_threshold": 0.8, "poor_threshold": 0.5}
      },
      "representations": {
        "enabled": ["numeric", "text"],
        "numeric": {"round_digits": 4, "include_quality": true},
        "text": {"include_quality": true, "include_methods": false,
                 "float_format": ".4g", "max_flags": 10}
      },
      "embeddings": {"model": "mxbai-embed-large",
                     "source": "text",
                     "batch_size": 1,
                     "include_quality": true,
                     "enabled_for_llm": false,
                     "note": "Material complementario. No se usa como entrada al LLM."}
      "package": {"corpus_version": "0.1.0"},
      "persist": {"canonical_arrays": true, "compression": false}
    }
  }
}
```

Un subject puede sobrescribir este bloque en
`results/input/{subject}/config.json`; el merge es profundo.

---

## 6. QC (H2)

Cada canal produce:

- `nan_count` / `nan_fraction`, `inf_count`;
- `mean`, `std`, `min`, `max`, cuantiles 1/5/50/95/99;
- `zero_diff_fraction` (sensor atascado);
- `railing_fraction` (saturación en mínimo/máximo);
- curtosis y asimetría (scipy);
- `range_violations` (de la verificación de unidades);
- `flags`: `high_nan`, `infinite`, `constant`, `flatline`, `saturated`,
  `range_violation`.

El informe incluye además los contadores y transiciones de etiqueta. Los flags
**no excluyen** datos por sí solos: la política de exclusión vive en `window`.

En S2 real: 10 canales, 0 flags, `quality_ok = true`.

## 7. Segmentación (H3)

Reglas:

1. Se detectan los tramos contiguos de etiqueta.
2. Solo se usan las etiquetas de `labels.keep` (1–4).
3. Cada tramo se recorta `guard_s` por ambos extremos.
4. Se generan ventanas de `length_s` con paso `stride_s` dentro del tramo.
5. Se exige `purity >= purity_min` y cobertura por modalidad
   `>= min_valid_fraction`; si no, la ventana se descarta y se contabiliza
   (`short_run`, `low_purity`, `low_coverage`, `label_not_kept`).

Cada ventana guarda rangos de muestras a la frecuencia nativa de cada
modalidad, pureza, cobertura, flags, fuente y fingerprint. La etiqueta se
conserva en el índice para segmentar/evaluar, pero **no** se copiará a las
representaciones.

En S2 real: 45 ventanas (18 baseline, 10 stress, 5 amusement, 12 meditation),
pureza 1.0 y cobertura 1.0.

### Paso intermedio: `degrade` (experimento de resolubilidad)

El paso `degrade` se ejecuta entre `window` y `features`. Genera versiones
degradadas del array canónico de un sujeto aplicando degradación controlada a
los canales especificados en la configuración (`experiment.degradation_conditions`).
Cada condición produce su propio ``.npz`` en
``results/output/{subject}/INGEST/degraded/{condition_id}/{subject}.npz``.

Tipos de degradación:

| Severidad | Tipo | Efecto |
|-----------|------|--------|
| 0.0 | Ninguna | Copia del canónico original (condición limpia) |
| 0.5 | Parcial | 50 % de las muestras del canal sustituidas por NaN |
| 1.0 | Total | Canal plano (todos los valores = 0) |

La etiqueta (label) **nunca** se degrada; solo los canales de señal.

Configuración en `config.json`:

```json
"experiment": {
    "minimal_channels": ["chest_Resp", "chest_EDA"],
    "degradation_enabled": true,
    "degradation_conditions": [
        {"id": "C0_clean", "channels": [], "severity": 0.0},
        {"id": "C3_total_resp", "channels": ["chest_Resp"], "severity": 1.0},
        {"id": "C5_total_both", "channels": ["chest_Resp", "chest_EDA"], "severity": 1.0}
    ]
}
```

### ⚠️ Factor experimental: tamaño de ventana (60 s)

La ventana de **60 s sin solape** es una decisión operativa inicial, no una
decisión fisiológicamente validada. Para el piloto se debe justificar y
analizar su sensibilidad:

#### Justificación fisiológica pendiente (60 s)

Los estados emocionales de WESAD (baseline, stress, amusement, meditation) son
**estados sostenidos** que duran minutos, no segundos. Una ventana de 60 s
captura un segmento estable dentro de cada tramo, pero:

- **HRV temporal:** SDNN, RMSSD y pNN50 necesitan suficientes latidos para ser
  fiables. A 60 s con ~70-90 latidos/min se obtienen ~70-90 RR, suficiente
  para RMSSD y pNN50 pero marginal para SDNN estable.
- **SCR (EDA):** las respuestas cutáneas galvánicas tienen latencia de 1-3 s y
  duración de 5-20 s; una ventana de 60 s captura múltiples SCR pero puede
  diluir respuestas transitorias breves.
- **Respiración:** a ~12-20 respiraciones/min, 60 s contiene ~12-20 ciclos,
  suficiente para estimar frecuencia pero con variabilidad entre ventanas.

#### Análisis de sensibilidad propuesto (piloto)

Probar ventanas de **30 s, 60 s y 120 s** para evaluar:

| Ventana | Ventajas potenciales | Riesgos potenciales |
|---------|---------------------|---------------------|
| **30 s** | Más ventanas por sujeto (máx. estadística), mejor resolución temporal para transiciones | Pocos latidos para HRV (SDNN inestable), menos SCR por ventana, más ruido |
| **60 s** | (actual) Equilibrio entre estabilidad y cantidad de ventanas | Puede diluir eventos breves de estrés/respiración |
| **120 s** | HRV más estable (SDNN), más SCR por ventana, mejor estimación de tendencias | Menos ventanas (máx. estadística), riesgo de mezclar sub-estados dentro de una ventana, más vulnerabilidad a tramos cortos |

**Métricas de comparación propuestas:**
- Nº de ventanas por sujeto y por etiqueta
- SQI medio y distribución por modalidad
- Estabilidad de features (varianza entre ventanas del mismo estado)
- Capacidad de discriminación entre clases (baseline vs stress vs amusement vs meditation)

#### Decisión para el piloto

Mantener **60 s** como valor por defecto en el pipeline, pero ejecutar el
análisis de sensibilidad con 30 s y 120 s como experimento paralelo antes de
cerrar la decisión metodológica definitiva.

---

## 8. Features y SQI (H4)

Cada ventana produce un registro en `features/{SX}.features.jsonl` con 63
valores de feature (10 modalidades) y un bloque de calidad. **El registro no
contiene la etiqueta**: la unión con las ventanas se hace por `window_id`.

| Modalidad | Fuente | Features principales |
|---|---|---|
| `ecg` | `chest_ECG` | HR, RR medio, SDNN, RMSSD, pNN50, outliers, nº latidos |
| `eda` | `chest_EDA` | media, std, pendiente, tónico, fásico, SCR (candidatos, filtrados, tasa, amplitud) |
| `resp` | `chest_Resp` | frecuencia, CV de intervalos, amplitud, std, nº ciclos |
| `temp` | `chest_Temp` | media, std, pendiente, rango |
| `acc` | `chest_ACC` | magnitud media/std/p95, energía, fracción de movimiento |
| `emg` | `chest_EMG` | media absoluta, std, RMS, p95 |
| `bvp` | `wrist_BVP` | HR, RR medio, RMSSD, outliers, nº latidos |

> **Nota (2026-11-08):** `bvp` procede de `wrist_BVP`, por lo que es una
> modalidad de muñeca. La vista LLM principal (`numeric_llm`/`text_llm`)
> incluye solo `{ecg, eda, resp, temp}`; `bvp` se reserva para el experimento
> de sobre-confianza en medidas derivadas (HR de BVP vs ECG de referencia).
| `wrist_eda` | `wrist_EDA` | mismas features EDA a 4 Hz |
| `wrist_temp` | `wrist_TEMP` | mismas features de temperatura |
| `wrist_acc` | `wrist_ACC` | mismas features ACC |

Cada valor declara `value`, `unit`, `method`, `method_version`, `params` y
`kind` (`observed`, `derived`, `quality`, `error`).

### ⚠️ Factor de diseño: codificación de features no calculables

Cuando una característica no puede calcularse (p. ej. 0 o 1 R-picos en ECG
→ HRV indefinido), se codifica de forma explícita para evitar ambigüedades
que afecten a la abstención del LLM:

| Campo | Valor | Significado |
|-------|-------|-------------|
| `value` | `null` (JSON) / "not available" (texto) | No hay dato, no es cero |
| `kind` | `"error"` | Marca programática de no-calculable |
| `unit` | unidad normal | Se mantiene para contexto |
| `method` | nombre del extractor | Para auditoría |

**Por qué `null` y no `0` o omisión:**

- **`0`** sería interpretado por el LLM como un valor real (ej. HR = 0 bpm = parada cardiaca), induciendo a error.
- **Omisión** (no incluir la clave) impediría al consumidor saber qué features faltan para decidir la abstención.
- **`null` + `kind="error"`** deja claro que el dato no está disponible, permitiendo al constructor de prompts y al LLM decidir si abstenerse.

**Impacto en la abstención:** el consumidor debe contar las features con `kind="error"` y combinarlo con el SQI para decidir si la ventana tiene evidencia suficiente. Un SQI bajo con múltiples `kind="error"` es candidato a abstención.

**HRV solo en dominio temporal.** No se calcula LF/HF: una ventana de 60 s no
tiene resolución suficiente para estimarlas de forma fiable. Esta decisión está
registrada en `feature_spec.json`.

**SQI.** Cada modalidad devuelve un valor en `[0, 1]` y flags. El agregado es
una media ponderada (pesos iguales por defecto, configurables) y se publica con
`sqi_min`, `sqi_max` y `category` (`good`/`fair`/`poor`). Es una variable
experimental de calidad, no una validación clínica. Un fallo en una modalidad
no invalida la ventana: se registra `extractor_error` con SQI 0.

En S2 real: 45 ventanas × 63 features = 2.835 valores, SQI medio 0,934,
todas las ventanas en categoría `good`.

---

## 9. Representaciones (H5)

Dos formatos versionados (`representation_version = "1.3"`). Ambos se generan
con las mismas ventanas y features, y ambos excluyen la etiqueta.

### Numérica

JSON autodescriptivo por ventana, pensado para que un constructor de prompts
seleccione campos:

```json
{
  "window": {"window_id": "WESAD:S2:w00001", "t_start_s": 311.5,
             "t_end_s": 371.5, "duration_s": 60.0, "coverage": {"chest": 1.0}},
  "modalities": {
    "ecg": {
      "ecg_hr_bpm": {"value": 79.5686, "unit": "bpm", "kind": "derived", "method": "neurokit2.ecg_peaks"},
      "ecg_rmssd_ms": {"value": null, "unit": "ms", "kind": "error", "method": "neurokit2.ecg_peaks"},
      "ecg_n_beats": {"value": 0, "unit": "count", "kind": "observed", "method": "neurokit2.ecg_peaks"}
    }
  },
  "quality": {"sqi_overall": 0.3, "category": "poor", "flags": ["few_beats"]},
  "feature_version": "1.0"
}
```

### Textual

Texto estructurado por secciones, con unidades y un bloque final de calidad:

```text
PHYSIOLOGICAL SIGNAL SUMMARY
==================================================

Window: WESAD:S2:w00001
Subject: S2
Time: 311.5 s - 371.5 s (duration 60 s)

CARDIOVASCULAR (ECG)
  Heart rate: 79.57 bpm
  RMSSD: not available
  Beats detected: 0 count
  ...

SIGNAL QUALITY
  Overall SQI: 0.3 (poor)
  Concerns: few_beats
  ...
```

### ⚠️ Factor de diseño: diferencia de precisión numérica vs textual

Las dos representaciones usan **precisión diferente intencionalmente**:

| Representación | Configuración | Semántica | Ejemplo (HR: 79.5686 bpm) |
|---|---|---|---|
| Numérica | `round_digits: 4` | 4 **decimales** | `79.5686` |
| Textual | `float_format: ".4g"` | 4 **dígitos significativos** | `79.57` |

Esto es un **factor de diseño deliberado**:

- **Numérica:** prioriza precisión computacional para cálculos posteriores, prompt engineering o comparaciones exactas.
- **Textual:** prioriza legibilidad humana; `.4g` elimina ceros innecesarios y usa notación compacta (`1e-05` en lugar de `0.00001`).

Si se requiere **consistencia exacta** entre ambas representaciones, cambiar en `config.json`:

```json
"representations": {
  "text": {
    "float_format": ".4f",  // 4 decimales fijos, igual que numérica
    ...
  }
}
```

Con `.4f`, `79.5686` se muestra como `79.5686` en la representación textual también.

---

### Vista de auditoría vs vista para el LLM

Se generan dos vistas independientes para cada representación:

| Vista | Directorio | Contenido | Uso |
|-------|-----------|-----------|-----|
| **Auditoría** | `representations/{kind}/` | Completa: tiempos, IDs reales, SQI, cobertura, flags, kind, method | Análisis, depuración, H6 |
| **LLM** | `representations/{kind}_llm/` | Filtrada: solo `value`+`unit`, sin tiempos, sin IDs reales, sin SQI, sin cobertura, sin flags | Prompting al LLM |

La vista para el LLM se activa con:

```json
"representations": {
  "numeric": {
    "llm_view": {
      "enabled": true,
      "opaque_window_id": true,
      "allowed_fields": ["value", "unit"]
    }
  },
  "text": {
    "llm_view": {
      "enabled": true,
      "opaque_window_id": true
    }
  }
}
```

#### Reglas de la vista LLM (lista de campos permitidos, no por exclusión)

**Numérica (`numeric_llm`):**
- Solo `value` y `unit` por feature (si `allowed_fields` lo incluye)
- Sin `kind`, sin `method` (procedencia queda en `features/`)
- Sin `window` con tiempos ni IDs reales; solo `window_id` opaco y `duration_s`
- Sin `quality` (SQI, categoría, flags)

**Textual (`text_llm`):**
- Solo secciones por modalidad con valor y unidad
- Sin header de ventana (`Window`, `Subject`, `Time`)
- Sin bloque `SIGNAL QUALITY`
- Sin `Coverage`, sin `Concerns`
- Identificador de ventana opaco (`win_<hash8>`)

#### ⚠️ Fuga indirecta de etiqueta

WESAD tiene un **protocolo con orden fijo por sujeto**. El instante de inicio de una ventana correlaciona con la condición experimental. Mostrar tiempos absolutos (`Time: 311.5 s - 371.5 s`) o identificadores reales (`w00001`, `S2`) permite al modelo:

1. Inferir la condición a partir del tiempo de inicio (memorización del protocolo)
2. Reconocer el sujeto y recordar respuestas de evaluaciones previas

Por eso la vista LLM usa **identificadores opacos** (`win_<sha256[:8]>`) y omite tiempos absolutos. La presencia o ausencia de calidad (SQI, cobertura, flags) se deja como **factor experimental explícito** para el análisis secundario (H6): se evalúa si el modelo detecta degradación por sí mismo o si simplemente copia la señalización de calidad cuando se le proporciona.

---

### Embebidos (vectores densos) — Material complementario

Generados con Ollama (`/api/embeddings`) sobre el texto de la representación
textual. Cada ventana produce un vector de 1024 dimensiones con el modelo
`mxbai-embed-large`.

**No se usan como entrada al LLM.** Los modelos de chat (p. ej. Qwen en Ollama)
reciben texto o imágenes, no vectores densos; para inyectar un embebido haría
falta entrenar un adaptador específico. Por tanto, los embebidos se mantienen
como material complementario, no como representación candidata para el LLM en
esta fase del proyecto.

**Usos potenciales (pendiente de exploración):**

- **Línea base densa:** entrenar un clasificador sencillo sobre los vectores
  para medir cuánta información conserva el texto, y compararlo con el LLM.
- **Recuperación:** buscar ventanas parecidas para mostrarlas como ejemplos.
  Riesgo: los vecinos recuperados pueden filtrar etiquetas.
- **Sondas (H exploratoria):** requerirían los estados internos del propio LLM
  evaluado, no vectores de `mxbai-embed-large`, que es otro modelo.

**Nota sobre la precisión del vector:** el texto se divide en fragmentos
solapados y sus vectores se promedian. Esto evita el truncado, pero el promedio
no es el embedding del texto completo: mezcla fragmentos, las zonas solapadas
pesan doble, y los modelos de embedding codifican mal las cifras numéricas.
Es una aproximación, no una preservación fiel de la información. Se podría
medir entrenando una regresión que reconstruya features (p. ej. frecuencia
cardíaca) a partir del vector y reportando cuánta varianza se recupera.

**Estado:** pendiente de validación futura. No está activo en el pipeline
principal de representaciones para el LLM.

Si Ollama no está disponible o el modelo no está instalado (`ollama pull mxbai-embed-large`),
el paso de embebidos se omite sin afectar a `numeric` y `text`.

### Invariantes

- Solo se copian de la ventana: `window_id`, tiempos, duración y cobertura.
- No aparecen etiquetas ni nombres de condición (`baseline`, `stress`, etc.).
- Los métodos y parámetros no se incluyen en la representación: quedan en
  `features/feature_spec.json` para auditoría.
- En S2 real: 45 registros por formato (90 ventanas-representación).

### Ejemplos de output reales (sujeto S2, ventana w00001)

#### Numérica (JSONL)

```json
{"window_id":"WESAD:S2:w00001","subject":"S2","representation":"numeric","representation_version":"1.3","payload":{"window":{"window_id":"WESAD:S2:w00001","subject":"S2","t_start_s":311.5,"t_end_s":371.5,"duration_s":60.0,"coverage":{"chest":1,"wrist_acc":1,"wrist_bvp":1,"wrist_eda":1,"wrist_temp":1}},"modalities":{"ecg":{"ecg_n_beats":{"value":79,"unit":"count"},"ecg_hr_bpm":{"value":79.5686,"unit":"bpm"},"ecg_rmssd_ms":{"value":64.02,"unit":"ms"}},"eda":{"eda_mean_us":{"value":4.848,"unit":"uS"}}},"quality":{"sqi_overall":0.8988,"category":"good"}},"feature_version":"1.0","schema_version":"1.0","config_fingerprint":"..."}
```

#### Textual (JSONL, campo `payload` como string)

```json
{"window_id":"WESAD:S2:w00001","subject":"S2","representation":"text","representation_version":"1.3","payload":"PHYSIOLOGICAL SIGNAL SUMMARY\n==================================================\n\nWindow: WESAD:S2:w00001\nSubject: S2\nTime: 311.5 s - 371.5 s (duration 60 s)\n\nCARDIOVASCULAR (ECG)\n  Heart rate: 79.57 bpm\n  RMSSD: 64.02 ms\n  ...\n\nSIGNAL QUALITY\n  Overall SQI: 0.8988 (good)\n  ...\n","config_fingerprint":"...","feature_version":"1.0","schema_version":"1.0"}
```

#### Embebidos (JSONL) — Material complementario

```json
{"window_id":"WESAD:S2:w00001","subject":"S2","representation":"embeddings","representation_version":"1.3","payload":{"vector":[-0.0057,0.0638,-0.0023,0.0245,...],"dimension":1024,"model":"mxbai-embed-large","source":"text"},"config_fingerprint":"...","feature_version":"1.0","schema_version":"1.0"}
```

*Nota: este vector no se presenta al LLM. Se mantiene como material complementario para exploración futura (línea base densa, recuperación, sondas).*

---

## 10. Package (H6)

`package` valida y publica el corpus. Antes de escribir el manifiesto
comprueba:

- presencia de todos los artefactos por sujeto;
- hash del `.npz` canónico contra `.meta.json`;
- hash de origen contra `dataset_manifest.json`;
- alineación estricta de `window_id` entre ventanas, features y
  representaciones;
- etiquetas dentro de `labels.keep`;
- ausencia de la clave `label` en features y representaciones;
- número de features constante por ventana;
- recuentos de reports e índices globales.

Si falla, escribe `corpus_errors.json` y **no** publica un manifiesto válido.

Separación de artefactos:

- `corpus_index.jsonl` — metadatos, calidad, representaciones y fingerprints;
  **sin etiqueta**.
- `groundtruth/labels.jsonl` — etiqueta y tiempos por `window_id`.
- `corpus_manifest.json` — `corpus_version`, versiones, git,
  `config_fingerprint`, recuentos, integridad y lista de artefactos con SHA-256.
- `corpus_checksums.sha256` — checksums de todos los artefactos y del
  manifiesto.

En S2 real: 45 ventanas, 63 features/ventana, 2.835 valores, 24 artefactos más
el manifiesto, y 9/9 comprobaciones de integridad correctas. Una segunda
ejecución devuelve `reused` sin recalcular el índice.

---

## 11. Uso

```bash
source .venv/bin/activate

# Resolver configuración y rutas sin ejecutar
python main.py --proc INGEST --subject WESAD --dry-run

# Solo descubrimiento (manifiesto + hashes)
python main.py --proc INGEST --subject WESAD --step discover

# Desarrollo con un subconjunto, corpus completo
python main.py --proc INGEST --subject WESAD --subjects S2 --step package

# Corpus completo (todos los sujetos)
python main.py --proc INGEST --subject WESAD
```

# Representaciones para el LLM (numérica y textual)
python main.py --proc INGEST --subject WESAD --step represent --force

# Corpus completo (incluye embebidos como material complementario)
python main.py --proc INGEST --subject WESAD --step package --force
```

Flags nuevas: `--subjects`, `--force`, `--dry-run`.

`--step X` ejecuta hasta X inclusive. Por ejemplo, `--step represent` ejecuta
todo hasta las representaciones numérica y textual para el LLM, y
`--step package` ejecuta la Fase 1 completa.

**Nota sobre embebidos:** `mxbai-embed-large` se genera como material
complementario (no como entrada al LLM). Requiere Ollama en ejecución
(`http://localhost:11434`) con el modelo instalado:

```bash
ollama pull mxbai-embed-large
```

---

## 12. Reproducibilidad

Cada fingerprint combina:

- `schema_version`,
- hash del fichero fuente,
- configuración resuelta (tasas, unidades, etiquetas),
- commit de git,
- versiones de Python/numpy/scipy/neurokit2.

Repetir un paso con el mismo fingerprint reutiliza el artefacto. `--force` lo
recalcula. Los tests verifican la idempotencia.

El `package` añade una capa final de reproducibilidad:

- `corpus_manifest.fingerprint` se calcula sobre los artefactos de datos
  (canónico, ventanas, features y representaciones), los hashes de origen, la
  configuración y el commit;
- `corpus_checksums.sha256` permite verificar la integridad de cualquier
  copia del corpus;
- `corpus_version` identifica la versión publicada (por defecto `0.1.0`).

---

## 13. Tests

```bash
python -m pytest -q
```

Cobertura actual (49 tests):

- rutas y escritura atómica;
- fingerprints deterministas;
- conversión y verificación de unidades;
- descubrimiento, hashes cacheados y ficheros ignorados;
- validación del esquema `.pkl` y coherencia temporal;
- canonicalización, idempotencia y `--force`;
- QC por canal, flags de canal plano/NaN, idempotencia;
- segmentación: pureza, guardas, rangos nativos, no cruce de transiciones,
  descarte de tramos cortos e índice global;
- features: procedencia, ausencia de etiqueta, HRV temporal, idempotencia,
  índice global y robustez ante señal plana;
- representaciones: numérica y textual, anti-leakage, idempotencia, índices
  globales y compatibilidad con el formato de lista;
- empaquetado: manifiesto, índice, etiquetas separadas, checksums,
  idempotencia y detección de corrupción del canónico;
- orquestador, `--step` y `--dry-run`.

Los tests usan un dataset sintético diminuto; no requieren WESAD.

---

## Empezar de cero

Un servidor nuevo, o artefactos copiados que no deberían estar, se resuelven
borrando **solo lo derivado**:

```sh
python main.py --proc INGEST --subject WESAD --fresh --dry-run   # inventario y MB
python main.py --proc INGEST --subject WESAD --fresh --yes       # borra y recalcula
```

Qué hace `--fresh`:

1. Lista el contenido de `results/output/{subject}/INGEST/` con su tamaño.
2. Sin `--yes` **no borra nada**: registra el aviso y sale con código 2.
3. Con `--yes`, borra el árbol completo (`canonical`, `qc`, `windows`, `degraded`,
   `degraded_features`, `features`, `representations`, `steps.json`,
   `corpus_*`) y deja el directorio vacío.
4. Avisa de que existen `PHASE2`/`PHASE3` del corpus anterior, pero **no los
   toca**.

Qué **no** toca nunca: `results/input/` (WESAD y datasets externos) ni
`config.json`.

### Migrar de servidor

```sh
# máquina antigua o copia manual
rsync -av results/input/ nuevo_servidor:~/proyectos/ACL2027/results/input/
scp config.json nuevo_servidor:~/proyectos/ACL2027/

# servidor nuevo
python main.py --proc INGEST --subject WESAD --fresh --yes
```

El fingerprint no interviene en la decisión: con `--fresh` se recalcula todo por
definición. Los fingerprints sirven para lo contrario, para **no** recalcular
cuando el contenido no ha cambiado (mismo `.pkl`, mismo código, misma config).

