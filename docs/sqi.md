# SQI — Indicadores de Calidad de Señal (pipeline INGEST)

**Estado:** documentado  
**Versión:** 1.0  
**Módulos implicados:** `sources/ingestion/features/sqi.py`, `sources/ingestion/features/common.py`, `sources/ingestion/features/registry.py`, y los extractores por modalidad (`ecg.py`, `eda.py`, `resp.py`, `temp.py`, `acc.py`, `emg.py`, `bvp.py`).

---

## 1. Visión general

El **SQI** (Signal Quality Index) es una variable experimental de calidad de señal, no una validación clínica. Cada ventana de 60 s recibe:

1. Un **SQI individual por modalidad** (ECG, EDA, Resp, Temp, ACC, EMG, BVP y sus versiones de muñeca), acotado a `[0, 1]`.
2. Un **SQI agregado por ventana** (`sqi_overall`), calculado como media ponderada de los SQI de las modalidades disponibles.
3. Una **categoría de calidad** (`good`, `fair`, `poor`) basada en dos umbrales configurables.
4. Una lista de **flags** por modalidad y por ventana (`per_modality`, `flags`).

El SQI sirve para:

- Detectar ventanas con señal corrupta o insuficiente antes de enviar al LLM.
- Filtrar o ponderar ventanas en análisis posteriores (Fase 2 y Fase 3).
- Auditar el impacto de condiciones de degradación (experimento `degrade`).

**Importante:** el SQI no aparece en la representación filtrada para el LLM por defecto (ver `docs/ingestion.md`, separación vista de auditoría vs vista LLM). Está disponible en la vista de auditoría y en los artefactos de features.

---

## 2. Flujo de datos

```
ventana (60 s)  ──►  coverage por modalidad  ──►  extractor(ecg|eda|resp|...)
                                                    │
                                                    ▼
                                           sqi_modalidad + flags
                                                    │
                                                    ▼
                                    sqi.combine(quality, cfg)
                                                    │
                      ┌───────────────┬───────────┼───────────┐
                      ▼               ▼           ▼           ▼
                 sqi_overall   sqi_min   sqi_max   category   flags
                      │                                   │
                      └───────────────┬───────────────────┘
                                      ▼
                         publicado en features/{subject}.features.jsonl
```

El orquestador es `registry.extract_window()` (`sources/ingestion/features/registry.py:41-80`). Recorre todas las modalidades habilitadas, extrae el array correspondiente del canónico para el rango de la ventana, llama al extractor de esa modalidad y acumula resultados. Si un extractor lanza una excepción, no invalida la ventana: se asigna SQI `0.0` con flag `extractor_error` y se continúa con las demás (`registry.py:70-76`).

---

## 3. Cálculo del SQI por modalidad

Cada modalidad define su propio `extract()` y devuelve una tupla `(features_dict, quality_dict)`. El `quality_dict` se construye con `sqi.modality_quality(sqi, flags, **extra)` (`sqi.py:15-26`), que acota el `sqi` con `clip01()` y guarda los flags.

Todas las modalidades empiezan desde `sqi = coverage`, donde `coverage` es la fracción de la ventana con datos válidos para esa canal (proviene del paso `window`). Luego aplican penalizaciones específicas.

### 3.1 ECG (pecho) — `sources/ingestion/features/ecg.py:91-95`

```python
sqi = coverage * max(0.0, min(1.0, mean_quality))
if outlier is not None:
    sqi *= (1.0 - outlier)
if n_beats < min_beats:
    sqi *= 0.5
```

- `mean_quality`: calidad media devuelta por `nk.ecg_quality(clean, rpeaks=peaks, sampling_rate=fs, method="averageQRS")`. Acotada a `[0, 1]`. Si el cálculo falla, vale `0.0`.
- `outlier`: fracción de intervalos RR fuera de `[outlier_low_ms, outlier_high_ms]`. `outlier = 1.0 - valid.size / rr.size`.
- `n_beats`: número de picos R detectados por `nk.ecg_peaks`.
- `min_beats`: umbral mínimo de latidos (configurable, default en config).

**Flags:**
- `few_beats` si `n_beats < min_beats`.
- `rr_outliers` si `outlier > 0.10`.

**Features relacionadas publicadas:** `ecg_n_beats`, `ecg_hr_bpm`, `ecg_rr_mean_ms`, `ecg_sdnn_ms`, `ecg_rmssd_ms`, `ecg_pnn50_pct`, `ecg_rr_min_ms`, `ecg_rr_max_ms`, `ecg_rr_outlier_fraction`, `ecg_amplitude_std_mv`.

---

### 3.2 BVP (muñeca) — `sources/ingestion/features/bvp.py:75-79`

```python
sqi = coverage * max(0.0, min(1.0, mean_quality))
if outlier is not None:
    sqi *= (1.0 - outlier)
if n_beats < min_beats:
    sqi *= 0.5
```

- `mean_quality`: calidad media de `nk.ppg_quality(clean, peaks=peaks, sampling_rate=fs, method="templatematch")`. **Si el cálculo de calidad lanza excepción, se usa `0.5` por defecto** (a diferencia de ECG, que usa `0.0`).
- `outlier`: fracción de intervalos inter-picos fuera del rango configurado.
- `n_beats`: número de picos de pulso detectados por `nk.ppg_peaks`.

**Flags:**
- `few_beats` si `n_beats < min_beats`.
- `rr_outliers` si `outlier > 0.10`.

**Features relacionadas:** `bvp_n_beats`, `bvp_hr_bpm`, `bvp_rr_mean_ms`, `bvp_rmssd_ms`, `bvp_rr_outlier_fraction`, `bvp_amplitude_std`.

---

### 3.3 EDA (pecho y muñeca) — `sources/ingestion/features/eda.py:102-108`

```python
sqi = coverage
if std <= 1e-12:
    sqi *= 0.5
    flags.append("constant")
if "phasic_failed" in flags:
    sqi *= 0.5
```

- `std`: desviación estándar de los valores finitos de la señal EDA cruda.
- Si la señal es constante (`std <= 1e-12`), se penaliza con `× 0.5` y flag `constant`.
- Si la descomposición tónico/fásica falla (`nk.eda_clean` / `nk.eda_phasic` / `nk.eda_peaks` lanza excepción), se añade flag `phasic_failed` y se penaliza `× 0.5`.

**Flags:**
- `constant` si `std <= 1e-12`.
- `phasic_failed` si falló la limpieza o detección de picos SCR.

**Features relacionadas:** `eda_mean_us`, `eda_std_us`, `eda_slope_us_per_s`, `eda_tonic_mean_us`, `eda_phasic_std_us`, `eda_scr_candidates`, `eda_scr_count`, `eda_scr_rate_per_min`, `eda_scr_amplitude_mean_us`, `eda_scr_amplitude_max_us`.

---

### 3.4 Resp (pecho) — `sources/ingestion/features/resp.py:61-67`

```python
sqi = coverage
if cv is not None:
    sqi *= max(0.3, 1.0 - min(cv, 1.0))
else:
    sqi *= 0.5
if n_cycles < min_cycles:
    sqi *= 0.5
```

- `cv`: coeficiente de variación de los intervalos entre valles consecutivos (`RSP_Troughs`). `cv = std(intervals) / mean(intervals)`.
- Si hay al menos 2 intervalos y la media es > 0, se calcula `cv`. El factor `max(0.3, 1.0 - min(cv, 1.0))` asegura que el multiplicador nunca baje de 0.3.
- Si no hay `cv` (menos de 2 intervalos), se aplica `× 0.5`.
- Si `n_cycles < min_cycles`, se aplica `× 0.5`.

**Flags:**
- `few_cycles` si `n_cycles < min_cycles`.

**Features relacionadas:** `resp_rate_bpm`, `resp_interval_cv`, `resp_amplitude_pct`, `resp_std_pct`, `resp_n_cycles`.

---

### 3.5 Temp (pecho y muñeca) — `sources/ingestion/features/temp.py:39-44`

```python
sqi = coverage
if std <= 1e-12:
    sqi *= 0.8
    flags.append("constant")
```

- `std`: desviación estándar de los valores finitos de temperatura.
- Si es constante (`std <= 1e-12`), se penaliza suavemente con `× 0.8` (no se considera señal inválida, puede ser cuantización del sensor).

**Flags:**
- `constant` si `std <= 1e-12`.

**Features relacionadas:** `temp_mean_degc`, `temp_std_degc`, `temp_slope_degc_per_s`, `temp_range_degc`.

---

### 3.6 ACC (pecho y muñeca) — `sources/ingestion/features/acc.py:58-61`

```python
sqi = coverage
if std <= 1e-12:
    sqi *= 0.5
    flags.append("constant")
```

- La señal puede ser 1D o 2D (3 ejes). Si es 2D, se calcula magnitud vectorial `sqrt(x² + y² + z²)` en g.
- `std`: desviación estándar de la magnitud.
- Si es constante, `× 0.5`.

**Flags:**
- `constant` si `std <= 1e-12`.

**Features relacionadas:** `acc_mean_g`, `acc_std_g`, `acc_p95_g`, `acc_energy_g2`, `acc_motion_fraction`.

---

### 3.7 EMG (pecho) — `sources/ingestion/features/emg.py:37-40`

```python
sqi = coverage
if std <= 1e-12:
    sqi *= 0.5
    flags.append("constant")
```

- `std`: desviación estándar de los valores finitos de EMG.
- Si es constante, `× 0.5`.

**Flags:**
- `constant` si `std <= 1e-12`.

**Features relacionadas:** `emg_mean_abs_mv`, `emg_std_mv`, `emg_rms_mv`, `emg_p95_abs_mv`.

---

## 4. Agregación del SQI por ventana — `sqi.combine()`

`src/sources/ingestion/features/sqi.py:29-85`.

### 4.1 Entrada

Recibe:
- `quality`: un `Mapping[str, dict]` donde cada clave es el nombre de una modalidad y el valor es el bloque devuelto por `modality_quality()` (contiene `sqi`, `flags`, y campos extra).
- `cfg`: configuración completa del pipeline.

### 4.2 Pasos

**Paso 1 — Filtrar modalidades disponibles:**

```python
available = {name: block for name, block in quality.items()
             if block.get("sqi") is not None}
```

Las modalidades con `sqi is None` se excluyen del agregado (aunque siguen apareciendo en `per_modality`).

**Paso 2 — Pesos:**

```python
sqi_cfg = cfg["features"]["sqi"]
weights = sqi_cfg.get("weights", {})  # default: cada modalidad pesa 1.0
```

Cada modalidad usa `weights.get(name, 1.0)`.

**Paso 3 — Media ponderada:**

```
sqi_overall = Σ (peso_i × sqi_i) / Σ pesos_i
```

si `Σ pesos > 0`; en caso contrario, `0.0`.

**Paso 4 — Estadísticas y categoría:**

```python
sqi_min = min(values)
sqi_max = max(values)

if overall >= good_threshold (default 0.8):
    category = "good"
elif overall >= poor_threshold (default 0.5):
    category = "fair"
else:
    category = "poor"
```

**Paso 5 — Flags globales:**

Se concatenan todos los flags de todas las modalidades con el formato `"modalidad:flag"`, por ejemplo `"ecg:few_beats"`, `"eda:constant"`.

**Paso 6 — Caso sin modalidades:**

Si `available` está vacío:
```python
{
    "per_modality": {...},
    "sqi_overall": 0.0,
    "sqi_min": 0.0,
    "sqi_max": 0.0,
    "category": "poor",
    "flags": flags + ["no_modalities"],
}
```

### 4.3 Salida

```json
{
  "per_modality": {
    "ecg": {"sqi": 0.95, "flags": []},
    "eda": {"sqi": 0.80, "flags": ["constant"]},
    ...
  },
  "sqi_overall": 0.88,
  "sqi_min": 0.50,
  "sqi_max": 1.00,
  "category": "good",
  "flags": ["eda:constant"]
}
```

Todos los valores numéricos pasan por `clip01()` (`common.py:80-83`): si son `None`, `NaN`, `inf` o están fuera de `[0, 1]`, se convierten a `0.0`.

---

## 5. Configuración (`config.json`)

El bloque de SQI vive bajo `params.ingestion.features.sqi` (o directamente `cfg["features"]["sqi"]` según cómo se resuelva la config). Ejemplo:

```json
{
  "features": {
    "sqi": {
      "weights": {
        "ecg": 1.0,
        "eda": 1.0,
        "resp": 1.0,
        "temp": 1.0,
        "acc": 1.0,
        "emg": 1.0,
        "bvp": 1.0,
        "wrist_eda": 1.0,
        "wrist_temp": 1.0,
        "wrist_acc": 1.0
      },
      "good_threshold": 0.8,
      "poor_threshold": 0.5
    }
  }
}
```

- **`weights`**: pesos por modalidad en la media ponderada. Cualquier modalidad no listada usa peso `1.0`.
- **`good_threshold`**: umbral inferior para categoría `good` (default `0.8`).
- **`poor_threshold`**: umbral inferior para categoría `fair` (default `0.5`). Entre `poor_threshold` y `good_threshold` es `fair`.

Otros parámetros que afectan indirectamente al SQI (definidos en `params.ingestion.features.<modalidad>`):
- `ecg.min_beats`, `ecg.outlier_low_ms`, `ecg.outlier_high_ms`
- `eda.min_scr_amplitude_us`, `eda.clean_method`, `eda.phasic_method`, `eda.highpass_hz`
- `resp.min_cycles`, `resp.clean_method`
- `acc.motion_threshold_g`
- `bvp.min_beats`, `bvp.outlier_low_ms`, `bvp.outlier_high_ms`

---

## 6. Ejemplo numérico paso a paso

Supongamos una ventana con estas modalidades y SQI individuales:

| Modalidad | SQI | Flags |
|-----------|-----|-------|
| ecg | 0.92 | [] |
| eda | 0.50 | ["constant"] |
| resp | 0.85 | [] |
| temp | 1.00 | [] |
| acc | 1.00 | [] |
| emg | 0.50 | ["constant"] |
| bvp | 0.90 | [] |
| wrist_eda | 0.70 | [] |
| wrist_temp | 0.95 | [] |
| wrist_acc | 1.00 | [] |

Todos con peso 1.0.

**Media ponderada:**
```
(0.92 + 0.50 + 0.85 + 1.00 + 1.00 + 0.50 + 0.90 + 0.70 + 0.95 + 1.00) / 10
= 8.32 / 10
= 0.832
```

**Categoría:** `0.832 >= 0.8` → `"good"`.

**Estadísticas:**
- `sqi_overall = 0.832`
- `sqi_min = 0.50`
- `sqi_max = 1.00`
- `flags = ["eda:constant", "emg:constant"]`

Aunque el agregado sea `good`, `sqi_min = 0.50` revela que al menos dos modalidades tienen calidad regular. Por eso se publica `sqi_min` y `sqi_max`: para no ocultar modalidades malas detrás de un promedio alto.

---

## 7. Edge cases y robustez

| Situación | Comportamiento |
|-----------|----------------|
| Extractor lanza excepción | SQI `0.0`, flag `extractor_error`, la ventana continúa con las demás modalidades (`registry.py:70-76`). |
| Sin valores finitos en la señal | El extractor lanza `ValueError`, caught como excepción arriba. |
| Modalidad no habilitada | Se salta (`registry.py:56`). |
| `sqi` es `None` o `NaN` | `clip01()` lo convierte a `0.0` antes de publicar. |
| Sin ninguna modalidad disponible | `sqi_overall = 0.0`, categoría `poor`, flag `no_modalities`. |
| `coverage = 0.0` | Todo el SQI de esa modalidad será `0.0` independientemente de otras penalizaciones. |
| Señal constante (std ≈ 0) | EDA/ACC/EMG: `× 0.5`. Temp: `× 0.8`. ECG/BVP: no aplican este filtro, dependen de `mean_quality` y outliers. |

---

## 8. Archivos de código y responsabilidades

| Archivo | Responsabilidad |
|---------|-----------------|
| `sources/ingestion/features/sqi.py` | `modality_quality()` (constructor de bloque de calidad) y `combine()` (agregación, umbrales, categoría). |
| `sources/ingestion/features/common.py` | `clip01()` (acotado a [0,1]), `FeatureBuilder` (estándar de publicación de features), `finite()`, `linear_slope()`. |
| `sources/ingestion/features/registry.py` | `extract_window()` (orquestador por ventana, aislamiento de fallos, llamada a `sqi.combine()`). |
| `sources/ingestion/features/ecg.py` | SQI ECG: `coverage × quality_averageQRS × (1-outlier) × (0.5 si few_beats)`. |
| `sources/ingestion/features/bvp.py` | SQI BVP: igual que ECG pero calidad `templatematch` y fallback `0.5` si falla. |
| `sources/ingestion/features/eda.py` | SQI EDA: `coverage × (0.5 si constante) × (0.5 si phasic_failed)`. |
| `sources/ingestion/features/resp.py` | SQI Resp: `coverage × max(0.3, 1-cv) × (0.5 si few_cycles o sin cv)`. |
| `sources/ingestion/features/temp.py` | SQI Temp: `coverage × (0.8 si constante)`. |
| `sources/ingestion/features/acc.py` | SQI ACC: `coverage × (0.5 si constante)`. |
| `sources/ingestion/features/emg.py` | SQI EMG: `coverage × (0.5 si constante)`. |

---

## 9. Limitaciones conocidas (pendientes de validar)

1. **Umbral SCR:** El filtro de amplitud de EDA (`min_scr_amplitude_us = 0.05 µS`) se calibró empíricamente sobre S3 (2026-11-07): con 0.01 µS el conteo SCR era ruido (baseline 52 > stress 40, invertido); con 0.05 µS separa correctamente stress (≈5.4/ventana) de baseline/amusement/meditation (≈0). Pendiente de validar con literatura y anotación humana.
2. **cvxEDA:** No disponible (falta `cvxopt`). Se usa `highpass` para la descomposición fásica de EDA.
3. **EDA de muñeca a 4 Hz:** `nk.eda_clean` emite un aviso no bloqueante ("very low frequency").
4. **Solo S2 validado end-to-end:** El pipeline completo con SQI se ha verificado en S2 (45 ventanas, SQI medio 0,934, 45/45 `good`). Falta validar en S3–S17.
5. **Promedio de embebidos:** Si en el futuro se usan embebidos como línea base densa, el promedio de chunks no preserva toda la información (ver `docs/ingestion.md`).

---

## 10. Próximos pasos documentados

- Validar el umbral SCR y evaluar `cvxEDA` con `cvxopt`.
- Calibrar el SQI con anotación humana.
- Validar el confusor del movimiento (ACC/EMG) en los 14 sujetos restantes.
- Explorar embebidos como línea base densa (regresión que reconstruya features a partir del vector).
