# P3 — Control de generalidad con PhysioNet/CinC 2017 (ritmo en ECG)

**Fecha:** 2026-11-10
**Adaptador:** `sources/external/physionet2017.py`
**Runner:** `scripts/run_p3_physionet.py`
**Salidas:** `results/output/PHYSIONET2017/P3/`

---

## 1. Por qué este dataset

La tesis no va de afecto: va de si el LLM dice lo que la evidencia respalda y
cómo reacciona a la degradación de la señal. WESAD aporta el experimento
controlado; falta un **control de generalidad** con etiquetas de experto y, si
es posible, una clase natural de evidencia inutilizable.

PhysioNet/CinC 2017 cumple las tres cosas:

- **8.528 registros** de ECG de una derivación (AliveCor), 9–61 s, 300 Hz.
- Etiquetas de experto: **N** 5.050, **A** (fibrilación auricular) 738,
  **O** (otro ritmo) 2.456 y **~** (*Too Noisy to Process*) 284.
- La clase `noisy` es exactamente lo que H4 necesita: evidencia que **no permite
  interpretación**, etiquetada por un experto, no inducida por nosotros.

## 2. Diseño

| Elemento | Decisión |
|---|---|
| Muestreo | 150 registros por clase (600 en total), estratificado, semilla 42 |
| Split | **CRC32 del identificador**: 30 % desarrollo (200), 70 % evaluación (400) |
| Features | 5 canónicas de ECG: `ecg_hr_bpm`, `ecg_rr_mean_ms`, `ecg_sdnn_ms`, `ecg_rmssd_ms`, `ecg_n_beats` |
| Estandarización | **z poblacional** con mediana/IQR de los registros de **desarrollo** |
| Prompt | ritmo en 4 clases + `abstain`, sin few-shot, con nota de dataset |
| Tarea | `normal` / `af` / `other-rhythm` / `noisy` / `abstain` |

No hay línea base por sujeto (cada registro es una persona distinta), así que la
dirección de los claims se define respecto a la **referencia poblacional**
calculada solo con desarrollo. El split es determinista y reproducible por
CRC32, de modo que el conjunto de evaluación no participa ni en la referencia ni
en el modelo de referencia.

## 3. Qué se mide

1. **Corrección**: accuracy y balanced accuracy (4 clases) frente a la etiqueta
   de experto, y por clase (especialmente `noisy`).
2. **Abstención y clase `noisy`** (H4): tasa de abstención cuando la evidencia
   es inutilizable, y falsos `noisy` cuando no lo es.
3. **Respaldo de claims** (H5): el verificador funciona igual porque los valores
   son z poblacionales y las métricas son las canónicas de ECG.
4. **Certeza lingüística** (H2b).
5. **Modelo de referencia**: regresión logística entrenada en desarrollo y
   evaluada en el mismo conjunto que el LLM.

## 4. Muestreo y prevalencia

El muestreo es **balanceado por clase** (150 de cada una), no representativo de
la prevalencia original (el 0,5 % de `noisy`). Por tanto las métricas describen
la capacidad de discriminación entre clases, no el rendimiento en despliegue.
Los resultados deben reportarse con esa salvedad.

## 5. Limitaciones

- Solo ECG: no hay EDA, respiración ni temperatura; los claims quedan limitados
  a métricas cardiacas.
- Una sola derivación: la morfología es más pobre que en un ECG de 12.
- Etiquetado por un único experto (el propio reto anunciaba re-etiquetado).
- `ecg_n_beats` y las métricas de HRV dependen de la detección de R-peaks; en
  registros ruidosos el extractor puede fallar y producir valores `null`.
- La clase `noisy` se sobremuestrea; no debe interpretarse como tasa de error
  en producción.

## 6. Resultados (evaluación, 403 registros)

| Métrica | LLM | Referencia (logreg) |
|---|---:|---:|
| Accuracy | **0,243** | 0,548 |
| Balanced accuracy | **0,250** | 0,552 |
| Abstención | **0,000** | — |
| Decisiones | 398 `normal`, 1 `af` | — |
| Claims respaldados | 0,524 | — |
| Claims contradichos | **0,476** | — |
| Certeza lingüística | −1,17 | — |
| Confianza | 85 en 405/406 | — |

Recall por clase: `normal` 1,000; `af` 0,000; `other-rhythm` 0,000;
`noisy` 0,000.

### Lectura

- **Colapso a la clase mayoritaria**: el modelo responde `normal` en 398 de 399
  decisiones válidas y nunca predice `af`, `other-rhythm` ni `noisy`. El
  balanced accuracy (0,250) es exactamente el azar de 4 clases.
- **No es falta de información**: la referencia con las mismas 5 features
  alcanza 0,552 de balanced accuracy (AF se separa por SDNN/RMSSD altos y
  `noisy` por menos latidos). Es **coste en exceso**, no ausencia de señal.
- **H4 con evidencia real inutilizable**: 0 % de abstención y 0 % de detección
  de `noisy` en 103 registros etiquetados por experto como no interpretables.
- **H5**: 47,6 % de claims contradichos (frente a ~22 % en WESAD limpio),
  mientras la confianza declarada es 85 y el lenguaje es cauto (índice −1,17):
  confianza numérica, certeza lingüística y respaldo **se desacoplan**.
- **Salvedad de prompt**: este brazo se ejecutó **sin few-shot** (no había
  ejemplos del dataset). La ablación en desarrollo (con y sin 4 ejemplos
  construidos con registros de dev) comprueba si el colapso es inducido por el
  prompt o es un límite del modelo. El conjunto de test no se usa para ajustar
  nada.

## 7. Reproducción

```bash
# Descarga (una vez)
mkdir -p /tmp/opencode/physionet2017 && cd /tmp/opencode/physionet2017
curl -sLO https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
unzip -q training2017.zip

# Piloto y corpus
python scripts/run_p3_physionet.py --per-class 6 --max-records 6
python scripts/run_p3_physionet.py --per-class 150
```

Salidas:

- `physionet2017.responses.jsonl` (sin etiquetas)
- `physionet2017.labels.jsonl`
- `physionet2017.features.json` (caché de features)
- `physionet2017_summary.json`
