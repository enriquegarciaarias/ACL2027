# P3 — Control de generalidad con PhysioNet/CinC 2017 (ritmo en ECG)

**Fecha:** 2026-11-10 (ejecución) · 2026-11-11 (análisis)
**Adaptador:** `sources/external/physionet2017.py`
**Runner:** `scripts/run_p3_physionet.py`
**Análisis:** `scripts/analyze_p3_physionet.py`
**Salidas:** `results/output/PHYSIONET2017/P3/`
**Informe:** `results/output/PHYSIONET2017/P3/physionet2017_analysis.json`

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

Es, además, el contraste más duro para la tesis: otra tarea (ritmo cardiaco),
otro sensor (una derivación de ECG), otras escalas (z poblacional en lugar de z
intra-sujeto) y **ninguna** de las modalidades del experimento principal (sin
EDA, sin respiración).

## 2. Diseño

| Elemento | Decisión |
|---|---|
| Muestreo | 150 registros por clase (600 en total), estratificado, semilla 42 |
| Split | **CRC32 del identificador**: 30 % desarrollo (197), 70 % evaluación (403) |
| Features | 5 canónicas de ECG: `ecg_hr_bpm`, `ecg_rr_mean_ms`, `ecg_sdnn_ms`, `ecg_rmssd_ms`, `ecg_n_beats` |
| Estandarización | **z poblacional** con mediana/IQR de los registros de **desarrollo** |
| Prompt | ritmo en 4 clases + `abstain`, sin few-shot, con nota de dataset |
| Tarea | `normal` / `af` / `other-rhythm` / `noisy` / `abstain` |
| Modelo | `qwen2.5:7b`, temperatura 0,3, semilla 42, `max_retries` 3 |

No hay línea base por sujeto (cada registro es una persona distinta), así que la
dirección de los claims se define respecto a la **referencia poblacional**
calculada solo con desarrollo. El split es determinista y reproducible por
CRC32, de modo que el conjunto de evaluación no participa ni en la referencia ni
en el modelo de referencia.

### Composición del conjunto de evaluación (403 registros)

| Clase | n | Duración media (s) |
|---|---:|---:|
| `normal` | 97 | 32,3 |
| `af` | 94 | 31,0 |
| `other-rhythm` | 105 | 33,0 |
| `noisy` | 103 | 23,1 |

La clase `noisy` tiene registros más cortos (23 s frente a ~32 s), un efecto
esperable: es más probable que un registro corto no alcance calidad suficiente.

## 3. Qué se mide

1. **Corrección** (H1): accuracy y balanced accuracy (4 clases) frente a la
   etiqueta de experto, por clase, y **exceso de coste** frente a una referencia
   features-only (regresión logística entrenada en desarrollo).
2. **Abstención y clase `noisy`** (H4): tasa de abstención cuando la evidencia
   es inutilizable, y falsos `noisy` cuando no lo es.
3. **Respaldo de claims** (H5): el verificador funciona igual porque los valores
   son z poblacionales y las métricas son las canónicas de ECG. Se añade el
   desglose del error (dirección invertida, dirección sin desviación, magnitud).
4. **Certeza lingüística** (H2b), con contraste estadístico entre `noisy` y el
   resto de clases.
5. **Señal de calidad en la entrada** (H6): AUROC univariante y multivariante de
   `noisy` frente al resto, y curva de compuerta (cuánta cobertura limpia se
   pierde para detectar ruido).
6. **Transferibilidad de la compuerta de consistencia** de P1 y **sistema
   compuesto** LLM + compuerta.

## 4. Muestreo y prevalencia

El muestreo es **balanceado por clase** (150 de cada una), no representativo de
la prevalencia original (el 0,5 % de `noisy`). Por tanto las métricas describen
la capacidad de discriminación entre clases, no el rendimiento en despliegue.
Los resultados deben reportarse con esa salvedad. La prevalencia de `noisy` en
el conjunto de evaluación es del 25,6 %.

---

## 5. Resultados

### 5.1 Corrección: colapso en la clase ya predicha, no en la mayoritaria

| Métrica | LLM | Referencia (logreg) | Mayoría trivial (`other-rhythm`) |
|---|---:|---:|---:|
| Accuracy (decididas) | **0,243** | 0,548 | 0,266 |
| Balanced accuracy | **0,250** | 0,552 | 0,250 |
| Cobertura (decididas / válidas) | 399 / 399 | 403 / 403 | — |
| Abstención | **0,000** | — | — |

- El balanced accuracy (0,250) es **exactamente el azar de 4 clases**: el modelo
  no aporta información sobre el ritmo.
- La accuracy (0,243) queda **por debajo de la mayoría trivial** (0,266), porque
  la clase mayoritaria del conjunto no es la que el modelo elige: predice
  `normal` 398 veces y `af` una.
- **Exceso de coste: −0,305** de accuracy y −0,302 de balanced frente a la
  referencia. Es el mismo patrón que en WESAD (el LLM queda por debajo de una
  regresión logística con las mismas features), aquí más extremo.

### 5.2 Por clase: el LLM solo sabe decir «normal»

| Clase | n | Recall LLM | Recall referencia | Δ | Abstención LLM |
|---|---:|---:|---:|---:|---:|
| `normal` | 97 | **1,000** | 0,711 | +0,289 | 0,000 |
| `af` | 94 | 0,000 | 0,454 | −0,454 | 0,000 |
| `other-rhythm` | 105 | 0,000 | 0,236 | −0,236 | 0,000 |
| `noisy` | 103 | 0,000 | **0,806** | −0,806 | 0,000 |

La referencia logística **sí** detecta `noisy` en el 80,6 % de los casos con las
mismas cinco features, y AF en el 45,4 %. El LLM no predice `noisy` **ni una
vez**. `other-rhythm` es difícil para ambos (0,236), lo que es coherente con la
propia definición de la clase («cualquier otra cosa»).

### 5.3 ¿Hay información en la entrada? Sí

AUROC univariante por clase en el conjunto de evaluación (orientado: >0,5 separa
la clase del resto en la dirección natural):

| Clase | Mejor feature | AUROC univariante |
|---|---|---:|
| `normal` | `ecg_rmssd_ms` | 0,785 |
| `af` | `ecg_rr_mean_ms` | 0,731 |
| `noisy` | `ecg_sdnn_ms` | 0,816 |
| `other-rhythm` | `ecg_hr_bpm` | 0,622 |

Es decir: **cada clase, salvo `other-rhythm`, tiene una única feature que la
separa del resto con AUROC 0,73–0,82**, y una logística con las cinco alcanza
0,552 de balanced. El fallo del modelo es de **lectura e integración**, no de
información disponible. Es la misma conclusión que en WESAD, con otro sensor.

Sobre la escala: las medianas de desarrollo son `sdnn` 39 ms (normal), 159 ms
(af), 237 ms (noisy) y `rmssd` 35 / 209 / 320 ms. Con la referencia poblacional
(centro 146,7 ms, escala IQR 162,4 ms) esas diferencias clínicamente grandes se
comprimen en z moderados (normal ≈ −0,66, af ≈ +0,08, noisy ≈ +0,56). No impide
la separación —la referencia la consigue—, pero deja al lector con desviaciones
que parecen pequeñas, un factor de representación relevante para H3.

### 5.4 Abstención con evidencia inutilizable (H4)

| Medida | Valor |
|---|---|
| Registros `noisy` (etiqueta de experto) | 103 |
| Abstuvo el LLM | **0** (IC95 Wilson 0,000–0,036) |
| Predijo `noisy` | **0** |
| Predijo `normal` | 102 |
| Predijo `af` | 1 |

En 103 registros que un experto declaró **no interpretables**, el modelo emitió
un juicio firme en el 100 % de los casos y **nunca** usó la etiqueta que el
propio esquema le ofrece para ese caso. No es que eligiera mal entre «noisy» y
«normal»: la opción `noisy` no aparece ni una vez en 403 respuestas.

### 5.5 Respaldo de claims (H5)

1.727 claims verificables (0 `unsupported`: el modelo nunca cita una métrica que
no recibió).

| Categoría | n | Tasa |
|---|---:|---:|
| Respaldados | 905 | **0,524** |
| Contradichos | 822 | **0,476** |
| No respaldados | 0 | 0,000 |
| Indeterminados | 0 | 0,000 |
| Acierto de dirección | — | 0,644 |
| Acierto de magnitud | — | 0,778 |

Desglose de los 822 contradichos (un claim solo cae en una categoría):

| Tipo de error | n | Lectura |
|---|---:|---|
| Dirección invertida (`increase` con z<0 o `decrease` con z>0, \|z\|≥0,2) | 209 | Afirmación contraria al dato |
| Dirección sobre valor en la referencia (\|z\|<0,2) | 306 | «Ve» una tendencia donde no la hay |
| Solo magnitud (dos niveles de diferencia) | 208 | Dirección bien, intensidad mal |
| `stable` con desviación real (\|z\|>0,5) | 99 | No ve la desviación que existe |

Por clase verdadera:

| Clase | n claims | Respaldados | Contradichos | Acierto dirección |
|---|---:|---:|---:|---:|
| `normal` | 424 | 0,533 | 0,467 | 0,712 |
| `af` | 411 | 0,518 | 0,482 | 0,591 |
| `other-rhythm` | 457 | 0,549 | 0,451 | 0,654 |
| `noisy` | 435 | **0,494** | **0,506** | 0,618 |

Por métrica (contradichos / total):

| Métrica | Contradichos | Total |
|---|---:|---:|
| `ecg_hr_bpm` | 246 | 398 (61,8 %) |
| `ecg_rmssd_ms` | 203 | 398 (51,0 %) |
| `ecg_sdnn_ms` | 192 | 398 (48,2 %) |
| `ecg_rr_mean_ms` | 127 | 398 (31,9 %) |
| `ecg_n_beats` | 54 | 135 (40,0 %) |

Dos ejemplos reales (mismo registro, `A00591`, predicho `normal`):

- claim `ecg_hr_bpm: decrease (low)` con **z = +0,83** → contradicho.
- claim `ecg_rmssd_ms: decrease (low)` con **z = −0,03** → contradicho
  (dirección declarada sobre un valor en la referencia).

**El respaldo no predice la corrección**: la tasa de claims respaldados es 0,541
en las decisiones acertadas y 0,527 en las falladas (AUROC del respaldo para
predecir el acierto = **0,510**). Es la misma disociación observada en WESAD
entre «describir bien los números» y «acertar la inferencia», aquí con el 47,6 %
de claims contradichos: ni siquiera la descripción es fiable.

### 5.6 Certeza lingüística (H2b): más seguridad donde no hay evidencia

| Clase | Índice de certeza | n |
|---|---:|---:|
| `normal` | −1,27 | 97 |
| `af` | −1,32 | 94 |
| `other-rhythm` | −1,47 | 105 |
| **`noisy`** | **−0,64** | 103 |

- Kruskal-Wallis entre las cuatro clases: H = 11,74, **p = 0,008**.
- `noisy` frente al resto: Mann-Whitney **p = 0,0008**, Cliff δ = **+0,60**
  (efecto grande): el modelo escribe con **más seguridad** precisamente en los
  registros que el experto descartó por ilegibles.
- En WESAD la certeza bajaba con la degradación controlada; aquí **sube** con la
  degradación natural. La dirección del efecto depende de si el modelo *sabe*
  que hay un problema: cuando la señal se degrada por un experimento, el texto
  se vuelve cauto; cuando el ruido viene del mundo real, se vuelve asertivo.

### 5.7 Señal de calidad en la entrada (H6)

Detector entrenado **solo con desarrollo** (197 registros) y evaluado en los 403
de test.

| Feature | AUROC (noisy vs resto) | Mediana `noisy` | Mediana limpio |
|---|---:|---:|---:|
| `ecg_sdnn_ms` | **0,816** | 236,9 ms | 109,5 ms |
| `ecg_rmssd_ms` | **0,809** | 311,3 ms | 145,3 ms |
| `ecg_n_beats` | 0,264 (orientado 0,736) | 28 | 37 |
| `ecg_hr_bpm` | 0,506 | 75,4 bpm | 74,4 bpm |
| `ecg_rr_mean_ms` | 0,494 | 796,1 ms | 806,2 ms |
| **Multivariante (5 features)** | **0,873** (AP 0,692) | — | — |

A diferencia de WESAD —donde el indicador derivado de las features (compuerta de
consistencia) no aportaba AUROC—, aquí **la variabilidad del propio ECG identifica
el ruido**: SDNN y RMSSD altos con menos latidos detectados. La diferencia es
conceptual, no de dataset: en WESAD la corrupción era una manipulación externa
que no dejaba rastro en la geometría de las features; aquí el ruido real **sí**
deforma las features.

Simulación de compuerta (abstenerse si p(noisy) ≥ umbral):

| Recall de `noisy` | Umbral | `noisy` detectados | Limpios perdidos | Cobertura | Accuracy retenida |
|---:|---:|---:|---:|---:|---:|
| 0,50 | 0,305 | 52 | 21 (7,0 %) | 0,809 | 0,294 |
| 0,80 | 0,133 | 82 | 69 (23,0 %) | 0,615 | 0,359 |
| 0,90 | 0,076 | 92 | 90 (30,0 %) | 0,538 | 0,387 |

La mejora aparente de accuracy es **mecánica** (se eliminan registros que el
modelo fallaba), no discriminativa: el balanced accuracy sigue en 0,250. Un umbral
razonable (recall 0,5 al 7 % de cobertura limpia perdida) convertiría 52 juicios
sin fundamento en abstenciones explícitas.

### 5.8 Compuerta de consistencia: transferencia parcial

La compuerta determinista de P1 se aplica aquí sin features de EDA ni
respiración; solo dos de sus ocho comprobaciones son evaluables
(`ecg_hr_vs_rr_mean` y `ecg_n_beats_vs_hr`).

| Resultado | Valor |
|---|---|
| Comprobaciones evaluables | 2 de 8 (403/403 registros) |
| Fallos | 32, **todos** en `ecg_n_beats_vs_hr` |
| Vetos por clase | `noisy` 24 · `normal` 4 · `af` 1 · `other-rhythm` 3 |
| Abstención inducida | 8,0 % global; 23,3 % en `noisy`; 2,7 % en limpio |

- La compuerta **detecta el 23,3 % de los registros ilegibles** con un coste del
  2,7 % de cobertura limpia, sin aprender nada ni usar etiquetas: el ruido real
  rompe la identidad `n_beats ≈ HR × duración`.
- **No es transferible tal cual**: con `require_key_features=True` (la
  configuración de WESAD) vetaría el 100 % del dataset, porque sus features clave
  son de EDA y respiración. Las compuertas de calidad deben declararse **por
  dataset**; este es un resultado de diseño, no un defecto de implementación.
- **Sistema compuesto** LLM + compuerta: 367 decididas, 8,0 % de abstención,
  accuracy 0,253, balanced 0,250, accuracy incluyendo abstención 0,233. La
  compuerta elimina juicios sin fundamento pero no puede crear acierto donde el
  modelo está en el azar.

### 5.9 Ablación few-shot en desarrollo (40 registros)

Se ejecutaron los mismos 40 registros de desarrollo (13 `af`, 11 `normal`,
10 `noisy`, 6 `other-rhythm`) con y sin ejemplos construidos con medianas de
desarrollo.

| Ejecución | Accuracy | Balanced | Decisiones | Inválidas | Claims resp. | Certeza |
|---|---:|---:|---|---:|---:|---:|
| Sin few-shot | 0,282 | 0,250 | 39 `normal` | 1 | 0,503 | −1,78 |
| Con few-shot | 0,375 | 0,302 | 29 `af`, 11 `normal` | 0 | 0,401 | −0,44 |

- Acuerdo entre ambas: **0,275** (11/40). El few-shot **cambia el colapso de
  clase** (`normal` → `af`) más que arreglarlo: nunca predice `noisy` ni
  `other-rhythm` en ninguno de los dos casos.
- Pares discordantes: 7 aciertan solo sin few-shot, 11 solo con few-shot;
  McNemar exacto **p = 0,481** (n = 40, sin evidencia de mejora).
- El few-shot **reduce el respaldo de claims** (0,503 → 0,401) y **sube la
  certeza** (−1,78 → −0,44): más seguridad con menos respaldo, el patrón
  contrario al deseable.
- **Aviso metodológico:** los ejemplares son medianas por clase y el de `af`
  tiene `ecg_sdnn_ms` = +0,07 SD y `ecg_rmssd_ms` = +0,12 SD, pero su
  razonamiento fijo dice «intervalos RR marcadamente irregulares con SDNN/RMSSD
  elevados». El ejemplo funciona como **prior de etiqueta** (empuja hacia `af`),
  no como patrón numérico. La ablación mide, por tanto, el efecto de la etiqueta
  más que el del patrón; una versión corregida debería usar registros
  prototípicos o varios ejemplos por clase.

### 5.10 Respuestas inválidas y recuperación

- 4 inválidas (0,99 %) y todas por el mismo motivo: `direction: "high"` (el
  modelo puso la magnitud en el campo de dirección).
- La recuperación tolerante (`sources/phase3/recovery.py`, ahora con
  `decision_mode="rhythm4"`) recupera esas 4 respuestas: 403 decididas, accuracy
  0,241, balanced 0,250. Las 4 recuperadas estaban **equivocadas**, así que no
  cambia ninguna conclusión; se documenta por consistencia con el brazo textual
  de WESAD.
- El fichero de respuestas tiene 406 líneas: 3 registros pertenecen a una
  ejecución previa de prueba y no están en el conjunto de evaluación. El análisis
  los descarta (ver `inputs.dropped_unlabeled` del informe).

---

## 6. Interpretación para la tesis

1. **El fallo es del lector, no de la información.** Cada clase tiene una feature
   con AUROC 0,73–0,82 y una logística con las mismas cinco entradas alcanza
   0,552 de balanced; el LLM se queda en el azar exacto (0,250) y por debajo de
   la mayoría trivial. Coincide con WESAD: el coste en exceso se reproduce en
   otro sensor y otra tarea.
2. **La abstención no aparece ni cuando el experto la pide.** 0/103 en evidencia
   no interpretable, con 0 usos de la etiqueta `noisy`. H4 se refuerza: no es un
   problema de calibración del umbral, es que la opción no se usa.
3. **La certeza lingüística se invierte con la degradación natural.** En WESAD
   (degradación controlada) el texto se vuelve cauto; en PhysioNet (ruido real)
   se vuelve **más** asertivo (Cliff δ = +0,60, p = 0,0008). El modelo reacciona
   a la *señal de que algo va mal*, no a la falta de información: si el ruido le
   parece un patrón, lo describe con seguridad.
4. **El respaldo de claims es bajo (0,476 contradichos) y no informa del acierto**
   (AUROC 0,510). Es la versión extrema del hallazgo de WESAD: allí el modelo
   citaba bien y fallaba al combinar; aquí falla ya en la lectura.
5. **La calidad sí deja rastro en este dataset** (AUROC multivariante 0,873), a
   diferencia del SQI/compuerta en WESAD. Un sistema con compuerta de calidad
   explícita convertiría ~23 % de los juicios sobre ruido en abstenciones a un
   coste del 2,7 % de cobertura limpia. Es la prueba de concepto de H6 que WESAD
   no podía dar, porque allí la degradación era externa.
6. **El few-shot no es una solución de fondo:** mueve el colapso de clase y sube
   la confianza sin mejorar la corrección de forma significativa (p = 0,481).

## 7. Limitaciones

- Solo ECG: no hay EDA, respiración ni temperatura; los claims quedan limitados
  a métricas cardiacas y la compuerta de P1 solo puede evaluar 2 de 8
  comprobaciones.
- Una sola derivación: la morfología es más pobre que en un ECG de 12.
- Etiquetado por un único experto (el propio reto anunciaba re-etiquetado).
- Muestreo balanceado por clase: no refleja la prevalencia real (0,5 % de
  `noisy`); las métricas miden discriminación, no rendimiento en despliegue.
- El conjunto de desarrollo tiene 197 registros y solo 40 se usaron en la
  ablación few-shot: potencia baja (McNemar p = 0,481 no distingue mejora de
  azar).
- La referencia logística está entrenada con 197 registros y `class_weight`
  balanceado; su 0,552 es un suelo razonable, no un techo.
- Los ejemplares few-shot son medianas por clase con razonamiento fijo, lo que
  introduce un prior de etiqueta (véase §5.9).
- El detector de calidad de §5.7 es exploratorio: entrenado en desarrollo y
  evaluado en test, pero no preregistrado; su curva de compuerta sirve para
  dimensionar el problema, no como resultado confirmatorio.

## 8. Reproducción

Los datos viven **dentro del proyecto** en
`results/input/EXTERNAL/physionet2017/training2017/` (~211 MB: 8.528 `.mat` +
`.hea` + `REFERENCE.csv`). Se puede reubicar con `--data-dir` o con
`params.external.physionet.data_dir`. Ver [`datos_entrada.md`](datos_entrada.md).

```bash
# Descarga (una vez, si no está)
cd results/input/EXTERNAL/physionet2017
curl -sLO https://physionet.org/files/challenge-2017/1.0.0/training2017.zip
unzip -q training2017.zip && rm training2017.zip

# Piloto y corpus (LLM)
python scripts/run_p3_physionet.py --per-class 6 --max-records 6
python scripts/run_p3_physionet.py --per-class 150

# Ablación few-shot (desarrollo)
python scripts/run_p3_physionet.py --eval-split dev --max-records 40
python scripts/run_p3_physionet.py --eval-split dev --max-records 40 --few-shot

# Análisis (sin llamadas al LLM)
python scripts/analyze_p3_physionet.py
```

Salidas:

- `physionet2017.responses.jsonl` (sin etiquetas)
- `physionet2017.labels.jsonl`
- `physionet2017.features.json` (caché de features)
- `physionet2017_summary.json` (resumen de la ejecución)
- `physionet2017_analysis.json` (informe del análisis: corrección, referencia,
  abstención, claims, certeza, señal de calidad, consistencia, ablación)
