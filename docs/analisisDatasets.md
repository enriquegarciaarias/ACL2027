# Compendio de datasets de los papers recabados (corpus SLIP)

Documento **consolidado**: reúne en una sola tabla los datasets de los papers del
corpus SLIP que hasta ahora estaban repartidos en tres bloques (dos tablas y las
30 fichas de `gemini_notebook_03.md`). Tabla completa, sin duplicados,
**con los papers que usan WESAD en las primeras filas** y con dos columnas nuevas:
relevancia bibliográfica y similitud con esta tesis.

---

## WESAD: descripción del dataset, señales y uso en el experimento

### WESAD

WESAD (*Wearable Stress and Affect Detection*) es un dataset público recogido en laboratorio con **15 participantes** (S2–S17, sin S12) durante una **única sesión** por persona. Cada sesión sigue un protocolo secuencial con **dos versiones (A y B)** que contrabalancean el orden de las condiciones para mitigar efectos de orden:

- **baseline** (reposo, lectura neutral),
- **stress** (inducción tipo Trier / Trier Social Stress Test adaptada),
- **amusement** (videos divertidos),
- **meditación 1** y **meditación 2** (respiración guiada, dos fases, para "de-excitar" al sujeto),
- **rest** (periodo de descanso de 10 min tras el stress, solo en versión B).


Las etiquetas provienen de la **condición de protocolo**, no de los autoinformes, y se alinean a **ventanas de 60 s sin solape**. El dataset suma unas **12,9 GB** de señales crudas y es el vehículo principal de esta tesis (H1–H6): permite degradar canales de forma controlada y comparar la respuesta del LLM contra una referencia de estado conocida.

### Duración del protocolo y tiempos por condición

El estudio completo dura aproximadamente **dos horas** por sujeto (sin incluir preparación y recuperación). Cada sesión sigue un protocolo secuencial con **dos versiones (A y B)** que contrabalancean el orden de *stress* y *amusement* para mitigar efectos de orden.

| Condición | Duración en paper | Duración media en datos | Rango entre sujetos | Contenido |
|---|---|---|---|---|
| Baseline (L1)| 20 minutos | 19,6 min | 19,0 – 20,0 min | Lectura neutral en escritorio, sentado o de pie junto a una mesa |
| Stress (TSST modificado) (L2) | ~10 minutos | 11,1 min | 10,2 – 12,1 min | Exposición oral y cuenta atrás desde 2023 de 17 en 17 |
| Amusement (L3)| ~6,5 min (392 s) | 6,2 min | 6,0 – 6,4 min | 11 videos humorísticos con pausas neutras de 5 s entre ellos |
| Meditación 1 (L4)| ~7 min | ~6,6 min | 6,1 – 6,8 min | Respiración guiada vía audio |
| Meditación 2 (L4)| ~7 min | ~6,5 min | 6,1 – 6,8 min | Respiración guiada vía audio |
| Rest | 10 min | – | – | Descanso tras el stress (solo en versión B) |
| Label 5 | – | 0,9 min | 0,8 – 1,2 min | No definido explícitamente en el paper (no presente en S2) |
| Label 6 | – | 0,9 min | 0,7 – 1,1 min | No definido explícitamente en el paper |
| Label 7 | – | 0,9 min | 0,3 – 1,2 min | No definido explícitamente en el paper |

Datos afectivos por sujeto: unos **37 minutos** en total (baseline + stress + amusement + meditaciones).

> **Nota sobre las etiquetas 5, 6 y 7:** el paper de Schmidt et al. (2018) no define explícitamente qué representan las etiquetas numéricas 5, 6 y 7. En los datos `.pkl` aparecen como segmentos muy cortos (~0,9 min cada uno) entre las condiciones principales. Dado que el protocolo incluye autoinformes (PANAS, STAI, SAM, SSSQ) entre condiciones —marcados como "red/dark boxes" en Figure 2 del paper— es plausible que estas etiquetas correspondan a esos periodos de cuestionario o a transiciones breves no clasificadas como ninguna de las 4 condiciones afectivas principales.

> **Nota:** las duraciones en la columna «Duración en paper» se refieren al protocolo descrito en Schmidt et al. (2018), *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*.

#### Orden de las fases

Existen dos versiones del protocolo, contrabalanceadas entre sujetos para mitigar efectos de orden:

- **Versión A:** baseline → amusement → meditación 1 → stress → **rest** → meditación 2  
- **Versión B:** baseline → stress → **rest** → meditación 1 → amusement → meditación 2

En ambas versiones hay dos meditaciones (Medi I y Medi II). En la versión B, el periodo de **rest** (10 min) se sitúa justo después del stress; en la versión A, el rest se sitúa entre el stress y la meditación 2. Según el paper de Schmidt et al. (2018), las meditaciones se introdujeron explícitamente para "de-excite the subjects and bring them back to a close to neutral affective state".

Esa estructura de contrabalanceo es relevante para el análisis: el estrés puede venir justo después del baseline o tras amusement más una meditación.

#### Qué ocurre entre condiciones

El paper de Schmidt et al. (2018) documenta un periodo de **rest** de 10 minutos tras el stress (solo en versión B). Además, entre condiciones se llenan autoinformes (PANAS, STAI, SAM, SSSQ — marcados como "red/dark boxes" en Figure 2 del paper). Al final del protocolo hay un **recovery** (retirada de sensores y desvelo del engaño del TSST), pero **no es una condición experimental**: no se recogen datos etiquetados durante él y no influye en las ventanas de 60 s que el LLM ve.

Los huecos entre fases que no corresponden a una condición afectiva definida quedan registrados como **etiqueta 0 (transitoria)** en las señales. Se ha verificado empíricamente en los 15 sujetos (S2–S17, sin S12):

| Métrica de etiqueta 0 | Valor | Rango entre sujetos |
|---|---|---|
| % del registro en etiqueta 0 | **45,0 %** (media) | 39,3 % – 55,2 % |
| Número de gaps por sujeto | **8,9** (media) | 8 – 9 |
| Tiempo total en gaps por sujeto | **43,9 min** (media) | 34,2 – 65,1 min |
| Duración media de cada gap | **4,9 min** (media) | – |
| Gap más corto observado | – | 26 s |
| Gap más largo observado | – | 19,0 min (S4) |

**Detalle por sujeto (duración de cada gap en minutos):**

| Subj | n_gaps | Gap 1 | Gap 2 | Gap 3 | Gap 4 | Gap 5 | Gap 6 | Gap 7 | Gap 8 | Gap 9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 8 | 2.4 | 6.4 | 6.4 | 4.6 | 14.5 | 11.5 | 10.2 | 3.0 | – | 51.0 |
| S3 | 9 | 0.7 | 17.3 | 5.1 | 5.1 | 6.2 | 4.8 | 6.2 | 5.1 | 5.1 | 55.8 |
| S4 | 9 | 0.8 | 19.0 | 4.8 | 4.8 | 6.2 | 4.8 | 5.1 | 4.8 | 5.1 | 55.1 |
| S5 | 9 | 0.7 | 16.1 | 4.5 | 4.5 | 5.7 | 4.5 | 6.1 | 4.5 | 5.1 | 51.0 |
| S6 | 9 | 0.6 | 18.9 | 5.3 | 5.3 | 7.2 | 5.3 | 6.2 | 5.3 | 5.3 | 65.1 |
| S7 | 9 | 0.6 | 10.8 | 3.0 | 3.0 | 3.9 | 3.0 | 5.1 | 3.0 | 5.1 | 35.0 |
| S8 | 9 | 0.8 | 11.2 | 3.0 | 3.0 | 4.3 | 3.0 | 5.1 | 3.0 | 5.1 | 38.5 |
| S9 | 9 | 0.8 | 11.2 | 2.3 | 2.3 | 3.8 | 2.3 | 5.1 | 2.3 | 5.1 | 34.2 |
| S10 | 9 | 0.9 | 11.2 | 3.0 | 3.0 | 4.2 | 3.0 | 5.1 | 3.0 | 5.1 | 37.8 |
| S11 | 9 | 0.9 | 11.6 | 2.4 | 2.4 | 3.8 | 2.4 | 5.1 | 2.4 | 5.1 | 34.4 |
| S13 | 9 | 0.8 | 11.2 | 3.5 | 3.5 | 4.4 | 3.5 | 5.1 | 3.5 | 5.1 | 40.0 |
| S14 | 9 | 0.4 | 11.7 | 3.0 | 3.0 | 4.4 | 3.0 | 5.1 | 3.0 | 5.1 | 39.2 |
| S15 | 9 | 0.8 | 11.1 | 3.0 | 3.0 | 3.9 | 3.0 | 5.1 | 3.0 | 5.1 | 34.7 |
| S16 | 9 | 0.8 | 11.7 | 3.5 | 3.5 | 4.5 | 3.5 | 5.1 | 3.5 | 5.1 | 40.9 |
| S17 | 9 | 0.9 | 11.1 | 4.9 | 4.9 | 5.1 | 4.9 | 5.1 | 4.9 | 5.1 | 45.7 |

> **Nota:** S2 es un caso especial: solo tiene 8 gaps (no 9) y no presenta label 5. Sus gaps son más largos (media 6,4 min vs ~4–5 min del resto).

Tras cada fase se recogieron autoinformes que son son la "opinión subjetiva" del participante sobre su estado afectivo, (entre ellos **PANAS**, y en la descripción original también aparecen STAI, SAM y SSSQ).

> **Nota metodológica:** el lapso entre condiciones **no es un tiempo fijo**; varía por sujeto y queda capturado como etiqueta 0 transitoria. Las duraciones de cada condición arriba son las reportadas por el paper original (Schmidt et al., 2018) y confirmadas en los datos etiquetados.

### Señales que recoge WESAD

| Canal | Ubicación | Tasa aprox. | Unidades | Uso en esta tesis |
|---|---|---|---|---|
| ECG | pecho | 700 Hz | mV | Sí (HR, HRV) |
| EDA (electrodermal) | pecho | 4 Hz | µS | Sí (tonic, phasic, SCR) |
| Respiración | pecho | 700 Hz | — | Sí (rate, ciclos) |
| Temperatura | pecho | 700 Hz | °C | **Canónica: sí; primario LLM: no.** Es un confusor del orden del protocolo verificado empíricamente (r=0,801 posición vs temperatura) y la única modalidad de pecho que no se degrada; el experimento primario la excluye de la evidencia del LLM. El few-shot nunca la incluye. |
| ACC | muñeca | 700 Hz | g | No (confusor de movimiento; excluido de la vista LLM) |
| EMG | muñeca | 700 Hz | — | No (confusor de movimiento; excluido) |
| BVP | muñeca | 64 Hz | — | No (canal de muñeca; excluido de la vista LLM principal) |
| EDA | muñeca | 4 Hz | µS | No (vista LLM usa solo pecho) |
| TEMP | muñeca | 4 Hz | °C | No |

### Señales utilizadas en el experimento

Usamos exclusivamente los canales de **pecho**. La representación **canónica** conserva **23 features** (`LLM_ALLOWED_FEATURES`) de ECG, EDA, respiración y temperatura. El **experimento primario** excluye la temperatura de la evidencia del LLM y trabaja con **19 features** de ECG, EDA y respiración (`--temperature exclude`); las sensibilidades declaradas pueden incluirla. El few-shot nunca incluye temperatura.

### Exclusión de confusores

Se excluyen las siguientes señales:
- ACC/EMG: ACC (acelerómetro de muñeca): mide aceleración en 3 ejes → detecta movimiento físico del brazo. EMG (electromiograma de muñeca): mide actividad eléctrica muscular → detecta contracción muscular. Si ACC o EMG se incluyen en la representación que ve el LLM, el modelo podría predecir "estrés" basándose únicamente en el movimiento físico, no en la evidencia fisiológica afectiva (EDA, ECG, respiración)
- orden del protocolo (balanced 0,857 en 4 clases usando solo la fase, sin fisiología);

- Temperatura de pecho como confusor del orden del protocolo. La temperatura de pecho aumenta sistemáticamente a lo largo del protocolo (r = 0,801 de media), independientemente de la condición afectiva. Como el protocolo es secuencial (no aleatorio por ventana), la temperatura está correlacionada con **cuándo** aparece cada condición, no solo con **qué** condición es. Versiones A y B del protocolo: El orden de las condiciones varía entre sujetos (versión A: baseline→amusement→med1→stress→rest→med2; versión B: baseline→stress→rest→med1→amusement→med2). Este contrabalanceo mitiga pero no elimina el confusor de orden, ya que el stress aparece en posición 2 (versión B) o posición 4 (versión A), y la temperatura difiere sistemáticamente entre esas posiciones. **En la práctica:** el corpus exploratorio (4 clases y E12) se generó con temperatura en la evidencia; el **experimento primario la excluye** (`--temperature exclude`) porque, además de confusor, es la única modalidad de pecho que **no** se degrada y podría amortiguar el efecto de C1/C2/C4/C5. La omisión en el few-shot no elimina el confusor: el modelo podía usar la temperatura en la evidencia. La separación limpia es: el LLM primario ve `ecg`, `eda` y `resp`; el análisis de resolubilidad conserva `temp` en el corpus canónico para **cuantificar** el confusor.

### Por qué se utilizan estas señales

1. **Canales separables para degradación controlada (H1).** EDA y respiración del pecho pueden anularse o degradarse independientemente sin destruir todo el registro, lo que permite medir el coste de cada modalidad por separado.
2. **EDA y ECG son las más informativas para el estrés.** La referencia features-only con las mismas 23 features separa stress vs baseline con balanced ≈ 0,87–0,92, de modo que cualquier error en la respuesta del LLM se debe a su interpretación de la evidencia y no a la falta de señal en los datos de entrada.
3. **Etiquetas de protocolo conocidas.** Cada ventana tiene *ground truth* (baseline / stress / amusement / meditation), lo que permite evaluar corrección, abstención y respaldo de claims sin depender de anotación subjetiva continua.
4. **Tamaño manejable y reproducible.** 15 sujetos y 705 ventanas permiten ejecutar la matriz completa de experimentos en horas, y reconstruir el corpus en minutos mediante fingerprints de contenido.



### Relevancia bibliográfica de WESAD

A continuación se recogen las declaraciones de relevancia del dataset WESAD tanto en el paper original de Schmidt et al. (2018) como en los papers secundarios del corpus SLIP que lo utilizan como dataset primario o secundario.

#### Schmidt et al. (2018) — paper original de WESAD

> "In this paper we present a novel dataset for stress and affect detection. The subjects (n = 15) were exposed to different affective stimuli (stress and amusement). In addition, a baseline and two meditation periods (introduced to de-excite the participants after a stimulus) were recorded. The dataset contains high resolution physiological (ECG, EDA, EMG, RESP, and TEMP) and motion (ACC) data sampled at 700 Hz from a chest-worn device, and lower resolution data from a wrist-worn device. Finally, the data of each subject is linked to several self-reports, which represent the subjective experience during an affective stimulus. The dataset is well-suited to benchmark (personalised) stress and affect detection algorithms, a first evaluation is presented in this paper."
>
> — *Schmidt et al. (2018), Introducing WESAD, ICMI'18*

> "(1) A new multimodal, publicly available dataset is presented. The data has been recorded using two different devices (one chest-based and one wrist-based), each including high-resolution physiological (BVP, ECG, EDA, EMG, RESP, and TEMP) and motion (ACC) modalities.
> (2) The dataset bridges the gap between previous lab studies on stress and emotions, by containing three different affective states (neutral, stress, amusement). In addition, the dataset features self-reported values on the perceived affective state of the subjects, which were obtained using several established questionnaires. These self-reports can be used to train personalised classifiers.
> (3) A benchmark is created using a large amount of well-known features (extracted from physiological and motion signals)"

#### Papers que usan WESAD como dataset primario

| Paper | Año | Qué dice de la relevancia de WESAD |
|---|---|---|
| `ieee_88` | 2026 | "An empirical study for effective LLM based human state prediction from physiological data." Usa WESAD junto con HHAR y DREAMT para evaluar combinaciones de representación de entrada, uso de LLM y forma de salida en predicción de estados humanos. Concluye que feature input + few-shot + label-with-explanation fue la configuración efectiva. |
| `scopus_111` | 2025 | "StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data." Usa WESAD (junto con PMData) para identificar biomarcadores óptimos para predicción de estrés con LLM y ablación de características. Evalúa zero-shot, few-shot y Chain-of-Thought prompting sobre WESAD. |
| `pubmed_150` | 2026 | "Seamless monitoring of stress levels leveraging a foundational model for time sequences." Usa WESAD (junto con DREAMER, MAHNOB-HCI y @HOME) para detección de estrés como anomalía personalizada con UniTS y explicaciones LLM. Reporta resultados para WESAD (ECG y BVP) por separado. |
| `arxiv_51` | 2026 | "Beyond Latent Personalization: Integrating HRV-Conditioned Personalization via LLM for Interpretable Stress Detection." Usa WESAD (junto con TILES-2018) para validar la generalización de su framework de personalización por HRV. Declara que "its performance on WESAD supports its generalizability across datasets". |

#### Papers que mencionan WESAD en surveys y reviews

| Paper | Tipo | Mención de WESAD |
|---|---|---|
| `scopus_293` | Survey | "A survey on large language models for medical time series." Incluye WESAD entre los datasets de series temporales médicas revisados (ECG, PPG/BVP, signos vitales, wearables). |
| `wos_386` | Scoping review | "A scoping review of large language models in personal wellness." Menciona WESAD entre los 21 estudios revisados en bienestar personal y sueño. |


## 1. Papers que usan WESAD

**Escala de similitud** (`Sim.`):

- **A1** — usa **WESAD**. Similitud máxima: mismo dato que esta tesis.
- **A2** — señal fisiológica **y** ejes de la tesis (degradación, ruido, calidad,
  incertidumbre, calibración, abstención, claims, fiabilidad).
- **M** — señal fisiológica con LLM en otra tarea, o métodos de
  incertidumbre/calibración/abstención en texto (transferibles).
- **B** — sin señal fisiológica ni ejes de incertidumbre.


| Paper | Título | Dataset(s) | Señal | Objetivo | Resultado | Citas | Rel. | Sim. |
|---|---|---|---|---|---|---:|---:|:--:|
| `scopus_111` | StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data | PMData; WESAD | Datos fisiológicos/wearables (HR, sueño, actividad, EDA, ECG, etc.) | Identificar biomarcadores óptimos para predicción de estrés con LLM y ablación de características | El tamaño del modelo no siempre correlaciona con accuracy; edad ayuda; género poco; biomarcadores óptimos varían por dataset/modelo | 14 | 0.76 | A1 |
| `pubmed_150` | Seamless monitoring of stress levels leveraging a foundational model for time sequences | DREAMER; MAHNOB-HCI; WESAD; @HOME | ECG, BVP/PPG, HR/HRV, señales ambientales | Detección de estrés como anomalía personalizada con UniTS y explicaciones LLM | UniTS supera 12 métodos; F1 medio 0.821; en @HOME 30/32 anomalías confirmadas; robusto con wearables | 2 | 0.76 | A1 |
| `scopus_293` | A survey on large language models for medical time series | Survey: **MIMIC-III, MIMIC-IV, eICU-CRD, HiRID, AmsterdamUMCdb, OhioT1DM, VitalDB, WESAD, PPG-DaLiA, TROIKA, QT Database, MIT-BIH Arrhythmia, Apnea-ECG, Sleep-EDF, CHB-MIT**; PhysioNet benchmark | ECG, EEG, PPG/BVP, signos vitales, wearables | Revisar LLMs para series temporales médicas | Taxonomía, estrategias de prompting/evaluación, retos de datos, multimodalidad, eficiencia, interpretabilidad y privacidad | 2 | 0.75 | A1 |
| `wos_386` | A scoping review of large language models in personal wellness | **21 estudios revisados** (sleep education, OSA, CBT-I, coaching) — Revisión; menciona WESAD entre los estudios revisados. | Señales de sueño/wearables (Fitbit, Oura, PSG) | Revisión scoping de LLMs en bienestar del sueño | Evidencia preliminar; evaluaciones a corto plazo, expert-rated o proxy-based | 1 | 0.75 | M |
| `ieee_88` | An empirical study for effective LLM based human state prediction from physiological data | WESAD; HHAR; DREAMT | ECG, datos de sensores fisiológicos/wearables | Evaluar combinaciones de representación de entrada, uso de LLM y forma de salida en predicción de estados humanos | Feature input + few-shot + label-with-explanation fue la configuración efectiva; diferencias individuales y temporales afectan el rendimiento | 0 | 0.77 | A1 |
| `arxiv_51` | Beyond Latent Personalization: Integrating HRV-Conditioned Personalization via LLM for Interpretable Stress Detection | TILES-2018; WESAD | HRV | Integrar HRV y atributos personales mediante interpretación semántica LLM para detección de estrés | MCC 0.261 en TILES-2018; +29.9% frente a baseline más fuerte; ablation confirma HRV, atributos y embeddings | — | 0.75 | A1 |

**Cuatro** papers usan WESAD como dataset primario (`ieee_88`,
`scopus_111`, `pubmed_150`, `arxiv_51`), uno lo incluye en una revisión de
series temporales médicas (`scopus_293`) y otro lo menciona entre los estudios
de una revisión scoping (`wos_386`). Ninguno evalúa el efecto de la degradación
de la señal de forma controlada: ese hueco es el de esta tesis.

### 1.1 Papers prioritarios

1. `ieee_88` — WESAD + HHAR + DREAMT: qué representación de entrada y qué
   configuración de few-shot funcionan para predecir estados humanos.
2. `scopus_111` — StressLLM: biomarcadores óptimos y efecto del tamaño del
   modelo en WESAD y PMData.
3. `pubmed_150` — estrés como anomalía personalizada con explicaciones LLM
   (WESAD, DREAMER, MAHNOB-HCI).
4. `arxiv_51` — personalización por HRV para detección de estrés interpretable
   (TILES-2018, WESAD).
5. `scopus_293` — revisión de LLM sobre series temporales médicas; sitúa WESAD
   en el mapa.
6. `arxiv_288` / `manual_2` — **el más cercano a H5/H2**: sobreconfianza en
   medidas derivadas (PPG–ECG) y reparación con evidencia de fiabilidad.
7. `manual_5` — **el más cercano a H1**: degradación de telemetría y estabilidad
   secuencial (FlipRate) en un sistema de decisión con LLM.
8. `manual_3` — PhysioBench: techo actual del QA sobre señales fisiológicas.
9. `arxiv_7` — revisión de LLM en series temporales médicas: ruido, calidad y
   evaluación.
10. `scopus_68` — señales cognitivas como proxy para alinear y mitigar
    alucinaciones.

---

## 2. Resto de papers

Orden: similitud (A2 → M → B) y, dentro de cada grupo, citas descendentes.

| Paper | Título | Dataset(s) | Señal | Objetivo | Resultado | Citas | Rel. | Sim. |
|---|---|---|---|---|---|---:|---:|:--:|
| `scopus_68` | Integrating cognitive processing signals into language models: a review of advances, applications, and future directions | **Revisión** (ZUCO dataset: EEG+ET; múltiples corpora ET) | **Eye-tracking (ET), EEG, fMRI** | Revisión de integración de señales cognitivas en LMs/MLLMs | Identifica ET como proxy prometedor para data augmentation, alineación humana y mitigación de hallucinations | 3 | 0.74 | A2 |
| `arxiv_288` | When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence | MIMIC-III Matched Waveform Database v1.0: 50,000 registros PPG–ECG de 1,275 pacientes; test bloqueado de 187 pacientes | PPG y ECG | Definir y cuantificar sobreconfianza en medidas derivadas (DFOT) y evaluar reparación con evidencia de fiabilidad | K2 mejora CRR/ESRM y especificidad en 1.82–6.69 pp; UHR +0.67 pp; intervalos pareados excluyen cero | — | 0.77 | A2 |
| `arxiv_7` | Large language models in medical time series analysis | Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc. | ECG, EEG, PPG, signos vitales, wearables | Revisar arquitecturas, prompting, aplicaciones y evaluación de MedTSLLM | Taxonomía y guías clínicas; retos de alucinación, datos, benchmarks, razonamiento mecanístico, cómputo y regulación | — | 0.75 | A2 |
| `manual_2` | When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence | **MIMIC-III Matched Waveform Database v1.0** (50,000 registros PPG–ECG sincronizados de 1,275 pacientes; 36,115 train / 6,464 val / 7,421 test de 187 pacientes) | **PPG + ECG** (ritmo cardiaco) | Definir y cuantificar Derived-Feature Over-Trust (DFOT); 5 estimandos (COTR, CIR, CRR, ESRM, UHR) | K2 vs B2: mejoras de **1.82–6.69 pp** en 4 endpoints de repair/specificity; todos los IC pareados excluyen cero; UHR +0.67 pp (IC −0.4 a +1.7) | — | 1.00 | A2 |
| `manual_3` | PhysioBench: A Unified Benchmark for Physiological Signal Question Answering | **PhysioBench** (22 datasets públicos, **61,418,821 preguntas**, 30 tareas, 240,171 sujetos) — PSG 74.76%, ECG 23.19%, EEG 1.90%, PPG 0.15% | **ECG, PPG, EEG, PSG** | Benchmark unificado de QA sobre señales fisiológicas | 21 modelos evaluados: **ninguno robusto**; zero-shot LLMs/VLMs/TSLMs cerca o bajo azar; PSFMs con encoder congelado superan azar pero con alta heterogeneidad por tarea | — | 1.00 | A2 |
| `manual_5` | Telemetry degradation and sequential stability in LLM-integrated IoT decision systems | **AI-tennis coaching case study** (telemetría sintética de smartwatch, 40 features IMU, ventanas de 20 golpes) | **IoT telemetry / smartwatch IMU** (biomecánica) | Evaluar estabilidad conductual de LLM como capa de decisión bajo degradación de telemetría (dropout, downsampling, reliability signaling) | FlipRate: **0.144 (clean) → 0.207 (downsampling, +44%)** en régimen balanceado; **0.300** en régimen serve-dominant; ViolationRate=0 (modelo open-loop); structured prompting mitiga parcialmente | — | 1.00 | A2 |
| `scopus_198` | Multimodal LLMs for health grounded in individual-specific data | **UK Biobank** (50,000 registros; 36,115 train / 6,464 val / 7,421 test de 187 pacientes) | Datos tabulares clínicos + **spirogram** (time-series de función pulmonar) | Estimación de riesgo de enfermedad con LLM multimodal (HeLM) | **AUROC 0.75 para asma** con tabular+spirogram vs 0.49 solo tabular; supera o iguala a LogReg/XGBoost en 8 traits | 58 | 0.74 | M |
| `wos_308` | A personal health large language model for sleep and fitness coaching | **PH-LLM case studies**; **629 MCQs sueño**; **99 fitness**; **Google DWB/PROMIS**; 833 survey responses; ratings 4.265/5.049 y 2.606/3.335 | Wearables: sueño, actividad, HR, HRV, respiración | Evaluar PH-LLM en exámenes, coaching y PROs | 79% sueño, 88% fitness; case studies cercanos a expertos; PROs con adapter comparable a modelos especializados | 57 | 0.76 | M |
| `pubmed_168` | Integrating large language model, EEG and eye tracking for word-level neural state classification | **ZuCo 1.0 Task 3** | EEG + eye-tracking | Clasificar relevancia de palabras con etiquetas LLM | >60% accuracy; HRW más fijaciones; primer uso de LLM para etiquetas word-level | 40 | 0.74 | M |
| `scopus_197` | EmoSavior: depression recognition and intervention via multimodal physiological signals and LLMs | Dataset propio de 200 participantes; EEG, ECG, GSR; PHQ-9; estímulos IADS-2 | EEG, ECG, GSR | Reconocimiento de depresión e intervención cerrada con LLM | Accuracy 0.7667; F1 0.7666; supera baselines y SOTA; ablación confirma componentes | 22 | 0.76 | M |
| `scopus_80` | Improving dietary supplement information retrieval: RAG with LLMs | **iDISK2.0**; fuentes **NMCD, MSKCC, DSLD, LNHPD**; preguntas derivadas de **MSKCC** | No fisiológica; texto/knowledge graph | RAG sobre knowledge base de suplementos dietéticos para QA | RAG >95% accuracy; GPT-4o 62% y 52% en true/false y MCQ de interacciones; reduce hallucinations | 16 | 0.75 | M |
| `ieee_430` | Speech recognition in a smart home: some experiments for telemonitoring | **HIS (Health Smart Home) Grenoble** (47 m², 8 micrófonos); **Normal/Distress corpus** (126 frases en francés: 66 normales, 60 de distress) | Audio/voz (no señal fisiológica corporal, pero señal acústica biomédica) | Reconocimiento de voz y sonidos para telemonitoreo de personas mayores | Global Error Rate **15.6%** (10 sujetos, 429 frases) | 15 | 0.74 | M |
| `ieee_676` | Multi modal integration analysis of Alzheimer’s disease using LLMs and knowledge graphs | Cinco datasets públicos independientes (MRI, EEG, biomarcadores, clínicos, expresión génica); tres cohortes de validación independientes (n=28–45) | MRI, EEG, biomarcadores, datos clínicos, expresión génica | Integrar datos fragmentados de Alzheimer sin matching de IDs mediante grafos y LLM | Encuentra relaciones cruzadas; ruta metabólico-inflamatoria-tau con r>0.6; correlaciones EEG-gen r=0.42–0.58; validación con efectos consistentes | 10 | 0.76 | M |
| `scopus_118` | Multimodal large language models in human-centered health: practical insights | **Ocho datasets públicos de sensores inerciales**; se nombran **mHealth, DLA, ComplexHAD, DSADS, FLAAP, RealWorld, PAMAP2**; el octavo no se nombra en el extracto | Acelerómetro/giroscopio wearable | Dual-view FM para forecasting de señales inerciales | Supera TimeGPT-1 hasta 36% MAE; fine-tuning >7% MAE; mejora clasificación downstream | 9 | 0.74 | M |
| `acl_anthology_8` | Evaluating Subjective Cognitive Appraisals of Emotions from Large Language Models | **COVIDET-APPRAISALS** (241 posts de Reddit; 24 dimensiones de appraisal; 21 aplicables) | — (texto) | Evaluar capacidad de LLMs para predecir cognitive appraisals y generar rationales | ChatGPT: ρ=0.388, MAE=1.694; open-source LLMs (Alpaca, Dolly) con correlaciones negativas; ChatGPT ≈ anotadores humanos en calidad de rationales | 8 | 0.74 | M |
| `ieee_598` | Artificial intelligence for speech classification and enhancement of speech and language disorders | Revisión sistemática: **TORGO, UA-Speech, PC-GITA, SVD/Saarbruecken, LAANA, Fluency Bank, SEP-28k, KSoF, MEEI, NEMOURS, TCC-300, NCMMSC2021, TIMIT, LibriSpeech** | Habla/audio (acústica, prosódica, lingüística); no señal fisiológica única | Revisar IA/ML/DL para clasificación y mejora de trastornos del habla y lenguaje | Síntesis de 231 estudios; progresión SVM→DL; brechas en datasets longitudinales, multilingües y explicabilidad; mejora del habla como preprocessing | 8 | 0.75 | M |
| `scopus_97` | A novel chain of thought reasoning approach for Alzheimer’s disease detection using LLM and vision-language models | ADReSS / DementiaBank Pitt Corpus (descripción de imagen “cookie theft”) | Voz/audio, imagen, texto | Detectar Alzheimer con ASR, VQA y Chain-of-Thought durante fine-tuning | Accuracy 86.67%; F1 86.34%; CoT mejora relativa 16.7% | 8 | 0.76 | M |
| `pubmed_128` | On the limits of LLM surprisal as a functional explanation of the N400 and P600 | Tres datasets ERP controlados de los Estudios 1–3 (no nombrados) | EEG (N400/P600) | Evaluar si el surprisal de LLM explica N400/P600 y si está contaminado por asociación | El surprisal LLM no captura consistentemente ambos componentes; está contaminado por asociación, sobre todo en modelos pequeños; propone P600 como mejor índice de surprisal verdadero | 7 | 0.77 | M |
| `acl_anthology_2` | Psycholinguistic Diagnosis of Language Models' Commonsense Reasoning | **Singh et al. (2016) stimuli** (82 items de presuposición); **Nieuwland et al. (2010) data** (168 items de implicatura escalar); **ImpPress dataset** (>25k pares; 14,100 Presp + 6,000 SI) | — (behavioral/EEG-ERP referenciado, no recolectado) | Evaluar razonamiento pragmático de LMs (presuposición) | DistillBERT detecta Presp; fine-tuning no mejora | 4 | 0.74 | M |
| `acl_anthology_57` | Teaching LLMs to Abstain across Languages via Multilingual Feedback | **M-MMLU** (Multilingual MMLU); **M-Hellaswag**; **Belebele** (apéndice) | — (texto QA) | Abstención multilingüe vía feedback en lenguajes relacionados | Hasta **9.2% mejora** en lenguajes de bajos recursos; reducción de Gini (equidad) 12.9% | 4 | 0.74 | M |
| `acl_anthology_148` | Do language models mirror human confidence? Exploring psychological insights to address overconfidence in LLMs | **MMLU** (high school/college; physics, chemistry, biology) y **GPQA** (expert) | No fisiológica | Analizar calibración/confianza y proponer AFCE | AFCE reduce ECE; LLMs menos sensibles a dificultad; sesgos por personas demográficas; GPT-4o más balanceado | 3 | 0.75 | M |
| `pubmed_187` | MedTsLLM: Medical Time Series Analysis Using Multimodal LLMs | No se nombran datasets en el extracto; tareas de ECG, monitoreo respiratorio y arritmias | ECG, señales respiratorias, series temporales fisiológicas | Integrar señales fisiológicas con contexto clínico textual mediante patch reprogramming y prompting contextual | Supera baselines en segmentación semántica, detección de límites, anomalías y clasificación; encoder-decoder y decoder-only tienen ventajas según tarea | 3 | 0.75 | M |
| `pubmed_25` | Low frequency neural activity tracks syntactic information through semantic mediation | **EEG Mandarin frequency-tagging**: 25 participantes; 4 condiciones; 30 trials/cond | EEG | Examinar tracking sintáctico mediado por semántica y simulación GPT-2 | Picos 1 Hz en antónimos, living-nonliving y NV-sensical; no en NV-nonsensical; 2 Hz en todas; GPT-2 con limitaciones | 3 | 0.74 | M |
| `scopus_180` | Guiding LLMs to decode text via aligning semantics in EEG signals and language | **ZuCo benchmark** (ZuCo 1.0 + 2.0) | EEG; eye-tracking para segmentar palabras | EEG-to-text como generación controlada con re-ranking por LLM | SOTA en multi-sujeto, single-sujeto y cross-subject; mejores BLEU/ROUGE/BERTScore que baselines | 3 | 0.74 | M |
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | **TriviaQA**; **NQ-Open**; **PopQA** | — (texto QA) | Detectar y mitigar hallucinations calibrando incertidumbre verbal | Reducción relativa media de ~30% en hallucinations confiadas | 2 | 0.75 | M |
| `acl_anthology_34` | Can LLMs understand unvoiced speech? Exploring EMG-to-text conversion with LLMs | **Gaddy & Klein unvoiced EMG**: 1 hablante, 8 canales, 67 palabras, ~26 min, 500 utterances; **CSL-EMG_Array** para person ID | EMG no vocalizado | Convertir EMG no vocalizado a texto sin audio/voz | WER 0.49; con 6 min mejora ~20% a modelos especializados; person ID 96% | 2 | 0.76 | M |
| `acl_anthology_56` | Decompose and Compare Consistency: Measuring VLMs' Answer Reliability via Task-Decomposition Consistency Comparison | **SNLI-VE**; **VCR**; **A-OKVQA**; **Winoground**; **MMMU**; **MathVista** | — (visión-lenguaje) | Medir fiabilidad de respuestas de VLMs | DeCC mejora Brier Score y Effective Reliability | 2 | 0.74 | M |
| `scopus_272` | LLM-assisted explainable daily stress recognition: physiologically grounded threshold rules from PPG features | **Can et al. (2020) multimodal physiological database** (14 estudiantes, TSST + 12h/día durante 1 semana, ~1000h, 388 EMA) | **PPG, HRV (RMSSD, HF, pNN50, LF/HF, stdV), EDA, temperatura cutánea, aceleración** (Empatica E4) | Detección de estrés explicable con reglas de umbral justificadas por LLM | **~75% accuracy binaria** (75.4%); comparable a CNN/LSTM/Transformer (75.8–78.1%) pero totalmente interpretable | 2 | 0.74 | M |
| `scopus_401` | Reducing annotation burden in physical activity research using vision language models | CAPTURE24 Oxfordshire (161 participantes) y CAPTURE24-CN Sichuan (111 participantes) | Imágenes de cámara wearable y acelerometría de muñeca | Clasificar intensidad de actividad física con VLMs y modelos discriminativos | SB bien predicho; LIPA/MVPA peor; caída fuerte al generalizar a Sichuan; modelos útiles para reducir anotación, no superan anotadores humanos | 2 | 0.77 | M |
| `scopus_460` | Harnessing LLM for noise-robust cognitive diagnosis in web-based intelligent education systems | **ASSIST0910** (2,485 estudiantes, 16,818 ejercicios, 263,209 logs); **Junyi1808** (2,619 estudiantes, 18,047 ejercicios); **Eedi50** (4,918 estudiantes, 948 ejercicios, 1.38M logs) | — (logs de respuesta, no fisiológico) | Diagnóstico cognitivo robusto a ruido con LLM + difusión | DLLM-NCDM: ACC 74.46%, AUC 77.92%, F1 82.06% | 2 | 0.74 | M |
| `acl_anthology_231` | ClinDet-Bench: Beyond abstention, evaluating judgment determinability of LLMs in clinical decision-making | **ClinDet-Bench**: 16 sistemas de scoring; 188 casos; 8 LLMs | No fisiológica | Evaluar determinabilidad bajo información incompleta | LLMs fallan: conclusiones prematuras y abstención excesiva; imputan información faltante (75.9% errores) | 1 | 0.75 | M |
| `ieee_205` | Using generative AI for neurofeedback content personalization | **OpenNeuro ds002718, ds002778, ds003064** | EEG | Framework LLM para neurofeedback personalizado | Framework propuesto; experimentos preliminares muestran eficacia en selección de features y diferenciación de casos | 1 | 0.74 | M |
| `ieee_392` | NarratingSleep: a language model based framework for sleep staging | **Sleep-EDF SC-EDF-20** (20 sujetos); **Sleep-EDF SC-EDF-78** (78 sujetos); **ISRUC-Sleep** (100 sujetos) | **EEG** (PSG) | Sleep staging usando LM preentrenado (cross-modal transfer) | **ACC 85.72%, MF1 81.65** | 1 | 0.74 | M |
| `ieee_828` | Predicting fall risk in community dwelling older adults using a fine tuned quantized large language model | Cohorte prospectiva de 206 adultos mayores comunitarios (217 basal) a 6 meses; posturografía, cuestionarios, pruebas físicas, sensores inerciales/plataforma de presión | Posturografía, acelerometría/inercial, datos clínicos | Predecir riesgo de caída con LLM cuantizado fine-tuned (QLoRA) | Modelo I con LLM: accuracy 0.86; recall 0.92; especificidad 0.83; AUC 0.88; supera ML tradicional | 1 | 0.75 | M |
| `ieee_886` | Counterfactual modeling with fine tuned LLMs for health intervention design and sensor data augmentation | AI-READI Flagship Dataset (1,067 participantes; CGM, actividad, sueño, estrés, sensores ambientales) | Datos fisiológicos, conductuales y ambientales de wearables | Generar contra factuales clínicamente plausibles y usarlos para augmentación bajo escasez de etiquetas | LLaMA-3.1-8B fine-tuned: plausibilidad hasta 99%, validez hasta 0.99; recuperación media ~20% F1 en escasez de etiquetas | 1 | 0.76 | M |
| `pubmed_106` | Validating large language model assisted data extraction from clinical notes | 60 pacientes; 1,482 páginas; 29 categorías; 2,555 valores; cáncer de cabeza y cuello | N/A (notas clínicas) | Validar extracción de datos clínicos por LLM vs extracción manual | Accuracy 74–90%; tiempo 8.6 → 1.9 min; alucinaciones 0.16%; 68 errores de alto impacto | 1 | 0.75 | M |
| `pubmed_161` | LLM powered dysphagia screening with multimodal physiological signal analysis and medically informed prompts | Dataset clínico propio: 217 participantes (109 con disfagia post-ictus, 108 sanos); 1,391 muestras disfágicas y 1,273 sanas | Vibración laríngea triaxial, flujo nasal, sonido de deglución | Screening de disfagia con señales multimodales y prompts médicos | Accuracy 96.3%; sensibilidad 95.2%; especificidad 97.4%; estable en few-shot | 1 | 0.76 | M |
| `pubmed_178` | Sleep disorder analysis using large language models | **CAP Sleep Database; Sleep Accel Dataset; Sleep Bioradiolocation Database; Sleep-EDF Expanded; Sleep Heart Health Study PSG; HMC Sleep Database; Charité University Hospital Berlin; NCH Sleep DataBank** | EEG/PSG, stages de sueño | Fine-tuning BERT/Longformer para clasificar sueño sano/no sano | Accuracies 70–80%; BERT mejor en secuencias cortas, Longformer en largas; overfitting; no clasifica trastornos específicos | 1 | 0.74 | M |
| `pubmed_28` | Resting state fMRI foundation models enable robust and generalizable latent neural target discovery in cognitive aging interventions | ADNI (fine-tuning); ACT; CogTE; pretraining de FMs en UK Biobank/HCP | rsfMRI/BOLD | Evaluar foundation models de rsfMRI para clasificar respuesta de memoria episódica a intervenciones | BrainLM supera SVM/MLP/GCN; BrainLM-ADNI mejora ACC/F1; robusto a sitio y movimiento; RSA alinea embeddings con redes cerebrales | 1 | 0.75 | M |
| `pubmed_39` | Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands | Comandos de voz de 35 adultos mayores; 15 con interacción diaria en casa durante 18 meses | Voz/audio y transcripciones | Detectar deterioro cognitivo leve mediante comandos de voz longitudinales y refinamiento iterativo de prompts | Accuracy 73.80%; F1 72.67%; mejora 27.13% sobre baseline | 1 | 0.77 | M |
| `scopus_223` | Large language models in sports cardiology: challenges, governance and human in the loop integration | No dataset primario; revisión narrativa | ECG, imagen cardiovascular, síntomas, historia familiar | Revisar rol de LLM en cardiología deportiva y gobernanza | Propone paradigma text-proxy; LLM como capa semántica supervisada, no diagnóstico autónomo; validación por etapas | 1 | 0.76 | M |
| `scopus_355` | Mind the data: evaluating data quality sensitivity in medical LLMs | LQMR: Heart Disease, Heart Failure, PIMA, ILPD, Vertebral; MIMIC-IV-ED: Hospitalization outcome y Critical outcome | N/A (EHR estructurado/tabular) | Evaluar sensibilidad de LLM a datos médicos de baja calidad | AUC cae hasta 16.79% con anomalías de plausibilidad; LLM detecta mal anomalías de plausibilidad; prompting simple no mitiga | 1 | 0.75 | M |
| `scopus_484` | Graph representations for reading comprehension analysis using LLM and eye tracking biomarker | ZuCo 1.0 (12 sujetos; 21,629 palabras; 1,107 frases; 154,173 fijaciones) | EEG y eye-tracking | Analizar comprensión lectora con grafos generados por LLM y validar con fijaciones | PageRank obtiene las mejores AUC (hasta 0.651); nodos importantes reciben más fijaciones (diferencia media 0.291) | 1 | 0.75 | M |
| `scopus_500` | Lightweight vision language model guided gesture recognition based on electromyography | **EMG pretraining dataset** (33 sujetos); **4 nuevos participantes**; **Flickr30k** para VLM | EMG + visión | Pseudolabeling con VLM para decodificación EMG | 100% decoding accuracy en 4 gestos/4 participantes bajo distribution shift; VLM reliability 1 vs MVC 0.8 | 1 | 0.74 | M |
| `scopus_69` | 2-D GAF-enhanced multimodal vision language model for breathing patterns analysis via ECG sensing | **Long-Term Infant Cardio-Respiratory Signals DB; Stress Recognition in Automobile Drivers DB; Fantasia DB; Combined ECG/breathing/seismocardiogram; ECG spider video clips; UoS ECG test** | ECG | Clasificar patrones respiratorios con VLLM sobre GAF 2D | Idefics BLEU 0.572 (2D) vs 0.549 (1D); Florence 0.626 en 1D; GAF mejora visualización | 1 | 0.74 | M |
| `acl_anthology_245` | Speech disfluencies and LLM confidence: length bias and pragmatic insensitivity in Brazilian Portuguese | **Roda Viva corpus**: 344 turnos | Habla/transcripciones (no fisiológica) | Estudiar confianza LLM vs incertidumbre pragmática | Longitud de turno domina confianza (β=+14.47); marcadores pragmáticos no significativos; ECE 41.95/41.14 | 0 | 0.76 | M |
| `acl_anthology_286` | Vulnerability of LLMs’ stated belief? LLMs belief resistance check through strategic persuasive conversation interventions | **BoolQ; PubMedQA; LatentHatred** | No fisiológica | Evaluar vulnerabilidad a persuasión multi-turno | Vulnerabilidad alta y desigual; confianza verbalizada aumenta vulnerabilidad; fine-tuning adversarial dependiente del modelo | 0 | 0.75 | M |
| `acl_anthology_288` | When LLMs Disagree with Human Experts: Understanding LLM Annotation Failures in Nutrition Misinformation | **169 posts de Instagram sobre seed oils** (anotados por 3 nutricionistas, Fleiss' κ=0.813) | — (texto de redes sociales) | Evaluar LLMs como anotadores de desinformación en salud | Taxonomía jerárquica de errores; sobreconfianza sistemática (149/169 errores con confianza ≥0.8) | 0 | 0.74 | M |
| `acl_anthology_303` | Can a remedy find a researcher? Exploring semantic knowledge in Italian BabyLMs | **Benchmark italiano de pares mínimos**: 272 AI, 332 AA, 282 AA-rev; validación con 232 participantes; modelos BAMBI y LMs italianos | No fisiológica | Evaluar competencia semántica de BabyLMs italianos | BabyLMs sensibles a violaciones fuertes; calibración baja ECE; datos child-directed algo mejores que web | 0 | 0.75 | M |
| `acl_anthology_65` | LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models | **LENS dataset**: 258 participantes; 50.957 EMAs; >100.000 pares sensor-texto; Garmin vivoactive 3 + Android | GPS, pasos, acelerómetro ZCR/energía, conversación, unlock, HR, sueño, estrés | Alinear sensores con LLM para narrativas clínicas de salud mental | LENS supera baselines en métricas NLP y alineación clínica; expertos valoran narrativas; eficiencia token ~94% menor que texto | 0 | 0.77 | M |
| `acl_anthology_72` | Multimodal Affective Modeling in an LLM-based Intelligent Tutoring System for Foreign Language Learning | **EmoBot proof-of-concept** (sin dataset público nombrado) | Audio, vídeo, biosignales (HR, SpO₂, HRV desde smartwatch) | Reconocimiento multimodal de emociones en aprendizaje de alemán | Audio: 81.3% accuracy en habla alemana; Visual: ~66% en FER-2013 (nivel humano 65±5%) | 0 | 0.74 | M |
| `arxiv_163` | Large language models surpass domain-specific architectures for antepartum electronic fetal monitoring analysis | **Oxford Maternity (OxMat)**: 23.127 segmentos train; 2.500 eval; 4.000 limited | CTG: frecuencia cardíaca fetal (FHR) y actividad uterina (UA) | Benchmark de modelos para clasificación CTG APO/NPO | Llama-3B fine-tuned mejor AUC 0.852; data-limited 0.800; sin UA, 1D SE-ResNet 0.815; LLMs requieren mucho más cómputo | — | 0.75 | M |
| `arxiv_190` | Towards a Personal Health Large Language Model | **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment** | Wearables: sueño, actividad, HR, HRV, respiración, pasos | Fine-tuning de Gemini para salud personal (sueño/fitness) | PH-LLM 79% sueño y 88% fitness; rendimiento cercano a expertos; PROs con adapter comparable a modelos especializados | — | 0.75 | M |
| `arxiv_199` | Hindsight Bias in Clinical Temporal Reasoning: How Future Data Exposure Affects Large Language Model Judgment | 171 case reports de PMC OA: 40 sepsis y 131 GLP1/diabetes; TTS humano y generado por LLM | N/A (texto clínico/series temporales textuales) | Medir sesgo de retrospección en razonamiento clínico temporal | Exposición a timeline completo aumenta movimiento hacia trampas de retrospección; el enmascaramiento temporal reduce sesgo sin pérdida de accuracy | — | 0.75 | M |
| `arxiv_276` | Integrating LLM, EEG and eye-tracking biomarker analysis for word-level neural state classification | **ZuCo Task 3**: 12 sujetos; 407 oraciones; 356 usadas | EEG + eye-tracking | Clasificar HRW vs LRW con etiquetas LLM | >60% validation accuracy; HRW más fijaciones; primer intento de clasificación word-level con LLM | — | 0.74 | M |
| `arxiv_280` | Benchmarking LLM Summaries of Multimodal Clinical Time Series for Remote Monitoring | TIHM-1.5 (56 personas con demencia; 2,803 días-paciente; test de 100 días-paciente) | HR, presión arterial, temperatura, actividad de habitaciones, sueño | Evaluar fidelidad clínica de resúmenes LLM de series temporales multimodales | Métricas NLP altas no implican corrección clínica; pipeline visual Gemini-2.5-Pro logra 45.7% recall de anomalías y 100% recall de duración/cobertura | — | 0.76 | M |
| `arxiv_317` | Neurocognitive Modeling for Text Generation: Deep Learning Architecture for EEG Data | ImageNet of the Brain (Emotiv Insight; 5 canales EEG; 3 s a 128 Hz; estímulos ImageNet ILSVRC2013) | EEG | Generar texto a partir de EEG con encoder RNN + classifier-LLM (Gemma 2B) | Accuracy 66.67% en 2 clases, 51.85% en 3 clases; perplexity 24.81–73.54 según clases; eficiente en datos | — | 0.75 | M |
| `arxiv_367` | LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models | 46 categorías semánticas con 478–624 valores; 39/54 LLMs | N/A (texto/LLM) | Comparar interferencia retroactiva y proactiva en LLM | PI > RI en todos los modelos (d=1.73); tamaño predice RI pero no PI; errores distintos; alucinación <1% | — | 0.75 | M |
| `arxiv_70` | Large Language Models are Few-Shot Health Learners | **MIT-BIH Atrial Fibrillation Database; MIMIC-III; PAMAP2; dataset sintético metabólico; Fitbit longitudinal (stress EMA/PHQ-8)** | ECG/IBI, acelerómetro, HR/sueño/pasos de Fitbit | Few-shot prompt tuning para tareas de salud | Mejora sobre zero-shot y baselines supervisados; contexto en prompt mejora; zero-shot pobre; hasta 130% mejora en cardiovasculares | — | 0.75 | M |
| `arxiv_77` | Fine-tuning LLMs using EEG microstate features for mental workload assessment | **Zyma et al.** (36 sujetos) + **Shin et al.** (29 sujetos); paper reporta 103 sujetos | EEG microstates | Fine-tuning LLM para carga cognitiva Rest/Load | Accuracy 97% tras fine-tuning vs 4.5% antes; ~24× mejora | — | 0.74 | M |
| `ieee_116` | Multi task multi emotion joint representation and LLM based posterior calibration for mental health risk assessment | Dataset propio multitarea/multiemoción con 237 participantes | Onda pupilar | Evaluar depresión, ansiedad y estrés con representación conjunta y calibración posterior con LLM | Reduce MAE/RMSE frente a modelos de una tarea; calibración LLM mejora errores adicionalmente | 0 | 0.77 | M |
| `ieee_255` | Multimodal to semantics large language model framework for loneliness assessment | Dataset de soledad de trabajo previo [11]: Oura Ring, Samsung Gear Sport y smartphone AWARE; 16 participantes | HRV, HR, sueño, actividad, uso de smartphone | Evaluar soledad con representaciones semánticas relativas a baseline individual y LLM | No mejora el ajuste numérico global, pero reduce subestimación de soledad alta; falsos negativos 39 vs 48 del ML | 0 | 0.76 | M |
| `ieee_259` | A review of multimodal sensing enabled large language models for automated emotional regulation in mental health applications | IEMOCAP; DEAP; revisión de 120 publicaciones | EEG, HRV, GSR, facial, voz, IoT | Revisar y evaluar LLM multimodales para regulación emocional | Accuracy 92.7%; F1 0.91; +14.3% vs EEG unimodal; federated learning/differential privacy reduce riesgo de fuga | 0 | 0.75 | M |
| `ieee_314` | MM-LLM for depression: towards an augmented intelligent diagnosis and intervention by fusing multimodal data and LLMs | Dataset 1, Dataset 2 y Dataset 3 (no nombrados en el extracto) | EEG, habla y texto | Reconocimiento de depresión, soporte de evaluación y generación de respuestas de intervención | Proof-of-concept; mejora discriminabilidad EEG; piloto pre-post con 20 participantes muestra reducción de síntomas (p<0.01), preliminar | 0 | 0.75 | M |
| `ieee_378` | AI driven wearable sensor analytics for dementia monitoring: machine learning, deep learning and emerging foundation model opportunities | Revisión de 42 estudios; menciona SINEW (n=204); actigrafía research-grade 87.8%, wearables consumer <15% | IMU, PPG, ECG, actigrafía, sueño, señales oculares | Revisar biomarcadores digitales y paradigmas ML/DL/FM en demencia | ML es más maduro (TRL 6–7); DL limitado por datos (TRL 4–5); FM incipiente (TRL 2–3); propone arquitecturas híbridas edge-cloud | 0 | 0.76 | M |
| `ieee_401` | The impact of generative AI and large language models on telemedicine diagnostics: a systematic review | Revisión sistemática; no dataset primario | Texto, imagen, señal multimodal en telemedicina | Revisar evidencia de GenAI/LLM en diagnóstico telemedicine | Evidencia preliminar; alta accuracy en contextos estructurados y reducción de documentación; retos de alucinación, sesgo, interpretabilidad y regulación | 0 | 0.75 | M |
| `ieee_449` | Validated multi agent simulation for dementia care: LLM based agents with biometric feedback | 120 interacciones simuladas cuidador-paciente; sin dataset real | HRV/EDA simulados | Simulación multiagente con LLM y biometría para validar razonamiento clínico sin circularidad | Estimación de confusión r=0.915; compliance predicha r=−0.933; asociaciones biométricas mediadas por acción (parcial r≈−0.009) | 0 | 0.75 | M |
| `ieee_59` | Large language model driven feature weighting for cross device harmonization of wearable data | Cohortes ETRI, MDCRC, LifeSnaps, Fitbit Versa, Actigraphy, Empatica E4; cohorte externa independiente | HR, actividad, sueño, GPS, EMA de ánimo/estrés | Armonizar datos wearable entre cohortes con pesos generados por LLM | GPT-OSS mejora F1 externo de forma más consistente; SHAP muestra que los pesos se usan y correlacionan con variables fisiológicas/conductuales | 0 | 0.75 | M |
| `ieee_651` | A communication robot for elderly persons using LLM to generate responses according to emotions estimated from physiological signals | Dos participantes mayores (80 y 90 años) en residencia de cuidado | EEG (Muse S Gen2) y HRV (Polar H10); voz | Robot conversacional que adapta respuestas a emociones estimadas | Robot propuesto reduce estrés y estimula función cerebral frente al robot previo; comunicación más activa | 0 | 0.75 | M |
| `manual_1` | WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data | **WearableQA benchmark** (200 usuarios reales, 4,084 preguntas MCQ de 10 opciones, hasta 500 días de datos) | **Wearable time-series** (16 métricas diarias: RHR, HRV, VO₂max, pasos, calorías, sueño, estrés), **blood biomarkers** (17), demographics | Benchmark de razonamiento en salud sobre datos wearables reales | 14 LLMs: accuracy **19.6%–72.9%** (baseline azar 10%); Gemini-3.1-Pro el mejor (72.9%); data reasoning más difícil que health reasoning | — | 1.00 | M |
| `manual_4` | An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation | **PPG–ECG dataset** (46 participantes; 12 held-out, 621 grabaciones de 15 min; Samsung Gear Sport + Shimmer3) | **PPG + ECG** (frecuencia cardiaca) | Agente LLM (OpenCHA, GPT-3.5-turbo) orquestando pipeline de estimación de HR desde PPG | **MAE 2.83, RMSE 5.47, MAPE 0.04, MAD 1.90** — vs GPT-4o (MAE 8.93) y GPT-4o-mini (MAE 9.41); outliers: 0.08% vs 6.20%/17.55% | — | 1.00 | M |
| `pubmed_105` | Multimodal large language models in health care: applications, challenges and future | No dataset primario; revisión. Menciona MIMIC-IV, PMC-VQA, LLaVA-Med y caso CONCH | Imágenes médicas, series temporales, audio, texto, vídeo, ómicas | Revisar principios, aplicaciones, retos y futuro de M-LLM en salud | Marco conceptual; destaca fusión de datos, benchmarks, interpretabilidad, privacidad y regulación | — | 0.76 | M |
| `pubmed_12` | The role of multimodal generative AI in older adults’ health management: systematic scoping review | Revisión sistemática de 28 estudios; no dataset primario | Modalidades multimodales (texto, audio, imagen, sensores) | Revisar rol, impactos y tendencias de GenAI multimodal en salud de mayores | 82% de publicaciones 2024–2025; predominan chatbots; evidencia limitada a mayores cognitivamente normales; faltan estudios longitudinales y equidad | — | 0.76 | M |
| `pubmed_157` | Integrating NLP into radiation oncology: a practical guide to transformer architecture and LLMs | No dataset primario; revisión | No fisiológica; texto clínico | Revisar NLP/LLM en oncología radioterápica y proponer framework | Framework de evaluación: propósito, fit clínico, rendimiento, sesgo/confianza, legal/ético, QA | — | 0.74 | M |
| `pubmed_177` | The first step is the hardest: pitfalls of representing and tokenizing temporal data | **Sample mobile sensing dataset** (case study con tokenizador OpenAI) | Time-series de wearables (genérico) | Analizar retos de tokenización de datos temporales en LLMs | Case study: tokenizadores dividen timestamps y valores en tokens no significativos; se aboga por prompt tuning, model grafting y tokenizadores mejorados | — | — | M |
| `pubmed_41` | The effects of maturation and dyslexia risk on neural speech sound encoding and discrimination at preschool stage | **DyslexiaBaby project**: 146 a 28 meses; 143 a 4–5 años | EEG/ERP/MMR | Estudiar maduración y riesgo de dislexia | P1–N2; MMN+LDN; riesgo de dislexia con MMN reducida; sin diferencias en ERPs obligatorios | — | 0.74 | M |
| `scopus_176` | A review of the emergence of AI-driven cognitive digital phenotyping: multimodal sensing, foundation models and the path toward digital twins for human cognition | **Revisión** (sin dataset propio) | Behavioral, neurophysiológico (EEG, fNIRS, fMRI), smartphone, wearables | Revisión y hoja de ruta hacia gemelos digitales cognitivos (CDT) | Propone pipeline CDP→CDT con FMs/VLMs; identifica retos de estandarización, simulación en tiempo real y privacidad | 0 | 0.74 | M |
| `scopus_207` | Exploring the feasibility of generating physical prescriptions using large language models with motion tracking inputs | **30 testers** (15 pacientes clínicos + 15 voluntarios sanos) | **Motion tracking / gait analysis** (Vicon: 24 cámaras ópticas, IMU, marker-less) | Generación de prescripciones físicas personalizadas con LLM + datos cinemáticos | Expertos puntúan LLM mejor que humano en 6/10 casos; MAE tiempo de generación 46s vs 35min humano; el LLM **no interpreta bien** datos cinemáticos sin fine-tuning | 0 | 0.74 | M |
| `scopus_226` | Structural priming in humans and large language models | Estímulos dativos en inglés: **64 verbos, 256 oraciones, 128 pares prime-target, 396 fillers**; **EEG de 27 participantes**; **norming con 60 participantes**; suite de 13 LMs | EEG y eye-tracking | Comparar priming estructural humano vs LMs | Ambos muestran priming; LMs más fuerte en DO; predicción ítem-a-ítem limitada; ningún LM predice consistentemente el priming humano | 0 | 0.75 | M |
| `scopus_244` | Integrating large language models with formal planning to automate the design and validation of biosignal processing pipelines | Dataset novel de 78 descripciones de pipelines ECG/EEG de PubMed; corpus preliminar de escenarios | ECG y EEG (descripciones de procesamiento) | Traducir lenguaje natural a pipelines válidos de procesamiento biosignal con LLM + planificación HTN | Precisión 0.89–0.98 mapeando bloques; F1 0.97–0.98 generando pipelines válidos de 2–15 pasos | 0 | 0.76 | M |
| `scopus_263` | AWARE Narrator and the utilization of large language models to extract behavioral insights from smartphone sensing data | StudentSense: estudiantes universitarios durante 17 semanas; análisis de una semana de un participante; AWARE-Light | Sensado de smartphone (GPS, acelerómetro, uso de apps, teclado, etc.) | Convertir datos tabulares de smartphone en narrativas y extraer insights conductuales/psicológicos con LLM | Demuestra factibilidad; narrativas permiten resúmenes y predicción exploratoria de escalas DASS-21/I-PANAS-SF | 0 | 0.75 | M |
| `scopus_271` | EEG-informed adaptive UX questionnaire generation with multi-agent LLMs | **EEG DSI-24**: 6 participantes; 4 condiciones visuo-hápticas; 24 trials | EEG | Generar cuestionarios UX adaptativos con multi-agent LLMs | G-Eval 80.96; +45.32% sobre baseline single-stage | 0 | 0.74 | M |
| `scopus_341` | DiscBioSem-LM: a framework for anxiety detection via semantic fusion of multimodal physiological signals and LLMs | Dataset propio de escenarios de intervención psicológica en VR (180 participantes; EMG, EDA, HR, BVP, RESP, ACC; SAI) | EMG, EDA, HR, BVP, respiración, acelerometría | Detección de ansiedad mediante descripciones semánticas y fine-tuning discriminativo de LLM | Accuracy 0.8100; F1 0.8102; supera baselines multimodales | 0 | 0.77 | M |
| `scopus_39` | Leveraging LLMs for automated feature extraction and model training in EMG-based motion decoding | **EMG in-hand manipulation dataset**: 11 participantes; 16 músculos; 3 objetos/movimientos | EMG | Automatizar extracción de features y entrenamiento con LLM | Rendimiento comparable a código manual; sin diferencia significativa (p=0.16); pequeñas desviaciones por precisión/parámetros | 0 | 0.74 | M |
| `scopus_396` | Understanding uncertainty in large language model predictions of early death in critically ill patients: a conformal prediction approach | **MIMIC-III** (4,714 admisiones únicas; 910 muertes; cohorte balanceada final 1,354) | — (notas clínicas de texto de UCI) | Predicción de muerte intrahospitalaria con GPT-4o zero-shot + conformal prediction | Cobertura empírica **90.4%** (global), 99.7% muerte / 81.1% supervivencia; recall muerte 93%, supervivencia 13% | 0 | 0.74 | M |
| `scopus_56` | Toward multimodal AI for dementia: a challenge driven survey of LLMs and VLMs | DementiaBank; ADReSS; TIHM; dataset multimodal ADL [13] (no nombrado); ADRD-Bench | Voz/audio, vídeo, sensores, datos fisiológicos, texto | Revisar LLM/VLM para demencia en grounding, multimodalidad y personalización | Taxonomía de tres retos; RAG y grafos de conocimiento ayudan; fusión multimodal y personalización siguen abiertas | 0 | 0.76 | M |
| `scopus_77` | Explaining recovery trajectories of older adults post lower limb fracture using modality wise multiview clustering and LLMs | MAISON-LLF (560 días; 10 adultos mayores; aceleración, pasos, movimiento ambiental, GPS, HR, sueño; SIS, OHS, OKS, TUG, 30sCST) | Acelerometría, HR, sueño, GPS, movimiento ambiental | Interpretar clústeres multimodales con LLM en recuperación tras fractura | La mayoría de etiquetas de clúster fueron estadísticamente significativas frente a escalas clínicas; aceleración no significativa | 0 | 0.77 | M |
| `scopus_85` | Large language models integrated into brain computer interfaces for communication | No dataset primario; revisión sistemática de 11 estudios; escenarios P300, SSVEP, cVEP, copy-spelling y ALS message-bank | EEG/BCI (P300, SSVEP, cVEP) | Revisar integración LLM-BCI para comunicación asistiva | Taxonomía de 5 patrones de integración; reporta ahorros >50% de tecleo y 42 caracteres/min en un estudio ALS; advierte falta de pacientes con discapacidad motora y poca latencia reportada | — | — | M |
| `wos_200` | The shrinking landscape of linguistic diversity in the age of large language models | **Reddit r/WritingPrompts** (N=318,490 historias); **Patch News** (N=379,583 artículos); **arXiv preprints** (N=80,238 abstracts CS); **Essays dataset** (N=2,348); **YourMorals** (N=3,641); **Empathic Conversations** (N=711); **Congress** (N=710) | — (texto) | Medir homogenización lingüística por LLMs | Reducción significativa de varianza en complejidad de escritura | 10 | 0.75 | B |
| `wos_399` | Multimodal large language models in medical research and clinical practice: development, applications, challenges and future | **MIMIC-CXR** (377k imágenes CXR + 227k informes); **COVID-19-CT-CXR** (1,000 imágenes); **IU-Xray, COVID19-CT, NIH Chest X-ray, CheXpert, Shenzhen TB**; **MedQA (USMLE), NEJM CPC, GeneTuring, Slake-VQA, Path-VQA, ECG-VQA, MIMIC-III** | Imagen médica (rayos X) | Revisión de MLLMs médicos (M³FM, Med-Gemini, MUSK, PathChat, GPT-4V(Med), Med-MLLM) | Revisión; M³FM: BLEU-4 38.8% zero-shot chino; Med-Gemini: 91.1% USMLE; MUSK: C-index 0.747 pancáncer | 7 | 0.74 | B |
| `ieee_196` | Reducing diagnostic uncertainty using large language models | **MIMIC-III** | No fisiológica; notas clínicas/EHR | Predecir categorías de sistema corporal a partir de notas clínicas de los primeros 3 días | ~80% modelos F1>0.7; varios sistemas >0.8; bajos en malformaciones congénitas, ojo y oído; con 1 día vs 3 días F1 baja pero la mayoría mantiene rendimiento satisfactorio | 5 | 0.75 | B |
| `scopus_149` | Agent-based large language model system for extracting structured data from breast cancer synoptic reports: a dual validation study | **864 casos sintéticos** (validación controlada); **90 informes de patología reales** (6,651 campos anotados) | — (texto de patología) | Extracción estructurada de datos de informes de cáncer de mama (229 campos CAP) | Accuracy sintética **93.8–99.0%** | 5 | 0.74 | B |
| `ieee_271` | Developing a RAG agent for personalized fitness and dietary guidance | **10 PDFs** de dieta/ejercicio; datos de wearables “en tiempo real” previstos, sin dataset fisiológico nombrado | No fisiológica; voz/texto; wearable previsto | RAG para fitness/dieta personalizada | Relevancia ~80–90%; faithfulness 40% (F1 57%); latencia 2–3 s; voz 85%; memoria 95% | 4 | 0.75 | B |
| `wos_262` | Large language models in mental health: a systematic review of applications, innovations and ethical challenges | Revisión: **PHQ-9**, datasets de redes sociales, notas clínicas; 92 estudios finales | Texto/social media; no fisiológica única | Revisar aplicaciones LLM en salud mental | Aplicaciones en detección, agentes conversacionales, apoyo terapéutico; retos de datos, sesgo, privacidad, validación clínica | 4 | 0.75 | B |
| `scopus_128` | Predicting momentary suicidal ideation from smartphone screenshots using vision language models: prospective machine learning study | 2.5 millones de capturas de pantalla de 70 participantes durante 28 días; EMA | N/A (capturas de pantalla; no fisiológica) | Predecir ideación suicida momentánea desde screenshots con VLM | Temporal holdout AUC 0.83/AUPRC 0.77; subject holdout ≈0.50; screening léxico AUC 0.62; modelos pequeños viables on-device | 1 | 0.75 | B |
| `acl_anthology_169` | Vers une évaluation rigoureuse des systèmes RAG : le défi de la due diligence | **Dataset RAG due diligence**: 300 documentos DataCorp; 121 preguntas; 20 respuestas/pregunta; anotaciones humanas/LLM; público `gmartinonQM/eval-rag` | No fisiológica | Evaluar RAG industrial con PPI/human+LLM judge | Métricas automáticas y PPI; LLM-Judge sobreestima pertinencia; acuerdo bajo hace marginal la ganancia de PPI | — | 0.75 | B |
| `acl_anthology_243` | Beyond Toxic Positivity: Interpersonal Affect Regulation in LLM-Based Dialogue Agents using Discourse Politeness Theory | **Simulación longitudinal de 8 semanas** (18 observadores terceros) | — (diálogo texto; sin señal fisiológica) | Regulación afectiva interpersonal vía Discourse Politeness Theory | 72% seguridad psicológica vs 22% baseline; 72% preferencia por modelo propuesto | 0 | 0.74 | B |
| `acl_anthology_73` | Gladiator at MEDIQA-SYNUR 2026: Contextual Clinical Extraction Integrating Foundation Models with Domain-Specific Validation Rules | **MEDIQA-SYNUR shared task dataset** (122 train, 101 dev, held-out test) | — (texto clínico: dictados de enfermería) | Extracción híbrida de información clínica guiada por esquema (193 conceptos, 4 tipos de valor) | **74.27% precision, 84.32% recall, 78.98% F1** en dev set | 0 | 0.75 | B |
| `arxiv_286` | Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data | Dreaddit; DepSeverity; SDCNL; CSSRS-Suicide; Red-Sam; Twt-60Users; SAD | N/A (texto online) | Evaluar zero-shot, few-shot e instruction fine-tuning en predicción de salud mental | Mental-Alpaca y Mental-FLAN-T5 superan GPT-3.5/GPT-4 y se acercan a Mental-RoBERTa; fine-tuning multi-dataset mejora generalización | — | 0.76 | B |
| `arxiv_365` | Daily and Weekly Periodicity in Large Language Model Performance and Its Implications for Research | Humanity’s Last Exam (tarea de física; consultas repetidas a GPT-4o) | N/A | Testar invariancia temporal del rendimiento de GPT-4o bajo condiciones fijas | ~20% de la varianza atribuible a componentes periódicos; variación pico-pico ~14% de la escala; no hay deriva lineal significativa | — | 0.76 | B |
| `pubmed_100` | Automated generation of discharge summaries leveraging large language models with clinical data | 25 casos de cirugía pancreática del Heidelberg University Hospital; EHR estructurado y resúmenes médicos en alemán | N/A (texto clínico) | Generar resúmenes de alta en alemán desde datos estructurados con LLM open-source | ~2.84 errores/resumen; ROUGE-1 0.25; BERTScore 0.64; valoraciones 3.72–3.96/5; solo 60% evaluó completitud como buena | — | 0.76 | B |
| `pubmed_123` | DataAtlas: automatic generation of data dictionaries using large language models | **ORCHID** (64 columnas, PhysioNet); **CarpeDiem** (101 columnas, PhysioNet); **INSPIRE** (62 columnas, PhysioNet) | — (datos tabulares clínicos) | Generación automática de diccionarios de datos | GPT-4 preferido sobre documentación oficial en 46.9% de comparaciones | 0 | 0.74 | B |
| `pubmed_182` | Using large language models for chronic disease management tasks: scoping review | Revisión de 29 estudios; no dataset primario | N/A (revisión) | Revisar aplicaciones de LLM en manejo de enfermedades crónicas | Tareas centradas en paciente 62%; retos: respuestas inexactas 62%, ética/privacidad 35%, usabilidad 31% | — | 0.75 | B |
| `pubmed_194` | Large language models to extract cancer staging data from clinical documentation | 700 notas anotadas de un sistema de salud integrado; 2.5M registros de estadificación de 217,768 pacientes desde >2M notas; cáncer vesical, cervical, colorrectal, mama y próstata | N/A (documentación clínica) | Extraer estadificación oncológica desde notas clínicas con LLM fine-tuned | Precisión a nivel relación 0.77–1.0 en tres cánceres y 0.83–1.0 sin entrenamiento adicional en dos cánceres | — | — | B |
| `pubmed_69` | Unveiling the potential of large language models in transforming chronic disease management: mixed methods systematic review | Revisión sistemática de 20 estudios; no dataset primario | N/A (revisión) | Sintetizar evidencia sobre LLM en manejo de enfermedades crónicas | Tasa de acierto agrupada 71%; RAG mejora precisión (OR 2.89); retos de privacidad, idioma, cultura y tareas avanzadas | — | 0.75 | B |
| `pubmed_78` | Automating clinical phenotyping using natural language processing | MSDW/MSCCR: 49,572 notas clínicas y 2,204 informes de radiología de 584 pacientes con Crohn; test de 280 textos; 134 pacientes para nivel paciente | N/A (texto clínico y radiológico) | Comparar reglas spaCy vs GPT-4 para fenotipado de Crohn | GPT-4 similar o mejor; F1 nota ≥0.90 conducta y ≥0.82 edad; nivel paciente ≥0.66 y ≥0.71; sin diferencia significativa vs expertos | — | 0.76 | B |
| `scopus_50` | Effectiveness, safety, and workflow burden of LLM-based medical report generation: systematic review | Revisión: **101 estudios**; menciona **MIMIC-CXR** y corpus públicos | Imágenes médicas (X-ray, CT, MRI, US, endoscopia, patología, ECG); no señal fisiológica única | Revisar efectividad, seguridad y carga de trabajo | Sin estudios bajo riesgo bajo; 15 moderado, 72 alto, 14 serio; evidencia heterogénea/no poolable; certeza muy baja | 0 | 0.74 | B |
| `wos_230` | Leveraging large language models for automated psychological analysis in electronic sandboxes | **Electronic sandbox data** (60 participantes planificados, 18–35 años) | — (imágenes de sandbox, logs de interacción, movimiento de ratón) | Análisis psicológico automatizado integrando LLMs + visión por computador | Sistema propuesto; validación experimental futura (sin resultados aún) | 0 | 0.75 | B |
| `wos_327` | Integrating NLP into radiation oncology: a practical guide to transformer architecture and LLMs | No dataset primario; duplicado de `pubmed_157` | No fisiológica; texto clínico | Revisar NLP/LLM en oncología radioterápica | Framework de evaluación clínica; mismos hallazgos que `pubmed_157` | 0 | 0.75 | B |
| `wos_403` | Evaluation of LLMs for structured data extraction from interstitial lung disease clinical notes | **Stanford ILD clinic notes** 2018–2022: 10 prompt engineering + 100 validación; 12 LLMs | No fisiológica; notas clínicas | Extraer datos binarios y multiclase de notas ILD | 7/12 modelos a nivel humano (96.2%); gpt-oss-120b 97.5%; multiclase 91.1%/88.0% | 0 | 0.75 | B |
| `wos_85` | Enabling dietary assessment with smart wearables: a passive, scalable approach for precision nutrition | **Estudio de viabilidad**: 30 participantes UK; Hammersmith y Reading; 8 días; cámaras wearable; menú de referencia dietista | Imágenes de cámara wearable (no fisiológica) | Evaluar pipeline LVLM para evaluación dietética pasiva | Recall de alimentos 82%; MAE porción 44.7 g alimentos y 70.4 mL bebidas | 0 | 0.75 | B |

---

## 3. Cobertura por tipo de señal fisiológica

Cada fila agrega **todos** los papers del compendio (no un subconjunto), los
datasets implicados y el número de papers. Un paper puede aparecer en varias
filas si usa varias modalidades.

| Señal | n | Papers | Datasets implicados |
|---|---:|---|---|
| **ECG** | 17 | `arxiv_288`, `arxiv_7`, `arxiv_70`, `ieee_378`, `ieee_88`, `manual_2`, `manual_3`, `manual_4`, `pubmed_150`, `pubmed_187`, `scopus_111`, `scopus_197`, `scopus_223`, `scopus_244`, `scopus_293`, `scopus_50`, `scopus_69` | MIMIC-III Matched Waveform Database v1.0: 50,000 registros PPG–ECG de 1,275 pacientes; test bloqueado de 187 pacientes; Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc.; **MIT-BIH Atrial Fibrillation Database; MIMIC-III; PAMAP2; dataset sintético metabólico; Fitbit longitudinal (stress EMA/PHQ-8)**; Revisión de 42 estudios; menciona SINEW (n=204); actigrafía research-grade 87.8%, wearables consum… |
| **PPG / BVP** | 10 | `arxiv_288`, `arxiv_7`, `ieee_378`, `manual_2`, `manual_3`, `manual_4`, `pubmed_150`, `scopus_272`, `scopus_293`, `scopus_341` | MIMIC-III Matched Waveform Database v1.0: 50,000 registros PPG–ECG de 1,275 pacientes; test bloqueado de 187 pacientes; Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc.; Revisión de 42 estudios; menciona SINEW (n=204); actigrafía research-grade 87.8%, wearables consumer <15%; **MIMIC-III Matched Waveform Database v1.0** (50,000 registros PPG–ECG sincronizados de 1,275 pacientes; 36,115 train / 6,… |
| **EEG / PSG** | 28 | `acl_anthology_2`, `arxiv_276`, `arxiv_317`, `arxiv_7`, `arxiv_77`, `ieee_205`, `ieee_259`, `ieee_314`, `ieee_392`, `ieee_651`, `ieee_676`, `manual_3`, `pubmed_128`, `pubmed_168`, `pubmed_178`, `pubmed_25`, `pubmed_41`, `scopus_176`, `scopus_180`, `scopus_197`, `scopus_226`, `scopus_244`, `scopus_271`, `scopus_293`, `scopus_484`, `scopus_68`, `scopus_85`, `wos_386` | **Singh et al. (2016) stimuli** (82 items de presuposición); **Nieuwland et al. (2010) data** (168 items de implicatura escalar); **ImpPress dataset** (>25k pares; 14,100 Presp + 6,000 SI); **ZuCo Task 3**: 12 sujetos; 407 oraciones; 356 usadas; ImageNet of the Brain (Emotiv Insight; 5 canales EEG; 3 s a 128 Hz; estímulos ImageNet ILSVRC2013); Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc.; **Z… |
| **HRV** | 11 | `acl_anthology_72`, `arxiv_190`, `arxiv_51`, `ieee_255`, `ieee_259`, `ieee_449`, `ieee_651`, `manual_1`, `pubmed_150`, `scopus_272`, `wos_308` | **EmoBot proof-of-concept** (sin dataset público nombrado); **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment**; TILES-2018; WESAD; Dataset de soledad de trabajo previo [11]: Oura Ring, Samsung Gear Sport y smartphone AWARE; 16 participantes; IEMOCAP; DEAP; revisión de 120 publicaciones; … |
| **EDA / temperatura** | 7 | `arxiv_280`, `ieee_259`, `ieee_449`, `scopus_111`, `scopus_197`, `scopus_272`, `scopus_341` | TIHM-1.5 (56 personas con demencia; 2,803 días-paciente; test de 100 días-paciente); IEMOCAP; DEAP; revisión de 120 publicaciones; 120 interacciones simuladas cuidador-paciente; sin dataset real; PMData; WESAD; Dataset propio de 200 participantes; EEG, ECG, GSR; PHQ-9; estímulos IADS-2; **Can et al. (2020) multimodal physiological database** (14 estudiantes, TSST + 12h/día durante 1 semana, ~1000h, 388 EMA); Datas… |
| **Respiración / voz** | 18 | `acl_anthology_245`, `acl_anthology_72`, `arxiv_190`, `ieee_259`, `ieee_271`, `ieee_314`, `ieee_430`, `ieee_598`, `ieee_651`, `pubmed_105`, `pubmed_12`, `pubmed_161`, `pubmed_187`, `pubmed_39`, `scopus_341`, `scopus_56`, `scopus_97`, `wos_308` | **Roda Viva corpus**: 344 turnos; **EmoBot proof-of-concept** (sin dataset público nombrado); **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment**; IEMOCAP; DEAP; revisión de 120 publicaciones; **10 PDFs** de dieta/ejercicio; datos de wearables “en tiempo real” previstos, sin dataset fisio… |
| **EMG** | 4 | `acl_anthology_34`, `scopus_341`, `scopus_39`, `scopus_500` | **Gaddy & Klein unvoiced EMG**: 1 hablante, 8 canales, 67 palabras, ~26 min, 500 utterances; **CSL-EMG_Array** para person ID; Dataset propio de escenarios de intervención psicológica en VR (180 participantes; EMG, EDA, HR, BVP, RESP, ACC; SAI); **EMG in-hand manipulation dataset**: 11 participantes; 16 músculos; 3 objetos/movimientos; **EMG pretraining dataset** (33 sujetos); **4 nuevos participantes**; **Flickr3… |
| **IMU / ACC / movimiento** | 14 | `acl_anthology_65`, `arxiv_70`, `ieee_259`, `ieee_378`, `ieee_828`, `manual_5`, `scopus_118`, `scopus_207`, `scopus_263`, `scopus_272`, `scopus_341`, `scopus_401`, `scopus_77`, `wos_230` | **LENS dataset**: 258 participantes; 50.957 EMAs; >100.000 pares sensor-texto; Garmin vivoactive 3 + Android; **MIT-BIH Atrial Fibrillation Database; MIMIC-III; PAMAP2; dataset sintético metabólico; Fitbit longitudinal (stress EMA/PHQ-8)**; IEMOCAP; DEAP; revisión de 120 publicaciones; Revisión de 42 estudios; menciona SINEW (n=204); actigrafía research-grade 87.8%, wearables consumer <15%; Cohorte prospectiva de … |
| **Eye-tracking / pupila** | 7 | `arxiv_276`, `ieee_116`, `pubmed_168`, `scopus_180`, `scopus_226`, `scopus_484`, `scopus_68` | **ZuCo Task 3**: 12 sujetos; 407 oraciones; 356 usadas; Dataset propio multitarea/multiemoción con 237 participantes; **ZuCo 1.0 Task 3**; **ZuCo benchmark** (ZuCo 1.0 + 2.0); Estímulos dativos en inglés: **64 verbos, 256 oraciones, 128 pares prime-target, 396 fillers**; **EEG de 27 participantes**; **norming con 60 participantes**; suite de 13 LMs; ZuCo 1.0 (12 sujetos; 21,629 palabras; 1,107 frases; 154,173 fija… |
| **Imagen médica** | 8 | `ieee_401`, `ieee_676`, `pubmed_12`, `scopus_149`, `scopus_223`, `scopus_50`, `scopus_97`, `wos_399` | Revisión sistemática; no dataset primario; Cinco datasets públicos independientes (MRI, EEG, biomarcadores, clínicos, expresión génica); tres cohortes de validación independientes (n=28–45); Revisión sistemática de 28 estudios; no dataset primario; **864 casos sintéticos** (validación controlada); **90 informes de patología reales** (6,651 campos anotados); No dataset primario; revisión narrativa; Revisión: **101 … |
| **Wearable / multimodal** | 22 | `acl_anthology_72`, `arxiv_190`, `arxiv_7`, `arxiv_70`, `ieee_271`, `ieee_676`, `ieee_88`, `ieee_886`, `manual_1`, `manual_5`, `pubmed_12`, `pubmed_177`, `scopus_111`, `scopus_118`, `scopus_176`, `scopus_272`, `scopus_293`, `scopus_401`, `scopus_56`, `wos_308`, `wos_386`, `wos_85` | **EmoBot proof-of-concept** (sin dataset público nombrado); **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment**; Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc.; **MIT-BIH Atrial Fibrillation Database; MIMIC-III; PAMAP2; dataset sintético metabólico; Fitbit longitudinal… |
| **Sueño (actigrafía)** | 13 | `acl_anthology_65`, `arxiv_190`, `arxiv_280`, `arxiv_70`, `ieee_255`, `ieee_378`, `ieee_59`, `manual_1`, `pubmed_178`, `scopus_111`, `scopus_77`, `wos_308`, `wos_386` | **LENS dataset**: 258 participantes; 50.957 EMAs; >100.000 pares sensor-texto; Garmin vivoactive 3 + Android; **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment**; TIHM-1.5 (56 personas con demencia; 2,803 días-paciente; test de 100 días-paciente); **MIT-BIH Atrial Fibrillation Database; M… |
| **Sin señal fisiológica (texto, EHR, logs)** | 37 | `acl_anthology_118`, `acl_anthology_148`, `acl_anthology_169`, `acl_anthology_231`, `acl_anthology_243`, `acl_anthology_286`, `acl_anthology_288`, `acl_anthology_303`, `acl_anthology_56`, `acl_anthology_57`, `acl_anthology_73`, `acl_anthology_8`, `arxiv_163`, `arxiv_199`, `arxiv_286`, `arxiv_365`, `arxiv_367`, `ieee_196`, `pubmed_100`, `pubmed_106`, `pubmed_123`, `pubmed_157`, `pubmed_182`, `pubmed_194`, `pubmed_28`, `pubmed_69`, `pubmed_78`, `scopus_128`, `scopus_198`, `scopus_355`, `scopus_396`, `scopus_460`, `scopus_80`, `wos_200`, `wos_262`, `wos_327`, `wos_403` | **TriviaQA**; **NQ-Open**; **PopQA**; **MMLU** (high school/college; physics, chemistry, biology) y **GPQA** (expert); **Dataset RAG due diligence**: 300 documentos DataCorp; 121 preguntas; 20 respuestas/pregunta; anotaciones humanas/LLM; público `gmartinonQM/eval-rag`; **ClinDet-Bench**: 16 sistemas de scoring; 188 casos; 8 LLMs; **Simulación longitudinal de 8 semanas** (18 observadores terceros); **BoolQ; PubMed… |

### Lectura de la cobertura

- **ECG, PPG/BVP y EEG** concentran la literatura; **EDA** aparece sobre todo en
  datasets multimodales (WESAD, EmoSavior, el dataset de Can et al. y los
  conjuntos propios de detección de ansiedad).
- Las **revisiones** se cuentan por las modalidades que cubren, no por datos
  propios; por eso aparecen en varias filas.
- Solo **seis papers** combinan señal fisiológica con evaluación de la evidencia
  (grupo A2), y **cuatro** del compendio usan WESAD como dato primario (§1).
  Ninguno cruza ambos ejes: el hueco de esta tesis sigue abierto.
- La modalidad **wearable multimodal** es la más frecuente en los datasets
  recientes y la que más se solapa con el diseño de esta tesis (ventanas
  cortas, múltiples canales, etiquetas de estado).

---

## 4. Apéndice: papers del corpus sin ficha en este documento

`papers_metadata.json` contiene 126 papers; este compendio cubre 117. Los 12
restantes están en la metadata pero no tienen ficha de datasets en los
documentos de partida (ninguno menciona WESAD):

| Paper | Título | Año |
|---|---|---:|
| `arxiv_162` | From Classification to Clinical Insights: Towards Analyzing and Reasoning About Mobile and Behavioral Health Data With Large Language Models | 2023 |
| `arxiv_169` | EEG-Based Brain-LLM Interface for Human Preference Aligned Generation | 2026 |
| `arxiv_25` | Mind the Missing: Variable-Aware Representation Learning for Irregular EHR Time Series using Large Language Models | 2025 |
| `arxiv_64` | MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis | 2024 |
| `ieee_387` | clinical summarization of bradycardia and desaturation episodes in preterm neonates using llm models | 2025 |
| `ieee_483` | lstm in recursive feedback loops a study on textual evolution and complexity | 2025 |
| `ieee_62` | PhysioLLM: Supporting Personalized Health Insights with Wearables and Large Language Models | 2024 |
| `ieee_801` | multimodal sleep stage classification with llm based sleep report | 2026 |
| `pubmed_117` | incorporating advanced language models into the p300 speller using particle filtering | 2015 |
| `pubmed_122` | natural language processing with dynamic classification improves p300 speller accuracy and bit rate | 2012 |
| `pubmed_147` | robust assessment of the cortical encoding of word level expectations using the temporal response function | 2025 |
| `wos_300` | scientific production in the era of large language models outcome triggered treatment timing and spurious event study dynamics | 2026 |

Tres fichas no tienen entrada en la metadata (sin citas ni `relevance_score`):
`pubmed_177`, `pubmed_194`, `scopus_85`.

---

## 5. Competidores: qué dataset usa cada uno y a qué hipótesis compiten

Hipótesis: 
- **H1** corrección bajo degradación
- **H2a** confianza numérica vs acierto
- **H2b** certeza lingüística (hedges/boosters)
- **H3** representación
- **H4** adecuación de la abstención
- **H5** respaldo de las afirmaciones
- **H6** indicador externo de calidad (secundario).

| Paper | Dataset(s) | Compite en | Nivel | Qué hace | Coincidencias con esta tesis |
|---|---|:--:|:--:|---|---|
| `manual_5` Telemetry degradation | Telemetría sintética de smartwatch (IMU, 40 features; caso AI-tennis) | H1, H3 (H5 media) | **Muy alta** | Degrada telemetría (dropout, downsampling, señalización de fiabilidad) y mide estabilidad de la decisión (FlipRate) | Degradación controlada con parámetros conocidos; LLM como intérprete de datos numéricos; evaluación del cambio de conducta |
| `manual_2` = `arxiv_288` DFOT | MIMIC-III Matched Waveform (50.000 registros PPG–ECG; 1.275 pacientes) | H1, H2a, H5 (+H6) | **Muy alta** | Derived-Feature Over-Trust: el LLM acepta medidas derivadas poco fiables; repara con evidencia de fiabilidad | Señal fisiológica; fiabilidad de la evidencia; sobreconfianza; controles emparejados y evidencia barajada |
| `manual_1` WearableQA | WearableQA: 200 usuarios, 4.084 preguntas, hasta 500 días | H1, H5 | Alta | Benchmark de razonamiento sobre datos reales de wearables | Ground truth verificable; interpretación de datos fisiológicos; evaluación de corrección |
| `manual_3` PhysioBench | 22 datasets públicos (ECG, PPG, EEG, PSG); 61,4 M preguntas | H1, H3 | Alta | QA unificado sobre señales fisiológicas | Mismas modalidades; preguntas en lenguaje natural; comparación entre tareas y datasets |
| `acl_anthology_231` ClinDet-Bench | 16 sistemas de scoring; 188 casos; 8 LLM | H4 | Alta | Determinabilidad: ¿la información disponible permite decidir? | Información incompleta; abstención excesiva y conclusiones prematuras; límites de la evidencia |
| `acl_anthology_118` Verbal uncertainty | TriviaQA, NQ-Open, PopQA | H2a, H2b, H5 | Alta | Incertidumbre verbal como rasgo lineal para reducir alucinación confiada | Relación entre certeza expresada y error; hedges/boosters |
| `acl_anthology_148` AFCE | MMLU, GPQA | H2a | Alta | Sobreconfianza y estimación de confianza sin respuesta | Confianza desacoplada del acierto y de la dificultad |
| `acl_anthology_245` Disfluencias | Roda Viva (344 turnos) | H2b | Alta | Confianza del LLM frente a incertidumbre pragmática | Sesgo de longitud y baja sensibilidad a marcadores epistémicos |
| `scopus_355` Mind the data | LQMR (5 tablas) + MIMIC-IV-ED | H1, H6 | Alta | Sensibilidad del LLM a la **calidad** de los datos médicos | Las anomalías de plausibilidad reducen el AUC hasta un 16,8 % y el LLM las detecta mal |
| `arxiv_280` Clinical TS summaries | TIHM-1.5 (56 personas; 2.803 días-paciente) | H5, H1 | Alta | Fidelidad clínica de resúmenes de series temporales | Métricas NLP altas ≠ corrección; evaluación por eventos |
| `acl_anthology_57` Multilingual abstention | M-MMLU, M-Hellaswag, Belebele | H4 | Media | Enseñar a abstenerse con feedback multilingüe | Abstención como comportamiento entrenable, no como umbral fijo |
| `acl_anthology_56` DeCC | SNLI-VE, VCR, A-OKVQA, Winoground, MMMU, MathVista | H5 | Media | Consistencia por descomposición de la tarea | Fiabilidad de la respuesta más allá de la accuracy puntual |
| `manual_4` PPG→HR agent | PPG–ECG de 46 participantes (621 grabaciones) | H2a, H5 | Media | Agente LLM que orquesta la estimación de FC desde PPG | Señal fisiológica + LLM + evaluación de error cuantitativa |
| `scopus_244` Biosignal pipelines | 78 descripciones de pipelines ECG/EEG | H3 | Media | LLM + planificación formal para diseñar pipelines de biosignal | Representación del procesamiento y validación formal |

### Lectura de la competencia

- Los tres que marcan el listón son **`manual_5`** (degradación controlada y
  estabilidad), **`manual_2`/`arxiv_288`** (evidencia fisiológica y
  sobreconfianza) y **`manual_1`** (evaluación sobre datos de wearables).
- **Ninguno combina las cinco dimensiones** (corrección, confianza declarada,
  certeza lingüística, respaldo de claims y abstención) dentro del mismo
  experimento, ni distingue **degradación física** de **pérdida real de
  información** y de **resolubilidad**.
- **Ningún competidor usa WESAD bajo degradación**: de los 117 papers del
  compendio, cuatro lo usan como dato (y ninguno manipula la señal). El hueco
  experimental es real; lo que no es novedad por sí solo es «que los LLM son
  sobreconfiados» o «que la degradación reduce la accuracy» (ya documentado en
  `manual_5` y `scopus_355`).
- La contribución es **medir conjuntamente** cómo se adapta el comportamiento lingüístico a cambios conocidos de evidencia, separando degradación de resolubilidad y tratando la representación como factor.

---

## 6. Datasets externos ya probados: PhysioNet 2017 y K-EmoCon

Ambos se ejecutaron con el mismo pipeline (`GENERATE`/`EVALUATE`) y sirven para
una cosa distinta que WESAD: **no validan el endpoint afectivo**, ponen a prueba
la generalidad de los mecanismos.

| Dataset | Tarea | n evaluado | Papel en la tesis | Resultado observado | Utilidad |
|---|---|---:|---|---|---|
| **PhysioNet/CinC 2017** | Ritmo en ECG de una derivación (N/A/O/`noisy`) | 403 registros (de 8.528; 400 llamadas) | Control de generalidad con **evidencia no interpretable etiquetada por experto** (`noisy`) | LLM en el **azar exacto** (0,243 acc / 0,250 bal) frente a 0,548/0,552 de la referencia; **0/103 abstenciones** en `noisy`; 47,6 % de claims contradichos; certeza **mayor** en `noisy` (−0,64 vs −1,36; p=0,0008, Cliff δ=+0,60); calidad detectable (AUROC multivariante **0,873**) | **Alta** (H4/H5/H6, otra tarea y otro sensor) |
| **K-EmoCon** | Afecto (arousal/valencia) en habla espontánea con muñeca | 44 ventanas (piloto LLM); 5.528 ventanas de 5 s en la tabla de referencia | Intento de validación externa del componente afectivo | Referencia features-only: mejor AUROC **0,609** (`external_arousal`), 0,527 auto-arousal, 0,417 auto-valencia (bajo azar); LLM 0,614 balanced / 0,642 AUROC; el contraste `pre`/`debate` (0,708 bal / 0,785 AUROC) mide **tarea/actividad**, no afecto | **Baja** (limitación documentada) |

**Qué aporta cada uno.**

- **PhysioNet** es el mejor banco de pruebas para H4/H5 que hemos tenido: contiene
  la clase que un experto declara ilegible y el modelo **no se abstiene ni una
  vez** y además **sube** su certeza. Refuerza la lectura de que la certeza
  lingüística sigue la *forma* del dato, no su calidad. También muestra que la
  calidad **sí** deja rastro cuando las features son buenas (AUROC 0,873), lo que
  sostiene H6.
- **K-EmoCon** se cerró como **resultado negativo documentado**: no hay etiqueta
  de estrés, las etiquetas afectivas son dispersas y la referencia con EDA/HR de
  muñeca no pasa de AUROC 0,61. Su valor es metodológico: (i) delimita qué no se
  puede afirmar con datos de muñeca, (ii) muestra el confusor de orden del
  protocolo y (iii) justifica no repetir la ingesta.

**Qué falta para una validación externa del endpoint.** Un dataset con
**inducción de estrés** y canales de pecho o EDA fiables. Candidatos no
incorporados aún (pendientes de verificar acceso y canales): **UBFC-Phys** y
**SWELL-KW**. Sin uno de ellos, la validez externa del endpoint de estrés queda
como limitación declarada.

---

## 7. Por qué WESAD: ventajas, alternativas y debilidades

### 7.1 Comparación con los datasets alternativos analizados (sección 7.4)

La comparación relevante para justificar WESAD se hace frente a los demás
datasets fisiológicos que se han revisado en este documento (sección 7.4), no
frente a los controles externos de sección 7.5, que responden a una pregunta
distinta.

| Criterio | **WESAD** | DREAMT | TILES-2018 | PMData | PPG-DaLiA | HHAR | UBFC-Phys / SWELL-KW |
|---|---|---|---|---|---|---|---|
| Fenómeno | Estrés/afecto inducido (TSST, amusement, meditación) | Sueño y transiciones de estadio | Estrés ecológico/longitudinal | Estrés vitalicio (Fitbit) | Actividad física | Actividad (teléfono/reloj) | Estrés inducido (candidatos) |
| Etiquetas | Condición de protocolo (4 clases) | Estadio PSG (referencia técnica) | Autoinformes (encuesta) | Autoinforme (Fitbit) | Etiqueta de actividad | Etiqueta de actividad | No verificados aún |
| Referencia externa | Limitada | **PSG sincronizada** | No | No | No | No | No |
| Modalidades | Pecho (ECG, EDA, resp, temp) + muñeca | Wearable + PSG | Muñeca (EDA/HR) | Fitbit (HR, sueño, pasos) | Muñeca + pecho (PPG, EDA, temp, ACC) | Teléfono/reloj (IMU) | EDA/ECG (pendiente verificar) |
| Participantes | 15 | 100 | Trabajadores sanitarios | 16 sujetos, 5 meses | 15 sujetos | Múltiples | Pendiente |
| Ventanas usadas | 60 s | 30 s (PSG) | Ventanas continuas | Continuo | Continuo | Continuo | Pendiente |
| Acceso | Público (licencia investigación) | **Restringido** (PhysioNet) | Pendiente de verificación | Público | Público | Público | Pendiente |
| Confusor principal | Orden del protocolo (secuencial) | Nocturno/ordenado por sueño | Etiquetas escasas; deriva temporal | Etiquetas escasas | Tarea de actividad, no afecto | Sin fisiología afectiva | Pendiente |
| Papel en la tesis | **Vehículo principal** (H1–H6) | Complemento para fidelidad situacional | Segunda validación del endpoint | Complemento para personalización | Control de representación PPG/calidad | Eje representación/ruido | Validación externa (pendiente) |

### 7.2 Ventajas de elegir WESAD

1. **Es el único de los candidatos con las dos cosas a la vez**: inducción
   experimental de estados afectivos **y** canales de pecho con EDA, ECG,
   respiración y temperatura. La degradación controlada necesita canales
   separables; el análisis de evidencia necesita una referencia de estado.
2. **Etiquetas por condición de protocolo**, alineadas con ventanas de 60 s, sin
   depender de anotación subjetiva continua.
3. **Cuatro clases con dificultad desigual**: `baseline` y `stress` separables,
   `amusement` fisiológicamente próxima a `baseline` y `meditation` distinguible
   por respiración. Esa gradación es útil: permite separar «no acierta porque no
   hay información» de «no acierta porque las clases son confundibles».
4. **Es un estándar de facto**: aparece como dato en 4 de los 117 papers del
   compendio (y en uno más como mención), lo que hace sus resultados comparables
   y auditables por terceros.
5. **Tamaño manejable y reproducible**: 15 sujetos, 12,9 GB, corpus de 705
   ventanas que se reconstruye en minutos con fingerprints; permitió ejecutar la
   matriz completa (2 modelos × 2 tareas × 5 condiciones) en horas, no semanas.
6. **Ya está resuelto el problema de la referencia**: la referencia features-only
   con las mismas 23 features da 0,64–0,68 en 4 clases y 0,92 en binario, de modo
   que el error en la respuesta del LLM se debe a su interpretación de la
   evidencia y no a la falta de señal en los datos de entrada.

### 7.3 Debilidades que hay que declarar

1. **n = 15 y una sola sesión por sujeto**: poca potencia para generalizar entre
   personas; el bootstrap agrupa por sujeto, pero no sustituye a más sujetos.
2. **`stress` es una condición inducida, no una medida de la experiencia
   subjetiva**; y `amusement` también produce activación fisiológica. No se puede
   convertir una etiqueta de protocolo en una afirmación sobre lo que siente la
   persona.
3. **Protocolo secuencial**: la fase predice la clase (modelo solo-fase:
   balanced 0,857 en 4 clases). Se controló residualizando en fase²
   ([`resolubilidad.md`](resolubilidad.md)), pero el confusor existe y hay que
   citarlo.
4. **Sin anotaciones de decisiones o actuaciones**: permite evaluar la
   *justificación* de una implicación, no su daño real (limitación que también
   señalaba el análisis de fidelidad situacional).
5. **Confusor de movimiento**: ACC y EMG discriminan estrés vs baseline
   (AUC hasta 0,93). Se excluyó la muñeca de la vista del LLM, lo que a su vez
   reduce la señal disponible.

### 7.4 Alternativas plausibles y para qué servirían

| Dataset | Punto fuerte | Punto débil | Papel recomendado |
|---|---|---|---|
| **DREAMT** (100 sujetos; wearable + PSG) | Referencia externa **independiente** (PSG) y 100 sujetos: el mejor banco para *fidelidad situacional* (¿la descripción del sueño resiste la degradación?) | Acceso restringido; mide sueño, no estrés; ventanas de 30 s; otra tarea | **Complemento principal** de una extensión; no sustituye a WESAD para H4 |
| **TILES-2018** (trabajadores sanitarios; wearable continuo) | Estrés «ecológico» y longitudinal, con EDA/HR de muñeca | Sin canal de pecho; etiquetas por encuesta; menos control experimental | Segunda validación del endpoint si se consigue acceso |
| **PMData** (16 sujetos, 5 meses) | Longitud y multimodalidad (Fitbit + encuestas) | Etiquetas escasas; sin EDA de pecho | Complemento para personalización y deriva temporal |
| **PPG-DaLiA** (15 sujetos; muñeca + pecho) | PPG/EDA/temp/ACC con referencia de pecho y actividad | Tarea de actividad, no afecto | Control para representación de PPG y calidad |
| **HHAR** (actividad, teléfono/reloj) | Muy usado; útil para degradación de IMU | Sin fisiología afectiva | Solo para el eje de representación/ruido |
| **UBFC-Phys**, **SWELL-KW** | Inducción de estrés con EDA/ECG (candidatos) | No incorporados; hay que verificar canales, licencia y calidad | **Validación externa del endpoint** (pendiente) |
| **WearableQA**, **PhysioBench** | Benchmarks listos para comparar | No son datos crudos; tareas ya definidas | Comparabilidad, no experimento propio |

### 7.5 Datasets externos ya probados: PhysioNet 2017 y K-EmoCon (controles externos, no alternativas a WESAD)

PhysioNet/CinC 2017 y K-EmoCon **no son alternativas a WESAD**: responden a una
pregunta distinta y fueron analizados inicialmente dentro del estudio PSRI, de
donde proviene su posible relevancia para esta tesis. No se fuerzan como
validación externa del endpoint de estrés porque su fenómeno es distinto; su
valor está en poner a prueba la generalidad de los mecanismos (H4, H5, H6) con
otra tarea y otro sensor.

| Dataset | Tarea | n evaluado | Resultado observado | Papel en la tesis | Contribución | Limitaciones |
|---|---|---:|---|---|---|---|
| **PhysioNet/CinC 2017** | Ritmo en ECG de una derivación (N/A/O/`noisy`) | 403 registros (de 8.528; 400 llamadas) | LLM en el **azar exacto** (0,243 acc / 0,250 bal) frente a 0,548/0,552 de la referencia; **0/103 abstenciones** en `noisy`; 47,6 % de claims contradichos; certeza **mayor** en `noisy` (−0,64 vs −1,36; p=0,0008, Cliff δ=+0,60); calidad detectable (AUROC multivariante **0,873**) | Control de generalidad con **evidencia no interpretable etiquetada por experto** (`noisy`) | Alta para H4/H5/H6: muestra que la certeza lingüística sigue la *forma* del dato, no su calidad; y que la calidad **sí** deja rastro cuando las features son buenas | Concepto diferente: ritmo cardiaco, no afecto/estrés inducido |
| **K-EmoCon** | Afecto (arousal/valencia) en habla espontánea con muñeca | 44 ventanas (piloto LLM); 5.528 ventanas de 5 s en la tabla de referencia | Referencia features-only: mejor AUROC **0,609** (`external_arousal`), 0,527 auto-arousal, 0,417 auto-valencia (bajo azar); LLM 0,614 balanced / 0,642 AUROC; el contraste `pre`/`debate` (0,708 bal / 0,785 AUROC) mide **tarea/actividad**, no afecto | Intento de validación externa del componente afectivo | Baja: delimita qué no se puede afirmar con datos de muñeca; muestra el confusor de orden del protocolo; justifica no repetir la ingesta | No hay etiqueta de estrés; etiquetas afectivas dispersas; referencia con EDA/HR de muñeca no pasa de AUROC 0,61 |

**Qué aporta cada uno:**

- **PhysioNet** es el mejor banco de pruebas para H4/H5 que hemos tenido: contiene
  la clase que un experto declara ilegible (`noisy`) y el modelo **no se abstiene ni una
  vez** y además **sube** su certeza. Refuerza la lectura de que la certeza
  lingüística sigue la *forma* del dato, no su calidad. También muestra que la
  calidad **sí** deja rastro cuando las features son buenas (AUROC 0,873), lo que
  sostiene H6.
- **K-EmoCon** se cerró como **resultado negativo documentado**: no hay etiqueta
  de estrés, las etiquetas afectivas son dispersas y la referencia con EDA/HR de
  muñeca no pasa de AUROC 0,61. Su valor es metodológico: (i) delimita qué no se
  puede afirmar con datos de muñeca, (ii) muestra el confusor de orden del
  protocolo y (iii) justifica no repetir la ingesta.

**Qué falta para una validación externa del endpoint.** Un dataset con
**inducción de estrés** y canales de pecho o EDA fiables. Candidatos no
incorporados aún (pendientes de verificar acceso y canales): **UBFC-Phys** y
**SWELL-KW**. Sin uno de ellos, la validez externa del endpoint de estrés queda
como limitación declarada.

**Nota sobre no forzar inclusión:** ni PhysioNet ni K-EmoCon se incluyen como
alternativas válidas para el endpoint de estrés porque su fenómeno es distinto.
Su inclusión en la tesis está justificada únicamente como controles externos de
generalidad de mecanismos, no como sustitutos de WESAD.

### 7.6 Conclusión

WESAD es la elección correcta **para esta tesis** porque el objeto de estudio es
la sensibilidad del lector a la evidencia, no el fenómeno del estrés: se necesita
un dataset con canales separables, etiquetas de referencia y tamaño que permita
la matriz completa. Sus debilidades (n, secuencialidad, etiqueta de protocolo) no
afectan a la comparación interna entre condiciones, que es donde viven H1–H6, y
están cuantificadas.

Las alternativas no lo desplazan: **DREAMT** mejora la referencia externa pero
cambia de fenómeno; **K-EmoCon** y **PhysioNet 2017** son controles externos de
concepto diferente (no validan el endpoint de estrés); **DREAMT (o un dataset
con inducción de estrés)** como extensión para validez externa. El diseño
más defendible es el que ya está en marcha: **WESAD como experimento principal,
PhysioNet como control de generalidad (H4/H5/H6) y DREAMT o un dataset con
inducción de estrés como extensión para validez externa del endpoint.**

### 7.7 Ranking de datasets en papers con alta similitud a esta tesis (sim. A1/A2/M)

El corpus SLIP contiene 117 papers con ficha. De ellos, **94 papers** tienen similitud **A1** (usa WESAD), **A2** (señal fisiológica + ejes de la tesis: degradación, ruido, calidad, incertidumbre, calibración, abstención, claims, fiabilidad) o **M** (señal fisiológica con LLM en otra tarea, o métodos de incertidumbre/calibración/abstención en texto). A continuación se presenta el ranking de los **20 datasets más utilizados** en esos 94 papers, ordenados por número de apariciones. WESAD aparece en 4 de ellos; el resto son datasets alternativos que abordan problemas similares (degradación de señal, fisiología wearable, inferencia con LLM sobre datos fisiológicos, confianza/abstención en LLM).

| Rank | Dataset | N papers | Papers |
|------|---------|----------|--------|
| 1 | **WESAD** | 4 | scopus_111, pubmed_150, ieee_88, arxiv_51 |
| 2 | **MIMIC-III Matched Waveform Database v1.0** | 4 | arxiv_288, manual_2, arxiv_70, scopus_396 |
| 3 | **ZuCo 1.0** | 2 | pubmed_168, scopus_484 |
| 4 | **PH-LLM / Google Digital Wellbeing Study** | 2 | wos_308, arxiv_190 |
| 5 | PMData | 1 | scopus_111 |
| 6 | DREAMER | 1 | pubmed_150 |
| 7 | MAHNOB-HCI | 1 | pubmed_150 |
| 8 | @HOME | 1 | pubmed_150 |
| 9 | PhysioNet benchmark | 1 | scopus_293 |
| 10 | HHAR | 1 | ieee_88 |
| 11 | DREAMT | 1 | ieee_88 |
| 12 | TILES-2018 | 1 | arxiv_51 |
| 13 | PhysioBench | 1 | manual_3 |
| 14 | AI-tennis coaching case study | 1 | manual_5 |
| 15 | UK Biobank | 1 | scopus_198 |
| 16 | PH-LLM case studies | 1 | wos_308 |
| 17 | 99 fitness (MCQs) | 1 | wos_308 |
| 18 | Google DWB/PROMIS | 1 | wos_308 |
| 19 | 833 survey responses | 1 | wos_308 |
| 20 | ZuCo 1.0 Task 3 | 1 | pubmed_168 |

**Total datasets únicos en papers con similitud A1/A2/M:** 218  
**Total papers con similitud A1/A2/M:** 94  
**Papeles que usan WESAD como dataset primario:** 4 (scopus_111, pubmed_150, ieee_88, arxiv_51)

**Observaciones:**

1. **WESAD y MIMIC-III son los únicos datasets con 4 apariciones** en papers de alta similitud, lo que los sitúa como los dos recursos de referencia más utilizados en este campo.
2. **ZuCo 1.0 y PH-LLM / Google Digital Wellbeing Study** aparecen en 2 papers cada uno, mostrando uso emergente pero limitado.
3. La gran mayoría de datasets (203 de 218) aparecen en **solo 1 paper**, lo que indica una fuerte fragmentación: no existe un estándar dominante fuera de WESAD y MIMIC-III para este tipo de investigación.
4. Muchos papers usan **datasets propios o sintéticos** (AI-tennis coaching case study, WearableQA benchmark, PPG–ECG dataset, etc.), lo que dificulta la comparabilidad entre estudios.
5. WESAD es el **único dataset** entre los 20 primeros que combina señal fisiológica de pecho (ECG, EDA, respiración), etiquetas de protocolo conocidas (4 clases afectivas), degradación controlada posible y uso por múltiples papers de alta similitud.

### 7.8 Papers de ACL Anthology con mayor relevancia y sus datasets

Solo los papers con prefijo `acl_anthology_*` pertenecen a **ACL Anthology**
(Association for Computational Linguistics). Los papers `ieee_*`, `scopus_*`,
`pubmed_*`, `arxiv_*`, `wos_*` y `manual_*` provienen de otros repositorios y
**no son de ACL**.

De los **94 papers con similitud A1/A2/M** del corpus SLIP, **14 son de ACL
Anthology** (todos con similitud **M**: métodos de incertidumbre, calibración,
abstención o fiabilidad en texto, transferibles a esta tesis). A ellos se suman
**3 papers ACL con similitud B**. En total, 17 papers ACL en la franja alta de
relevancia; **ninguno usa WESAD**. La tabla los ordena por relevancia conceptual
para la tesis (degradación de la entrada, confianza/certeza, respaldo de
afirmaciones, abstención y representación).

| # | Paper | Título | Dataset(s) | Sim. | Relevancia (hipótesis) |
|---|-------|--------|------------|:----:|------------------------|
| 1 | `acl_anthology_65` | LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models | **LENS**: 258 participantes; 50.957 EMAs; >100.000 pares sensor-texto; Garmin vivoactive 3 + Android (GPS, pasos, ACC, HR, sueño, estrés) | 0.77 M | Señal wearable → lenguaje; el más cercano al puente sensor-texto de H3/H5 |
| 2 | `acl_anthology_34` | Can LLMs understand unvoiced speech? Exploring EMG-to-text conversion with LLMs | **Gaddy & Klein unvoiced EMG**: 1 hablante, 8 canales, 67 palabras, ~26 min, 500 utterances; **CSL-EMG_Array** | 0.76 M | Señal fisiológica → texto (EMG): representación de señal para el LLM (H3) |
| 3 | `acl_anthology_245` | Speech disfluencies and LLM confidence: length bias and pragmatic insensitivity in Brazilian Portuguese | **Roda Viva corpus**: 344 turnos | 0.76 M | Confianza del LLM frente a incertidumbre pragmática; sesgo de longitud (H2b) |
| 4 | `acl_anthology_148` | Do language models mirror human confidence? Exploring psychological insights to address overconfidence in LLMs | **MMLU** (high school/college) y **GPQA** (expert) | 0.75 M | Sobreconfianza y calibración de la confianza (H2a) |
| 5 | `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | **TriviaQA**; **NQ-Open**; **PopQA** | 0.75 M | Incertidumbre verbal como rasgo lineal; hedges/boosters frente a alucinación (H2b, H5) |
| 6 | `acl_anthology_231` | ClinDet-Bench: Beyond abstention, evaluating judgment determinability of LLMs in clinical decision-making | **ClinDet-Bench**: 16 sistemas de scoring; 188 casos; 8 LLMs | 0.75 M | Determinabilidad y abstención con información incompleta (H4) |
| 7 | `acl_anthology_286` | Vulnerability of LLMs' stated belief? LLMs belief resistance check through strategic persuasive conversation interventions | **BoolQ**; **PubMedQA**; **LatentHatred** | 0.75 M | Confianza verbalizada y vulnerabilidad a la persuasión (H2b) |
| 8 | `acl_anthology_303` | Can a remedy find a researcher? Exploring semantic knowledge in Italian BabyLMs | **Benchmark italiano de pares mínimos**: 272 AI, 332 AA, 282 AA-rev; validación con 232 participantes; BAMBI y LMs italianos | 0.75 M | Competencia semántica y calibración (ECE) de LMs pequeños (H3) |
| 9 | `acl_anthology_57` | Teaching LLMs to Abstain across Languages via Multilingual Feedback | **M-MMLU**; **M-Hellaswag**; **Belebele** | 0.74 M | Abstención como comportamiento entrenable, no umbral fijo (H4) |
| 10 | `acl_anthology_56` | Decompose and Compare Consistency: Measuring VLMs' Answer Reliability via Task-Decomposition Consistency Comparison | **SNLI-VE**; **VCR**; **A-OKVQA**; **Winoground**; **MMMU**; **MathVista** | 0.74 M | Fiabilidad/consistencia de respuestas; Brier y Effective Reliability (H2, H5) |
| 11 | `acl_anthology_8` | Evaluating Subjective Cognitive Appraisals of Emotions from Large Language Models | **COVIDET-APPRAISALS**: 241 posts de Reddit; 21 dimensiones de appraisal | 0.74 M | Appraisal cognitivo desde texto; rationales y acuerdo con humanos (H5) |
| 12 | `acl_anthology_2` | Psycholinguistic Diagnosis of Language Models' Commonsense Reasoning | **Singh et al. (2016)**; **Nieuwland et al. (2010)**; **ImpPress** | 0.74 M | Razonamiento pragmático de LMs (evaluación de la representación textual) |
| 13 | `acl_anthology_288` | When LLMs Disagree with Human Experts: Understanding LLM Annotation Failures in Nutrition Misinformation | **169 posts de Instagram sobre seed oils** (3 nutricionistas; Fleiss' κ=0.813) | 0.74 M | Fallos de anotación y sobreconfianza sistemática (H2a, H5) |
| 14 | `acl_anthology_72` | Multimodal Affective Modeling in an LLM-based Intelligent Tutoring System for Foreign Language Learning | **EmoBot** proof-of-concept (sin dataset público nombrado; HR, SpO₂, HRV de smartwatch) | 0.74 M | Afecto multimodal con señal de wearable (H3) |
| 15 | `acl_anthology_169` | Vers une évaluation rigoureuse des systèmes RAG : le défi de la due diligence | **Dataset RAG due diligence**: 300 documentos DataCorp; 121 preguntas; 20 respuestas/pregunta | 0.75 B | Rigor de evaluación con jueces LLM (H5) |
| 16 | `acl_anthology_243` | Beyond Toxic Positivity: Interpersonal Affect Regulation in LLM-Based Dialogue Agents using Discourse Politeness Theory | **Simulación longitudinal de 8 semanas** (18 observadores terceros) | 0.74 B | Regulación afectiva en diálogo (sin señal fisiológica) |
| 17 | `acl_anthology_73` | Gladiator at MEDIQA-SYNUR 2026: Contextual Clinical Extraction Integrating Foundation Models with Domain-Specific Validation Rules | **MEDIQA-SYNUR** (122 train, 101 dev, test held-out) | 0.75 B | Extracción clínica guiada por esquema (texto) |

**"Ranking" de datasets usados por papers de ACL.** Ningún dataset se repite
entre los 17 papers: cada uno usa su propio conjunto. No existe, por tanto, un
dataset dominante comparable al caso WESAD/MIMIC-III de §7.7. Por tipo de dato:

| Tipo de dato | N papers ACL | Papers |
|---|---:|---|
| Texto, QA o diálogo (sin señal fisiológica) | 14 | `acl_anthology_2`, `_8`, `_56`, `_57`, `_73`, `_118`, `_148`, `_169`, `_231`, `_243`, `_245`, `_286`, `_288`, `_303` |
| Señal fisiológica o wearable → lenguaje | 3 | `acl_anthology_34` (EMG), `acl_anthology_65` (LENS), `acl_anthology_72` (EmoBot) |

**Observaciones clave:**

1. **Ninguno de los 17 papers de ACL usa WESAD.** Los papers que usan WESAD
   (`scopus_111`, `pubmed_150`, `ieee_88`, `arxiv_51`) tienen prefijo no-ACL
   (Scopus, PubMed, IEEE, arXiv): son de otros repositorios.
2. **Fragmentación total**: 17 papers ACL, 17 conjuntos distintos. No hay un
   dataset de referencia compartido, a diferencia de WESAD/MIMIC-III en §7.7.
3. **LENS** (`acl_anthology_65`) es el más cercano conceptualmente: alinea
   sensores wearables (incluye HR, sueño y estrés) con lenguaje para narrativas
   clínicas de salud mental.
4. **EMG-to-text** (`acl_anthology_34`) y **EmoBot** (`acl_anthology_72`) aportan
   señal fisiológica, pero en dominios distintos (comunicación sin voz y
   aprendizaje de idiomas).
5. **Implicación para la tesis**: el hueco que ocupa este trabajo —fisiología de
   pecho con etiquetas de protocolo, degradación controlada de la entrada y
   evaluación de corrección, confianza, claims y abstención— no lo cubre ningún
   paper de ACL del corpus. Los papers ACL aportan los **métodos** (calibración,
   hedges/boosters, abstención, fiabilidad de claims), no el **dato**.

**Conclusión:** WESAD sigue siendo el vehículo adecuado y el vacío de
investigación es real; la literatura ACL con la que dialoga esta tesis aporta
los ejes metodológicos (H2, H4, H5), no datasets fisiológicos de pecho.

---
