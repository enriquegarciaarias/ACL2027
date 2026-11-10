# Análisis de literatura del corpus SLIP para las fases de ingesta y evaluación

**Fecha:** 2026-11-08
**Corpus:** `docs/SLIP/papers_text.json` y `docs/SLIP/papers_zoned.json` (126 registros)
**Alcance:** solo Fase 1 (`INGEST`) y Fase 3 (evaluación). No se analiza aquí la Fase 2
salvo cuando una aportación afecta directamente a la evaluación.
**Herramienta de búsqueda:** `scripts/search_slip_docs.py` (BM25, stdlib).

---

## 0. Resumen ejecutivo

1. **Documento recomendado para buscar: `papers_zoned.json`.** Es la misma
   colección de papers que `papers_text.json`, pero con el texto completo
   dividido por secciones (`abstract`, `introduction`, `methodology`,
   `results`, `conclusion`, `figures_tables_captions`, `references`) y
   etiquetado por rol (`method`, `evidence`, `claim`, `background`, `gap`,
   `future_work`). Permite filtrar por sección y recuperar el pasaje donde
   está la evidencia. `papers_text.json` queda como complemento para dos
   cosas: el abstract compacto (`clean_text`) y la ruta al PDF (`pdf_file`).
2. **El corpus no es específico de WESAD.** Es una colección sobre LLM +
   sensores fisiológicos/wearables, series temporales clínicas, calibración,
   abstención y sesgos epistémicos. Solo unos pocos papers usan WESAD
   (`scopus_111` explícitamente). Sirve para diseñar experimentos y
   justificar decisiones, no como evidencia directa sobre nuestros datos.
3. **Los papers más accionables para la ingesta** son `manual_5` (degradación
   graduada y estabilidad secuencial), `scopus_355` (taxonomía de anomalías:
   missing/plausibility/conformance), `manual_2`/`arxiv_288`
   (sobre-confianza en medidas derivadas, con evidencia emparejada vs
   barajada), `arxiv_25` (missingness explícita), `scopus_244` (validación
   formal de pipelines) y `scopus_111` (ablación de features en WESAD).
4. **Los papers más accionables para la evaluación** son `acl_anthology_231`
   (determinabilidad vs abstención), `arxiv_280` (evaluación a nivel de
   evento/claim), `acl_anthology_148` (AFCE y sobreconfianza),
   `acl_anthology_245` (sesgo de longitud en la confianza y hedges),
   `acl_anthology_118` (mismatch incertidumbre verbal/semántica),
   `acl_anthology_56` (consistencia por descomposición), `scopus_396`
   (conformal prediction) y `arxiv_199` (fuga temporal/hindsight).
5. **Cinco oportunidades nuevas de mayor impacto** para nuestros objetivos:
   - **E1. Taxonomía de degradación fisiológica** (missing, plausibility,
     conformance) en vez de solo caída/NaN/flat.
   - **E2. Barrido de severidad + estabilidad secuencial (`FlipRate`)** en vez
     de condiciones C0–C5 discretas.
   - **E3. Señalización de fiabilidad vs pérdida real de información** (2×2,
     incluida señalización engañosa).
   - **E5. Determinabilidad vs abstención**: distinguir “no hay información”
     de “hay información suficiente aunque falte un canal”.
   - **E6. Evaluación a nivel de claim/evento fisiológico**, que hoy no
     medimos (solo decisión y confidence).

---

## 1. Comparación de los dos documentos

### 1.1. Estructura

| Dimensión | `papers_text.json` | `papers_zoned.json` |
|---|---|---|
| Registros | 126 | 126 |
| Texto completo | `clean_sections` (9,74 M chars) | `paragraphs` (9,63 M chars) |
| Texto compacto | `clean_text` (3,15 M chars) | — |
| Granularidad | Sección entera en un campo | Una fila por sección (747 filas) con `section` + `zone` + `paragraph_idx` |
| Metadatos | `paper_id`, `pdf_file`, `word_count`, `char_count` | `paper_id`, `title` (vacío), `paragraphs` |
| Filtros posibles | Ninguno (todo el paper) | `section`, `zone`, `paper_id` |
| Provenance | Ruta al PDF | — |

### 1.2. Evidencia empírica de la comparación

- **Mismo contenido subyacente.** La ratio de longitud
  `zoned / clean_sections` tiene mediana **1,000**; la cobertura de tokens
  (palabras de ≥4 letras) tiene mediana **1,000**. En 123/126 papers el texto
  es equivalente. No se pierde contenido al usar `zoned`.
- **`clean_text` es casi siempre el abstract, no el texto completo.** La
  ratio `clean_text / clean_sections` tiene mediana **0,187**; en 72/126
  papers es <30 %. Ejemplo: `pubmed_128` declara `word_count=729` y
  `char_count=4966`, pero sus secciones suman 112.554 caracteres. Los campos
  `word_count`/`char_count` de `papers_text.json` no son fiables para saber
  cuánto texto hay.
- **Buscar solo en `clean_text` perdería evidencia técnica.** Términos que
  aparecen en el cuerpo pero no en el abstract (calculado sobre `zoned`):

  | Término | Papers con el término | Solo en cuerpo (no en abstract) |
  |---|---|---|
  | `quality` | 87 | 30 |
  | `reliab` | 79 | 31 |
  | `evidence` | 75 | 24 |
  | `window` | 47 | 21 |
  | `calibrat` | 29 | 17 |
  | `noise` | 39 | 15 |
  | `claim` | 23 | 15 |
  | `artifact` | 25 | 10 |
  | `missing` | 40 | 10 |
  | `abstain` | 7 | 5 |
  | `hedg` | 5 | 5 |

  Es decir: una búsqueda sobre el abstract habría perdido, por ejemplo, el
  59 % de los papers que hablan de calibración y el 100 % de los que usan el
  término “hedge”.
- **Prueba de recuperación con una consulta real.** Para
  `"calibration abstention confidence"`, la vista `text` devuelve papers
  completos y coloca en el top-5 el par duplicado `arxiv_288`/`manual_2`;
  la vista `zoned` devuelve pasajes concretos de `results`/`methodology` y
  recupera en el top-5 `acl_anthology_231` (ClinDet-Bench), que es el paper
  más pertinente para abstencion/determinabilidad. Además permite filtrar
  por `--section results` para excluir introducciones y opiniones.
- **`zoned` permite filtrar por sección/rol**, lo que reduce falsos positivos
  y acelera la lectura. `text` obliga a leer el paper entero o a hacer un
  split propio.

### 1.3. Limitaciones de `zoned` (y cómo mitigarlas)

- **No tiene títulos.** Todos los `title` están vacíos. Se resuelven con
  `papers_metadata.json` del proyecto SLIP (título, año, venue, DOI). El
  buscador lo usa solo para identificar el paper; la búsqueda no depende de él.
- **Los “párrafos” son en realidad bloques de sección**, no párrafos finos:
  747 bloques / 126 papers ≈ 5,9 bloques por paper, uno por sección. Un bloque
  de metodología puede tener 50.000 caracteres (`manual_5`). Para leer, el
  buscador devuelve snippets; para análisis fino hay que trocear por frases.
- **Las etiquetas de sección son heurísticas y a veces fallan.** Ejemplo:
  `manual_2`/`arxiv_288` tiene 32.607 caracteres de cuerpo etiquetados como
  `introduction` y solo 73 como `methodology`. No debemos confiar ciegamente
  en la etiqueta; sí nos avisa de que algo raro pasa.
- **Duplicados:** `arxiv_288` y `manual_2` son el mismo paper (título y texto
  idénticos). `arxiv_276` y `pubmed_168` comparten prefijo de abstract (casi
  duplicados). El corpus tiene 126 registros pero ~124 papers únicos.

### 1.4. Recomendación operativa

- **Índice primario:** `papers_zoned.json` a nivel de sección, con filtros
  `--section` y `--zone`.
- **Triage rápido:** `clean_text` de `papers_text.json` como abstract.
- **Provenance/cita:** `pdf_file` de `papers_text.json` + metadatos del SLIP.
- **Comando tipo:**

  ```bash
  python scripts/search_slip_docs.py "data quality anomaly degradation" \
    --top 15 --papers --section methodology,results
  python scripts/search_slip_docs.py "calibration confidence" \
    --top 10 --section results --zone evidence
  ```

---

## 2. Método de búsqueda

`scripts/search_slip_docs.py` implementa BM25 Okapi (k1=1,5; b=0,75) sobre
los bloques de `zoned` o sobre los papers de `text`. Añade:

- stopwords académicas (incluye `study`, `results`, `method`, `figure`…),
- pesos por sección (`abstract` 1,20; `methodology` 1,10; `results` 1,05;
  `conclusion` 0,95; `introduction` 0,90; `captions` 0,55; `references` 0,15),
- pesos por rol (`method` 1,15; `evidence` 1,10; `claim` 1,10; `gap` 1,20;
  `future_work` 1,20; `background` 0,90; `rest` 0,80),
- soporte de frases entre comillas y salida agrupada por paper.

Consultas ejecutadas (todas reproducibles):

```text
sensor degradation noise robustness telemetry
verbalized confidence calibration overconfidence abstention
missing data imputation irregular time series wearable
electrodermal activity EDA skin conductance SCR feature extraction
evidence grounding claims verification faithfulness hallucination
selective prediction abstain insufficient evidence uncertainty
hedges boosters epistemic modality certainty language
signal quality index artifact rejection ECG PPG
inter-subject variability personalization baseline physiological
prompt sensitivity question formulation benchmark
window length segmentation sampling rate signal processing
multimodal synchronization alignment different sampling rates
data leakage label leakage evaluation contamination
stress detection wearable WESAD chest ECG EDA
LLM annotation error taxonomy overconfidence claim strength
numerical time series serialization tokenization prompt representation
randomized controlled prompt variants temperature reproducibility seeds
```

---

## 3. Mapa del corpus

- **126 registros**, ~124 papers únicos.
- **Fuentes:** `scopus` (31), `ieee` (23), `pubmed` (22), `arxiv` (18),
  `acl_anthology` (17), `wos` (10), `manual` (5).
- **Años:** 2009–2026, con concentración en 2025–2026.
- **Temáticas dominantes:**
  1. LLM + sensores fisiológicos/wearables y salud mental.
  2. Series temporales médicas y *time-series LLMs*.
  3. Calibración, incertidumbre, abstención y sobreconfianza.
  4. Anotación clínica, extracción estructurada y evaluación de fidelidad.
  5. Benchmarks de razonamiento fisiológico (`PhysioBench`, `WearableQA`).
- **Hueco:** no hay papers específicos sobre WESAD y degradación controlada
  de EDA/ECG con LLM; la oportunidad experimental es real.

---

## 4. Papers relevantes para la INGESTA

### 4.1. Degradación y calidad de datos

#### `manual_5` — *Telemetry degradation and sequential stability in LLM-integrated IoT decision systems* (2026)

**Qué aporta.** Framework para evaluar estabilidad de decisiones LLM bajo
degradación controlada de telemetría. Define dos ejes de severidad —
**dropout** \(D \in \{0;0{,}1;0{,}2;0{,}3;0{,}4;0{,}6\}\) y **downsampling**
\(K \in \{1,2,3,5,10\}\)— más una condición de **reliability signaling**
(metadatos de retraso/obsolescencia sin cambiar los datos). Métricas:
`FlipRate` (volatilidad de decisión entre ventanas consecutivas),
`EscalationRate`, `IncoherenceRate`, `ViolationRate`, `time-to-violation`,
`NetLoadDrift`; barrido de temperatura \(\{0;0{,}2;0{,}5\}\). Resultado:
`FlipRate` sube de 0,144 a 0,207 (+44 %) con downsampling; los regímenes con
una clase dominante amplifican la inestabilidad; el prompt estructurado
amortigua pero no corrige.

**Cómo encaja.** Nuestro `degrade.py` ya hace 0/0,5/1,0 sobre canales. Este
paper justifica: (a) convertir la severidad en un eje continuo; (b) añadir
downsampling temporal (no solo dropout); (c) separar “pérdida de
información” de “señalización de fiabilidad”; (d) medir estabilidad
secuencial y no solo acierto por ventana. Nuestro diseño es *open-loop* como
el suyo, así que es directamente comparable.

**Límite.** Es un caso de tenis con IMU, no fisiología; las métricas hay que
adaptarlas (p. ej. `FlipRate` entre ventanas del mismo sujeto/condición).

#### `scopus_355` — *Mind the data: evaluating data quality sensitivity in medical LLMs* (2026, Pattern Recognition Letters)

**Qué aporta.** Dataset LQMR con tres familias de anomalías sobre datos
estructurados: **missing values**, **plausibility violations** (valores
individualmente normales que violan dependencias entre variables) y
**conformance errors** (violaciones de esquema/columnas). 5 datasets, 15
subdatasets, 6.789 casos. Métricas: AUC-ROC y cambio porcentual; además,
capacidad de detección de anomalías y prompts de mitigación. Resultado:
las anomalías de plausibilidad son las más dañinas (−16,79 %), seguidas de
conformance (−11,45 %); los LLM detectan mal los errores sutiles.

**Cómo encaja.** Nuestra degradación actual cubre sobre todo “missing” y
“sensor plano”. Falta:
- **plausibilidad fisiológica**: combinaciones imposibles (p. ej. HR 55 bpm
  con `eda_scr_rate` 10/min; RR incompatible con HR; temperatura 42 °C con
  EDA normal);
- **conformance**: unidades o escalas incoherentes, ventanas con sampling
  rate distinto, canales con longitud distinta.
Esto permite separar “el modelo falla porque falta información” de “el
modelo falla porque la información es incoherente”, que son hipótesis
distintas en H1.

#### `manual_2` / `arxiv_288` — *When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence* (2026)

**Qué aporta.** Define **derived-feature over-trust (DFOT)**: el LLM trata
una medida derivada (p. ej. HR de PPG) como un hecho directo, ignorando que
su validez depende de la instancia. Usa una modalidad privilegiada (ECG de
referencia) que el LLM nunca ve. Dos escenarios: **D1** el LLM acepta una
medida derivada contradicha por ECG; **D2** el LLM rechaza una medida
derivada fiable por un historial engañoso. Cinco estimandos: `COTR` (aceptar
a pesar del conflicto), `CIR` (rechazar por contexto), `CRR` (reparación),
`ESRM` (evidencia emparejada vs **barajada**), `UHR` (verificación
innecesaria). La comparación emparejada vs barajada es un control negativo
de especificidad: distingue “usa la evidencia” de “reacciona a la palabra
fiabilidad”.

**Cómo encaja.** WESAD tiene ECG de pecho y BVP/EDA de muñeca: podemos
construir HR derivada de muñeca y ECG de referencia. Es un experimento
nuevo y muy alineado con RQ2/RQ3: ¿sobreconfía el LLM en una HR derivada
cuando el ECG la contradice? ¿Mejora si le damos evidencia de fiabilidad?
¿Y si la evidencia es barajada (control)?. Además, `ESRM` es exactamente el
tipo de control que nos falta para demostrar que el LLM usa la calidad y no
solo el formato.

#### `scopus_460` — *Harnessing LLM for noise-robust cognitive diagnosis* (2026, WWW)

**Qué aporta.** Muestra que ruido en los logs puede invertir completamente
el perfil semántico generado; propone integración robusta al ruido.

**Cómo encaja.** Evidencia adicional de que la degradación no es solo
pérdida de acierto: puede cambiar el *significado* de la explicación. Sirve
para justificar métricas de estabilidad semántica, no solo exactitud.

---

### 4.2. Missingness y representación

#### `arxiv_25` — *Mind the Missing: VITAL* (2025)

**Qué aporta.** Distingue variables frecuentes (signos vitales) de
esporádicas (labs); codifica la ausencia con un token aprendible
`[Not measured]` en lugar de imputar; robustez creciente con el nivel de
missingness en PhysioNet.

**Cómo encaja.** Nuestra vista textual ya escribe “not available” y la
numérica `value=null`. Podemos convertirlo en un **factor experimental**:
tres políticas (eliminar la feature, imputar con mediana, marcar
explícitamente “not measured”) × nivel de missingness. Medir decisión,
abstención y claims. Esto también conecta con `scopus_198`, que sugiere que
los LLM pueden manejar missingness si se les indica explícitamente.

#### `scopus_198` — *Multimodal LLMs for health grounded in individual-specific data (HeLM)* (2024)

**Qué aporta.** Serializa datos tabulares a texto y usa encoders para
modalidades de alta dimensión; señala que los LLM pueden ser flexibles con
missing data sin imputar.

**Cómo encaja.** Justifica probar variantes de serialización (tabla vs
texto vs JSON) y no asumir que imputar es siempre lo correcto. Nuestro
`numeric_llm` y `text_llm` ya son dos variantes; faltaría una tercera
“tabla” explícita y una condición de missingness.

#### `arxiv_7` / `scopus_293` / `arxiv_64` — revisiones de *medical time series LLMs*

**Qué aportan.** Taxonomías de arquitecturas, representaciones (raw signal,
plot, resumen textual, embeddings), prompts, tareas y evaluación. `arxiv_7`
insiste en que la evaluación debe cubrir *grounding* fisiológico,
consistencia temporal, robustez clínica y razonamiento basado en evidencia,
no solo exactitud.

**Cómo encaja.** Marco para Related Work y para justificar nuestras
elecciones de representación. También advierte de que un LLM no ingiere
bien secuencias largas (`acl_anthology_65` lo dice explícitamente), lo que
refuerza nuestro enfoque de resúmenes estadísticos.

---

### 4.3. Validación y harmonización del pipeline

#### `scopus_244` — *LLMs + formal planning para diseñar y validar pipelines de biosignales* (2025)

**Qué aporta.** Combina un LLM (comprensión del lenguaje) con un planner
HTN (razonamiento formal) para generar y **validar** pipelines de
procesamiento de biosignales; marca violaciones de buenas prácticas.
Reporta precisión 0,89–0,98 y F1 0,97–0,98 en pipelines de 2–15 pasos.

**Cómo encaja.** No necesitamos el planner, pero sí su idea: un **contrato
de pipeline** comprobable. Reglas candidatas: Nyquist, bandas de filtro
válidas, orden de filtro, longitud mínima de ventana para HRV, rangos
fisiológicos, unidades, coherencia de sampling rate, missingness máxima.
Nuestro `units.py`/`validation.py` ya hace parte; el paper sugiere
formalizarlo y reportar violaciones como artefacto QC estructurado.

#### `scopus_118` — *Multimodal LLMs in human-centered health: practical insights* (2024)

**Qué aporta.** Protocolo de alineación entre datasets: unidades (m/s²,
rad/s), resampling, interpolación de missing, ventanas de 2,56 s con 50 %
solape, resampling a 50 Hz, normalización reversible (RevIN) y embedding de
posición del sensor.

**Cómo encaja.** WESAD mezcla 700 Hz (pecho) y 64/4 Hz (muñeca). Documentar
y testear la alineación es clave para reproducibilidad y para una futura
extensión multi-dataset. Propone una ablación de harmonización (resamplear
a una tasa común, normalizar por sujeto) que hoy no medimos.

#### `ieee_59` — *LLM-driven feature weighting for cross-device harmonization* (2026)

**Qué aporta.** El LLM, con few-shot, genera reglas/pesos por cohorte para
armonizar datos de distintos dispositivos; mejora F1 en validación externa
frente a pesos clásicos.

**Cómo encaja.** Experimento secundario: usar el LLM para derivar pesos de
features por dispositivo (pecho vs muñeca) o por sujeto. No es central para
nuestra tesis, pero es una línea de “LLM como herramienta de ingeniería de
datos” que puede dar un resultado publicable.

#### `pubmed_123` — *DataAtlas: data dictionaries with LLMs* (2026)

**Qué aporta.** Genera diccionarios de datos automáticamente; los valores
de muestra son clave para que el LLM “ancle” las descripciones; advierte de
sensibilidad a la calidad de la muestra.

**Cómo encaja.** Nuestro `feature_spec.json` es un diccionario de features;
podemos enriquecerlo con valores representativos y unidades, y usar ese
contexto en el prompt (si se decide) o en la documentación del corpus.

---

### 4.4. Features, fisiología y personalización

#### `scopus_111` — *StressLLM: LLMs for stress prediction via wearable sensor data* (2025)

**Qué aporta.** Usa PMData y **WESAD**; hace ablación sistemática de
biomarcadores. Hallazgos: few-shot suele ayudar; en WESAD, para
BioMistral-DARE el mejor MAE (0,73) se obtiene **eliminando ECG de pecho**,
y el peor (0,86) eliminando el acelerómetro de muñeca. La importancia de
features depende del modelo y del prompt.

**Cómo encaja.** Justifica una ablación *leave-one-modality-out* y
*leave-one-feature-out* más allá de nuestras C1–C5. También es un aviso: no
asumir que ECG es imprescindible; nuestro resultado de que ACC/EMG
discriminan estrés (confusor de movimiento) es coherente con que la muñeca
aporte señal, pero también con riesgo de confusor.

#### `pubmed_150` — *Seamless stress monitoring with a time-series foundation model* (2026, AI in Medicine)

**Qué aporta.** Reformula la detección de estrés como **detección de
anomalías** respecto a una **línea base personalizada**; robustez a señales
ruidosas de wearables de baja calidad.

**Cómo encaja.** Oportunidad para el problema de 4 clases: en vez de pedir
al LLM que clasifique valores absolutos, darle la **desviación respecto al
baseline del sujeto**. Puede separar mejor baseline vs estrés y reducir
variabilidad inter-sujeto. Es compatible con nuestras ventanas y con el
ground truth de WESAD.

#### `arxiv_51` — *HRV-conditioned personalization via LLM* (2026)

**Qué aporta.** Integra features HRV con atributos personales mediante
interpretación semántica del LLM; +29,9 % MCC relativo en TILES-2018 y
descripciones legibles.

**Cómo encaja.** Podemos añadir contexto del sujeto (edad, sexo,
antropometría si está disponible) y features HRV; pero cuidado con fugas y
confusores. Es un factor experimental, no una mejora gratuita.

#### `scopus_272` — *LLM-assisted explainable stress recognition with PPG threshold rules* (2026)

**Qué aporta.** Extrae HRV (RMSSD, HF, pNN50), deriva umbrales con
ROC-Youden y construye tres reglas lógicas simples; el LLM razona sobre
ellas. Rendimiento comparable con interpretabilidad alta.

**Cómo encaja.** Dos usos: (a) **modelo de referencia** interpretable para
la condición de resolubilidad; (b) **ground truth de claims**: si el LLM
afirma “HRV bajo”, se puede verificar contra la regla/umbral. Encaja con la
evaluación a nivel de claim (E6).

#### `ieee_88` — *An empirical study for effective LLM-based human state prediction from physiological data* (2026)

**Qué aporta.** Organiza el diseño en tres capas: **input representation**,
**LLM usage** y **output form**. La mejor combinación es feature input +
few-shot + **label-with-explanation**; los factores de sujeto y posición
temporal importan.

**Cómo encaja.** Nuestro diseño ya usa features + few-shot + decisión con
`reasoning`, pero no lo hemos medido como factorial. Este paper da la
estructura para un experimento 3×2×2 de bajo coste que responde a RQ4.

#### `acl_anthology_65` — *LENS: LLM-enabled narrative synthesis for mental health* (2026)

**Qué aporta.** Los LLM no ingieren secuencias largas de sensores; entrena
un encoder a nivel de patch para proyectar señales al espacio del LLM y
genera narrativas clínicas.

**Cómo encaja.** Frontera a medio plazo: si quisiéramos ir más allá de
resúmenes estadísticos, un encoder entrenado sería la vía. Para esta tesis,
justifica no intentar meter series crudas en el prompt.

### 4.5. Implicaciones inmediatas verificadas en nuestros datos

Al contrastar la literatura con el corpus WESAD aparecieron tres hechos que
afectan al diseño experimental:

1. **`bvp` es una modalidad de muñeca, no de pecho.** El registro
   `ModalitySpec("bvp", "wrist_BVP", ...)` lo confirma. Estaba incluida por
   error en `LLM_CHEST_MODALITIES`; se ha movido a `WRIST_MODALITIES`. Esto
   refuerza la oportunidad E4: HR derivada de BVP frente a ECG de referencia.
2. **La frecuencia respiratoria separa `meditation`.** Medias de las 705
   ventanas: baseline 16,0 rpm, stress 14,2, amusement 16,7, meditation 9,4.
   Es la señal con mayor potencial para la clase que hoy está en AUROC ≈0,5.
   El few-shot se ha actualizado para mostrarla.
3. **La temperatura media es un posible confusor de orden.** Baseline 33,4 °C
   frente a 34,1–34,2 °C en las demás condiciones; el protocolo WESAD empieza
   por baseline, así que la deriva térmica puede codificar el orden de la
   sesión. Se recomienda un análisis de sensibilidad excluyendo `temp` del
   prompt (E12) y no usarla como ejemplo few-shot.
4. **`numeric_llm` y `text_llm` no contenían el mismo conjunto de features.**
   El filtro por feature (`EDA_HR_FEATURES` + `ADDITIONAL_FEATURES`) solo se
   aplicaba al prompt numérico; la vista textual incluía todas las features de
   las modalidades permitidas. Se ha centralizado el allowlist en
   `LLM_ALLOWED_FEATURES` (`represent.py`) y ambas vistas lo aplican. Sin esta
   corrección, la comparación de representaciones (RQ4) estaba confundida por
   un conjunto distinto de features, no solo por el formato.

---

## 5. Papers relevantes para la EVALUACIÓN

### 5.1. Abstención, determinabilidad y selectividad

#### `acl_anthology_231` — *ClinDet-Bench: Beyond Abstention* (2026)

**Qué aporta.** Benchmark que separa **determinable** vs **undeterminable**
bajo información incompleta. Tres etiquetas de respuesta (`Met`, `Not met`,
`Unable to determine`) y tres prompts (base, CoT, “safe”). 16 sistemas de
scoring clínico. Hallazgo: los LLM producen **conclusiones prematuras** y
**abstenciones excesivas**; incluso explican bien la regla pero fallan al
aplicarla con información faltante.

**Cómo encaja.** Es el complemento que falta a nuestros controles
negativos. Hoy `NC_nan/NC_zero/NC_absent` solo prueban el extremo
“indeterminable”. Necesitamos condiciones **determinable-incompletas**
(p. ej. falta EDA pero HR y respiración bastan) y medir:
- `over_abstention_rate` en determinables,
- `premature_conclusion_rate` en indeterminables,
- accuracy en determinables.
Esto responde directamente a RQ2 y a la parte de abstención de H2b.

#### `acl_anthology_56` — *DeCC: Decompose and Compare Consistency* (2024)

**Qué aporta.** Estima fiabilidad comparando la respuesta directa con la
respuesta obtenida descomponiendo la pregunta en subpreguntas; mejor
correlación con accuracy que likelihood o confianza verbalizada; útil para
*selective prediction*.

**Cómo encaja.** Podemos implementar una variante barata: pedir al modelo
(a) decisión directa y (b) respuestas a subpreguntas (“¿HR elevada?”,
“¿EDA elevada?”, “¿movimiento alto?”) y medir consistencia. Es una señal de
incertidumbre sin necesidad de logits ni de entrenamiento.

#### `scopus_396` — *Conformal prediction para incertidumbre de LLM* (2026)

**Qué aporta.** Conformal prediction sobre predicciones de GPT-4o;
cobertura empírica 90,4 % con objetivo 90 %, pero cobertura por clase
desequilibrada (99,7 % muerte vs 81,1 % supervivencia). Expone la
sobreconfianza en errores.

**Cómo encaja.** Framework para umbrales de abstención con garantía de
cobertura y para reportar cobertura por clase (crítico en 4 clases
desbalanceadas). Requiere un conjunto de calibración por sujeto/split; es
una mejora metodológica fuerte para RQ2.

---

### 5.2. Confianza, calibración y lenguaje

#### `acl_anthology_148` — *Do Language Models Mirror Human Confidence?* (2025)

**Qué aporta.** Los LLM son menos sensibles a la dificultad que los humanos
y cambian su confianza con la **persona** (experto vs lego, demografía) sin
cambiar el acierto. Propone **AFCE** (Answer-Free Confidence Estimation):
primero se pide solo la confianza, luego la respuesta; reduce ECE en
preguntas difíciles.

**Cómo encaja.** Dos experimentos: (a) variante AFCE del prompt para medir
si mejora ECE/Brier; (b) control de persona: nuestro prompt usa “You are a
physiological signal analysis assistant”, que es una persona. Conviene
medir su efecto o eliminarla como confusor.

#### `acl_anthology_245` — *Speech disfluencies and LLM confidence* (2026)

**Qué aporta.** La confianza del modelo está dominada por la **longitud de
la respuesta** (β estandarizado +14,47) y no por marcadores pragmáticos;
los hedges no son significativos tras controlar longitud. Propone
calibración normalizada por longitud y un índice de compromiso epistémico
con hedges/boosters.

**Cómo encaja.** Nuestro plan de medir hedges/boosters tiene un confusor
claro: la longitud de `reasoning`. Hay que reportar por 100 palabras,
incluir longitud como covariable y no interpretar un aumento de hedges si
viene acompañado de respuestas más largas. Es una corrección metodológica
directa a H2b.

#### `acl_anthology_118` — *Calibrating Verbal Uncertainty as a Linear Feature* (2025)

**Qué aporta.** La incertidumbre verbal (VU) y la semántica (SU) no
coinciden; el **mismatch VU–SU** predice alucinaciones mejor que SU sola.
La VU es una dirección lineal en el espacio de representaciones; modularla
reduce ~30 % las alucinaciones confiadas.

**Cómo encaja.** Con Qwen2.5 (pesos abiertos) y varias muestras por prompt
podemos estimar SU (consistencia semántica) y VU (hedges/confianza), y
medir el mismatch como predictor de claims incorrectos. Encaja con el
análisis exploratorio de representaciones internas de la tesis.

#### `acl_anthology_288` — *When LLMs Disagree with Human Experts* (2026)

**Qué aporta.** Taxonomía jerárquica de errores de anotación: **dirección**
(over/under), **mecanismo** (interpretación, razonamiento) y **factores**
(fuerza del claim: absoluto/calificado/opinión; tipo de evidencia:
ninguna/citada/datos). 149 de 169 errores se reportaron con confianza ≥0,8.

**Cómo encaja.** Podemos adoptar la taxonomía para analizar nuestros errores
de decisión y de claim: ¿el modelo se equivoca por leer mal una feature, por
ignorar missingness, por sobre-generalizar? ¿Los errores se concentran en
claims absolutos? Esto da contenido a RQ3 y a la parte lingüística de H2b.

#### `arxiv_365` — *Daily and Weekly Periodicity in LLM Performance* (2026)

**Qué aporta.** Bajo condiciones fijas, GPT-4o muestra variabilidad
periódica diaria/semanal que explica ~20 % de la varianza.

**Cómo encaja.** Nuestros runs de 6–8 h cruzan horas del día. Debemos
registrar `generated_at` (ya se hace), aleatorizar/bloquear el orden de
condiciones, incluir hora/día como covariable y replicar condiciones clave
en días distintos. Es una amenaza a la reproducibilidad que hoy no está
documentada.

---

### 5.3. Claims, eventos, benchmarks y fuga temporal

#### `arxiv_280` — *Benchmarking LLM Summaries of Multimodal Clinical Time Series* (2026)

**Qué aporta.** Evaluación **a nivel de evento**: `abnormality recall`,
`duration recall`, `measurement coverage`, `hallucinated event mentions`.
Los eventos se definen con umbrales y persistencia temporal. Hallazgo
central: la “ilusión de fluidez”: modelos con alta similitud semántica
tienen recall de anomalías casi nulo (0–2 %); el prompting estadístico
mejora; la visualización es la mejor.

**Cómo encaja.** Es la plantilla para RQ3. En WESAD podemos definir eventos
fisiológicos verificables (p. ej. ráfaga SCR, elevación sostenida de HR,
episodio de movimiento) y medir si los claims del LLM los recuperan. Hoy
solo evaluamos decisión y confianza; este paper muestra que eso puede
ocultar omisiones graves.

#### `manual_3` — *PhysioBench* (2026)

**Qué aporta.** 22 datasets públicos, 61,4 M preguntas, 30 tareas, 21
modelos, 3 ajustes de evaluación. Cada par pregunta-respuesta está anclado
a un segmento de señal y es trazable a su anotación. Conclusión: ningún
modelo es consistente entre modalidades/tareas y el rendimiento es
sensible a la formulación de la pregunta.

**Cómo encaja.** Dos ideas: (a) construir un subconjunto WESAD trazable
(ventana → features → anotación) para comparabilidad; (b) tratar la
formulación de la pregunta como factor (paráfrasis, orden, vocabulario).

#### `manual_1` — *WearableQA* (2026)

**Qué aporta.** Benchmark de razonamiento sobre datos wearables reales.
Cada relación retenida se apoya en ≥2 estudios independientes en la misma
dirección, verificado contra PubMed/PMC; se registra en una ficha con
DOI/PMID, umbrales literales, cohorte y **“qué no establece la fuente”**.
Los patrones solo se promueven a pregunta si superan puertas de
consistencia (efecto |ρ|≥0,5, bootstrap ≥80 %, leave-one-out).

**Cómo encaja.** Protocolo ejemplar para construir ground truth defendible y
evitar sobre-claim. Podemos adoptar las “fichas de evidencia” para nuestros
items y las puertas de consistencia para validar eventos/umbrales (p. ej.
el umbral SCR 0,05 µS).

#### `acl_anthology_169` — *Evaluación rigurosa de sistemas RAG* (2025)

**Qué aporta.** Protocolo con anotación humana + LLM-juez para clasificar
fallos (alucinación, fuera de tema, cita defectuosa, abstención) y
`Prediction Powered Inference` (PPI) para métricas con garantías
estadísticas.

**Cómo encaja.** Si usamos un LLM-juez para evaluar claims, hay que
calibrarlo contra anotación humana y reportar acuerdo/incertidumbre. PPI es
una vía para no inflar la confianza en métricas estimadas por juez.

#### `arxiv_199` — *Hindsight Bias in Clinical Temporal Reasoning* (2026)

**Qué aporta.** Mide si el modelo usa información futura. Benchmark pareado:
respuesta prospectiva vs “trampa” consistente con el desenlace; métricas
`accuracy`, `hindsight trap rate`, `answer instability rate`,
`hindsight bias rate`. El enmascaramiento temporal reduce el sesgo sin
bajar la accuracy.

**Cómo encaja.** Nuestra anti-leakage actual cubre etiquetas, IDs y tiempos
absolutos, pero no “información futura”. Podemos construir trampas de
hindsight en WESAD: incluir en el prompt un resumen de ventanas posteriores
o el desenlace de la sesión, y comprobar si la decisión cambia. Es una
prueba de validez ecológica y de fuga.

#### `arxiv_162` — *From Classification to Clinical Insights* (2023)

**Qué aporta.** Cadena de pensamiento sobre datos móviles/de comportamiento;
robustez a modalidad faltante; hallazgo de alucinación severa en
interpretación de ECG; recompensas basadas en evidencia diagnóstica.

**Cómo encaja.** Refuerza la necesidad de condiciones de modalidad faltante
y de evaluación de claims, no solo de clasificación.

---

## 6. Oportunidades experimentales nuevas (priorizadas)

> Prioridad = impacto esperado sobre RQ1–RQ4 / coste de implementación.
> “Coste” es una estimación relativa en nuestro código actual.

### E1. Taxonomía de degradación fisiológica: missing / plausibility / conformance

- **Refs:** `scopus_355`, `manual_5`.
- **Diseño:** extender `degrade.py` con tres familias:
  1. *missing*: dropout de canales/ventanas (lo actual);
  2. *plausibility*: valores individualmente válidos pero incoherentes entre
     features (HR vs RR, SCR vs EDA media, temp vs EDA);
  3. *conformance*: unidades/escalas/sampling incoherentes o columnas
     intercambiadas.
- **Métricas:** accuracy, macro-F1, abstención, tasa de detección de la
  anomalía, cambio de decisión respecto a C0.
- **Coste:** bajo (config + generador de anomalías + evaluación).
- **Aporta:** separa “falta información” de “información incoherente”;
  refuerza H1 y RQ1.

### E2. Barrido de severidad y estabilidad secuencial (`FlipRate`)

- **Refs:** `manual_5`.
- **Diseño:** sustituir 0/0,5/1,0 por \(D \in \{0,0{,}1,\dots,0{,}6\}\) y
  \(K \in \{1,2,3,5,10\}\), más una condición de solo *reliability
  signaling*. Evaluar secuencias de ventanas consecutivas del mismo
  sujeto/condición.
- **Métricas:** `FlipRate`, `EscalationRate` (cambios de dirección),
  `IncoherenceRate`, `time-to-first-flip`, además de accuracy.
- **Coste:** bajo-medio (más llamadas; se puede hacer con un subconjunto).
- **Aporta:** mide estabilidad, no solo acierto; H1 y RQ1.

### E3. Señalización de fiabilidad vs pérdida real de información (2×2)

- **Refs:** `manual_5`, `manual_2`, `scopus_355`.
- **Diseño:** factorial: datos {limpios, degradados} × metadatos de calidad
  {ausentes, presentes, **presentes pero engañosos**}. La condición engañosa
  es clave: un LLM que solo reacciona a la palabra “quality” se delata.
- **Métricas:** accuracy, abstención, `ESRM`-like (evidencia correcta vs
  barajada), cambio de confianza.
- **Coste:** bajo (variantes de prompt + metadatos).
- **Aporta:** RQ2/RQ3; distingue uso de evidencia de cautela genérica.

### E4. Sobre-confianza en medidas derivadas (DFOT) con ECG como referencia privilegiada

- **Refs:** `manual_2` / `arxiv_288`.
- **Diseño:** HR derivada de BVP/PPG de muñeca vs HR de ECG de pecho.
  Condiciones D1 (conflicto) y D2 (historial engañoso), con evidencia de
  fiabilidad emparejada vs barajada. Estimar `COTR`, `CIR`, `CRR`, `ESRM`,
  `UHR`.
- **Métricas:** las cinco anteriores + accuracy de decisión.
- **Coste:** medio (calcular HR de muñeca, generar conflictos, parser).
- **Aporta:** experimento novedoso y publicable; RQ2/RQ3.

### E5. Determinabilidad vs abstención

- **Refs:** `acl_anthology_231`.
- **Diseño:** tres condiciones por ventana:
  1. completa;
  2. determinable-incompleta (falta un canal, pero la evidencia restante
     basta para la etiqueta);
  3. indeterminable (falta la evidencia crítica).
  Instrucción con tres salidas: `baseline/stress/...` + `abstain`, y una
  variante que permita “no determinable”.
- **Métricas:** `over_abstention_rate` (en determinables),
  `premature_conclusion_rate` (en indeterminables), accuracy en
  determinables, abstención en indeterminables.
- **Coste:** bajo (enmascarar canales en la representación).
- **Aporta:** corrige el sesgo actual de que todos los NC son
  indeterminables; RQ2/H2b.

### E6. Evaluación a nivel de evento y claim fisiológico

- **Refs:** `arxiv_280`, `scopus_272`, `acl_anthology_288`.
- **Diseño:** definir eventos verificables por ventana (ráfaga SCR, HR
  elevada sostenida, movimiento, HRV baja) con umbrales y persistencia.
  Parsear los `claims` del LLM y clasificarlos en
  `supported / contradicted / unsupported / indeterminate` contra los
  eventos/features.
- **Métricas:** `event recall`, `duration recall`, `measurement coverage`,
  `hallucinated event rate`, precisión de claims.
- **Coste:** medio (parser + extractor de eventos).
- **Aporta:** operacionaliza RQ3, que hoy no se mide.

### E7. AFCE, longitud y marcadores epistémicos

- **Refs:** `acl_anthology_148`, `acl_anthology_245`, `acl_anthology_288`.
- **Diseño:** (a) variante AFCE en dos pasos (confianza sola → respuesta);
  (b) análisis de hedges/boosters por 100 palabras con longitud como
  covariable; (c) clasificación de fuerza del claim
  (absoluto/calificado/opinión).
- **Métricas:** ECE, Brier, sobreconfianza, pendientes estandarizadas,
  correlación con acierto.
- **Coste:** bajo.
- **Aporta:** H2a/H2b con control de confusores.

### E8. Consistencia como señal de fiabilidad (DeCC / self-consistency)

- **Refs:** `acl_anthology_56`, `acl_anthology_118`.
- **Diseño:** decisión directa vs decisión por subpreguntas
  (“¿HR elevada?”, “¿EDA elevada?”, “¿movimiento alto?”); varias semillas
  (config ya soporta `repetitions`/`seeds`). Consistencia → score de
  fiabilidad.
- **Métricas:** AUROC de la consistencia para predecir acierto; curvas de
  selective prediction; comparación con confidence verbalizada.
- **Coste:** medio (más llamadas; se puede hacer en un subconjunto).
- **Aporta:** RQ2 y análisis exploratorio de incertidumbre.

### E9. Conformal prediction para abstención

- **Refs:** `scopus_396`.
- **Diseño:** split por sujeto (calibración/test); aplicar conformal a la
  confianza o al modelo de referencia; reportar cobertura global y por clase.
- **Métricas:** cobertura empírica vs nominal, tamaño del conjunto de
  predicción, abstención.
- **Coste:** medio.
- **Aporta:** umbrales de abstención con garantía estadística; RQ2.

### E10. Trampas de hindsight / fuga temporal

- **Refs:** `arxiv_199`.
- **Diseño:** tres variantes del prompt: solo ventana; ventana + resumen de
  ventanas pasadas; ventana + resumen de ventanas futuras/desenlace.
  Medir cambio de decisión.
- **Métricas:** `hindsight trap rate`, `answer instability rate`,
  `hindsight bias rate`.
- **Coste:** medio.
- **Aporta:** validez; complementa la anti-leakage actual.

### E11. Sensibilidad a formulación y persona

- **Refs:** `manual_3`, `acl_anthology_148`.
- **Diseño:** factor prompt: base / paráfrasis / sin abstención / persona
  “experto” / pregunta reordenada. Medir varianza entre variantes.
- **Métricas:** accuracy, ECE, abstención, acuerdo entre variantes.
- **Coste:** bajo.
- **Aporta:** RQ4 y robustez; detecta conclusiones que dependen del prompt.

### E12. Ablación de features y representación factorial

- **Refs:** `scopus_111`, `ieee_88`, `pubmed_150`, `arxiv_51`.
- **Diseño:** (a) leave-one-modality-out y leave-one-feature-out;
  (b) 3 capas de `ieee_88`: input (features vs texto vs tabla) × usage
  (zero/few-shot) × output (etiqueta vs etiqueta+explicación);
  (c) condición “desviación respecto al baseline del sujeto”.
- **Métricas:** accuracy/macro-F1, AUROC por clase, contribución marginal.
- **Coste:** medio.
- **Aporta:** RQ4 y ataca el problema de 4 clases desde la evidencia.

### E13. Contrato formal de validación del pipeline

- **Refs:** `scopus_244`, `scopus_118`.
- **Diseño:** reglas comprobables (Nyquist, bandas/orden de filtro, longitud
  mínima de ventana para HRV, rangos, unidades, sampling, missingness) y
  reporte estructurado de violaciones.
- **Métricas:** nº de violaciones por sujeto/condición, correlación con SQI.
- **Coste:** medio.
- **Aporta:** rigor y reproducibilidad de Fase 1; base para auditoría.

### E14. Robustez a la hora del día

- **Refs:** `arxiv_365`.
- **Diseño:** aleatorizar/bloquear orden de condiciones; registrar
  timestamp; incluir hora/día como covariable; replicar condiciones clave
  en días distintos.
- **Métricas:** varianza explicada por hora/día; intervalos de confianza.
- **Coste:** bajo en análisis; alto en re-ejecución.
- **Aporta:** control de una amenaza real a la reproducibilidad.

---

## 7. Amenazas y limitaciones

1. **Sesgo del corpus.** Predominan papers clínicos y de salud mental; muy
   pocos usan WESAD. Las conclusiones son inspiración de diseño, no
   evidencia directa sobre nuestros datos.
2. **Duplicados.** `arxiv_288` = `manual_2`; `arxiv_276` ≈ `pubmed_168`.
   Contar papers únicos, no registros.
3. **Etiquetas de sección heurísticas.** Pueden estar mal asignadas
   (`manual_2`: cuerpo como `introduction`). Verificar el texto, no solo la
   etiqueta.
4. **Bloques gruesos.** `zoned` no son párrafos finos; para claims hay que
   trocear por frases.
5. **Sin títulos en `zoned`.** Usar metadatos del SLIP o el `pdf_file` para
   citar.
6. **`clean_text` no es el texto completo** y los campos `word_count`/
   `char_count` de `text` no son fiables.
7. **Confusor temporal.** `arxiv_365` muestra ~20 % de varianza por hora/día;
   nuestros runs largos deben controlarlo.
8. **LLM-juez.** Si se usa para claims, validar contra humanos y reportar
   acuerdo (idealmente con PPI, `acl_anthology_169`).
9. **Validez externa.** Los umbrales y reglas de otros papers no son
   transferibles sin verificación; el umbral SCR 0,05 µS sigue pendiente de
   validación con literatura/anotación.

---

## 8. Anexo: papers seleccionados

| ID | Título | Uso principal |
|---|---|---|
| `manual_5` | Telemetry degradation and sequential stability in LLM-integrated IoT decision systems | Degradación graduada, FlipRate |
| `scopus_355` | Mind the data: evaluating data quality sensitivity in medical LLMs | Taxonomía de anomalías |
| `manual_2` / `arxiv_288` | When Derived Measurements Mislead | DFOT, evidencia emparejada/barajada |
| `arxiv_25` | Mind the Missing (VITAL) | Missingness explícita |
| `scopus_198` | Multimodal LLMs for health grounded in individual-specific data | Serialización/missing |
| `scopus_244` | LLMs + formal planning for biosignal pipelines | Validación de pipeline |
| `scopus_118` | Multimodal LLMs in human-centered health | Alineación unidades/sampling |
| `ieee_59` | LLM-driven feature weighting for cross-device harmonization | Armonización |
| `scopus_111` | StressLLM | Ablación de features en WESAD |
| `pubmed_150` | Seamless stress monitoring with a time-series foundation model | Baseline personalizado/anomalías |
| `arxiv_51` | HRV-conditioned personalization via LLM | Personalización |
| `scopus_272` | LLM-assisted explainable stress recognition (PPG) | Reglas/umbrales interpretables |
| `ieee_88` | Empirical study of LLM human-state prediction | Factorial representación/uso/salida |
| `acl_anthology_231` | ClinDet-Bench | Determinabilidad vs abstención |
| `acl_anthology_56` | DeCC | Consistencia/selective prediction |
| `scopus_396` | Conformal prediction for LLM uncertainty | Cobertura/abstención |
| `acl_anthology_148` | Do LMs mirror human confidence? | AFCE/sobreconfianza |
| `acl_anthology_245` | Speech disfluencies and LLM confidence | Longitud/hedges/ECE |
| `acl_anthology_118` | Calibrating verbal uncertainty | VU vs SU/alucinación |
| `acl_anthology_288` | When LLMs disagree with human experts | Taxonomía de errores |
| `arxiv_280` | Benchmarking LLM summaries of clinical time series | Evaluación por eventos |
| `manual_3` | PhysioBench | Benchmark fisiológico |
| `manual_1` | WearableQA | Fichas de evidencia/consistencia |
| `acl_anthology_169` | Evaluación rigurosa de RAG | LLM-juez/PPI |
| `arxiv_199` | Hindsight bias in clinical temporal reasoning | Fuga temporal |
| `arxiv_162` | From classification to clinical insights | Modalidad faltante/alucinación |
| `arxiv_365` | Daily and weekly periodicity in LLM performance | Confusor temporal |
| `acl_anthology_65` | LENS | Límites de ingesta de series largas |

### Comandos de reproducción

```bash
# Búsqueda general agrupada por paper
python scripts/search_slip_docs.py "degradation quality anomaly" --papers --top 20

# Solo metodología y resultados
python scripts/search_slip_docs.py "calibration abstention confidence" \
  --section methodology,results --top 15

# Un paper concreto
python scripts/search_slip_docs.py "estimands reliability evidence" \
  --paper manual_2 --top 20

# Comparar con la vista text
python scripts/search_slip_docs.py "missing data imputation" --view text --top 10
```
