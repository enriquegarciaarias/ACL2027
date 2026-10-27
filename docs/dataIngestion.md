He revisado los tres documentos buscando trabajos que describan operaciones de tratamiento de señales fisiológicas antes de su interpretación por un LLM o VLM, así como trabajos que permitan decidir cómo representar esas señales para que el modelo produzca respuestas correctas, justificadas y compatibles con información textual.

La selección distingue entre el procesamiento clásico de la señal, su transformación en una representación interpretable por modelos de lenguaje y los mecanismos para controlar la fiabilidad de las respuestas. No todos los artículos son igual de pertinentes: algunos proponen directamente una arquitectura de procesamiento y otros aportan métodos parciales o marcos de evaluación.

## 1. Clasificación de los papers por tipo de tratamiento

Pertinencia estimada para el proyecto, valorada cualitativamente a partir del contenido de los documentos, no del Relevance Score de la base bibliográfica.

A. Filtrado, eliminación de artefactos y control de calidad

Prioridad muy alta

Tratamiento de ruido, detección de segmentos poco fiables, reconstrucción de tramos degradados y extracción de medidas fisiológicas. Es la etapa que evita que el LLM interprete como evidencia real una medición incorrecta.

B. Extracción de características y representación textual

Prioridad muy alta

Transformación de series temporales en características, estadísticas y descripciones fisiológicas estructuradas. Reduce el volumen de entrada y permite combinar mediciones con contexto textual.

C. Codificación de señales y alineamiento multimodal

Prioridad muy alta

Representación de señales mediante imágenes, parches temporales o embeddings, y alineamiento con texto. Es fundamental para comparar entradas numéricas, visuales y semánticas.

D. Segmentación temporal y diseño de pipelines

Prioridad alta

Ventanas, épocas, sincronización, orden de las operaciones y validación de las cadenas de procesamiento. Permite construir un procedimiento reproducible y comparable entre modalidades.

E. Validación de la evidencia y robustez del LLM

Prioridad alta

Evalúa qué ocurre cuando la señal contiene ruido, las características derivadas son incorrectas o se combinan varias señales. Permite estudiar si el modelo detecta una evidencia insuficiente o responde con una seguridad injustificada.

## 2. Selección principal de artículos

### A. Filtrado, reconstrucción y calidad de señal

1\. `manual_4` — An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation

Muy alta

Tratamiento identificado: filtrado paso alto de PPG con corte en 0,5 Hz; evaluación de calidad de señal; reconstrucción de segmentos con ruido o pérdidas; cálculo de frecuencia cardiaca mediante herramientas analíticas.

Pertinencia: es uno de los precedentes más directamente aplicables a una arquitectura en la que el LLM no interpreta sin más la señal cruda, sino que utiliza una cadena explícita de procesamiento y herramientas. Permite separar el cálculo fisiológico de la explicación lingüística.

Qué aprovechar: una etapa previa de control de calidad y cálculo determinista antes de generar la respuesta. También conviene estudiar qué información del procesamiento se entrega al LLM: señal original, señal corregida, indicadores de calidad y medidas derivadas.&#x20;

gemini_notebook_03.md



2\. `scopus_197` — EmoSavior: Depression Recognition and Intervention via Multimodal Physiological Signals

Muy alta

Tratamiento identificado: reconstrucción de señales fisiológicas ruidosas o incompletas mediante LLM, generación de características de orden superior y alineamiento multimodal mediante Cross-Modal Attention.

Pertinencia: aborda directamente dos dificultades de tu proyecto: la degradación de los sensores y la integración de modalidades con diferentes escalas temporales y respuestas fisiológicas.

Qué aprovechar: estudiar por separado la reconstrucción, la extracción de características y el alineamiento. La reconstrucción requiere una precaución importante: una señal reconstruida puede parecer plausible sin recuperar necesariamente la información real perdida. Por ello, el proyecto debería medir también el error introducido por la reconstrucción.&#x20;

gemini_notebook_01.md



3\. `scopus_272` — LLM-Assisted Explainable Daily Stress Recognition: Physiologically Grounded Thresholds

Alta

Tratamiento identificado: extracción de características HRV a partir de PPG, cálculo algorítmico de umbrales mediante optimización ROC–Youden y utilización del LLM para justificar las reglas a partir de literatura fisiológica.

Pertinencia: es especialmente interesante porque separa el análisis cuantitativo de la interpretación lingüística. El LLM no tiene que estimar directamente todas las propiedades de una onda: recibe características y reglas obtenidas mediante procedimientos explícitos.

Qué aprovechar: establecer qué cálculos deben realizarse fuera del LLM y qué resultados debe explicar. El artículo se centra en estrés y reglas de clasificación, por lo que no valida por sí mismo una solución general para múltiples sensores.&#x20;

gemini_notebook_03.md



### B. Extracción de características y conversión de señales a texto

4\. `scopus_341` — DiscBioSem-LM: A Framework for Anxiety Detection via Semantic Fusion of Multimodal Physiological Signals

Muy alta

Tratamiento identificado: conversión de características de señales fisiológicas multimodales en descripciones textuales enriquecidas con conocimiento del dominio; selección de características y ajuste discriminativo del modelo.

Pertinencia: constituye un precedente directo de la estrategia de representar las mediciones mediante lenguaje antes de introducirlas en un modelo de lenguaje. Trabaja con modalidades que incluyen EMG, EDA y frecuencia cardiaca.

Qué aprovechar: diseñar descripciones textuales basadas en valores calculados, tendencias y unidades explícitas. Conviene comprobar si la transformación textual conserva la información necesaria y si las descripciones inducen al modelo a inferir más de lo que permiten los datos.&#x20;

gemini_notebook_01.md



5\. `ieee_801` — Multimodal Sleep Stage Classification with LLM-Based Sleep Report

Alta

Tratamiento identificado: preprocesamiento de señales, ingeniería de características en los dominios temporal y frecuencial, selección mult criterio de características y clasificación de fases del sueño. Un LLM genera después interpretaciones de las salidas cuantitativas.

Pertinencia: es un ejemplo de arquitectura por etapas: la señal se procesa primero y el LLM transforma los resultados cuantitativos en explicaciones inteligibles.

Qué aprovechar: el diseño de una interfaz estructurada entre el módulo de análisis fisiológico y el generador de lenguaje. Su limitación es que el LLM explica resultados de un clasificador; no demuestra necesariamente que pueda razonar correctamente sobre la señal original.

6\. `scopus_111` — StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data

Alta

Tratamiento identificado: integración de biomarcadores procedentes de sensores vestibles en prompts y evaluación de estrategias de selección de biomarcadores para predicción del estrés.

Pertinencia: ayuda a decidir cómo condensar las mediciones en una entrada manejable por el LLM y qué variables merece la pena proporcionar.

Qué aprovechar: comparar una entrada con todas las variables frente a otra con características seleccionadas. No debe confundirse la selección de biomarcadores con la limpieza de la señal cruda: son etapas distintas del proceso.

### C. Codificación de la señal y alineamiento con texto

7\. `acl_anthology_65` — LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensor Streams

Muy alta

Tratamiento identificado: construcción de más de 100.000 pares sensor-texto a partir de respuestas EMA y entrenamiento de un codificador temporal por parches que proyecta señales crudas directamente al espacio de representación del LLM.

Pertinencia: probablemente uno de los trabajos más importantes para la parte de arquitectura de tu proyecto. Evita convertir cada muestra numérica en tokens de texto y permite conectar secuencias temporales con generación de narrativas.

Qué aprovechar: comparar el enfoque de características textuales con una representación temporal aprendida. Además, el trabajo ofrece un precedente para alinear datos fisiológicos y descripciones lingüísticas. No demuestra por sí solo que el modelo sea robusto ante degradaciones controladas de la señal.&#x20;

gemini_notebook_02.md



8\. `arxiv_64` — MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis

Muy alta

Tratamiento identificado: codificación de series temporales mediante patch embeddings, agrupando múltiples puntos —habitualmente 16— y transformándolos en embeddings continuos; inclusión de información textual específica del paciente en el prompt.

Pertinencia: aborda el problema de representar secuencias largas sin serializar cada punto como texto. Es directamente útil para comparar una entrada compacta con otra que preserve mejor la estructura temporal.

Qué aprovechar: el tamaño de parche, la precisión numérica y la fusión entre señal y contexto textual como factores experimentales. Hay que verificar cómo se adapta el codificador a cada modalidad y qué información temporal pierde la compresión.

9\. `scopus_69` — 2D GAF-Enhanced Multimodal Vision Language Model for Breathing Patterns Analysis

Alta

Tratamiento identificado: transformación de señales ECG en imágenes bidimensionales mediante Gramian Angular Fields (GAF) y ajuste de modelos visión-lenguaje para clasificar patrones respiratorios.

Pertinencia: ofrece una alternativa a las representaciones numéricas y textuales: codificar la estructura temporal como imagen para que la procese un VLM.

Qué aprovechar: comparar la representación visual con los valores numéricos y las características estadísticas. Es importante comprobar si la transformación conserva patrones relevantes y si la resolución o el escalado modifican la respuesta del modelo.

10\. `ieee_392` — NarratingSleep: A Language Model-Based Framework for Sleep Staging

Alta

Tratamiento identificado: creación de un vocabulario temporal (`time series vocabulary`) para conectar medidas continuas con tokens discretos y uso de un clasificador temporal para procesar múltiples pasos.

Pertinencia: aporta una alternativa intermedia entre serializar valores en texto y emplear imágenes: aprender una representación discreta que preserve información temporal.

Qué aprovechar: incluir una representación temporal aprendida en la comparación experimental, especialmente si el objetivo es analizar secuencias y no solo estadísticas agregadas.

### D. Segmentación y construcción de pipelines reproducibles

11\. `scopus_244` — Integrating Large Language Models with Formal Planning to Automate the Design and Validation of Biosignal Processing Pipelines

Muy alta

Tratamiento identificado: conversión de instrucciones en lenguaje natural a operaciones de procesamiento de bios señales y validación formal del flujo mediante planificación HTN. Incluye operaciones como filtrado, segmentación y comprobación del orden correcto de los pasos.

Pertinencia: es relevante para especificar un pipeline reproducible y evitar que las decisiones de preprocesamiento se tomen de forma improvisada. También permite separar la interpretación del lenguaje de la validación técnica.

Qué aprovechar: definir una cadena modular con operaciones, parámetros, precondiciones y comprobaciones explícitas. No es principalmente un método de interpretación fisiológica por LLM, sino una contribución a la ingeniería del proceso de tratamiento.&#x20;

gemini_notebook_01.md



12\. `pubmed_150` — Seamless Monitoring of Stress Levels Leveraging a Foundational Model for Time Series

Media-alta

Tratamiento identificado: empleo de un modelo fundacional para series temporales (UniTS) en monitorización del estrés con señales vestibles, teniendo en cuenta la degradación de calidad propia de las mediciones de uso cotidiano.

Pertinencia: aporta contexto sobre el uso de modelos temporales para datos fisiológicos de larga duración y sobre la diferencia entre sensores clínicos y vestibles.

Qué aprovechar: utilizar modelos temporales especializados como referencia frente a las representaciones que recibe un LLM. El interés principal está en el modelado de series temporales, no necesariamente en convertirlas a texto.

### E. Fiabilidad de la evidencia y evaluación de respuestas

Estos trabajos no son todos métodos de preprocesamiento, pero son necesarios para que el estudio no se limite a demostrar que una representación es cómoda o que mejora una métrica de clasificación.

13\. `manual_2` — When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Physiological Evidence

Muy alta

Tratamiento o problema estudiado: fiabilidad de medidas derivadas de PPG, contraste con información ECG de referencia y evaluación del exceso de confianza del LLM cuando la evidencia de entrada es incorrecta o poco fiable.

Pertinencia: es especialmente relevante para tu pregunta sobre la exactitud de la información generada. Un LLM puede recibir una medida calculada por otro modelo y confiar en ella aunque el cálculo esté equivocado.

Qué aprovechar: evaluar no solo si la respuesta es correcta, sino también si cambia de forma apropiada al degradar la evidencia, si identifica insuficiencia de datos y si expresa incertidumbre proporcional al error de la medición.&#x20;

gemini_notebook_01.md



14\. `manual_1` — WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data

Muy alta

Tratamiento identificado: evaluación de razonamiento sobre registros longitudinales reales de dispositivos vestibles, incluidas preguntas sobre señales individuales y razonamiento entre señales. Los datos conservan ruido de dispositivo y variabilidad entre personas.

Pertinencia: permite evaluar si una representación que funciona con una señal aislada sigue funcionando al combinar señales y contexto. El trabajo encuentra dificultades en el razonamiento directo a partir de mediciones crudas y en el razonamiento entre señales.

Qué aprovechar: una matriz de evaluación que distinga interpretación de una sola señal, cálculo temporal y combinación de señales. Su ventaja es el realismo; su limitación para tu experimento es que el ruido no está controlado necesariamente por niveles de severidad conocidos.&#x20;

gemini_notebook_03.md



15\. `manual_3` — PhysioBench: A Unified Benchmark for Physiological Signal Question Answering

Muy alta

Tratamiento identificado: armonización de anotaciones de 22 conjuntos de datos, segmentación según el alcance temporal de las anotaciones y conversión de tareas fisiológicas en pares de pregunta-respuesta sobre segmentos de señal.

Pertinencia: es útil para definir tareas comparables y evaluar si el modelo interpreta correctamente diferentes modalidades y preguntas sobre ellas.

Qué aprovechar: preguntas con respuestas verificables, ventanas temporales alineadas y un conjunto de tareas que incluya interpretación de patrones, detección de eventos y caracterización fisiológica. No debe interpretarse como una receta universal de limpieza de señales: su contribución principal es el benchmark y la evaluación.&#x20;

gemini_notebook_03.md



16\. `scopus_293` — A Survey on Large Language Models for Medical Time Series

\*\*badge>Referencia de síntesis

Tratamiento identificado: revisión de codificación de características, arquitecturas de modelos temporales, estrategias de prompting, fusión multimodal y tareas de generación o reconstrucción de series.

Pertinencia: es útil para ampliar la búsqueda bibliográfica y evitar que el estado del arte se limite a aplicaciones de salud mental.

Qué aprovechar: utilizar sus categorías para organizar los métodos y rastrear los trabajos primarios. Al ser una revisión, sus afirmaciones sobre métodos concretos deben comprobarse en los artículos originales.

## 3. Comparación: ¿qué representación conviene entregar al LLM?

Los artículos seleccionados sugieren cuatro familias de representación. No existe base suficiente en los documentos para declarar una ganadora universal: la elección depende de la modalidad, de la tarea y de qué información temporal se necesita preservar.

| Representación                        | Ejemplos de la selección                   | Ventaja principal                                                                          | Riesgo principal                                                               |
| ------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Valores y características numéricas   | `scopus_272`, `scopus_111`                 | Entrada compacta, auditable y relativamente fácil de verificar                             | Se pierden patrones de la onda si las características son insuficientes        |
| Descripciones textuales estructuradas | `scopus_341`                               | Fácil integración con contexto, instrucciones y texto libre                                | La descripción puede ocultar detalles o introducir interpretaciones prematuras |
| Imágenes de señales                   | `scopus_69`                                | Conserva ciertos patrones visuales y permite usar VLM                                      | Sensibilidad al escalado, resolución y transformación de la señal              |
| Embeddings o parches temporales       | `acl_anthology_65`, `arxiv_64`, `ieee_392` | Reduce el coste de representar secuencias largas y puede conservar dependencias temporales | Menor transparencia y dependencia de entrenamiento y arquitectura              |

La comparación debería controlar también la calidad de la señal de origen. Si se entrega una señal limpia a un modelo y una señal degradada a otro, no se puede atribuir la diferencia exclusivamente a la representación.

## 4. Pipeline que recomiendo investigar

La síntesis de los trabajos apunta a un pipeline experimental con etapas separadas y verificables:

### 1. Señal fisiológica original

ECG, PPG/HR, EDA, EEG, respiración u otras modalidades

### 2. Preprocesamiento específico por modalidad

Filtrado · detección de artefactos · segmentación · sincronización · unidades y frecuencia de muestreo

Registrar parámetros, muestras descartadas y transformaciones realizadas

### 3. Evaluación de calidad y extracción de evidencia

Indicadores de calidad · características temporales y frecuenciales · estadísticos · medidas derivadas

Conservar la distinción entre medida observada, medida estimada y medida reconstruida

### 4. Generación de representaciones alternativas

A. Valores / estadísticas

B. Texto estructurado

C. Imagen / gráfico

D. Embeddings temporales

### 5. Integración con información textual

Pregunta · contexto de la persona o sesión · información temporal · datos de calidad · evidencia textual pertinente

### 6. Evaluación del LLM / VLM

Corrección de afirmaciones · errores numéricos · inferencias no sustentadas · incertidumbre · abstención

La idea metodológica clave es no fusionar todas las etapas en una única transformación opaca. Si la respuesta es incorrecta, necesitas poder determinar si el error se originó en la adquisición, el filtrado, el cálculo de características, la representación, la integración con texto o el razonamiento del LLM.

## 5. Cómo convertir la selección en un estudio experimental sólido

Propongo organizar los experimentos en tres ejes.

### Eje 1. Calidad de señal

Partir de segmentos de referencia y generar degradaciones controladas, por ejemplo:

- Ruido añadido y artefactos.
- Pérdida de muestras o segmentos.
- Reducción de frecuencia de muestreo.
- Desplazamiento temporal entre modalidades.
- Errores en las medidas derivadas, diferenciándolos de la degradación de la señal cruda.

Debe conocerse la severidad introducida y conservarse la señal original como referencia.

### Eje 2. Representación

Comparar, con las mismas ventanas y condiciones de calidad:

1. Valores numéricos serializados.
2. Resumen estadístico y características fisiológicas.
3. Descripción textual derivada de esas características.
4. Representación gráfica para VLM.
5. Codificación temporal mediante parches, cuando sea viable.

En cada condición hay que mantener constante el contexto textual, el modelo y la pregunta. De lo contrario, no podrás aislar el efecto de la representación.

### Eje 3. Calidad de la respuesta

No basta con medir exactitud de clasificación. Para tu objetivo —obtener información acertada de señales individuales y combinadas con texto— mediría:

- Corrección factual: si cada afirmación está respaldada por la señal o por el texto.
- Precisión numérica: error absoluto o relativo respecto de medidas de referencia, con tolerancias definidas por variable.
- Interpretación individual: capacidad de describir cada modalidad sin confundirla con otras.
- Razonamiento multimodal: corrección al integrar señales y contexto textual, comparada con la evaluación individual.
- Sensibilidad a la degradación: si la calidad de la respuesta empeora de forma consistente al aumentar el deterioro.
- Calibración lingüística: si la seguridad expresada por el modelo se corresponde con la corrección real.
- Abstención: si reconoce cuándo la evidencia no permite responder.

El precedente de `manual_2` es especialmente importante aquí: una respuesta puede ser incorrecta y, aun así, expresarse con mucha seguridad. `manual_1` y `manual_3` aportan marcos complementarios para evaluar razonamiento sobre registros reales y preguntas verificables.&#x20;

gemini_notebook_01.md

gemini_notebook_03.md

gemini_notebook_03.md



## 6. Prioridad de lectura

Si el objetivo es avanzar con rapidez hacia un diseño experimental defendible, leería los trabajos en este orden:

| Orden | ID                      | Motivo                                                                               |
| ----- | ----------------------- | ------------------------------------------------------------------------------------ |
| 1     | `manual_4`              | Pipeline explícito de PPG, calidad, reconstrucción y análisis mediante herramientas. |
| 2     | `scopus_244`            | Diseño y validación de pipelines reproducibles.                                      |
| 3     | `scopus_341`            | Conversión de características fisiológicas en descripciones textuales.               |
| 4     | `acl_anthology_65`      | Integración de series temporales directamente en el espacio del LLM.                 |
| 5     | `arxiv_64`              | Codificación temporal mediante parches e integración de contexto textual.            |
| 6     | `scopus_197`            | Reconstrucción y alineamiento de señales multimodales.                               |
| 7     | `scopus_69`             | Representación gráfica mediante GAF para modelos visión-lenguaje.                    |
| 8     | `scopus_272`            | Separación entre cálculo cuantitativo y explicación lingüística.                     |
| 9     | `manual_2`              | Evaluación del exceso de confianza ante medidas fisiológicas erróneas.               |
| 10    | `manual_1` y `manual_3` | Diseño de evaluaciones de razonamiento fisiológico individual y multimodal.          |

## Conclusión

Los documentos permiten identificar tres líneas complementarias: limpiar y verificar la señal antes de interpretarla; representar la información fisiológica de forma compacta sin perder la estructura relevante; y comprobar que el LLM adapta sus afirmaciones a la calidad de la evidencia.

Para tu proyecto, evitaría plantear el preprocesamiento como una simple fase técnica previa. Conviene tratarlo como una variable experimental: comparar representaciones y cadenas de tratamiento bajo degradaciones controladas, y medir no solo la exactitud de las respuestas, sino también su fidelidad a la evidencia y su calibración lingüística.

Esta selección está limitada a los tres documentos proporcionados. Los títulos se presentan según los registros bibliográficos incluidos en ellos; antes de cerrar el estado del arte conviene comprobar las versiones originales, especialmente en los trabajos de prepublicación y en los artículos cuyo registro contiene títulos abreviados.