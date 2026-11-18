# Metodología

He revisado los tres documentos (`gemini_notebook_01`, `gemini_notebook_02` y `gemini_notebook_03`), centrándome en los trabajos que evalúan directamente las respuestas de los LLM: su corrección, confianza, incertidumbre, consistencia, abstención y errores.

## **1. Conclusión general**

**Sí existen metodologías replicables para analizar respuestas de LLM en los artículos incluidos, pero no hay una metodología común y consolidada que cubra todas las dimensiones que estamos estudiando.**

He identificado tres niveles:

1. **Metodologías experimentales bastante replicables:** fijan las condiciones de entrada, los modelos, las perturbaciones, las respuestas esperadas y las métricas.
2. **Metodologías replicables para dimensiones concretas:** analizan confianza, incertidumbre lingüística, alucinaciones o errores, pero no evalúan conjuntamente el comportamiento completo del modelo.
3. **Evaluaciones poco estandarizadas:** presentan resultados de accuracy, confianza o análisis cualitativos sin describir suficientemente cómo se han obtenido o validado las medidas.

La distinción importante es que una metodología puede ser replicable para evaluar una dimensión concreta sin constituir un protocolo completo de análisis de respuestas.

## **2. Papers que ofrecen metodologías más aprovechables**

Estos son los trabajos que considero más relevantes para identificar procedimientos que podamos reutilizar.

### **A. Degradación de datos y estabilidad de las respuestas**

`manual_5`**&#x20;— Telemetry Degradation and Sequential Stability in LLM-Integrated IoT Decision Systems**

Alta replicabilidad experimental

Este trabajo presenta uno de los protocolos más próximos a nuestro experimento de WESAD.

**Qué hace:**

- Introduce degradación controlada de telemetría mediante dropout y downsampling.
- Define diferentes niveles de degradación.
- Mantiene condiciones experimentales comparables.
- Ejecuta diez réplicas por condición.
- Analiza las decisiones generadas por el LLM a lo largo del tiempo.

**Cómo analiza las respuestas:**

No se limita a medir si una respuesta es correcta. Evalúa cómo cambia el comportamiento del sistema mediante:

- `FlipRate`: frecuencia con la que cambia la decisión.
- `EscalationRate`: tendencia a incrementar la intensidad de las decisiones.
- `ViolationRate`: consecuencias de seguridad derivadas de las secuencias de decisiones.

**Qué podemos aprovechar:** la evaluación de estabilidad, la repetición de ejecuciones y la comparación sistemática entre niveles de degradación.

**Limitación:** su objeto de análisis son decisiones estructuradas dentro de un sistema de entrenamiento de tenis. No desarrolla una metodología de verificación semántica de afirmaciones, certeza lingüística o abstención clínica.&#x20;

gemini_notebook_03(1).md

gemini_notebook_03(1).md



`manual_2`**&#x20;/&#x20;**`arxiv_288`**&#x20;— When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence**

Alta replicabilidad de los estimandos

Es uno de los trabajos más relevantes para nuestra investigación.

**Qué hace:**

Define cinco estimandos para medir el comportamiento del LLM ante medidas fisiológicas derivadas cuya fiabilidad puede ser diferente:

| Estimando | Qué evalúa                                                             |
| --------- | ---------------------------------------------------------------------- |
| COTR      | Aceptación de una medida derivada en conflicto con la referencia.      |
| CIR       | Errores inducidos por el contexto.                                     |
| CRR       | Reparación correcta de errores.                                        |
| ESRM      | Si la reparación depende de que la evidencia sea realmente pertinente. |
| UHR       | Verificaciones innecesarias en casos de alta fiabilidad.               |

La evaluación utiliza registros PPG–ECG y un test bloqueado por paciente de 187 sujetos.

**Qué podemos aprovechar:** la definición previa de situaciones experimentales, la construcción de casos con evidencia contradictoria, la comparación entre condiciones y la utilización de estimandos específicos para cada comportamiento.

**Limitación:** mide principalmente aceptación, rechazo y reparación de información derivada. No analiza de manera integral la estructura lingüística de las respuestas ni la veracidad de cada afirmación generada.&#x20;

gemini_notebook_03(1).md



### **B. Análisis de confianza e incertidumbre expresada**

`Verbal Uncertainty Feature / Mechanistic Uncertainty Calibration`

Metodología cuantitativa replicable

Este trabajo es especialmente interesante para nuestra evaluación de certeza lingüística.

Separa dos conceptos:

- **Semantic Uncertainty (SU):** incertidumbre sobre el contenido semántico de la respuesta.
- **Verbal Uncertainty (VU):** grado de incertidumbre que el modelo transmite mediante su lenguaje.

**Procedimiento:**

1. Genera varias respuestas para cada pregunta.
2. Agrupa las respuestas semánticamente equivalentes.
3. Calcula la entropía semántica.
4. Utiliza un segundo LLM como evaluador para asignar una puntuación de incertidumbre lingüística.
5. Calcula la puntuación media de VU.
6. Contrasta ambas medidas para detectar desajustes entre incertidumbre semántica y lingüística.

Además, contrasta las puntuaciones del evaluador con similitudes de embeddings respecto a expresiones prototípicas de incertidumbre.

**Qué podemos aprovechar:** es una metodología explícita para analizar el lenguaje de las respuestas, en lugar de limitarse a la confianza numérica que declara el modelo.

**Limitación:** el evaluador también es un LLM. La validación mediante embeddings no sustituye por completo a una evaluación humana independiente. Además, la investigación se centra en preguntas y respuestas breves, no en evidencia fisiológica.&#x20;

gemini_notebook_03(1).md

gemini_notebook_03(1).md



`gemini_notebook_01`**&#x20;— estudio sobre anomalías en datos médicos y comportamiento de LLM**

Replicabilidad parcial

El artículo analiza el efecto de tres tipos de anomalías en datos médicos: ausencia de información, problemas de conformidad y problemas de plausibilidad.

**Qué hace:**

- Compara el rendimiento diagnóstico bajo diferentes anomalías.
- Evalúa la capacidad del LLM para detectar anomalías.
- Prueba CoT, Self-Refine, Self-Consistency y T3.
- Realiza ejecuciones repetidas para estudiar la confianza.
- Observa que las puntuaciones de confianza superan el 90 % incluso ante datos anómalos.

**Qué podemos aprovechar:** la comparación entre tipos de degradación y la evaluación de la confianza declarada ante evidencia problemática.

**Limitación:** el análisis de confianza es más limitado que el de rendimiento. No presenta una taxonomía suficientemente completa de las respuestas ni una evaluación integral de sus afirmaciones.&#x20;

gemini_notebook_01(1).md



### **C. Verificación de respuestas y análisis de errores**

`acl_anthology_73`**&#x20;— Gladiator at MEDIQA-SYNUR 2026**

Alta replicabilidad del análisis estructurado

Este trabajo presenta un procedimiento muy claro para evaluar extracciones realizadas por un LLM.

**Qué hace:**

- Utiliza un esquema de 193 conceptos clínicos.
- Define tipos de salida y valores válidos.
- Utiliza un prompt documentado y ejemplos few-shot.
- Aplica validaciones y filtros posteriores.
- Compara las extracciones con anotaciones de referencia.
- Clasifica los errores en falsos positivos, omisiones y valores incorrectos.

La evaluación se realiza mediante precisión, recall y F1, con un criterio de correspondencia exacta entre identificador y valor.

**Qué podemos aprovechar:** la clasificación de errores y la separación entre extracción incorrecta, información omitida e información no respaldada.

**Limitación:** se trata de extracción estructurada. No evalúa directamente la calidad de respuestas abiertas, la confianza lingüística ni la adecuación de la abstención.&#x20;

gemini_notebook_03(1).md



**Estudio de clasificación de publicaciones sobre desinformación nutricional (**`gemini_notebook_03`**)**

Buena replicabilidad del análisis de errores

Este trabajo utiliza:

- Un prompt común para los modelos evaluados.
- Etiquetas contrastadas con consenso experto.
- Parsing determinista mediante expresiones regulares.
- Accuracy, precision, recall, macro-F1 y Cohen's κ.
- Fleiss' κ para acuerdo entre modelos.
- Una taxonomía jerárquica de errores.

La taxonomía diferencia dirección del error, mecanismo de fallo y factores contribuyentes.

**Qué podemos aprovechar:** la definición de categorías mutuamente excluyentes, el manual de codificación y la separación entre el error de interpretación y el error de razonamiento.

**Limitación:** la tarea es clasificación de publicaciones, no interpretación de señales fisiológicas.&#x20;

gemini_notebook_03(1).md



### **D. Benchmarks con evaluación automatizada**

`manual_3`**&#x20;— PhysioBench: A Unified Benchmark for Physiological Signal Question Answering**

Alta replicabilidad del scoring

Este trabajo ofrece un procedimiento particularmente interesante para evaluar respuestas sobre señales fisiológicas:

- Convierte anotaciones de 22 datasets en preguntas y respuestas mediante plantillas.
- Define un conjunto de respuestas candidatas por tarea.
- Comprueba que cada instancia tenga una única respuesta correcta.
- Compara la respuesta semántica predicha con la referencia.
- Considera incorrectas las respuestas que no pueden parsearse o mapearse a una categoría válida.
- Informa accuracy y macro-F1.

Su fortaleza es la normalización del proceso de evaluación y la trazabilidad entre pregunta, señal y anotación original.

Sin embargo, es un benchmark de preguntas cerradas. No permite evaluar de forma suficiente la calidad de las explicaciones, las afirmaciones adicionales o la incertidumbre expresada en respuestas abiertas.&#x20;

gemini_notebook_03(1).md

gemini_notebook_03(1).md



## **3. Comparación de las metodologías**

La siguiente tabla resume qué dimensiones permiten analizar los trabajos identificados. La valoración se refiere a la metodología descrita en los documentos, no a una reproducción independiente de sus experimentos.

| Paper                      | Corrección | Confianza numérica | Certeza lingüística | Verificación de claims | Abstención | Análisis de errores |
| -------------------------- | ---------- | ------------------ | ------------------- | ---------------------- | ---------- | ------------------- |
| `manual_5`                 | Parcial    | —                  | —                   | —                      | —          | Sí                  |
| `manual_2`                 | Sí         | Parcial            | —                   | Parcial                | Parcial    | Sí                  |
| Verbal Uncertainty / MUC   | Sí         | Indirecta          | Sí                  | Sí                     | Sí         | Parcial             |
| Anomalías médicas          | Sí         | Sí                 | —                   | —                      | —          | Parcial             |
| Gladiator MEDIQA           | Sí         | —                  | —                   | Parcial                | —          | Sí                  |
| Desinformación nutricional | Sí         | Sí                 | Parcial             | —                      | —          | Sí                  |
| PhysioBench                | Sí         | —                  | —                   | —                      | —          | Parcial             |

*Parcial significa que el trabajo aporta algún procedimiento relacionado, pero no una evaluación completa y equivalente a la dimensión que estamos considerando.*

## **4. ¿Qué metodología se puede considerar realmente replicable?**

A partir de los trabajos revisados, propongo distinguir cuatro requisitos.

| Requisito                                 | Qué debe quedar documentado                                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reproducibilidad de la generación         | Modelo y versión, prompt completo, parámetros de inferencia, número de ejecuciones y formato de salida.                                                 |
| Reproducibilidad de la evaluación         | Referencia de verdad, reglas de comparación, definición de métricas y procedimiento de parsing.                                                         |
| Reproducibilidad del análisis lingüístico | Taxonomía, manual de codificación, criterios para identificar certeza, hedges, boosters y abstenciones.                                                 |
| Reproducibilidad estadística              | Unidad de análisis, tratamiento de repeticiones, intervalos de confianza, pruebas estadísticas y corrección por comparaciones múltiples cuando proceda. |

Un punto adicional: **que un artículo publique métricas no significa que su análisis de respuestas sea replicable**. Para poder repetirlo necesitamos conocer cómo se transforma cada respuesta textual en una observación evaluable.

Por ejemplo, si un artículo afirma que la certeza aumenta ante una señal degradada, debería permitir reconstruir cómo se ha calculado esa certeza a partir de las respuestas originales.

## **5. Qué implicaciones tiene para nuestro experimento WESAD–PhysioNet**

Aquí está la cuestión más importante.

En nuestros experimentos ya estamos aplicando varios elementos presentes en los trabajos anteriores:

- Condiciones de evidencia controladas en WESAD.
- Degradación natural en PhysioNet.
- Referencias de clasificación conocidas.
- Evaluación de corrección.
- Medición de confianza numérica.
- Análisis de certeza lingüística.
- Identificación de abstenciones.
- Verificación de afirmaciones.
- Indicadores externos de calidad.

Sin embargo, los trabajos revisados muestran que debemos consolidar especialmente el procedimiento de análisis de las respuestas.

**Mi recomendación es construir un protocolo común de evaluación que se aplique a todas las respuestas generadas, independientemente del dataset, modelo o condición de degradación.**

Lo organizaría así:

**Protocolo común de análisis de respuestas LLM**

1. **1. Registro inalterado de la respuesta**

   Conservar prompt, modelo, versión, parámetros, condición experimental y respuesta original. No modificar el texto antes del análisis.
2. **2. Evaluación de corrección**

   Comparar la respuesta con el ground truth específico de la tarea. Distinguir clasificación correcta, incorrecta e indeterminada.
3. **3. Descomposición en afirmaciones**

   Separar las afirmaciones verificables de la respuesta. Registrar qué evidencia respalda cada una.
4. **4. Verificación de afirmaciones**

   Clasificar cada claim como respaldado, contradicho o no verificable con la evidencia disponible. Establecer tolerancias para las afirmaciones cuantitativas.
5. **5. Análisis de incertidumbre**

   Medir confianza numérica y certeza lingüística por separado. Definir categorías y reglas explícitas para hedges, boosters y expresiones de duda.
6. **6. Análisis de abstención**

   Distinguir entre abstención apropiada, abstención inapropiada y ausencia de abstención cuando la evidencia es insuficiente.
7. **7. Análisis de errores**

   Aplicar una taxonomía predefinida para clasificar los errores de interpretación, inferencia, contradicción y exceso de afirmación.
8. **8. Validación del propio evaluador**

   Contrastar una muestra de respuestas y claims mediante anotación humana independiente, calcular acuerdo interanotador y resolver discrepancias.

Hay una cuestión metodológica que considero fundamental: no debemos utilizar el mismo criterio de verdad para todas las afirmaciones. Una respuesta puede ser incorrecta respecto a la clasificación de un registro, pero contener una afirmación fisiológica verdadera; o puede acertar la clase y añadir una explicación que no está respaldada por los datos.

Por eso, el análisis de corrección de la respuesta completa y la verificación de claims deben ser dos evaluaciones independientes.

## **6. Mi valoración final**

Los documentos contienen precedentes suficientes para construir una metodología sólida y replicable. No necesitamos inventar desde cero el análisis de las respuestas de los LLM.

Las aportaciones más útiles son complementarias:

- `manual_5` aporta el diseño de degradación y estabilidad.
- `manual_2` aporta la evaluación de aceptación y reparación de evidencia fisiológica.
- Verbal Uncertainty aporta la medición de incertidumbre lingüística y semántica.
- Gladiator y el estudio de desinformación aportan taxonomías y protocolos de análisis de errores.
- PhysioBench aporta normalización de respuestas y evaluación semántica automatizada.

**Pero ninguno de estos trabajos, por sí solo, proporciona el protocolo completo que estamos desarrollando.** La oportunidad de nuestra investigación está precisamente en integrar esas dimensiones bajo un diseño experimental común, donde podamos observar no solo si el LLM acierta, sino también si sus afirmaciones, su certeza y su decisión de responder son coherentes con la evidencia fisiológica disponible.

Una última observación: antes de presentar el protocolo como validado, necesitamos comprobar su reproducibilidad entre evaluadores humanos. Especialmente en la clasificación de claims y certeza lingüística. Si esas categorías dependen excesivamente del criterio de quien las interpreta, las métricas resultantes no serán suficientemente robustas.

Mi siguiente paso sería convertir esta propuesta en un **manual de anotación operacional**, con categorías mutuamente excluyentes, ejemplos positivos y negativos, reglas de decisión y métricas de acuerdo interanotador. Así podríamos aplicarlo retrospectivamente a las respuestas de WESAD y PhysioNet y comprobar si los resultados se mantienen bajo un procedimiento uniforme.

# Health-LLM y BioMistral

Sí. De hecho, **Health-LLM y BioMistral son referencias importantes para vuestro experimento**, pero conviene distinguir qué aporta cada uno. Hay una diferencia fundamental entre un modelo especializado en conocimiento biomédico y un modelo adaptado específicamente a interpretar datos de wearables.

## **1. Health-LLM: probablemente la referencia más próxima a vuestro planteamiento**

**Health-LLM** (Kim et al., CHIL 2024) no es un modelo único, sino un framework para evaluar y adaptar LLM a tareas de predicción de salud utilizando datos de sensores y contexto personal.

Su trabajo contempla:

- 12 LLM, entre ellos BioMistral, MedAlpaca, BioMedGPT, Llama 2, GPT-4 y Gemini-Pro.
- Cuatro conjuntos de datos: PMData, LifeSnaps, GLOBEM y AW_FB.
- Diez tareas de predicción relacionadas con salud mental, actividad, metabolismo y sueño.
- Zero-shot, few-shot, Chain-of-Thought, self-consistency y fine-tuning.
- Incorporación de contexto personal, conocimiento sanitario e información temporal.

Su modelo adaptado, **HealthAlpaca**, obtuvo el mejor rendimiento en 8 de las 10 tareas.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://orsonxu.com\&sz=32)

orsonxu.com+1



Fuentes originales

- [Health-LLM — proyecto y resultados](https://health-llm.github.io/?utm_source=chatgpt.com)
- [Artículo original](https://arxiv.org/abs/2401.06866?utm_source=chatgpt.com)

**¿Qué relación tiene con vuestro trabajo?**

Es una referencia muy pertinente porque aborda precisamente el problema de hacer que un LLM interprete información fisiológica y conductual presentada como datos numéricos y temporales.

Pero hay una diferencia metodológica importante:

| Health-LLM                                              | Vuestro experimento                                   |
| ------------------------------------------------------- | ----------------------------------------------------- |
| Predice variables de salud y bienestar                  | Evalúa la fidelidad de las respuestas a la evidencia  |
| Utiliza datos de wearables agregados                    | Introduce degradaciones controladas y conocidas       |
| Evalúa principalmente rendimiento predictivo            | Evalúa corrección, claims, certeza y abstención       |
| Mejora la interpretación mediante contexto y adaptación | Estudia si el modelo reconoce la pérdida de evidencia |

Por tanto, Health-LLM es un precedente directo para la entrada de datos fisiológicos, pero no resuelve vuestra pregunta central sobre la sensibilidad de las respuestas a la degradación.

## **2. BioMistral: modelo biomédico, pero no especializado en señales fisiológicas**

[BioMistral-7B: Mistral 7B Based LLM for Medical Domains](https://images.openai.com/static-rsc-4/iSxtd_NLxcA38E8VfllCTOnHoJA8lDp-jmOIujYuSLSVv7hywhv9X8_mWQ5u8uc3pr8p73karww08TuoTVRIpF78QvPlHX66z-2egZX9bPATEvJOTBV7FCAh3t9Xp1Dmhc2azl3IuSOXhU9Cxqi9-wtJWGdbUlJUhNY-J7cACNw?purpose=inline)

**BioMistral**

Labrak et al. (2024). Modelo de 7.000 millones de parámetros basado en Mistral 7B Instruct, con preentrenamiento adicional sobre literatura de PubMed Central.

[Artículo original](https://arxiv.org/abs/2402.10373?utm_source=chatgpt.com)

Su especialización es el conocimiento biomédico y la comprensión del lenguaje médico. El artículo evalúa diez tareas de preguntas y respuestas médicas, además de aspectos de veracidad y calibración.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arxiv.org



Pero no debemos confundir esa especialización con capacidad para interpretar directamente señales fisiológicas.

BioMistral no incorpora por sí mismo un codificador especializado para ECG, EDA, respiración o EEG. Puede razonar sobre características fisiológicas que se le proporcionen como texto, pero eso no demuestra que interprete adecuadamente las dinámicas temporales de las señales.

## **3. ¿Aparecen en los documentos que analizamos?**

Sí, con distinto grado de presencia.

| Referencia      | Presencia en el conjunto                                                   | Relevancia |
| --------------- | -------------------------------------------------------------------------- | ---------- |
| Health-LLM      | Citado como framework de referencia y estrategia de prompting en StressLLM | Alta       |
| BioMistral      | Evaluado dentro de Health-LLM y otros trabajos                             | Alta       |
| BioMistral-DARE | Evaluado en el estudio StressLLM                                           | Alta       |
| HealthAlpaca    | Modelo adaptado dentro de Health-LLM                                       | Muy alta   |

Hay dos detalles que merece la pena destacar:

- **StressLLM** utiliza la estrategia de prompting de Health-LLM como punto de partida para sus experimentos con PMData y WESAD. También evalúa BioMistral-DARE.&#x20;

  training_materials(1).md

- En el material de Health-LLM aparecen BioMistral y otros modelos biomédicos como parte de la comparación. Sin embargo, su página oficial indica que BioMistral fue excluido de algunos experimentos de mejora por prioridades de integración.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://orsonxu.com\&sz=32)

  orsonxu.com+1



## **4. Mi recomendación para vuestro experimento**

No limitaría la comparación a Qwen2.5 y Qwen3. Incorporaría al menos un modelo biomédico y, si los recursos lo permiten, uno adaptado específicamente a datos de wearables.

Mi propuesta sería:

| Modelo        | Función experimental                            | Prioridad |
| ------------- | ----------------------------------------------- | --------- |
| Qwen2.5:7B    | Referencia generalista                          | Mantener  |
| Qwen3:14B     | Generalista de mayor capacidad                  | Mantener  |
| BioMistral-7B | Efecto de la especialización biomédica          | Alta      |
| HealthAlpaca  | Efecto de adaptación a predicción con wearables | Alta      |



Recomendación

**HealthAlpaca sería especialmente interesante como comparación adicional**, porque permite separar dos cuestiones que ahora mismo están mezcladas:



1. ¿Mejora el conocimiento biomédico la interpretación de señales degradadas?
2. ¿Mejora la adaptación específica a datos de wearables la capacidad de reconocer cuándo la evidencia es insuficiente?

Hay, no obstante, una precaución: HealthAlpaca fue adaptado para tareas de predicción de salud, no para calibrar incertidumbre o abstenerse ante señales degradadas. No debemos presuponer que vaya a ser más prudente que Qwen.

### **Una cuestión metodológica importante**

Si incorporamos estos modelos, mantendría exactamente las mismas condiciones C0–C5, las mismas representaciones de entrada y el mismo protocolo de evaluación. Así podríamos estudiar una interacción especialmente interesante:

**Especialización del modelo × calidad de la evidencia.**

Por ejemplo, un modelo biomédico podría ser más preciso con señales limpias, pero también más propenso a producir explicaciones convincentes cuando las señales están degradadas. Ese resultado sería científicamente más interesante que demostrar simplemente que un modelo especializado obtiene mayor accuracy.

Mi recomendación es incorporar BioMistral como comparación biomédica y estudiar la viabilidad de HealthAlpaca como comparación adaptada a wearables. No introduciría todavía más modelos hasta comprobar que el coste experimental y el tamaño de la muestra permiten hacer comparaciones sólidas.

# Health-LLM metodo y HealthAlpaca

Sí. Y aquí hay una distinción importante: Health-LLM no se limita a proponer una estrategia de prompting ni a presentar HealthAlpaca. Aporta una metodología experimental bastante más amplia para adaptar y evaluar LLM con datos de wearables.

He revisado el artículo publicado en CHIL 2024, no solamente su resumen. Sus métodos se pueden organizar en seis bloques.
1. Enriquecimiento contextual de la evidencia
Es probablemente la aportación metodológica más interesante del trabajo, aparte del fine-tuning.

Health-LLM plantea que no basta con introducir las medidas fisiológicas: el modelo puede necesitar información adicional para interpretarlas correctamente.

Define tres tipos de contexto:
Componente	Información que incorpora	Finalidad
User Context	Edad, sexo, altura, peso, etc.	Personalizar la interpretación
Health Context	Definición del indicador, conocimiento fisiológico y ecuaciones	Proporcionar conocimiento sobre la variable que debe estimar
Temporal Context	Secuencias de observaciones a lo largo del tiempo	Preservar la evolución temporal

Además, estudia la combinación de los tres componentes.

Por ejemplo, para estimar readiness, no se limita a proporcionar frecuencia cardíaca, sueño y actividad. También puede explicar qué representa readiness y proporcionar información del usuario.

El artículo encuentra mejoras de hasta un 23,8 % mediante estrategias de enriquecimiento contextual. 

raw.githubusercontent.com+1

Relevancia para vuestro proyecto: esta metodología permitiría estudiar si el conocimiento fisiológico adicional hace que el modelo interprete mejor las señales degradadas o, por el contrario, le induce a completar información que ya no está presente.
2. Representación y codificación de las series temporales
Este bloque es especialmente relevante para vuestra investigación sobre cómo representar señales fisiológicas para LLM.

Health-LLM estudia distintas formas de introducir la información temporal y selecciona la representación denominada Natural Language String.

La idea es transformar las secuencias en texto legible por el modelo, en lugar de limitarse a proporcionar estadísticas agregadas.

Por ejemplo:
- Representación agregada: frecuencia cardíaca media de un periodo.
- Representación temporal: secuencia de valores de frecuencia cardíaca a lo largo del periodo.
El artículo destaca tres aspectos:
1. La representación textual permite conservar información temporal.
2. Puede representar valores ausentes mediante símbolos como NaN, sin necesidad de imputarlos previamente.
3. Es interpretable y compatible con los prompts convencionales.
Los autores comparan distintas representaciones temporales y encuentran que la cadena en lenguaje natural ofrece los mejores resultados en sus experimentos. 

raw.githubusercontent.com+1

Para vosotros: es un precedente directo para comparar representación numérica, resumen estadístico y secuencia textual. No obstante, Health-LLM no estudia sistemáticamente la pérdida de información causada por degradaciones controladas.
3. Razonamiento mediante múltiples estrategias de prompting
Además del enriquecimiento contextual, el trabajo compara distintas estrategias de inferencia:
- Zero-shot.
- Few-shot (3 ejemplos).
- Chain-of-Thought (CoT).
- Self-Consistency (SC).
En Self-Consistency genera cinco posibles trayectorias de razonamiento y utiliza la consistencia entre ellas para mejorar la respuesta.

Esto permite estudiar no solo si el modelo conoce una relación fisiológica, sino también cómo cambia su rendimiento cuando se le proporcionan ejemplos o se le solicita desarrollar un razonamiento.

Hay un resultado importante: la combinación CoT + Self-Consistency no mejora necesariamente todas las tareas ni todos los modelos. Su efecto depende del modelo y del objetivo predictivo. 

raw.githubusercontent.com+1

Relevancia: os proporciona estrategias de inferencia que podrían incorporarse como factores experimentales, aunque no las convertiría en el núcleo de vuestro estudio.
4. Adaptación del modelo mediante Instruction Tuning y PEFT
Aquí aparece HealthAlpaca.

Los autores no se limitan a cambiar los prompts: también adaptan los parámetros de los modelos.

Utilizan:
- Instruction tuning para especializar el modelo en las tareas de predicción de salud.
- LoRA como alternativa de Parameter-Efficient Fine-Tuning.
- Variantes de 7B y 13B.
La comparación entre HealthAlpaca y sus variantes LoRA permite estudiar cómo afecta la adaptación al rendimiento.

El modelo ajustado alcanza resultados competitivos frente a modelos mucho mayores en las tareas evaluadas. 

raw.githubusercontent.com+1

Esto es importante porque separa dos mecanismos distintos:
- Contextualización: proporcionar más información mediante el prompt.
- Adaptación paramétrica: modificar los parámetros del modelo para especializarlo en una tarea.
Para vuestro trabajo, esta distinción permitiría estudiar si un modelo adaptado a datos fisiológicos es más resistente a la degradación que uno generalista.
5. Evaluación cruzada entre datasets
Este es otro aspecto metodológico que considero especialmente interesante.

Health-LLM no se limita a comparar modelos sobre un único conjunto de datos. Utiliza cuatro datasets públicos:
- PMData.
- LifeSnaps.
- GLOBEM.
- AW_FB.
También estudia la adaptación de modelos con datos de un dataset y su rendimiento en otros.

Esto permite examinar hasta qué punto el aprendizaje obtenido en un contexto se transfiere a otro.

No es exactamente una validación externa de la interpretación de señales degradadas, pero sí constituye un precedente metodológico para estudiar generalización entre fuentes de datos. 

raw.githubusercontent.com+1

6. Ablation studies y comparación con modelos convencionales
Finalmente, el artículo incorpora una metodología de evaluación por ablación.

Estudia qué sucede cuando se eliminan determinados componentes del contexto:
- Información del usuario.
- Conocimiento sanitario.
- Información temporal.
- Combinaciones de estos elementos.
También compara los LLM con modelos convencionales como MLP, SVM y Random Forest, además de modelos de lenguaje especializados.

Esto permite determinar si las mejoras se deben realmente al contexto añadido, al modelo o a la propia información de entrada. 

raw.githubusercontent.com+1


Qué aporta Health-LLM y qué no aporta a vuestro proyecto
Esta es mi síntesis de pertinencia metodológica:
Metodología	Health-LLM	Aplicación potencial en vuestro estudio
Enriquecimiento contextual	Sí	Estudiar el efecto del conocimiento fisiológico
Representación temporal textual	Sí	Comparar representaciones de señales
Few-shot / CoT / Self-Consistency	Sí	Analizar sensibilidad a estrategias de inferencia
Instruction tuning / LoRA	Sí	Comparar modelos generalistas y adaptados
Evaluación cruzada entre datasets	Sí	Examinar generalización
Ablation studies	Sí	Identificar qué información contribuye a las respuestas
Degradación controlada de señales	No	Es uno de vuestros elementos diferenciales
Evaluación de claims generados	No como protocolo sistemático	Verificar fidelidad de las explicaciones
Calibración ante degradación	No	Estudiar si la certeza refleja la calidad de la evidencia
Abstención ante evidencia insuficiente	No como objetivo experimental	Evaluar autorregulación

Mi conclusión: qué deberíamos aprovechar
No copiaría Health-LLM como arquitectura experimental completa. Extraería tres elementos metodológicos especialmente útiles:
1. Context enhancement

Introducir conocimiento fisiológico y contexto temporal como variables experimentales. Esto permitiría comprobar si el conocimiento previo ayuda a interpretar señales incompletas o genera una falsa sensación de suficiencia.

2. Temporal encoding

Comparar las representaciones de la señal. Esta dimensión conecta directamente con vuestra pregunta sobre cómo la pérdida de información afecta al razonamiento del LLM.

3. Ablation y cross-dataset evaluation

Aislar el efecto de cada componente y estudiar si los resultados se mantienen al cambiar de dataset, modalidad o tarea.

Dejaría CoT, Self-Consistency y fine-tuning como extensiones o factores secundarios, porque introducirlos todos simultáneamente complicaría la interpretación de los resultados.

Una precisión importante: Health-LLM no desarrolla un sistema de control de calidad de señales ni un mecanismo para detectar que la evidencia es insuficiente. Su objetivo es mejorar la predicción de variables de salud. Por eso, vuestra investigación puede partir de su metodología de entrada y adaptación de datos fisiológicos, pero estudiar un problema diferente: qué sucede con la fidelidad de las respuestas cuando la evidencia que fundamenta esas respuestas pierde calidad.

Ese contraste es, a mi juicio, una forma sólida de posicionar vuestra contribución frente a Health-LLM.

Referencia principal: Kim et al. (2024), Health-LLM, Proceedings of Machine Learning Research, CHIL 2024.

# Repeticiones LLM

Sí. En la investigación con LLM se utilizan estrategias de **repetición de prompts y generación de múltiples respuestas**. Sin embargo, conviene distinguir tres procedimientos que persiguen objetivos diferentes.

## **1. Repetir exactamente el mismo prompt (*repeat prompting*)**

Se envía varias veces la misma entrada al mismo modelo y se comparan las respuestas.

Por ejemplo, ante una ventana de señales fisiológicas:

- Ejecución 1: «¿Qué estado fisiológico reflejan estas señales?»
- Ejecución 2: exactamente la misma pregunta y los mismos datos.
- Ejecución 3: exactamente la misma pregunta y los mismos datos.

El objetivo es medir:

- Estabilidad de las respuestas.
- Variabilidad entre ejecuciones.
- Frecuencia con la que cambia la clasificación.
- Variabilidad de la confianza declarada.
- Consistencia de los hechos y afirmaciones generados.

Esto es especialmente interesante para vuestro experimento: permite determinar si una misma evidencia degradada produce respuestas más variables que una evidencia limpia.

## **2. Self-consistency prompting**

Es una estrategia algo diferente. Se generan varias respuestas o cadenas de razonamiento para una misma pregunta y después se agregan los resultados, normalmente mediante votación mayoritaria.

**Ejemplo: cinco ejecuciones**

Run 1

**Stress**

Run 2

**Stress**

Run 3

**Non-stress**

Run 4

**Stress**

Run 5

**Stress**

**Resultado agregado**

Stress · 4/5

La respuesta final se obtiene por mayoría, no seleccionando arbitrariamente una ejecución.

En el artículo **Health-LLM**, Kim et al. utilizan esta estrategia con cinco cadenas de razonamiento candidatas (N=5). También combinan Self-Consistency con Chain-of-Thought y few-shot prompting.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com\&sz=32)

raw\.githubusercontent.com+1



## **3. Variaciones del prompt (*prompt ensembling*)**

En lugar de repetir exactamente la misma pregunta, se utilizan distintas formulaciones o estructuras para evaluar la sensibilidad del resultado:

- Prompt A: pregunta directa.
- Prompt B: pregunta con contexto fisiológico.
- Prompt C: pregunta con información temporal.
- Prompt D: pregunta con instrucciones explícitas sobre las limitaciones de los datos.

Esto permite estudiar si el modelo depende excesivamente de la formulación de la pregunta.

## **¿Qué recomendaría para vuestro experimento?**

Creo que tiene sentido incorporar la repetición de ejecuciones como una dimensión experimental, pero no confundirla con Self-Consistency.

| Estrategia                    | Qué permite estudiar                                | Pertinencia                                     |
| ----------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| Repetición idéntica           | Estabilidad de las respuestas                       | Muy alta                                        |
| Self-Consistency              | Mejora mediante agregación                          | Alta, como experimento secundario               |
| Variación de prompts          | Sensibilidad a la formulación                       | Alta                                            |
| Combinación de las anteriores | Robustez frente a distintas fuentes de variabilidad | Interesante, pero aumenta el coste experimental |

**Una precaución estadística importante:** las cinco respuestas de un mismo caso no son cinco observaciones independientes. Un estudio sobre repetición de prompts encontró una correlación intraclase (ICC) de 0,69; ignorarla puede inflar artificialmente el tamaño muestral y la significación estadística.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://pmc.ncbi.nlm.nih.gov\&sz=32)

pmc.ncbi.nlm.nih.gov+1



Mi propuesta sería mantener el mismo prompt y repetirlo, por ejemplo, cinco veces por caso, con configuración de generación controlada. Después mediría la variabilidad de:

1. La clasificación fisiológica.
2. La corrección de las afirmaciones.
3. La certeza lingüística.
4. La confianza numérica, si se solicita.
5. La abstención.

Esto permitiría formular una pregunta de investigación adicional: **¿la degradación de la evidencia fisiológica incrementa la inestabilidad de las respuestas del LLM, además de afectar a su corrección?**

Es una dimensión distinta de la precisión y puede aportar bastante al estudio.

Sí. **100 repeticiones por prompt es una cifra perfectamente posible y existen estudios que la utilizan. Incluso hay precedentes de 250 repeticiones por cada entrada.** No obstante, hay que distinguir entre lo que se puede hacer computacionalmente y lo que está justificado metodológicamente.

### **1. Precedentes publicados**

He localizado tres referencias especialmente ilustrativas:

| Estudio                                                                                                    | Repeticiones | Objetivo                                                              |
| ---------------------------------------------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------- |
| *Affiliation bias in peer review of abstracts by an LLM* (2024), analizado posteriormente por Gallo et al. | 250          | Estudiar variabilidad y sesgo en decisiones de revisión científica.   |
| *One Run Is Not Enough* (2026, preprint)                                                                   | 100          | Evaluar variabilidad de respuestas mediante un protocolo Monte Carlo. |
| *Health-LLM* (2024)                                                                                        | 5            | Self-Consistency: agregar distintas cadenas de razonamiento.          |

El primer caso es particularmente interesante: 900 combinaciones de resumen y afiliación, repetidas 250 veces, generaron 225.000 observaciones.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://oup.silverchair-cdn.com\&sz=32)

oup.silverchair-cdn.com+1



Por tanto, 100 no es una cifra excepcional ni un límite superior.

### **2. ¿Qué se gana aumentando las repeticiones?**

Depende de qué quieras medir:

- **5 repeticiones:** permiten detectar cierta inestabilidad, pero ofrecen una estimación limitada de la distribución de respuestas.
- **10-20:** permiten caracterizar mejor la variabilidad y estimar frecuencias de resultados.
- **30-50:** proporcionan una caracterización más sólida de la distribución, especialmente si existen respuestas minoritarias.
- **100:** permiten realizar un análisis Monte Carlo más detallado, estimar probabilidades de resultados y estudiar comportamientos poco frecuentes.
- **250 o más:** pueden justificarse cuando interesa caracterizar colas de distribución, eventos raros o variabilidad muy pequeña.

Estas cifras son orientativas, no umbrales estadísticos universales.

### **3. Una advertencia importante: repetir 100 veces no equivale a tener 100 observaciones independientes**

Este es un aspecto que afecta directamente a vuestro diseño.

Gallo et al. (2025) encontraron un ICC de 0,69 en las respuestas repetidas. En su experimento, 250 ejecuciones de una misma combinación equivalían aproximadamente a 1,45 observaciones efectivas bajo ese modelo de correlación.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://oup.silverchair-cdn.com\&sz=32)

oup.silverchair-cdn.com



Es decir, multiplicar las ejecuciones no multiplica proporcionalmente la información estadística.

Además, conviene diferenciar:

- Variabilidad estocástica del modelo.
- Variabilidad entre señales fisiológicas diferentes.
- Variabilidad introducida por degradar las señales.
- Variabilidad causada por cambiar la formulación del prompt.

No deberíamos mezclarlas.

### **4. Qué plantearía para vuestro experimento WESAD**

No descartaría utilizar 100 repeticiones. De hecho, propondría estudiar primero si aportan información adicional respecto a 10 o 20.

Una posible estrategia sería:

**Estudio piloto de repetibilidad**

1. **Seleccionar un subconjunto de señales**

   Casos de C0, C1, C2, C4 y C5, procurando incluir las distintas clases fisiológicas.
2. **Ejecutar 100 veces cada caso**

   Mismo modelo, prompt, parámetros de generación y evidencia. Registrar cada respuesta individual.
3. **Evaluar la convergencia**

   Comparar las estimaciones obtenidas con 5, 10, 20, 50 y 100 ejecuciones.
4. **Determinar el número adecuado**

   Identificar cuándo se estabilizan la clasificación, la certeza lingüística, la confianza y la abstención.

Posteriormente, ese número se aplicaría al experimento completo si los resultados lo justifican.

**Mi recomendación:** no establecería cinco repeticiones como una decisión definitiva. Para vuestro trabajo, probaría 100 en un subconjunto y determinaría empíricamente si 20, 50 o 100 son necesarios.

Hay una cuestión especialmente interesante desde el punto de vista científico: no estudiar solamente si las respuestas son correctas, sino si **la distribución de respuestas generadas cambia cuando se degrada la evidencia fisiológica**.

Por ejemplo, podríamos observar si C0 produce 95 respuestas coincidentes de 100, mientras que C5 produce cinco respuestas diferentes con frecuencias mucho más dispersas. Esa distribución puede aportar información sobre la estabilidad del modelo que una única respuesta o una mayoría de cinco ejecuciones no permite observar.

Y esto encaja directamente con vuestra investigación sobre fiabilidad de los LLM ante evidencia fisiológica degradada.

# Modelos y frameworks LLM fisiología

# **Evaluación y alineación** de modelos de lenguaje con señales fisiológicas

La evaluación de respuestas generadas por grandes modelos de lenguaje (LLM) en el contexto de señales fisiológicas abarca dos vertientes principales: la evaluación de la coherencia clínica cuando el LLM interpreta bioseñales y el uso de biomarcadores fisiológicos humanos para evaluar objetivamente las respuestas del modelo.

## Evaluación de la Coherencia Fisiológica

Evaluar si un LLM realmente interpreta patrones biomédicos o se basa en correlaciones espurias requiere metodologías estructuradas que verifiquen la alineación fisiológica y la robustez clínica (Bolpagni et al., 2026; Han et al., 2026).

* Modelos subrogados (*surrogate models*) emparejados con explicabilidad SHAP permiten auditar la coherencia fisiológica de predicciones de estrés basadas en variabilidad de la frecuencia cardíaca (VFC), revelando que los modelos más grandes mantienen una mayor consistencia fisiológica (Bolpagni et al., 2026).
* Las rúbricas booleanas precisas y adaptativas (*Adaptive Precise Boolean rubrics*) descomponen criterios clínicos complejos en preguntas binarias para evaluar respuestas de LLM sobre salud metabólica y datos de ponibles (*wearables*), duplicando la eficiencia de las escalas Likert (Mallinar et al., 2026).
* Los esquemas de evaluación en tres niveles (representación de entrada, uso del modelo y formato de salida) confirman que combinar características estructuradas, aprendizaje con pocos ejemplos (*few-shot*) y salidas explicadas optimiza la precisión en la predicción del estado humano (Wang et al., 2026).

**Figure 1.** A precision rubric for evaluating open-ended language responses to health queries. (A) A set of representative health queries and wearable data are used to construct inputs to the language model, which are then evaluated using our proposed evaluation rubric. (B) Our Results indicate that the proposed Precise Boolean rubric leads to higher agreement with humans (both experts and non-expert). Moreover, leveraging our proposed Adaptive Precise Boolean, the time being used in evaluation is halved compared to the traditional Likert scale, while maintaining high rater agreement and high quality evaluation. (C) An example of a query and response highlighting references to specific relevant parts of (Mallinar et al., 2026)

*How to read: El diagrama esquematiza en (A) la entrada de datos biomédicos hacia el modelo de lenguaje, en (B) la comparación metodológica entre la escala Likert y la rúbrica booleana precisa adaptativa, y en (C) un ejemplo de consulta evaluada mediante criterios binarios específicos.*

## Marcos para Interpretar Bioseñales

Para procesar e interpretar bioseñales adecuadamente, la literatura científica propone arquitecturas modulares, agentes coordinados y representaciones intermedias que evitan que el LLM trabaje directamente con secuencias numéricas extensas (Feli et al., 2025; Han et al., 2026).

| Framework / Modelo | Tipo de Bioseñal | Enfoque de Evaluación y Funcionamiento |
| :--- | :--- | :--- |
| **E²-LLM** (Fei et al., 2026) | EEG (Electroencefalografía) | Codificador de señales neuronales proyectado al espacio latente para análisis afectivo interpretable (Fei et al., 2026). |
| **OpenCHA Agent** (Feli et al., 2025) | PPG y ECG | Orquestador que conecta herramientas analíticas externas con el LLM para estimación fiable de frecuencia cardíaca (Feli et al., 2025). |
| **MedTsLLM** (Chan et al., 2025) | Series temporales médicas | Reprogramación de parches (*patch reprogramming*) y contextualización clínica para detección de anomalías y segmentación (Chan et al., 2025). |
| **Pipeline VFC en tiempo real** (Gellisch & Burr, 2025) | VFC / Intervalos R–R | Transmisión por API/WebSocket que modula la retroalimentación conversacional según carga cognitiva y estrés (Gellisch & Burr, 2025). |

**Figure 1:** Comparativa de marcos de integración y procesamiento de bioseñales basados en modelos de lenguaje.

## Evaluación con Señales Neurofisiológicas Humanas

Un avance complementario consiste en utilizar las respuestas neurofisiológicas del propio usuario para cuantificar la calidad y satisfacción ante el texto generado por el LLM sin depender de autoinformes subjetivos (Park et al., 2026).

* Las interfaces cerebro-computadora pasivas registran correlaciones implícitas de satisfacción mediante electroencefalografía (EEG), mostrando aumentos sostenidos de potencia delta frontotemporal ante respuestas de alta calidad (Park et al., 2026).
* El procesamiento de respuestas insatisfactorias desencadena una supresión localizada de la potencia gamma en regiones frontotemporales derechas durante los primeros 4 segundos, señalando sorpresa o violación de expectativas (Park et al., 2026).
* Clasificadores basados en ShallowFBCSPNet discriminan entre satisfacción e insatisfacción extrema frente al texto del LLM con precisiones medias de aproximadamente el 81% (Park et al., 2026).

En conjunto, estos marcos permiten auditar formalmente la veracidad clínica de las respuestas y cerrar el ciclo de interacción adaptativa mediante el registro continuo de biomarcadores fisiológicos (Gellisch & Burr, 2025; Park et al., 2026; Fei et al., 2026).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
F., Lin, H., Xie, Y., Ren, H., Shen, X., Ding, W., & Qi, T. (2026). E^2-LLM: Bridging Neural Signals and Interpretable Affective Analysis. *arXiv (Cornell University), abs/2601.07877*. https://doi.org/10.48550/arxiv.2601.07877
 
Bolpagni, M., Pozza, M., & Gabrielli, S. (2026). Assessing physiological coherence in stress related predictions of large language models: a surrogate based analysis of the Mistral 3 family using wearable HRV data. *medRxiv*. https://doi.org/10.64898/2026.04.24.26351717
 
Chan, N., Parker, F., Zhang, C., Bennett, W., Jia, M. Y., Fackler, J. C., & Ghobadi, K. (2025). MedTsLLM: Medical Time Series Analysis Using Multimodal LLMs. *IEEE Journal of Biomedical and Health Informatics, PP*, 1-14. https://doi.org/10.1109/jbhi.2025.3621512
 
Feli, M., Azimi, I., Liljeberg, P., & Rahmani, A. M. (2025). An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation. *Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), 2025*, 1-7. https://doi.org/10.1109/embc58623.2025.11254428
 
Gellisch, M., & Burr, B. (2025). Establishing a real-time biomarker-to-LLM interface: a modular pipeline for HRV signal acquisition, processing, and physiological state interpretation via generative AI. *Frontiers in Digital Health, 7*, 1670464-1670464. https://doi.org/10.3389/fdgth.2025.1670464
 
Han, Y., Beyan, C., Zhang, X., Liu, X., Liu, N., Sun, J., Hong, S., Ding, C., & Murino, V. (2026). Large language models in medical time series analysis. *arXiv (Cornell University), abs/2609.22262*. https://doi.org/10.48550/arxiv.2609.22262
 
Mallinar, N., Heydari, A. A., Liu, X., Faranesh, A. Z., Winslow, B. D., Hammerquist, N., Graef, B., Speed, C. A., Malhotra, M., Patel, S., Prieto, J. L., McDuff, D., & Metwally, A. A. (2026). A scalable framework for evaluating health language models. *npj Digital Medicine, 9*. https://doi.org/10.1038/s41746-026-02492-x
 
Park, W., Mansour, S., Kim, L., & Eid, M. (2026). Decoding user satisfaction during reading of pre-defined question–answer pairs: BCI approach using delta and gamma EEG. *Frontiers in Neuroscience, 20*, 1934150-1934150. https://doi.org/10.3389/fnins.2026.1934150
 
Wang, L., Guo, A., & J. (2026). An Empirical Study for Effective LLM-Based Human-State Prediction from Physiological Data. *International Conference on Affective Computing and Intelligent Interaction Workshops and Demos (ACIIW)*, 1-5. https://doi.org/10.1109/aciiw71585.2026.11712086
 
