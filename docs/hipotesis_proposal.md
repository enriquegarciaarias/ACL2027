# # Do LLMs Say What the Evidence Supports? Evidence Sensitivity under Controlled Input Degradation
¿Dicen los LLM lo que la evidencia respalda? Sensibilidad a la evidencia ante la degradación controlada de la entrada

## 1. Planteamiento del problema

### 1.1. Contexto

Ante el incremento del uso de modelos de lenguaje de gran tamaño (LLM) para transformar datos estructurados y señales numéricas o fisiológicas en explicaciones, resúmenes e interpretaciones en lenguaje natural, estas interfaces facilitan su consulta tanto por parte de personas sin formación técnica como por otros sistemas o procesos posteriores. Sin embargo, una respuesta fluida no garantiza que su contenido esté respaldado por la evidencia recibida ni que la certeza expresada por el modelo sea la adecuada. Para auditar este fenómeno, el proyecto analiza de forma conjunta cinco propiedades críticas del comportamiento de los sistemas: la corrección de la respuesta, la confianza declarada por el modelo, la certeza transmitida por el lenguaje mediante marcadores epistémicos (*hedges* y *boosters*), el respaldo que los datos proporcionan a las afirmaciones y la decisión de responder o abstenerse.

Nuestro enfoque aborda hasta qué punto el lenguaje generado por el LLM refleja la calidad y las limitaciones de la evidencia recibida. Para estudiar este comportamiento, las señales fisiológicas se emplean como un banco de pruebas experimental, aclarando que el propósito del estudio no es desarrollar un nuevo método de procesamiento fisiológico ni un nuevo índice de calidad de sensores, sino utilizar dichas señales para evaluar el comportamiento de los modelos.

### 1.2. Problema

El problema central que identificamos es que no podemos dar por sentada la relación entre la calidad de la evidencia de entrada y las respuestas o el comportamiento de los LLM. La degradación de una señal puede incidir en diversas dimensiones, que incluyen la corrección de las respuestas, la confianza declarada, la certeza expresada en el lenguaje y el contenido de las afirmaciones. Se trata de dimensiones que pueden evolucionar de forma coherente o desacoplarse entre sí ante el deterioro y la pérdida de calidad de la evidencia de entrada.

### 1.3. Brecha de investigación

La brecha que se pretende investigar es la **evaluación conjunta del comportamiento lingüístico de los LLM bajo degradación controlada**. En concreto: si la degradación afecta de forma coherente a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención; si dichos efectos dependen de la representación de la señal; y si las expresiones de certeza reflejan fielmente las limitaciones de la evidencia.

La revisión preliminar sugiere que esta combinación delimita nuestra contribución al evaluar a los LLM no solo por si producen respuestas correctas, sino por cómo expresan y gestionan las limitaciones de la evidencia que reciben, aportando una perspectiva distinta de la que ofrecen la exactitud o la confianza numérica por sí solas y determinando qué aspectos de dichas limitaciones se reflejan efectivamente en las respuestas y cuáles permanecen sin expresar.

---

## 2. Preguntas de investigación

* **RQ1:** ¿Cómo y en qué medida la corrección de las respuestas generadas por los LLM se ve afectada por el tipo y la severidad de la degradación de las señales de entrada?
* **RQ2:** ¿Cómo se relacionan la confianza declarada, la certeza expresada lingüísticamente y la decisión de abstenerse con la corrección de las respuestas y con la información disponible de entrada?
* **RQ3:** ¿Cómo cambia la distribución de afirmaciones categorizadas como respaldadas, contradichas, no respaldadas e indeterminadas en función del tipo y la severidad de la degradación?
* **RQ4:** ¿Hasta qué punto la representación de la señal (secuencia numérica o resumen estadístico) modifica los efectos de la degradación sobre las respuestas de los LLM?
* **RQ5:** ¿Aporta un indicador externo de calidad (p. ej., PSRI) información adicional que permita mejorar la identificación de errores o afirmaciones no respaldadas en el LLM?

---

## 3. Hipótesis

### H1. Corrección bajo degradación

* La corrección de las respuestas disminuirá con la degradación, con un efecto que depende del tipo de perturbación y que excederá la pérdida de información medida por el modelo de referencia (*coste en exceso*).

### H2a. Confianza numérica y corrección

* La confianza declarada discriminará las respuestas correctas de las incorrectas en la condición original, y esa capacidad disminuirá y la sobreconfianza aumentará bajo degradación.

### H2b. Certeza lingüística

* La frecuencia de *hedges* aumentará y la de *boosters* disminuirá con la degradación, pero el desplazamiento será menor que la pérdida de corrección; es decir, la certeza lingüística se desacoplará de la corrección.

### H3. Interacción entre representación y degradación

* El efecto de la degradación sobre la corrección y el comportamiento lingüístico diferirá entre representaciones (secuencia numérica cruda frente a resumen estadístico agregado) dentro de un mismo modelo.

### H4. Adecuación de la abstención

* La decisión de abstenerse del LLM seguirá la resolubilidad de forma imperfecta: habrá infra-abstención en condiciones irresolubles y/o abstenciones innecesarias en condiciones resolubles.

### H5. Respaldo de las afirmaciones generadas

* La proporción de afirmaciones contradichas (alucinaciones) y no respaldadas aumentará con la degradación y variará según el tipo de perturbación.

### H6. Valor adicional de un indicador externo de calidad (Análisis secundario)

* Un indicador externo de calidad de señal aportará información adicional a la confianza declarada para identificar respuestas incorrectas o afirmaciones no respaldadas, con una utilidad que varía según el tipo y la severidad de la degradación.

---

## 4. Objetivos

### Objetivo general

Evaluar cómo la degradación controlada de señales fisiológicas afecta a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención de los LLM, examinando el papel de la representación de la entrada y estableciendo criterios reproducibles de resolubilidad para interpretar los resultados.

### OE1. Delimitar el estado del arte y la contribución

Completar la revisión sistemática en tres flujos (LLM y señales fisiológicas; calibración, incertidumbre lingüística y abstención; fidelidad y verificación de afirmaciones) y elaborar una tabla de diferenciación analítica frente a la literatura previa.

### OE2. Construir un conjunto experimental con degradación controlada

Preparar señales con referencias adecuadas y generar condiciones originales y degradadas con perturbaciones reproducibles (ruido, pérdida aleatoria y pérdida en bloque), caracterizándolas con métricas objetivas de calidad.

### OE3. Medir corrección, confianza, certeza lingüística y abstención

Aplicar el diseño experimental principal para estimar los efectos cuantitativos y discursivos sobre las variables centrales en cada condición, siguiendo el análisis estadístico preespecificado.

### OE4. Construir y validar los instrumentos de medida

Desarrollar el verificador programático de afirmaciones en JSON y la métrica de certeza lingüística, validándolos empíricamente mediante anotación humana sobre un subconjunto de control.

### OE5. Analizar el efecto de la representación y del modelo

Comparar sistemáticamente las representaciones de entrada dentro de cada familia de modelos mediante contrastes preespecificados, documentando los factores de confusión.

### OE6. Análisis secundario y exploratorio (condicionales)

Estudiar el valor predictivo incremental de un indicador externo de calidad (H6) y examinar las representaciones internas en modelos de pesos abiertos (HE).

### OE7. Documentar y liberar un recurso reproducible

Publicar el protocolo completo, los *prompts*, las salidas en bruto, el código fuente y las consideraciones éticas y de limitación en un repositorio abierto.