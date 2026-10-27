# Do LLMs Say What the Evidence Supports? Evidence Sensitivity under Controlled Input Degradation

## 1. Planteamiento del problema

### 1.1. Contexto

Ante el incremento del uso
Los modelos de lenguaje de gran tamaño (LLM) se utilizan cada vez más para convertir datos estructurados y señales numéricas en explicaciones, resúmenes e interpretaciones en lenguaje natural. Estas interfaces permiten que personas sin formación técnica consulten datos que de otro modo requerirían un análisis especializado, y sus salidas son consumidas también por otros sistemas, como agentes o canales de procesamiento posteriores. Sin embargo, una respuesta fluida no garantiza que su contenido esté respaldado por la evidencia recibida ni que la certeza que expresa sea adecuada.

En estos sistemas deben distinguirse cinco propiedades relacionadas pero no equivalentes:

1. **Corrección** de la respuesta.
2. **Confianza declarada** por el modelo (por ejemplo, una confianza numérica verbalizada).
3. **Certeza transmitida por el lenguaje**, mediante marcadores epistémicos como *hedges* (posibilidad, duda, cautela) y *boosters* (seguridad, énfasis).
4. **Respaldo** que los datos proporcionan a cada afirmación concreta.
5. **Decisión de responder o abstenerse.**

Un sistema puede dar una respuesta correcta sin reconocer las limitaciones de la entrada, expresar una seguridad elevada ante información insuficiente, introducir afirmaciones que no pueden verificarse con los datos recibidos o abstenerse cuando la información disponible sí permite resolver la tarea.

Estas diferencias plantean una cuestión central para el procesamiento del lenguaje natural (PLN):

**¿Hasta qué punto el lenguaje generado por un LLM refleja la calidad y las limitaciones de la evidencia que recibe?**

Para estudiarla se utilizan señales fisiológicas como **banco de pruebas experimental**. Son series numéricas cuya alteración puede controlarse y cuantificarse, y permiten definir tareas con referencias de evaluación. A partir de una misma señal pueden generarse versiones modificadas (ruido, pérdida de muestras, interrupciones) y compararse las respuestas lingüísticas obtenidas.

El estudio **no** pretende desarrollar un método de procesamiento fisiológico ni un nuevo índice de calidad de sensores. Las señales son un entorno experimental, y los resultados se tratarán como evidencia sobre el comportamiento de los LLM, sin presuponer que se generalicen a otros dominios; esa generalidad se comprobará con un control adicional (véase OE3).

### 1.2. Problema

El problema central es que la relación entre la calidad de la evidencia de entrada y el comportamiento lingüístico de los LLM **no puede darse por supuesta**. La degradación de una entrada puede modificar la corrección, pero también puede afectar de manera distinta a la confianza declarada, a la certeza expresada en el lenguaje y al contenido de las afirmaciones. Estas dimensiones podrían cambiar de forma coherente con la pérdida de información o desacoplarse entre sí.

Por ejemplo, ante una señal degradada, un modelo podría mantener una respuesta categórica aunque haya disminuido su capacidad para inferir el estado de referencia; introducir expresiones de cautela que no se correspondan con una reducción efectiva de la confianza; o abstenerse aunque la información restante siga siendo suficiente. La evaluación exclusiva de la exactitud no permite distinguir estos comportamientos.

La dificultad metodológica principal es que **degradación, pérdida de información y falta de resolubilidad no son conceptos equivalentes**. Se adoptan las siguientes definiciones operativas:

- **Degradación:** alteración aplicada a la entrada con parámetros conocidos (tipo y severidad). Es la variable experimental.
- **Pérdida de información:** reducción de la información disponible para resolver la tarea, estimada empíricamente y no inferida de la severidad.
- **Resolubilidad:** una condición se considera *resoluble* para una tarea si un modelo de referencia, entrenado y evaluado sobre esa condición con partición por sujeto, supera el azar con un margen preespecificado (límite inferior del intervalo de confianza por encima de azar más un margen *m*); *irresoluble* si no lo hace; y *parcialmente resoluble* en la zona intermedia. La resolubilidad depende de la tarea y del modelo de referencia; no es una propiedad absoluta de la señal.

Una señal modificada puede conservar información suficiente para responder bien, y una señal original no garantiza por sí sola que una inferencia sea válida. Por tanto, **la severidad de la degradación no se utilizará como criterio para decidir si una respuesta debería ser correcta o si el modelo debería abstenerse**. Primero debe establecerse qué información conserva cada condición y en qué medida la tarea sigue siendo resoluble; sobre esa base se analiza si el lenguaje generado refleja las limitaciones de la evidencia.

### 1.3. Brecha de investigación

La literatura relacionada aborda distintos aspectos del problema, que deben compararse atendiendo a qué dimensiones evalúan realmente y bajo qué condiciones:

1. **Degradación controlada de entradas y estabilidad de decisiones.** Trabajos como *Telemetry degradation and sequential stability in LLM-integrated IoT decision systems* estudian cómo la alteración controlada de la telemetría (abandono gradual de datos, submuestreo, señalización de fiabilidad) afecta a la estabilidad de las decisiones del LLM. Constituyen un antecedente metodológico directo. Su foco es la estabilidad decisional, no la corrección, la calibración ni las propiedades del lenguaje generado.
2. **Fiabilidad de las mediciones y evidencia sobre su calidad.** *When Derived Measurements Mislead* (DFOT) formaliza la sobreconfianza del LLM ante mediciones derivadas y el papel de la evidencia externa de fiabilidad, con intervenciones emparejadas frente a barajadas para separar la reparación específica de la cautela genérica. Es el antecedente más próximo para el análisis secundario y para el diseño de controles.
3. **Evaluación de respuestas y razonamiento sobre datos.** WearableQA y benchmarks similares evalúan el razonamiento de los LLM sobre datos de wearables con comprobaciones computacionales deterministas. Sirven de antecedente para la verificación, que deberá distinguir entre resultados numéricos, afirmaciones sobre los datos e inferencias sobre estados fisiológicos.
4. **Predicción selectiva y conformal.** ConMIL (combinación de modelos pequeños y LLM con predicción conformal sobre señales fisiológicas) ofrece un antecedente de predicción con garantías de cobertura y de separación entre casos seguros e inciertos.
5. **Calibración, expresión lingüística de la incertidumbre y fidelidad.** La literatura de PLN sobre confianza verbalizada, marcadores epistémicos (*hedges* y *boosters*), abstención y detección de afirmaciones no respaldadas proporciona los métodos y las métricas que este estudio aplica. **La revisión sistemática de este bloque está pendiente** y condiciona la delimitación definitiva de la contribución.

A partir de estos antecedentes, la brecha que se pretende investigar es la **evaluación conjunta del comportamiento lingüístico de los LLM bajo degradación controlada, con niveles de alteración conocidos por construcción y con criterios explícitos de resolubilidad, de corrección y de respaldo de las afirmaciones**. En concreto: si la degradación afecta de forma coherente a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención; si esos efectos dependen de la representación de la señal; y si las expresiones de certeza reflejan las limitaciones de la evidencia.

La revisión preliminar sugiere que esta combinación puede ofrecer un espacio de contribución, especialmente desde el análisis lingüístico. **No se afirma** que ninguna de las dimensiones haya sido estudiada de forma aislada, ni se presupone novedad absoluta: la delimitación final requiere comprobar sistemáticamente los métodos, las métricas y las condiciones experimentales de los trabajos más próximos (OE1).

### 1.4. Justificación y alineamiento con la comunidad de PLN

La investigación se justifica por la necesidad de evaluar los LLM no solo por si producen respuestas correctas, sino por cómo **expresan y gestionan las limitaciones de la evidencia** que reciben. Desde el PLN, el estudio relaciona tres ámbitos que conviene analizar conjuntamente: la corrección factual, la expresión lingüística de la incertidumbre y la verificación del contenido generado. El análisis de *hedges*, *boosters* y otros marcadores epistémicos permite estudiar si las elecciones lingüísticas cambian ante modificaciones de la evidencia y si esos cambios se corresponden con la corrección, una perspectiva distinta de la que ofrecen la exactitud o la confianza numérica por sí solas.

El planteamiento se alinea con problemas centrales de la comunidad: fiabilidad de las respuestas generadas, calibración de la incertidumbre, fidelidad a la evidencia de entrada, detección de afirmaciones no respaldadas, abstención selectiva y análisis del comportamiento de los modelos. Su contribución potencial reside en estudiar estas propiedades **dentro de un mismo diseño experimental**, en lugar de asumir que una mejora en una de ellas implica mejoras en las demás.

El estudio no presupone que los modelos posean una capacidad intrínseca para evaluar la calidad de sus entradas. Busca determinar qué aspectos de la limitación de la evidencia se reflejan efectivamente en las respuestas y cuáles permanecen sin expresar. Desde el punto de vista experimental, la validez dependerá de establecer referencias apropiadas para la tarea y de comprobar qué condiciones conservan información suficiente para resolverla.

### 1.5. Contribuciones esperadas e implicaciones

**Contribuciones** (a confirmar con los resultados):

1. Una caracterización empírica conjunta de la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención bajo degradación controlada.
2. Un protocolo reutilizable que separa degradación, pérdida de información y resolubilidad mediante un modelo de referencia.
3. Instrumentos de medida validados: un verificador de afirmaciones con categorías explícitas y una medida de certeza lingüística.
4. Un recurso reproducible: protocolo, prompts, salidas en bruto y código, con limitaciones y consideraciones éticas documentadas.

**Implicaciones.** Cada hallazgo tendrá consecuencias prácticas que se formularán como implicaciones prudentes, no como productos prometidos:

| Pregunta | Qué se aprenderá sobre los LLM | Implicación (condicionada a los resultados) |
|---|---|---|
| RQ1 | Cómo cae la corrección según el tipo y la severidad de la degradación, y si cae más que la información perdida | Para quien consume el texto, hasta qué punto de degradación deja de ser un indicador fiable |
| RQ2 | Si la confianza numérica, los *hedges* y la abstención siguen la calidad de la evidencia | Si pueden usarse como señales de fiabilidad o conviene no fiarse de ellos |
| RQ3 | Qué proporción de lo que afirma el modelo está respaldada por los datos | Cuánto del texto requiere verificación independiente |
| RQ4 | Si la representación de la entrada cambia estos comportamientos | Cómo evitar confundir un efecto de formato con un efecto del modelo |

---

## 2. Preguntas de investigación

### RQ1. Corrección bajo degradación

**¿Cómo afectan el tipo y la severidad de la degradación de las señales de entrada a la corrección de las respuestas generadas por los LLM, y en qué medida excede ese efecto la pérdida de información que sufre un modelo de referencia?**

La severidad se trata como variable experimental, no como medida directa de la dificultad de la tarea ni como criterio automático de incorrección.

### RQ2. Confianza, certeza lingüística y abstención

**¿Cómo se relacionan la confianza declarada, la certeza expresada lingüísticamente y la decisión de abstenerse con la corrección de las respuestas y con la información disponible para resolver la tarea?**

Se distingue la confianza numérica de los marcadores lingüísticos de certeza y se analizan ambas en relación con la corrección. También se estudia si la abstención ocurre cuando la evidencia es insuficiente y si se evita cuando la tarea sigue siendo resoluble. La evaluación de la abstención se basa en los criterios empíricos de resolubilidad del piloto: no se considera que una respuesta deba omitirse solo porque la señal haya sido degradada.

### RQ3. Respaldo de las afirmaciones generadas

**¿Cómo cambia la distribución de afirmaciones respaldadas, contradichas, no respaldadas e indeterminadas en función del tipo y la severidad de la degradación?**

Se distinguen las afirmaciones numéricas, verificables por cálculo, de las interpretativas, que requieren una referencia adicional. La ausencia de respaldo no se interpreta automáticamente como falsedad.

### RQ4. Efecto de la representación de la señal

**¿Hasta qué punto la representación de la señal (secuencia numérica o resumen estadístico) modifica los efectos de la degradación sobre la corrección y el comportamiento lingüístico?**

La comparación se limita a condiciones que permitan separar razonablemente los factores. No se atribuye a la representación una diferencia que esté confundida con la familia de modelo, el formato de la entrada o la dificultad de la tarea.

### Análisis secundario: información externa sobre la calidad de la señal

¿Aporta un indicador externo de calidad información adicional para identificar respuestas incorrectas o afirmaciones no respaldadas, más allá de la confianza declarada? Su inclusión depende de disponer de un indicador válido, independiente de las etiquetas de corrección y sin filtración de información hacia la evaluación. No es un objetivo central del estudio.

### Análisis exploratorio: representaciones internas

¿Contienen las representaciones internas de modelos de pesos abiertos información sobre el estado de degradación de la entrada, aunque la salida lingüística no la refleje? Es un análisis condicionado a la viabilidad y no se plantea como confirmatorio.

---

## 3. Hipótesis

### 3.0. Principios de formulación y análisis

- Las hipótesis, las variables, los márgenes de equivalencia, las tolerancias y las reglas de exclusión se **registran antes del análisis principal**.
- Cada hipótesis incluye un **patrón esperado** y un **criterio de refutación**, para que pueda fallar.
- Análisis con **modelos mixtos** (efectos aleatorios de sujeto y ventana; modelo, representación, perturbación y severidad como efectos fijos), **bootstrap por sujeto**, corrección de Holm, tamaños de efecto con intervalos y **contrastes de equivalencia** (TOST) cuando la hipótesis predice ausencia de diferencia relevante.
- Se informa la sensibilidad a las **variantes de prompt** (varias paráfrasis), a la temperatura y a las repeticiones. Las versiones de modelo se fijan con fecha.
- **ECE y Brier** solo se emplean cuando la confianza pueda interpretarse como probabilidad de corrección y haya casos suficientes por intervalo; en otro caso se usan métricas de discriminación (AUROC) y de sobreconfianza.

### H1. Corrección bajo degradación

**Enunciado.** La corrección de las respuestas disminuirá con la degradación, con un efecto que depende del tipo de perturbación y que excederá la pérdida de información medida por el modelo de referencia.

**Patrón esperado.** Caída mayor ante perturbaciones que eliminan información estructural (por ejemplo, bloques de muestras) que ante ruido leve, sin presuponer una relación estrictamente monotónica con la severidad. El coste en exceso del LLM respecto al modelo de referencia es positivo en las condiciones de pérdida estructural.

**Operacionalización.** Corrección (exactitud y macro-F1) por condición. Interacción tipo × severidad en un modelo mixto. Coste en exceso = Δ corrección del LLM − Δ corrección del modelo de referencia.

**Criterio de refutación.** Las diferencias entre la condición original y las degradadas quedan dentro del margen de equivalencia (±Δ) en todas las condiciones, o el intervalo del coste en exceso incluye el cero en todas ellas.

### H2a. Confianza numérica y corrección

**Enunciado.** La confianza declarada discriminará las respuestas correctas de las incorrectas en la condición original, y esa capacidad disminuirá y la sobreconfianza aumentará bajo degradación.

**Operacionalización.** AUROC de la confianza como predictor de corrección; sobreconfianza = confianza media − exactitud entre las respuestas emitidas; ECE y Brier cuando procedan.

**Criterio de refutación.** La variación del AUROC entre la condición original y las degradadas queda dentro del margen y la sobreconfianza no aumenta (intervalo incluye 0), o la confianza cae al menos tanto como la corrección.

### H2b. Certeza lingüística

**Enunciado.** La frecuencia de *hedges* aumentará y la de *boosters* disminuirá con la degradación, pero el desplazamiento será menor que la pérdida de corrección, es decir, la certeza lingüística se desacoplará de la corrección.

**Operacionalización.** Índice de certeza lingüística (boosters − hedges por cada 100 palabras), obtenido con un procedimiento de extracción o anotación validado frente a anotación humana. Comparación de pendientes estandarizadas respecto a la corrección y a la confianza numérica.

**Criterio de refutación.** Los marcadores no cambian (equivalencia) o su pendiente estandarizada es igual o mayor que la de la pérdida de corrección.

**Nota.** No se presupone que la certeza lingüística sea una probabilidad calibrada ni que todos los marcadores se interpreten igual.

### H3. Interacción entre representación y degradación

**Enunciado.** El efecto de la degradación sobre la corrección y el comportamiento lingüístico diferirá entre representaciones (secuencia numérica frente a resumen estadístico) dentro de un mismo modelo.

**Patrón esperado.** No se fija una dirección a priori. Se prevé que el resumen estadístico conserve mejor la corrección bajo ruido y que la secuencia numérica sea más sensible a la pérdida estructural, pero esto se evalúa como contraste preespecificado.

**Operacionalización.** Contrastes de interacción representación × severidad por modelo, con TOST y márgenes definidos antes del análisis.

**Criterio de refutación.** Todos los contrastes de interacción quedan dentro del margen de equivalencia. Las diferencias entre familias de modelos no se atribuyen a la representación cuando ambos factores estén confundidos.

### H4. Adecuación de la abstención

**Enunciado.** La abstención del LLM seguirá la resolubilidad de forma imperfecta: habrá infra-abstención en condiciones irresolubles y/o abstenciones innecesarias en condiciones resolubles. Una mayor frecuencia de abstención no implicará una abstención más adecuada.

**Operacionalización.** Tasa de abstención por nivel de resolubilidad. Curva riesgo–cobertura y área bajo la curva (AURC). Comparación con una política de referencia basada en el modelo de referencia (abstenerse si la condición es irresoluble). Abstenciones innecesarias. Variantes de prompt con y sin opción explícita de abstención, y control de abstención informada frente a precaución genérica con intervenciones emparejadas frente a barajadas, adaptado de DFOT.

**Criterio de refutación.** En condiciones irresolubles la abstención es al menos 1 − δ₁ y en condiciones resolubles es como máximo δ₂, sin diferencia relevante respecto a la política de referencia.

### H5. Respaldo de las afirmaciones generadas

**Enunciado.** La proporción de afirmaciones contradichas y no respaldadas aumentará con la degradación y variará según el tipo de perturbación, y las afirmaciones numéricas se comportarán de forma distinta a las interpretativas.

**Operacionalización.** Salida estructurada con afirmaciones (métrica, ventana, dirección, magnitud). Verificación por cálculo reproducible con tolerancias preespecificadas. Categorías: respaldada, contradicha, no respaldada, indeterminada (nunca automáticamente falsa). Análisis de sensibilidad a las tolerancias. Validación del verificador con anotación humana sobre un subconjunto, con acuerdo mínimo preespecificado.

**Criterio de refutación.** La diferencia de proporciones respecto a la condición original queda dentro del margen en todas las perturbaciones, o las conclusiones no son robustas a las tolerancias.

### H6. Valor adicional de un indicador externo de calidad (análisis secundario)

**Enunciado.** Un indicador externo de calidad de señal aportará información adicional a la confianza declarada para identificar respuestas incorrectas o afirmaciones no respaldadas, con una utilidad que varía según el tipo y la severidad de la degradación.

**Operacionalización.** AUROC incremental de la combinación (indicador + confianza) frente a la confianza sola, y comparación con indicadores simples (reglas de varianza, índices de calidad estándar). Solo se estudia si existe un indicador válido, independiente de las etiquetas de corrección.

**Criterio de refutación.** El incremento de AUROC tiene un intervalo que incluye 0 o queda por debajo del margen preespecificado.

### HE. Representaciones internas (análisis exploratorio)

**Enunciado.** En modelos de pesos abiertos, las representaciones internas permitirán discriminar entre entrada original y degradada por encima del azar, incluso cuando la salida lingüística no lo refleje.

**Operacionalización.** Sondas lineales sobre estados ocultos con controles (etiquetas permutadas y tarea de control) y comparación con el comportamiento observable.

**Criterio de refutación.** La precisión de la sonda no supera la de los controles. Resultado exploratorio, no confirmatorio.

---

## 4. Objetivos

### Objetivo general

**Evaluar cómo la degradación controlada de señales fisiológicas afecta a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención de los LLM, examinando el papel de la representación de la entrada y estableciendo criterios reproducibles de resolubilidad para interpretar los resultados.**

### OE1. Delimitar el estado del arte y la contribución

Completar la revisión en tres flujos: (A) LLM y señales fisiológicas; (B) calibración, incertidumbre lingüística y abstención; (C) fidelidad y verificación de afirmaciones sobre datos numéricos o entradas corruptas. Elaborar una tabla de diferenciación frente a *Telemetry degradation and sequential stability*, DFOT, WearableQA, ConMIL y los trabajos que aparezcan, con las columnas: tarea, intervención o degradación, referencia de corrección, medida de confianza, certeza lingüística, abstención, verificación de afirmaciones y representación. El resultado es una delimitación documentada de la contribución.

### OE2. Construir un conjunto experimental con degradación controlada

Preparar un conjunto de señales con referencias adecuadas y generar condiciones originales y degradadas con perturbaciones reproducibles (ruido añadido, pérdida aleatoria de muestras, interrupciones o pérdida en bloque y, según viabilidad, artefactos de movimiento simulados). Documentar los parámetros y caracterizar las señales con métricas objetivas de calidad. No tratar como equivalentes la ausencia de señal, la señal constante, la pérdida aleatoria y la eliminación estructurada de información.

### OE3. Definir y validar el diseño y la resolubilidad

Establecer tarea, referencias de corrección, unidades de análisis, representaciones y modelos. Realizar un **piloto** que estime variabilidad y márgenes, defina la resolubilidad con el modelo de referencia y verifique que el LLM supera el azar en la condición original (si no, se replantea la tarea). Incluir **al menos un control de generalidad** (un segundo conjunto con artefactos reales o una serie no fisiológica).

### OE4. Medir corrección, confianza, certeza lingüística y abstención

Aplicar el diseño principal y estimar los efectos sobre la corrección, la confianza declarada, la certeza lingüística y la abstención en cada condición (H1, H2, H4), con el análisis estadístico preespecificado.

### OE5. Construir y validar los instrumentos de medida

Desarrollar el verificador de afirmaciones (salida estructurada, ejecución numérica, tolerancias y categorías) y la medida de certeza lingüística, y validarlos con anotación humana sobre un subconjunto, informando el acuerdo y los casos discrepantes (H2b, H5).

### OE6. Analizar el efecto de la representación y del modelo

Comparar representaciones dentro de cada modelo mediante contrastes preespecificados y documentar qué factores quedan confundidos (H3).

### OE7. Análisis secundario y exploratorio (condicionales)

Estudiar el valor adicional de un indicador externo de calidad (H6) y, para modelos de pesos abiertos, las representaciones internas (HE), solo si se dispone de un indicador válido e independiente y si el tiempo lo permite.

### OE8. Documentar y liberar un recurso reproducible

Publicar protocolo, prompts, salidas en bruto, código y hoja de datos, con limitaciones, consideraciones éticas y licencias de los datos.

**Hitos de difusión (no científicos).** Preparación y envío del manuscrito, ajustados al formato, los plazos y el alcance de la convocatoria.

---

## 5. Trazabilidad

| Pregunta | Hipótesis | Objetivos |
|---|---|---|
| RQ1 | H1 | OE2, OE3, OE4 |
| RQ2 | H2a, H2b, H4 | OE3, OE4, OE5 |
| RQ3 | H5 | OE5 |
| RQ4 | H3 | OE6 |
| Secundario | H6 | OE7 |
| Exploratorio | HE | OE7 |
| Transversal | | OE1, OE8 |

---

## 6. Alcance, limitaciones y ética

- **Alcance:** una tarea, un conjunto de datos principal y un control de generalidad. Los resultados no se extrapolan automáticamente a otros dominios, tareas o modelos.
- **No es un estudio clínico:** no evalúa utilidad diagnóstica ni debe usarse para decisiones médicas.
- **Datos:** respetar las licencias y condiciones de uso de los conjuntos empleados. No enviar datos sensibles a servicios externos sin comprobar sus términos y el consentimiento original.
- **Reproducibilidad:** al menos dos modelos de pesos abiertos, versiones fijadas con fecha, caché de llamadas y publicación de prompts y salidas. Los modelos de API pueden cambiar o retirarse.
- **Medición del lenguaje:** los marcadores de certeza dependen del idioma, del dominio y del prompt, y su interpretación no es uniforme.
- **Resolubilidad:** depende de la tarea y del modelo de referencia elegido; se documentan los criterios y se analiza su sensibilidad.
- **Estado de la evidencia:** la afirmación sobre la brecha es provisional hasta completar los flujos B y C y leer en detalle los antecedentes más próximos.

---

## 7. Decisiones a fijar en el piloto

1. **Tarea y datos:** candidato, estrés frente a reposo con WESAD (sujetos y ventanas por sujeto a definir). Criterio de paso: el LLM supera claramente el azar en la condición original y existen condiciones resolubles e irresolubles.
2. **Modelo de referencia** para la resolubilidad (clasificador supervisado sobre características) y margen *m*.
3. **Perturbaciones y niveles:** tres tipos × tres niveles + condición original.
4. **Representaciones:** secuencia numérica y resumen estadístico en modelos de texto. Una representación gráfica solo con un modelo con visión, como extensión.
5. **Modelos:** al menos tres, con al menos dos de pesos abiertos, versiones y fechas fijadas.
6. **Márgenes y tolerancias:** Δ (equivalencia), δ₁ y δ₂ (abstención), tolerancias numéricas del verificador y acuerdo mínimo de la validación humana.
7. **Tamaño orientativo:** 450 ventanas × 10 condiciones × 2 representaciones × 3 modelos ≈ 27.000 llamadas por pasada, a multiplicar por las repeticiones y las variantes de prompt. Ajustar tras el piloto.
8. **Controles negativos:** ventanas sin información, señal barajada y ausencia de señal, para comprobar si el modelo responde con el conocimiento previo del prompt.