# Do LLMs Say What the Evidence Supports? Evidence Sensitivity under Controlled Input Degradation
*(¿Dicen los LLM lo que la evidencia respalda? Sensibilidad a la evidencia ante la degradación controlada de la entrada)*

## 1. Planteamiento del problema

### 1.1. Contexto: El giro de los sensores a los LLM
Ante el incremento del uso de modelos de lenguaje de gran tamaño (LLM) para transformar datos estructurados y señales numéricas o fisiológicas en explicaciones, resúmenes e interpretaciones en lenguaje natural, estas interfaces facilitan su consulta tanto por personas sin formación técnica como por otros sistemas o agentes de procesamiento posterior. Sin embargo, una respuesta fluida no garantiza que su contenido esté respaldado por la evidencia recibida ni que la certeza expresada por el modelo sea la adecuada.

Este proyecto nace de una trayectoria previa centrada en la ciberseguridad y el análisis de sesgos (proyecto EXIST). En trabajos anteriores, se observó que añadir canales fisiológicos (sensores) como *features* a los procesos de NLP no mejoraba automáticamente los resultados si no se auditaba la calidad de la señal. Para resolver esto, se desarrolló el marco de auditoría **PSRI** (fiabilidad fisiológica, coherencia biológica y condición conductual), validado empíricamente con datos de PhysioNet, K-EmoCon y WESAD. 

El **giro metodológico** de esta tesis doctoral consiste en dejar de auditar los sensores (lo físico) para evaluar a los LLM (lo epistémico). La pregunta central ya no es solo si el LLM acierta, sino **si ajusta su respuesta a la evidencia disponible**. Para auditar este fenómeno, el proyecto analiza de forma conjunta cinco propiedades críticas del comportamiento de los sistemas:
1. **Corrección** de la respuesta.
2. **Confianza declarada** por el modelo (por ejemplo, confianza numérica verbalizada).
3. **Certeza transmitida por el lenguaje**, mediante marcadores epistémicos (*hedges* y *boosters*).
4. **Respaldo** que los datos proporcionan a cada afirmación concreta.
5. **Decisión de responder o abstenerse.**

### 1.2. Problema y Definiciones Operativas
El problema central es que la relación entre la calidad de la evidencia de entrada y el comportamiento lingüístico de los LLM no puede darse por supuesta. Un sistema puede dar una respuesta correcta sin reconocer las limitaciones de la entrada, expresar una seguridad elevada ante información insuficiente, introducir afirmaciones no verificables o abstenerse cuando la información sí permitía resolver la tarea. 

Para abordar esto metodológicamente, se establecen tres definiciones operativas fundamentales, evitando asumir que son conceptos equivalentes:
*   **Degradación:** alteración aplicada a la entrada con parámetros conocidos (tipo y severidad). Es la variable experimental.
*   **Pérdida de información:** reducción de la información disponible para resolver la tarea, estimada empíricamente y no inferida de la severidad.
*   **Resolubilidad:** una condición es *resoluble* si un modelo de referencia supera el azar con un margen preespecificado; *irresoluble* si no lo hace. La severidad de la degradación **no** se usará como criterio para decidir si el modelo debería abstenerse; primero se establece qué información conserva la señal.

### 1.3. Brecha de investigación
La literatura previa ha abordado la degradación de telemetría, la calibración y la predicción selectiva, pero **falta estudiar conjuntamente las cinco dimensiones bajo degradación controlada** (ACL carece de estudios que correlacionen LLM y sensores de esta forma). La brecha que se pretende investigar es la evaluación conjunta del comportamiento lingüístico de los LLM bajo degradación controlada, determinando si la degradación afecta de forma coherente a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención; y si las expresiones de certeza reflejan fielmente las limitaciones de la evidencia.

---

## 2. Preguntas de Investigación (RQs)

**RQ1. Corrección bajo degradación**
¿Cómo y en qué medida la corrección de las respuestas generadas por los LLM se ve afectada por el tipo y la severidad de la degradación de las señales de entrada?

**RQ2. Confianza, certeza lingüística y abstención**
¿Cómo se relacionan la confianza declarada, la certeza expresada lingüísticamente y la decisión de abstenerse con la corrección de las respuestas y con la información disponible de entrada?

**RQ3. Respaldo de las afirmaciones generadas**
¿Cómo cambia la distribución de afirmaciones categorizadas como respaldadas, contradichas, no respaldadas e indeterminadas en función del tipo y la severidad de la degradación?

**RQ4. Efecto de la representación de la señal**
¿Hasta qué punto la representación de la señal (secuencia numérica o resumen estadístico) modifica los efectos de la degradación sobre las respuestas de los LLM?

**Análisis secundario: Información externa sobre la calidad**
¿Aporta un indicador externo de calidad (p. ej., PSRI) información adicional que permita mejorar la identificación de errores o afirmaciones no respaldadas en el LLM, más allá de la confianza declarada?

**Análisis exploratorio: Representaciones internas**
¿Contienen las representaciones internas de modelos de pesos abiertos información sobre el estado de degradación de la entrada, aunque la salida lingüística no la refleje?

---

## 3. Hipótesis

*Nota metodológica: Las hipótesis incluyen su operacionalización y criterios de refutación para garantizar que puedan ser falsadas mediante modelos mixtos, bootstrap y correcciones estrictas por comparaciones múltiples (Holm, FDR).*

### H1. Corrección bajo degradación
*   **Enunciado:** La corrección de las respuestas disminuirá con la degradación, con un efecto que depende del tipo de perturbación y que excederá la pérdida de información medida por el modelo de referencia (*coste en exceso*).
*   **Operacionalización:** Corrección (exactitud y macro-F1) por condición. Interacción tipo × severidad. Coste en exceso = Δ corrección del LLM − Δ corrección del modelo de referencia.
*   **Criterio de refutación:** Diferencias dentro del margen de equivalencia (±Δ) en todas las condiciones, o el intervalo del coste en exceso incluye el cero.

### H2a. Confianza numérica y corrección
*   **Enunciado:** La confianza declarada discriminará las respuestas correctas de las incorrectas en la condición original, y esa capacidad disminuirá y la sobreconfianza aumentará bajo degradación.
*   **Operacionalización:** AUROC de la confianza como predictor de corrección; sobreconfianza = confianza media − exactitud; ECE y Brier cuando procedan.
*   **Criterio de refutación:** La variación del AUROC queda dentro del margen y la sobreconfianza no aumenta.

### H2b. Certeza lingüística
*   **Enunciado:** La frecuencia de *hedges* aumentará y la de *boosters* disminuirá con la degradación, pero el desplazamiento será menor que la pérdida de corrección; es decir, la certeza lingüística se desacoplará de la corrección.
*   **Operacionalización:** Índice de certeza lingüística (boosters − hedges por cada 100 palabras). Comparación de pendientes estandarizadas respecto a la corrección y a la confianza numérica.
*   **Criterio de refutación:** Los marcadores no cambian o su pendiente estandarizada es igual o mayor que la de la pérdida de corrección.

### H3. Interacción entre representación y degradación
*   **Enunciado:** El efecto de la degradación sobre la corrección y el comportamiento lingüístico diferirá entre representaciones (secuencia numérica cruda frente a resumen estadístico agregado) dentro de un mismo modelo.
*   **Operacionalización:** Contrastes de interacción representación × severidad por modelo, con TOST (equivalencia) y márgenes predefinidos.
*   **Criterio de refutación:** Todos los contrastes de interacción quedan dentro del margen de equivalencia.

### H4. Adecuación de la abstención
*   **Enunciado:** La decisión de abstenerse del LLM seguirá la resolubilidad de forma imperfecta: habrá infra-abstención en condiciones irresolubles y/o abstenciones innecesarias en condiciones resolubles.
*   **Operacionalización:** Tasa de abstención por nivel de resolubilidad. Curva riesgo–cobertura (AURC). Comparación con una política de referencia basada en el modelo predictivo.
*   **Criterio de refutación:** En condiciones irresolubles la abstención es al menos 1 − δ₁ y en resolubles como máximo δ₂, sin diferencia relevante respecto a la política de referencia.

### H5. Respaldo de las afirmaciones generadas
*   **Enunciado:** La proporción de afirmaciones contradichas (alucinaciones) y no respaldadas aumentará con la degradación y variará según el tipo de perturbación, y las afirmaciones numéricas se comportarán de forma distinta a las interpretativas.
*   **Operacionalización:** Salida estructurada. Verificación por cálculo reproducible. Categorías: respaldada, contradicha, no respaldada, indeterminada. Validación con anotación humana.
*   **Criterio de refutación:** La diferencia de proporciones respecto a la condición original queda dentro del margen en todas las perturbaciones.

### H6. Valor adicional de un indicador externo de calidad (Análisis secundario)
*   **Enunciado:** Un indicador externo de calidad de señal aportará información adicional a la confianza declarada para identificar respuestas incorrectas o afirmaciones no respaldadas, con una utilidad que varía según el tipo y la severidad de la degradación.
*   **Operacionalización:** AUROC incremental de la combinación (indicador + confianza) frente a la confianza sola.
*   **Criterio de refutación:** El incremento de AUROC tiene un intervalo que incluye 0 o queda por debajo del margen.

### HE. Representaciones internas (Análisis exploratorio)
*   **Enunciado:** En modelos de pesos abiertos, las representaciones internas permitirán discriminar entre entrada original y degradada por encima del azar, incluso cuando la salida lingüística no lo refleje.
*   **Operacionalización:** Sondas lineales sobre estados ocultos con controles (etiquetas permutadas) y comparación con el comportamiento observable.
*   **Criterio de refutación:** La precisión de la sonda no supera a los controles.

---

## 4. Objetivos

### Objetivo general
Evaluar cómo la degradación controlada de señales fisiológicas afecta a la corrección, la confianza declarada, la certeza lingüística, el respaldo de las afirmaciones y la abstención de los LLM, examinando el papel de la representación de la entrada y estableciendo criterios reproducibles de resolubilidad para interpretar los resultados.

### OE1. Delimitar el estado del arte y la contribución
Completar la revisión en tres flujos: (A) LLM y señales fisiológicas; (B) calibración, incertidumbre lingüística y abstención; (C) fidelidad y verificación de afirmaciones. Elaborar una tabla de diferenciación frente a la literatura previa (Telemetry degradation, DFOT, WearableQA, etc.).

### OE2. Construir un conjunto experimental con degradación controlada
Preparar señales con referencias adecuadas y generar condiciones originales y degradadas con perturbaciones reproducibles (ruido, pérdida aleatoria y pérdida en bloque), caracterizándolas con métricas objetivas de calidad.

**Control de confusor del movimiento (decisión C):** Los canales de acelerómetro (ACC) y EMG se incluyen como factor experimental. Se generan dos vistas LLM para cada representación:
- `numeric_llm` / `text_llm`: sin canales de movimiento (vista limpia, evita confusor).
- `numeric_llm_motion` / `text_llm_motion`: con canales de movimiento (vista con confusor).

Esto permite reportar resultados con y sin canales de movimiento y evaluar si el LLM detecta la degradación fisiológica por sí mismo o si se basa en el movimiento como atajo.

**Embebidos (material complementario):** Los vectores de `mxbai-embed-large` no se usan como entrada al LLM (los modelos de chat reciben texto, no vectores). Se mantienen como material complementario para posible uso futuro como línea base densa, recuperación de ventanas similares, o sondas sobre estados internos del LLM. No forman parte del diseño principal de representaciones para el LLM en esta fase.

### OE3. Definir y validar el diseño y la resolubilidad
Establecer la tarea y realizar un piloto que estime variabilidad, defina la resolubilidad con el modelo de referencia y verifique que el LLM supera el azar en la condición original. Incluir al menos un control de generalidad (ej. un segundo conjunto con artefactos reales).

### OE4. Medir corrección, confianza, certeza lingüística y abstención
Aplicar el diseño experimental principal para estimar los efectos cuantitativos y discursivos sobre las variables centrales en cada condición, siguiendo el análisis estadístico preespecificado (H1, H2a, H2b, H4).

### OE5. Construir y validar los instrumentos de medida
Desarrollar el verificador programático de afirmaciones y la métrica de certeza lingüística, validándolos empíricamente mediante anotación humana sobre un subconjunto de control (H2b, H5).

### OE6. Analizar el efecto de la representación y del modelo
Comparar sistemáticamente las representaciones de entrada dentro de cada familia de modelos mediante contrastes preespecificados, documentando los factores de confusión (H3).

### OE7. Análisis secundario y exploratorio (condicionales)
Estudiar el valor predictivo incremental de un indicador externo de calidad (H6) y examinar las representaciones internas en modelos de pesos abiertos (HE).

### OE8. Documentar y liberar un recurso reproducible
Publicar el protocolo completo, los *prompts*, las salidas en bruto, el código fuente y las consideraciones éticas y de limitación en un repositorio abierto.

---

## 5. Trazabilidad

| Pregunta | Hipótesis | Objetivos |
| :--- | :--- | :--- |
| **RQ1** | H1 | OE2, OE3, OE4 |
| **RQ2** | H2a, H2b, H4 | OE3, OE4, OE5 |
| **RQ3** | H5 | OE5 |
| **RQ4** | H3 | OE6 |
| **Secundario** | H6 | OE7 |
| **Exploratorio** | HE | OE7 |
| **Transversal** | | OE1, OE8 |

---

## 6. Alcance, limitaciones y ética
*   **Alcance:** Una tarea principal y un control de generalidad. Los resultados no se extrapolan automáticamente a otros dominios o modelos.
*   **No es un estudio clínico:** No evalúa utilidad diagnóstica ni debe usarse para decisiones médicas.
*   **Datos y Reproducibilidad:** Respetar licencias de los datasets (WESAD, etc.). Uso de al menos dos modelos de pesos abiertos, versiones fijadas, caché de llamadas y publicación de salidas. Los modelos de API pueden cambiar o retirarse.
*   **Medición del lenguaje:** Los marcadores de certeza dependen del idioma, del dominio y del prompt.

## 7. Confusor natural del movimiento

### 7.1. Descripción del problema

En WESAD el estrés se induce con tareas de hablar en público y aritmética. Al hablar y moverse, las ventanas de estrés tienen **más movimiento y artefactos reales** que las de reposo. Esto significa que la calidad de la señal ya correlaciona con la etiqueta de forma natural:

| Señal | Baseline | Estrés | Relación con etiqueta |
|-------|----------|--------|----------------------|
| Movimiento (ACC) | Bajo | Alto | **Natural** |
| EMG | Bajo | Alto (tensión muscular) | **Natural** |
| Calidad de señal (SQI) | Buena | Peor (más artefactos) | **Indirecta** |

### 7.2. Por qué importa

El SQI, la fracción de movimiento del acelerómetro y el EMG pueden actuar como **atajos**: el modelo puede acertar "estrés" porque detecta movimiento, no porque interprete la fisiología. Cuando se inyecte degradación (paso `degrade`), el efecto de lo inyectado se mezclará con la diferencia natural entre estrés y reposo. Si el LLM parece "notar" que la señal es mala, podría estar notando el estrés a través del movimiento, no la pérdida de información fisiológica.

### 7.3. Qué comprobar antes de interpretar H1 y H4

1. **Distribución del SQI por etiqueta y sujeto** (diagramas de caja o histogramas): verificar si `SQI_overall` difiere significativamente entre baseline y estrés.
2. **Features que separan las clases por sí solas** (AUC univariante, importancia en el modelo de referencia): las candidatas a ser atajos son la fracción de movimiento, el EMG y las del acelerómetro.
3. **Modelo de referencia con y sin canales de movimiento**: entrenar dos veces (con y sin ACC/EMG). Si el rendimiento cae drásticamente al quitar movimiento, el modelo estaba usando el atajo.

### 7.4. Qué hacer según el resultado

| Resultado | Acción |
|-----------|--------|
| Movimiento/EMG separan clases con AUC > 0,80 | Excluir esas features de la vista LLM, o incluirlas como factor experimental y reportar con/sin ellas |
| SQI difiere por etiqueta | Reportar resultados estratificados por sujeto (diseño dentro de sujeto) para controlar el confusor |
| El modelo de referencia cae al quitar movimiento | El confusor es real; diseñar el experimento para comparar dentro de cada sujeto, no entre sujetos |

### 7.5. Resultados de verificación preliminar (S2)

Se ejecutó `scripts/verify_motion_confounder.py` sobre las 45 ventanas de S2. Resultados:

**SQI por etiqueta:**
- Baseline (1): media=0.942, std=0.014
- Estrés (2): media=0.894, std=0.016
- Amusement (3): media=0.945, std=0.015
- Meditación (4): media=0.953, std=0.019

El estrés tiene SQI sistemáticamente menor que baseline, aunque la diferencia es modesta (0.048 puntos).

**AUC univariante (estrés vs baseline):**
- `acc.acc_std_g`: AUC=0.928 (sesgo hacia estrés) ← **confusor fuerte**
- `acc.acc_motion_fraction`: AUC=0.825 (sesgo hacia estrés) ← **confusor moderado**
- Otras features de movimiento: AUC entre 0.45 y 0.52 (no discriminativas en este sujeto)

**Modelo de referencia (validación cruzada 5-fold):**
| Modelo | Todas las features | Sin movimiento | Caída AUC | Caída F1 |
|--------|-------------------|----------------|-----------|----------|
| Regresión Logística | 0.972 | 0.961 | +0.011 | +0.000 |
| Random Forest | 1.000 | 1.000 | +0.000 | +0.048 |

**Conclusión preliminar:** El confusor del movimiento existe en S2 (ACC tiene AUC > 0.8), pero no es dominante para este sujeto específico. Se recomienda:
1. Ejecutar la misma verificación en los otros 14 sujetos antes de decidir
2. Si el patrón se confirma, excluir canales de movimiento de la vista LLM (`numeric_llm`, `text_llm`) o reportar resultados con/sin ellos como factor experimental

### 7.6. Implementación

Si el confusor se confirma en múltiples sujetos, se añadirá a `config.json` el bloque:

```json
"experiment": {
  "control_motion_confounder": true,
  "exclude_motion_features_from_llm": true
}
```

Y la vista LLM (`numeric_llm`, `text_llm`) filtrará los canales `acc` y `emg` de las representaciones cuando ese flag esté activo.

---

## 8. Decisiones a fijar en el piloto (Octubre-Noviembre)
1.  **Tarea y datos:** Candidato estrés vs. reposo con WESAD. Criterio de paso: el LLM supera claramente el azar en la condición original.
2.  **Modelo de referencia** para la resolubilidad y margen *m*.
3.  **Perturbaciones y niveles:** Tres tipos × tres niveles + condición original.
4.  **Representaciones:** Secuencia numérica y resumen estadístico.
5.  **Modelos:** Al menos tres (dos de pesos abiertos).
6.  **Tamaño orientativo:** ~27.000 llamadas por pasada (450 ventanas × 10 condiciones × 2 representaciones × 3 modelos).
7.  **Controles negativos:** Ventanas sin información, señal barajada y ausencia de señal, para comprobar si el modelo alucina basándose en el conocimiento previo del prompt.