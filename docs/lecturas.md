# Lecturas prioritarias

Diez papers para entender el estado del arte de esta tesis, con su relevancia
para la investigación y una versión en español de su contenido por secciones.

Selección tomada de [`analisisDatasets.md`](analisisDatasets.md) §1.1 («Qué leer
primero»), contrastada con el [análisis de competidores](analisis_competidores_fidelidad_situacional_WESAD_DREAMT.md)
y con [`literatura_slip.md`](literatura_slip.md).

---

## Cómo se ha construido este documento

**De dónde sale el texto.** El corpus SLIP ofrece dos representaciones del mismo
material: los *notebooks* (`docs/SLIP/notebook/gemini_notebook_01..03.md`), que
contienen el **texto completo** de cada paper con una cabecera
(`TITULO DEL PAPER`, `Paper ID`, `Relevance Score`, `Palabras`), y
`docs/SLIP/papers_zoned.json`, que guarda **un párrafo por sección**
(`abstract`, `introduction`, `methodology`, `results`, `conclusion`).

- Para **leer el paper entero**, la fuente adecuada es el **notebook**: es la
  exportación de lectura, con el texto íntegro y sin recortes.
- Para **extraer y ordenar por secciones**, `papers_zoned` es mejor: su unidad ya
  es la sección, lo que evita reconstruirla desde un texto continuo.

Decisión: el texto íntegro de referencia es el **notebook** (se indica el fichero
en cada ficha y en el apéndice); la segmentación de este documento se apoya en
`papers_zoned`, verificado contra el notebook.

**Qué es «el texto» de cada ficha.** No es una traducción automática literal.
Es una **versión ampliada en español, organizada por secciones**, que sigue el
argumento del paper, conserva las cifras, los datasets y los términos técnicos
(*few-shot*, *prompt*, etc.) y cubre el contenido sustantivo disponible en el
corpus: prácticamente todo en los papers cortos (`ieee_88`) y en la parte
recuperada de `pubmed_150` y `arxiv_51`; las secciones principales en los
extensos y las revisiones. No se reproducen listas de referencias, tablas
numéricas completas ni pies de página.

**Hipótesis de referencia:** H1 corrección bajo degradación · H2a confianza
numérica frente a acierto · H2b certeza lingüística (hedges/boosters) · H3 efecto
de la representación · H4 adecuación de la abstención · H5 respaldo de las
afirmaciones · H6 valor de un indicador externo de calidad.

---

## Mapa de lecturas

| # | Paper | Dataset principal | Hipótesis | Papel |
|:--|:-----------------------------|:--------------------|:---------|:---------------------------|
| 1 | `ieee_88` Estudio empírico de predicción de estados humanos | WESAD, HHAR, DREAMT | H1, H3 | Diseño de la entrada y del *few-shot* |
| 2 | `scopus_111` StressLLM | PMData, WESAD | H1, H3 | Biomarcadores y ablación de features |
| 3 | `pubmed_150` Estrés como anomalía personalizada (UniTS) | WESAD, DREAMER, MAHNOB-HCI, @HOME | H1, H5 | Robustez a wearables y explicaciones |
| 4 | `arxiv_51` Personalización con HRV (BLP) | TILES-2018, WESAD | H3 | Representación y normalización por sujeto |
| 5 | `scopus_293` Revisión de LLM para series temporales médicas | MIMIC, VitalDB, PhysioNet, WESAD | H1–H3 | Mapa del campo |
| 6 | `arxiv_288` = `manual_2` DFOT | MIMIC-III *matched waveform* (PPG–ECG) | H2a, H5, H6 | Competidor conceptual |
| 7 | `manual_5` Degradación de telemetría | Telemetría sintética de smartwatch | H1, H3 | Competidor metodológico |
| 8 | `manual_3` PhysioBench | 22 datasets públicos | H1, H3 | Techo de comparabilidad |
| 9 | `arxiv_7` Revisión de MedTSLLM | ECG, EEG, PPG, signos vitales | H1–H3 | Marco de referencia |
| 10 | `scopus_68` Señales cognitivas para LM | Eye-tracking, EEG (ZuCo) | H2b, H5 | Motivación y alucinaciones |

---

## 1. `ieee_88` — Estudio empírico de predicción de estados humanos con LLM

**Título original:** *An empirical study for effective LLM based human state
prediction from physiological data* (Wang, Guo y Ma; ACIIW 2026).

**Ficha.** Datasets: **WESAD, HHAR, DREAMT**. Señal: ECG y datos de sensores
fisiológicos/wearables. Tamaño del texto: 1.318 palabras. Relevancia SLIP: 0,77.

**Por qué es relevante.** Es el paper más próximo a nuestras decisiones de
diseño: compara *cómo se presenta la señal* (features frente a datos crudos) y
*cómo se usa el LLM* (zero-shot frente a few-shot) sobre WESAD, que es nuestro
mismo dataset. Justifica dos elecciones que ya están en el pipeline —entrada por
features y ejemplos few-shot de S2–S4— y aporta evidencia de que el rendimiento
varía por sujeto y por posición temporal, lo que respalda normalizar por sujeto
(E12) y agrupar el bootstrap por sujeto. No estudia degradación ni abstención.

### Texto
**Resumen.** Los estudios recientes emplean LLM para predecir estados humanos
a partir de datos fisiológicos, pero difieren en tres decisiones: cómo se
representa la señal, cómo se usa el modelo y qué forma tiene la salida. No está
claro qué combinación funciona ni si las conclusiones de un sistema se trasladan
a otro. Los autores proponen organizar el sistema en **tres capas** —entrada,
uso del LLM y salida— y comparar sus opciones bajo un marco común.

**Motivación y estado del arte.** Los trabajos previos ilustran el problema: Wang
et al. estudiaron la inferencia de estados de salud con ECG; HARGPT introduce
datos crudos de sensores como entrada; Liu et al. usan ejemplos few-shot para
adaptar el modelo a la tarea; ZARA combina un agente LLM con conocimiento de
features y genera explicaciones en lenguaje natural. Cada uno explora una opción
distinta, en un sistema distinto y con un dataset distinto. Lo que falta es una
comparación **unificada** de esas opciones sobre las mismas tareas y los mismos
datos.

**Marco de tres capas.** La capa de **representación de entrada** distingue entre
pasar la señal en bruto y pasar **features** calculadas. La capa de **uso del
LLM** distingue entre **zero-shot** y **few-shot** (con ejemplos en el prompt).
La capa de **forma de salida** distingue entre devolver solo la **etiqueta** o
devolver la etiqueta **acompañada de una explicación**. El producto cartesiano de
las opciones se evalúa en varios LLM.

**Tareas y datos.** Tres tareas comunes de estado humano: **predicción de
estrés**, **sueño-vigilia** y **estado de actividad**. Las muestras se convierten
en entradas textuales. Los datasets son **WESAD, HHAR y DREAMT**: el primero
aporta fisiología de pecho y muñeca con condiciones afectivas inducidas; el
segundo, actividad con teléfono y reloj; el tercero, sueño con referencia de
polisomnografía.

**Resultados.** La configuración eficaz es **features + few-shot + etiqueta con
explicación**, y se mantiene entre tareas y modelos. La entrada por features
**supera a los datos crudos**; el few-shot **mejora siempre** la accuracy; la
diferencia entre las dos formas de salida es pequeña. Es decir: la mayor parte de
la ganancia está en cómo se presenta la evidencia y en dar ejemplos, no en pedir
una explicación.

**Análisis por sujeto.** El rendimiento **varía entre individuos**. Añadir un
ejemplo específico por etiqueta y por sujeto eleva la accuracy media por sujeto
en los tres datasets. Los autores lo interpretan como que una **personalización
limitada** reduce los errores atribuibles a diferencias individuales, sin
necesidad de reentrenar.

**Análisis temporal.** La accuracy también cambia según la **posición de la
muestra dentro de la secuencia**: no todas las fases de un registro se predicen
igual. Esto sugiere que el orden y la deriva temporal son una fuente de error
que merece tratamiento propio.

**Conclusiones, limitaciones y ética.** Los autores señalan tres limitaciones:
solo usan LLM cerrados y métodos de prompting; faltan métricas como macro-F1,
precisión y recall en más datasets; y conviene estudiar las muestras mal
clasificadas para identificar factores temporales o individuales. Añaden una
advertencia ética relevante para nuestra tesis: las predicciones **no deben
interpretarse como mediciones directas del estado interno**; los datos
fisiológicos y de movimiento permiten reidentificación, y cualquier uso requiere
consentimiento, validación por población y supervisión humana.
---

## 2. `scopus_111` — StressLLM

**Título original:** *StressLLM: Large Language Models for Stress Prediction via
Wearable Sensor Data* (2025).

**Ficha.** Datasets: **PMData** y **WESAD**. Señal: datos fisiológicos y de
wearables (HR, sueño, actividad, EDA, ECG). Tamaño del texto: 5.535 palabras.
Relevancia SLIP: 0,76; 14 citas.

**Por qué es relevante.** Es el competidor más directo en WESAD. Confirma tres
cosas que hemos observado: el *few-shot* cambia el resultado, el tamaño del
modelo no garantiza acierto, y en zero-shot los modelos **colapsan a «estrés»**.
Su conversión de WESAD a binario (estrés / no estrés) con remuestreo a 20 Hz es
un precedente directo de nuestro brazo binario. No mide degradación, confianza,
claims ni abstención.

### Texto
**Resumen.** El estrés afecta a la salud mental, al sistema inmunitario y al
bienestar, y reduce la productividad. Los LLM se han explorado en salud mental,
actividad y sueño, pero la predicción de estrés —y sobre todo **qué
biomarcadores** importan— está menos estudiada. El trabajo evalúa la capacidad de
varios LLM para predecir estrés con datos de wearables y hace una **ablación
sistemática de features** para identificar los biomarcadores que aportan y los
que solo añaden ruido.

**Introducción y motivación.** Los wearables (pulseras y relojes) aportan datos
fisiológicos continuos que, combinados con modelos, permiten evaluar el estrés en
tiempo casi real. Los LLM manejan información multimodal y no estructurada, y
además pueden **explicar** sus decisiones, lo que les da ventaja frente a los
sistemas clásicos de aprendizaje automático; su coste de inferencia limita el uso
en tiempo real. El antecedente directo es **Health-LLM**, que evalúa LLM con
datos de wearables para varias condiciones, incluido el estrés, pero sin analizar
qué biomarcadores son realmente útiles. Este trabajo cubre ese hueco.

**Contribuciones.** (i) Pruebas en varios datasets de estrés con cuatro familias
de modelos —BioMistral-DARE, Gemini, Llama y variantes de GPT— para evaluar el
efecto del número de parámetros; (ii) análisis del efecto de la **edad**; (iii)
análisis del efecto del **género**; (iv) identificación del **conjunto óptimo de
biomarcadores** para un despliegue eficiente en dispositivos con recursos
limitados.

**Datasets.** *PMData*: se construye una ventana deslizante de **14 días** y se
usa la valoración diaria de estrés como etiqueta; el modelo debe responder en una
escala **1–5** a partir de la historia de salud y actividad. *WESAD*: se
convierten las clases originales (**Baseline, Amusement, Stress, Meditation y
Recovery**) en **binario estrés / no estrés**; se seleccionan biomarcadores de
pecho y muñeca; se remuestrean todas las señales (originalmente entre **4 y 700
Hz**) a **20 Hz** para homogeneizar; se generan **500 prompts** tomando muestras
equilibradas por participante; la etiqueta de cada ventana se asigna por **valor
mayoritario**.

**Prompts y estrategias.** Se adopta el esquema de *prompting* de Health-LLM como
línea base. Cada prompt tiene un **preámbulo** (tarea e historial del participante
con sus biomarcadores) y un **formato de ejemplo** (uno o varios ejemplos de
respuesta). Se prueban **zero-shot, few-shot y cadena de pensamiento (CoT)**; en
few-shot se usan **3 ejemplos en PMData y 2 en WESAD**. En la fase preliminar se
generaron ~300 prompts por modo y se amplió a 500.

**Ablación y métricas.** La ablación elimina un biomarcador cada vez y reformula
el prompt para medir el efecto de su ausencia. Las métricas son **accuracy** y
**error absoluto medio (MAE)**. Se comparan con **siete modelos clásicos**:
árboles de decisión, Random Forest, SVM, gradient boosting, kNN, Naive Bayes y
regresión logística. Cada experimento se repite **tres veces** para calcular la
desviación típica.

**Resultados en PMData.** En los modelos clásicos, **SVM** obtiene la mejor
accuracy, **67,53 %**, con MAE **0,3593**, al excluir la feature «steps» (sugiere
que aporta poco). En los LLM, BioMistral-DARE logra su mejor MAE al quitar «sleep
minutes» (el sueño podría introducir ruido); GPT-4 mejora al quitar el estado de
ánimo autoinformado y GPT-3.5-Turbo al quitar las calorías quemadas; Gemini-Pro
mejora con few-shot y CoT y alcanza su mejor MAE, **0,7667**, al quitar la
frecuencia cardiaca en reposo. En general, añadir ejemplos ayuda a Gemini-Pro, y
el zero-shot rinde mejor sin la variable de estado de ánimo.

**Resultados en WESAD.** En los modelos clásicos, **Random Forest** alcanza
**91,13 %** de accuracy y MAE **0,0887** al excluir el **género**. BioMistral-DARE
rinde mejor sin el **ECG de pecho** (MAE **0,73**) y empeora al quitar el
acelerómetro de muñeca (MAE **0,86**). En zero-shot, GPT-4 y GPT-3.5-Turbo
obtienen MAE de **0,8067** y **0,8133** al excluir el acelerómetro de pecho y el
ECG de pecho, respectivamente; pero, pese a su tamaño, **respondieron
«estresado» a casi todas las muestras**. Introducir ejemplos mejora mucho a
GPT-3.5-Turbo. Gemini-Pro con zero-shot y CoT llega a MAE **0,4825** al quitar el
acelerómetro de pecho, y con few-shot (2 ejemplos) y CoT alcanza el mejor
resultado global: **MAE 0,1504** y **accuracy 84,96 %** al excluir el **EMG de
pecho**. Llama2 rinde mal: predice «estresado» en la mayoría de las muestras y su
mejor MAE en zero-shot es **0,8274** con todos los biomarcadores.

**Edad y género.** Quitar la **edad** reduce el rendimiento un **3,79 %**, lo que
indica que es informativa. Quitar el **género** con zero-shot **mejora** un
**2,37 %**. Los autores concluyen que el género no aporta y que los modelos
podrían diseñarse para evitar sesgos por género, mientras la edad sí debe
incorporarse.

**Conclusiones.** (i) El **tamaño del modelo no correlaciona siempre** con la
accuracy: GPT-3.5-Turbo rinde de forma comparable a GPT-4, y Gemini-Pro supera a
GPT-4 en WESAD; importan más las características del dataset y la arquitectura.
(ii) La edad influye y debe incluirse. (iii) El género no influye de forma
significativa. (iv) Ciertos biomarcadores pueden eliminarse sin pérdida, lo que
abarata el despliegue.

**Límites para nuestra tesis.** La comparación es de **accuracy y MAE**: no se
manipula la calidad de la señal, no se mide la confianza declarada, no se
verifican afirmaciones y no hay abstención. La conversión de WESAD a binario y el
remuestreo a 20 Hz son decisiones propias de los autores; nuestro diseño conserva
las ventanas de 60 s y la escala centrada por sujeto. Su hallazgo del **colapso a
«estrés» en zero-shot** es exactamente el patrón que observamos en el 7b con
cuatro clases, y su mejora con few-shot respalda nuestro uso de ejemplares de
desarrollo (S2–S4).
---

## 3. `pubmed_150` — Estrés como anomalía personalizada con UniTS y explicaciones LLM

**Título original:** *Seamless monitoring of stress levels leveraging a
foundational model for time sequences* (Artificial Intelligence in Medicine,
2026).

**Ficha.** Datasets: **DREAMER, MAHNOB-HCI, WESAD** (ECG y BVP) y **@HOME**
(pacientes mayores con deterioro cognitivo, en su domicilio). Tamaño del texto:
8.459 palabras. Relevancia SLIP: 0,76.

**Por qué es relevante.** Es el trabajo que mejor articula la motivación de esta
tesis: el ECG clínico es invasivo y limitado, y **los wearables degradan la
señal**. Su solución es un modelo fundacional de series temporales (UniTS) que
detecta estrés como **anomalía personalizada**, con un LLM que **explica** los
resultados. Nos interesa por dos motivos: demuestra robustez frente a
submuestreo y ruido, y evalúa las explicaciones del LLM contra un clínico. El
LLM no decide: explica.

### Texto
**Resumen.** El estrés agrava enfermedades neurodegenerativas como Alzheimer y
Parkinson y su seguimiento ayuda a ajustar el tratamiento. Los métodos basados en
ECG son invasivos y de duración limitada; los wearables son cómodos pero **su
señal se degrada** frente a la de grado clínico. El artículo propone monitorizar
el estrés con **UniTS**, un modelo fundacional de series temporales, reformulando
la detección como **anomalía personalizada**: se establece una línea base por
paciente y se marcan las desviaciones. Un LLM añade explicaciones legibles para
el clínico.

**Contexto clínico.** En pacientes con deterioro cognitivo el estrés aparece por
cambios de ánimo (depresión, apatía), confinamiento, aislamiento social,
incertidumbre sobre la evolución y factores económicos y familiares. Medirlo
permite entender la progresión y ajustar planes; también mejora la calidad de
vida. Las técnicas habituales incluyen voz y vídeo, ECG, EEG y dispositivos
implantables: dan información más precisa y continua, pero implican coste,
incomodidad, privacidad y procedimientos manuales.

**El problema de la calidad de la señal.** El ECG es invasivo y limitado en el
tiempo; los wearables ligeros, aunque prácticos, **pierden calidad**. El reto es
un sistema robusto, de largo plazo y amable para el paciente, que además ofrezca
interpretaciones accionables y se adapte a la variabilidad individual.

**Formulación.** En lugar de clasificar estados con un umbral fijo, se aprende el
comportamiento fisiológico **típico de cada paciente** y se detectan desviaciones
contextuales. Esto convierte el problema en **detección de anomalías
personalizada**, lo que permite adaptarse a patrones individuales y a la deriva
temporal sin reetiquetar.

**Modelo y ajuste.** Se parte de **UniTS**, un modelo fundacional multitarea para
series temporales, y se **ajusta** para detección de anomalías contextuales. El
sistema se integra después con un LLM que redacta explicaciones estructuradas de
cada anomalía detectada.

**Datos y evaluación.** Se evalúa en tres datasets de referencia —**DREAMER**,
**MAHNOB-HCI** y **WESAD** (con ECG y con BVP)— y en **@HOME**, un pilotaje
naturalista de pacientes mayores con deterioro cognitivo monitorizados en su
domicilio con wearables de consumo. La comparación es contra **12 métodos** de
detección de anomalías del estado del arte.

**Resultados.** UniTS **supera a los 12 métodos** y alcanza un **F1 medio de
0,821**. Con wearables de bajo coste rinde de forma **comparable a las señales
clínicas** (ECG), lo que habilita la monitorización no invasiva. Mantiene el
rendimiento al **reducir la frecuencia de muestreo**, un escenario habitual por
consumo y hardware. En @HOME se confirmaron clínicamente **30 de 32 anomalías**,
lo que indica precisión en un entorno no supervisado.

**Validación de las explicaciones.** Un profesional sanitario y GPT-4o puntuaron
significancia y criticidad de las explicaciones en escala 0–5. El **error medio**
de GPT-4o fue **0,58 ± 0,46** en significancia y **0,43 ± 0,39** en criticidad;
por dataset, el error baja en @HOME (0,00–0,21) y sube en los controlados
(DREAMER 1,25; MAHNOB-HCI 1,13; WESAD-ECG 0,75; WESAD-BVP 0,63). La
**especificidad** de las explicaciones en @HOME fue 3,90 ± 0,54 (hipertensión),
3,50 ± 0,67 (estrés) y 3,00 (calidad del sueño): el contexto disponible mejora la
explicación.

**Aportaciones.** Teóricas: nueva formulación como anomalía personalizada, primera
adaptación de un modelo fundacional multitarea de series temporales a monitorización
fisiológica y demostración de transferencia. Prácticas: robustez con señal ruidosa
y submuestreada, explicaciones legibles y despliegue real con alertas. Empíricas:
mejor rendimiento que 12 métodos y validación clínica de la mayoría de anomalías.

**Límites para nuestra tesis.** El LLM **explica, no decide**: no hay decisión
clasificatoria que evaluar, ni confianza declarada, ni verificación de cada
afirmación, ni abstención. La degradación se **observa** (wearables peores,
submuestreo) pero no se **induce** de forma controlada. Es, aun así, la mejor
referencia para nuestra motivación: la pérdida de calidad de la señal es el
problema de partida, y su solución pasa por un modelo robusto más una capa
lingüística evaluada contra un experto.
---

## 4. `arxiv_51` — Personalización con HRV y descripciones de LLM (BLP)

**Título original:** *Beyond Latent Personalization: Integrating HRV-Conditioned
Personalization via LLM for Interpretable Stress Detection* (2026).

**Ficha.** Datasets: **TILES-2018** (212 trabajadores sanitarios, 185 tras
filtrado; 10 semanas) y **WESAD** (15 participantes). Señal: HRV (61 features en
4 dominios). Tamaño del texto: 5.117 palabras. Relevancia SLIP: 0,75.

**Por qué es relevante.** Es el mejor argumento bibliográfico para dos decisiones
nuestras: **la representación importa** (H3) y **normalizar por sujeto** mejora
la clasificación (E12). Además usa el LLM como *traductor* de señal a lenguaje,
igual que nuestro paso `represent`, no como clasificador final. Sirve para
justificar la personalización y para diferenciar «LLM que decide» de «LLM que
describe».

### Texto
**Resumen.** La variabilidad de la frecuencia cardiaca (HRV) se usa para
detectar estrés, pero su interpretación depende de atributos personales (edad,
género, turno). Los enfoques personalizados actuales integran esos atributos en
**representaciones latentes**, lo que mejora el rendimiento pero **oculta cómo
interactúan** con la fisiología. Los autores proponen **BLP** (Beyond-Latent
Personalization): un LLM genera descripciones específicas por dominio fisiológico
que integran HRV y atributos personales, y esas descripciones se fusionan con la
señal temporal para clasificar. En TILES-2018 logra una **mejora relativa del
29,9 % en MCC** frente al mejor baseline.

**El problema.** La HRV es un indicador consolidado de estrés, pero responde de
forma distinta según la persona y el contexto laboral. Los métodos previos alinean
HRV y encuestas con aprendizaje contrastivo, regularizan embeddings con
similitudes de atributos o modelan interacciones de orden superior con hipergrafos.
Son eficaces, pero codifican la interacción en un espacio latente y **no explican**
cómo un atributo relativamente estático contextualiza una señal dinámica.

**Por qué un LLM.** Los LLM pueden expresar contexto en lenguaje: se han usado
para integrar texto clínico en tareas posteriores y para alinear señales
fisiológicas de alta dimensión con lenguaje. Esa capacidad permite **describir**
la relación entre atributos y fisiología en lugar de comprimirla en un vector
opaco.

**BLP en cuatro bloques.** (1) **Representación de HRV**: las features de cada
sujeto se ordenan temporalmente, se añaden codificaciones posicionales y un
Transformer modela dependencias; un proyector las lleva al espacio de fusión.
(2) **Interpretación semántica con LLM**: se construyen *prompts* por dominio
fisiológico —tiempo, frecuencia, RR multiescala y dRR multiescala— con el perfil
basal del sujeto, el contexto de interpretación y los marcadores del día; el LLM
produce una descripción por dominio **sin predecir la etiqueta**. (3)
**Representación semántica**: las descripciones se codifican con un modelo de
*embeddings* (1024 dimensiones), se proyectan por dominio y se agregan con pesos
aprendidos; una **pérdida de ortogonalidad** reduce redundancia entre dominios.
(4) **Fusión y clasificación**: atención cruzada bidireccional entre señal y
semántica, concatenación y MLP.

**El prompt.** El perfil basal es una plantilla con edad, género, formación,
puesto, turno, horas, antigüedad, estado civil, ingresos, hijos y desplazamiento.
El contexto de interpretación explica que las métricas son **agregados diarios**
extraídos de ventanas de 5 minutos de intervalos NN y qué significa cada sufijo
(media, desviación, coeficiente de variación, asimetría, curtosis, cuartiles). Las
restricciones de salida son estrictas: un solo párrafo, interpretación
fisiológica y psicológica, **sin decir si la persona está estresada**, sin repetir
los números y usando descripciones cualitativas.

**Entrenamiento.** Se combina entropía cruzada con una **pérdida intra-sujeto
multi-similitud** que separa estrés y no estrés dentro del mismo participante, y
con la pérdida de ortogonalidad entre dominios; sendos pesos escalares controlan la contribución de cada término auxiliar.

**Datos y features.** **TILES-2018** (212 trabajadores sanitarios, 185 tras el
filtrado de calidad; 10 semanas) y **WESAD** (15 participantes; baseline, estrés y
amusement). Ambas usan las mismas **61 features de HRV** en cuatro dominios,
extraídas de segmentos RR de 5 minutos con 4 minutos de solape y agregadas por día
con 13 estadísticos.

**Resultados.** BLP **supera a todos los baselines** en TILES-2018 (MCC 0,261;
**+29,9 % relativo**) y en WESAD confirma la generalización. La ablación verifica
que HRV, atributos personales, interpretación semántica y extracción de embeddings
aportan de forma complementaria. Los pesos por dominio indican qué familia de
features contribuye más a la decisión, y las descripciones hacen legible la
relación entre contexto y fisiología.

**Límites.** El marco **sigue dependiendo de la representación de HRV**: las
representaciones semánticas por sí solas no son suficientemente discriminativas.
No hay degradación de señal, ni incertidumbre, ni abstención. Las etiquetas de
TILES-2018 provienen de encuestas y las de WESAD de protocolo.

**Relación con esta tesis.** Es la mejor evidencia de que **la representación y la
normalización por sujeto cambian el resultado** (H3; nuestro E12) y de que el LLM
puede actuar como **traductor de señal a lenguaje** en lugar de clasificador. Su
límite —la semántica sola no basta— coincide con nuestra observación de que el
texto es una vista útil pero no superior en todas las condiciones.
---

## 5. `scopus_293` — Revisión de LLM para series temporales médicas

**Título original:** *A survey on large language models for medical time series*
(2025).

**Ficha.** Datasets revisados: **MIMIC-III/IV, eICU-CRD, HiRID, AmsterdamUMCdb,
OhioT1DM, VitalDB, WESAD, PPG-DaLiA, TROIKA, QT Database, MIT-BIH Arrhythmia,
Apnea-ECG, Sleep-EDF, CHB-MIT** y el *benchmark* PhysioNet. Señal: ECG, EEG,
PPG/BVP, signos vitales y wearables. Tamaño del texto: 14.409 palabras.
Relevancia SLIP: 0,75.

**Por qué es relevante.** Es el mapa del campo y la cita natural para la sección
de trabajos relacionados: sitúa a WESAD entre los datasets de referencia y ordena
las tareas en **decisión semántica** (salida de texto, la nuestra) frente a
**generación/reconstrucción** (salida de serie temporal). También resume las
métricas y los protocolos de validación, lo que ayuda a justificar nuestra
evaluación por condición y el uso de una referencia *features-only*.

### Texto
**Resumen.** Los LLM están transformando el dominio médico por su
generalización y su capacidad de modelar patrones complejos. La revisión cubre
los **LLM multimodales para series temporales médicas**: fundamentos,
metodologías y aplicaciones clínicas, con una taxonomía doble por **tipo de
tarea** y por **modalidad de señal**.

**Antecedentes.** En texto clínico, ClinicalBERT y BioBERT mostraron ventaja en
historias clínicas electrónicas, minería de literatura y consulta automatizada.
En pregunta-respuesta visual médica, los sistemas multimodales basados en
Transformers visuales combinan imagen y texto para apoyar el diagnóstico. Más
allá, se han explorado modelos que integran EHR, genética, imagen y series
temporales como base de la medicina de precisión.

**Por qué series temporales.** Las señales fisiológicas (ECG, EEG) son centrales
en diagnóstico, predicción y gestión de la salud, con tareas de clasificación,
predicción, detección de anomalías y segmentación semántica. Los métodos clásicos
—ARIMA con transformada wavelet, CWT para descomposición multiescala— dependen de
una **estacionalidad fuerte** y de ingeniería de features experta, y sufren con la
no estacionalidad, el ruido y los eventos abruptos de las señales médicas. Ese es
el hueco que ocupan los modelos temporales profundos y, después, los LLM.

**Fundamentos.** Se revisan arquitecturas y estrategias de preentrenamiento de
modelos temporales generales: **extensiones temporales de Transformers** y
**aprendizaje autosupervisado**, que son la base sobre la que se construyen las
adaptaciones al dominio médico.

**Datos y taxonomía.** Integra los principales conjuntos —**MIMIC-IV, VitalDB y
PhysioNet**, entre otros— y propone una **clasificación doble**: por tipo de tarea
(**decisión semántica con salida de texto** frente a **generación o
reconstrucción con salida temporal**) y por modalidad de señal. Analiza
diferencias de frecuencia de muestreo, tipo de señal y calidad de anotación, y
construye un **mapeo datos-tarea** que sirve de referencia estandarizada para
comparar entrenamientos.

**Método.** Examina cuatro aspectos: (i) elección del **backbone** y su
adaptabilidad; (ii) **diseño de arquitectura**, con innovaciones para la
interacción semántica entre señal y texto (mecanismos de atención y proyección de
features); (iii) **mecanismos de alineación**, como aprendizaje contrastivo e
ingeniería de prompts; y (iv) **diseño de la función de pérdida**, con
equilibrado en aprendizaje multitarea y optimización conjunta para predicción
diagnóstica y generación de informes.

**Evaluación.** Resume métricas y protocolos de validación, con atención a la
**generalización** y a la efectividad clínica, y compara *pipelines* en escenarios
clínicos representativos.

**Retos.** En datos: estandarizar fuentes heterogéneas, mitigar el desbalance de
clases y proteger la privacidad. En modelo: eficiencia computacional,
interpretabilidad y generalización fuera de distribución. En clínica: validación
de fiabilidad e integración con los sistemas sanitarios. Propone soluciones como
entrenamiento con privacidad (aprendizaje federado) y optimización de eficiencia
con grafos de conocimiento.

**Relación con esta tesis.** La revisión ordena el campo y **sitúa a WESAD** entre
los datasets de referencia. Su taxonomía nos sirve para justificar que nuestra
tarea es de **decisión semántica** (salida de texto) y no de generación de señal.
Lo que **no** cubre es el eje de esta tesis: no hay degradación controlada de la
evidencia, ni medición de la adecuación de la incertidumbre, ni verificación del
respaldo de cada afirmación.
---

## 6. `arxiv_288` = `manual_2` — Cuando las mediciones derivadas engañan (DFOT)

**Título original:** *When Derived Measurements Mislead: Quantifying and
Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence* (2025).

**Ficha.** Dataset: **MIMIC-III Matched Waveform Database v1.0** — 50.000
registros PPG–ECG sincronizados de 1.275 pacientes; test bloqueado de 187
pacientes. Señal: PPG y ECG. Tamaño del texto: 9.277 palabras. Relevancia SLIP:
0,76.

**Por qué es relevante.** Es el **competidor conceptual** más cercano: formaliza
el *derived-feature over-trust* (DFOT), el fallo por el que un LLM trata una
medición derivada como un hecho directo. Coincide con nuestra tesis en señal
fisiológica, fiabilidad de la evidencia, sobreconfianza e intervenciones
controladas. Se diferencia en el mecanismo: ellos **inyectan evidencia de
fiabilidad**; nosotros **degradamos físicamente la señal**. Su contraste entre
evidencia emparejada y barajada es el antecedente directo de nuestros controles
negativos (`NC_shuffled`).

### Texto
**Resumen.** Muchas arquitecturas actuales son modulares: un modelo aguas
arriba convierte la observación en una **medida derivada** y un LLM posterior
razona sobre ella. La modularidad mejora la accesibilidad, pero puede transformar
una estimación incierta en un **token con apariencia de hecho**. Los autores
definen *derived-feature over-trust* (**DFOT**) como el fallo en el que el modelo
asigna a esa medida el estatus de hecho directo o la usa fuera de su ámbito
válido.

**El problema, con precisión.** En muchos despliegues el LLM **solo ve** la medida
derivada: no accede a la medición original, ni a las condiciones de adquisición,
ni a evidencia sobre si la estimación es fiable para ese caso concreto. El fallo
resultante es distinto de una alucinación y de un error de predicción: el modelo
puede razonar de forma coherente con lo que recibe y, aun así, equivocarse porque
concede a la estimación un estatus epistémico que no tiene. Es un problema **de
interfaz**, no de una modalidad concreta.

**Por qué no basta con la calibración.** La calibración probabilística pregunta si
la confianza del modelo aguas arriba refleja su acierto empírico. DFOT pregunta si
el decisor aguas abajo **usa adecuadamente** el valor derivado en ese caso. La
predicción selectiva tampoco lo cubre: aquí la incertidumbre **nace en otro
canal** y debe sobrevivir a una interfaz semántica entre modelos distintos.

**Diseño experimental.** Dos escenarios: **D1** mide la aceptación de un ritmo
derivado del PPG que el ECG contradice; **D2** mide el rechazo de un ritmo PPG
fiable confirmado por ECG cuando el historial clínico es engañoso. El **ECG nunca
se muestra al LLM**: aporta supervisión y referencia offline.

**Cadena de cinco estimandos.** (1) **COTR**, tasa de sobreconfianza en conflicto
(D1); (2) **CIR**, tasa de error inducida por contexto (D2); (3) **CRR**, tasa de
reparación correcta del error congelado; (4) **ESRM**, margen de reparación
específico de la evidencia, que contrasta evidencia **emparejada** con evidencia
**barajada y disjunta por paciente**; (5) **UHR**, tasa de daño por utilidad, que
cuenta verificaciones innecesarias en casos de alta fiabilidad que en la línea
base se usaban sin verificar.

**Datos.** **50.000 registros PPG–ECG emparejados** de MIMIC-III *matched
waveform* (1.275 pacientes; test bloqueado de 187). El protocolo está bloqueado
antes del análisis.

**Resultados.** La intervención **K2** (evidencia de fiabilidad destilada de la
modalidad privilegiada) mejora **CRR, ESRM y especificidad entre 1,82 y 6,69
puntos porcentuales**; el **UHR sube 0,67 pp**; los intervalos de confianza
pareados **excluyen el cero** en los cuatro endpoints de reparación. K2 mejora
además el *ranking* de fiabilidad, y el contraste entre evidencia emparejada y
barajada distingue **reparación del caso concreto** de una cautela genérica. Los
autores reconocen margen para métodos de mitigación más fuertes.

**Conclusiones.** El marco es reproducible y discriminativo entre pacientes,
familias de LLM y formulaciones de prompt, y se declara aplicable a cualquier
arquitectura que razone sobre medidas derivadas. Deja abierta una agenda de
investigación sobre *pipelines* medición→LLM más fiables.

**Diferencias con esta tesis.** DFOT **inyecta evidencia de fiabilidad**; nosotros
**degradamos físicamente la señal** y observamos qué hace el modelo sin ayuda
externa. Ellos miden la confianza del decisor sobre medidas derivadas; nosotros
medimos cinco dimensiones (corrección, confianza declarada, certeza lingüística,
respaldo de claims y abstención) y separamos degradación de resolubilidad. Su
contraste emparejado/barajado es, aun así, el antecedente directo de nuestros
controles negativos (`NC_shuffled`) y su idea de «evidencia sobre la evidencia»
inspira nuestra H6.
---

## 7. `manual_5` — Degradación de telemetría y estabilidad secuencial

**Título original:** *Telemetry degradation and sequential stability in
LLM-integrated IoT decision systems*.

**Ficha.** Dataset: **telemetría sintética de smartwatch** (40 features IMU,
ventanas de 20 golpes) en un caso de entrenamiento de tenis con IA. Tamaño del
texto: 15.422 palabras. Relevancia SLIP: 0,72.

**Por qué es relevante.** Es el **competidor metodológico** más próximo a H1 y
H3: degrada la entrada de forma controlada y mide cómo cambia la conducta del
sistema. Su hallazgo de que el **submuestreo sistemático perturba más que el
dropout aleatorio** anticipa nuestro contraste entre huecos (`C2`, features
`null`) y valores imposibles (`C4`, canal a cero), que producen reacciones
distintas del modelo. Su métrica *FlipRate* es el antecedente de medir la
**tasa de cambio de decisión** entre condiciones, que podríamos calcular sobre
nuestro corpus.

### Texto
**Resumen.** Los sistemas de decisión con LLM convierten telemetría IoT en
salida semántica accionable. Faltan evaluaciones de **estabilidad conductual**
cuando los datos se degradan. Los autores proponen un marco de estabilidad
local-global y lo validan con un caso de **entrenamiento de tenis con IA**:
secuencias de golpes registradas con smartwatch (40 features IMU, ventanas de 20
golpes).

**Motivación.** Los LLM se despliegan en sistemas ciberfísicos e IoT donde
consumen *streams* estructurados y producen salidas que influyen en el mundo
físico. Hay mucha evaluación de razonamiento y de robustez estática del prompt,
pero poca sobre cómo se comporta la **política de decisión** con telemetría
degradada. En salud digital, rehabilitación, deporte o *Living Labs* (como el
centro UKDRI), la salida del LLM no es un resumen: es **guía accionable**. En
operación real hay pérdida de paquetes, reducción de resolución e inestabilidad
de transporte, que alteran el contenido informativo antes de llegar al decisor.
La inestabilidad no siempre se manifiesta como error semántico: puede aparecer
como **volatilidad** entre salidas sintácticamente válidas y plausibles.

**Perturbaciones.** Tres familias: **dropout graduado**, **reducción de
resolución** (submuestreo) y **señalización de fiabilidad** (se informa al modelo
de la calidad). El objetivo es mapear perturbaciones locales en cambios globales
de la política.

**Hipótesis locales (¿la degradación perturba decisiones individuales?).**
**HL1** estocasticidad intrínseca: el FlipRate es distinto de cero con telemetría
ideal, luego hay variabilidad de base ajena a la degradación. **HL2** amplificación
local: la volatilidad **crece de forma monótona** con la severidad. **HL3**
estructura de la ausencia: a igual pérdida nominal de información, el
**submuestreo sistemático** perturba más que el **dropout aleatorio**, por la
relación entre regularidad temporal y estabilidad de la salida.

**Hipótesis globales (¿las decisiones locales se componen en una política
coherente?).** **HG1** ganancia por régimen: los regímenes balanceados amortiguan
la propagación; los dominantes la amplifican. **HG2** interacción con la
estocasticidad: una temperatura de decodificación mayor amplifica la
inestabilidad de trayectoria, porque la incertidumbre de muestreo y la de entrada
se componen. **HG3** amortiguación por interfaz: el prompting estructurado
restaura parte de la estabilidad, pero la brecha local-global persiste con
suficiente degradación; las restricciones de interfaz **amortiguan**, no corrigen.

**Métricas.** **FlipRate** (volatilidad temporal de la decisión), escalada
direccional y resultados acumulados de seguridad (violaciones). Se evalúan además
los efectos de la temperatura y del prompting estructurado.

**Resultados.** En régimen **balanceado**, el FlipRate pasa de **0,144 en limpio a
0,207 con submuestreo (+44 %)**; en régimen **dominante** alcanza **0,300**. La
tasa de violaciones (ViolationRate) es 0 en el modelo de lazo abierto. La
**señalización de fiabilidad** produce efectos menores que la pérdida estructural
de información: las pistas informativas **no compensan** la telemetría perdida. El
prompting estructurado mitiga parcialmente la inestabilidad, sin eliminarla. Los
regímenes con una feature claramente dominante son **más sensibles** a la pérdida.

**Conclusiones.** La degradación **aumenta la inestabilidad** de sistemas que ya
eran inestables, y parte de la variabilidad secuencial proviene del **muestreo
estocástico** del modelo, no solo de la entrada. Los autores defienden evaluar la
estabilidad conductual **junto con** la fiabilidad del sensado y la estructura del
régimen.

**Diferencias con esta tesis.** Es el competidor metodológico más cercano, pero su
variable es la **estabilidad decisional**, no la corrección. No mide confianza
declarada, ni hedges/boosters, ni respaldo de afirmaciones, ni abstención. Su
hallazgo de que el **submuestreo perturba más que el dropout** anticipa nuestro
contraste entre `C2` (features `null`) y `C4` (canal a cero), donde el modelo
reacciona de forma distinta; y su **FlipRate** es el antecedente directo de una
métrica que podemos calcular sobre nuestro corpus: la tasa de cambio de decisión
de cada ventana entre condiciones.
---

## 8. `manual_3` — PhysioBench

**Título original:** *PhysioBench: A Unified Benchmark for Physiological Signal
Question Answering* (2026).

**Ficha.** Datasets: **22 conjuntos públicos armonizados** (ECG 23,19 %, PPG
0,15 %, EEG 1,90 %, PSG 74,76 %); **61,4 millones de preguntas**, 30 tareas,
240.171 sujetos. Tamaño del texto: 10.497 palabras. Relevancia SLIP: 0,72.

**Por qué es relevante.** Define el **techo actual** del razonamiento sobre
señales fisiológicas y demuestra que ningún modelo es robusto entre modalidades y
tareas, con rendimiento sensible a la **formulación** de la pregunta. Eso apoya
nuestro eje de representación (H3) y sitúa el listón: nuestro trabajo no compite
en cobertura de tareas, sino en medir la **sensibilidad a la evidencia**. Su
diseño no manipula la degradación ni pide abstención o claims.

### Texto
**Resumen.** Las señales fisiológicas sirven para tareas clínicas y de
monitorización muy diversas, pero los modelos fundacionales de señal suelen
necesitar **adaptación supervisada por tarea**. El lenguaje natural ofrece una
interfaz común para especificar objetivos, pero no se había evaluado bien si los
modelos actuales siguen esas instrucciones entre modalidades. **PhysioBench**
armoniza anotaciones de **22 datasets públicos** en **61,4 millones de preguntas**
y **30 tareas**; cada par pregunta-respuesta está anclado a un segmento de señal y
es **trazable** a su anotación de origen.

**Construcción.** Las preguntas se derivan de **plantillas específicas por tarea**
y las respuestas de las **anotaciones estructuradas** de origen. Ese diseño
permite escalar a decenas de millones de pares conservando la trazabilidad, a
cambio de una diversidad lingüística limitada (los propios autores lo señalan y
proponen en el futuro preguntas redactadas por expertos y formulaciones
alternativas).

**Cobertura.** Predominan las tareas de **PSG (74,76 %)**, seguidas de **ECG
(23,19 %)**, **EEG (1,90 %)** y **PPG (0,15 %)**; en total, 240.171 sujetos. La
distribución no es un defecto del benchmark, sino un reflejo de la literatura
disponible, y condiciona la comparación: las conclusiones sobre PPG descansan
sobre una fracción pequeña del total.

**Evaluación.** Se prueban **21 modelos** representativos en **tres ajustes
complementarios**: LLM, modelos visión-lenguaje, modelos de series temporales con
lenguaje y modelos fundacionales de señal fisiológica.

**Resultados.** **Ningún modelo** logra un rendimiento consistentemente fuerte
entre modalidades y tareas. La incorporación del lenguaje permite una predicción
unificada, pero el rendimiento **sigue siendo sensible a la formulación** de la
pregunta. Los modelos con representaciones fisiológicas preentrenadas superan el
azar, pero su transferencia es **desigual por tarea**; los modelos habilitados
para lenguaje muestran una comprensión **zero-shot limitada** de la señal. Los
retos que quedan son la representación heterogénea de señales, el aprendizaje con
tareas desbalanceadas y una alineación señal-texto robusta.

**Conclusiones y continuidad.** El benchmark se plantea como plataforma
extensible: más modalidades, más datasets, tareas más diversas y respuestas más
ricas (explicaciones clínicamente fundamentadas).

**Relación con esta tesis.** Es el **techo de comparabilidad** y una advertencia
metodológica: si la formulación de la pregunta cambia el resultado, la
**representación** es un factor experimental, no un detalle (nuestra H3). Su
diseño, sin embargo, no manipula la calidad de la evidencia ni pide al modelo que
reconozca sus límites: mide comprensión y predicción, no fidelidad ni abstención.
---

## 9. `arxiv_7` — Revisión de LLM para series temporales médicas (MedTSLLM)

**Título original:** *Large language models in medical time series analysis*
(2025).

**Ficha.** Datasets mencionados: ECG, EEG, PPG, signos vitales y wearables;
PTB-XL, MIMIC-III/IV, entre otros. Tamaño del texto: 44.434 palabras (la revisión
más extensa del corpus). Relevancia SLIP: 0,74.

**Por qué es relevante.** Es el marco de referencia para la introducción y para
la taxonomía de *prompting*. Su propio planteamiento deja fuera lo que esta tesis
mide: evalúa arquitecturas, datos y aplicaciones, pero **no la sensibilidad del
modelo a la degradación de la evidencia** ni el respaldo de sus afirmaciones.
Sirve también para justificar por qué las métricas de NLP no equivalen a
corrección clínica.

### Texto
**Resumen.** Las series temporales médicas (ECG, EEG, PPG y signos vitales)
son centrales en diagnóstico y monitorización. La revisión sintetiza el trabajo
sobre **MedTSLLM** —LLM aplicados a series temporales médicas— cubriendo
arquitecturas, recursos de datos, *pipelines* de procesamiento, estrategias de
prompting y aplicaciones clínicas, además de los problemas de despliegue real.

**Punto de partida.** Frente a la imagen médica o el texto clínico, las series
temporales son **más continuas, dinámicas y multicanal**, y la información
clínica está en la morfología de la onda, las dependencias temporales, las
interacciones entre canales y la variación entre pacientes. Los LLM aportan
comprensión del lenguaje, seguimiento de instrucciones, integración de
conocimiento y generación de texto: capacidades que la interpretación clínica
necesita además del reconocimiento de patrones.

**Flujo de adaptación.** La revisión explica los conceptos básicos y el proceso
para llevar un LLM a una serie temporal: **tokenización**, **preentrenamiento** y
**ajuste fino**, con las decisiones que cada etapa implica.

**Estrategias de prompting.** Cuatro familias: **instrucción** (la tarea se
describe en lenguaje), **razonamiento estructurado** (se guía el proceso),
**guiado clínico** (criterios diagnósticos, escalas como el HEART score, umbrales
de onda o flujos de evaluación clínica) y ***soft prompting*** (vectores
entrenables que se anteponen a la entrada o se inyectan en capas intermedias).
El *soft prompting* es eficiente en parámetros y flexible, pero **pierde
interpretabilidad** porque los vectores no son legibles. Además se revisan
**plantillas de prompt** y su diseño por escenario.

**Aplicaciones.** Organiza seis escenarios clínicos: interpretación diagnóstica,
generación de informes, monitorización longitudinal, síntesis de señal y otros,
con atención a la formulación de la tarea, al papel funcional del LLM y a los
protocolos de evaluación.

**Evaluación.** Revisa métricas y protocolos, y subraya que las **métricas de
lenguaje no equivalen a corrección clínica**: una respuesta bien formada puede
ser clínicamente falsa.

**Retos.** Alucinación, escasez y heterogeneidad de datos, falta de *benchmarks*
adecuados, razonamiento mecanístico, coste computacional y regulación. La
revisión propone avanzar hacia modelos más fiables, fundamentados clínicamente y
útiles en la práctica.

**Relación con esta tesis.** Es el **marco de referencia** para la introducción y
la taxonomía de prompting, y su propio planteamiento deja ver el hueco: evalúa
**capacidades** (qué tareas resuelve el modelo), no **sensibilidad a la
evidencia** (qué hace cuando la evidencia empeora). No hay degradación
controlada, ni medición de la adecuación de la incertidumbre, ni verificación de
afirmaciones. Nuestra contribución se sitúa exactamente en ese eje, sobre un
dataset y unas condiciones que la revisión no cubre.
---

## 10. `scopus_68` — Señales cognitivas para mejorar modelos de lenguaje

**Título original:** *Integrating cognitive processing signals into language
models: a review of advances, applications, and future directions* (2025).

**Ficha.** Datasets: **ZuCo** (EEG + eye-tracking) y múltiples corpus de
*eye-tracking*. Señal: eye-tracking, EEG y fMRI. Tamaño del texto: 9.467
palabras. Relevancia SLIP: 0,74.

**Por qué es relevante.** Aporta la motivación «de arriba» de la tesis: usar
señales humanas para **anclar** el lenguaje, mejorar la alineación y **mitigar
alucinaciones**. No compite con el diseño experimental, pero justifica por qué
tiene sentido evaluar la fidelidad del lenguaje del modelo respecto a la
evidencia. Es la cita adecuada para la introducción, no para los resultados.

### Texto
**Resumen.** La integración de la neurociencia cognitiva en el procesamiento
del lenguaje ha crecido. La revisión cubre el uso de **señales cognitivas** —sobre
todo *eye-tracking* (ET)— para mejorar modelos de lenguaje y modelos multimodales.
Esas señales abordan dos problemas de fondo: la **escasez de datos** y el **coste
ambiental** del entrenamiento, y aportan aumentación eficiente, convergencia más
rápida y mejor **alineación humana**.

**Motivación.** Los Transformers y los LLM (BERT, GPT, LLaMA) han convertido el
lenguaje en un problema de predicción a gran escala, pero entrenar esos modelos
tiene un coste creciente: el Transformer original emitió del orden de **0,012
toneladas de CO2**, mientras que GPT-3 se estimó en **~502 toneladas**. A la vez
persisten problemas de alineación (seguir instrucciones, evitar daño) y de
**alucinación**. Enriquecer los datos con señales derivadas del usuario es una vía
para reducir requisitos de datos y mejorar la alineación.

**Cómo se integran las señales.** Se revisan trabajos que incorporan **ET y EEG**
en el entrenamiento: módulos paralelos con *scanpaths*, medidas de lectura
inyectadas en la atención de BERT, y arquitecturas como **CogAlign** (encoder
compartido entrenado de forma adversarial para transferir conocimiento cognitivo
a datasets sin señal) y **CogAware** (features entre dominios preservando lo
específico de cada modalidad). Muchos usan **ZuCo**, el corpus que combina EEG y
ET.

**Aplicaciones revisadas.** Comprensión lectora, atención, clasificación de
sentimiento, comprensión del lenguaje, modelado del lenguaje, alineación humana,
pregunta-respuesta, **VQA**, detección de alucinaciones, discurso de odio y
sarcasmo. En varias de ellas la señal cognitiva se usa como **señal auxiliar**
(atención, reordenación, máscara) más que como entrada principal.

**Resultados que destaca.** Las señales cognitivas permiten aumentar datos y
mejorar la convergencia; el ET es especialmente prometedor para VQA y para
**detectar o mitigar alucinaciones** en modelos multimodales, porque aporta
información sobre qué mira el usuario y cuándo se pierde.

**Retos.** Coste y disponibilidad de las señales, heterogeneidad entre sujetos y
dispositivos, privacidad, y la dificultad de generalizar el conocimiento cognitivo
a dominios sin registro.

**Relación con esta tesis.** Es una **motivación conceptual**, no un competidor:
sostiene que la evidencia humana (fisiológica o cognitiva) debe informar al
lenguaje y que el lenguaje puede desviarse de ella. No evalúa degradación, ni
abstención, ni respaldo de afirmaciones; su interés para nosotros está en la
introducción y en el argumento de que la fidelidad del lenguaje a la evidencia es
un problema abierto.
---

## Apéndice: dónde está el texto completo

Los notebooks son la fuente íntegra. Cada paper aparece con la cabecera
`TITULO DEL PAPER: <id>-<título>`; la posición es aproximada (carácter dentro del
fichero) y sirve para localizarlo con una búsqueda.

| Paper | Fichero | Posición | Palabras |
|:-----------------|:------------------------------------|----------:|---------:|
| `ieee_88` | `docs/SLIP/notebook/gemini_notebook_01.md` | 283.096 | 1.318 |
| `arxiv_288` (= `manual_2`) | `docs/SLIP/notebook/gemini_notebook_01.md` | 360.361 | 9.277 |
| `scopus_111` | `docs/SLIP/notebook/gemini_notebook_01.md` | 426.929 | 5.535 |
| `pubmed_150` | `docs/SLIP/notebook/gemini_notebook_01.md` | 547.105 | 8.459 |
| `arxiv_51` | `docs/SLIP/notebook/gemini_notebook_01.md` | 2.841.662 | 5.117 |
| `arxiv_7` | `docs/SLIP/notebook/gemini_notebook_01.md` | 2.910.513 | 44.434 |
| `scopus_293` | `docs/SLIP/notebook/gemini_notebook_02.md` | 284.495 | 14.409 |
| `scopus_68` | `docs/SLIP/notebook/gemini_notebook_03.md` | 1.408.001 | 9.467 |
| `manual_3` | `docs/SLIP/notebook/gemini_notebook_03.md` | 1.894.602 | 10.497 |
| `manual_5` | `docs/SLIP/notebook/gemini_notebook_03.md` | 2.018.266 | 15.422 |

Para localizar uno:

```bash
grep -n "TITULO DEL PAPER: ieee_88-" docs/SLIP/notebook/gemini_notebook_01.md
```

Los metadatos bibliográficos (DOI, año, venue, citas, `relevance_score`) están en
`docs/SLIP/papers_metadata.json`; la ficha de cada paper en el compendio de
datasets, en [`analisisDatasets.md`](analisisDatasets.md).
