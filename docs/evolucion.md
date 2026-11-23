# Evolución de la investigación — sinopsis por sesión

**Actualizado:** 2026-11-11

Este documento cuenta **cómo cambió el experimento** a medida que llegaban los
resultados: qué preguntas obligaron a reorientar el diseño, qué conclusiones se
descartaron y con qué evidencia se sostienen las actuales. No es un historial de
tareas (eso está en `AGENTS.md` y `SESSION_STATE.md`) sino la **línea
argumental** de la tesis.

**Cómo se mantiene** (`AGENTS.md` §0, regla 7): al cerrar **cada sesión**, se
añade una entrada breve en la sección [Registro de sesiones](#registro-de-sesiones)
con el formato *pregunta → hallazgo → decisión*. El relato de los actos solo se
reescribe si la línea argumental cambia de verdad.

---

## La línea argumental en cinco frases

1. Empezamos midiendo acierto y encontramos un sesgo aparente.
2. Al auditar el instrumento, el sesgo era un artefacto.
3. Al medir resolubilidad, el problema dejó de ser el dato y pasó a ser la
   lectura.
4. Al congelar el diseño, apareció el hallazgo real: **corrección y respaldo de
   afirmaciones se desacoplan**.
5. Al salir de WESAD, el patrón se repite y se le añade una vuelta de tuerca:
   **donde no hay evidencia, la certeza sube**.

---

## Acto 0 — Punto de partida

Hipótesis H1–H6 sobre si el LLM **dice lo que la evidencia respalda**, con WESAD
como banco de pruebas y la **degradación controlada** como palanca. Se construye
primero el corpus (705 ventanas de 60 s, features + SQI, representaciones
numérica y textual) y el preregistro de condiciones C0–C5.

→ [`hipotesis.md`](hipotesis.md) · [`ingestion.md`](ingestion.md) ·
[`experiment_degradation.md`](experiment_degradation.md)

## Acto 1 — El primer piloto y el espejismo del sesgo

46 ventanas de S3, 460 llamadas. Resultado aparente: **97,8 % de respuestas
«stress»**. La lectura fácil era «el modelo tiene sesgo». Antes de escribirla, se
revisa el instrumento.

## Acto 2 — Auditar el instrumento antes de creer el resultado

Seis fallos invalidaban las cifras: AUROC mal calculado (daba 2,0), `accuracy`
confundida con `accuracy_incl_abstain`, colisión de caché entre `NC_nan` y
`NC_absent`, controles negativos de texto mal aplicados, canales sin normalizar y
umbral SCR invertido (0,01 → **0,05 µS**). Con el instrumento corregido, el
«colapso» desaparecía: AUROC de stress 0,60 → **0,896**.

> **Reorientación:** ninguna conclusión se publica sin auditar la métrica. Lo
> medido antes queda como material de desarrollo.

## Acto 3 — ¿Falla el modelo o fallan los datos?

Patrón incómodo: el LLM **nunca predice `amusement` ni `meditation`** (0/705).
Antes de aceptar que «EDA+HR no bastan», se mide la **resolubilidad** con las
mismas 23 features: una logística separa las 4 clases (0,64–0,68), los pares
superan 0,86 y, controlando la fase del protocolo, todo sigue por encima de 0,72.

> **Reorientación:** la información **existe**; el fallo es de
> representación/razonamiento. Se adopta el **endpoint binario
> stress/no-stress** como primario, declarándolo decisión **pragmática**, no por
> falta de separabilidad.

→ [`resolubilidad.md`](resolubilidad.md) · [`split_dev_test.md`](split_dev_test.md)

## Acto 4 — Atacar la representación (E12)

Si el modelo juzga valores absolutos sin referencia, se le da la referencia: z
intra-sujeto. El binario mejora mucho (accuracy 0,930; AUROC stress 0,958), pero
**las 4 clases siguen colapsadas** y aparece un efecto colateral: con datos
barajados (`NC_shuffled`) la abstención cae de 95,3 % a 23,6 %.

> **Reorientación:** la representación centrada se queda para el binario, y el
> efecto colateral abre un problema nuevo: hace falta una comprobación de
> coherencia **externa al modelo**.

→ [`e12_resultados.md`](e12_resultados.md)

## Acto 5 — La abstención no se puede pedir por prompt

Se implementa la **compuerta determinista de consistencia** (8 identidades
fisiológicas): 0 % de vetos en limpio, 100 % en datos incoherentes. La ablación
con la regla escrita en el prompt es **negativa**: el modelo no ejecuta la
comprobación aunque pueda.

> **Reorientación:** la calidad se resuelve con código, no con instrucciones. La
> abstención pasa a ser un problema **de sistema** (LLM + compuerta), no del
> modelo.

→ [`consistencia.md`](consistencia.md)

## Acto 6 — Descubrir la fuga y congelar el diseño

Auditando el few-shot se encuentra una fuga: los ejemplos usaban medias de **las
705 ventanas evaluadas** (HR 72,4 bpm, EDA 3,9 µS). Se corrigen con medianas de
**S2–S4** y se congela el split (desarrollo S2–S4; evaluación S5–S17 salvo S12).

> **Reorientación mayor:** todo lo anterior pasa a **exploratorio** y el
> experimento se re-ejecuta: 5.670 respuestas (12 sujetos × 567 ventanas × 5
> condiciones × 2 representaciones).

→ [`split_dev_test.md`](split_dev_test.md)

## Acto 7 — El resultado que da valor (y el que refuta)

Con el diseño congelado llega la historia real:

- **H1 apoyada:** accuracy 0,611 → 0,263; la referencia cae menos → exceso de
  coste 0,15–0,20. EDA es la modalidad crítica.
- **H5, el hallazgo principal:** el respaldo de claims **sube** a 0,83 mientras
  la accuracy cae a 0,29. El modelo cita bien los valores y falla al combinarlos;
  `unsupported = 0`.
- **H2b apoyada:** la certeza lingüística baja (+0,13 → −1,12).
- **H3 apoyada:** el texto acierta menos (−0,09) pero degrada menos (interacción
  +0,065) y en degradación severa se vuelve **más asertivo** (+1,36).
- **H4 refutada:** infra-abstención (brecha error−abstención hasta +0,58) y
  sobre-abstención en limpio (12,5 %).
- **H6 matizada:** el SQI aporta (+0,05/+0,09); la compuerta no (mide integridad,
  no información).

> El valor no está en «el LLM acierta poco», sino en **disociar** corrección,
> confianza, certeza y respaldo: cada eje se mueve en una dirección distinta.

→ [`resultados_degradacion.md`](resultados_degradacion.md) ·
[`claims_h5.md`](claims_h5.md)

## Acto 8 — ¿Se sostiene fuera de WESAD?

**PhysioNet/CinC 2017** (otra tarea, otro sensor, clase `noisy` real): balanced
0,250 —azar exacto— frente a 0,552 de la referencia; **0/103 abstenciones** en
evidencia ilegible; 47,6 % de claims contradichos; y un matiz nuevo: la certeza
**sube** donde no hay evidencia (Cliff δ = +0,60) y la calidad **sí** deja rastro
en las features (AUROC 0,873). **K-EmoCon** se cierra como resultado negativo
documentado (sus etiquetas afectivas no son predecibles, AUROC ≤ 0,61).

> **Reorientación final:** el hallazgo se formula como **sensibilidad a la
> evidencia**, no como clasificación de estrés.

→ [`p3_physionet.md`](p3_physionet.md) · [`p3_kemocon.md`](p3_kemocon.md)

## Acto 9 — La infraestructura que lo sostiene

Comparar modelos exige aislar artefactos (`llm_responses/{modelo}/`), el CLI se
homogeneiza en tres fases (`INGEST` → `GENERATE` → `EVALUATE`) con el experimento
como parámetro, y la réplica en el servidor obliga a tres arreglos: **errores
silenciosos** (códigos de salida y traza en el log), **datos dentro del
proyecto** (`results/input/EXTERNAL/`) y **comprobación de entorno** (venv +
dependencias). No es ciencia, pero sin ello los resultados no son reproducibles.

→ [`fase2_diseno.md`](fase2_diseno.md) §8 · [`datos_entrada.md`](datos_entrada.md)

---

## Tabla de reorientaciones

| Reorientación | Disparador | Qué se cambió | Evidencia que la sostiene |
|---|---|---|---|
| Auditar antes de concluir | Sesgo 97,8 % en S3 | 6 correcciones de métrica y controles | AUROC 0,60 → 0,896 |
| Binario como primario | Nunca predice 2 clases | Endpoint stress/no-stress (desviación declarada) | 0/705; resolubilidad 0,64–0,68 |
| Representación centrada (E12) | Valores absolutos sin referencia | z intra-sujeto | Binario 0,930; 4 clases sin cambio |
| Compuerta determinista (P1) | E12 pierde sensibilidad a `NC_shuffled` | Comprobación de coherencia en código | 0 % → 100 % de abstención |
| Congelar el diseño | Fuga del few-shot | Split dev/test + re-ejecución | 5.670 respuestas |
| Foco en H5 | El respaldo sube mientras cae la accuracy | Claims como eje principal | 0,78 → 0,83 con accuracy 0,29 |
| Generalización | Resultados solo de WESAD | PhysioNet + K-EmoCon | 0/103 abstenciones; AUROC 0,873 de calidad |
| Segunda tarea, no segunda versión | K-EmoCon no valida (AUROC 0,61) y DREAMT no tiene estrés | Eje de tarea (sueño) con su propia matriz ausencia/degeneración | 100 % de abstención en `C4`; H4 25,0 % vs 98,8 % según representación |

---

## Registro de sesiones

Bitácora breve: una entrada por sesión con **pregunta → hallazgo → decisión**.
Las sesiones anteriores a la creación de este documento se resumen a partir de
`AGENTS.md`.

| Fecha | Pregunta / foco | Hallazgo | Decisión |
|---|---|---|---|
| 2026-11-01 | ¿Se puede construir el corpus? | Pipeline completo sobre 705 ventanas, SQI 0,934, corpus 0.1.0 | Fijar Fase 1 y avanzar a prompts |
| 2026-11-04 | ¿Cómo degradar de forma controlada? | `degrade` C0–C5; el confusor del movimiento discrimina (AUC 0,928) | Añadir canal de movimiento como factor y excluirlo de la vista LLM |
| 2026-11-05 | ¿Qué prompt y qué modelo? | qwen3:8b vaciaba `content`; esquema JSON y orquestador funcionan | Usar qwen2.5:7b y cachear por hash de prompt |
| 2026-11-06 | ¿Acierta el LLM? | Sesgo masivo a «stress»; controles negativos no se aplicaban | Corregir controles y añadir few-shot |
| 2026-11-07 | ¿Son fiables las métricas? | 6 bugs invalidaban las cifras (AUROC 2,0; SCR invertido…) | Rehacer evaluación; descartar conclusiones previas |
| 2026-11-08 | ¿Falla el modelo o el dato? | Resolubilidad 0,64–0,68 con las mismas features; LLM 0,468 | Binario como primario (pragmático); E12 centrado |
| 2026-11-09 | ¿Cómo mejorar el sistema? | E12 pierde sensibilidad a `NC_shuffled`; la regla en prompt no se ejecuta | Compuerta determinista (P1); split congelado por fuga del few-shot |
| 2026-11-10 | ¿Qué pasa al degradar? | H5: respaldo 0,78→0,83 con accuracy 0,29; H4 refutada; H3 texto degrada menos | Foco en H5; representación como factor |
| 2026-11-10 | ¿Generaliza? | PhysioNet: azar (0,250), 0/103 abstenciones, certeza invertida | Reformular como sensibilidad a la evidencia |
| 2026-11-11 | ¿Se puede replicar? | Faltaban datos internos, entorno y errores silenciosos | Datos en `results/input/EXTERNAL/`, preflight de entorno, Python 3.12 |
| 2026-11-11 | ¿Cómo contamos la evolución? | El relato estaba disperso en la memoria operativa | Crear este documento como **sinopsis obligatoria por sesión** (`AGENTS.md` §0, regla 7) |
| 2026-11-11 | Cierre: ¿está todo reproducible y documentado? | 339 tests en 3.12, 0 enlaces rotos, externos dentro del proyecto, fallos de arranque con mensaje y código de salida | Memoria consolidada en `AGENTS.md` §43 y `SESSION_STATE.md` |
| 2026-11-12 | ¿Por qué rehízo el corpus entero un commit de documentación? | El commit estaba en los 6 fingerprints: 14/15 sujetos recanonicalizados por 19 GB; verificación 5.670/5.670 prompts idénticos | Fingerprint por contenido de código (`code_fingerprint`), `features_sha256` por respuesta y `--noEmbeddings` |
| 2026-11-12 | ¿Cómo afecta tener otra GPU? | El log decía «No CUDA devices» con una A5000: el respaldo por `nvidia-smi` llamaba a una función inexistente | Detección con doble vía y sin ocultar la GPU; torch solo informa, la inferencia es de Ollama |
| 2026-11-12 | ¿Se puede fijar torch a la CUDA del servidor? | El canal `cu128` llega a torch 2.11.0; PyPI compila 2.11/2.15 contra CUDA 13.x | `torch` se queda en `requirements.txt` con el índice `cu128`; `gpu_report()` en log, `--dry-run` y manifiesto |
| 2026-11-12 | ¿Y si migramos a otro servidor? | No había forma de empezar de cero: el fingerprint se volvía un requisito en lugar de una facilidad | `--fresh --yes`: borra solo `results/output` y recalcula; copiar `results/input` basta |
| 2026-11-12 | ¿Sirve de algo el fingerprint nuevo? | Tras `--fresh --yes`, la segunda ejecución da `reused` en los 15 sujetos y las 4 representaciones | Fingerprints estables: la reutilización funciona y un commit no invalida el corpus |
| 2026-11-12 | ¿Qué pasa si Ollama no está? | El paso `represent` habría fallado tras reintentar por ventana; los embeddings son complementarios | Ollama caído ⇒ `embeddings: skipped` y la ingesta sigue con numeric/text |
| 2026-11-12 | ¿Cómo se lanza en el clúster sin depender de `~/data`? | Los scripts quedaban fuera del repo y el lanzador apuntaba a `run_slip_gpu.sh` | `scripts/run_ACL.sh` (GPU, Ollama acotado, código de salida) + `scripts/launch_lsi.sh`; documentado en `docs/ejecucion_slurm.md` |
| 2026-11-13 | ¿Se sostiene el sistema tras estabilizar el corpus? | Job `INGEST` COMPLETED 0:0 (3:32, 211 artefactos); siguiente run 100 % `reused` | Corpus y fingerprints cerrados; el trabajo pasa a la fase LLM |
| 2026-11-13 | ¿Responde un modelo mayor a la degradación? | Humo `qwen3:14b` (25 llamadas, 0 errores): abstiene en C2/C4/C5 y decide en C0/C1 | Lanzar el barrido completo de 14b (~5 h, `--time 12:00:00`) y comparar con 7b |
| 2026-11-14 | ¿Resiste mejor el modelo grande? | Con prompt binario en limpio empatan (Δ +0,014 [−0,012; +0,040]); con EDA a cero el 7b sobrerreclama estrés (bal 0,63) y el 14b dice non-stress para todo (bal 0,500, acc 0,845 = tasa base) | La ventaja del 14b en 4 clases era la abstención; la robustez no escala, solo cambia el modo de fallo |
| 2026-11-14 | ¿Por qué se abstiene el modelo? | El prompt binario incluye la misma regla de abstención y aun así abstención ≈0: `non-stress` absorbe la duda. En 4 clases, sin clase comodín, abstención 31,7 % (C0) y 96–100 % (EDA destruida) | La abstención es el **residuo del espacio de etiquetas**, no una auditoría de la evidencia |
| 2026-11-14 | ¿Es fiable la accuracy para medir robustez? | A cobertura comparable, el 14b «gana» en C5 con 0,845 mientras su balanced es 0,500: 0,845 es la tasa base de non-stress | Reportar siempre balanced y cobertura; la accuracy premia al clasificador degenerado |
| 2026-11-14 | ¿Se sostiene H2b? | Con la certeza separada por decisión: monótona en 7b-4c (+0,11→−0,99) y 14b-bin (−1,10→−3,04); plana en 14b-4c (el veto asertivo la enmascaraba) y 7b-bin | H2b depende de modelo × tarea; no es propiedad general del lector |
| 2026-11-14 | ¿Qué aporta el SQI? | ΔAUROC +0,163/+0,243 en el 7b binario (confianza constante, AUROC 0,50) y +0,043 en el 14b (confianza AUROC 0,79) | H6 reformulada: el indicador de calidad **sustituye** a una confianza que no informa |
| 2026-11-14 | ¿Qué papers compiten con la tesis y con qué dataset? | 14 competidores mapeados a H1–H6; ninguno combina las cinco dimensiones y **ninguno usa WESAD bajo degradación** (4 lo usan como dato, sin manipular la señal) | Tabla de competidores + posicionamiento: la novedad es la medición conjunta y la separación degradación/resolubilidad/información |
| 2026-11-14 | ¿Cómo justificar WESAD y qué aportan PhysioNet/K-EmoCon? | PhysioNet da la clase `noisy` real (0/103 abstenciones, certeza invertida, calidad AUROC 0,873); K-EmoCon no tiene etiqueta de estrés y su referencia no pasa de AUROC 0,61; DREAMT mejora la referencia externa (PSG) pero cambia de fenómeno | WESAD como vehículo principal; PhysioNet como control; DREAMT (o un dataset con inducción de estrés) como extensión |
| 2026-11-14 | ¿Podemos leer los papers en español sin perder el contenido? | El texto íntegro del corpus suma ~124k palabras (arxiv_7: 44k); una versión ampliada cubre ~4,5 % y 40 % en el paper corto | Documento de lecturas por lotes (2 papers/turno) o texto íntegro en inglés como alternativa |
| 2026-11-13 | ¿Desde dónde se lanzan los scripts? | Se documentaban como `./scripts/launch_lsi.sh`, pero en el servidor el despliegue vive en `~/data` y esa ruta no existe en el repo | Regla permanente (`AGENTS.md` §0.8): repo = fuente, `~/data` = despliegue; todos los comandos usan `~/data/launch_lsi.sh` |
| 2026-11-13 | ¿Se puede ejecutar `python main.py` directamente en el servidor? | No por defecto: el trabajo va por Slurm (GPU, Ollama, cola); solo lo pequeño (sin GPU/Ollama, minutos) cabe en la terminal SSH | Regla `AGENTS.md` §0.9 + tabla «qué se lanza y qué se corre a mano» en `docs/ejecucion_slurm.md` |
| 2026-11-16 | ¿La tarea binaria es la de Schmidt y qué papel juega la temperatura? | El prompt incluía `meditation` en `non-stress` (567) y la temperatura entraba en la evidencia (r=0,801 con la posición); restringido a las 3 clases de Schmidt, la distribución es 53,8/29,8/16,3 (paper: 53/30/17) | Fijar `schmidt` (424) sin temperatura como primario y 355/498/567 + con-temperatura como sensibilidades; bootstrap agrupado por sujeto; segunda desviación post hoc |
| 2026-11-17 | ¿Se sostiene el binario al evaluar 424 ventanas de Schmidt sin temperatura? | En limpio empatan (7b 0,908 / 14b 0,915, IC solapados) con perfiles opuestos (ECE 0,068 vs 0,125; AUROC 0,912 vs 0,964); con EDA a cero el 7b cae a 0,548 (tasa base 0,700) y el 14b predice estrés 1–3 veces de 424 (balanced 0,508) | El colapso del 14b es estructural (bal 0,522 en las 4 variantes) y su accuracy de 0,82 es artefacto de cobertura (acc* 0,70) |
| 2026-11-17 | ¿Dónde se abstiene el 14b y por qué? | 63,8 % de las ventanas de estrés en C2, 47,2 % en C4 y 50,4 % en C5, frente a 0,3–4,4 % de las de `non-stress`; el 7b no se abstiene nunca | La abstención del 14b es **dependiente de la etiqueta** (se retira cuando falta la señal que define la clase positiva), no ruido ni azar |
| 2026-11-17 | ¿El 7b mejora al fijar el diseño primario? | C5 pasa de balanced 0,613 / acc 0,495 (567, `all` + temperatura) a 0,723 / 0,682 (424, `schmidt` sin temperatura) | Parte del daño atribuido al 7b era de la temperatura y de las clases ajenas a la tarea; el diseño primario se separa del histórico en la documentación |
| 2026-11-17 | ¿El fallo bajo degradación se debe a falta de información? | En limpio el LLM **iguala a la referencia supervisada** con las mismas 424 ventanas y 19 features (0,908 / 0,915 vs 0,905); con EDA a cero el exceso de coste es +0,317 (7b) y +0,166 (14b) | H1 apoyada con más margen que en 567; el límite es del lector, no del dato |
| 2026-11-17 | ¿Se sostiene H5 en el diseño primario? | El respaldo sube con la degradación (7b 0,768→0,811; 14b 0,876→0,915), `unsupported` = 0 y dirección 0,81–0,95, mientras la decisión se hunde (0,45 de error en C4 del 7b) | H5 replicada: claims localmente ciertos, inferencia global falsa |
| 2026-11-17 | ¿Se sostiene H6 («el SQI sustituye a la confianza»)? | En 424 el ΔSQI del 7b cambia de signo (−0,139 en C0, +0,119 en C2, −0,042 en C5) y el del 14b es ≤ +0,112; el gate sigue en Δ = 0 | **H6 no se replica fuera del corpus de 567**: citarla como dependiente del corpus, no como propiedad del indicador |
| 2026-11-17 | ¿Resiste más el modelo grande en el diseño primario? | En limpio empatan (+0,007 [−0,024; +0,042]); en C2 +0,059 de accuracy pero −0,151 de `acc*` y −0,265 de balanced; en C4 el 14b gana +0,277 de accuracy y +0,158 de `acc*` pero **pierde −0,126 de balanced** (se abstiene en el 47 % de las ventanas de estrés); en C5 +0,145 / +0,014 / −0,224 | La comparación no es identificable sin una regla de coste: fijar balanced y `acc*` como métricas de decisión y reportar accuracy solo como descriptivo |
| 2026-11-17 | ¿Modula el modelo su lenguaje y su confianza? | 7b: confianza con **dos valores** (AUROC 0,50–0,54) y certeza no monótona (−0,58 → −0,31); 14b: AUROC 0,867 → 0,731 y certeza sobre decididas −0,25 → −0,85 | H2a y H2b **dependen del modelo**; el índice global del 14b está inflado por el texto del veto (−2,4 en abstenciones) |
| 2026-11-18 | ¿Está la respuesta estabilizada por la semilla? | Con 20 repeticiones: 22/30 ventanas no cambian nunca y **0 de las 9 de estrés**; las 8 inestables son `non-stress` y el cambio va siempre hacia `stress`. Una pasada: accuracy 0,648 ± 0,043 (rango 0,533–0,700) | **R=3 basta para el agregado** (±3 pp); no se lanza corpus completo de repeticiones |
| 2026-11-18 | ¿Mide la confianza la incertidumbre? | Una ventana sale **10/10** entre clases con **confianza 85 ± 0,0** en las 20 repeticiones; el 7b no se abstiene nunca, ni ahí | Repetir **revela** la indecidibilidad, no la arregla: la incertidumbre no está en el número que emite el modelo |
| 2026-11-18 | ¿Forma parte del experimento el motor de inferencia? | Mismo prompt y misma semilla: **0 abstenciones** en el servidor (Ollama 0.40.6) y **28** en local (0.30.7); accuracy 0,682 vs 0,614 | `run_provenance.json` con versión y digest; las cifras se citan solo del servidor y los runs locales no se documentan |
| 2026-11-18 | ¿Se puede analizar en el nodo de login? | `import numpy` aborta con `Illegal instruction` en `sirius`, también con un venv limpio y el wheel oficial; el mismo `.venv` funciona en los nodos de cálculo | Análisis y `--dry-run` por `srun`/lanzador; mensaje al administrador pendiente |
| 2026-11-18 | ¿La representación cambia el fallo bajo degradación? | El texto **duplica el coste**: −0,510 vs −0,226 (7b) y −0,442 vs −0,220 (14b). El 7b responde mal y con confianza (0,252 en C5, sin abstenerse); el 14b se retira (cobertura 0,480) y su `Δacc` +0,126 es artefacto de cobertura (en `acc*`, −0,239) | H3 apoyada con la **dirección invertida** respecto del corpus de 4 clases (+0,065); aquel resultado queda como exploratorio |
| 2026-11-18 | ¿Se sostiene el formato cuando falta la evidencia? | 155 inválidas del 7b textual (5,5 %; **140 en C4**) frente a 2 del 14b (0,07 %), todas por copiar `not available` en `direction`/`magnitude` | La recuperación tolerante rescata la decisión; el «claim de ausencia» no tiene categoría en el esquema y se declara como limitación, sin añadir un alias post hoc |
| 2026-11-19 | ¿Los claims son triviales porque el prompt ya da los valores? | No: 22 % de claims contradichos en limpio (1.853 en `C0`); cita 3,31 de 17,4 features (19 %) con percentil de \|z\| **61,6** (50 = azar, 100 = siempre el más extremo); con EDA a cero, el **68 %** de sus claims al decidir `stress` son ciertos y **contrarios** a la decisión | H5 se declara **cota inferior** (claim necesario, no suficiente) y se registra el eje de relevancia: verdad ≠ relevancia ≠ suficiencia |
| 2026-11-19 | ¿Qué falta para cerrar H5? | `claims.py` solo verifica verdad: un claim verdadero pero contrario a la conclusión cuenta como `supported`; ni relevancia, ni suficiencia, ni selección se miden | Implementar `claims_coherent_with_decision` + contraprueba de suficiencia en sesión nueva, con la tabla de firma **preregistrada** |
| 2026-11-19 | ¿Se respetó el split dev/test? | **En datos sí** (0 ventanas de S2–S4 en `degradation/` y `degradation_text/`); **en diseño no**: 6 decisiones en producción se tomaron con datos de evaluación y 3 no estaban declaradas (normalización z de E12 que usa E1, conjunto de 23 features, umbrales de la compuerta) | Declarar desviaciones **7–10** y matizar E1 como «firme en los datos, no en la independencia de su diseño»; **no** re-ejecutar (cambiaría el prompt de 5.670 respuestas) |
| 2026-11-21 | ¿Basta con medir la verdad del claim para sostener el hallazgo de H5? | No: el verificador daba por `supported` un claim **verdadero y contrario** a la decisión; con la firma preregistrada, la coherencia del lado `stress` cae de 0,677 [0,641; 0,713] en `C0` a 0,315 [0,285; 0,354] en `C4`, y `against_supported_share` sube de 0,270 a 0,737 (exploratorio, corpus de 4 clases) | Implementar el eje de relevancia (`signatures.py` + `claim_coherence`) con la tabla fijada antes de medir; `stable` neutro, celdas < 30 claims `insufficient` |
| 2026-11-21 | ¿La firma describe los datos que el LLM vio? | **P0 11/12** en `C0`; la única que falla es `resp_rate_bpm` (z −0,16 en `stress` vs +0,07) porque `meditation` (≈9 rpm) entra en `non-stress` | No se toca la tabla: el fallo se explica por la variante `all` y la jerarquía Schmidt ya excluye `meditation`; P0 se repite en el diseño primario |
| 2026-11-22 | ¿Las hipótesis están atadas a estrés vs no-estrés? | **No.** H1–H6 se enuncian sobre corrección, confianza, certeza, abstención y respaldo bajo degradación; el estrés solo aparece en §7 (confusor propio de WESAD) y §8.1 (elección de tarea) | La tarea pasa a ser **parámetro del experimento**; se añade un segundo eje (sueño, DREAMT) sin tocar H1–H6 |
| 2026-11-22 | ¿Sirve DREAMT para el mismo proceso? ¿Tiene estrés? | **No tiene ninguna etiqueta de estrés ni afectiva** (100 sujetos, solo estadio de sueño y apnea), pero **la infraestructura se reutiliza entera**: los 7 extractores corren sobre sus canales (ECG 93 latidos/60 s, EDA 9,21 µS, ACC 0,232 g, resp 25,4 rpm); las etiquetas viajan en la carpeta de 64 Hz, así que el ground truth PSG no exige los 101 GB | DREAMT entra como **segundo eje de tarea**, no como réplica del endpoint afectivo. Rama wearable (14 GB) preregistrada |
| 2026-11-22 | ¿Qué hacer con N3? | 5,3 % de las épocas y **presente en 6 de 13 sujetos**; sus direcciones son las mismas que N2 en las 4 features de firma | **Excluir N3** del espacio de clases (4 clases: W/N1/N2/R); fusionarlo con N2 inyectaría ruido de etiqueta justo en el eje de H5 |
| 2026-11-22 | ¿Sirve la matriz C0–C5 en la rama wearable? | No: degrada respiración, que el E4 no mide; y en WESAD «50 % de NaN» ya equivale a «100 %» (el extractor devuelve `null`), así que severidad parcial y total colapsan | Eje nuevo: **ausencia** (`C1_absent_eda`) frente a **degeneración** (`C2_degenerate_eda`, ceros imposibles), más `C3_absent_hrv`, `C4_absent_core` y `NC_absent` |
| 2026-11-22 | ¿Hay HRV en la rama *wearable*? | **No**: `IBI` está cuantizado a 1/64 s con *dither* (0–21 intervalos útiles por ventana tras deduplicar) y los picos de `BVP` dan RR de 328 a 5.641 ms; la columna `HR` del dispositivo (1 Hz) sí es utilizable | Desviaciones **D7** (bloque cardiaco desde `HR`, sin SDNN/RMSSD) y **D8** (firma de 3 features; `rem` no firma HR porque está a −0,2 SD) declaradas antes de generar respuestas |
| 2026-11-22 | ¿Es resoluble la tarea de sueño con 18 features de muñeca a 60 s? | La referencia supervisada con **las mismas features y la misma estandarización** (LOSO, 1.309 ventanas) da accuracy **0,494** y balanced **0,418**; el azar es 0,25 | El piloto **no puede** sostener afirmaciones de precisión: se declara techo de la evidencia en la rama wearable y H6 (calidad) queda sin evaluar aquí |
| 2026-11-22 | ¿Replica el mecanismo de la tesis en otra tarea y otro sensor? | **Sí, parcialmente**: `C4`/`NC_absent` → **100 %** de abstención en las dos vistas; `unsupported = 0` en las seis condiciones; `C3` numérico con **53,7 %** de claims verdaderos **contrarios** a la decisión (coherencia 0,719 → 0,491); confianza con 2 valores (75/85) y sobreconfianza **+0,53** | E12 entra en [`experimentos.md`](experimentos.md) como **demostración de mecanismo** y **negativo de precisión** (LLM 0,300/0,275 vs 0,494) |
| 2026-11-22 | ¿La abstención ante la falta de evidencia es del modelo o de la representación? | De la **representación**: con EDA ausente (`C1`) el numérico se abstiene **25,0 %** y el texto **98,8 %**; con EDA degenerada a ceros (`C2`), **0 % en ambas** | **H4 pasa a formularse como par (modelo, representación)**; falsadores F1, F3, F4 y F5 no se activan y F2 solo parcialmente |

### Plantilla para la próxima sesión

```
| AAAA-MM-DD | <pregunta o foco> | <hallazgo con cifra> | <decisión o cambio> |
```

Si la sesión no cambia el diseño ni las conclusiones (p. ej. solo
infraestructura), basta una entrada que lo diga: sirve para no releer lo mismo.
