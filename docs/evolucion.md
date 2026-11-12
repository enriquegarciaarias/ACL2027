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

### Plantilla para la próxima sesión

```
| AAAA-MM-DD | <pregunta o foco> | <hallazgo con cifra> | <decisión o cambio> |
```

Si la sesión no cambia el diseño ni las conclusiones (p. ej. solo
infraestructura), basta una entrada que lo diga: sirve para no releer lo mismo.
