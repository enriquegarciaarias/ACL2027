# Conclusiones — síntesis entre modelos y datasets

Este documento cruza los experimentos **entre modelos** y **entre datasets** y
fija qué se puede escribir en las conclusiones de un paper. No sustituye a los
informes de resultados (enlazados al final): es la lectura conjunta.

Todo lo que sigue son cifras de los corpus en disco, no estimaciones. Cuando algo
es exploratorio o está contaminado, se dice en la misma frase.

**Volumen**: 42.288 respuestas de LLM registradas en `results/output` en local
—21.502 en WESAD (incluye los corpus exploratorios de E12), 20.160 en DREAMT
(3 modelos × [piloto 960 + escala local 5.760]), 486 en PhysioNet/CinC 2017 y 140
en K-EmoCon— **más 17.280 de la re-ejecución de la escala de DREAMT en el
servidor, que es la que se cita**. Los corpora del diseño primario de WESAD se
generaron también en el servidor y se citan desde sus informes de evaluación.

---

## 1. El diseño en una tabla

| Dataset | Tarea (etiqueta) | Modelos | Condiciones | Unidad de evaluación | Motor |
|---|---|---|---|---|---|
| **WESAD** | binario `stress` / `non-stress` (`schmidt`: baseline + amusement), sin temperatura | `qwen2.5:7b`, `qwen3:14b` | `C0`, `C1` resp, `C2` EDA parcial, `C4` EDA a cero, `C5` ambas | 424 ventanas, 12 sujetos (S5–S17 sin S12) | servidor, Ollama 0.40.6 |
| **WESAD** | 4 clases afectivas (baseline/amusement/stress/meditation) | los mismos 2 | las mismas 5 | 567 ventanas, 12 sujetos | servidor, Ollama 0.40.6 |
| **DREAMT** | 4 estadios (wake/n1/n2/rem), N3 excluido | `qwen2.5:7b`, `llama3.1:8b`, `medgemma:4b` | `C0`, `C1` EDA ausente, `C2` EDA degenerada, `C3` cardíaco ausente, `C4` núcleo ausente, `NC_absent` | 480 ventanas/condición, 12 sujetos, 2 vistas | servidor, Ollama 0.40.6 |
| **PhysioNet 2017** | 4 ritmos (normal/af/other/noisy) | `qwen2.5:7b` | una sola (limpia) + clase `noisy` de experto | 403 registros | 0.30.7 (histórico) **y** 0.40.6 (`srv`, pareado) |

Los motores **no se mezclan**: con el mismo prompt y la misma semilla, Ollama
0.40.6 dio 0 abstenciones y 0.30.7 dio 28. WESAD y DREAMT comparten el motor del
servidor. **PhysioNet se re-ejecutó allí** (2026-11-28, `--tag srv`, 403
llamadas) y el contraste pareado registro a registro da la medida exacta de lo
que cambia el motor ([`p3_physionet.md`](p3_physionet.md) §5.11): **402 de 403
predicciones idénticas** (McNemar p = 1,000), 0/103 abstenciones en `noisy` en
los dos brazos, y en cambio **+6,4 pp de claims respaldados** y **+0,57 de
índice de certeza** (menos *hedges*). Es decir: la **corrección y la abstención**
(H1, H4) no dependen del motor; el **nivel** de las métricas de lenguaje (H2b,
H5) sí. Las comparaciones entre modelos se hacen **dentro** de cada dataset, y
las comparaciones entre datasets se hacen sobre el **patrón**, nunca sobre el
número.

---

## 2. Resultado 1 — La abstención detecta la *ausencia*, no la *inutilidad*

Es el hallazgo más consistente de los tres datasets y el que da forma a H4.

| Situación de la entrada | WESAD | DREAMT | PhysioNet |
|---|---|---|---|
| Todas las features ausentes (`null`) | **100 %** abstención (`NC_nan`/`NC_absent`, corpus de 4 clases) | **100 %** en 6/6 brazos modelo × vista (`NC_absent`) | no aplicable |
| Canal clave **a cero** (valor imposible) | 0 % (`C4`: 7b 0,0 %; 14b 14,4 %) | 0–31 % (`C2`, num.: 31,0 / 0,0 / 25,4 %) | — |
| Canal clave **ausente** (parcial) | 22,2 % (`C2`, 14b) | 0–88 % (`C1` num.: 87,9 / 1,3 / 17,1 %) y 0–57 % (`C3` num.: 45,8 / 0,2 / 56,5 %) | — |
| Valores **presentes pero inutilizables** | — | — | **0/103** en `noisy` (0–3,6 % IC Wilson) |

Lectura: los modelos detectan la **sintaxis de la ausencia** (campos vacíos) y no
la **semántica de la inutilidad** (ceros, ruido, incoherencia). Un canal a cero
es un valor perfectamente legible, y el modelo lo lee.

Dos consecuencias prácticas:

1. **No existe un detector genérico de «no puedo decidir»**. La abstención depende
   del par (modelo, canal) y de la vista: `qwen` en WESAD se retira con la EDA
   destruida; en DREAMT, con el canal cardíaco ausente, `qwen` se abstiene el 46 %
   y `medgemma` el 57 %, mientras `llama3.1:8b` decide en el 99,8 %.
2. **La compuerta determinista sí recupera la abstención**, y solo ella: en
   `NC_shuffled` de WESAD pasa de 95,3 % a **100 %** (numeric) y de 86,5 % a
   **100 %** (text), sin tocar el limpio (C0 idéntico). La misma regla **en el
   prompt** no funciona: en la ablación solo-prompt, `NC_shuffled` pasó de 13,0 %
   a **0,0 %** de abstención. Los modelos no ejecutan la comprobación que se les
   pide, aunque la representación la permita.

---

## 3. Resultado 2 — Sin evidencia hay varias formas de fallar, y la accuracy premia la peor

Con el núcleo de la evidencia destruido, los brazos fallan en direcciones
distintas, y la métrica que se elija decide qué modelo «gana».

**WESAD `C4` (EDA a cero; HR, respiración y temperatura siguen vivas), 424
ventanas, prevalencia de estrés 0,300:**

| Brazo | Cobertura | Accuracy | Balanced | Precision | Recall | Estrategia |
|---|---:|---:|---:|---:|---:|---|
| Referencia supervisada (19 features, CV por sujeto) | 1,000 | 0,861 | — | — | — | la señal existe |
| `qwen2.5:7b` | 0,998 | 0,547 | **0,637** | 0,387 | 0,858 | sobrerreclama estrés (173 FP) |
| `qwen3:14b` | 0,856 | **0,824** | **0,522** | 1,000 | 0,045 | clasificador nulo: 3 TP de 127 (64 FN, 60 abstenciones) |

El 14b gana por 27,7 puntos de accuracy y pierde en balanced: con prevalencia
0,30, decir siempre «non-stress» da 0,70 de accuracy. Contando la abstención como
error, su ventaja se reduce a +0,158; en balanced es **−0,126**.

**DREAMT (480 ventanas/condición, azar 0,250, referencia 0,408 / 0,362): la
degradación no separa «mejores» y «peores», separa tres modos de fallo.**

| Brazo | `C0` | `C1` EDA ausente | `C2` EDA a cero | `C3` cardíaco | Estrategia |
|---|---:|---:|---:|---:|---|
| `qwen2.5:7b` | 0,315 | **0,052** (abs 88 %) | 0,254 | 0,144 | se retira en proporción a lo que falta |
| `llama3.1:8b` | 0,298 | 0,329 | 0,329 | 0,269 | **no se retira** (≤ 1,3 %): salida independiente de la evidencia |
| `medgemma:4b` | 0,304 | 0,290 | 0,177 | **0,067** | se retira y **selecciona**: en `C3` conserva 15,6 % con balanced 0,417 |

(cifras de `acc*`; en `C4` y `NC_absent` los tres se retiran al 100 %, salvo
`medgemma` textual, 77,3 % + 22,7 % de inválidas.)

La diferencia entre modelos aparece en las ausencias **parciales**, y ahí `acc*`
ordena por cobertura, no por calidad: `medgemma` es el que mejor conserva —en
`C3` su balanced sobre el 15,6 % retenido es 0,417 [0,333; 0,491], por encima del
azar y sin poder separarse de la referencia— y el que peor `acc*` obtiene
(`Δacc* −0,077` frente a `qwen` y `−0,202` frente a `llama`). Es la misma lección
que en WESAD: **la métrica premia al que no se calla**.

Regla de lectura que se deriva de aquí, y que hay que llevar al paper: **toda
comparación entre modelos reporta balanced y cobertura junto a la accuracy**. La
accuracy sobre decididas es una medida de selectividad, no de discriminación.

---

## 4. Resultado 3 — Los claims son locales y verdaderos; la inferencia no

H5 se replica en los tres datasets con la misma forma: lo que el modelo cita de
la entrada suele ser cierto, y la conclusión global puede ser falsa.

| Dataset | Condición | Claims | Respaldados | Comentario |
|---|---|---:|---:|---|
| WESAD (primario) | `C0` → `C5`, 7b | 1.862 | 0,768 → **0,811** | el respaldo sube mientras la decisión se hunde |
| WESAD (primario) | `C0` → `C5`, 14b | — | 0,876 → **0,915** | ídem |
| WESAD (affective) | todas | 9.371 | 0,697–0,828 | `unsupported` = 0 en las cinco condiciones |
| DREAMT (escala) | `C0` limpio | — | 0,58–0,69 | `unsupported` ≈ 0; `medgemma` mantiene **0,463** en `C4` numérico con abstención 98 % |
| PhysioNet 2017 | única | 1.727 | 0,524 | **0,476 contradichos**, 0 `unsupported` |

`unsupported = 0` es informativo: cuando un canal desaparece, el modelo **deja de
citarlo**. No alucina la fuente; lo que falla es la combinación.

**Eje de relevancia (exploratorio, corpus de 4 clases, 15 sujetos).** Verificar
*verdad* no es verificar *coherencia con la decisión*. Con EDA destruida, de las
respuestas que deciden `stress`, el **68,1 %** (`C4`) y el **65,0 %** (`C5`) de sus
claims verdaderos apuntan **en contra** de esa decisión. Es la forma más nítida
del hallazgo: el modelo cita correctamente la evidencia que lo contradice y
concluye igual. Estos números son exploratorios; el diseño primario solo se
evaluó con el eje de verdad.

---

## 5. Resultado 4 — La representación cambia el modo de fallo, no la capacidad

**WESAD (424 ventanas, `schmidt` sin temperatura):**

| Condición | 7b num. | 7b txt. | Δ | 14b num. | 14b txt. | Δ |
|---|---:|---:|---:|---:|---:|---:|
| `C0` | 0,908 | 0,762 | **−0,146** | 0,915 | 0,903 | −0,012 |
| `C5` | 0,682 | 0,252 | **−0,429** | 0,826 | 0,952 | artefacto de cobertura |

- Efecto de la degradación (C0→C5): numérico **−0,226**, texto **−0,510** en el 7b;
  **−0,220** y **−0,442** en el 14b. **El texto duplica el coste de la degradación.**
- El +0,126 del 14b textual en `C5` es cobertura: se abstiene en el 52 % de las
  ventanas. En utilidad (`acc*`) pierde **−0,239**.
- El 7b textual produce **155 inválidas** (5,5 %, 140 en `C4`) por
  `direction: "not available"`: el esquema **no tiene una categoría para «claim de
  ausencia»**. Es una limitación de diseño, no un alias que se pueda añadir después.

**DREAMT (480 ventanas/condición), interacción con el modelo:**

| Condición (abstención) | `qwen` num. | `qwen` txt. | `llama` num. | `llama` txt. |
|---|---:|---:|---:|---:|
| `C1` EDA ausente | 0,879 | 1,000 | 0,013 | 0,000 |
| `C4` núcleo ausente | 1,000 | 1,000 | 1,000 | 0,952 |

**Con el motor del servidor, la interacción de DREAMT desaparece**: los dos
formatos detectan la ausencia total (95–100 %) y `qwen` detecta la EDA ausente en
las dos vistas (0,879 frente a 1,000). El «espejo» de 0,294 frente a 0,996 que se
midió en local era del motor 0.30.7. El único efecto de representación que
sobrevive es el de **WESAD** —el texto duplica el coste de la degradación, −0,51
frente a −0,23, y pierde 14,6 puntos ya en limpio—, y por eso H3 se sostiene con
ese brazo y no con DREAMT.

---

## 6. Resultado 5 — Ni la confianza declarada ni la certeza del lenguaje son incertidumbre

- **WESAD, 7b**: la confianza toma **dos valores** en 703 de 705 respuestas; su
  AUROC es 0,51–0,54 y la sobreconfianza crece hasta **+0,300** en `C4`, con ECE
  0,068 → 0,298.
- **WESAD, 14b**: la confianza **sí** ordena (AUROC 0,964 → 0,746) y la certeza
  lingüística modula sobre las decididas (−0,25 → −0,85). Es un caso distinto, no
  un matiz.
- **DREAMT** (servidor): la certeza lingüística es **plana** —índice 0,00 en las
  respuestas decididas de los tres modelos— y solo es negativa en las
  abstenciones (`qwen` `C1` −28,57). La confianza ya se documentó con dos valores
  en el piloto.
- **Repeticiones (E11, 20 pasadas por ventana)**: la confianza **no codifica** la
  incertidumbre. Una ventana empata 10/10 entre clases con confianza **85 ± 0,0**
  en las 20 repeticiones. Repetir **revela** la incertidumbre, no la arregla; con
  el motor del servidor, promediar repeticiones sube el AUROC de confianza
  **+0,064** (≈1,9 sd). Para describir una ventana harían falta k ≈ 20; para el
  agregado, k = 3.
- **PhysioNet**: el efecto se **invierte**. Con evidencia inutilizable (`noisy`),
  el modelo no solo no se abstiene: se vuelve **más asertivo** (P(`noisy`>resto)
  = 0,603, **Cliff δ = +0,206**, p = 0,0008), mientras en WESAD la degradación
  aumenta los *hedges*. El **signo** replica con el motor 0.40.6 (δ = +0,139,
  p = 0,019) pero la **magnitud es del motor**: el índice global pasa de −1,17 a
  −0,60 y la heterogeneidad entre las cuatro clases deja de ser significativa
  (Kruskal p 0,008 → 0,078) con las mismas decisiones. Se cita la **dirección**;
  el nivel, con su motor.

Conclusión para el paper: la confianza y la certeza son **canales de la
respuesta**, no lecturas de la evidencia. Reportar ECE y AUROC(confianza) es
obligatorio; ninguno de los dos sustituye a la repetición o a la predicción
selectiva.

---

## 7. Resultado 6 — Las señales de calidad son de la tarea, no del modelo

- **WESAD**: el SQI **no se replica** en el diseño primario (ΔSQI inestable en el
  7b: −0,139 a +0,119; ≤ +0,112 en el 14b) y la compuerta de consistencia aporta
  **ΔAUROC ≈ 0** en las diez celdas. La compuerta sirve para **recuperar
  abstención** (100 % en `NC_shuffled`), no para mejorar la decisión.
- **PhysioNet**: aquí la calidad **sí** deja rastro. Un detector multivariante de
  5 features, entrenado en desarrollo, alcanza **AUROC 0,873** (AP 0,692) para
  separar `noisy`; un umbral con recall 0,5 pierde 7,0 % de cobertura limpia.
- **DREAMT**: no hay indicador de calidad propio todavía; H6 no es evaluable.

El contraste es la lección: un score de calidad vale cuando la corrupción tiene
**firma medible** (ruido en ECG) y no cuando el canal simplemente **falta** (lo
detecta la compuerta) o **está a cero** (nadie lo detecta).

---

## 8. Qué replica y qué no (tabla de decisión)

| Fenómeno | WESAD | DREAMT | PhysioNet | Veredicto |
|---|---|---|---|---|
| Abstención al faltar **todas** las features | 100 % | 100 % (6/6) | n/a | **Replica** (todos los pares medidos) |
| Abstención con valores **inutilizables** | 0 % (`C4`, EDA a cero) | 0–57 % (`C2`/`C3`) | 0 % (0/103) | **Ningún modelo lo hace de forma fiable** |
| Respaldo de claims con la degradación | sube (0,768→0,915) | `unsupported` ≈ 0 | 47,6 % contradichos | **Replica en forma**: los claims son locales |
| Confianza informativa | 14b sí / 7b no | no | no | **Depende de modelo y tarea** |
| Certeza lingüística como señal | 14b sí, 7b no | no (0,00) | **invertida** | **No replica** fuera de WESAD-14b |
| Representación textual | peor y más frágil | sin efecto (0,95–1,00) | n/a | **Solo WESAD: el texto cuesta más** |
| SQI / calidad como señal | no replica (Δ≈0) | no evaluable | sí (AUROC 0,873) | **Específico de la tarea** |

---

## 9. Amenazas a la validez (para la sección de limitaciones)

1. **Motor.** WESAD, DREAMT (escala) y **PhysioNet (`srv`)** se generaron con
   Ollama **0.40.6**; el corpus histórico de PhysioNet es de 0.30.7 y se conserva
   como **contraste de motor** (la procedencia solo se escribe si hay llamadas
   pendientes, así que reanudar sin `--tag` no puede reetiquetarlo). El contraste
   pareado cuantifica el efecto por primera vez: con la misma entrada y **402 de
   403 decisiones idénticas**, el motor nuevo produce **+6,4 pp de claims
   respaldados** y **+0,57 de índice de certeza**. En DREAMT el motor alteró la
   abstención de forma aún mayor: la misma condición pasó de 0,294 a 0,879 en
   `qwen`/`C1`, y un hallazgo local (`llama3.1:8b` decidiendo en `C4`)
   desapareció. Las comparaciones entre datasets son de **patrón**; las cifras de
   claims y certeza exigen motor único (`run_provenance.json` lo registra).
2. **Unidad de generalización.** Las 424–480 ventanas vienen de **12 sujetos**; los
   IC son bootstrap agrupado por sujeto. El n efectivo es ≈12, no ≈500.
3. **Diseño no completamente preregistrado.** El endpoint binario, la jerarquía
   `schmidt`, la exclusión de temperatura, la normalización z intra-sujeto, el
   conjunto de 23 features y los umbrales de la compuerta se decidieron viendo
   datos de evaluación; están declarados como desviaciones 1–10 en
   [`split_dev_test.md`](split_dev_test.md). Los experimentos E4 (binario) y E5
   (calibración) siguen marcados como **exploratorios/contaminados** y no se citan
   como firme.
4. **Verificación de claims sin anotación humana.** `supported`/`contradicted` salen
   de una regla determinista sobre las features del prompt (intervalos de
   compatibilidad), no de jueces humanos. Falta un segundo anotador.
5. **Techo de la tarea.** En DREAMT la referencia supervisada con las **mismas 18
   features** alcanza 0,408 de accuracy (0,362 balanced) frente a 0,250 de azar:
   ninguna conclusión sobre «el LLM lee mal» puede exceder ese techo.
6. **Few-shot.** Los ejemplares se construyeron al principio con estadísticos del
   conjunto evaluado (fuga detectada y corregida con S2–S4); el corpus de
   PhysioNet conserva ejemplares que actúan como prior de etiqueta, declarado en
   su informe.

---

## 10. Lo que firmaría en un paper (cinco afirmaciones)

1. **La abstención de los LLM responde a la ausencia declarada de datos, no a su
   inutilidad.** Con todos los campos vacíos se abstienen el 100 % de las veces
   (todos los pares modelo–dataset medidos); con un canal **ausente o
   degenerado** no hay regla compartida: en DREAMT `C3` el rango va del 0,2 % al
   57 % de abstención según el modelo, y `llama3.1:8b` decide en más del 98 % de
   las ventanas en todas las condiciones parciales.
2. **La corrección bajo degradación no se explica por falta de información.** En
   limpio el LLM iguala a una referencia supervisada con las mismas features
   (0,908/0,915 frente a 0,905 en WESAD); con EDA destruida el exceso de coste es
   +0,32 (7b) y +0,17 (14b) y la referencia apenas se mueve (−0,044).
3. **Los claims que el modelo emite son localmente verdaderos incluso cuando la
   conclusión es falsa**, y su respaldo **sube** al degradar la evidencia (0,768 →
   0,811 y 0,876 → 0,915) mientras la decisión se hunde. En el eje exploratorio de
   relevancia, dos de cada tres claims verdaderos contradicen la decisión tomada.
4. **La representación no es un factor de mejora sino de modo de fallo**, y su
   efecto es específico de la tarea: en WESAD el texto **duplica** el coste de la
   degradación (−0,51 frente a −0,23) y pierde 14,6 puntos ya en limpio; en
   DREAMT, con el motor del servidor, ambos formatos detectan igual la ausencia y
   el efecto desaparece.
5. **La confianza y la certeza no son incertidumbre, y la calidad sí que depende de
   la tarea**: dos valores de confianza con ECE hasta 0,298, una ventana 10/10 con
   confianza 85 ± 0,0 en 20 repeticiones, certeza *creciente* con el ruido en
   PhysioNet y un SQI que no replica en WESAD pero un detector de calidad con
   AUROC 0,873 en ECG.

**Implicación de diseño**: si el objetivo es que un sistema clínico se calle
cuando no puede decidir, **no basta con pedírselo en el prompt** (ablación
solo-prompt: la abstención en datos corruptos cae a 0 %). Hace falta una
**comprobación estructural determinista** de la entrada, y esa compuerta recupera
la abstención sin coste en limpio.

---

## 11. Dónde está cada número

| Qué | Documento |
|---|---|
| Diseño primario WESAD (424, dos modelos, H1–H6) | [`resultados_jerarquia.md`](resultados_jerarquia.md) |
| Matriz 2 modelos × 2 tareas × 5 condiciones (567) | [`resultados_binario.md`](resultados_binario.md) |
| Corpus de 4 clases, H1–H6 y degradación | [`resultados_degradacion.md`](resultados_degradacion.md) |
| Repeticiones y varianza de muestreo | [`repetitions.md`](repetitions.md) |
| Compuerta de consistencia (P1) | [`consistencia.md`](consistencia.md) |
| Claims: verdad y relevancia | [`claims_h5.md`](claims_h5.md) · [`h5_relevancia.md`](h5_relevancia.md) |
| DREAMT piloto (3 modelos, 4 sujetos) | [`piloto_dreamt.md`](piloto_dreamt.md) |
| DREAMT escala (3 modelos, 12 sujetos) | [`escala_dreamt.md`](escala_dreamt.md) |
| PhysioNet/CinC 2017 | [`p3_physionet.md`](p3_physionet.md) |
| Inventario de experimentos y estados | [`experimentos.md`](experimentos.md) |
