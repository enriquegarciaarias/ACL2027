# Repeticiones de prompt: varianza de muestreo

**Pregunta.** Si se envía **exactamente el mismo prompt** R veces cambiando solo
la semilla de decodificación, ¿cuánto cambia la respuesta y cuántas repeticiones
hacen falta para que la métrica agregada sea estable?

Este documento declara el diseño **antes** de mirar los resultados del piloto.
Es el paso previo a lanzar un corpus de repeticiones completo: con una muestra
reducida se decide si el efecto existe y qué `R` lo estabiliza, en lugar de
gastar horas de LLM por intuición.

---

## 1. Por qué importa

Tres resultados de la tesis dependen de que la respuesta sea estable:

| Hipótesis | Dependencia |
|---|---|
| **H2a** (confianza) | `qwen2.5:7b` emite solo **dos valores** de confianza (85 y 75) en 703/705 ventanas. Si el valor depende de la semilla, el AUROC de confianza (0,50–0,54) mide ruido de decodificación, no calibración. |
| **H1** (corrección) | Una accuracy de una sola pasada es una muestra de tamaño 1 por ventana. Si la varianza es alta, parte del «coste de la degradación» puede ser varianza y no pérdida de información. |
| **H4** (abstención) | La abstención depende de que el modelo «sepa» que no puede decidir; si la decisión cambia con la semilla, la abstención es un umbral inestable. |

El barrido publicado usa `--seed 42` y una pasada. Las repeticiones no cambian
ese corpus: lo **auditan**.

---

## 2. Diseño

- **Mismo prompt, distintas semillas.** El prompt se construye **una vez** por
  `(ventana, condición)` y se reutiliza en las R repeticiones; solo cambia
  `options.seed`. Si se reconstruyera por repetición, cualquier no
  determinismo de la construcción se confundiría con muestreo.
- **Semillas** `--seed + r` con `r = 0..R-1` (42, 43, …, por defecto). La
  repetición 0 es exactamente la del corpus de una pasada, así que un barrido
  con `--repetitions 10` empieza por la misma muestra que el publicado.
- **Muestra sistemática** (`--sample-windows K`): K ventanas repartidas por todo
  el registro del sujeto. No se usa `--max-windows`, que corta por el principio
  y en WESAD eso son casi todas `baseline` (el protocolo es secuencial). La
  selección **no consulta etiquetas**: es ciega a la clase real.
- **Sujetos de evaluación** (S5–S17 salvo S12); los de desarrollo se bloquean
  salvo `--allow-dev`.
- **Condiciones**: `C0_clean` (referencia) y `C5_total_both` (degradación
  máxima) en el 14b; para el 7b, medido después, solo `C5`, que es donde
  aparece la varianza (en `C0` y `C4` el acuerdo era ≥ 0,98).
- **Sin temperatura** (`--temperature exclude`) y variante **`schmidt`**:
  el diseño primario.
- **Salida aislada**: `{nombre}_rep{R}/{modelo}/`, de modo que el corpus de una
  pasada (el que sostiene las cifras publicadas) no se mezcla nunca.

Cada registro guarda `repetition` y `seed`. Los corpus antiguos, sin el campo,
se leen como repetición 0.

---

## 3. Métricas

Por condición:

1. **Acuerdo modal por ventana** — fracción de la decisión más frecuente sobre
   las R repeticiones. `1,0` = respuesta determinista; `0,5` con dos clases =
   moneda al aire.
2. **Entropía** de la distribución de decisiones por ventana (bits) y número de
   decisiones distintas.
3. **Dispersión de la confianza** intra-ventana (`sd`, mínimo, máximo, número de
   valores distintos).
4. **Variabilidad de una pasada** — accuracy, balanced, cobertura y abstención
   calculadas con **cada repetición por separado**. Su desviación típica es el
   error de muestreo de un experimento de una sola pasada.
5. **Convergencia** — las mismas métricas con `k = 1..R` repeticiones
   agregadas por **voto mayoritario** (empate → `abstain`).
6. **Extrapolación** — `k` necesario para que `sd` baje de 1, 2 o 3 puntos, con
   la ley `sd_k ≈ sd_1 / sqrt(k)` (cota optimista: supone independencia).
7. **Resolución de la confianza** — valores distintos y AUROC con una pasada
   frente a confianza promediada en R repeticiones.

---

## 4. Regla de decisión (predeclarada)

| Observación en el piloto | Decisión |
|---|---|
| Acuerdo modal **≥ 0,98** en las tres condiciones | El decodificado es prácticamente determinista. **No** se lanza corpus de repeticiones: H2a se cita como límite del modelo (dos valores de confianza), no del muestreo. |
| Acuerdo modal **< 0,90** en C4 o C5 | Hay varianza sustancial. Se fija `R` con la extrapolación a **±2 pp**, con mínimo 5 y máximo 20. |
| Caso intermedio (0,90–0,98) | Se fija `R` con la extrapolación a **±3 pp**, mínimo 3. |
| La confianza promediada sube el AUROC **≥ 0,05** | Las repeticiones se reportan como **intervención sobre H2a**: crean resolución donde el modelo solo emite dos valores. |

El umbral no se mueve después de ver los resultados. Si el piloto cae cerca de
un corte, se declara y se reportan las dos lecturas.

---

## 5. Nota sobre empates y valores pares de `k`

Con `k` par y un modelo cercano al azar, los empates son frecuentes y la regla
conservadora los convierte en `abstain`. La curva de convergencia puede
zigzaguear (k=2 peor que k=1 y k=3). No es un fallo del análisis: dos muestras
en desacuerdo no sostienen una decisión. Por eso se reportan también
**cobertura** y **abstención**, y la comparación de `k` se hace con la curva
completa, no con un punto.

---

## 6. Coste

Llamadas = sujetos × ventanas por sujeto × condiciones × R.

| Fase | Sujetos | Ventanas | Condiciones | R | Llamadas | Tiempo real |
|---|---|---:|---:|---:|---:|---|
| Pre-piloto 14b | 1 (S5) | 12 | 2 (C0, C5) | 5 | 120 | 9 min 51 s |
| Piloto 7b | 3 (S5–S7) | 12 | 1 (C5) | **20** | 720 | 25 min 36 s |

No se ejecutó corpus completo: la regla de decisión lo descarta (§8.6). Si se
quisiera una tasa de inestabilidad con más ventanas, el siguiente paso es
**ampliar sujetos** (S8–S10), no repeticiones: 3 × 12 × 1 × 20 = **720
llamadas**, ~25 min al ritmo medido del piloto (720 en 25 min 36 s).

---

## 7. Reproducción

```bash
# Piloto 7b (servidor; regla 8 de AGENTS.md: lanzador desde ~/data)
~/data/launch_lsi.sh --proc run_ACL.sh --time 02:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --subjects S5,S6,S7 --sample-windows 12 --conditions C5_total_both \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --repetitions 20 --model qwen2.5:7b

# Pre-piloto 14b
~/data/launch_lsi.sh --proc run_ACL.sh --time 01:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --subjects S5 --sample-windows 12 --conditions C0_clean,C5_total_both \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --repetitions 5 --model qwen3:14b

# Análisis (en nodo de cálculo: en sirius numpy aborta con SIGILL)
srun --partition=A5000 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:15:00 --pty bash
cd ~/data/proyectos/ACL2027
.venv/bin/python scripts/analyze_repetitions.py \
  --dir results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp_rep20/qwen2_5_7b \
  --output results/output/WESAD/PHASE2/repetitions_rep20_qwen2_5_7b.json
# o, con el despachador: --proc EVALUATE --experiment degradation --repetitions 20
```

Sin `--output-dir` (bandera del script, no del CLI), el barrido escribe en
`degradation_binary_schmidt_notemp_rep{R}/{modelo}/`. Lanzado desde
`scripts/run_degradation_sweep.py` se puede forzar otro destino con
`--output-dir`.

El análisis deduce la tarea y la variante de la procedencia de las respuestas y
descarta las ventanas fuera de la variante (`meditation` en `schmidt`).

### 7.1 Ampliación pendiente en el servidor (S8–S10)

Lo que falta para citar el +0,064 de AUROC con intervalos es **más ventanas, no
más repeticiones**: S8–S10 con la misma muestra sistemática (`--sample-windows
12`, ciega a las etiquetas) y las mismas 20 semillas (42…61). El barrido es
reanudable por `(ventana, condición, repetición)`, así que S5–S7 se omiten solos.

```bash
# 1) Preflight EN NODO DE CÁLCULO (en sirius numpy aborta con SIGILL)
srun --partition=A5000 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:10:00 --pty bash
cd ~/data/proyectos/ACL2027
.venv/bin/python main.py --proc GENERATE --experiment degradation --subject WESAD \
  --model qwen2.5:7b --decision-mode binary --nonstress-variant schmidt \
  --temperature exclude --conditions C5_total_both \
  --subjects S8,S9,S10 --sample-windows 12 --repetitions 20 --dry-run
# → estimated_llm_calls: 720 (3 × 12 × 1 × 20)
exit

# 2) Generación
~/data/launch_lsi.sh --proc run_ACL.sh --time 02:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --model qwen2.5:7b --decision-mode binary --nonstress-variant schmidt \
  --temperature exclude --conditions C5_total_both \
  --subjects S8,S9,S10 --sample-windows 12 --repetitions 20

# 3) Análisis (despachador; resuelve el directorio _rep20 solo)
srun --partition=A5000 --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:15:00 --pty bash
cd ~/data/proyectos/ACL2027
.venv/bin/python main.py --proc EVALUATE --experiment degradation --subject WESAD \
  --model qwen2.5:7b --repetitions 20 --decision-mode binary \
  --nonstress-variant schmidt \
  --output results/output/WESAD/PHASE2/repetitions_rep20_qwen2_5_7b.json
```

Al analizar el corpus completo (S5–S10) el informe sale con 6 sujetos; si se
lanza solo la ampliación y el informe se escribe en el mismo `--output`, se
pierde el de S5–S7: usar un `--output` distinto y compararlos, o mover los
`.jsonl` a un directorio que reúna las seis series.

**Guardia de motor (nueva).** Antes de escribir `run_provenance.json`, el
barrido compara `model`, `model_digest` y `ollama_version` con los del corpus
existente; si difieren, **aborta** (código 2) en lugar de reanudar. Motivo: la
caché de reanudación no indexa el motor, y en un experimento de varianza de
muestreo un cambio de build sería indistinguible del ruido que se mide. Solo se
omite con `--allow-engine-change` (explícito, para un caso declarado), también
desde el despachador (`main.py … --allow-engine-change`). Además, un `timeout`
nunca cuenta como respuesta cacheada y el barrido aborta si el modelo no está
instalado. Antes de lanzar, comprobar que el corpus del piloto sigue allí y con
qué motor:

```bash
ls results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp_rep20/qwen2_5_7b/
cat results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp_rep20/qwen2_5_7b/run_provenance.json
```

Si el corpus se copió de otro entorno (p. ej. un piloto local con Ollama
0.30.7), **muévelo o renómbralo** antes de reanudar: mezclar motores no se
puede separar después con los datos.

---

## 8. Resultados

### 8.1 Ejecuciones citables

| Run | Entorno | Diseño | Llamadas | Errores |
|---|---|---|---:|---:|
| Pre-piloto 14b | **servidor** (Ollama 0.40.6) | S5 · C0 + C5 · R=5 | 120 | 0 |
| Piloto 7b | **servidor** (Ollama 0.40.6) | S5–S7 · 12 ventanas · C5 · **R=20** | 720 | 0 |

Los runs locales no se citan: sirvieron para descubrir que el motor importa
(§7.4). 30 ventanas útiles por condición tras descartar `meditation`.

### 8.2 El agregado es estable; la ventana no (7b, C5, R=20)

| Acuerdo modal por ventana | Ventanas |
|---|---:|
| 20/20 (nunca cambia) | **22** (73,3 %) |
| 18/20 | 3 |
| 16/20 | 1 |
| 15/20 | 2 |
| 11/20 | 1 |
| **10/20 (empate)** | **1** |

Lo decisivo no es cuántas cambian, sino **dónde**:

| Clase real | Ventanas | Inestables |
|---|---:|---:|
| baseline | 15 | 4 |
| amusement | 6 | 4 |
| **stress** | **9** | **0** |

**Ninguna ventana de estrés cambió ni una vez en 20 repeticiones**, y las ocho
inestables son `non-stress` que a veces se reclasifican como `stress`. La
inestabilidad es, otra vez, el error del modelo: sobre-reclamar estrés con EDA
destruida.

Una pasada frente al voto mayoritario:

| Métrica | Una pasada (20 valores) | Agregado k=20 |
|---|---|---|
| Accuracy | 0,648 ± 0,043 (rango 0,533–0,700) | 0,690 |
| Balanced | 0,749 ± 0,031 | 0,775 |
| AUROC confianza | 0,761 ± 0,033 | **0,825** |
| Cobertura | 1,000 (0 abstenciones) | 0,967 |

| k | Accuracy | Balanced | sd |
|---:|---|---|---:|
| 1 | 0,648 | 0,749 | 0,043 |
| 3 | 0,661 | 0,758 | 0,025 |
| 5 | 0,667 | 0,762 | 0,019 |
| 10 | 0,676 | 0,766 | 0,011 |
| 20 | 0,690 | 0,775 | — |

Extrapolación: **k ≈ 3 para ±3 pp** y **k ≈ 5 para ±2 pp** sobre esta muestra.
La accuracy sube +4,2 pp al agregar 20 repeticiones (≈1 desviación típica: **no
concluyente**); el AUROC sube +0,064 (≈1,9 desviaciones: **sugerente**).

### 8.3 El caso que decide la interpretación

| Ventana | Clase real | Reparto (20) | Confianza |
|---|---|---|---|
| `win_f87d0217` | baseline | **10 `non-stress` / 10 `stress`** | **85,0 ± 0,0** |
| `win_25ab1b5e` | amusement | 11 / 9 | 82,0 ± 4,6 |
| `win_03ce68c4` | amusement | 15 / 5 | 85,0 ± 0,0 |
| `win_97be8f9f` | amusement | 15 / 5 | 85,0 ± 0,0 |
| `win_df16c22a` | baseline | 16 / 4 (hacia stress) | 84,5 ± 2,2 |
| `win_f5602f53` | baseline | 18 / 2 | 85,0 ± 0,0 |
| `win_8259b39e` | amusement | 18 / 2 | 84,0 ± 3,0 |
| `win_aefe7971` | baseline | 18 / 2 | 85,0 ± 0,0 |

Una ventana sale **exactamente 10 a 10** entre dos clases y el modelo declara
**confianza 85 en las veinte repeticiones, sin una sola variación**. No es que
la confianza esté mal calibrada: es que **no codifica la incertidumbre**. La
incertidumbre está en el muestreo, no en el número.

Y eso no se arregla repitiendo. Repetir **revela** la indecidibilidad; no la
resuelve. Es la diferencia entre un indicador de calidad calculado desde fuera
(la repetición, el SQI, la compuerta) y una señal que el modelo emite.

### 8.4 `qwen3:14b`: determinista en la decisión, graduado en la confianza (R=5)

S5, C0 + C5, 10 ventanas por condición:

| Condición | Acuerdo modal | Deterministas | sd(conf) intra | Valores distintos | Acc | Balanced |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | **1,000** | 100 % | 0,64 | 5 | 0,900 | 0,929 |
| `C5_total_both` | **1,000** | 100 % | 0,20 | 3 | 1,000 | 0,500 |

Ni una decisión cambió entre semillas. La varianza está toda en la confianza,
que además ya es graduada (75/85/90/95/50): promediar reduce ruido, no aporta
resolución.

| | `qwen2.5:7b` | `qwen3:14b` |
|---|---|---|
| Decisión entre semillas | varía (8/30 en C5) | **no varía** |
| Confianza entre semillas | fija (85 / 50) | **varía** |
| Abstención en C5 | 0 % | 30 % |
| Accuracy / balanced en C5 | 0,648 / 0,749 | **1,000 / 0,500** |

Cada modelo tiene un grado de libertad y es el contrario del otro. Y la trampa
de la accuracy del 14b (1,000 con balanced 0,500 y cobertura 0,70) es
**determinista**: no es una casualidad de semilla.

### 8.5 Qué se puede afirmar

1. **H1 no es un artefacto de semilla.** El agregado se mueve ±4,3 pp en 30
   ventanas y las 9 ventanas de estrés no cambian nunca; escalado al corpus
   completo (424) queda del orden de **±1 pp**, muy por debajo de la
   incertidumbre por sujeto.
2. **H2a: la confianza no mide evidencia.** El caso 10/10 con 85 ± 0,0 lo
   demuestra sin estadística. Agregar repeticiones mejora el AUROC +0,064
   (sugerente, ~1,9 sd), pero eso es una propiedad del *promedio*, no del
   modelo.
3. **La abstención no es la vía:** el 7b no se abstiene nunca (0/30, cobertura
   1,000) ni cuando la respuesta es 50/50.
4. **Para cifras agregadas basta k=3** (regla predeclarada). Para describir una
   ventana hace falta k≈20, y aun así una ventana 50/50 no tiene respuesta
   estable que reportar.

### 8.6 Regla de decisión aplicada

| Condición | Acuerdo modal (R=5/R=20) | Banda | Decisión |
|---|---:|---|---|
| 7b `C0_clean` | 0,993 | ≥ 0,98 | determinista |
| 7b `C4_total_eda` | 0,987 | ≥ 0,98 | determinista |
| 7b `C5_total_both` | 0,935 | 0,90–0,98 | intermedia → **R = 3** para el agregado |
| 14b `C0_clean` | 1,000 | ≥ 0,98 | determinista |
| 14b `C5_total_both` | 1,000 | ≥ 0,98 | determinista |

### 8.7 Límites

- **30 ventanas y 3 sujetos** en la condición inestable: suficiente para
  demostrar que la inestabilidad existe y dónde está, insuficiente para fijar
  su tasa con precisión. El +0,064 de AUROC necesita más ventanas para citarse.
- El 14b se midió en **un sujeto y dos condiciones**.
- **Dependencia del motor** (§7.4): los números son de Ollama 0.40.6 con el
  digest registrado en `run_provenance.json`. Otro build no es comparable.
- Solo se auditó la representación **numérica** y la condición C5 del 7b.

### 8.8 Fallos corregidos por el camino

- `"direction": "slight increase"` invalidaba la respuesta: añadido
  `_normalize_direction` (slight, sharply, markedly, somewhat…), con tests.
- El corpus no registraba la versión de Ollama ni el digest del modelo:
  añadidos `ollama_run_provenance()` y `run_provenance.json`.

---

## 9. Plan completo pendiente (estado guardado, 2026-11-26)

Esta sección fija **qué falta, con qué `R`, en qué brazos y a qué coste**, para
que la campaña pueda lanzarse más tarde sin volver a deducir el diseño. No se ha
ejecutado nada de lo que sigue salvo lo indicado como «hecho».

### 9.1 `R` ya está determinado (y por objetivo)

La regla del §4 está congelada y aplicada en §8.6. No hay un único `R`: depende
de para qué se use la repetición.

| Objetivo | `R` | Fundamento |
|---|---:|---|
| **Cifra agregada** (accuracy, balanced, cobertura, abstención) | **3** | Banda intermedia (acuerdo 0,935 en 7b `C5`): extrapolación a ±3 pp, mínimo 3. Verificado en el piloto: `sd` 0,043 → 0,025 con k=3. |
| **Describir una ventana** (¿esta ventana es indecidible?) | **≈20** | La ventana 10/10 sigue 10/10; con menos `k` el empate no se ve. Y aun con 20 no hay respuesta estable que reportar. |
| **Brazos deterministas** (acuerdo ≥ 0,98) | **1** (auditoría puntual) | 14b `C0`/`C5` y 7b `C0`/`C4`: la decisión no cambió ni una vez. Repetir no aporta; una confirmación basta. |
| **Intervención sobre H2a** (¿promediar crea resolución?) | 20 | El +0,064 de AUROC de confianza es del promedio; con k=3 no es medible. |

Si en la ampliación el acuerdo de un brazo cae por debajo de 0,98, se aplica la
misma regla sin cambiarla (esto ya está predeclarado, no se decide después).

### 9.2 Lo que se va a repetir y lo que no

**No** se repiten todos los datasets × todos los modelos. Se repite donde la
varianza puede cambiar una conclusión:

| Se repite | No se repite | Motivo |
|---|---|---|
| WESAD numérico 7b `C5` (único brazo con acuerdo < 0,98) | 7b `C0`/`C4`, 14b (deterministas) | Nada que medir; el corpus de una pasada ya lo demuestra. |
| WESAD **texto** (nunca auditado; 155 inválidas de formato en 7b) | WESAD muñeca/BVP (no está en la vista LLM) | El brazo textual puede tener otra varianza y otra tasa de inválidas. |
| DREAMT `C5` y **`C3` de `medgemma`** (detección parcial 0,200 [0,167;0,235]) | DREAMT `C0`/`NC_absent` (0 % / 100 % deterministas) | Hay un hallazgo nuevo cuya estabilidad no se ha comprobado. |
| PhysioNet (se re-ejecuta en el servidor de todos modos) | K-EmoCon (exploratorio, referencia ≈ azar) | Se aprovecha el mismo job; K-EmoCon no sostiene ninguna conclusión. |

### 9.3 Matriz y coste

Ritmos medidos: 7b ≈ 2,1 s/llamada; 14b ≈ 4,9 s; DREAMT ≈ 2,1 s; PhysioNet
`qwen2.5:7b` ≈ 1,5 s.

**Fase A — WESAD, sin código nuevo** (el barrido ya soporta todo):

| # | Brazo | Sujetos | Ventanas | `R` | Llamadas | Tiempo | Para qué |
|---|---|---|---:|---:|---:|---:|---|
| A1 | 7b numérico `C5` | S8–S10 | 12/sujeto | 20 | 720 | ~25 min | Ampliar la auditoría de ventana a 6 sujetos; citar el +0,064 de AUROC. |
| A2 | 7b numérico `C5` | 12 eval | **todas (567)** | 3 | 1.701 | ~1 h | Verificar a escala completa el `R=3` predeclarado. |
| A3 | 14b numérico `C5` | S5–S7 | 12/sujeto | 20 | 720 | ~1 h | ¿La **dispersión** de confianza predice errores mejor que su nivel? (único modelo con confianza graduada). |
| A4 | 7b texto `C5` | 12 eval | todas (567) | 3 | 1.701 | ~1 h | Varianza del brazo textual y estabilidad de las inválidas de formato. |

Total Fase A ≈ **4.842 llamadas**, ~3,5 h. A2/A4 admiten un atajo declarado:
copiar los registros de la repetición 0 desde el corpus base (mismo prompt y
semilla 42) ahorra 567 llamadas por brazo; si se hace, se anota en el informe y
se verifica con `scripts/verify_corpus_rebuild.py`.

**Fase B — DREAMT, ya implementado** (`scripts/run_p4_dreamt.py --repetitions R`;
mismo diseño que el barrido: prompt construido una vez en `prepare`, semillas
`--seed + r`, reanudación por `(ventana, condición, rep)`, salida aislada
`responses_rep{R}/`, guardia de motor y procedencia con `repetitions`/`seeds`):

| # | Brazo | Condición | Sujetos | Ventanas | `R` | Llamadas | Tiempo |
|---|---|---|---|---:|---:|---:|---:|
| B1 | `qwen2.5:7b` numérico | `C1_absent_eda` | 12 escala | 480 | 3 | 1.440 | ~50 min |
| B2 | `qwen2.5:7b` texto | `C1_absent_eda` | 12 escala | 480 | 3 | 1.440 | ~50 min |
| B3 | `medgemma:4b` numérico | `C3_absent_hrv` | 12 escala | 480 | 3 | 1.440 | ~1,5 h |

Condiciones elegidas por **poder mover una conclusión**, no por ser las más
duras: `C1` es donde el brazo numérico se retira 0,294 y el textual 0,996 (la
diferencia de representación que sostiene H3/H4), y `C3` es donde `medgemma`
detecta el bloque cardíaco ausente (0,200) y nadie lo ha replicado. `C4` y
`NC_absent` quedan fuera: los tres modelos se abstienen al 100 %, y una
condición determinista no necesita réplicas. (El plan del 2026-11-26 citaba «C5»
para DREAMT; esa condición **no existe** en esta tarea: su matriz es C0–C4 +
`NC_absent`.)

**Fase C — PhysioNet, ya implementado** (`scripts/run_p3_physionet.py
--repetitions R`, ficheros `physionet2017[_tag]_rep{R}.*`; 400 registros × 3 =
1.200 llamadas, 800 nuevas sobre la pasada base, ~20 min). El reparto por
registro (no por llamada) garantiza que un registro conserve sus tres réplicas
aunque se use `--max-records`.

Coste total de las tres fases ≈ **9.300 llamadas**, ~6–7 h de GPU. Orden
sugerido: A1 → A2 → B3 → A3 → A4 → B1/B2 → C. Criterio de parada por brazo: si
el acuerdo modal sale ≥ 0,98 en la ampliación, el brazo se cierra con una nota
(no se escala a más sujetos).

### 9.4 Prerrequisitos y estado

- [x] Guardia de identidad del motor en el barrido (`engine_conflicts`,
  `--allow-engine-change`).
- [x] Procedencia (`run_provenance.json`), `timeout` no cacheado, aborto si el
  modelo falta, salida `_rep{R}` aislada y análisis en el despachador.
- [x] Corpus del piloto en el servidor: 7b (S5–S7, `C5`, R=20; 720) y 14b (S5,
  `C0`+`C5`, R=5; 120). En local solo el informe.
- [x] Fase B: `--repetitions` en `run_p4_dreamt.py` y análisis externo
  (`scripts/analyze_repetitions_external.py`, 2026-11-28).
- [x] Fase C: `--repetitions` en `run_p3_physionet.py` (mismo analizador) y
  guardia de motor que **aborta** en vez de avisar.
- [x] **B1 ejecutado** (2026-11-28): `qwen2.5:7b` numérico `C1_absent_eda`, R=3
  (semillas 42–44), 1.440 llamadas en 17 min 34 s, 2 inválidas (0,14 %).
  `acuerdo 0,988`, `sd1` de accuracy **0,001**, `cov1` 0,115 → abstención
  **0,885** frente a **0,879** [0,852; 0,908] de la pasada única y `acc*`
  implícito **0,052** (publicado 0,052 [0,035; 0,073]), `k(2pp)=1`.
  **Brazo cerrado con nota** por el criterio de parada (acuerdo ≥ 0,98): `C1`
  numérico es determinista bajo tres semillas y no se escala.
- [x] **B2 ejecutado** (2026-11-28): `qwen2.5:7b` texto `C1_absent_eda`, R=3,
  1.440 llamadas en 12 min 49 s, **0 inválidas**; acuerdo modal **1,000**,
  `cov1` **0,000** (abstención 1,000, como la publicada) → brazo cerrado con
  nota. Detalle en §9.7.
- [ ] B3 (`medgemma:4b` `C3_absent_hrv`) y Fase C (PhysioNet) ejecutados en el
  servidor. **El primer intento de B3 (job 16719) abortó** en la guardia de
  motor porque la procedencia de DREAMT era un fichero por cohorte: ver §9.8.

- [ ] Copia opcional de la repetición 0 desde el corpus base (A2/A4), con
  verificación de prompts.

### 9.5 Qué NO se puede afirmar con el plan actual

- La varianza de la **representación textual** en WESAD (no auditada) ni la de
  DREAMT/PhysioNet (el mecanismo está implementado, pero no hay corpus de
  repeticiones todavía).
- La tasa de inestabilidad con precisión: A1 la mejora (6 sujetos, 60 ventanas
  aprox.), pero el n efectivo sigue siendo de sujetos, no de ventanas.
- Nada de un brazo ejecutado con otro motor: la guardia impide reanudar, y el
  `R` medido en Ollama 0.40.6 no se traslada a otro build.

### 9.6 Cómo se ejecuta (DREAMT y PhysioNet, código listo desde 2026-11-28)

Los dos runners externos replican el diseño del barrido de WESAD:

- **el prompt se construye una vez** por `(ventana/registro, condición)` y solo
  cambia `options.seed` (`--seed + r`);
- **reanudación por repetición**: la clave es
  `(ventana/registro, condición, repetición)`, así que un corte no repite
  llamadas ni mezcla réplicas;
- **salida aislada**: `responses_rep{R}/` en DREAMT y
  `physionet2017[_tag]_rep{R}.*` en PhysioNet. Con `R = 1` las rutas son las de
  siempre, de modo que el corpus publicado no se toca;
- **guardia de motor**: si el `run_provenance*.json` existente declara otra
  versión de Ollama o otro `model_digest`, el runner **aborta** (código 2) salvo
  `--allow-engine-change`. La caché de reanudación no indexa el motor;
- **procedencia**: `repetitions` y `seeds` quedan en el JSON de procedencia, y
  cada registro guarda `repetition`, `seed` y `model`.

**Antes de lanzar, comprueba qué código hay en el servidor.** El 2026-11-28 un
job se lanzó sin `git pull`: el despachador antiguo **ignoró `--repetitions` en
silencio**, hizo 0 llamadas y el log no lo delataba (solo `0 pendientes`). Desde
entonces el arranque imprime `código: <commit> (<rama>, <fecha>)` y el
despachador imprime el `argv` exacto con el que invoca al runner. Si en el log no
aparece `--repetitions` o el commit no es el esperado, el job **no** está
ejecutando el código nuevo:

```bash
grep -n "repetitions" ~/data/proyectos/ACL2027/scripts/run_p4_dreamt.py | head -3
```

Y el análisis, sin coste de LLM:

```bash
python scripts/analyze_repetitions_external.py --experiment dreamt --cohort scale --model qwen2.5:7b --repetitions 3
python scripts/analyze_repetitions_external.py --experiment physionet --tag srv --repetitions 3
```

Por CLI de fases, con enrutado automático al mismo analizador:

```bash
python main.py --proc EVALUATE --experiment dreamt --cohort scale --model qwen2.5:7b --subject DREAMT --repetitions 3
python main.py --proc EVALUATE --experiment physionet --tag srv --subject PHYSIONET2017 --repetitions 3
```

> En el servidor, `run_ACL.sh` **exige** `--subject` (`WESAD`, `DREAMT`,
> `PHYSIONET2017`) y sale con código 2 si falta, aunque el análisis resuelva su
> ruta con `--cohort`/`--tag`. Es la única bandera obligatoria que el lanzador
> impone y no el analizador: un EVALUATE sin ella muere antes de arrancar
> (job 16715, 1 s, `2:0`).

El informe (`repetitions_rep{R}_{modelo}.json` / `repetitions_{tag}_rep{R}.json`)
agrupa por `(representación, condición)` y da: acuerdo modal, entropía, fracción
de ventanas deterministas, dispersión de la confianza, variabilidad de la
métrica con **una** pasada (su `sd` es el error de muestreo de un experimento de
una pasada), convergencia por voto mayoritario `k = 1..R` y `k` implícito para
±2/3 pp. Las etiquetas se leen de `labels/*.labels.jsonl` (DREAMT, con el id
opaco recalculado) y de `physionet2017[_tag]_rep{R}.labels.jsonl`; las clases
son las de cada tarea (`wake/n1/n2/rem` y `normal/af/other-rhythm/noisy`), no
las afectivas de WESAD.

### 9.7 Primeros resultados (B1/B2, 2026-11-28): `C1` es determinista en las dos vistas

`qwen2.5:7b`, `C1_absent_eda` (EDA ausente), 480 ventanas × 3 semillas
(42, 43, 44), Ollama 0.40.6, digest `845dbda0…` (el mismo del corpus
publicado). Numérico: 1.440 llamadas, 17 min 34 s, 2 inválidas (0,14 %). Texto:
1.440 llamadas, 12 min 49 s, **0 inválidas**.

| Vista | n | Acuerdo modal | `sd` de accuracy (una pasada) | `cov1` | Abstención publicada | `k` ±2 pp |
|---|---:|---:|---:|---:|---:|---:|
| numérico | 480 | 0,988 | **0,001** | 0,115 | **0,879** [0,852; 0,908] | 1 |
| texto | 480 | **1,000** | 0,000 | **0,000** | **1,000** | 1 |

Lectura:

1. **El brazo se cierra con una nota**: el acuerdo (0,988) supera el umbral
   pre-registrado de 0,98, así que no se escala a más sujetos ni se sube `R`.
   La retirada del modelo ante la EDA ausente es una propiedad estable, no un
   artefacto de una semilla.
2. **La reproducción es exacta**: la abstención cae dentro del IC publicado y el
   `acc*` coincide al milésimo, con una `sd` de 0,001 entre semillas. Sirve de
   control de que el corpus de repeticiones y el publicado son comparables
   (mismo motor, mismo `model_digest`, mismos prompts).
3. **La referencia correcta es la del servidor.** La cifra 0,294 de `C1`
   numérico que circula en `piloto_dreamt.md` §19 es del piloto local con Ollama
   **0.30.7**; en 0.40.6 la misma condición abstiene 0,879 (contraste en
   `escala_dreamt.md` §7.6). Comparar contra 0,294 habría hecho parecer un
   cambio de comportamiento donde solo hay un cambio de motor.
4. **Las dos representaciones son deterministas aquí, y difieren de forma
   estable.** El texto se abstiene en las 480 ventanas (`cov1` 0,000) con
   acuerdo modal 1,000 y cero inválidas, frente al 0,115 de cobertura del
   numérico. Es decir: la diferencia 0,885 vs 1,000 que sostiene el «espejo» de
   H4/H3 en esta condición **no es ruido de una pasada**; es una propiedad del
   formato que tres semillas reproducen exactamente. Sin un brazo de
   repeticiones esto solo se podía suponer.
5. **Con cobertura 0 la accuracy no se publica como 0.** Una pasada sin ninguna
   ventana decidida deja la accuracy indefinida; el impresor muestra `—` en
   `acc1`/`sd1` y `0.000` en `cov1` (antes, el idioma `valor or nan` convertía
   ese 0 en `nan`, ver la incidencia de esta sesión). El dato que sostiene la
   conclusión es la cobertura, no una accuracy de 0 que describiría a un
   clasificador que falla cuando en realidad no clasificó.

### 9.8 Procedencia por modelo (incidencia del job 16719)

El intento de B3 (`medgemma:4b`, `C3_absent_hrv`) abortó en 25 s con código 1:

```
[ERROR] El corpus de run_provenance_rep3.json se generó con otro motor
        (model: qwen2.5:7b -> medgemma:4b; model_digest: 845dbda0… -> 9fe4e9a6…)
```

El diagnóstico del guard era correcto; el **alcance** no: en DREAMT el corpus de
respuestas vive en `responses_rep{R}/{modelo}/`, pero la procedencia era un
único `run_provenance_rep{R}.json` por cohorte. Consecuencias:

1. El segundo modelo de una cohorte **no podía ejecutarse nunca** con `R > 1`
   (y con `R = 1` solo a costa de sobrescribir el registro del primero, que es
   lo que ya había pasado: `run_provenance.json` de `scale/` dice `medgemma:4b`
   aunque el corpus lo generaron tres modelos).
2. La alternativa que sugería el mensaje («usa otro `--tag`»), que es la correcta
   en PhysioNet, no existe en DREAMT: `--cohort` no separa modelos.

Arreglo:

- `provenance.repetition_provenance_name(R, slug)` →
  `run_provenance_rep{R}_{modelo}.json` (y `run_provenance_{modelo}.json` con
  `R = 1`). El nombre sin modelo queda como **legado de lectura**.
- `provenance.guard_corpus_engine(current_path, legacy_path, …)` compara la
  procedencia del modelo y, además, la antigua compartida **solo si su `model`
  coincide**: si no, describe otro corpus y abortar por ella impediría correr un
  segundo modelo. Un cambio de motor dentro del mismo modelo sigue abortando.
- `analyze_repetitions_external.dreamt_provenance_path()` lee la del modelo y cae
  a la antigua con la misma regla, para que las cifras publicadas de B1/B2
  conserven su motor y un modelo ajeno no pueda heredarlas.

`--tag` sigue siendo la separación correcta en PhysioNet, donde la procedencia y
el fichero de respuestas son por tag y mezclar modelos en el mismo fichero sí
sería un error. 5 tests nuevos (nombres por modelo, guardia que ignora la
procedencia ajena, guardia que aborta dentro del mismo modelo y resolución del
analizador).
