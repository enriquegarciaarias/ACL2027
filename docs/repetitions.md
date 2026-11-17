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
- **Condiciones**: `C0_clean` (referencia), `C4_total_eda` (donde el 7b
  sobre-reclama estrés) y `C5_total_both` (degradación máxima). Se excluye
  `C2` del piloto por coste; si la varianza depende de la condición, C4/C5 son
  los extremos donde debe verse.
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

| Fase | Sujetos | Ventanas | Condiciones | R | Llamadas | Tiempo aprox. |
|---|---|---:|---:|---:|---:|---|
| Pre-piloto | 1 (S5) | 12 | 2 (C0, C5) | 5 | 120 | ~10–15 min |
| Piloto | 3 (S5–S7) | 12 | 3 | 10 | 1.080 | ~1,5–2 h |
| Corpus completo (si procede) | 12 | 424 en total | 2 | según piloto | 424 × 2 × R | 5 h × R/5 |

---

## 7. Reproducción

```bash
# Pre-piloto (servidor; regla 8 de AGENTS.md: lanzador desde ~/data)
~/data/launch_lsi.sh --proc run_ACL.sh --time 04:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --subjects S5 --sample-windows 12 --conditions C0_clean,C5_total_both \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --repetitions 5 --model qwen2.5:7b

# Análisis (minutos, sin LLM)
python scripts/analyze_repetitions.py \
  --dir results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp_rep5/qwen2_5_7b \
  --output results/output/WESAD/PHASE2/repetitions_prepilot_report.json
```

Sin `--output-dir` (bandera del script, no del CLI), el barrido escribe en
`degradation_binary_schmidt_notemp_rep{R}/{modelo}/`. Lanzado desde
`scripts/run_degradation_sweep.py` se puede forzar otro destino con
`--output-dir`.

El análisis deduce la tarea y la variante de la procedencia de las respuestas y
descarta las ventanas fuera de la variante (`meditation` en `schmidt`).

---

## 8. Resultados

### 8.1 Pre-piloto (S5, 12 ventanas, C0 + C5, R = 5, `qwen2.5:7b`)

120 llamadas, 1 inválida de formato (0,8 %). Tras descartar `meditation`
(variante `schmidt`) quedan **10 ventanas por condición** (5 baseline, 3 stress,
2 amusement).

| Condición | Acuerdo modal | Deterministas | Entropía | sd(conf) intra | Acc 1 pasada | sd | Balanced | Cobertura |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_clean` | **1,000** | 100 % | 0,000 | 0,00 | 0,900 | **0,000** | 0,929 | 1,000 |
| `C5_total_both` | **0,960** | 80 % | 0,144 | 1,40 | 0,911 | **0,044** | 0,933 | 0,920 |

Extrapolación en C5: **k ≈ 5** para ±2 pp y **k ≈ 3** para ±3 pp. En C0 no hay
varianza que reducir.

**Las dos únicas ventanas que cambian** (ambas en C5, ninguna en C0):

| Ventana | Repeticiones | Acuerdo |
|---|---|---|
| una de baseline | `non-stress` × 1, `stress` × 4 | 0,80 |
| una de baseline | `non-stress` × 1, `abstain` × 4 | 0,80 |

Es decir: la varianza aparece **solo cuando la evidencia está destruida**, y
aparece como un desplazamiento hacia `stress` o hacia la abstención, no como
ruido simétrico.

### 8.2 Hallazgo que no dependía del muestreo: la confianza es una etiqueta

Distribución de los 120 valores de confianza:

| Valor | Registros | Situación |
|---:|---:|---|
| 85 | 112 | decisión tomada |
| 50 | 4 | `abstain` |
| 75 | 3 | decisión tomada (caso raro) |
| 85 (inválida) | 1 | fallo de vocabulario |

**La confianza no varía con la evidencia, varía con la decisión.** Consecuencia
directa para H2a:

- Promediar repeticiones **no crea resolución**: en las ventanas decididas, la
  confianza agregada toma **un solo valor** (85).
- El AUROC de confianza coincide con el AUROC del signo de la decisión. En este
  piloto sale 0,93 porque el modelo acertó; en el corpus salía 0,50 porque
  predecía todo igual. No mide calibración: mide el acierto.
- Por tanto, **las repeticiones no arreglan H2a en `qwen2.5:7b`**. Debe citarse
  como propiedad del modelo (el número es una etiqueta, no una probabilidad),
  no como ruido de decodificación.

### 8.3 Aplicación de la regla predeclarada

| Condición | Acuerdo modal | Banda | Decisión |
|---|---:|---|---|
| `C0_clean` | 1,000 | ≥ 0,98 | determinista: no requiere repeticiones |
| `C5_total_both` | 0,960 | 0,90–0,98 | intermedia: `R` por **±3 pp** → **R = 3** (mínimo 3) |

Con la ley `sd_1/√k`, `R = 3` da ±2,5 pp en C5 y `R = 5` da ±2,0 pp.

### 8.4 Límites de esta lectura

- **Un solo sujeto y 10 ventanas por condición**: la `sd` se estima con 5
  valores y su propio error relativo es ~35 %. El orden de magnitud es fiable;
  el decimal no.
- **`C4_total_eda` no se midió**, y es la condición que sostiene el exceso de
  coste de H1 (+0,317 en el 7b) y donde el modelo sobre-reclama estrés.
- **`qwen3:14b` no se ha medido**: su abstención es dependiente de la etiqueta
  (47 % de las ventanas de estrés en C4), así que su varianza puede ser mucho
  mayor y no se puede extrapolar desde el 7b.

### 8.5 Fallo corregido

Una respuesta se invalidó por `"direction": "slight increase"`. El validador ya
toleraba `decreased`/`increased`, pero no el adverbio antepuesto. Añadido
`_normalize_direction` (slight, sharply, markedly, somewhat, moderately…), con
tests. Afecta solo a ejecuciones futuras.

---

### 8.6 Piloto (S5–S7, 12 ventanas, C0 + C4 + C5, R = 5, `qwen2.5:7b`)

540 llamadas, **0 errores**. Tras descartar `meditation`: **30 ventanas por
condición** (S5–S7).

| Condición | Acuerdo modal | Deterministas | Ventanas que cambian | Acc 1 pasada | sd | Balanced |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 0,993 | 96,7 % | 1/30 | 0,860 | 0,013 | 0,900 |
| `C4_total_eda` | 0,987 | 93,3 % | 2/30 | 0,480 | 0,027 | 0,629 |
| `C5_total_both` | **0,947** | **76,7 %** | **7/30** | 0,614 | 0,014 | 0,720 |

Convergencia (accuracy, media ± sd entre subconjuntos de `k`):

| k | C0 | C4 | C5 |
|---:|---|---|---|
| 1 | 0,860 ±0,013 | 0,480 ±0,027 | 0,614 ±0,014 |
| 2 | 0,865 ±0,002 | 0,480 ±0,016 | 0,627 ±0,018 |
| 3 | 0,867 ±0,000 | 0,467 ±0,000 | 0,610 ±0,016 |
| 5 | 0,867 ±0,000 | 0,467 ±0,000 | 0,621 ±0,000 |

#### El resultado importante: el agregado es robusto, la ventana no

La inestabilidad **no es ruido simétrico**. Las 10 ventanas que cambian en las
tres condiciones son casi todas de clase **`non-stress`** empujadas hacia
`stress`:

| Condición | Ventana (clase real) | Reparto |
|---|---|---|
| C0 | amusement | `non-stress` 4 / `stress` 1 |
| C4 | baseline | `stress` 4 / `non-stress` 1 |
| C4 | baseline | `stress` 4 / `non-stress` 1 |
| C5 | baseline | `stress` 4 / `non-stress` 1 |
| C5 | baseline | `stress` 4 / `non-stress` 1 |
| C5 | baseline | `non-stress` 4 / `abstain` 1 |
| C5 | baseline | `non-stress` 4 / `stress` 1 |
| C5 | amusement | `non-stress` 3 / `stress` 2 ← **único 3:2** |
| C5 | amusement | `non-stress` 4 / `stress` 1 |
| C5 | amusement | `stress` 4 / `abstain` 1 |

Dos lecturas, ambas necesarias:

1. **Agregado estable**: 6 de los 7 cambios de C5 son 4:1, así que la mayoría se
   mantiene y la accuracy se mueve 1–2 pp entre repeticiones. El exceso de coste
   de H1 (+0,317) **no es un artefacto de semilla**.
2. **Ventana frágil**: en C5 el **23 % de las ventanas** cambia su respuesta en
   al menos una de cinco repeticiones, y el sentido del cambio es siempre el
   mismo (falso positivo de `stress`). Cualquier afirmación *por ventana* —«esta
   ventana es un falso positivo», «este claim es correcto»— tiene una fiabilidad
   de aproximadamente 4:1, no de certeza.

La segunda lectura es la que conecta con H5: los claims se verifican ventana a
ventana, y una decisión 4:1 no sostiene una claim individual.

#### Confianza: se confirma que es una etiqueta

| Condición | Valores distintos (1 pasada → 5 reps) | AUROC 1 pasada | AUROC agregado |
|---|---:|---:|---:|
| `C0_clean` | 2 → 2 | 0,914 | 0,929 |
| `C4_total_eda` | **1 → 1** | 0,629 | 0,619 |
| `C5_total_both` | 2 → 1 | 0,720 | 0,725 |

En C4 la confianza toma **un solo valor** y su AUROC (0,629) coincide con el
balanced accuracy (0,629): la «confianza» es el signo de la decisión. Promediar
repeticiones no la convierte en probabilidad.

#### Aplicación de la regla

| Condición | Acuerdo modal | Banda | Decisión |
|---|---:|---|---|
| `C0_clean` | 0,993 | ≥ 0,98 | determinista |
| `C4_total_eda` | 0,987 | ≥ 0,98 | determinista |
| `C5_total_both` | 0,947 | 0,90–0,98 | intermedia → **R = 3** (suelo de la regla) |

La extrapolación sugiere `k ≈ 1–2` para ±2 pp, pero con `sd` estimada sobre 5
valores; el suelo de 3 de la regla protege frente a esa imprecisión.

#### Comprobación de cordura

El piloto reproduce los dos patrones del corpus completo, con cifras algo más
bajas por composición de la muestra: C4 por debajo de la tasa base (0,480 con
prevalencia 0,70; balanced 0,629 frente a 0,637 del corpus) y C5 con balanced
0,720 frente a 0,723. La caída no es de la muestra: es la firma del modelo.

---

### 8.7 Pre-piloto de `qwen3:14b` (S5, 12 ventanas, C0 + C5, R = 5)

120 llamadas, **0 errores**, 9 min 51 s (4,9 s/llamada). 10 ventanas útiles por
condición tras descartar `meditation`.

| Condición | Acuerdo modal | Deterministas | sd(conf) intra | Conf. constante | Valores distintos | Acc 1 pasada | sd |
|---|---:|---:|---:|---:|---:|---:|---:|
| `C0_clean` | **1,000** | 100 % | 0,64 | 70 % | **5** | 0,900 | 0,000 |
| `C5_total_both` | **1,000** | 100 % | 0,20 | 90 % | 3 | 1,000 | 0,000 |

Ni una sola decisión cambió entre las cinco semillas, en ninguna de las dos
condiciones. La varianza está **toda** en la confianza.

#### El contraste que ordena la interpretación

| | `qwen2.5:7b` | `qwen3:14b` |
|---|---|---|
| Decisión entre semillas | varía en C5 (7/30 ventanas) | **no varía nunca** |
| Confianza entre semillas | fija (85 decidido / 50 abst.) | **varía** (sd 0,64; 5 valores) |
| Promediar repeticiones | no aporta resolución | reduce ruido, no cambia el AUROC |
| Abstención en C5 | 3,3 % | **30,0 %** |
| Accuracy / balanced en C5 | 0,614 / 0,720 | **1,000 / 0,500** |

Cada modelo tiene un solo grado de libertad y es el contrario del otro: **el 7b
es determinista en la confianza y ruidoso en la decisión; el 14b es determinista
en la decisión y ruidoso en la confianza.**

#### La trampa de la accuracy, confirmada como determinista

En C5 el 14b da **accuracy 1,000 con balanced 0,500** y cobertura 0,70: se
abstiene en las tres ventanas de estrés, decide las siete restantes y acierta
todas. Repetir no lo cambia ni una vez. La cifra de 0,845 del corpus de 424 no
era una casualidad de semilla: es el comportamiento estable del modelo.

#### Confianza: aquí sí hay resolución, y no hace falta promediarla

| Condición | Valores distintos (1 pasada → 5 reps) | AUROC 1 pasada | AUROC agregado |
|---|---:|---:|---:|
| `C0_clean` | 5 → 5 | 1,000 | 1,000 |
| `C5_total_both` | 3 → 2 | 0,500 | 0,500 |

El 14b ya emite confianza graduada (75, 85, 90, 95, 50), así que la media de
repeticiones no añade resolución: reduce el ruido de ±1-2 puntos sin mover el
AUROC. **H2a no mejora repitiendo en ninguno de los dos modelos**, por razones
distintas: en el 7b porque el número es una etiqueta; en el 14b porque ya era
informativo.

#### Regla aplicada

| Condición | Acuerdo modal | Banda | Decisión |
|---|---:|---|---|
| `C0_clean` | 1,000 | ≥ 0,98 | determinista |
| `C5_total_both` | 1,000 | ≥ 0,98 | determinista |

**El 14b no necesita corpus de repeticiones.** Para el 7b, la única condición
fuera de la banda determinista es C5 (0,947) → **R = 3 y solo en C5**
(424 × 3 = 1.272 llamadas, ~1,3 h), que es el alcance que la regla autoriza.

#### Límite

El 14b se ha medido en **un sujeto** (S5) y dos condiciones. Es suficiente para
la pregunta que se hizo —¿cambia la respuesta con la semilla?— porque la
respuesta fue «ni una vez en 10 ventanas × 5 semillas × 2 condiciones», pero no
para estimar una tasa de inestabilidad con precisión.
