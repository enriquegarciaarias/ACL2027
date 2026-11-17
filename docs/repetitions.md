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
