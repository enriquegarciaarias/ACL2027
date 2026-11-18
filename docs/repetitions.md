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
**ampliar sujetos** (S8–S10), no repeticiones: 1.440 llamadas, ~50 min.

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
