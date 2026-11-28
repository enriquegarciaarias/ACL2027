# Escala de DREAMT — cohorte de 12 sujetos

> **Estado**: cohorte **congelada** el 2026-11-24, antes de generar una sola
> respuesta de esta cohorte. `prepare` ejecutado (P0 8/8), generación en curso.
> Documento de preregistro de la escala; el piloto de 4 sujetos sigue siendo
> [`piloto_dreamt.md`](piloto_dreamt.md).

---

## 1. Pregunta

El piloto de 4 sujetos (S034, S051, S076, S084) dejó una matriz de abstención
muy contrastada (§19 de [`piloto_dreamt.md`](piloto_dreamt.md)): el modelo se
retira al desaparecer el **núcleo** de la evidencia, pero casi nunca ante la
ausencia de **un** canal, y esa sensibilidad depende de la representación y del
modelo.

Con cuatro sujetos no se puede afirmar que eso sea una propiedad de la tarea:
las tasas por sujeto tienen una dispersión que el rango [mín–máx] no cuantifica
y el efecto podría venir de dos noches atípicas. La escala responde a:

> ¿Se sostiene la matriz de abstención y el desacople evidencia↔decisión al
> ampliar la cohorte, y con qué precisión?

---

## 2. Cohorte congelada

### 2.1 Regla de selección

Se fijó **antes** de generar y es ciega a los resultados:

1. **Disponibilidad**: sujetos con **≥ 12 ventanas puras** (60 s, pureza 1,0,
   N3 excluido) en **cada una** de las cuatro clases (`wake`, `n1`, `n2`,
   `rem`). Solo usa el etiquetado de sueño; no interviene ninguna respuesta del
   LLM.
2. **Exclusión** de los sujetos de desarrollo (`S002`, `S010`, `S026`) y del
   piloto (`S034`, `S051`, `S076`, `S084`).
3. **Reparto uniforme**: de la lista ordenada resultante —**50 sujetos**— se
   toman **12** en posiciones `round(i·(n−1)/11)`, `i = 0…11`, para no
   concentrar la cohorte en un tramo de identificadores (el orden de
   identificador puede correlacionar con el reclutamiento).

Medición de disponibilidad: recuento de ventanas puras por clase sobre los 100
sujetos, leyendo **solo** la columna `Sleep_Stage` (127 s, sin GPU ni LLM).

### 2.2 Sujetos y disponibilidad

| Sujeto | `wake` | `n1` | `n2` | `rem` |
|---|---:|---:|---:|---:|
| S008 | 20 | 13 | 202 | 65 |
| S015 | 23 | 20 | 254 | 92 |
| S021 | 158 | 31 | 104 | 30 |
| S035 | 25 | 17 | 226 | 54 |
| S039 | 29 | 161 | 135 | 49 |
| S045 | 148 | 26 | 160 | 15 |
| S061 | 34 | 12 | 208 | 45 |
| S067 | 61 | 40 | 202 | 84 |
| S071 | 88 | 19 | 169 | 21 |
| S086 | 94 | 43 | 181 | 36 |
| S093 | 42 | 35 | 242 | 35 |
| S103 | 52 | 75 | 175 | 15 |

Contexto del dataset: **59** de los 100 sujetos cumplen ≥ 10 ventanas por clase
y **55** cumplen ≥ 12; `rem` es la clase escasa (30 sujetos no tienen ninguna
ventana REM pura). La cohorte se apoya en el criterio ≥ 12 para que el muestreo
de 10 por clase no quede al límite.

---

## 3. Diseño

| Elemento | Piloto | **Escala** |
|---|---|---|
| Sujetos | 4 | **12** (tabla §2.2) |
| Ventanas puras por clase y sujeto | 20 | **10** |
| Ventanas por sujeto | 80 | **40** |
| Condiciones | 6 | 6 (`C0`, `C1`, `C2`, `C3`, `C4`, `NC_absent`) |
| Representaciones | numérica y textual | numérica y textual |
| Prompts por representación | 480 | **2.880** |
| Llamadas por modelo | 960 | **5.760** |

Invariantes que **no** cambian:

- Few-shot y pre-chequeo **P0** siguen calculándose con los sujetos de
  desarrollo (`S002`, `S010`, `S026`); la cohorte de escala nunca alimenta el
  prompt. P0: **8/8** comprobaciones OK antes de generar.
- Referencia de estandarización: mediana/IQR de **la propia noche** del sujeto,
  como en el piloto.
- Matriz de condiciones, exclusiones de features y reglas del prompt: idénticas
  (ver [`piloto_dreamt.md`](piloto_dreamt.md) §5–§6).
- La cohorte de escala **no se fusiona** con la del piloto en ningún informe:
  son dos corpus con `n` distinto por sujeto y muestreo distinto.

### 3.1 Desviación declarada

Se reduce de **20 a 10 ventanas por clase y sujeto**. Motivo: el coste por
modelo se duplica con 20 (11.520 llamadas) y la pregunta de la escala es de
**generalización entre sujetos**, no de resolución intra-sujeto. El piloto se
mantiene con 20 y se reporta por separado. La consecuencia: las tasas por
sujeto de la escala tienen más ruido de muestreo; por eso el análisis publica
intervalos **sobre sujetos**, que es la unidad de generalización.

---

## 4. Qué se mide

| Hipótesis | Métrica de la escala |
|---|---|
| **H4** (abstención) | Tasa de abstención por condición, con IC 95 % bootstrap sobre **sujetos** (2.000 remuestreos). Es la hipótesis que motivó la escala. |
| **H1** (corrección) | Accuracy sobre decididas, accuracy con la abstención como error (`acc*`), balanced y cobertura; referencia features-only LOSO con las mismas 18 features. |
| **H3** (representación) | Diferencia numérico − texto por condición, con IC bootstrap sobre sujetos. |
| **H5** (claims) | Respaldo de claims, coherencia con la decisión y `unrecognized_metric_names` (residual del verificador). |
| **H2a/H2b** | AUROC de confianza, sobreconfianza e índice de certeza, separando decididas de abstenidas. |

### 4.1 Reglas de lectura (fijadas antes de medir)

1. **Accuracy bruta no decide nada**: con prevalencia cercana a 0,3 y
   abstención selectiva, un clasificador degenerado puede subir la accuracy.
   Se reporta siempre junto a `balanced` y `acc*`.
2. **El n efectivo es el número de sujetos** (12), no de ventanas. Los
   intervalos se calculan re-muestreando sujetos.
3. `rem` con recall 0 se documenta como límite de la evidencia, no como error.
4. Si la escala contradice al piloto, se reporta la contradicción: no se
   re-elige la cohorte ni la métrica.

---

## 5. Ejecución

La ejecución publicada (§7) se lanzó en el **servidor** con el lanzador y
`--subject DREAMT`; los comandos de abajo son la versión local equivalente,
válida para desarrollo. Reglas de nodo y de Ollama en
[`ejecucion_slurm.md`](ejecucion_slurm.md).

```bash
# 1) preparar (features de la noche, referencia intra-sujeto, prompts, P0)
.venv/bin/python scripts/run_p4_dreamt.py --phase prepare --cohort scale --windows-per-class 10

# 2) generar (reanudable por ventana × condición × representación)
.venv/bin/python scripts/run_p4_dreamt.py --phase generate --cohort scale --windows-per-class 10 --representation both --model qwen2.5:7b

# 3) evaluar
.venv/bin/python scripts/run_p4_dreamt.py --phase evaluate --cohort scale --model qwen2.5:7b --recovery --reference

# 4) comparar modelos (IC sobre sujetos y deltas pareados de todos los pares)
.venv/bin/python scripts/compare_dreamt_models.py --cohort scale --models qwen2.5:7b,llama3.1:8b,medgemma:4b --representation both --recovery --output results/output/DREAMT/PHASE2/scale/model_comparison.json
```

O por el CLI de fases (en el servidor, por `sbatch` o el lanzador; ver
[`ejecucion_slurm.md`](ejecucion_slurm.md)):

```bash
python main.py --proc GENERATE --experiment dreamt --subject DREAMT --cohort scale --windows-per-class 10 --model qwen2.5:7b
python main.py --proc EVALUATE --experiment dreamt --subject DREAMT --cohort scale --model qwen2.5:7b --recovery --reference
```

Artefactos en `results/output/DREAMT/PHASE2/scale/` (gitignorado):
`features/`, `reference/`, `labels/`, `prompts/`, `p0.json`, `manifest.json`,
`responses/{modelo}/`, `evaluation_*.json` y `model_comparison.json`
(media y rango por modelo, `ci95` por condición y
`deltas.<vista>.pairs` con **todos** los pares: `delta`, `ci` bootstrap sobre
sujetos y `n_subjects`).

---

## 6. Estado

- [x] Cohorte congelada (§2) y documentada antes de generar.
- [x] `prepare`: 15 sujetos (3 de desarrollo + 12 de escala), 2.880 prompts por
  representación, P0 8/8.
- [x] Generación en el **servidor** (Ollama 0.40.6), 5.760 llamadas por modelo:
  `qwen2.5:7b` 69 inválidas (1,2 %), 2 h 00 · `llama3.1:8b` 0 inválidas, 2 h 16
  · `medgemma:4b` **1.144 inválidas** (19,9 %; 618 irrecuperables con
  `--recovery`), 3 h 57.
- [x] Evaluación con intervalos de los tres modelos y comparación pareada (§7).
- [ ] Repeticiones de prompt (§7.8): una pasada por modelo. El run local previo
  (0.30.7) queda solo como registro de desarrollo y no se cita (§7.6).

---

## 7. Resultados (servidor, Ollama 0.40.6)

Todas las cifras de esta sección están generadas en el **servidor** (partición
`A5000`, Ollama **0.40.6**), que es el motor de los resultados citables; el run
local con 0.30.7 queda como desarrollo y su diferencia se declara en §7.6.

Procedencia: generación en los jobs 16630 (`qwen2.5:7b`, 2 h 00), 16653
(`llama3.1:8b`, 2 h 16) y 16692 (`medgemma:4b`, 3 h 57); evaluaciones 16699,
16700 y 16702; comparación pareada 16704. Cada brazo escribe
`run_provenance.json` con versión de Ollama, `model_digest`, tamaño y parámetros
por defecto del modelo.

Convenciones de lectura (fijadas antes de medir, §4.1):

- 480 ventanas por condición y modelo (12 sujetos × 40).
- `abs` (abstención) y `cov` (cobertura) dividen por **todas** las ventanas;
  `inv` es la salida irrecuperable, que no cuenta ni como decisión ni como
  abstención. Cuando `abs + cov = 1` el corpus no tiene inválidas.
- `acc*` = accuracy con la abstención como error = `acc × cov`.
- Tasa base 0,25 (cuatro clases); referencia LOSO **acc 0,408 · balanced 0,362**
  (4.065 ventanas, las mismas 18 features).

### 7.1 Formato y recuperación

| Modelo | Inválidas al generar | % de 5.760 | Irrecuperables tras `--recovery` | % |
|---|---:|---:|---:|---:|
| `qwen2.5:7b` | 69 | 1,2 | 0 | 0,0 |
| `llama3.1:8b` | 0 | 0,0 | 0 | 0,0 |
| `medgemma:4b` | 1.144 | 19,9 | 618 | 10,7 |

`medgemma:4b` falla en la **forma** del claim (`direction: "high"`,
`channel: "fraction"`); la recuperación tolerante conserva la decisión y
descarta el claim mal formado. El fallo no es uniforme: se concentra donde
escasea la evidencia — 143 irrecuperables en `C2` numérico, 134 en `C3`
numérico y 109 en `C4` textual. Ahí su cobertura deja de ser comparable.

### 7.2 Abstención

Fracción de las 480 ventanas de cada condición:

| Condición | `qwen` num. | `llama` num. | `medgemma` num. | `qwen` txt. | `llama` txt. | `medgemma` txt. |
|---|---:|---:|---:|---:|---:|---:|
| `C0` limpio | 0,025 | 0,000 | 0,073 | 0,004 | 0,000 | 0,092 |
| `C1` EDA ausente | **0,879** | 0,013 | 0,171 | **1,000** | 0,000 | 0,185 |
| `C2` EDA a cero | 0,310 | 0,000 | 0,254 | 0,054 | 0,000 | 0,242 |
| `C3` cardíaco ausente | 0,458 | 0,002 | **0,565** | 0,462 | 0,000 | 0,310 |
| `C4` núcleo ausente | 1,000 | 1,000 | 0,983 | 1,000 | **0,952** | 0,773 ¹ |
| `NC_absent` | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |

¹ el 22,7 % restante son inválidas irrecuperables: `medgemma` no decide, pero
tampoco emite una abstención bien formada.

IC 95 % sobre sujetos del brazo **numérico** (`abs`): `qwen` 0,025
[0,010; 0,040] · 0,879 [0,852; 0,908] · 0,310 [0,148; 0,477] · 0,458
[0,406; 0,515]; `llama` 0,000 · 0,013 [0,002; 0,025] · 0,000 · 0,002
[0,000; 0,006]; `medgemma` 0,073 [0,054; 0,094] · 0,171 [0,131; 0,210] ·
0,254 [0,175; 0,335] · 0,565 [0,506; 0,625]. En texto, las celdas extremas son
exactas: `qwen` `C1` = 1,000 [1,000; 1,000] y `llama` `C4` = 0,952
[0,929; 0,971].

### 7.3 Corrección y utilidad

`acc*` (abstención como error) y balanced, por condición:

| Condición | `qwen` acc\* | `llama` acc\* | `medgemma` acc\* | `qwen` bal | `llama` bal | `medgemma` bal |
|---|---:|---:|---:|---:|---:|---:|
| `C0` | 0,315 | 0,298 | 0,304 | 0,323 | 0,298 | 0,356 |
| `C1` | **0,052** | 0,329 | 0,290 | 0,354 | 0,335 | 0,375 |
| `C2` | 0,254 | 0,329 | 0,177 | 0,353 | 0,329 | **0,443** |
| `C3` | 0,144 | 0,269 | **0,067** | 0,284 | 0,270 | **0,417** |
| `C4` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |
| `NC_absent` | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 | 0,000 |

IC 95 % de `acc*` (numérico): `qwen` `C0` [0,273; 0,362] · `C1` [0,035; 0,073]
· `C2` [0,198; 0,317] · `C3` [0,119; 0,169]; `llama` [0,258; 0,335] ·
[0,300; 0,360] · [0,300; 0,358] · [0,229; 0,308]; `medgemma` [0,273; 0,337] ·
[0,252; 0,327] · [0,117; 0,238] · [0,050; 0,085].

Texto: `acc*` en `C0` 0,275 / 0,275 / 0,240 y en `C1` **0,000** / 0,321 / 0,212.
El `acc*` 0,010 de `llama` en `C4` textual se calcula sobre cobertura 0,048
(23 ventanas): no se cita.

Ningún brazo alcanza la referencia en limpio (0,30 frente a 0,408), y en DREAMT
no hay referencia por condición —a diferencia de WESAD—, así que el coste se
mide contra `C0` y contra el techo de la tarea, no como exceso sobre una
referencia degradada.

### 7.4 Tres modos de fallo sobre la misma evidencia

El resultado de la escala no es un ranking sino una **tipología**: con la misma
degradación, los tres modelos fallan de forma distinta y medible.

1. **Retirada graduada** (`qwen2.5:7b`). Se retira en proporción a lo que falta
   (`C1` 0,879 · `C3` 0,458 · `C2` 0,310 en numérico; `C1` 1,000 · `C3` 0,462 ·
   `C2` 0,054 en texto) y su utilidad se hunde donde se retira: `acc*` 0,052
   [0,035; 0,073] en `C1`. Detecta la ausencia de EDA, pero no da servicio.
   Su reducción de accuracy solo es significativa en `C1` y `C3`; en `C2`
   (valor imposible) se mantiene a nivel de `C0`.
2. **Invariancia** (`llama3.1:8b`). No se retira nunca fuera de `C4`/`NC`
   (≤ 0,013) y su `acc*` va de 0,298 a 0,329 a 0,269 con **IC solapados**: no
   pierde corrección porque **nunca usó la evidencia**. Es la forma limpia de lo
   que en el run local se leyó como «atajo de la temperatura»; aquí no decide en
   `C4` (0,952 de abstención textual), simplemente responde con la misma
   distribución en todas las condiciones.
3. **Selección** (`medgemma:4b`). Es el único cuya accuracy **sobre lo que
   conserva** sube mientras cae la cobertura: `C2` 0,460 con cobertura 0,448
   (balanced 0,443 [0,363; 0,562]) y `C3` 0,458 con cobertura 0,156 (balanced
   0,417 [0,333; 0,491]). En `C3` su balanced está por encima del azar
   (0,25) y su intervalo incluye la referencia (0,362); frente a `qwen`
   `Δacc = +0,194 [+0,087; +0,300]` y frente a `llama` `+0,189 [+0,052; +0,316]`,
   absteniéndose 57 puntos más.

El precio de esa selección es la métrica: `Δacc*` de `medgemma` frente a `qwen`
en `C3` es **−0,077 [−0,096; −0,060]** y frente a `llama` **−0,202**. Es decir,
**el modelo que mejor selecciona es el que peor puntúa en `acc*`**, porque cada
abstención correcta se contabiliza como error. `acc*` ordena a los modelos por
cobertura, no por calidad de la decisión.

### 7.5 Claims, coherencia y certeza

Respaldo (`supported`) y coherencia (fracción de claims cuya dirección favorece
la clase decidida) por condición:

| Condición | `qwen` sop / coh | `llama` sop / coh | `medgemma` sop / coh |
|---|---|---|---|
| num. `C0` | 0,663 / 0,831 | 0,579 / 0,944 | 0,694 / 0,947 |
| num. `C2` | 0,684 / 0,860 | 0,493 / 0,966 | 0,754 / 0,895 |
| num. `C3` | 0,619 / 0,659 | 0,607 / 0,927 | 0,760 / 0,985 |
| num. `C4` | — | — | 0,463 / — |
| txt. `C0` | 0,548 / 0,869 | 0,433 / 0,902 | 0,520 / 0,978 |
| txt. `C3` | 0,488 / 0,816 | 0,561 / 0,933 | 0,593 / 1,000 |
| txt. `C4` | — | 0,551 / — | 0,252 / — |

1. **El respaldo no cae donde cae la decisión**: `medgemma` mantiene 0,463 de
   respaldo en `C4` numérico con abstención 0,983, y `llama` 0,551 en `C4`
   textual. Los claims son **locales**: puede citar correctamente la temperatura
   que sí ve mientras retira la decisión.
2. **La coherencia es alta incluso cuando la decisión es mala** (`llama` 0,93–
   0,97 en todas las condiciones): el modelo cita evidencia que *favorece* su
   respuesta, y su respuesta sigue estando al nivel del azar. Coherencia mide
   consistencia interna, no corrección.
3. **`—` significa «sin claims evaluables»**: cuando la respuesta es una
   abstención limpia no hay claims que verificar; no es un fallo del
   verificador.
4. **La certeza lingüística es plana** en las respuestas decididas: índice 0,00
   en casi todas las celdas de los tres modelos, y negativo solo en las
   abstenciones (`qwen` `C1` −28,57; `llama` `C4` textual −5,72; `medgemma`
   `C3` −0,98). Coincide con WESAD: la certeza no informa de la calidad de la
   evidencia.

### 7.6 Qué cambió respecto del run local (0.30.7)

La estructura **cualitativa** replica y el **nivel** no:

| Celda | Local 0.30.7 | Servidor 0.40.6 |
|---|---:|---:|
| `qwen` num. `C1` (EDA ausente) | 0,294 | **0,879** |
| `qwen` num. `C2` (EDA a cero) | 0,006 | **0,310** |
| `qwen` num. `C3` (cardíaco) | 0,008 | **0,458** |
| `medgemma` num. `C1` / `C3` | 0,023 / 0,200 | 0,171 / **0,565** |
| `llama` num. `C4` (núcleo) | 0,523 | **1,000** |
| `llama` txt. `C4` | 0,000 | **0,952** |

1. **El perfil de canal se conserva**: `qwen` reacciona a la EDA (`C1`) y
   `medgemma` al bloque cardíaco (`C3`) en los dos motores. Eso es lo que
   sostiene la lectura de H4.
2. **El nivel de abstención es del motor**: el 0.40.6 se retira entre 3 y 30
   veces más en las condiciones parciales. Un cambio de motor mueve una métrica
   de 0,29 a 0,88 sin tocar el prompt ni la semilla.
3. **Un hallazgo local no replica**: `llama` decidía en `C4` con la temperatura
   y su balanced era azar exacto. En el servidor se retira al 95–100 %. El
   «atajo de la temperatura» era del motor 0.30.7, no del modelo; **no se cita**.
4. Por eso todos los números publicados de DREAMT son los de esta sección y los
   locales quedan solo como registro de desarrollo (§19 de
   [`piloto_dreamt.md`](piloto_dreamt.md)).

### 7.7 Lectura para la tesis

- **H1 (coste)**: ningún brazo alcanza la referencia en limpio (0,30 vs 0,408) y
  la degradación no produce un coste común sino tres perfiles. El coste de
  `qwen` es de cobertura; el de `llama`, nulo en corrección porque su salida ya
  era independiente de la evidencia; el de `medgemma`, de cobertura con
  corrección retenida por encima del azar.
- **H4 (abstención)**: el único umbral compartido es «ninguna feature» (100 % en
  `C4`/`NC_absent` en los seis brazos). Por debajo, la abstención es
  **(modelo × canal × representación)** y ahora está medida con intervalos.
- **H5 (respaldo)**: el respaldo de claims no acompaña a la corrección; en
  `medgemma` incluso sube de `C0` a `C3` (0,694 → 0,760) mientras su `acc*` cae
  de 0,304 a 0,067.
- **H2 (calibración/certeza)**: la certeza es plana en las decididas; la
  confianza ya se documentó con dos valores en el piloto. DREAMT no aporta
  resolución nueva a H2a.
- **H6**: no evaluable en esta tarea; no hay indicador de calidad propio para
  sueño.

### 7.8 Pendiente

- Indicador de calidad para sueño (cobertura + coherencia interna) para que H6
  sea evaluable en esta tarea.
- Validación humana de claims y hedges/boosters en los tres modelos.
- Repeticiones de prompt en DREAMT: una pasada por modelo no cuantifica la
  varianza de muestreo. **Mecanismo implementado (2026-11-28)**: `--repetitions
  R` en `scripts/run_p4_dreamt.py` (salida `responses_rep{R}/`, reanudación por
  ventana×condición×rep, semillas `--seed + r`, guardia de motor) y análisis en
  `scripts/analyze_repetitions_external.py`.
  **B1 y B2 medidos** (`qwen` `C1`, R=3, numérico y texto): numérico con
  abstención 0,885 frente a 0,879 publicada, `acc*` 0,052 frente a 0,052,
  acuerdo modal 0,988 y `sd` de accuracy 0,001; texto con acuerdo modal
  **1,000**, `cov1` **0,000** (abstención 1,000, como la publicada) y 0
  inválidas. Las dos vistas son **deterministas** en esta condición, así que los
  dos brazos se cierran con nota y la diferencia 0,885 vs 1,000 entre formatos
  no es varianza de muestreo. Falta B3 (`medgemma` `C3`). Detalle en
  [`repetitions.md`](repetitions.md) §9.7.
- Heredados de WESAD: `qwen3:8b` en `C0`/`C4`, sensibilidad de temperatura,
  decisión sobre H6 y el mensaje al administrador por `sirius`.
