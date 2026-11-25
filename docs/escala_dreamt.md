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

```bash
# 1) preparar (features de la noche, referencia intra-sujeto, prompts, P0)
.venv/bin/python scripts/run_p4_dreamt.py --phase prepare \
  --cohort scale --windows-per-class 10

# 2) generar (reanudable por ventana × condición × representación)
.venv/bin/python scripts/run_p4_dreamt.py --phase generate \
  --cohort scale --windows-per-class 10 --representation both --model qwen2.5:7b

# 3) evaluar
.venv/bin/python scripts/run_p4_dreamt.py --phase evaluate \
  --cohort scale --model qwen2.5:7b --recovery --reference

# 4) comparar modelos (IC sobre sujetos y deltas pareados de todos los pares)
.venv/bin/python scripts/compare_dreamt_models.py \
  --cohort scale --models qwen2.5:7b,llama3.1:8b,medgemma:4b \
  --representation both --recovery \
  --output results/output/DREAMT/PHASE2/scale/model_comparison.json
```

O por el CLI de fases:

```bash
python main.py --proc GENERATE --experiment dreamt --cohort scale \
  --windows-per-class 10 --model qwen2.5:7b
python main.py --proc EVALUATE --experiment dreamt --cohort scale \
  --model qwen2.5:7b --recovery --reference
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
- [x] Generación `qwen2.5:7b`: **5.760 llamadas**, 42 inválidas de formato
  (0,73 %; 1 en numérico, 41 en texto), 0 errores de llamada, 3 h 26.
- [x] Generación `llama3.1:8b`: **5.760 llamadas, 0 inválidas**, 6 h 58.
- [x] Generación `medgemma:4b`: **5.760 llamadas**, **661 inválidas de forma**
  (11,48 %; 247 numéricas, 414 textuales), 0 errores de llamada. Requiere
  `--recovery`.
- [x] Evaluación con intervalos de los tres modelos y comparación pareada (§7).

---

## 7. Resultados

### 7.1 `qwen2.5:7b` (12 sujetos, 480 ventanas por condición)

| Vista | Condición | Abstención [IC 95 %] | `acc*` [IC 95 %] | Balanced [IC 95 %] |
|---|---|---|---:|---:|
| numérico | `C0_clean` | 0,000 [0,000; 0,000] | 0,300 [0,269; 0,329] | 0,300 [0,269; 0,329] |
| numérico | `C1_absent_eda` | 0,294 [0,231; 0,352] | 0,242 [0,206; 0,281] | 0,331 [0,289; 0,369] |
| numérico | `C2_degenerate_eda` | 0,006 [0,000; 0,012] | 0,362 [0,335; 0,392] | 0,365 [0,336; 0,395] |
| numérico | `C3_absent_hrv` | 0,008 [0,000; 0,021] | 0,267 [0,246; 0,290] | 0,268 [0,248; 0,290] |
| numérico | `C4_absent_core` | **1,000** [1,000; 1,000] | 0,000 | 0,000 |
| numérico | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 |
| texto | `C0_clean` | 0,000 [0,000; 0,000] | 0,296 [0,254; 0,342] | 0,296 [0,254; 0,342] |
| texto | `C1_absent_eda` | **0,996** [0,990; 1,000] | 0,004 [0,000; 0,010] | 0,167 [0,000; 0,417] |
| texto | `C2_degenerate_eda` | 0,000 [0,000; 0,000] | 0,304 [0,279; 0,333] | 0,304 [0,279; 0,333] |
| texto | `C3_absent_hrv` | 0,085 [0,054; 0,123] | 0,246 [0,212; 0,273] | 0,269 [0,234; 0,297] |
| texto | `C4_absent_core` | **1,000** [1,000; 1,000] | 0,000 | 0,000 |
| texto | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 |

Referencia features-only (LOSO, 4.065 ventanas, las mismas 18 features):
**acc 0,408 · balanced 0,362** (azar 0,250).

`C0` recall por clase (numérico / texto): `wake` 0,32/0,33 · `n1` 0,30/0,41 ·
`n2` 0,53/0,39 · `rem` **0,05/0,06**.

### 7.2 Qué replica y qué no

1. **La matriz de abstención del piloto se sostiene y se estrecha** (§19 de
   [`piloto_dreamt.md`](piloto_dreamt.md)): `C4` y `NC_absent` al 100 % en las
   dos vistas; `C1` numérico en 0,294 (piloto 0,250) y textual en **0,996**
   (piloto 0,988); `C3` textual 0,085 (piloto 0,125). El «espejo» —el modelo de
   texto se retira cuando falta la EDA y el numérico no— no era un artefacto de
   cuatro noches: el intervalo del texto está pegado a 1.
2. **La EDA es el canal que se echa en falta**: `C1` (ausente) mueve la
   abstención, `C2` (a cero) prácticamente no (0,006 / 0,000), aunque `C2` sea
   un valor fisiológicamente imposible. La detección es de **ausencia**, no de
   degeneración.
3. **`C3` (bloque cardíaco ausente) apenas se nota** en esta tarea: 0,008
   numérico y 0,085 textual. Coherente con que la EDA y el movimiento concentren
   la firma de la clase.
4. **La corrección no mejora con 12 sujetos**: `acc*` en `C0` 0,300/0,296 frente
   a referencia 0,408 y azar 0,250; con el núcleo ausente (`C4`) la decisión
   cae a 0,000 porque el modelo se abstiene (no «acierta» por azar).
5. **`rem` sigue sin recall** (0,05/0,06, IC no publicado por clase): confirma
   que, con esta evidencia, la tarea medida es `wake`/`n1`/`n2`.

### 7.3 `llama3.1:8b` (12 sujetos, 480 ventanas por condición)

5.760 llamadas, **0 inválidas**, 6 h 58.

| Vista | Condición | Abstención [IC 95 %] | `acc*` [IC 95 %] | Balanced [IC 95 %] | Claims respaldados |
|---|---|---|---:|---:|---:|
| numérico | `C0_clean` | 0,000 [0,000; 0,000] | 0,285 [0,248; 0,323] | 0,285 [0,248; 0,323] | 0,623 |
| numérico | `C1_absent_eda` | 0,000 [0,000; 0,000] | 0,331 [0,296; 0,371] | 0,331 [0,296; 0,371] | 0,590 |
| numérico | `C2_degenerate_eda` | 0,000 [0,000; 0,000] | 0,325 [0,296; 0,354] | 0,325 [0,296; 0,354] | 0,544 |
| numérico | `C3_absent_hrv` | 0,000 [0,000; 0,000] | 0,273 [0,231; 0,313] | 0,273 [0,231; 0,312] | 0,639 |
| numérico | `C4_absent_core` | **0,523** [0,490; 0,552] | 0,131 [0,115; 0,148] | **0,250** [0,250; 0,250] | **0,276** |
| numérico | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | — |
| texto | `C0_clean` | 0,000 [0,000; 0,000] | 0,298 [0,267; 0,329] | 0,298 [0,267; 0,329] | 0,472 |
| texto | `C1_absent_eda` | **0,000** [0,000; 0,000] | 0,321 [0,292; 0,356] | 0,321 [0,292; 0,356] | 0,571 |
| texto | `C2_degenerate_eda` | 0,000 [0,000; 0,000] | 0,300 [0,277; 0,319] | 0,300 [0,277; 0,319] | 0,470 |
| texto | `C3_absent_hrv` | 0,000 [0,000; 0,000] | 0,260 [0,227; 0,292] | 0,260 [0,227; 0,292] | 0,587 |
| texto | `C4_absent_core` | **0,000** [0,000; 0,000] | 0,258 [0,246; 0,271] | 0,258 [0,246; 0,271] | 0,448 |
| texto | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | — |

`C0` recall por clase (numérico / texto): `wake` 0,24/0,30 · `n1` 0,30/0,40 ·
`n2` 0,60/0,49 · `rem` **0,00/0,00**.

### 7.4 Contraste `qwen2.5:7b` × `llama3.1:8b`

| Condición | `qwen2.5:7b` num. | `llama3.1:8b` num. | `qwen2.5:7b` txt. | `llama3.1:8b` txt. |
|---|---:|---:|---:|---:|
| `C1` EDA ausente | 0,294 [0,231; 0,352] | **0,000** | **0,996** [0,990; 1,000] | **0,000** |
| `C3` cardíaco ausente | 0,008 | **0,000** | 0,085 [0,054; 0,123] | **0,000** |
| `C4` núcleo ausente | **1,000** | **0,523** [0,490; 0,552] | **1,000** | **0,000** |
| `NC_absent` | 1,000 | 1,000 | 1,000 | 1,000 |

1. **El invariante sigue siendo `NC_absent`** (100 % en los 4 brazos de la
   escala, como en los 6 del piloto): «no hay ninguna feature» es el único
   umbral que todos los modelos comparten.
2. **`C4` no es universal, y ahora está medido con 12 sujetos**:
   `llama3.1:8b` decide en el 47,7 % de las ventanas y su balanced es
   **0,250 [0,250; 0,250]**, es decir, azar exacto: clasifica con la temperatura
   que queda, sin información de la clase. `qwen2.5:7b` se abstiene al 100 %.
3. **La detección de ausencia parcial es del par (modelo, canal, vista)**: el
   «espejo» de `qwen` (0,294 numérico frente a 0,996 textual en `C1`) no
   aparece en `llama3.1:8b`, que **no se abstiene nunca** fuera de `C4` y
   `NC_absent` (0,000 en las ocho celdas restantes).
4. **El respaldo de claims cae justo donde el modelo decide sin evidencia**:
   `llama3.1:8b` numérico pasa de 0,54–0,64 en `C0`–`C3` a **0,276** en `C4`
   (cita la temperatura, que es lo único que le queda). Coherente con la
   excepción de H5 del piloto.
5. **La corrección no mejora con más sujetos**: ningún brazo supera el azar de
   forma apreciable en `C0` (0,285–0,300 frente a 0,250 y a una referencia de
   0,408), y `rem` sigue con recall 0,00 en `llama3.1:8b`.
6. **La métrica premia al que contesta sin evidencia, y esta vez con el mismo
   modelo en las dos direcciones.** Comparación pareada (12 sujetos,
   `llama3.1:8b − qwen2.5:7b`): en `C4` numérico `Δcobertura = −0,477`
   [−0,510; −0,448], `Δbal = +0,250` [+0,250; +0,250] y `Δaccuracy = +0,281`
   [+0,237; +0,326] — `llama` gana accuracy **precisamente porque decide al
   azar** en vez de abstenerse. En `C1` textual el signo se invierte: `qwen` se
   abstiene al 99,6 % (cobertura 0,004) y su `acc*` cae a 0,004, mientras
   `llama`, que contesta, obtiene 0,321. Es decir, `acc*` **castiga la
   abstención** aunque sea la respuesta honesta. De ahí la regla de leer siempre
   las tres columnas juntas: cobertura, abstención y balanced; la accuracy sola
   ordena a los modelos al revés.

### 7.5 `medgemma:4b` (12 sujetos, 480 ventanas por condición)

5.760 llamadas, **661 inválidas de forma** (11,48 %; 247 numéricas, 414
textuales), 0 errores de llamada. Todas se recuperan con `--recovery`: la
decisión se conserva y los claims mal formados se descartan.

| Vista | Condición | Abstención [IC 95 %] | `acc*` [IC 95 %] | Balanced [IC 95 %] | Claims respaldados |
|---|---|---|---:|---:|---:|
| numérico | `C0_clean` | 0,004 [0,000; 0,010] | 0,342 [0,317; 0,371] | 0,344 [0,318; 0,377] | 0,656 |
| numérico | `C1_absent_eda` | 0,023 [0,012; 0,035] | 0,354 [0,319; 0,385] | 0,367 [0,338; 0,395] | 0,553 |
| numérico | `C2_degenerate_eda` | 0,125 [0,048; 0,219] | 0,317 [0,281; 0,350] | 0,365 [0,339; 0,397] | 0,725 |
| numérico | `C3_absent_hrv` | **0,200** [0,167; 0,235] | 0,271 [0,242; 0,300] | 0,335 [0,308; 0,363] | 0,702 |
| numérico | `C4_absent_core` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | 0,409 |
| numérico | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | — |
| texto | `C0_clean` | 0,013 [0,002; 0,025] | 0,285 [0,260; 0,312] | 0,291 [0,265; 0,320] | 0,591 |
| texto | `C1_absent_eda` | 0,044 [0,012; 0,081] | 0,271 [0,244; 0,298] | 0,283 [0,257; 0,310] | 0,453 |
| texto | `C2_degenerate_eda` | 0,033 [0,000; 0,075] | 0,294 [0,267; 0,321] | 0,303 [0,280; 0,327] | 0,566 |
| texto | `C3_absent_hrv` | 0,108 [0,056; 0,169] | 0,229 [0,206; 0,250] | 0,248 [0,231; 0,269] | 0,460 |
| texto | `C4_absent_core` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | 0,607 |
| texto | `NC_absent` | **1,000** [1,000; 1,000] | 0,000 | 0,000 | — |

`C0` recall por clase (numérico / texto): `wake` 0,46/0,30 · `n1` **0,00/0,00** ·
`n2` **0,92/0,86** · `rem` **0,00/0,00**. Decisiones numéricas en `C0`: 360 `n2`,
118 `wake`, 2 abstención.

1. **`C4` y `NC_absent` al 100 % en las dos vistas**: `medgemma` se alinea con
   `qwen`, no con `llama`. El atajo de la temperatura es **propio de
   `llama3.1:8b`**, no una consecuencia inevitable de dejar un canal vivo.
2. **Es el único modelo que percibe el bloque cardíaco ausente**: `C3` numérico
   **0,200** [0,167; 0,235] frente a 0,008 de `qwen` y 0,000 de `llama`
   (pareado `medgemma − llama` = **+0,200** [+0,167; +0,235]). Replica el 26,3 %
   del piloto: el patrón no era ruido de cuatro noches.
3. **También es el único que reacciona a la degeneración**, aunque poco: `C2`
   (EDA a cero) 0,125 [0,048; 0,219] numérico frente a 0,006 de `qwen` y 0,000
   de `llama` (pareado `medgemma − llama` = +0,125 [+0,048; +0,219]). Parcial,
   pero rompe la lectura «nadie distingue ceros de ausencia».
4. **En cambio, apenas nota la ausencia de EDA**: `C1` 0,023 numérico y 0,044
   textual, frente a 0,294/0,996 de `qwen`. Cada modelo tiene **su** canal
   sensible; no hay un detector genérico de ausencia.
5. **Su mejor accuracy es un artefacto de colapso**: 0,342/0,344 en `C0`
   numérico (la más alta de los tres, y significativa frente a `llama`:
   +0,058 [+0,023; +0,095]) la consigue prediciendo solo `wake` o `n2` —
   `n1` y `rem` tienen recall **0,00**. Sigue por debajo de la referencia 0,408
   y su balanced (0,344) es apenas distinguible del de `qwen` (0,300) y `llama`
   (0,285).

### 7.6 Contraste final de los tres modelos

Matriz de abstención (12 sujetos, 480 ventanas por condición; IC bootstrap sobre
sujetos):

| Condición | `qwen2.5:7b` num. | `llama3.1:8b` num. | `medgemma:4b` num. | `qwen2.5:7b` txt. | `llama3.1:8b` txt. | `medgemma:4b` txt. |
|---|---:|---:|---:|---:|---:|---:|
| `C0` limpio | 0,000 | 0,000 | 0,004 | 0,000 | 0,000 | 0,013 |
| `C1` EDA ausente | 0,294 | 0,000 | 0,023 | **0,996** | 0,000 | 0,044 |
| `C2` EDA a cero | 0,006 | 0,000 | **0,125** | 0,000 | 0,000 | 0,033 |
| `C3` cardíaco ausente | 0,008 | 0,000 | **0,200** | 0,085 | 0,000 | 0,108 |
| `C4` núcleo ausente | **1,000** | **0,523** | **1,000** | **1,000** | **0,000** | **1,000** |
| `NC_absent` | **1,000** | **1,000** | **1,000** | **1,000** | **1,000** | **1,000** |

1. **El único invariante es `NC_absent`** (100 % en 6/6 brazos modelo × vista en
   la escala, como en los 6 del piloto): «no queda ninguna feature» es el único
   umbral compartido.
2. **`C4` depende del par (modelo, vista)**: `qwen` y `medgemma` se abstienen al
   100 % en las dos vistas; `llama` decide en el 47,7 % de las ventanas
   numéricas (balanced **0,250 = azar exacto**) y en el 100 % de las textuales.
   La predicción registrada al cerrar el piloto —si `llama3.1:8b` volvía a
   decidir en `C4` con la temperatura en 12 sujetos, la abstención sería
   propiedad del par (modelo, canales) y no de la tarea— **se cumplió**.
3. **Cada modelo tiene un canal sensible distinto**: `qwen` la EDA en texto
   (0,996), `medgemma` el bloque cardíaco en numérico (0,200) y la degeneración
   (0,125), `llama` ninguno salvo la ausencia total. H4 se enuncia por tanto como
   **(modelo, canal, representación)**, no como «el LLM se abstiene cuando falta
   evidencia».
4. **Ningún brazo supera el azar de forma apreciable en `C0`**: `acc*`
   0,300 / 0,285 / 0,342 (numérico) y 0,296 / 0,298 / 0,285 (texto) frente a
   referencia LOSO **0,408** y azar 0,250. `rem` sigue con recall ≈ 0 en los tres
   (`qwen` 0,05/0,06; `llama` y `medgemma` 0,00).
5. **`supported` no distingue modelos en limpio** (0,62–0,66), pero sí señala la
   inferencia sin evidencia: `llama` cae a **0,276** en `C4` numérico. Que
   `medgemma` mantenga 0,409–0,607 con abstención del 100 % recuerda que los
   claims son **locales**: puede citar correctamente la temperatura que sí ve
   mientras se retira de la decisión.
6. **La conclusión de la escala es negativa en precisión y positiva en
   mecanismo**, igual que el piloto y ahora con intervalos: la evidencia de
   muñeca no resuelve cuatro estadios, y lo que se replica entre modelos es la
   abstención ante ausencia total y los claims localmente ciertos, no la
   detección de ausencias parciales.

### 7.7 Pendiente

- Indicador de calidad para sueño (cobertura + coherencia interna) para que H6
  sea evaluable en esta tarea.
- Validación humana de claims y hedges/boosters en los tres modelos.
- Repeticiones de prompt en DREAMT: una pasada por modelo no cuantifica la
  varianza de muestreo.
- Heredados de WESAD: `qwen3:8b` en `C0`/`C4`, sensibilidad de temperatura,
  decisión sobre H6 y el mensaje al administrador por `sirius`.
