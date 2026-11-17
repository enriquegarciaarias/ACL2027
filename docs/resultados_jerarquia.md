# Jerarquía binaria `stress` vs `non-stress` — resultados (diseño primario)

**Fecha:** 2026-11-17
**Diseño:** [`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md)
(preregistro). Variante **`schmidt`** (`baseline` + `amusement`), **sin
temperatura**, bootstrap agrupado por sujeto.
**Corpus:** 12 sujetos de **evaluación** (S5–S17 salvo S12), 567 ventanas
recortadas a **424** (se descartan 143 de `meditation`) × 5 condiciones =
2.835 respuestas por modelo.
**Modelos:** `qwen2.5:7b` y `qwen3:14b`, T=0,3, seed 42, few-shot de S2–S4.
**Artefactos:**
`results/output/WESAD/PHASE2/degradation_binary_schmidt_notemp/{qwen2_5_7b,qwen3_14b}/`.
**Informes:** `evaluation_binary_hierarchy_{qwen2_5_7b,qwen3_14b}.json`.

---

## 0. Estado

| Bloque | Estado |
|---|---|
| Barrido `GENERATE` (dos modelos) | **Completo**: 2.835/2.835 cada uno; 0 errores de llamada; 3 respuestas inválidas en el 7b (0,11 %), recuperadas en el análisis |
| Clasificación por condición (§2–§4) | **Completo** |
| H1, H2a, H2b, H4, H5 y H6 (§5) | **Completo** (`degradation_report_schmidt_{qwen2_5_7b,qwen3_14b}.json`) |
| Comparación pareada de modelos (§6) | **Completo** (`model_comparison_schmidt.json`); deltas de `acc*`/`balanced` con IC al re-ejecutar el comparador actualizado |

Este documento es el informe del **diseño primario**. Las cifras proceden de
`scripts/analyze_degradation.py` sobre las 2.835 respuestas de cada modelo; la
tabla de clasificación de §2 se contrastó con
`scripts/analyze_binary_hierarchy.py` y coincide salvo ±0,001 en tres celdas del
7b (las 3 respuestas recuperadas).

---

## 1. Composición

| Clase real | Ventanas | Papel en `schmidt` |
|---|---:|---|
| `stress` | 127 | clase positiva |
| `baseline` | 157 | `non-stress` |
| `amusement` | 140 | `non-stress` |
| `meditation` | 143 | **fuera de la tarea** (descartada) |

Prevalencia de `stress`: **0,300** ⇒ **tasa base (siempre `non-stress`) =
0,700** y **balanced = 0,500**. Sin esas dos referencias, la accuracy no se
puede interpretar.

---

## 2. Corrección, cobertura y calibración (424 por condición)

`acc` = accuracy sobre decididas; `acc*` = contando la abstención como error;
`cov` = cobertura; `bal` = balanced accuracy (decididas).

| Condición | Modelo | acc | acc* | cov | bal | prec | rec | AUROC | ECE | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 7b | 0,908 | 0,908 | 1,000 | 0,907 | 0,810 | 0,906 | 0,912 | 0,068 | 0,086 |
| `C0_clean` | 14b | **0,915** | 0,915 | 1,000 | 0,888 | 0,889 | 0,819 | **0,964** | 0,125 | 0,072 |
| `C1_partial_resp` | 7b | 0,915 | 0,913 | 0,998 | 0,905 | 0,841 | 0,881 | 0,906 | 0,064 | 0,082 |
| `C1_partial_resp` | 14b | 0,917 | 0,917 | 1,000 | 0,892 | 0,890 | 0,827 | **0,972** | 0,120 | 0,067 |
| `C2_partial_eda` | 7b | 0,887 | 0,884 | 0,998 | **0,854** | 0,838 | 0,772 | 0,859 | **0,051** | 0,101 |
| `C2_partial_eda` | 14b | 0,945 | 0,736 | **0,778** | 0,804 | 1,000 | 0,609 | 0,922 | 0,114 | 0,061 |
| `C4_total_eda` | 7b | 0,548 | 0,547 | 0,998 | **0,637** | 0,387 | 0,858 | 0,638 | **0,298** | 0,336 |
| `C4_total_eda` | 14b | 0,824 | 0,705 | **0,856** | 0,522 | 1,000 | 0,045 | 0,746 | 0,093 | 0,137 |
| `C5_total_both` | 7b | 0,682 | 0,682 | 1,000 | **0,723** | 0,482 | 0,827 | 0,727 | 0,168 | 0,244 |
| `C5_total_both` | 14b | 0,826 | 0,696 | **0,842** | 0,508 | 1,000 | 0,016 | 0,761 | 0,115 | 0,138 |

IC 95 % por bootstrap agrupado por sujeto (2.000 remuestras):

| Condición | Modelo | acc [IC] | balanced [IC] | AUROC [IC] |
|---|---|---|---|---|
| `C0_clean` | 7b | 0,908 [0,859; 0,952] | 0,907 [0,834; 0,961] | 0,912 [0,841; 0,966] |
| `C0_clean` | 14b | 0,915 [0,866; 0,960] | 0,888 [0,809; 0,957] | 0,964 [0,932; 0,991] |
| `C4_total_eda` | 7b | 0,548 [0,469; 0,626] | 0,637 [0,571; 0,706] | 0,638 [0,561; 0,716] |
| `C4_total_eda` | 14b | 0,824 [0,759; 0,903] | 0,522 [0,500; 0,580] | 0,746 [0,657; 0,846] |
| `C5_total_both` | 7b | 0,682 [0,609; 0,759] | 0,723 [0,645; 0,804] | 0,727 [0,649; 0,809] |
| `C5_total_both` | 14b | 0,826 [0,760; 0,906] | 0,508 [0,500; 0,534] | 0,761 [0,665; 0,862] |

---

## 3. Dónde se abstiene cada modelo

Abstenciones por clase real (de 127 `stress` y 297 `non-stress`):

| Condición | Modelo | Abstención total | Sobre `stress` | Sobre `non-stress` |
|---|---|---:|---:|---:|
| `C2_partial_eda` | 14b | 94 (22,2 %) | **81 (63,8 %)** | 13 (4,4 %) |
| `C4_total_eda` | 14b | 61 (14,4 %) | **60 (47,2 %)** | 1 (0,3 %) |
| `C5_total_both` | 14b | 67 (15,8 %) | **64 (50,4 %)** | 3 (1,0 %) |
| todas | 7b | 0 | 0 | 0 |

El 14b se retira **casi solo cuando la ventana es de estrés y la señal que lo
distingue ha desaparecido** (SCR en C2; EDA completa en C4/C5). No es abstención
por ruido ni por clase: es abstención **dependiente de la etiqueta**.

---

## 4. Lectura

1. **En limpio, empate en corrección y reparto de perfil.** 0,908 vs 0,915 con
   IC solapados. El 14b rankea mejor (AUROC 0,964 vs 0,912) y acierta más cuando
   dice `stress` (precisión 0,889 vs 0,810); el 7b recupera más `stress`
   (recall 0,906 vs 0,819) y está **mejor calibrado** (ECE 0,068 vs 0,125).

2. **La pérdida de respiración es casi gratuita** (C1): 0,915 y 0,917, igual que
   en limpio. El 7b incluso mejora 0,007. Confirma que el modelo no usa la
   respiración de forma apreciable.

3. **Con EDA parcial (C2) los modelos divergen.** El 7b apenas se mueve
   (0,887; cobertura 1,00). El 14b sube la accuracy a 0,945 pero **bajando la
   cobertura al 0,778**: sobre las decididas su precisión es 1,000 y su recall
   0,609. Su `acc*` cae a 0,736, apenas por encima de la tasa base (0,700).

4. **Con EDA a cero (C4/C5) fallan en direcciones opuestas.**
   - **7b**: sobrerreclama estrés (282 y 218 predicciones de 424). Precisión
     0,387 / 0,482; accuracy 0,548 / 0,682. En C4 está **por debajo de la tasa
     base** (0,700), aunque su balanced (0,637 / 0,723) indica que conserva algo
     de señal.
   - **14b**: infra-reclama. En C4 hace 3 predicciones de `stress` en 424
     (todas correctas: precisión 1,000) y en C5 una sola. Balanced 0,522 / 0,508
     con IC que casi tocan 0,500: **clasificador degenerado**.

5. **La accuracy del 14b en C4/C5 es un artefacto de cobertura.** Sobre
   decididas da 0,824 / 0,826; contando la abstención como error cae a **0,705 /
   0,696**, es decir, al nivel de "decir siempre `non-stress`". Por eso este
   documento reporta siempre `acc*`, `cov` y `bal` juntas: sin ellas, abstenerse
   parece acertar.

6. **El AUROC de C4/C5 no es comparable entre modelos** con esta implementación:
   se calcula solo sobre decididas y el 14b decide un 84–86 % sesgado al
   negativo. Para C0–C2 (cobertura ≈ 1) sí es interpretable.

7. **La variante no cambia la conclusión** (sensibilidades declaradas, C0 / C4):

   | Variante | n | 7b acc C0 | 14b acc C0 | 7b acc C4 | 14b acc C4 | 14b bal C4 |
   |---|---:|---:|---:|---:|---:|---:|
   | `schmidt` | 424 | 0,908 | 0,915 | 0,548 | 0,824 | 0,522 |
   | `baseline` | 355 | 0,924 | 0,915 | 0,590 | 0,783 | 0,522 |
   | `no_amusement` | 498 | 0,904 | 0,930 | 0,529 | 0,854 | 0,522 |
   | `all` | 567 | 0,894 | 0,928 | 0,505 | 0,874 | 0,522 |

   El colapso del 14b con EDA destruida es **estructural** (balanced 0,522 en
   las cuatro variantes), no un efecto de qué clases entren en `non-stress`.

8. **Frente al corpus histórico (567, variante `all` + temperatura)**, el 7b
   mejora al fijar el diseño primario: C5 pasa de balanced 0,613 / accuracy
   0,495 a **0,723 / 0,682**; C4 se queda igual (0,63 → 0,637). Parte del daño
   que se atribuía al 7b era de la temperatura y de las clases ajenas a la
   tarea, no de la degradación. El 14b reproduce el mismo patrón degenerado
   (0,500 → 0,508).

9. **Las hipótesis (§5) sostienen H1, H4 y H5 y matizan H2a, H2b y H6.** En
   limpio el LLM **iguala a la referencia supervisada** (0,908 / 0,915 frente a
   0,905); con EDA destruida el exceso de coste es **+0,317 (C4) y +0,205 (C5)**
   para el 7b y **+0,166 / +0,198** para el 14b medido con `acc*`. El respaldo de
   claims **sube** (0,768 → 0,811 y 0,876 → 0,915) mientras la decisión se
   hunde. La confianza solo informa en el 14b (AUROC 0,87 → 0,73); el 7b emite
   dos valores y su sobreconfianza llega a +0,300. La certeza lingüística modula
   en el 14b (−0,25 → −0,85 sobre decididas) y no en el 7b. Y el SQI **no
   sustituye** a la confianza en este corpus (−0,139 en C0 y −0,042 en C5 del
   7b): el +0,243 que se citaba era del corpus de 567.

---

## 5. H1, H2, H4, H5 y H6 en el diseño primario

### 5.1 H1 — coste en exceso frente a la referencia

La referencia es una regresión logística con CV agrupada por sujeto sobre las
**mismas 424 ventanas** y las **19 features de pecho sin temperatura** (lo
único que cambia es el lector).

| Condición | Ref. binaria | 7b acc | 7b acc* | 14b acc | 14b acc* |
|---|---:|---:|---:|---:|---:|
| `C0_clean` | 0,905 | 0,908 | 0,908 | 0,915 | 0,915 |
| `C1_partial_resp` | 0,903 | 0,915 | 0,915 | 0,917 | 0,917 |
| `C2_partial_eda` | 0,912 | 0,887 | 0,887 | 0,945 | 0,736 |
| `C4_total_eda` | 0,861 | 0,547 | 0,547 | 0,824 | 0,705 |
| `C5_total_both` | 0,884 | 0,682 | 0,682 | 0,826 | 0,696 |

- **En limpio el LLM iguala a la referencia** (0,908 / 0,915 frente a 0,905): no
  hay techo de información que explique el fallo posterior.
- La referencia apenas se mueve (0,905 → 0,861 en C4: −0,044; → 0,884 en C5:
  −0,021), mientras el LLM se hunde.
- **Exceso de coste** (Δ LLM − Δ referencia): **+0,317 en C4** y **+0,205 en C5**
  para el 7b; **+0,166** y **+0,198** para el 14b medido con `acc*` (si se mide
  sobre decididas, el 14b «solo» pierde 0,091 en C4 y el exceso se oculta).
- **H1 apoyada**, y con más margen que en el corpus de 4 clases (0,15–0,20).

### 5.2 H4 — abstención frente a error

| Condición | 7b error (1−acc*) | 7b abst. | Brecha | 14b error (1−acc*) | 14b abst. | Brecha |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 0,092 | 0,000 | +0,092 | 0,085 | 0,000 | +0,085 |
| `C1_partial_resp` | 0,085 | 0,000 | +0,085 | 0,083 | 0,000 | +0,083 |
| `C2_partial_eda` | 0,113 | 0,000 | +0,113 | 0,264 | 0,222 | +0,042 |
| `C4_total_eda` | 0,453 | 0,000 | **+0,453** | 0,295 | 0,144 | **+0,151** |
| `C5_total_both` | 0,318 | 0,000 | **+0,318** | 0,304 | 0,158 | **+0,146** |

- **El 7b no se abstiene nunca** (0/424 en las cinco condiciones): falla el 45 %
  en C4 sin retirarse una sola vez.
- **El 14b se retira, pero no por dificultad**: su abstención se concentra en
  ventanas de estrés (§3), no en las que responde mal; la brecha sigue siendo
  +0,15 en C4/C5.
- Desaparece la **sobre-abstención en limpio** que se veía con el prompt de 4
  clases (31,7 % en C0): era un efecto del espacio de etiquetas, no del modelo.
- **H4 no se sostiene en su forma fuerte**: la abstención no audita la evidencia.

### 5.3 H5 — respaldo de afirmaciones

| Condición | 7b claims | 7b respaldadas | 7b dirección | 14b claims | 14b respaldadas | 14b dirección |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 2.544 | 0,768 | 0,850 | 2.918 | 0,876 | 0,947 |
| `C1_partial_resp` | 2.305 | 0,780 | 0,865 | 2.498 | 0,885 | 0,938 |
| `C2_partial_eda` | 1.940 | 0,740 | 0,811 | 2.583 | 0,842 | 0,921 |
| `C4_total_eda` | 2.317 | 0,794 | 0,868 | 2.995 | 0,915 | 0,951 |
| `C5_total_both` | 2.052 | 0,811 | 0,882 | 2.628 | 0,912 | 0,946 |

- **El respaldo sube con la degradación** (7b 0,768 → 0,811; 14b 0,876 →
  0,915) mientras la decisión se hunde; el acierto de dirección es 0,81–0,95.
- **`unsupported` = 0** en las dos series: el modelo deja de citar los canales
  ausentes, no inventa métricas.
- Los claims son **localmente ciertos** (EDA a cero ⇒ «conductancia muy
  disminuida» es verdad) y la inferencia global es falsa. Es el hallazgo central
  de la tesis, replicado ahora en el diseño primario.

### 5.4 H2a — la confianza como predictor de acierto

| Condición | 7b AUROC | 7b conf. media | 7b sobreconf. | 14b AUROC | 14b conf. media | 14b sobreconf. |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 0,537 | 84,9 | −0,059 | **0,867** | 84,5 | −0,071 |
| `C1_partial_resp` | 0,503 | 85,0 | −0,065 | **0,896** | 86,1 | −0,056 |
| `C2_partial_eda` | 0,540 | 84,9 | −0,038 | 0,781 | 74,6 | −0,200 |
| `C4_total_eda` | 0,513 | 84,7 | **+0,300** | 0,731 | 80,9 | −0,015 |
| `C5_total_both` | 0,511 | 84,9 | +0,168 | 0,755 | 81,5 | −0,011 |

- **El 7b emite dos valores de confianza** (75 y 85): su AUROC es azar por
  construcción y su ECE es puro sesgo (0,849 − accuracy).
- **El 14b sí informa** (0,87–0,90 en limpio) y pierde resolución al degradar
  (0,73–0,78), pero **no se sobreconfía** (sobreconfianza ≤ 0).
- La confianza media del 14b **baja** en C2 (74,6) porque las abstenidas no
  entran en el cálculo; en C4/C5 vuelve a 81 por selección de las decididas.

### 5.5 H2b — certeza lingüística

| Condición | 7b índice | 7b decididas | 7b hedges | 14b índice | 14b decididas | 14b abstenciones |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | −0,58 | −0,58 | 0,34 | −0,25 | −0,25 | — |
| `C1_partial_resp` | −0,34 | −0,34 | 0,25 | −0,34 | −0,34 | — |
| `C2_partial_eda` | −0,77 | −0,77 | 0,44 | −1,02 | −0,54 | **−2,71** |
| `C4_total_eda` | −0,71 | −0,71 | 0,39 | −0,84 | −0,59 | **−2,37** |
| `C5_total_both` | −0,31 | −0,31 | 0,25 | −0,82 | −0,85 | −0,67 |

- **El 7b no modula su lenguaje**: es negativo ya en limpio (−0,58), no es
  monótono (C2 −0,77; C5 −0,31) y sus *boosters* son residuales (~0,12).
- **El 14b sí modula, pero hay que medirlo sobre las decididas**: −0,25 → −0,85
  es monótono; el índice global exagera la caída porque el **texto del veto de
  las abstenidas es mucho más cauto** (−2,7 y −2,4).
- Es la misma trampa que en E9 (4 clases): sin separar decididas y abstenciones,
  H2b mide la redacción del veto, no el juicio sobre el afecto.
- **El nivel absoluto depende del prompt**: el mismo 7b tenía +0,13 en el corpus
  de 4 clases y −0,58 aquí. El índice compara condiciones **dentro** de una
  tarea; no es comparable entre tareas ni entre modelos.

### 5.6 H6 — indicadores de calidad

| Condición | 7b conf. | 7b Δgate | 7b ΔSQI | 14b conf. | 14b Δgate | 14b ΔSQI |
|---|---:|---:|---:|---:|---:|---:|
| `C0_clean` | 0,526 | 0,000 | **−0,139** | 0,889 | 0,000 | +0,024 |
| `C1_partial_resp` | 0,500 | 0,000 | +0,074 | 0,917 | 0,000 | +0,001 |
| `C2_partial_eda` | 0,527 | 0,000 | **+0,119** | 0,841 | 0,000 | +0,052 |
| `C4_total_eda` | 0,511 | 0,000 | +0,064 | 0,729 | 0,000 | **+0,112** |
| `C5_total_both` | 0,512 | 0,000 | −0,042 | 0,750 | 0,000 | +0,052 |

- **La compuerta de consistencia aporta exactamente 0** en las diez celdas: es
  un veto, no una señal de orden. Su valor está en anular datos incoherentes
  (0 % de falsos positivos en limpio), no en predecir acierto.
- **El SQI es pequeño e inestable en el 7b** (−0,139 en limpio, +0,119 en C2,
  −0,042 en C5). Con 424 ventanas de 12 sujetos, no sustituye de forma fiable a
  una confianza que no informa.
- En el 14b aporta poco (+0,001 a +0,112) y **donde más, cuando su confianza ya
  ha perdido resolución** (C4: 0,729 → 0,842).
- **H6 queda reformulada**: el SQI no «sustituye» a la confianza de forma
  general. El resultado de +0,243 que se citaba venía del corpus de 567
  ventanas con la variante `all` y temperatura; con el diseño primario **no se
  replica** y hay que citarlo como dependiente del corpus.

---

### 5.7 Resumen del diseño primario frente al corpus histórico

| Hipótesis | Corpus histórico (567, `all`+temp) | Diseño primario (424, `schmidt` sin temp) |
|---|---|---|
| H1 | Exceso 0,15–0,20 | **Exceso 0,21–0,32** (más claro) |
| H2a | 7b AUROC 0,51 → 0,76; sobreconfianza +0,47 | 7b AUROC ≈ 0,51 fija; sobreconfianza +0,30. **14b sí informa** (0,87 → 0,73) |
| H2b | 7b +0,11 → −0,99 (monótono) | 7b no monótono (−0,58 → −0,31). **14b decididas −0,25 → −0,85** |
| H4 | Infra-abstención, brecha +0,58 | **Brecha +0,45 (7b, sin abstenerse) y +0,15 (14b)** |
| H5 | 0,78 → 0,83 | **0,768 → 0,811 (7b) y 0,876 → 0,915 (14b)** |
| H6 | ΔSQI hasta +0,24 (7b) | **ΔSQI inestable en 7b; +0,11 máximo en 14b**; Δgate = 0 |

La conclusión robusta es **H1, H4 y H5**; **H2a, H2b y H6 dependen del modelo**
y del corpus, y así hay que declararlo. La comparación entre modelos (§6)
añade un tercer eje: **quién gana depende de la métrica** (accuracy bruta →
14b; balanced → 7b; calibración → 7b; respaldo de claims y ranking de
confianza → 14b).

## 6. Comparación pareada `qwen2.5:7b` vs `qwen3:14b`

Bootstrap agrupado por **sujeto** (2.000 remuestras, 12 sujetos) sobre las mismas
ventanas. Cuatro métricas pareadas, porque con prevalencia 0,30 la accuracy y la
discriminación de la clase positiva no dicen lo mismo:

- `accuracy` (decididas) y `balanced` (decididas) coinciden con las columnas
  `acc` y `bal` de §2.
- `acc*` y `balanced (abst.=error)` cuentan la abstención como fallo: son las
  versiones comparables entre modelos con coberturas distintas.

| Condición | Δ accuracy (decididas) [IC 95 %] | Δ acc* [IC 95 %] | Δ balanced (decididas) | Δ balanced (abst.=error) [IC 95 %] |
|---|---:|---:|---:|---:|
| `C0_clean` | +0,007 [−0,024; +0,042] | +0,007 [−0,024; +0,042] | −0,019 | −0,020 [−0,067; +0,022] |
| `C1_partial_resp` | +0,002 [−0,019; +0,021] | +0,002 [−0,019; +0,021] | −0,014 | −0,014 [−0,051; +0,014] |
| `C2_partial_eda` | **+0,059** [+0,023; +0,101] | **−0,151** [−0,205; −0,094] | −0,050 | **−0,265** [−0,356; −0,173] |
| `C4_total_eda` | **+0,277** [+0,197; +0,348] | **+0,158** [+0,076; +0,238] | −0,115 | **−0,126** [−0,188; −0,063] |
| `C5_total_both` | **+0,145** [+0,084; +0,211] | +0,014 [−0,057; +0,083] | −0,215 | **−0,224** [−0,293; −0,153] |

Signo positivo = el 14b es mejor. La Δ de `balanced (decididas)` es la
diferencia exacta de la columna `bal` de §2 (su IC sale del mismo bootstrap y
está en `model_comparison_schmidt.json`).

**Lectura.**

1. **En limpio y con pérdida de respiración no hay diferencia**: +0,007
   [−0,024; +0,042] y +0,002 [−0,019; +0,021], y el balanced estricto también
   incluye el 0. El empate no es falta de datos.
2. **En C2 el signo depende del coste de abstenerse**: +0,059 [+0,023; +0,101] si
   la abstención es gratis, **−0,151** [−0,205; −0,094] si cuesta, y **−0,265**
   en balanced estricto. El 14b compra su accuracy retirándose del 64 % de las
   ventanas de estrés.
3. **En C4 gana en las dos accuracies y pierde en balanced**: +0,277 y +0,158
   [+0,076; +0,238], pero −0,126 [−0,188; −0,063]. Es el retrato exacto del
   clasificador degenerado: acierta 296 de 297 ventanas de `non-stress` y 3 de
   127 de estrés; con prevalencia 0,30 eso basta para 0,824 de accuracy.
4. **En C5 el `acc*` se anula** (+0,014 [−0,057; +0,083]) mientras el balanced
   cae −0,224: al contar la abstención, la ventaja desaparece.
5. **La «accuracy a cobertura comparable» no rescata la comparación**: en C4 da
   0,540 (7b) vs 0,824 (14b) a la misma cobertura, pero mide **selectividad**
   (qué bien la confianza ordena los aciertos), no discriminación de clase. El
   14b tiene confianza informativa (§5.4) y el 7b solo dos valores; con
   prevalencia 0,30 un 0,824 es compatible con balanced ≈ 0,51.
6. **Conclusión operativa**: la elección de modelo **depende de la métrica y del
   coste declarado**. Para detección de estrés con prevalencia 0,30 → **7b**
   (balanced). Para ranking por confianza o respaldo de claims → **14b**
   (dirección 0,92–0,95). Para accuracy bruta → 14b, pero por el artefacto de la
   clase mayoritaria. La regla de decisión debe fijarse antes de comparar
   modelos; con estos datos, reportar solo accuracy es indefendible.

## 7. Amenazas a la validez

- **424 ventanas de 12 sujetos**: la unidad de generalización es el sujeto
  (n efectivo ≈ 12); los IC por ventana serían demasiado estrechos. Los IC de §2
  remuestrean sujetos, no ventanas.
- **Una sola semilla y una sola repetición**: no se mide la varianza de muestreo
  del modelo (pendiente: repeticiones de prompt). Con una abstención del 14b del
  14–22 %, parte de esa tasa puede ser ruido de muestreo.
- **Confianza de dos valores en el 7b** (75 y 85): H2a y ECE no son estimables
  con rigor en ese brazo; su AUROC ≈ 0,51 refleja la falta de resolución, no
  necesariamente incapacidad del modelo para ordenar ventanas.
- **H6 con 424 ventanas**: el ΔSQI se estima con una regresión logística en CV
  por sujeto; con 12 sujetos su varianza es alta y explica el signo negativo en
  C0/C5 del 7b. No se debe citar como propiedad del indicador sin repeticiones.
- **`amusement` ≈ `baseline`** fisiológicamente: parte de `non-stress` es
  indistinguible, lo que facilita la tarea en esa dirección.
- **El orden del protocolo** sigue confundido con la clase en WESAD; la
  exclusión de temperatura y de `meditation` lo reduce, no lo elimina.
- **La abstención del 14b** es informativa en la dirección correcta pero no se
  convierte en detección: precisión alta con recall 0,016–0,045 no es un
  detector.
- **Claims verificados automáticamente**: el verificador aplica intervalos de
  compatibilidad sobre los z-scores del prompt; no hay validación humana de una
  muestra (pendiente).

---

## 8. Reproducción

```bash
# 1) Barrido (servidor; regla 8 de AGENTS.md: lanzador desde ~/data)
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --model qwen3:14b
# repetir con --model qwen2.5:7b; los jobs son reanudables por (ventana, condición)

# 2) Clasificación por condición y variantes (sin LLM)
python main.py --proc EVALUATE --experiment degradation --model qwen3:14b \
  --hierarchy --bootstrap 2000 \
  --output results/output/WESAD/PHASE2/evaluation_binary_hierarchy_qwen3_14b.json

# 3) H1, H2a, H2b, H4, H5 y H6 (sin LLM; minutos)
.venv/bin/python scripts/analyze_degradation.py --model qwen2.5:7b \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --output results/output/WESAD/PHASE2/degradation_report_schmidt_qwen2_5_7b.json
.venv/bin/python scripts/analyze_degradation.py --model qwen3:14b \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --output results/output/WESAD/PHASE2/degradation_report_schmidt_qwen3_14b.json

# 4) Comparación pareada (bootstrap por sujeto + accuracy a cobertura comparable)
python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b \
  --mode degradation --decision-mode binary --nonstress-variant schmidt \
  --temperature exclude \
  --output results/output/WESAD/PHASE2/model_comparison_schmidt.json
```

Documentos relacionados: [`resultados_binario.md`](resultados_binario.md)
(corpus histórico `all` + temperatura, 567 ventanas),
[`resultados_degradacion.md`](resultados_degradacion.md) (4 clases, 7b),
[`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md)
(preregistro y desviaciones), [`experimentos.md`](experimentos.md) (inventario).
