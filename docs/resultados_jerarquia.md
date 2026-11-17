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
| Barrido `GENERATE` (dos modelos) | **Completo**: 2.835/2.835 cada uno; 0 errores de llamada; 3 respuestas inválidas en el 7b (0,11 %) |
| Clasificación por condición (este documento) | **Completo** |
| H1 (referencia con las mismas ventanas y features), H2a (AUROC de confianza), H2b (certeza), H4 (brecha error−abstención), H5 (respaldo de claims) y H6 (ΔSQI) | **Pendiente de ejecución** con el analizador ampliado (ver §5) |

Las cifras de este documento son de **clasificación**. El resto de hipótesis
necesita las features de `INGEST` y el verificador de claims, que viven en el
servidor; el análisis ya está preparado para el diseño primario.

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

---

## 5. Cómo se obtiene el resto de H1–H6 (pendiente)

El analizador de degradación ya entiende el diseño primario: resuelve el
directorio `degradation_binary_schmidt_notemp/{modelo}`, **deduce la tarea y la
variante de la procedencia del corpus**, descarta las ventanas de `meditation` y
calcula la referencia features-only **sin temperatura** para que vea lo mismo
que el LLM.

```bash
# Servidor, sin GPU ni Ollama (minutos)
python scripts/analyze_degradation.py --model qwen2.5:7b \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --output results/output/WESAD/PHASE2/degradation_report_schmidt_qwen2_5_7b.json

python scripts/analyze_degradation.py --model qwen3:14b \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --output results/output/WESAD/PHASE2/degradation_report_schmidt_qwen3_14b.json
```

Salida esperada (tablas de consola y bloques JSON):

- `reference_by_condition` → H1: `balanced_accuracy_binary` con las mismas 424
  ventanas y las 19 features sin `temp`.
- `conditions[*].claims` → H5 (`supported`/`contradicted`/`unsupported`/
  `indeterminate`, acierto de dirección).
- `conditions[*].certainty` → H2b (índice global, decididas y abstenciones).
- `conditions[*].confidence_auroc_correctness` → H2a.
- `conditions[*].coverage` / `accuracy_incl_abstain` → H4 (brecha error −
  abstención).
- `indicator_incremental` → H6 (ΔSQI y Δgate).

Comparación pareada de modelos (`paired_bootstrap_delta` remuestrea **sujetos**,
no ventanas; incluye accuracy a cobertura comparable):

```bash
python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b \
  --mode degradation --decision-mode binary --nonstress-variant schmidt \
  --temperature exclude
```

---

## 6. Amenazas a la validez

- **424 ventanas de 12 sujetos**: la unidad de generalización es el sujeto
  (n efectivo ≈ 12); los IC por ventana serían demasiado estrechos.
- **Una sola semilla y una sola repetición**: no se mide la varianza de muestreo
  del modelo (pendiente: repeticiones de prompt).
- **`amusement` ≈ `baseline`** fisiológicamente: parte de `non-stress` es
  indistinguible, lo que facilita la tarea en esa dirección.
- **El orden del protocolo** sigue confundido con la clase en WESAD; la
  exclusión de temperatura y de `meditation` lo reduce, no lo elimina.
- **La abstención del 14b** es informativa en la dirección correcta pero no se
  convierte en detección: precisión alta con recall 0,016–0,045 no es un
  detector.

---

## 7. Reproducción del corpus

```bash
# Servidor (regla 8 de AGENTS.md: lanzador desde ~/data)
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- \
  --proc GENERATE --experiment degradation --subject WESAD \
  --decision-mode binary --nonstress-variant schmidt --temperature exclude \
  --model qwen3:14b
# repetir con --model qwen2.5:7b; los jobs son reanudables por (ventana, condición)

# Evaluación de la jerarquía (sin LLM)
python main.py --proc EVALUATE --experiment degradation --model qwen3:14b \
  --hierarchy --bootstrap 2000 \
  --output results/output/WESAD/PHASE2/evaluation_binary_hierarchy_qwen3_14b.json
```

Documentos relacionados: [`resultados_binario.md`](resultados_binario.md)
(corpus histórico `all` + temperatura, 567 ventanas),
[`resultados_degradacion.md`](resultados_degradacion.md) (4 clases, 7b),
[`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md)
(preregistro y desviaciones), [`experimentos.md`](experimentos.md) (inventario).
