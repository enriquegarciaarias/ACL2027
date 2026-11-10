# Barrido de degradación con LLM (H1/H3) — diseño

**Fecha:** 2026-11-09
**Script:** `scripts/run_degradation_sweep.py`
**Salidas:** `results/output/WESAD/PHASE2/degradation/{subject}.responses.jsonl`
y `degradation_summary.json`

---

## 1. Qué mide

El núcleo de la tesis: cómo cambia la respuesta del LLM (corrección, confianza,
certeza lingüística, **respaldo de afirmaciones** y abstención) al degradar la
evidencia de entrada. Es la variable experimental de RQ1–RQ3.

## 2. Diseño congelado

| Elemento | Decisión |
|---|---|
| Sujetos | **evaluación** S5–S17 (salvo S12); desarrollo bloqueado |
| Ventanas | 567 (todas las del conjunto de evaluación) |
| Condiciones | `C0_clean`, `C1_partial_resp`, `C2_partial_eda`, `C4_total_eda`, `C5_total_both` |
| Representación | numérica estandarizada por sujeto (E12), 23 features de pecho |
| Referencia de estandarización | **mediana/IQR de las ventanas limpias C0 del sujeto**, aplicada a todas las condiciones |
| Prompt | clasificación afectiva de 4 clases (tarea preregistrada) |
| Few-shot | medianas por clase de S2–S4 (sin fuga del test) |
| Modelo | `qwen2.5:7b`, T=0,3, seed 42 |

La referencia de estandarización se calcula **solo con C0**. Si se recalculara
por condición, cada degradación se re-normalizaría y el efecto experimental se
perdería (las z-scores volverían a estar centradas).

## 3. Hallazgo de diseño: C1 ≡ C3

`C3_total_resp` (severidad 1,0) produce **exactamente las mismas features** que
`C1_partial_resp` (severidad 0,5): con ≥50 % de datos ausentes, el extractor
devuelve `null` para toda la modalidad respiratoria. Comprobado sobre los
ficheros degradados (0 valores distintos). Por eso el barrido ejecuta 5
condiciones efectivamente distintas y omite C3; el diseño preregistrado de 6 se
mantiene documentado, con la equivalencia declarada.

Consecuencia para el análisis: el eje de **severidad** no está materializado
para respiración. Sí lo está para EDA (`C2` conserva la media y pierde SCR;
`C4` deja EDA a cero). Las comparaciones de severidad deben hacerse solo en la
rama EDA (`C2` vs `C4`).

## 4. Qué se guarda por respuesta

`subject`, `split`, `condition_id`, `window_id` (real y opaco), `duration_s`,
`status`, `decision`, `confidence`, `claims` (métrica/canal/dirección/magnitud),
`reasoning`, `raw_output`, `prompt` completo, metadatos de Ollama y
`generated_at`. **No se guardan etiquetas** (están en
`INGEST/groundtruth/labels.jsonl`).

## 5. Coste y reanudación

- 567 ventanas × 5 condiciones = **2.835 llamadas** (~13 h a ~17 s/llamada).
- El script reanuda por `(window_id, condition_id)`: se puede interrumpir y
  relanzar sin perder trabajo.
- Log: `/tmp/opencode/degradation_sweep.log`.

## 6. Brazo textual (H3)

Para H3 (efecto de la representación) se ejecuta el mismo barrido con la
representación **textual**: los mismos 23 valores z, renderizados como resumen
con secciones y etiquetas legibles (`render_text_summary`), con los mismos
few-shot estandarizados de S2–S4 y la misma tarea de 4 clases.

- Salida: `results/output/WESAD/PHASE2/degradation_text/`.
- Comando: `python scripts/run_degradation_sweep.py --representation text`.
- El verificador de claims entiende ambas representaciones: añade
  `METRIC_ALIASES` para etiquetas legibles ("Heart rate", "SCR count",
  "Mean RR interval"…) y `parse_prompt_features` acepta `nombre: valor` sin
  guion.
- Contraste: `python scripts/analyze_degradation.py --compare-with
  results/output/WESAD/PHASE2/degradation_text` imprime, por condición,
  accuracy/abstención/respaldo/certeza de ambos brazos, sus deltas y el efecto
  de la degradación en cada representación.

## 7. Análisis previsto (siguiente paso)

1. **Corrección por condición** (H1/RQ1): accuracy, balanced, macro-F1 y
   *coste en exceso* frente a la referencia features-only por condición.
2. **Confianza y certeza** (H2): AUROC de la confianza como predictor de
   corrección, sobreconfianza, ECE/Brier; índice hedges−boosters por 100
   palabras sobre `reasoning`.
3. **Respaldo de afirmaciones** (H5/RQ3): verificación de cada claim contra las
   features de esa ventana y condición (respaldada / contradicha / no respaldada
   / indeterminada).
4. **Abstención** (H4): tasa por condición frente a la resolubilidad medida con
   el modelo de referencia.
5. **Indicador externo** (H6): AUROC incremental de la compuerta de
   consistencia (P1) sobre la confianza declarada.
