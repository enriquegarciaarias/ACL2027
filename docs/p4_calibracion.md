# P4 — Calibración y predicción selectiva del endpoint binario

**Fecha:** 2026-11-09
**Módulo:** `sources/phase3/calibration.py`
**Script:** `scripts/analyze_calibration.py`
**Reporte:** `results/output/WESAD/PHASE2/e12/calibration_report.json`

---

> **Aviso (2026-11-09):** resultados exploratorios: el corpus se generó con
> few-shot contaminado por estadísticos de evaluación. Recalcular tras la
> re-ejecución (ver `docs/split_dev_test.md`).

## 1. Problema

El prompt binario (P2) mejora la cobertura y la precisión, pero **satura la
confianza**: en el corpus completo, 703 de 705 respuestas declaran exactamente
`85` y las otras dos `75`. Una confianza con dos valores posibles:

- no permite ordenar la evidencia (el AUROC deja de medir separabilidad real y
  pasa a medir solo la dirección de la decisión);
- hace inútil el escalado de temperatura (una transformación monótona de un
  conjunto de dos puntos sigue teniendo dos puntos);
- hace inestable cualquier umbral de abstención (solo hay dos coberturas
  posibles: ~100 % o ~0 %).

P4 cuantifica el problema y prueba dos vías: **corrección post-hoc**
(temperatura, conformal) y **elicitación** (rúbrica de confianza graduada).

## 2. Herramientas

`sources/phase3/calibration.py` (stdlib + numpy):

| Función | Uso |
|---|---|
| `reliability_table`, `expected_calibration_error`, `maximum_calibration_error` | fiabilidad y ECE/MCE |
| `brier_score`, `log_loss` | puntuaciones propias binarias |
| `fit_temperature`, `apply_temperature` | escalado de temperatura (1 parámetro) |
| `risk_coverage_curve`, `evaluate_selective` | riesgo selectivo |
| `conformal_risk_threshold` | control conformal del riesgo (Angelopoulos et al., 2024) |
| `group_kfold_indices` | validación cruzada agrupada por sujeto (sin fuga) |
| `wilson_interval` | intervalo de confianza de una proporción |

`scripts/analyze_calibration.py` aplica todo lo anterior con **5 particiones
agrupadas por sujeto**: el umbral y la temperatura se ajustan en los sujetos de
entrenamiento y se evalúan en los sujetos no vistos.

## 3. Diagnóstico en el corpus (705 ventanas, `C0_clean`)

| Sistema | n | Valores de confianza | ECE (p_stress) | MCE | Brier | Log-loss | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prompt binario (P2) | 705 | **2** (0,85 ×703; 0,75 ×2) | 0,122 | 0,750 | 0,089 | 0,326 | 0,909 |
| Prompt 4 clases (mapeado) | 568 | 2 (0,85 ×381; 0,75 ×187) | 0,180 | 0,402 | 0,105 | 0,362 | 0,954 |

- El ECE sobre `p_stress` es alto en ambos casos, pero por motivos distintos:
  con el prompt binario, `p(non-stress)=0,85` se realiza como acierto el 97 %
  de las veces (**infraconfianza** en la clase mayoritaria) mientras que
  `p(stress)=0,85` acierta el 73 % (**sobreconfianza** en stress). Por eso el
  MCE es 0,75.
- Con la definición de confianza de la clase predicha (la que usa
  `evaluate_binary_records`), el ECE del prompt binario es **0,059**: el modelo
  acierta 0,904 cuando dice 0,85. Es decir, **globalmente está bien calibrado,
  pero no discrimina**: la confianza no distingue respuestas fáciles de difíciles.

## 4. Escalado de temperatura (validación cruzada agrupada)

| Sistema | T óptima (media) | Log-loss antes → después | Brier | ECE |
|---|---:|---|---|---|
| Prompt binario | 0,77 | 0,326 → **0,319** | 0,089 → 0,087 | 0,127 → 0,105 |
| Prompt 4 clases | 0,71 | 0,363 → 0,349 | 0,106 → 0,104 | 0,189 → 0,149 |

La mejora es marginal: con dos valores de confianza, la temperatura solo puede
acercar ambos puntos, no separarlos. **La corrección post-hoc no resuelve el
problema; hay que cambiar la elicitación.**

## 5. Predicción selectiva y control conformal del riesgo

Control conformal del riesgo (α = riesgo selectivo objetivo), umbral ajustado en
4/5 de los sujetos y evaluado en el sujeto restante:

| Sistema | α | Cobertura (test) | Precisión selectiva | Riesgo selectivo | Comportamiento por pliegue |
|---|---:|---:|---:|---:|---|
| Prompt binario | 0,05 | 0,00 | — | — | Ningún umbral cumple la cota en ningún pliegue |
| Prompt binario | 0,10 | 0,80 | 0,890 | 0,110 | 1 pliegue sin umbral + 4 con cobertura 100 % |
| Prompt binario | 0,20 | 1,00 | 0,905 | 0,095 | Acepta todo en los 5 pliegues |
| Prompt 4 clases | 0,05 | 0,669 | 0,959 | 0,041 | Umbral 0,76 en los 5 pliegues |
| Prompt 4 clases | 0,10 | 0,669 | 0,959 | 0,041 | Idéntico |
| Prompt 4 clases | 0,20 | 1,00 | 0,870 | 0,130 | Acepta todo |

Lectura: con el prompt binario **no existe un punto de operación selectivo
útil** (o se acepta todo o nada). Con el prompt de 4 clases, el único corte
disponible (0,75 vs 0,85) sí separa algo: cobertura 67 % con precisión selectiva
0,96, a costa de rechazar un tercio de las ventanas. Es el mejor resultado de
selectividad del proyecto, pero **no es aplicable al endpoint primario** porque
el prompt de 4 clases no decide `amusement`/`meditation`.

## 6. Intervención: rúbrica de confianza graduada

Se añadió `GRADED_CONFIDENCE_RUBRIC` al prompt (anclas 50-60 / 60-75 / 75-90 /
90-100 y prohibición explícita de un valor por defecto), con
`graded_confidence=True` en ambos constructores y el flag
`--graded-confidence` en `scripts/run_e12_pilot.py`.

Piloto S3 (`--binary --graded-confidence`, `C0_clean`, 46 llamadas):

| Variante (S3) | Valores de confianza | Accuracy | Balanced | Precision | Recall | AUROC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Binario base | 1 (85 ×46) | 0,891 | 0,931 | 0,667 | 1,000 | 0,931 | 0,041 | 0,099 |
| Binario + rúbrica graduada | 2 (85 ×39; 95 ×7) | 0,826 | 0,889 | 0,556 | 1,000 | **0,967** | 0,054 | 0,141 |

- La rúbrica **rompe parcialmente la saturación**: aparecen respuestas con
  confianza 95, todas en ventanas predichas como `stress`.
- En S3 ese grupo es **perfectamente preciso (7/7)**, y el AUROC sube de 0,931
  a 0,967. Es la primera señal selectiva útil del prompt binario.
- El coste en S3: más falsos positivos (5 → 8) y peor Brier. La muestra es
  demasiado pequeña (46 ventanas) para decidir, así que se lanza el corpus
  completo.

Corpus completo (`--binary --graded-confidence`, 705 llamadas):

| Sistema (705 ventanas) | Confianzas | Accuracy | Balanced | Precision | Recall | F1 | AUROC | ECE¹ | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Binario base | 85 ×703; 75 ×2 | **0,904** | **0,907** | **0,730** | 0,911 | **0,810** | 0,908 | 0,059 | **0,089** |
| Binario + graduada | 85 ×663; 95 ×42 | 0,873 | 0,893 | 0,650 | **0,929** | 0,765 | **0,912** | **0,017** | 0,110 |

¹ ECE con la confianza de la clase predicha (la definición de `evaluate_binary_records`).

- La rúbrica **sí crea resolución**: 42 ventanas con confianza 95 y **42/42
  correctas** (todas predichas como `stress`), frente al 86,6 % de acierto del
  grupo de confianza 85. El ECE baja de 0,059 a **0,017**.
- Coste: el modelo se vuelve más agresivo con `stress` (223 predicciones vs
  196), lo que sube los FP (53 → 78: baseline 24, meditation 40, amusement 14)
  y empeora precisión, F1 y Brier, a cambio de 2 TP más (145 vs 143) y 3 FN
  menos (11 vs 14).
- **Temperatura**: T=0,92 y sin mejora (log-loss 0,375 → 0,378). Sigue sin
  servir como corrección post-hoc.
- **Conformal** (CV por sujeto): con α=0,05 y α=0,10 el único umbral válido
  (≥0,90) acepta solo el grupo de confianza 95 → **cobertura 6,0 % con
  precisión selectiva 1,000** (riesgo 0; IC Wilson 95 % de la precisión:
  [0,916; 1,000] con 42 aciertos de 42). Con α=0,20 acepta todo (precisión
  0,873).

Este es el primer punto de operación selectivo válido del endpoint primario:
un 6 % de ventanas en las que el sistema puede afirmar `stress` con precisión
del 100 % en sujetos no vistos.

## 7. Interpretación y limitaciones

- La confianza del modelo binario base está **globalmente calibrada pero es
  degenerada**: dos valores en 705 ventanas. Ni la temperatura ni el conformal
  pueden arreglar una confianza sin resolución; el problema es de elicitación.
- La **rúbrica graduada funciona parcialmente**: aparecen 42 respuestas con
  confianza 95, todas correctas, y el ECE baja de 0,059 a 0,017. Pero también
  vuelve al modelo más propenso a predecir `stress`, con peor precisión y
  Brier. No es una mejora gratuita: es un **cambio de punto de operación**
  hacia mayor recall y menor precisión.
- El **conformal** con partición por sujeto solo es viable con la variante
  graduada y a cobertura muy baja (6 %). Es útil como modo de *triaje* ("estas
  ventanas son `stress` con garantía"), no como clasificador general.
- Recomendación operativa:
  - **Decisión general:** prompt binario base + compuerta de consistencia
    (accuracy 0,904, cobertura 99,4 %, ECE 0,059).
  - **Modo alta certeza:** prompt binario graduado, aceptar solo confianza ≥ 95
    (cobertura ~6 %, precisión 1,000 en sujetos no vistos).
  - **Descartado:** temperatura como corrección (sin efecto con 2 valores).
- Limitaciones:
  - `p_stress` se deriva de `confidence` y de la dirección de la decisión; no
    es una probabilidad elicitada de forma independiente.
  - La garantía conformal es marginal y con 42 muestras; el intervalo de Wilson
    de la precisión selectiva es [0,916; 1,000].
  - El grupo de confianza 95 está formado solo por predicciones `stress`; no
    hay señal equivalente para `non-stress`.
  - El corpus es un único dataset (WESAD, 15 sujetos); la validez externa de la
    calibración requiere K-EmoCon (P3).

## 8. Reproducción

```bash
# Análisis de calibración del corpus (sin LLM)
python scripts/analyze_calibration.py

# Piloto de la rúbrica graduada en S3
python scripts/run_e12_pilot.py --subjects S3 --conditions C0_clean \
  --scaling robust --binary --graded-confidence
```
