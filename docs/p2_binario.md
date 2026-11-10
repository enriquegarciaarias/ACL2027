# P2 — Prompt binario específico (stress vs non-stress)

**Fecha:** 2026-11-09
**Script:** `scripts/run_e12_pilot.py --binary`
**Datos:** representación E12 (z-scores intra-sujeto, 23 features de pecho)
**Modelo:** `qwen2.5:7b`, T=0,3, seed 42

---

> **Aviso (2026-11-09):** los resultados de este documento se obtuvieron con
> ejemplos few-shot calculados con estadísticos del conjunto de evaluación
> (fuga corregida después). Son **exploratorios**; deben re-ejecutarse con el
> split congelado y los ejemplos de desarrollo (ver `docs/split_dev_test.md`).

## 1. Motivación

Con el prompt de 4 clases el modelo **nunca predice** `amusement` ni
`meditation` (0/705 ventanas en el corpus): el espacio de decisión efectivo ya
es binario. La decisión metodológica (2026-11-09) es usar `stress` vs
`no-stress` como endpoint primario, justificada de forma pragmática.

P2 convierte esa realidad en un prompt explícito:

- El modelo debe elegir entre `stress`, `non-stress` y `abstain`.
- `non-stress` agrupa baseline, amusement y meditation del protocolo WESAD.
- Se mantiene la escala E12 (z-scores intra-sujeto) y el few-shot, con
  ejemplos binarios (stress / non-stress calmado / non-stress meditativo).
- El esquema de salida y el validador se parametrizan con
  `decision_mode="stress_binary"` (`sources/phase2/response_schema.py`).
- La compuerta de consistencia (P1) sigue disponible, pero esta ablation se
  ejecuta **sin** ella para medir el efecto puro del prompt.

## 2. Cambios de código

| Archivo | Cambio |
|---|---|
| `sources/phase2/response_schema.py` | `BINARY_DECISIONS`, `DECISION_MODES`, `DECISION_ALIASES`, `validate_llm_output(..., decision_mode=...)` |
| `sources/phase2/prompt_builder.py` | `decision_mode` en `build_numeric_prompt`/`build_text_prompt`, `BINARY_TASK_NOTE`, few-shot binario, esquema y reglas binarias |
| `sources/phase3/records.py` | `evaluate_binary_records`: accuracy, balanced, precisión/recall/F1, cobertura, abstención, AUROC, Brier, ECE y confusión 2×3 |
| `scripts/run_e12_pilot.py` | flag `--binary`, registro `decision_mode`, métricas binarias |

Normalizaciones aceptadas para `non-stress`: `non_stress`, `nonstress`,
`no-stress`, `no stress`, `not stress`.

## 3. Piloto en S3 (C0_clean, 46 ventanas)

Distribución real de S3: 18 baseline, 12 meditation, 10 stress, 6 amusement
(stress = 10 ventanas positivas).

| Sistema (S3, `C0_clean`) | Abstención | Accuracy | Balanced | Precision stress | Recall stress | AUROC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E12 4 clases, reinterpretado como binario | 21,7 % | 0,833 | 0,885 | 0,625 | 1,000 | 0,977 | 0,194 | 0,138 |
| **E12 + prompt binario** | **0,0 %** | **0,891** | **0,931** | **0,667** | **1,000** | 0,931 | **0,041** | **0,099** |

Matriz de confusión del prompt binario:

| Real \ Predicho | stress | non-stress | abstain |
|---|---:|---:|---:|
| stress (10) | **10** | 0 | 0 |
| non-stress (36) | 5 | **31** | 0 |

- **Recall de stress 100 %** con **cobertura 100 %**: la abstención desaparece
  (21,7 % → 0 %) y el modelo decide todas las ventanas.
- **Calibración muy superior** (ECE 0,194 → **0,041**; Brier 0,138 → **0,099**):
  con la tarea binaria la confianza declarada ya coincide con el acierto.
- **Precisión 0,667**: los 5 falsos positivos se reparten 2 baseline,
  2 amusement y 1 meditation. El modelo sobre-marca activación en ventanas
  no estresantes, algo coherente con que `amusement` sea fisiológicamente
  próximo a baseline en EDA+HR.
- **AUROC 0,931** (frente a 0,977 del 4 clases): la confianza es constante
  (85 en las 46 respuestas), así que el ranking se reduce a la dirección de la
  decisión. El AUROC alto del 4 clases aprovechaba confianzas más variadas.
- **Sensibilidad a la corrupción: nula.** El acuerdo entre la decisión limpia y
  la de `NC_shuffled` es **45/46 = 0,978** (4 clases: 0,761) y la "accuracy"
  sobre datos barajados es 0,913. El prompt binario hace al modelo todavía más
  insensible a que los valores estén permutados: predice prácticamente lo mismo.
  Esto **refuerza la necesidad de la compuerta determinista de P1** (que anula
  el 100 % de esas respuestas) y es un resultado de RQ2/RQ3.

> Nota: la fila "absoluto" de este piloto no es comparable (el prompt de 4
> clases se reinterpretó con un mapeo que aún no estaba aplicado en la versión
> ejecutada). El corpus completo se regenera con el mapeo corregido.

## 4. Corpus completo (15 sujetos, 705 ventanas, `C0_clean`)

Distribución real: 283 baseline, 179 meditation, 157 stress, 86 amusement
(157 ventanas positivas).

| Sistema | Cobertura | Accuracy | Balanced | Precision stress | Recall stress | AUROC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E12 4 clases (reinterpretado como binario) | 79,7 % | 0,870 | 0,905 | 0,673 | 0,980 | 0,954 | 0,096 | 0,106 |
| **E12 + prompt binario** | **99,4 %** | **0,904** | **0,907** | **0,730** | 0,911 | 0,908 | **0,059** | **0,089** |

- **Abstención 0 %** (4 inválidos, 0,6 %): el prompt binario decide (casi)
  todas las ventanas frente al 19,3 % de abstención del prompt de 4 clases.
- Confusión: **143 TP, 53 FP, 491 TN, 14 FN**.
- Falsos positivos por clase real: baseline 19/283 (6,7 %), amusement 11/86
  (12,8 %), meditation 23/179 (12,8 %).
- **Subconjunto stress vs baseline** (comparable con los resultados previos):
  accuracy 0,925, balanced 0,922, precision 0,883, recall 0,911, AUROC 0,925,
  frente a 0,930 / 0,937 / 0,867 / 0,980 del prompt de 4 clases con cobertura
  80,7 %.
- **Compuerta de consistencia (P1): 0 vetos en `C0_clean`**, así que las
  métricas del sistema compuesto son idénticas. La compuerta sigue anulando el
  100 % de `NC_shuffled` (ver `docs/consistencia.md`).

Lectura: ambos prompts **encuentran el mismo número de ventanas de stress**
(144 vs 143 TP). La diferencia es qué hacen con el resto: el de 4 clases se
abstiene en 136 ventanas (79,7 % de cobertura); el binario las resuelve como
`non-stress` con 53 FP (frente a 70 FP del 4 clases sobre su subconjunto
decidido). El coste es un AUROC algo menor (la confianza es constante 85) y
11 ventanas de stress que pasan a `non-stress`.

## 5. Interpretación

- El prompt binario **elimina el problema de abstención y de calibración** del
  caso de 4 clases: al preguntar exactamente lo que el modelo ya sabía hacer
  (separar stress del resto), responde a (casi) todas las ventanas y su
  confianza se ajusta al acierto real.
- A escala de corpus la ganancia es **cobertura y precisión**, no detección:
  ambos prompts encuentran las mismas ventanas de stress (144 vs 143 TP). Lo
  que cambia es el tratamiento del resto (abstención vs `non-stress`) y el
  número de falsos positivos (70 → 53).
- El coste es un **recall algo menor** (0,980 sobre decididas → 0,911 con
  cobertura 99,4 %) y un AUROC más bajo, porque la confianza es constante
  (703 de 705 respuestas con confianza 85; la confianza no gradúa la evidencia,
  solo la dirección).
- El modelo **no usa la estructura fina de las features**: barajar los valores
  cambia su decisión en menos del 3 % de los casos (S3). Acierta por el patrón
  global de activación, no por una lectura fisiológica detallada. La compuerta
  de P1 cubre el caso de evidencia incoherente.
- Recomendación: **endpoint primario = prompt binario + compuerta de
  consistencia + E12**; el 4 clases queda como análisis secundario.
- Pendiente derivado: la confianza constante limita AUROC y ECE. Una escala de
  confianza más expresiva (o preguntar por subcomponentes) es la vía natural
  para P4 (calibración).

## 6. Reproducción

```bash
# Piloto S3 (92 llamadas: C0_clean + NC_shuffled)
python scripts/run_e12_pilot.py --subjects S3 \
  --conditions C0_clean,NC_shuffled --scaling robust --binary

# Corpus completo (705 llamadas por condición)
python scripts/run_e12_pilot.py \
  --subjects S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17 \
  --scaling robust --binary
```

Salidas:

- `results/output/WESAD/PHASE2/e12/{subject}.numeric_centered_binary.responses.jsonl`
- `results/output/WESAD/PHASE2/e12/all.numeric_centered_binary.responses.jsonl`
- `results/output/WESAD/PHASE2/e12/e12_summary_binary.json`
