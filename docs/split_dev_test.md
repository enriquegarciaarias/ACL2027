# Split congelado de desarrollo/evaluación y corrección de la fuga del few-shot

**Fecha:** 2026-11-09
**Módulo:** `sources/phase2/development_split.py`
**Versión del split:** 1.0

---

## 1. Problema detectado

Dos prácticas de la puesta a punto invalidaban los resultados como medida de
generalización:

1. **Fuga del few-shot.** Los ejemplos del prompt se calcularon con las medias
   por clase de **las 705 ventanas que luego se evaluaron**. Es decir, el
   prompt contenía estadísticos del conjunto de test.
2. **Iteración sobre los mismos datos.** Rúbrica fisiológica, patrón dominante,
   regla de consistencia, prompt binario y centrado intra-sujeto se probaron
   sobre S3 y el corpus completo antes de fijar el diseño.

Consecuencia: los resultados previos (accuracy 0,904 del prompt binario;
binario 0,930 del centrado; ECE 0,059; gate 100 % en `NC_shuffled`) deben
tratarse como **exploratorios** hasta re-ejecutarse con el split congelado y
los ejemplos corregidos.

## 2. Split congelado

| Conjunto | Sujetos | Uso |
|---|---|---|
| **Desarrollo** | `S2`, `S3`, `S4` | diseño de prompts, reglas, ejemplos few-shot y sus estadísticos |
| **Evaluación** | `S5`–`S11`, `S13`–`S17` | resultados finales |

`S12` no existe en el corpus WESAD. El split está en
`sources/phase2/development_split.py` (`DEV_SUBJECTS`, `TEST_SUBJECTS`,
`SPLIT_VERSION`).

Reglas de uso:

- Los estadísticos de los ejemplos few-shot (medianas por clase) se calculan
  **solo con S2–S4** (138 ventanas: 55 baseline, 30 stress, 17 amusement,
  36 meditation) y quedan fijados en `prompt_builder.FEW_SHOT_EXEMPLARS`.
- `scripts/run_e12_pilot.py` usa por defecto los sujetos de evaluación y
  **bloquea** los de desarrollo salvo `--allow-dev`.
- `main.py --proc GENERATE` (`PHASE2` como alias) etiqueta cada respuesta y
  entrada de manifiesto con
  `split = development | evaluation` y avisa si la lista incluye S2–S4
  (`--allowDev` silencia el aviso).
- Cualquier respuesta con `split = development` es exploratoria y se excluye de
  las tablas finales.

## 3. Ejemplos few-shot corregidos

Medianas por clase en S2–S4 (redondeadas), sustituyendo las medias del corpus:

| Clase | HR (bpm) | RMSSD (ms) | EDA (µS) | SCR (n) | Resp (rpm) |
|---|---:|---:|---:|---:|---:|
| baseline | 58,7 | 91,0 | 1,6 | 0 | 17,6 |
| stress | 79,5 | 40,9 | 4,3 | 4 | 13,7 |
| amusement | 61,8 | 89,3 | 1,4 | 0 | 17,7 |
| meditation | 62,2 | 77,4 | 2,0 | 0 | 12,8 |

En z intra-sujeto (misma fuente):

| Clase | HR | RMSSD | EDA | SCR | Resp |
|---|---:|---:|---:|---:|---:|
| baseline | −0,1 | 0,0 | −0,1 | 0,0 | +0,2 |
| stress | +2,1 | −1,0 | +1,3 | +2,0 | −0,3 |
| amusement | −0,2 | +0,5 | 0,0 | 0,0 | +0,5 |
| meditation | −0,3 | +0,5 | −0,1 | 0,0 | −0,8 |

Diferencias frente a los valores anteriores (medias del corpus evaluado): la HR
baseline pasa de 72,4 a 58,7 y la EDA baseline de 3,9 a 1,6. La heterogeneidad
entre sujetos es grande, así que el cambio es sustancial.

Los razonamientos de los ejemplos se reescribieron para **no contradecir los
valores**: meditation ya no afirma "frecuencia cardiaca baja" (en dev su HR es
ligeramente superior a baseline), sino HRV alta y respiración baja.

## 4. Controles automáticos

`tests/test_prompt_leakage.py`:

- `test_few_shot_examples_come_from_development_subjects`: verifica que la
  procedencia son S2–S4 y que no reaparecen las medias del corpus evaluado
  (HR 72,4; EDA 3,9).
- `test_few_shot_values_match_development_features`: recalcula las medianas
  desde `results/output/WESAD/INGEST/` y las compara con las publicadas en el
  prompt.

`tests/test_development_split.py`: intersección vacía, cobertura de los 15
sujetos, normalización y avisos.

## 5. Qué hay que re-ejecutar

| Experimento | Estado | Acción |
|---|---|---|
| Corpus WESAD `C0` + `NC_*` (4 clases, numérico y texto) | contaminado | re-ejecutar con el few-shot corregido |
| Corpus binario (P2) | contaminado | re-ejecutar |
| Corpus graduado (P4) | contaminado | re-ejecutar |
| Pilotos S3 (P1, P2, P4) | exploratorios (dev) | conservar como evidencia de desarrollo |
| Gate sobre `NC_shuffled` | mecanismo determinista, no depende del prompt | válido (los vetos no usan few-shot) |
| Análisis de calibración / conformal | depende del corpus | recalcular tras la re-ejecución |

## 6. Desviaciones declaradas (para el artículo)

1. La decisión de **endpoint binario** (2026-11-09) es **post hoc**: se tomó tras
   observar el colapso del modelo en 4 clases. Se registra como desviación, no
   como decisión preregistrada.
2. El **few-shot con estadísticos del test** se detectó y corrigió; los
   resultados previos se presentan como exploratorios.
3. El **split dev/test** se formalizó después de las iteraciones; las decisiones
   de diseño (rúbrica, consistencia, binario, centrado) se tomaron sobre S3 y el
   corpus y se aplicarán una sola vez sobre S5–S17.
4. `NC_zero` **no es válido** en la escala centrada (0 = media del sujeto): se
   excluye del análisis.
5. La normalización E12 es **transductiva** (usa la sesión del sujeto); debe
   declararse en el artículo.
