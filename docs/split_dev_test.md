# Split congelado de desarrollo/evaluación y corrección de la fuga del few-shot

**Fecha:** 2026-11-09 · **Actualizado:** 2026-11-19 (auditoría §6.1:
desviaciones 7–10 declaradas)
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
6. La **jerarquía binaria y la exclusión de temperatura** (2026-11-16) son una
   **segunda desviación post hoc**. Se fijó tras ver los resultados por clase,
   pero **antes** de re-ejecutar: primario `schmidt` (`baseline` + `amusement`,
   424 ventanas) **sin temperatura**; sensibilidades `baseline` (355),
   `no_amusement` (498) y `all` (567). La exclusión de `meditation` se justifica
   por constructo (Schmidt no la usa como clase afectiva), no por resultado; la
   de temperatura, por ser confusor de orden y la única modalidad no degradada.
   Se reportan las cuatro variantes y la comparación con Schmidt es de
   **composición de clases**, no de rendimiento. Ver
   [`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md).
7. **La normalización intra-sujeto (E12) está en producción** (auditoría
   2026-11-19). La representación del experimento primario no es la escala
   absoluta: `sources/phase3/degradation.py` calcula `subject_reference()` con
   las ventanas `C0_clean` de cada sujeto y envía `z_scores()` al modelo. Esa
   normalización se eligió en **E6 (E12)**, un análisis marcado exploratorio que
   se ejecutó sobre las **705 ventanas de los 15 sujetos** (E12 conserva archivos
   de S2, S3 y S4 en `results/output/WESAD/PHASE2/e12/`). Consecuencia: E1 mide
   bien sobre sujetos de evaluación, pero su representación **no** es una
   decisión independiente de esos datos. No se re-ejecutó. Alternativa pendiente:
   fijar la referencia con S2–S4 (coste: 2.835 × 2 llamadas, porque cambia el
   prompt).
8. **El conjunto de 23 features de la vista LLM se fijó mirando salidas del
   corpus evaluado** (2026-11-08/09). La exclusión de `eda_scr_candidates`
   («diagnóstico interno que se confundía con el número real de SCR») y la
   inclusión de `resp` en los ejemplos salen de inspeccionar respuestas y errores
   de evaluación. Matiz importante: la exclusión de muñeca/BVP **sí** tiene
   justificación en desarrollo (el confusor de movimiento se analizó en S2,
   AUROC hasta 0,928), así que no todas las piezas del conjunto están
   contaminadas; conviene citar por separado las dos justificaciones.
9. **Los umbrales y la laxitud de la compuerta de consistencia** (`max_hard=0`,
   `max_soft=1`, `tolerance_scale=1.0`, activada por defecto en `config.json`)
   se fijaron con un barrido sobre las **705 ventanas** de los 15 sujetos
   ([`consistencia.md`](consistencia.md) §4.1). El mecanismo es determinista y
   no usa etiquetas ni prompt —por eso E3 se sostiene como mecanismo—, pero el
   **punto de operación** sí se eligió con datos de evaluación: con
   `tolerance_scale` 1→3 la detección de `NC_shuffled` no cambia y la
   sensibilidad al ruido baja de 96,2 % a 52,9 %. En limpio produce 0 vetos, así
   que las métricas de `C0` no dependen de esa elección; las de `NC_shuffled` sí.
10. **Decisiones menores con datos de evaluación.** (a) **R=3** repeticiones
    (E11) se fijó con 30 ventanas de **S5–S7**, que son de evaluación. (b) Los
    *smokes* operativos del barrido (numérico, textual y repeticiones) se
    hicieron sobre **S5**, con 1 ventana cada uno: no ajustan nada, pero gastan
    muestra de evaluación. Se listan por exhaustividad, no porque cambien
    conclusiones.

### 6.1. Auditoría 2026-11-19: qué se comprobó y qué no se arregló

Se verificó en disco que los corpus **de producción no contienen sujetos de
desarrollo** (`degradation/` y `degradation_text/`: S5–S11 y S13–S17; `e12/`:
los 15, incluidos S2–S4). Es decir, el split se respetó **en los datos** y no se
respetó del todo **en el diseño**: de las decisiones que siguen en producción,
3 estaban declaradas (1, 5–6 de esta lista) y 3 no lo estaban hasta hoy (7–9),
más la fuga del few-shot ya corregida y las decisiones menores (10).

No se re-ejecutó ningún brazo: la sesión fue de auditoría y documentación. La
corrección de mayor impacto sería la 7 (representación), y es también la más
cara, porque cambia el prompt de las 5.670 respuestas del diseño primario.
