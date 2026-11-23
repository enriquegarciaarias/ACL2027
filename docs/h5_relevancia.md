# H5 · Relevancia de las afirmaciones: preregistro de la firma por clase

**Fecha de fijación:** 2026-11-21 (antes de medir coherencia sobre ningún corpus).
**Estado:** especificación **cerrada**; **implementada** el 2026-11-21
(`sources/phase3/signatures.py`, `claim_coherence` en `claims.py`; ver §10).
Resultados del diseño primario: pendientes de ejecución.
**Complementa a:** [`claims_h5.md`](claims_h5.md) §5, donde se registró el hueco.

---

## 1. Qué se fija aquí y por qué ahora

`claims_h5.md` §5 dejó documentado que el verificador comprueba **verdad**
(el claim coincide con el valor mostrado), pero no **relevancia** (¿la evidencia
citada favorece la clase que el modelo acaba prediciendo?). Con el eje actual,
un claim verdadero *y contrario a la propia conclusión* cuenta como `supported`.

Este documento fija, **antes de medir**, cuatro cosas que no se pueden elegir
después sin convertirlo en análisis post hoc:

1. la **tabla de firma por clase** (dirección esperada de cada feature bajo
   `stress`), que es la única fuente de la noción de «a favor» / «en contra»;
2. qué features **entran** en el cálculo y cuáles quedan fuera, y por qué;
3. las **métricas** exactas y su agregación;
4. las **predicciones** y los **falsadores**, para que el resultado pueda
   contradecir la hipótesis en lugar de confirmarla por construcción.

### 1.1. Declaración de información previa

La tabla se fija por **mecanismo fisiológico y literatura**, no por el
rendimiento de las features en estos datos. Se declara, por transparencia, que
la dirección de tres features ya era conocida de un análisis anterior
(2026-11-07, AUROC `stress`-vs-`baseline` > 0,5 con `stress` como clase
positiva, lo que ya implica valores mayores en estrés):

| Feature | Dirección ya conocida | Cómo se trata |
|---|---|---|
| `ecg_hr_bpm` | mayor en estrés | se declara; coincide con la literatura |
| `eda_phasic_std_us` | mayor en estrés | se declara; coincide con la literatura |
| `eda_scr_rate_per_min` | mayor en estrés | se declara; coincide con la literatura (*) |

(*) La dirección de SCR ya se había discutido en la sesión del 2026-11-07 con
motivo de la recalibración del umbral (0,01 → 0,05 µS), que produjo el conteo
**invertido** (baseline 52 > stress 36). Es decir, también se sabía que el
sentido correcto era *subida* en estrés.

Las otras **9** features del conjunto primario (§3) no tenían dirección
declarada en ningún documento; su signo se fija aquí por primera vez. Las **11**
restantes quedan fuera del conjunto primario (§3.3 y §3.4), así que su dirección
no interviene en el resultado principal.

Para no depender de esa memoria, la §6 incluye la predicción **P0**, que
comprueba sobre los datos si la tabla se sostiene. **P0 no puede modificar la
tabla**: si falla, se reporta como predicción fallida.

---

## 2. Cuatro ejes que no son el mismo

| Eje | Pregunta | Estado |
|---|---|---|
| Verdad | ¿el claim coincide con el valor mostrado? | implementado (`supported` / `contradicted`) |
| **Relevancia** | ¿la dirección del claim favorece la clase predicha? | **este documento** |
| Suficiencia | ¿bastan esos claims para sostener la decisión? | fuera de alcance (contraprueba costosa) |
| Selección | ¿cita lo que apoya y omite lo que contradice? | parcialmente cubierto (percentil de \|z\|) |

La coherencia es **independiente de la verdad**: se evalúa la dirección
declarada contra la dirección esperada bajo la clase predicha, sin mirar el
valor. Por eso un claim puede ser:

- `supported` + coherente → el caso ideal;
- `supported` + contrario → **el hallazgo de H5** (narra bien, concluye mal);
- `contradicted` + coherente → se equivocó al leer pero concluyó por la vía
  esperada (menos grave, y distinguible);
- `contradicted` + contrario → doble fallo.

El instrumento principal es por tanto la **tabla conjunta 3 × 3**
(verdad × coherencia), no un único número.

---

## 3. Tabla de firma

### 3.1. Notación

`s(m) ∈ {+1, −1, 0}`: dirección esperada de la métrica `m` bajo `stress`,
relativa a la referencia intra-sujeto de la propia ventana.

`d(c) ∈ {+1, −1, 0}`: dirección declarada en el claim (`increase` = +1,
`decrease` = −1, `stable` = 0).

**Alineamiento:** `a = s(m) · d(c)`.

### 3.2. Features con firma

Dirección esperada **bajo `stress`** frente a la referencia del sujeto. En la
vista primaria (`numeric_llm` / `text_llm` con z intra-sujeto) la referencia son
las ventanas limpias `C0` del sujeto, así que «↑» significa «por encima de su
propia línea base».

| # | Feature | Modalidad | Firma | Nivel | ¿Primaria? | Justificación |
|---:|---|---|:---:|---|:---:|---|
| 1 | `eda_mean_us` | EDA | ↑ | fuerte | **sí** | nivel tónico de conductancia sube con activación simpática (Boucsein, 2012; Dawson et al., 2007) |
| 2 | `eda_tonic_mean_us` | EDA | ↑ | fuerte | **sí** | mismo mecanismo que (1); es su componente tónico |
| 3 | `eda_phasic_std_us` | EDA | ↑ | fuerte | **sí** | más actividad fásica / respuestas no específicas |
| 4 | `eda_scr_count` | EDA | ↑ | fuerte | **sí** | conteo en ventana **fija** de 60 s: no confundido con la duración |
| 5 | `eda_scr_rate_per_min` | EDA | ↑ | fuerte | **sí** | normalizado por duración; redundante con (4) |
| 6 | `eda_std_us` | EDA | ↑ | moderado | **sí** | variabilidad de la señal; sube con la actividad fásica |
| 7 | `eda_scr_amplitude_mean_us` | EDA | ↑ | moderado | **sí** | amplitud de respuesta tiende a crecer con la activación |
| 8 | `eda_scr_amplitude_max_us` | EDA | ↑ | moderado | **sí** | igual que (7), más sensible a un único evento |
| 9 | `ecg_hr_bpm` | ECG | ↑ | fuerte | **sí** | taquicardia de estrés agudo |
| 10 | `ecg_sdnn_ms` | ECG | ↓ | fuerte | **sí** | retirada vagal; SDNN global baja (Task Force, 1996; Kim et al., 2018) |
| 11 | `ecg_rmssd_ms` | ECG | ↓ | fuerte | **sí** | componente vagal; RMSSD baja |
| 12 | `resp_rate_bpm` | Resp | ↑ | moderado | **sí** | aumento de frecuencia respiratoria bajo demanda/estrés (Grossman, 1983) |

**Conjunto primario = 12 features.** Se reporta además una versión reducida de
**5** (una por mecanismo: `eda_mean_us`, `eda_phasic_std_us`, `ecg_hr_bpm`,
`ecg_rmssd_ms`, `resp_rate_bpm`) porque EDA aporta 8 de 12 y el conteo por claim
sobrepondera esa modalidad (§8).

### 3.3. Features sin firma (10)

No se les asigna dirección; sus claims quedan como `no_signature`, **fuera del
numerador y del denominador** y contados aparte.

| Feature | Motivo |
|---|---|
| `eda_slope_us_per_s` | deriva/adaptación dentro de 60 s; el signo no es identificable sin modelo de recuperación |
| `ecg_rr_mean_ms` | inversa aritmética de (9); incluirla **duplicaría** el peso de HR (se reporta aparte) |
| `ecg_n_beats` | conteo = HR × duración y nº de detecciones; no es un estado fisiológico |
| `resp_amplitude_pct` | profundidad: sube con el habla, baja con respiración superficial; ambiguo |
| `resp_std_pct` | variabilidad de amplitud; dirección no fijada por la literatura disponible |
| `resp_n_cycles` | conteo = frecuencia × duración; redundante con (12) |
| `temp_mean_degc` | ↓ por vasoconstricción periférica, pero **confundida con el orden del protocolo** (r = 0,801) y **fuera del conjunto primario** |
| `temp_std_degc` | idem; sin mecanismo direccional claro |
| `temp_slope_degc_per_s` | idem |
| `temp_range_degc` | idem |

### 3.4. Conjunto de sensibilidad (1 más)

| Feature | Firma | Nivel | Por qué no en la primaria |
|---|:---:|---|---|
| `resp_interval_cv` | ↑ | débil | irregularidad respiratoria; evidencia menos firme y podría reflejar el habla |

Se calcula todo con y sin ella. Si el resultado cambia de dirección, se declara
que depende de una feature de nivel débil.

### 3.5. Advertencia sobre `temp_*`

Aunque en la sensibilidad de temperatura (`--temperature include`) la modalidad
entra en la evidencia, **no** entra en la coherencia: la firma no está fijada y
el confusor de orden impediría interpretarla. Se mantiene la misma tabla en los
dos brazos para que sean comparables.

---

## 4. Regla de coherencia

Para cada claim con firma, en una respuesta con decisión `D`:

| `D` | coherente | contrario | neutro |
|---|---|---|---|
| `stress` | `a > 0` | `a < 0` | `a = 0` |
| `non-stress` | `a < 0` | `a > 0` | `a = 0` |

- `stable` produce siempre `a = 0` → **neutro**, no contrario. No se penaliza
  hedgear la dirección: se mide aparte.
- `D = abstain` no se evalúa; las abstenciones se cuentan aparte.
- Métrica sin firma o no listada → `no_signature` (fuera de los denominadores).
- No se usa la magnitud en la coherencia (solo la dirección).

### 4.1. La asimetría de `non-stress`, declarada

La regla espejo supone que en `non-stress` **está ausente la excursión típica de
estrés**, no que exista una excursión opuesta de igual tamaño. Es defendible
para `baseline`, pero **más débil para `amusement`**, que puede elevar la
activación. Consecuencias, fijadas ahora:

1. la coherencia de las decisiones `non-stress` se reporta **por separado** y no
   se agrega con la de `stress` sin declararlo;
2. se desglosa además por **clase real** (`baseline` vs `amusement`), como
   descriptivo;
3. si el desglose muestra que la diferencia entre clases reales explica el
   resultado, se reporta ese hecho y no se atribuye al modelo.

### 4.2. Mínimo para publicar una tasa

Si el denominador (coherentes + contrarios) de una celda es **< 30 claims**, la
celda se marca `insufficient` y no se publica como tasa. Motivo: el corpus tiene
celdas con pocas decisiones `stress` (en el primario `schmidt` solo **127** de
las 424 ventanas son de estrés).

---

## 5. Métricas

### 5.1. Por claim

`coherence ∈ {coherent, contrary, neutral, no_signature}` + `a`.

### 5.2. Por respuesta (solo si decidió)

| Campo | Definición |
|---|---|
| `n_confirming` | claims coherentes con `D` |
| `n_against` | claims contrarios a `D` |
| `n_against_supported` | contrarios **y** `supported` (el caso de §5.4) |
| `has_confirmation` | `n_confirming ≥ 1` |
| `evidence_conflict` | `n_against > n_confirming` |

### 5.3. Por (condición × modelo × representación)

| Métrica | Fórmula |
|---|---|
| `coherence_rate` | Σcoherentes / (Σcoherentes + Σcontrarios) |
| `against_supported_share` | Σ(contrarios ∧ `supported`) / Σ(`supported`) |
| `zero_confirmation_share` | decisiones `stress` con `n_confirming = 0` / decisiones `stress` |
| tabla conjunta | 3 verdades × 3 coherencias, con recuentos |
| IC 95 % | bootstrap **agrupado por sujeto**, 2.000 remuestreos |

Se publica **junto a** la tasa de respaldo ya existente, en la misma tabla, para
que las dos dimensiones se lean a la vez.

### 5.4. Regla de lectura

- `coherence_rate` mide **consistencia interna del relato**: un modelo puede ser
  coherente y estar equivocado. No es una medida de corrección.
- `against_supported_share` es la forma operativa de «cita evidencia verdadera
  contra su propia conclusión».
- Ninguna de las dos sustituye a accuracy/balanced/cobertura.

---

## 6. Predicciones y falsadores

### 6.1. Predicciones

| # | Predicción | Umbral |
|---|---|---|
| **P0** | La mediana de `z` por clase real en `C0` sigue la tabla §3.2 | ≥ **10 de 12** features con el signo esperado al comparar `stress` frente a `non-stress` (↑ en las diez de firma positiva; ↓ en `ecg_sdnn_ms` y `ecg_rmssd_ms`) |
| **P1** | Con señal intacta, el modelo cita sobre todo evidencia consistente | `coherence_rate(stress)` en `C0` ≥ **0,80** |
| **P2** | Al destruir EDA, cita evidencia verdadera contra su conclusión | `against_supported_share(stress)` en `C4` ≥ **0,40** |
| **P3** | El patrón es de la tarea, no de un modelo concreto | P1 y P2 se cumplen en **los dos modelos y las dos representaciones** (misma dirección) |
| **P4** | La coherencia cae **menos** que la corrección | caída C0→C5 de `coherence_rate` < caída de `balanced accuracy` |

**P0 se calcula sin LLM** (medianas de `z` sobre el corpus limpio) y **no puede
cambiar la tabla**: sirve para saber si la firma describe estos datos.

**P4** es la forma medible de «claims localmente ciertos, inferencia global
falsa»: si el relato se mantiene mientras el acierto se hunde, el fallo es de
inferencia; si caen juntos, hay además fallo de lectura.

#### Origen de los umbrales (declarado)

| Umbral | Origen | Por qué no es circular |
|---|---|---|
| P0 ≥ 10/12 | convención: tolera 2 discrepancias por ruido de medición en ventanas de 60 s | no se fija mirando los datos |
| P1 ≥ 0,80 | «mayoría amplia de evidencia consistente»; no sale de ningún dato | umbral de juicio, fijado a priori |
| P2 ≥ 0,40 | **derivado** del 0,681 exploratorio de `claims_h5.md` §5.4 (corpus de 4 clases), **rebajado** a la mitad | se declara como derivado: si se hubiera fijado en 0,68 sería un objetivo calcado de la observación previa. Al 0,40 el test sigue siendo informativo y no se aprueba por construcción |
| P4 (comparación de caídas) | no tiene umbral: es una desigualdad entre dos cantidades medidas en el mismo corpus | — |

### 6.2. Falsadores

| # | Si ocurre | Qué se concluye |
|---|---|---|
| **F1** | `supported` y coherencia caen **juntas** | la lectura «lee bien y concluye mal» **no** se sostiene; hay fallo de lectura y se declara |
| **F2** | `coherence_rate(stress)` en `C0` < 0,80 | el modelo cita evidencia contraria **incluso con señal intacta**: H5 pasa a describir un fallo de selección permanente, no un efecto de la degradación |
| **F3** | P3 se cumple en un modelo y no en el otro | el eje es **dependiente del modelo**; se declara como H2a y H6, no como propiedad general |
| **F4** | El modelo que nunca se abstiene (7b) da coherencia distinta del que se abstiene (14b) | no es un fallo: la coherencia se define sobre decisiones, y la abstención se reporta. Se interpreta junto a la cobertura |
| **F5** | El resultado desaparece al quitar `resp_interval_cv` o al usar el conjunto de 5 | el efecto depende de features débiles o de la ponderación por modalidad; se declara |

---

## 7. Corpus, agregación y reglas de ejecución

- **Sin llamadas al LLM.** Todo sale de artefactos en disco.
- **Brazos primarios (4):**
  `degradation_binary_schmidt_notemp/{qwen2_5_7b,qwen3_14b}` y
  `degradation_text_binary_schmidt_notemp/{qwen2_5_7b,qwen3_14b}`
  (2.835 registros cada uno; 424 ventanas tras descartar `meditation`).
- **No** se usan `degradation/` ni `degradation_binary/` (corpus histórico de
  4 clases y variante `all` con temperatura): sus cifras ya circulan y están
  marcadas como exploratorias.
- **Unidad de agregación:** respuesta. **Unidad de inferencia:** sujeto
  (bootstrap agrupado, 2.000 remuestreos), porque las ventanas de un sujeto no
  son independientes.
- **Recuperación tolerante:** se aplican los mismos criterios que
  `analyze_degradation.py` (decisión y claims bien formados; `recovered` se
  marca y se cuenta). Las respuestas `invalid_output` no entran en la tasa de
  coherencia y sí en el denominador de inválidas.
- **Cruce entre modelos:** prohibido comparar coherencias sin reportar cobertura
  y abstención, porque la coherencia se define solo sobre decisiones.

---

## 8. Limitaciones declaradas

1. **Consistencia ≠ corrección.** Un relato coherente puede sostener una
   conclusión falsa; el eje mide lo primero.
2. **Firma poblacional.** No hay firma por sujeto; un sujeto con respuesta
   atípica cuenta como incoherente.
3. **`amusement` en `non-stress`.** Eleva la activación: la coherencia de
   `non-stress` puede castigar claims fisiológicamente correctos (§4.1).
4. **Sobreponderación de EDA** (8 de 12 features). Mitigada con el conjunto de 5
   (§3.2) y las redundancias declaradas (`mean`/`tonic`, `count`/`rate`).
5. **El conjunto de 23 features tiene decisiones contaminadas del split**
   (desviación 8 de [`split_dev_test.md`](split_dev_test.md)). La firma **no**
   depende de esos umbrales, solo de la dirección fisiológica, así que la
   coherencia es menos sensible a esa contaminación que un umbral aprendido;
   aun así se declara.
6. **No mide suficiencia.** Un claim puede ser coherente y aun así no bastar
   para la decisión; eso requiere la contraprueba (retirar del prompt las
   features citadas), que queda fuera de alcance por coste (1 llamada por
   respuesta).
7. **El esquema no tiene categoría de «claim de ausencia»**, y no se añade
   (sería post hoc y solo para un brazo). Los claims con `not available` siguen
   cayendo en `indeterminate`.

---

## 9. Lo que NO se hará

- No se ajusta la tabla después de ver los resultados. Si P0 o P1 fallan, se
  publican como **predicciones fallidas**.
- No se añade `"not available"` como dirección válida en el validador.
- No se re-ejecuta el corpus ni se generan respuestas nuevas.
- No se comparan coherencias entre variantes de `non-stress` sin declararlo.

---

## 10. Implementación

Estado a **2026-11-21**: implementado y con tests. La tabla no cambió.

| Pieza | Dónde |
|---|---|
| Tabla de firma, niveles y conjuntos (12 / 5 / sensibilidad) | `sources/phase3/signatures.py` (`FeatureSignature`, `STRESS_SIGNATURE`, `EXCLUDED_FEATURES`, `PRIMARY_FEATURES`, `COMPACT_FEATURES`, `SENSITIVITY_FEATURES`, `sign_of`) |
| Coherencia por claim, por respuesta y agregados | `sources/phase3/claims.py` (`claim_coherence`, `response_coherence`, `summarize_coherence`, `MIN_CLAIMS_FOR_RATE`) |
| Bloque «H5 · coherencia» + IC por sujeto | `scripts/analyze_degradation.py` (`--coherence-set`) |
| Comprobación P0 (medianas de z por clase en `C0`) | `scripts/analyze_degradation.py` (`signature_check`, campo `signature_check` del informe) |
| CLI por fases | `--coherence-set` en `EVALUATE --experiment degradation` (`sources/pipelines.py`) |
| Tests | `tests/test_signatures.py` (12), `tests/test_claims.py` (+16 de coherencia), `tests/test_pipelines.py` (+2). Total del proyecto: **582** |

Tres detalles de implementación que conviene conocer:

1. **`stable` no se penaliza.** `a = signo × 0 = 0` es **neutro**: categoría
   propia, fuera del numerador y del denominador de la tasa. Un modelo que
   hedgea no empeora su coherencia, y la tabla lo muestra como columna aparte.
2. **La tasa y la tabla conjunta se separan.** `coherence_rate` sólo usa
   `coherent` y `contrary`; `against_supported_share` divide entre los claims
   **verdaderos** (`supported`), y `contrary_supported_rate` entre los
   **contrarios**. Son cocientes distintos y no se mezclan.
3. **El eje es del endpoint binario.** Para corpus de 4 clases se aplica el
   mismo mapa que la jerarquía Schmidt (`stress` frente al resto) y se declara
   como extensión, no como parte del preregistro.

El pre-chequeo P0 sobre el corpus histórico y la primera lectura exploratoria
están en [`claims_h5.md`](claims_h5.md) §5.7. Resultados del diseño primario:
pendientes de ejecución.

---

## 11. Reproducción

```bash
# análisis con coherencia, diseño primario, un modelo y una representación
python scripts/analyze_degradation.py \
  --representation numeric --decision-mode binary \
  --nonstress-variant schmidt --temperature exclude --model qwen3:14b

# los cuatro brazos
for m in qwen2.5:7b qwen3:14b; do
  python scripts/analyze_degradation.py --representation numeric \
    --decision-mode binary --nonstress-variant schmidt --temperature exclude --model "$m"
  python scripts/analyze_degradation.py --representation text \
    --decision-mode binary --nonstress-variant schmidt --temperature exclude --model "$m"
done

# por el CLI de fases (mismas banderas)
python main.py --proc EVALUATE --experiment degradation --model qwen3:14b \
  --representation numeric --decision-mode binary --nonstress-variant schmidt \
  --temperature exclude --coherence-set primary
```

En el servidor, por `srun --pty` (el nodo de login aborta con numpy).
`--coherence-set compact` y `--coherence-set sensitivity` son las dos
sensibilidades declaradas; `primary` es el valor por defecto.

---

## 12. Referencias

- Boucsein, W. (2012). *Electrodermal Activity* (2.ª ed.). Springer.
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2007). The electrodermal
  system. En *Handbook of Psychophysiology* (3.ª ed.).
- Grossman, P. (1983). Respiration, stress, and cardiovascular function.
  *Psychophysiology*, 20(3), 284–300.
- Kim, H.-G., Cheon, E.-J., Bai, D.-S., Lee, Y. H., & Koo, B.-H. (2018). Stress
  and heart rate variability: a meta-analysis and review of the literature.
  *Psychiatry Investigation*, 15(3), 235–245.
- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K.
  (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect
  detection. *ICMI*.
- Task Force of the ESC and the NASPE (1996). Heart rate variability: standards
  of measurement, physiological interpretation and clinical use.
  *Circulation*, 93, 1043–1065.
