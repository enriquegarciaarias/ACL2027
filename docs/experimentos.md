# Experimentos y resultados — visión general

**Actualizado:** 2026-11-15

Este documento es el **punto de entrada único** al trabajo experimental del
proyecto. Contiene, en este orden:

1. un **resumen ejecutivo** (qué se construyó, qué se midió, resultados en una
   página y qué es firme frente a exploratorio);
2. un **inventario de todos los experimentos** con su pregunta, datos, tamaño
   muestral, métrica clave, resultado, estado y documento de detalle;
3. el **detalle compacto de cada experimento**;
4. **limitaciones transversales**, **pendientes** y **mapa de documentos**.

Para el detalle metodológico completo de cada pieza, el documento enlazado en
cada apartado es la referencia; este documento no lo reemplaza.

---

# 1. Resumen ejecutivo

La tesis pregunta si los LLM **dicen lo que la evidencia respalda**. No se mide
solo el acierto: se mide cómo cambian cinco dimensiones cuando la entrada se
degrada de forma controlada y conocida.

| Dimensión | Instrumento |
|---|---|
| Corrección | accuracy y balanced accuracy frente a etiqueta y frente a referencia features-only |
| Confianza numérica | ECE, Brier, AUROC de confianza, calibración por temperatura, conformal |
| Certeza lingüística | índice de *hedges* y *boosters* por 100 palabras |
| Respaldo de afirmaciones | verificador determinista de claims contra el prompt recibido |
| Abstención | tasa por condición, sobre evidencia degradada y sobre evidencia ilegible real |

## 1.1 Qué se construyó

Un pipeline reproducible que convierte señales fisiológicas en **respuestas de
un LLM auditables**, más los instrumentos para evaluarlas contra la evidencia que
el modelo realmente recibió.

```
WESAD (15 sujetos)
   └─ INGEST ──► canonical ──► QC ──► ventanas 60 s ──► degradación ──► features + SQI
                                                              └──► representaciones (numérica / textual)
                                                                        └──► prompts ──► LLM ──► respuestas + procedencia
                                                                                              └──► evaluación (corrección,
                                                                                                   confianza, claims,
                                                                                                   certeza, abstención)
```

- **Corpus WESAD**: 15 sujetos, 12,9 GB, **705 ventanas** de 60 s (283 baseline,
  157 stress, 86 amusement, 179 meditation). Etiquetas aisladas en
  `groundtruth/`; las features y representaciones se construyen sin etiquetas y la unión etiqueta ↔ predicción ocurre solo en evaluación.
- **Representaciones**: numérica (23 features de pecho) y textual (mismo
  contenido con etiquetas legibles). Los embeddings quedan como material
  complementario: no se envían al LLM.
- **Degradación controlada**: `C0` limpia, `C1` respiración ausente, `C2` EDA
  parcial (sin SCR), `C4` EDA a cero, `C5` ambas. `C3` es idéntica a `C1` a nivel
  de features y se omite por diseño (declarado en el preregistro). El canal de
  **ECG se mantiene intacto en todas las condiciones** y no se degrada en ninguna;
  sirve como señal de referencia estable y las features de HR/HRV derivadas de
  ECG siguen disponibles para el LLM en cada condición.

  **Por qué C3 = C1 en respiración:** el extractor de respiración (`resp.py`) no
  lanza error ante datos faltantes, pero sus features clave (`rate`, `cv`,
  `amplitude`) dependen de detectar troughs/picos en la señal limpiada con
  `nk.rsp_peaks`. Con demasiados gaps (≥50 % de ausencia), la señal limpiada tiene
  pocos o ningún trough detectado → `n_cycles = 0`, `intervals` vacío →
  `rate = None`, `cv = None`, `amplitude = None`, `std = None`. En consecuencia,
  C1 (≥50 % ausencia) y C3 (respiración total) producen las mismas features
  (`None` para toda la modalidad de respiración), por lo que el LLM recibe una
  representación idéntica y `C3` se omite del barrido efectivo.
- **Instrumentos**: verificación de claims, índice de hedges/boosters,
  calibración (ECE/Brier), predicción selectiva (conformal), compuerta
  determinista de consistencia fisiológica y SQI.
- **Reproducibilidad**: fingerprints por artefacto, hashes de prompt como caché,
  procedencia completa por respuesta (modelo, semilla, temperatura, `num_ctx`,
  latencia), split desarrollo/evaluación congelado y determinista.
- **Modelos comparados**: `qwen2.5:7b` (referencia del corpus publicado) y
  `qwen3:14b` (matriz completa bajo degradación, numérica y binaria).

Detalle: [`ingestion.md`](ingestion.md), [`sqi.md`](sqi.md),
[`fase2_diseno.md`](fase2_diseno.md), [`degradacion_llm.md`](degradacion_llm.md).

## 1.2 Qué se midió

El experimento principal son **5.670 respuestas** del LLM (`qwen2.5:7b`, T=0,3,
semilla 42) sobre los **12 sujetos de evaluación** (S5–S17 salvo S12 que no tiene datos):
567 ventanas × 5 condiciones × 2 representaciones. Ese mismo diseño se
ejecutó con **`qwen3:14b`** bajo degradación (representación numérica,
task de 4 clases), y el **barrido binario** (`stress` / `non-stress`) se
ejecutó para ambos modelos sobre las mismas 567 ventanas × 5 condiciones
(5.670 respuestas cada modelo, 11.340 en total).

El split está congelado y determinista (documentado en docs/split_dev_test.md):
- Desarrollo: S2, S3, S4 → nunca aparecen como sujetos evaluados en los resultados finales.
- Evaluación: S5–S17 salvo S12 → aquí se miden accuracy, balanced, AUROC, abstención, etc.

Se incluyen adicionalmente dos validaciones externas (K-EmoCon y PhysioNet/CinC 2017).

### 1.2.1 4 clases vs binario: por qué el endpoint binario

El experimento preregistrado usaba un endpoint de **4 clases** (baseline, stress, amusement, meditation). Sin embargo, el análisis empírico del corpus completo (705 ventanas, 15 sujetos) mostró que el LLM **nunca predice `amusement` ni `meditation`** (0/705 en ambas representaciones, numérica y textual): todas las ventanas de esas dos clases se mapean a `baseline` o `abstain`. El problema efectivo del LLM es, por tanto, binario: `stress` vs `non-stress`.

Este cambio no se hizo por falta de separabilidad de las clases. La referencia features-only (misma regresión logística sobre las mismas 23 features) separa 4 clases con balanced 0,644–0,677 y stress vs baseline con balanced ≈ 0,87–0,92. La información fisiológica existe; el problema es que el LLM no la utiliza para distinguir las cuatro clases.

**Argumentos experimentales para el endpoint binario:**

1. **El modelo no usa el espacio de 4 clases.** En el corpus completo, `amusement` y `meditation` tienen AUROC ≈ 0,5 (casi azar) tanto para el 7b como para el 14b. El modelo las confunde sistemáticamente con `baseline`.
2. **En limpio, no hay ventaja del 14b en 4 clases que no sea abstención.** La ventaja observada del 14b (0,703 vs 0,611 del 7b) se debe principalmente a que el 14b se abstiene más (hasta 100 % en C5), no a que discrimine mejor. En limpio, los modelos empatan (Δ +0,014, IC [−0,012; +0,040]).
3. **Con degradación severa (EDA a cero), fallan en direcciones opuestas.** El 7b sobrerreclama estrés (balanced 0,63, recall 0,78); el 14b dice `non-stress` para todo (balanced 0,500 = tasa base, accuracy 0,845). La accuracy premia al clasificador degenerado; es obligatorio reportar balanced y cobertura.
4. **La abstención es el residuo del espacio de etiquetas.** En 4 clases, el modelo usa `abstain` cuando duda. En binario, `non-stress` absorbe toda la ambigüedad (baseline, amusement y meditation se agrupan), y la abstención desaparece (0 % en C0/C1 del 7b; 0–17,6 % en el 14b). El prompt binario contiene la misma regla de abstención que el de 4 clases, pero la diferencia es que existe un destino para la duda (`non-stress`).

**Evidencia en la literatura.** El uso de un endpoint binario con WESAD no es una decisión aislada de esta tesis. El corpus SLIP contiene seis papers que mencionan o usan WESAD; de los cuatro que ejecutan experimentos con él, dos emplean un espacio binario, uno usa tres clases y uno no clasifica (detección de anomalía personalizada):

| Paper | ID | Enfoque de clases en WESAD | Detalle |
|-------|-----|---------------------------|---------|
| StressLLM | `scopus_111` | **Binario** estrés / no estrés | Convierte las 5 clases originales (Baseline, Amusement, Stress, Meditation, Recovery) en binario. Remuestrea a 20 Hz, usa ventanas de 14 días para PMData, y para WESAD asigna etiqueta por valor mayoritario. |
| BLP | `arxiv_51` | **3 clases** (baseline, estrés, amusement) | Omite meditation y recovery. Usa las mismas 61 features de HRV que TILES-2018. |
| UniTS | `pubmed_150` | **No aplica** | No clasifica clases afectivas. Reformula el problema como **detección de anomalía personalizada** (desviación de la línea base individual por sujeto). Usa ECG y BVP de WESAD. |
| ieee_88 | `ieee_88` | No especificado claramente | Usa WESAD junto con HHAR y DREAMT para predicción de estados humanos (estrés, sueño-vigilia, actividad). No especifica cuántas clases de WESAD usa en el texto disponible. |

**Evidencia textual de `scopus_111` (StressLLM) sobre el colapso a estrés en zero-shot y el binario como espacio estable:**

> "En zero-shot, GPT-4 y GPT-3.5-Turbo obtienen MAE de 0,8067 y 0,8133 al excluir el acelerómetro de pecho y el ECG de pecho, respectivamente; pero, pese a su tamaño, **respondieron «estresado» a casi todas las muestras**." (`scopus_111`, líneas 216-217 de `docs/lecturas.md`)

> "Llama2 rinde mal: **predice «estresado» en la mayoría de las muestras** y su mejor MAE en zero-shot es 0,8274 con todos los biomarcadores." (`scopus_111`, línea 223 de `docs/lecturas.md`)

> "Su conversión de WESAD a binario (estrés / no estrés) con remuestreo a 20 Hz **es un precedente directo de nuestro brazo binario**." (`scopus_111`, línea 150 de `docs/lecturas.md`)

> "WESAD: <span style='color: red;'>se convierten las clases originales (**Baseline, Amusement, Stress, Meditation y Recovery**) en **binario estrés / no estrés**</span>" (`scopus_111`, líneas 181-182 de `docs/lecturas.md`)

**Citas:** `scopus_111` tiene 14 citas según `papers_metadata.json`. No hay información de shares (compartidos en redes sociales) en los documentos del proyecto.

**El paper original de Schmidt et al. (2018) ya define un endpoint binario.** El paper fundacional de WESAD no contempla 4 clases afectivas, sino **tres** (baseline, stress, amusement). Las dos fases de meditación se introducen explícitamente para "de-excitar" al sujeto tras el estrés o el amusement, no como una clase afectiva propia:

> "The amusement and stress conditions, which both aimed at exciting the subjects, were followed by a guided meditation. The aim of this meditation was to 'de-excite' the subjects and bring them back to a close to neutral affective state." (Schmidt et al., 2018, líneas 259-262 de `docs/Introducing WESAD.pdf`)

Además, la distribución de clases en los datos es **desigual**:

> "Out of these windows, **53 % belong to the baseline class, 30 % represent the stress class, and 17 % originate from the amusement condition**." (Schmidt et al., 2018, líneas 786-787 de `docs/Introducing WESAD.pdf`)

Y los propios autores **definen explícitamente una tarea binaria** combinando baseline y amusement en una clase "non-stress":

> "First, a three-class problem was defined: baseline vs. stress vs. amusement. [...] Second, a binary classification task was defined by combining the states baseline and amusement to a non-stress class, posing the stress vs. non-stress classification problem." (Schmidt et al., 2018, líneas 864-866 de `docs/Introducing WESAD.pdf`)

Los resultados del benchmark en el paper original confirman que el espacio binario es más estable y alcanza mejor precisión que el espacio multiclase:

| Tarea | Mejor accuracy | Mejor F1 |
|-------|---------------|----------|
| 3 clases (baseline vs stress vs amusement) | 80 % | – |
| Binario (stress vs non-stress) | **93 %** | – |

> "Depending on the input modalities, these classifiers reach scores up to 80 % for the three-class problem and up to 93 % for the binary task, respectively." (Schmidt et al., 2018, líneas 798-799 de `docs/Introducing WESAD.pdf`)

> "In both classification tasks the accuracies using one of the wrist-based physiological modalities range from 69 % to 86 % and the accuracies using one of the physiological chest-based modalities range from 67 % to 88 %. In both classification tasks the RESP is a particularly strong chest-based modality leading to the best result of a single modality." (Schmidt et al., 2018, líneas 814-818 de `docs/Introducing WESAD.pdf`)

> "The results reached using all physiological chest-based modalities (three-class accuracy: 80 %, binary accuracy: 93 %) are higher than the ones obtained using all physiological wrist-based" (Schmidt et al., 2018, líneas 831-832 de `docs/Introducing WESAD.pdf`)

**Conclusión.** El endpoint binario `stress` vs `non-stress` es el espacio de decisión **efectivo** del LLM en este experimento. No se elige por falta de separabilidad de los datos, sino porque el LLM no utiliza las cuatro clases de forma significativa. Se justifica de forma pragmática: relevancia clínica del estrés, potencia estadística y espacio de decisión que el modelo realmente utiliza. Además, el propio paper original de WESAD ya define y valida este espacio binario, alcanzando 93 % de accuracy frente al 80 % de la tarea de 3 clases, lo que refuerza que el endpoint binario no es una imposición arbitraria sino una decisión metodológica coherente con el diseño original del dataset.

### 1.2.2 Jerarquía binaria y temperatura (Fase 1, 2026-11-16)

La justificación anterior cita a Schmidt, pero la implementación del prompt binario
incluía `meditation` en `non-stress`. Eso **no es la tarea de Schmidt**: su
`non-stress` es `baseline` + `amusement`; `meditation` es una fase de
de-excitación, no una clase afectiva. Para alinear código y justificación se fija
una **jerarquía declarada** antes de re-ejecutar:

| Variante | `non-stress` | n (evaluación) | Prevalencia stress | Tasa base (siempre non-stress) | Papel |
|---|---|---:|---:|---:|---|
| `baseline` | baseline | 355 | 0,358 | 0,642 | Sensibilidad (constructo homogéneo) |
| **`schmidt`** | **baseline + amusement** | **424** | **0,300** | **0,700** | **Primario** |
| `no_amusement` | baseline + meditation | 498 | 0,255 | 0,745 | Sensibilidad (aísla amusement) |
| `all` | baseline + amusement + meditation | 567 | 0,224 | 0,776 | Sensibilidad (corpus histórico) |

Datos: 12 sujetos de evaluación (S5–S17 salvo S12); 228 `baseline`, 127
`stress`, 69 `amusement`, 143 `meditation`. Restringido a las tres clases de
Schmidt, la distribución es **53,8 % / 29,8 % / 16,3 %**, prácticamente la del
paper (53 / 30 / 17).

**Decisiones asociadas:**

1. **Primario: 424, `schmidt`.** Excluye `meditation` por constructo (no por
   resultado), conserva `amusement` (parte de Schmidt y contrapeso parcial del
   orden del protocolo) y reproduce el prior del benchmark.
2. **Temperatura fuera del primario.** Es un confusor verificado (r = 0,801 con
   la posición) y la **única modalidad de pecho que no se degrada**: podría
   amortiguar C1/C2/C4/C5 y atenuar H1. El primario ve `ecg`, `eda` y `resp`;
   la sensibilidad con temperatura mide cuánto importa.
3. **Sensibilidades: 355, 498 y 567.** No se eligen por acierto; se reportan
   todas. Si coinciden en dirección y significación, la conclusión es robusta a
   la composición de clases; si divergen, la divergencia es un hallazgo sobre la
   heterogeneidad de `non-stress`.
4. **Comparabilidad con Schmidt: de composición de clases, no de rendimiento.**
   Su 93 % es de clasificadores supervisados con LOSO y otras ventanas; el LLM
   zero-shot no replica ese benchmark.
5. **Segunda desviación post hoc.** La primera fue el binario (2026-11-09); esta
   es la jerarquía y la exclusión de temperatura. Se fija ahora, antes de
   re-ejecutar, y se reportan las cuatro variantes.
6. **H1–H5 se recalculan en binario 424.** Las cifras de 4 clases quedan como
   secundarias/exploratorias; no se mezclan con las del primario.
7. **Las cifras binarias de 567 de `docs/resultados_binario.md` son `all`**
   (incluyen `meditation` y temperatura), no la tarea de Schmidt. Se renombran
   como "stress vs all non-stress".

El análisis de la jerarquía se ejecuta sin LLM con
`scripts/analyze_binary_hierarchy.py` (bootstrap agrupado por sujeto). La
especificación completa está en [`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md).

**Resultados (2026-11-17).** Los dos modelos están ejecutados (2.835 respuestas
cada uno) y la clasificación del diseño primario está cerrada:

- **En limpio empatan** en corrección (7b 0,908 / 14b 0,915; IC solapados), con
  reparto de perfil: el 7b está mejor calibrado (ECE 0,068 vs 0,125) y el 14b
  rankea mejor (AUROC 0,964 vs 0,912).
- **La pérdida de respiración es casi gratuita** (C1: 0,915 y 0,917).
- **Con EDA parcial (C2) el 14b se retira** (cobertura 0,778) y sube su accuracy
  aparente a 0,945; el 7b apenas se mueve (0,887 con cobertura 1,00).
- **Con EDA a cero fallan en direcciones opuestas**: el 7b sobrerreclama estrés
  (accuracy 0,548 en C4, **por debajo** de la tasa base 0,700), el 14b
  infra-reclama (1–3 predicciones de estrés; balanced 0,522/0,508 ≈ azar) y su
  accuracy de 0,824/0,826 es un **artefacto de cobertura** (`acc*` 0,705/0,696).
- **La variante no cambia la conclusión**: el colapso del 14b es estructural
  (balanced 0,522 en las cuatro variantes).

Tablas con IC por sujeto, sensibilidades y comandos de reproducción en
[`resultados_jerarquia.md`](resultados_jerarquia.md). Las hipótesis H1, H2a,
H2b, H4, H5 y H6 del primario se calculan con `scripts/analyze_degradation.py`,
que ya resuelve esta variante, descarta `meditation` y construye la referencia
sin temperatura.

## 1.3 Resultados en una página (WESAD, experimento principal)

**Las hipótesis a prueba son:**

| Hipótesis | Qué pregunta |
|---|---|
| **H1** corrección bajo degradación | ¿El modelo acierta menos cuando la señal se degrada, y más de lo que caería solo por falta de señal? |
| **H2a** confianza y corrección | ¿La confianza numérica del modelo refleja su acierto, o está sobreconfianza cuando falla? |
| **H2b** certeza lingüística | ¿El lenguaje del modelo (hedges/boosters) cambia con la degradación de forma coherente con su certeza real? |
| **H3** representación × degradación | ¿La representación (numérica vs textual) afecta a cómo responde el modelo ante la degradación? |
| **H4** adecuación de la abstención | ¿El modelo se abstiene cuando la evidencia es insuficiente, o decide incluso cuando no debería? |
| **H5** respaldo de afirmaciones | ¿Las afirmaciones (claims) que hace el modelo están respaldadas por los valores que realmente vio? |
| **H6** indicador externo de calidad | ¿El SQI (indicador de calidad de señal) añade información útil más allá de la confianza del modelo? |

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| **H1** corrección bajo degradación | **Apoyada** | 4 clases: 0,611 → 0,263; la referencia cae menos (0,828 → 0,625): exceso 0,15–0,20. Primario (424): el LLM **iguala a la referencia en limpio** (0,908 vs 0,905) y el exceso sube a **+0,32 (C4)** y **+0,21 (C5)** |
| **H2a** confianza y corrección | Matizada | AUROC de confianza 0,51 en limpio → 0,76 en C5; la sobreconfianza pasa de +0,15 a +0,47. En el primario solo informa en el 14b (0,867 → 0,731); el 7b emite dos valores de confianza |
| **H2b** certeza lingüística | **Apoyada en dirección** | Índice +0,13 → −1,12; hedges 0,39 → 0,71 con boosters estables. En el primario depende del modelo: 14b decididas −0,25 → −0,85; 7b no monótono |
| **H3** representación × degradación | **Apoyada** | Texto −0,09 en accuracy; interacción +0,065 [0,025; 0,103]; en C5 el texto se vuelve más asertivo (+1,36) y el numérico más cauto (−1,12) |
| **H4** adecuación de la abstención | **Refutada** | Infra-abstención (brecha error−abstención hasta +0,58) y sobre-abstención en limpio (12,5 %). Primario: el 7b no se abstiene nunca (brecha **+0,45** en C4); el 14b se retira por etiqueta, no por dificultad |
| **H5** respaldo de afirmaciones | **Hallazgo principal** | El respaldo **sube** a 0,828 en C4 mientras la accuracy cae a 0,289. Primario: 0,768 → 0,811 (7b) y 0,876 → 0,915 (14b) mientras la decisión se hunde. `unsupported = 0` en todos los brazos |
| **H6** indicador externo de calidad | **Apoyada con SQI, dependiente del corpus** | El SQI añade AUROC incremental (+0,05 a +0,09 en 4 clases; +0,24 en el binario de 567) y la compuerta no añade (Δ≈0). **En el primario no se replica**: ΔSQI inestable en el 7b (−0,14 … +0,12) y ≤ +0,11 en el 14b |

### Detalle

#### H1 corrección bajo degradación

**Qué mide.** El acierto (accuracy) del LLM frente a la degradación controlada de la señal de entrada, comparado con una referencia de modelo clásico que ve las mismas 23 features numéricas. Se evalúa en 5 condiciones efectivas: C0 limpia, C1 respiración ausente, C2 EDA parcial (sin SCR), C4 EDA a cero, C5 ambas. El canal de ECG se mantiene intacto en todas las condiciones y sirve como señal de referencia estable.

**Qué significan los números.**

| | C0 (limpia) | C5 (degradación severa: EDA + respiración destruidas) |
|---|---|---|
| **LLM (qwen2.5:7b)** | 0,611 de accuracy | 0,263 de accuracy |
| **Referencia features-only** | 0,828 de accuracy | 0,625 de accuracy |

El LLM pasa de acertar el 61,1 % en señal limpia al 26,3 % con degradación severa (casi azar en 4 clases). La referencia también cae, pero menos: del 82,8 % al 62,5 %. Eso demuestra que **aún queda información fisiológica usable en C5** (el modelo clásico la aprovecha; el LLM no).

**Referencia features-only.** Es un modelo de **regresión logística (`LogisticRegression` de scikit-learn, `class_weight="balanced"`)** que infiere directamente sobre las mismas 23 features numéricas que recibe el LLM (HR, HRV, EDA tónica/fásica, SCR, respiración), con validación cruzada agrupada por sujeto. No ve texto ni prompts: recibe vectores numéricos puros y está entrenada específicamente para esta tarea de clasificación.

**El LLM cae más que la referencia.** Ambos reciben la misma información subyacente, pero de forma distinta:

| | Referencia features-only | LLM |
|---|---|---|
| Entrada | Vectores numéricos directos | Texto/números embebidos en un prompt |
| Tarea | Clasificar 4 clases con un modelo entrenado para eso | Interpretar texto y devolver JSON con decisión |
| Optimización | Entrenado específicamente en estas features | Modelo de propósito general (no entrenado para esta tarea) |
| Resultado C0 | 0,828 | 0,611 |
| Resultado C5 | 0,625 | 0,263 |

Las causas de la caída mayor del LLM son:

1. **Pérdida de precisión en la representación:** Las features numéricas se convierten a texto (prompt numérico o textual), lo que introduce redondeo, tokenización y formato que la regresión lineal no tiene.
2. **El LLM no está entrenado para esta tarea:** Es un modelo de lenguaje general, no un clasificador de señales fisiológicas. Tiene que "aprender" en el momento (vía few-shot/prompt) qué significa cada feature y cómo combinarla.
3. **Contexto limitado:** Aunque el prompt quepa en `num_ctx`, el LLM debe razonar sobre las features en formato de lenguaje natural, lo que añade ruido y ambigüedad que la regresión logística no tiene.

**Exceso de coste 0,15–0,20.** Es la caída **adicional** del LLM respecto a la referencia cuando la señal se degrada. Mide cuánto más pierde el LLM que el modelo clásico solo por ser un LLM interpretando texto, no por falta de señal:

- La referencia cae 0,203 (de 0,828 a 0,625) — todavía hay señal usable.
- El LLM cae 0,348 (de 0,611 a 0,263) — cae mucho más.
- **Exceso de coste ≈ 0,348 − 0,203 ≈ 0,15–0,20.**

En otras palabras: si el LLM razonara tan bien como la referencia con las mismas 23 features, en C5 estaría cerca de 0,625. Que caiga a 0,263 significa que **unos 0,15–0,20 de accuracy se pierden por el hecho de ser un LLM leyendo texto**, no porque los datos no tengan información.

**Conclusión para H1.** Los datos tienen señal (la referencia lo demuestra). El fallo es del **lector** (el LLM), no de la evidencia. Por eso H1 está apoyada: la degradación reduce el acierto del LLM más de lo que reduciría la mera falta de señal, y esa brecha se explica por la interpretación del modelo, no por ausencia de información fisiológica.

#### H2a confianza y corrección

El **AUROC de confianza** no mide si el modelo acierta clases, sino si su **confianza declarada separa aciertos de errores**: si el modelo dice “confianza 90” cuando acierta y “confianza 40” cuando falla, el AUROC será alto (cerca de 1). Un valor de **0,51 en limpio** indica que la confianza es casi azar: no sirve para saber cuándo el modelo tiene razón. Que suba a **0,76 en C5** significa que, con la señal muy degradada, la confianza empieza a informar algo sobre el error (aunque sigue siendo moderado).

La **sobreconfianza** se mide como la brecha entre **confianza media declarada** y **accuracy real**:

```
sobreconfianza = confianza_media − accuracy_media
```

Si el modelo declara una confianza media de 85 (de 100) pero acierta solo el 70 % de las veces, hay sobreconfianza de +15 puntos.

**Variables implicadas:**

| Variable | Escala | Qué representa |
|---|---|---|
| `confidence` | entero 0–100 | Certidumbre declarada por el LLM en su respuesta JSON |
| `correctness` | 0 o 1 | Si la decisión del LLM coincide con la etiqueta de protocolo (ground truth) |
| `decision` | clase (baseline/stress/amusement/meditation/abstain) | Lo que predijo el modelo |
| `accuracy` | 0–1 | Proporción de aciertos sobre las ventanas decididas |

El AUROC de confianza usa `confidence` como puntuación para predecir `correctness`; la sobreconfianza compara el promedio de `confidence` con el promedio de `correctness`.





#### H2b certeza lingüística

**Qué mide.** Si el lenguaje del modelo (uso de *hedges* y *boosters*) cambia de forma coherente con su certeza real cuando la señal se degrada.

**Índice de certeza.** Se calcula como:

```
índice de certeza = (boosters − hedges) / 100 palabras
```

- **Hedges**: palabras que rebajan la certeza ("puede", "posiblemente", "algo", "ligero", "parece", "aproximadamente").
- **Boosters**: palabras que refuerzan la certeza ("definitivamente", "claramente", "marcadamente", "muy", "significativamente").

El índice es positivo cuando hay más boosters que hedges (lenguaje asertivo) y negativo cuando predomina la cautela.

**Qué significan los números.**

| Métrica | C0 (señal limpia) | C5 (señal muy degradada) |
|---|---|---|
| **Índice de certeza** | **+0,13** | **−1,12** |
| **Hedges** | 0,39 por 100 palabras | 0,71 por 100 palabras |
| **Boosters** | estables | estables |

**Interpretación.**

- **En C0 (limpio):** el índice es **+0,13**, ligeramente positivo. El modelo usa más boosters que hedges (o un equilibrio leve hacia la certeza). Los hedges son escasos (0,39 por 100 palabras).
- **En C5 (degradación severa):** el índice cae a **−1,12**. Ahora hay muchos más hedges que boosters. El modelo se vuelve lingüísticamente cauteloso: usa casi el doble de palabras de cautela (0,71 vs 0,39) mientras los boosters **no aumentan** (se mantienen estables).

**Por qué los boosters son estables.** "Estables" significa que su frecuencia **no varía significativamente entre condiciones**: se mantienen en un número similar por 100 palabras de C0 a C5. El índice cae no porque el modelo use menos palabras de refuerzo, sino porque usa **más palabras de cautela** mientras los boosters se quedan igual. El equilibrio se inclina hacia la cautela sin que haya una reducción en el refuerzo.

**Disociación clave.** Aunque el modelo **no se abstiene** (H4 refutada: decide aunque la evidencia sea basura) y su **acierto cae** (H1), sí cambia su **lenguaje**: cuando la señal se degrada, usa más cautela lingüística (hedges) y menos certeza (índice negativo). Es decir, el modelo "siente" que algo falla y lo expresa con más cautela en el texto, pero **no traduce esa cautela en abstención ni en menor confianza numérica** (H2a: AUROC de confianza bajo, sobreconfianza alta). El lenguaje se vuelve cauteloso, pero la decisión no.

#### H3 representación × degradación

**Formatos de representación que recibe el LLM.**

Ambas vistas usan las mismas 23 features de 4 modalidades de pecho (`ecg`, `eda`, `resp`, `temp`). Las modalidades de muñeca (`bvp`, `wrist_eda`, `wrist_temp`, `wrist_acc`) y los canales de movimiento (`acc`, `emg`) quedan excluidos de la vista LLM para evitar el confusor de movimiento.

**Representación numérica** (`numeric_llm`): objeto JSON con las features agrupadas por modalidad. Cada feature es un par `value` + `unit`. Incluye un identificador de ventana opaco (`win_XXXXXXXX`) para evitar fugas de identidad:

```json
{
  "modalities": {
    "ecg": {
      "ecg_hr_bpm": {"value": 72.5, "unit": "bpm"},
      "ecg_rr_mean_ms": {"value": 828.0, "unit": "ms"},
      "ecg_sdnn_ms": {"value": 45.2, "unit": "ms"},
      "ecg_rmssd_ms": {"value": 32.1, "unit": "ms"},
      "ecg_n_beats": {"value": 72, "unit": "count"}
    },
    "eda": {
      "eda_mean_us": {"value": 4.15, "unit": "µS"},
      "eda_std_us": {"value": 1.23, "unit": "µS"},
      "eda_tonic_mean_us": {"value": 3.80, "unit": "µS"},
      "eda_phasic_std_us": {"value": 0.95, "unit": "µS"},
      "eda_scr_count": {"value": 3, "unit": "count"},
      "eda_scr_rate_per_min": {"value": 3.0, "unit": "1/min"},
      "eda_scr_amplitude_mean_us": {"value": 0.42, "unit": "µS"},
      "eda_scr_amplitude_max_us": {"value": 0.87, "unit": "µS"}
    },
    "resp": {
      "resp_rate_bpm": {"value": 15.2, "unit": "1/min"},
      "resp_interval_cv": {"value": 0.18, "unit": "ratio"},
      "resp_amplitude_pct": {"value": 4.5, "unit": "%"},
      "resp_std_pct": {"value": 2.1, "unit": "%"},
      "resp_n_cycles": {"value": 15, "unit": "count"}
    },
    "temp": {
      "temp_mean_degc": {"value": 36.4, "unit": "°C"},
      "temp_std_degc": {"value": 0.3, "unit": "°C"},
      "temp_slope_degc_per_s": {"value": -0.01, "unit": "°C/s"},
      "temp_range_degc": {"value": 1.2, "unit": "°C"}
    }
  },
  "window": {
    "window_id": "win_a1b2c3d4",
    "duration_s": 60.0
  }
}
```

**Representación textual** (`text_llm`): mismo contenido en formato de prosa, con secciones por modalidad, valor + unidad, e identificador de ventana opaco al inicio:

```
Window: win_a1b2c3d4

ELECTROCARDIOGRAMA (ECG)
- Heart rate: 72.5 bpm
- Mean RR interval: 828.0 ms
- SDNN: 45.2 ms
- RMSSD: 32.1 ms
- Beats detected: 72

ELECTRODERMAL ACTIVITY (EDA)
- Mean conductance: 4.15 µS
- Conductance variability: 1.23 µS
- Tonic level: 3.80 µS
- Phasic variability: 0.95 µS
- SCR count: 3
- SCR rate: 3.0 1/min
- Mean SCR amplitude: 0.42 µS
- Maximum SCR amplitude: 0.87 µS

RESPIRACIÓN
- Respiratory rate: 15.2 1/min
- Breath interval variability: 0.18 ratio
- Amplitude: 4.5 %
- Amplitude variability: 2.1 %
- Breath intervals detected: 15

TEMPERATURA
- Mean temperature: 36.4 °C
- Temperature variability: 0.3 °C
- Temperature trend: -0.01 °C/s
- Temperature range: 1.2 °C
```

¿La representación (numérica vs textual) afecta a cómo responde el modelo ante la degradación controlada de la señal?

Se comparan las dos representaciones sobre las mismas 567 ventanas de evaluación (S5–S17 salvo S12) y las 5 condiciones efectivas (C0, C1, C2, C4, C5). El modelo es `qwen2.5:7b`, semilla 42, sin few-shot de desarrollo en las ventanas de evaluación.

**Qué significan los números.**

| Hallazgo | Valor | Interpretación |
|---|---|---|
| **Texto −0,09 en accuracy** | −0,09 | La representación textual rinde 0,09 menos de accuracy que la numérica en promedio. El texto parte de una base ligeramente más baja de acierto. |
| **Interacción +0,065** | +0,065 [0,025; 0,103] | La brecha entre texto y numérico se reduce a medida que degrada: el texto cae menos que el numérico cuando la señal se destruye. El intervalo de confianza (bootstrap agrupado por sujeto) no cruza cero, por tanto la interacción es estadísticamente significativa: al re-muestrear sujetos, el efecto se mantiene positivo. |
| **En C5, texto más asertivo** | +1,36 | El índice de certeza lingüística del texto es +1,36 superior al del numérico en C5. El texto usa menos hedges y más boosters: se vuelve más asertivo a pesar de tener menos señal. |
| **En C5, numérico más cauto** | −1,12 | El índice de certeza lingüística del numérico en C5 es −1,12. El numérico usa muchos más hedges y menos boosters: se vuelve más cauteloso cuando la señal se degrada. |

La representación importa. El texto comienza con un accuracy algo menor (−0,09), pero es más robusto a la degradación: su caída es menos pronunciada y, en condiciones severas (C4/C5), el texto se vuelve más asertivo mientras el numérico se vuelve más cauteloso. Esta divergencia en certeza lingüística (+1,36 vs −1,12 en C5) es el núcleo de H3: no es solo cuánto acierta cada representación, sino cómo cambia el *comportamiento lingüístico* del modelo cuando la evidencia se destruye.

#### H4 adecuación de la abstención

Si el modelo se abstiene (`decision="abstain"`) cuando la evidencia es insuficiente o no permite decidir, y si se abstiene en la medida adecuada: ni demasiado (sobre-abstención) ni demasiado poco (infra-abstención).

**Infra-abstención.** El modelo **no se abstiene lo suficiente cuando debería**. En condiciones degradadas (especialmente C4 y C5, donde la EDA está destruida o a cero), el accuracy cae a valores cercanos al azar (0,26–0,49), pero el modelo sigue tomando decisiones en lugar de abstenerse. La *brecha error−abstención* mide cuánto supera el error a la abstención: en algunas condiciones alcanza **+0,58**, es decir, el error supera a la abstención en 58 puntos porcentuales. El modelo está equivocado la mayor parte del tiempo, pero apenas se abstiene.

**Sobre-abstención en limpio.** En C0 (señal limpia, tarea resoluble con referencia features-only de 0,828), el modelo se abstiene el **12,5 %** de las ventanas aunque la evidencia es suficiente para decidir con acierto. Es decir, rechaza casos que sí podría resolver.

**Texto del prompt que indica las condiciones de abstención** (instrucción enviada al LLM en cada llamada):

```
Rules:
- Use only the features provided above. Do not invent data.
- If the evidence is insufficient, set decision to "abstain".
- confidence must be an integer between 0 and 100.
- Each claim must reference a real feature from the input.
- Consider all four classes equally; do not bias toward any particular state.
- Respond with valid JSON only. No markdown, no extra text.
```

La regla clave es: *«If the evidence is insufficient, set decision to "abstain".* Sin embargo, en la práctica el modelo no cumple esta regla de forma adecuada: abstiene poco cuando debería (infra-abstención en degradación severa) y abstiene de más cuando no debería (sobre-abstención en señal limpia).

El modelo cambia su **lenguaje** (usa más hedges cuando degrada, H2b) y su **confianza numérica** está descalibrada (H2a), pero no traduce esa incertidumbre en una decisión de abstención proporcional a la calidad real de la evidencia.

#### H5 respaldo de afirmaciones

Las afirmaciones (*claims*) que el LLM hace en su respuesta están respaldadas por las features que realmente vio en el prompt. Un claim es una tupla `(métrica, canal, dirección, magnitud)`. Un claim está *respaldado* si la dirección y magnitud que declara el modelo son compatibles con el valor z de esa feature en los datos de entrada. Está *contradicho* si la dirección o magnitud se invierten respecto a los datos. Está *no respaldado (unsupported)* si la métrica no aparece en el input.

**Qué significan los números.**

| Condición | Accuracy | Respaldo de claims | Claims contradichos | Claims no respaldados (unsupported) |
|---|---|---|---|---|
| C0 limpia | 0,611 | 0,781 | ~22 % | 0 |
| C4 EDA a cero | 0,289 | 0,828 | ~17 % | 0 |
| C5 ambas degradadas | 0,263 | 0,762 | ~24 % | 0 |

**Hallazgo principal (H5).** El respaldo de claims **no cae** con la degradación: sube de 0,781 en C0 a 0,828 en C4, mientras la accuracy se desploma de 0,611 a 0,289. Es decir, el modelo hace claims que son *localmente ciertos* sobre las features que ve (por ejemplo, "EDA disminuye" cuando EDA está a cero o tiene valores muy bajos), pero esos claims locales no se traducen en una inferencia global correcta. El modelo puede describir correctamente cada pieza de evidencia por separado y, sin embargo, llegar a una decisión global errónea.

**unsupported = 0.** En ninguna condición el modelo inventa claims sobre features que no están en el input. Siempre cita métricas reales que vio. El problema no es que alucine features inexistentes, sino que combina claims locales ciertos en una inferencia global errónea.

**Por qué ocurre.** Cuando EDA se degrada (C4) o ambas señales (C5), el modelo sigue viendo features de ECG y respiración que sí tienen signal. Puede afirmar correctamente que "la frecuencia respiratoria es X" o "el ritmo cardíaco es Y" (claims respaldados), pero no logra integrar esa información de forma coherente para decidir el estado afectivo. La inferencia global es errónea aunque cada afirmación individual esté respaldada por los datos.

**Implicación.** La corrección y el respaldo de afirmaciones son ejes distintos. Un modelo puede ser muy bueno justificando cada pieza de evidencia por separado y, aun así, ser muy malo tomando la decisión final. Esto debilita la confianza en la explicabilidad del modelo: unas explicaciones locales correctas no garantizan una decisión global correcta.

#### H6 indicador externo de calidad

Si un indicador externo de calidad de la señal (SQI) añade información útil para predecir cuándo el LLM va a acertar, más allá de la confianza declarada por el propio modelo. También se evalúa si la compuerta de consistencia fisiológica (P1) añade capacidad de detección de errores.

**El SQI (Signal Quality Index).** Es un indicador derivado de las propias features fisiológicas, no una etiqueta externa. Por cada modalidad de pecho (ECG, EDA, respiración) se calcula un SQI parcial basado en propiedades de la señal (ej. variabilidad, presencia de picos, rango dinámico, continuidad temporal) y se agrega en un SQI global por ventana. Valores bajos de SQI indican que la señal es ruidosa, está aplanada o tiene gaps, por lo que las features extraídas de ella son poco fiables.

**La compuerta de consistencia (P1).** Es una comprobación determinística previa al LLM que verifica ocho identidades y desigualdades entre features sin usar etiquetas ni el modelo de lenguaje. Ejemplos:
- `ecg_hr_vs_rr_mean`: la frecuencia cardíaca derivada de los latidos detectados debe ser coherente con el intervalo RR medio.
- `ecg_n_beats_vs_hr`: el número de latidos en la ventana debe ser consistente con la HR reportada.
- `eda_scr_rate_vs_count`: el número de SCR detectados debe ser coherente con la tasa de SCR por minuto.
- `eda_mean_vs_tonic`: la media de EDA debe ser coherente con su componente tónica.
- `resp_n_cycles_vs_rate`: el número de ciclos respiratorios detectados debe ser coherente con la tasa respiratoria.

Cada comprobación tiene una tolerancia configurable. Si se superan los umbrales permitidos, la compuerta veta la ventana y el sistema se abstiene (`decision="abstain"`, `confidence=0`, `status="gate_abstention"`), conservando la decisión original del LLM como `decision_llm` para auditoría.

**Qué significan los números.**

| Indicador | Δ AUROC incremental | Interpretación |
|---|---|---|
| **SQI** | +0,05 a +0,09 | Añadir el SQI como feature adicional mejora la capacidad de detectar errores del LLM entre 5 y 9 puntos porcentuales de AUROC. El SQI deja rastro de la baja calidad de señal que la confianza del LLM no ve. |
| **Compuerta de consistencia (P1)** | Δ≈0 | La compuerta no añade capacidad predictiva. Mide integridad interna de las features, no información predictiva sobre el estado afectivo. Un veto de la compuerta indica que las features no son coherentes entre sí, no que la ventana sea de una clase u otra. |

**Interpretación.** El SQI sí deja rastro de la calidad de la señal: cuando el SQI es bajo, el LLM tiende a errar más, y usar el SQI como feature ayuda a predecir esos errores. En cambio, la compuerta de consistencia no aporta poder discriminatorio: vetar ventanas incoherentes no ayuda a distinguir clases, solo filtra entradas con problemas de integridad. Esto refuerza la lectura de H6: la calidad de la señal deja rastro detectable, pero una comprobación de integridad interna no es suficiente para mejorar la decisión del modelo.

**En el diseño primario (424, binario `schmidt` sin temperatura) el efecto no se
replica de forma estable.** En el 7b el ΔSQI va de **−0,139** en limpio a
+0,119 en C2 y −0,042 en C5 (la confianza es constante, AUROC ≈ 0,51); en el 14b
es pequeño y positivo (+0,001 a +0,112), y donde más aporta es en C4, cuando su
confianza ya ha perdido resolución. La compuerta sigue en Δ = 0 en las diez
celdas. Con 12 sujetos y 424 ventanas, la varianza del estimador es alta: H6 se
cita como **dependiente del corpus**, no como propiedad del indicador
([`resultados_jerarquia.md`](resultados_jerarquia.md) §5.6).

### Lectura de la matriz modelos × tarea (E9)

Con prompt binario y 2 modelos, la jerarquía se invierte al degradar: en
limpio están **empatados** (Δ +0,014, IC [−0,012; +0,040]) y con la EDA a cero
**fallan en direcciones opuestas** (el 7b sobrerreclama estrés, balanced 0,63;
el 14b dice `non-stress` para todo, balanced 0,50). La accuracy premia al
segundo porque 0,845 es la tasa base: **la evaluación de robustez debe usar
balanced y cobertura, no accuracy**. La abstención resulta ser el *residuo del
espacio de etiquetas*: desaparece cuando existe una clase comodín
(`non-stress`). Detalle completo en [`resultados_binario.md`](resultados_binario.md).

**Cifras clave de la matriz (prompt binario):**

| Condición | 7b acc | 7b balanced | 14b acc | 14b balanced | Δ acc (14b−7b, IC 95 %) |
|---|---:|---:|---:|---:|---:|
| C0 limpia | 0,914 | 0,897 | 0,928 | 0,881 | +0,014 [−0,012; +0,040] |
| C1 respiración ausente | 0,910 | 0,908 | 0,926 | 0,882 | +0,016 [−0,018; +0,048] |
| C2 EDA parcial | 0,894 | 0,867 | 0,955 | 0,756 | +0,061 [+0,041; +0,080] |
| C4 EDA a cero | 0,547 | 0,630 | 0,821 | 0,500 | +0,274 [+0,189; +0,361] |
| C5 ambas degradadas | 0,495 | 0,613 | 0,845 | 0,500 | +0,350 [+0,235; +0,463] |

En C4/C5 el 14b gana en accuracy (0,845) solo porque es la tasa base de
`non-stress`; su balanced (0,500) revela un clasificador constante. El 7b,
con menor accuracy (0,512), conserva balanced 0,613.

### Lectura de los controles externos

- **PhysioNet/CinC 2017** (ritmo en ECG, clase `noisy` real): el modelo cae al
  **azar exacto** (balanced 0,250 frente a 0,552 de la referencia), **no se
  abstiene en 103/103** registros ilegibles y su certeza lingüística **sube**
  donde no hay evidencia. Aquí, a diferencia de WESAD, la calidad **sí** deja
  rastro en las features (detector AUROC 0,873).
- **K-EmoCon** (afecto en habla espontánea): sus etiquetas de arousal/valencia no
  son predecibles con EDA/HR de muñeca (AUROC ≤ 0,609). Resultado negativo
  documentado: no sirve como validación del endpoint.

---

# 1.4 Evolución ACL: continuidad y avance sobre papers de ACL

Esta tesis se sitúa como continuación y avance de una línea de trabajos publicados en la conferencia **ACL (Association for Computational Linguistics)** que abordan aspectos parciales de la problemática tratada aquí: calibración de confianza, abstención, respaldo de afirmaciones (claims), certeza lingüística y uso de señales fisiológicas con LLM. La tabla siguiente sintetiza los papers de ACL más relevantes, qué resuelven y qué dejan sin resolver, y en qué consiste el avance de esta tesis sobre cada uno.

| Paper ACL | Problema que aborda | Qué hace | Qué deja sin resolver | Avance de esta tesis |
|-----------|---------------------|----------|----------------------|----------------------|
| `acl_anthology_231` ClinDet-Bench: Beyond abstention, evaluating judgment determinability of LLMs in clinical decision-making | Determinabilidad: ¿permite la información disponible decidir? | Evalúa 16 sistemas de scoring, 188 casos, 8 LLMs; detecta abstención excesiva y conclusiones prematuras; 75,9 % de errores por imputar información faltante. | No usa señales fisiológicas; no evalúa degradación controlada; no mide confianza numérica ni certeza lingüística; no analiza respaldo de claims sobre features fisiológicas. | Aplica el concepto de determinabilidad a señales fisiológicas (WESAD) bajo degradación controlada; cuantifica la brecha error–abstención (hasta +0,58); evalúa respaldo de claims sobre features de pecho (ECG, EDA, respiración). |
| `acl_anthology_118` Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | Incertidumbre verbal como rasgo lineal para reducir alucinación confiada. | TriviaQA, NQ-Open, PopQA; detecta y mitiga alucinaciones calibrando incertidumbre verbal; reducción relativa media de ~30 % en alucinaciones confiadas. | Texto QA, no señales fisiológicas; no evalúa degradación de señal; no mide certeza lingüística (hedges/boosters) sobre evidencia fisiológica; no evalúa degradación controlada. | Traslada la calibración de incertidumbre verbal a señales fisiológicas degradadas; mide certeza lingüística (índice hedges–boosters) bajo degradación C0→C5; evalúa respaldo de claims sobre evidencia fisiológica. |
| `acl_anthology_148` Do language models mirror human confidence? | Sobreconfianza y estimación de confianza sin respuesta. | MMLU, GPQA; propone AFCE; reduce ECE; LLMs menos sensibles a dificultad; sesgos demográficos. | Texto QA, no fisiológico; no evalúa degradación de señal; no evalúa respaldo de claims; no evalúa abstención bajo degradación. | Aplica sobreconfianza y calibración a señales fisiológicas degradadas; mide AUROC de confianza (0,51 en limpio → 0,76 en C5); sobreconfianza (+0,15 → +0,47). |
| `acl_anthology_65` LENS: LLM-Enabled Narrative Synthesis for Mental Health | Alinear sensores con LLM para narrativas clínicas de salud mental. | LENS dataset: 258 participantes, 50.957 EMAs, >100.000 pares sensor-texto; Garmin vivoactive 3 + Android; GPS, pasos, acelerómetro, HR, sueño, estrés. | No evalúa degradación controlada; no mide confianza numérica ni certeza lingüística; no evalúa respaldo de claims; no usa degradación C0–C5. | Aplica la idea de sensores+LLM a WESAD con degradación controlada (EDA, respiración); cuantifica cómo cambia la certeza lingüística y el respaldo de claims bajo degradación. |
| `acl_anthology_72` Multimodal Affective Modeling in an LLM-based Intelligent Tutoring System | Reconocimiento multimodal de emociones en LLM-based ITS para alemán. | Audio, vídeo, biosignales (HR, SpO₂, HRV desde smartwatch); audio 81,3 % accuracy, visual ~66 %. | No evalúa degradación de señal; no mide confianza ni certeza lingüística; no evalúa respaldo de claims; no usa degradación controlada. | Traslada el uso de biosignales con LLM a WESAD con degradación controlada; cuantifica cómo degradar EDA y respiración afecta a corrección, confianza y certeza lingüística. |
| `acl_anthology_57` Teaching LLMs to Abstain across Languages | Enseñar a abstenerse con feedback multilingüe. | M-MMLU, M-Hellaswag, Belebele; hasta 9,2 % mejora en lenguajes de bajos recursos; abstención como comportamiento entrenable. | Texto QA multilingüe, no fisiológico; no evalúa degradación de señal; no mide confianza numérica. | Aplica la abstención como comportamiento entrenable a señales fisiológicas degradadas; cuantifica infra-abstención (brecha error–abstención hasta +0,58) y sobre-abstención en limpio (12,5 %). |
| `acl_anthology_56` DeCC: Measuring VLMs' Answer Reliability via Task-Decomposition Consistency Comparison | Medir fiabilidad de respuestas de VLMs mediante descomposición de tarea. | SNLI-VE, VCR, A-OKVQA, Winoground, MMMU, MathVista; DeCC mejora Brier Score y Effective Reliability. | Visión-lenguaje, no fisiológico; no evalúa degradación de señal; no mide certeza lingüística. | Aplica la idea de consistencia y fiabilidad a señales fisiológicas; la compuerta de consistencia (P1) verifica 8 identidades fisiológicas (HR↔RR, n_beats↔HR, SCR_rate↔count, etc.) con 0 % de falsos positivos en limpio y 100 % en datos corruptos. |
| `acl_anthology_245` Speech disfluencies and LLM confidence | Confianza del LLM frente a incertidumbre pragmática. | Roda Viva corpus (344 turnos); longitud de turno domina confianza (β=+14,47); marcadores pragmáticos no significativos; ECE 41,95/41,14. | Texto (habla), no fisiológico; no evalúa degradación de señal; no mide certeza lingüística (hedges/boosters) sobre evidencia fisiológica. | Traslada el estudio de certeza lingüística a señales fisiológicas degradadas; mide índice de certeza (+0,13 → −1,12 de C0 a C5) y hedges (0,39 → 0,71 por 100 palabras). |

**Lectura de la evolución.** Ninguno de los papers de ACL revisados combina **degradación controlada de señal fisiológica** con **corrección, confianza numérica, certeza lingüística, respaldo de claims y abstención** en un mismo marco experimental. Los papers de ACL trabajan sobre texto (QA, habla, diálogo clínico) y, cuando usan señales fisiológicas (LENS, EmoBot), no degradan la señal ni evalúan el comportamiento del LLM bajo degradación controlada. Esta tesis cierra ese gap: WESAD proporciona la señal fisiológica de pecho (ECG, EDA, respiración) con etiquetas de protocolo conocidas; el pipeline de degradación controlada (C0–C5) permite aislar el efecto de perder EDA o respiración; y el marco de evaluación (H1–H6) cuantifica cómo cambia el comportamiento del LLM cuando la evidencia se degrada de forma conocida.

---
# 3. Detalle por experimento

`n` = unidades evaluadas. Estado: **Firme** (diseño congelado o mecanismo
determinista), **Exploratorio** (post hoc, declarado como desviación) o
**Contaminado** (ejecutado antes de corregir la fuga del few-shot: solo material
de desarrollo).

| # | Experimento | Pregunta | Datos / diseño | n | Métrica clave | Resultado | Estado | Doc |
|---|---|---|---|---|---:|---|---|---|
| E0 | Pipeline y corpus | ¿Se puede construir un corpus fisiológico→texto auditable y sin fugas? | WESAD, 15 sujetos, ventanas 60 s | 705 ventanas | SQI medio, integridad | SQI 0,934; 211 artefactos; 9/9 checks | Firme | [`ingestion.md`](ingestion.md) |
| E1 | **Barrido de degradación** (principal) | ¿Cómo cambian corrección, confianza, certeza, claims y abstención al degradar la señal? | 12 sujetos eval.; C0,C1,C2,C4,C5 × numérico/texto | 5.670 respuestas | accuracy y exceso de coste | 0,611 → 0,263; exceso 0,15–0,20 | **Firme** | [`resultados_degradacion.md`](resultados_degradacion.md) |
| E2 | Resolubilidad | ¿Se puede culpar a los datos del fallo del LLM? | Mismas 23 features, GroupKFold por sujeto | 705 ventanas | balanced accuracy | Referencia 0,644–0,677 (4 clases); LLM 0,468 | Firme (análisis) | [`resolubilidad.md`](resolubilidad.md) |
| E3 | Compuerta de consistencia (P1) | ¿Detecta el sistema datos internamente incoherentes sin LLM? | 8 identidades fisiológicas; C0 + controles negativos | 705 × 4 NC | abstención inducida | 0 % en limpio; 100 % en `NC_shuffled` | Firme (mecanismo) | [`consistencia.md`](consistencia.md) |
| E4 | Prompt binario (P2) | ¿Mejora el endpoint stress/no-stress? | Corpus 705 ventanas, `C0_clean` | 705 respuestas | accuracy, cobertura | 0,904 acc; cobertura 99,4 % | Exploratorio · **Contaminado** | [`p2_binario.md`](p2_binario.md) |
| E5 | Calibración y selectiva (P4) | ¿Se puede calibrar y abstenerse por riesgo? | Variante con rúbrica graduada + conformal por sujeto | 705 respuestas | ECE, cobertura selectiva | ECE 0,017; cobertura 6 % con precisión 1,000 | Exploratorio · **Contaminado** | [`p4_calibracion.md`](p4_calibracion.md) |
| E6 | Representación centrada (E12) | ¿Ayuda normalizar por sujeto? | z robusto intra-sujeto, 15 sujetos | 705 respuestas | accuracy, AUROC | 0,485→0,587; AUROC stress 0,744→0,958 | Exploratorio · **Contaminado** | [`e12_resultados.md`](e12_resultados.md) |
| E7 | Validación externa: K-EmoCon | ¿Se replica en afecto con muñeca? | 23 participantes, ventanas 60 s | ~5,5k ventanas | AUROC | Mejor 0,609; LLM 0,642 | Negativo documentado | [`p3_kemocon.md`](p3_kemocon.md) |
| E8 | Validación externa: PhysioNet 2017 | ¿Se replica con otra tarea, sensor y clase `noisy`? | 600 registros (197 dev / 403 test), 5 features ECG | 403 respuestas | balanced accuracy, abstención | 0,250 vs 0,552; 0/103 abstenciones | Firme (análisis) | [`p3_physionet.md`](p3_physionet.md) |
| E9 | **Modelos × tarea bajo degradación** | ¿Resiste mejor el modelo grande? ¿Cambia la tarea la abstención? | 2 modelos × {4 clases, binario} × C0,C1,C2,C4,C5; 12 sujetos eval. | 11.340 respuestas | balanced, cobertura, claims, certeza | En limpio empatan en binario (Δ +0,014 [−0,012; +0,040]); 14b en 4 clases tiene mejor acc (0,703 vs 0,611) pero por abstención (100 % en C5), no discriminación. Con EDA a cero fallan en direcciones opuestas (7b: bal 0,63 sobrerreclamando estrés; 14b: bal 0,50 diciendo non-stress) | **Firme** | [`resultados_binario.md`](resultados_binario.md) |
| **E10** | **Jerarquía binaria Schmidt sin temperatura (diseño primario)** | ¿Se sostiene el binario al alinear la tarea con el paper y quitar el confusor térmico? | 2 modelos × 5 condiciones × 424 ventanas (12 sujetos eval.) | 5.670 respuestas | accuracy, acc*, cobertura, balanced, AUROC, H1–H6 | Empate en limpio (0,908 vs 0,915) e **igualdad con la referencia supervisada** (0,905); exceso de coste **+0,32/+0,21** (H1); el 14b se abstiene 22 % en C2 y predice estrés 1–3 veces en C4/C5 (balanced ≈0,51; `acc*` 0,70); respaldo de claims **sube** 0,768→0,811 y 0,876→0,915 (H5); H6 no se replica (ΔSQI inestable); la comparación entre modelos depende de la métrica (Δ acc +0,277 en C4 con Δ balanced −0,114) | **Firme** | [`resultados_jerarquia.md`](resultados_jerarquia.md) |

### Ejecución y comparación por modelo

Tres fases (verbos) y el experimento como parámetro:
`INGEST` (corpus) → `GENERATE` (respuestas) → `EVALUATE` (métricas), con
`--experiment controls|degradation|centering|physionet|kemocon`.

Los experimentos se ejecutan **un modelo por vez** (sin cascada) y cada modelo
escribe en su propio subdirectorio (`qwen3:14b` → `qwen3_14b`), de modo que
comparar modelos nunca sobrescribe resultados:

```bash
python main.py --proc GENERATE --experiment controls    --subject WESAD --model qwen3:14b
python main.py --proc GENERATE --experiment degradation --subject WESAD --model qwen3:14b
python main.py --proc EVALUATE --experiment degradation --compare-models qwen2.5:7b,qwen3:14b
python main.py --proc GENERATE --experiment degradation --dry-run   # plan + llamadas estimadas
```

`compare_models.py` emite, por condición, abstención, accuracy, balanced, ECE,
respaldo de claims y certeza de cada modelo, más la diferencia de accuracy con
**bootstrap agrupado por sujeto** (IC 95 %). El modelo de referencia del corpus
publicado es `qwen2.5:7b`; el modo *thinking* de `qwen3` está desactivado por
defecto (`--think` lo activa) para mantener comparable el presupuesto de tokens.
Detalle técnico: [`fase2_diseno.md`](fase2_diseno.md) §8.

---



## 3.1 E0 — Pipeline y corpus (`INGEST`)

**Pregunta.** ¿Se puede construir un corpus fisiológico→representación auditable,
sin fugas de etiqueta ni de identidad, que sea reutilizable para auditar a un LLM?

**Diseño.** Pipeline de 8 pasos (`discover → ingest → qc → window → degrade →
features → represent → package`) con idempotencia por fingerprint, escritura
atómica, hashes y procedencia. Ventanas de 60 s sin solape, etiquetas 1–4.

**Resultado.** 15 sujetos (12,9 GB) → 705 ventanas (283/157/86/179). 63 features
por ventana en 10 modalidades, SQI por modalidad y agregado (media 0,934 en S2,
45/45 ventanas en categoría `good`). Corpus `0.1.0` con 211 artefactos, manifiesto
y 9/9 comprobaciones de integridad. Las etiquetas solo existen en
`groundtruth/labels.jsonl`; features y representaciones usan identificadores
opacos.

**Lectura.** La infraestructura no es el cuello de botella: permite congelar el
diseño y repetir cualquier análisis desde artefactos sin volver a las señales.

## 3.2 E1 — Barrido de degradación (experimento principal)

**Pregunta.** ¿Cómo responden corrección, confianza, certeza lingüística,
respaldo de claims y abstención cuando la señal de entrada se degrada de forma
controlada?

**Diseño.** 12 sujetos de evaluación (S5–S17 salvo S12), 567 ventanas, 5
condiciones efectivas × 2 representaciones = 5.670 respuestas. Referencia
features-only entrenada por condición con las mismas 23 features. Cada respuesta
guarda procedencia completa y el prompt exacto que la originó.

**Resultado (representación numérica).**

| Condición | Referencia (4 clases) | LLM accuracy | Abstención | ECE | Claims respaldados |
|---|---:|---:|---:|---:|---:|
| C0 limpia | 0,828 | 0,611 | 12,5 % | 0,190 | 0,781 |
| C1 sin respiración | 0,723 | 0,571 | 9,5 % | 0,250 | 0,697 |
| C2 EDA parcial | 0,826 | 0,601 | 20,5 % | 0,201 | 0,734 |
| C4 EDA a cero | 0,702 | **0,289** | 28,0 % | **0,489** | **0,828** |
| C5 ambas | 0,625 | **0,263** | 16,2 % | **0,512** | 0,762 |

**Lectura.** EDA es la modalidad crítica (la respiración es casi gratuita). El
respaldo de claims **no cae** con la degradación mientras la corrección se
desploma: es el hallazgo principal (H5). La abstención no sigue a la
resolubilidad (H4 refutada). El detalle por hipótesis, con intervalos de
confianza bootstrap por sujeto y el contraste numérico vs texto (H3), está en
[`resultados_degradacion.md`](resultados_degradacion.md); el preregistro y la
matriz de condiciones, en [`experiment_degradation.md`](experiment_degradation.md).

## 3.3 E2 — Resolubilidad

**Pregunta.** Antes de atribuir el fallo al modelo, ¿son las clases separables con
las mismas features que él recibe?

**Diseño.** Regresión logística y Random Forest con GroupKFold por sujeto,
intervalos bootstrap agrupados, pruebas de permutación y control del confusor
temporal (residualización en fase² del protocolo).

**Resultado.** 4 clases: 0,644 (logreg) / 0,677 (RF). Pares: stress-baseline
0,869; baseline-meditation 0,862; stress-meditation 0,912; amusement-meditation
0,926; baseline-amusement 0,621 (el más débil). Solo la fase temporal del
protocolo alcanza 0,857 (1,000 en los pares con baseline): WESAD es secuencial y
los benchmarks ingenuos están inflados. Residualizando en fase², todos los pares
superan 0,72. Centrado por sujeto: 0,823 en 4 clases.

**Lectura.** **La información existe.** El LLM queda en 0,468 (4 clases) y
0,739–0,805 en el binario, frente a 0,869 de la referencia: el fallo es de
representación/razonamiento, no de datos. Esto es lo que permite descartar
«no se puede» como explicación.

## 3.4 E3 — Compuerta de consistencia fisiológica (P1)

**Pregunta.** ¿Puede el sistema detectar que una ventana es internamente
incoherente (y abstenerse) sin llamar al LLM y sin señal externa de calidad?

**Diseño.** Ocho identidades y desigualdades entre features (HR↔RR,
n_latidos↔HR×duración, SCR_rate↔conteo/duración, EDA media↔tónica,
amplitud media≤máxima, ciclos↔frecuencia respiratoria, rango≥2·DE, DE≥DE fásica),
con tolerancias calibradas sobre datos limpios. Se evalúa sobre C0 y cuatro
controles negativos (barajado, ceros, ausentes, NaN) y sobre corrupciones
alternativas.

**Resultado.**

| Condición | Abstención de la compuerta | E12 solo | E12 + compuerta |
|---|---:|---:|---:|
| `C0_clean` (705) | **0,0 %** | 19,3 % | 19,3 % (sin coste) |
| `NC_shuffled` (705) | **100 %** | 23,6 % | **100 %** |
| `NC_nan` / `NC_absent` | 100 % | 100 % | 100 % |
| `NC_zero` | 0,0 % | — | no detectable (vector coherente) |

Robustez: `shuffle_modality_random` 99,9 %, `shuffle_global` 100 %, `swap_one`
33,5 %, `noise_10pct` 96,2 %, con **0 % de falsos positivos** en limpio. La
ablación con la regla en el prompt es negativa: `NC_shuffled` pasa de 13,0 % a
**0,0 %** de abstención y el acuerdo limpio↔corrupto sube de 0,761 a 0,826. Es
decir: el modelo **no ejecuta** la comprobación aunque la representación la
permita; hace falta código determinista.

**Lectura.** El mecanismo funciona y no cuesta nada en limpio. La transferencia a
otro dataset **no es automática**: en PhysioNet (solo ECG) únicamente 2 de las 8
comprobaciones son evaluables y la compuerta vetaría el 100 % si se exigen las
features clave de WESAD. Las compuertas de calidad deben declararse por dataset.

## 3.5 E4 — Prompt binario stress / no-stress (P2)

**Pregunta.** Si el modelo nunca predice `amusement` ni `meditation`, ¿mejora un
prompt explícitamente binario?

**Diseño.** Corpus completo (705 ventanas, `C0_clean`), prompt binario con
few-shot específico y esquema `stress`/`non-stress`/`abstain`.

**Resultado.** Cobertura 99,4 %, accuracy 0,904, balanced 0,907, precision 0,730,
recall 0,911, AUROC 0,908, ECE 0,059, Brier 0,089. Frente al prompt de 4 clases
mapeado a binario (cobertura 79,7 %, accuracy 0,870): gana cobertura, precisión y
calibración; pierde recall y AUROC.

**Lectura.** El endpoint binario es viable, pero es una **decisión post hoc** y se
ejecutó **antes** de corregir la fuga del few-shot. Queda como material de
desarrollo hasta re-ejecutarlo con el split congelado.

## 3.6 E5 — Calibración y predicción selectiva (P4)

**Pregunta.** ¿Se puede calibrar la confianza y abstenerse por riesgo de forma
útil?

**Diseño.** Tabla de fiabilidad, ECE/MCE, Brier, log-loss, escalado de temperatura,
curva riesgo-cobertura, control conformal del riesgo con validación cruzada
agrupada por sujeto y una rúbrica de confianza graduada.

**Resultado.** Con la confianza de la clase predicha el ECE era bueno (0,059) pero
sin resolución: 703/705 respuestas con confianza 85. La rúbrica graduada crea
resolución (42 ventanas con confianza 95, **42/42 correctas**, ECE 0,017) a costa
de más falsos positivos de `stress` (53 → 78; accuracy 0,904 → 0,873). El
conformal acepta solo confianza ≥ 0,90: **cobertura 6,0 % con precisión selectiva
1,000** (Wilson [0,916; 1,000]). La temperatura no aporta (T = 0,92).

**Lectura.** La calibración sirve para **triaje** (modo de alta certeza), no para
la decisión general. También exploratorio y contaminado por la fuga del few-shot.

## 3.7 E6 — Representación centrada por sujeto (E12)

**Pregunta.** ¿Parte del fallo es que el modelo juzga valores absolutos sin
referencia individual?

**Diseño.** Estandarización robusta intra-sujeto (mediana/IQR, recorte ±3) aplicada
a la representación y al prompt; comparación con la vista absoluta sobre las
mismas 705 ventanas.

**Resultado.**

| Métrica | Absoluto | Centrado |
|---|---:|---:|
| Accuracy (4 clases) | 0,485 | **0,587** |
| Balanced accuracy | 0,370 | **0,469** |
| AUROC stress | 0,744 | **0,958** |
| Binario accuracy | 0,787 | **0,930** |
| Binario balanced | 0,739 | **0,937** |
| Abstención | 8,0 % | 19,3 % |

**Lectura.** Mejora el binario y la detección de estrés, **no el problema de 4
clases** (sigue sin predecir `amusement` ni `meditation`). Además pierde
sensibilidad a la corrupción estructural: `NC_shuffled` cae de 95,3 % a 23,6 % de
abstención —por eso la compuerta de E3 es imprescindible—. Es una normalización
**transductiva** (usa la distribución del sujeto evaluado, sin etiquetas): debe
declararse en la tesis.

## 3.8 E7 — Validación externa: K-EmoCon

**Pregunta.** ¿Se replica el hallazgo en afecto con sensores de muñeca y habla
espontánea?

**Diseño.** 23 participantes; tabla de features agregada a ventanas de 60 s
(se usa la tabla ya derivada, en `results/input/EXTERNAL/KEMOCON/`; no se
reingieren las señales crudas); z intra-participante; etiquetas externas y autoinformadas de arousal y
valencia (alta vs baja); referencia logística con validación cruzada por
participante y un piloto LLM de arousal.

**Resultado.** Mejor AUROC de la referencia: **0,609** (`external_arousal`);
`self_arousal` 0,527; `external_valence` 0,587; `self_valence` 0,417 (bajo el
azar). Piloto LLM (4 participantes, 44 ventanas): balanced 0,614, AUROC 0,642, 0
abstenciones, confianza 85 en 42/44. El contraste de protocolo `pre`/`debate` sí
es detectable (0,785) pero mide tarea/actividad, no afecto.

**Lectura.** **Negativo documentado.** K-EmoCon no sirve para validar el endpoint
de estrés (no tiene esa etiqueta) ni el afectivo con estas features. Para validez
externa del estrés hace falta un dataset con inducción (UBFC-Phys, SWELL-KW).

## 3.9 E8 — Validación externa: PhysioNet/CinC 2017

**Pregunta.** ¿Se sostiene el diagnóstico con otra tarea (ritmo cardiaco), otro
sensor (una derivación de ECG) y una clase **natural** de evidencia inutilizable
(`noisy`, etiquetada por experto)?

**Diseño.** 600 registros muestreados (150 por clase), split determinista por
CRC32: 197 desarrollo / 403 evaluación. Datos en
`results/input/EXTERNAL/physionet2017/training2017/`. Cinco features canónicas de ECG en z
poblacional (referencia de desarrollo). Prompt de 4 clases + `abstain`, sin
few-shot. Referencia logística entrenada en desarrollo.

**Resultado.**

| Medida | LLM | Referencia |
|---|---:|---:|
| Accuracy | **0,243** | 0,548 |
| Balanced accuracy | **0,250** (azar exacto) | 0,552 |
| Recall `noisy` | **0,000** | 0,806 |
| Recall `af` | 0,000 | 0,454 |
| Abstención en `noisy` | **0/103** | — |

- Mayoría trivial: 0,266 de accuracy → el LLM queda **por debajo**.
- Exceso de coste: **−0,305**.
- Claims: 1.727, con **47,6 % contradichos** y 0 `unsupported`; el respaldo **no**
  predice el acierto (AUROC 0,510).
- Certeza lingüística **mayor** en `noisy` (−0,64 vs −1,36; p = 0,0008;
  Cliff δ = +0,60).
- La calidad **sí** es detectable aquí: AUROC multivariante 0,873; con recall 0,5
  en `noisy` se pierde 7,0 % de cobertura limpia. La compuerta de E3, sin
  aprender, detecta el 23,3 % de `noisy` con 2,7 % de falsos.
- Few-shot en desarrollo: mueve el colapso de `normal` a `af` (0,282 → 0,375;
  McNemar p = 0,481) y baja el respaldo de claims.

**Lectura.** Replica el diagnóstico con otro sensor y otra tarea: **la
información existe** (AUROC univariante 0,73–0,82 por clase) y el modelo está en
el azar. Aporta dos matices nuevos: (i) la certeza lingüística **sube** con la
degradación natural, al contrario que con la degradación controlada; (ii) la
calidad derivada de las features funciona cuando el ruido es real, lo que
convierte a H6 en una recomendación de diseño (compuerta por dataset).

## 3.10 E9 — Modelos × tarea bajo degradación (detalle)

**Pregunta.** ¿Resiste mejor el modelo grande (`qwen3:14b`) que el de referencia
(`qwen2.5:7b`)? ¿Cambia la tarea (4 clases vs binario) la abstención?

**Diseño.** 2 modelos × {4 clases, binario} × C0,C1,C2,C4,C5; 12 sujetos de
evaluación (S5–S17 salvo S12); 5.670 respuestas por modelo (11.340 en total).
Datos en `results/output/WESAD/PHASE2/degradation_binary/{modelo}/`. Split
congelado; referencia de estandarización calculada solo con C0 de cada sujeto;
few-shot de S2–S4; misma semilla y parámetros de generación. Lo único que cambia
entre columnas es el modelo.

**Resultado.**

*Prompt binario (ver [`resultados_binario.md`](resultados_binario.md)):*

| Condición | 7b acc | 7b balanced | 14b acc | 14b balanced | Δ acc (IC 95 %) |
|---|---:|---:|---:|---:|---:|
| C0 limpia | 0,914 | 0,897 | 0,928 | 0,881 | +0,014 [−0,012; +0,040] |
| C1 respiración ausente | 0,910 | 0,908 | 0,926 | 0,882 | +0,016 [−0,018; +0,048] |
| C2 EDA parcial | 0,894 | 0,867 | 0,955 | 0,756 | +0,061 [+0,041; +0,080] |
| C4 EDA a cero | 0,547 | 0,630 | 0,821 | 0,500 | +0,274 [+0,189; +0,361] |
| C5 ambas degradadas | 0,495 | 0,613 | 0,845 | 0,500 | +0,350 [+0,235; +0,463] |

En limpio, empatan (Δ +0,014, IC cruza cero). Con la EDA a cero, fallan en
direcciones opuestas: el 7b sobrerreclama estrés (balanced 0,63, recall 0,78);
el 14b dice `non-stress` para todo (balanced 0,50; accuracy 0,845 = tasa base).
La accuracy premia al clasificador degenerado: con balanced, el 7b (0,613 en
C5) supera al 14b (0,500).

*Tarea de 4 clases (qwen3:14b):* accuracy 0,703 en C0 (vs 0,611 del 7b) pero
abstención 100 % en C5 — la ventaja es abstención, no discriminación.

*H2b (certeza lingüística):* depende de modelo × tarea. Monótono en 7b-4 clases
(+0,11 → −0,99) y 14b-binario (−1,10 → −3,04); plano en 14b-4 clases y 7b-binario.

*H5 (respaldo de claims):* replica en los cuatro brazos. En C4/C5 binario, el
respaldo sube (0,864 → 0,918 para 14b) mientras la decisión cae al azar. Claims
localmente ciertos, inferencia global errónea.

*H6 (indicador de calidad):* el SQI sustituye a la confianza cuando esta no
informa (7b-binario: AUROC 0,50 → ΔSQI +0,243); aporta poco cuando la confianza
ya informa (14b: 0,79 → +0,043). La compuerta de consistencia no añade (Δ≈0).
En el diseño primario (424) **no se replica** (ΔSQI −0,14 … +0,12 en el 7b; ver
§3.11).

**Lectura.** El tamaño no compra acierto en limpio; compra calibración. La
abstención es el residuo del espacio de etiquetas: desaparece cuando existe una
clase comodín (`non-stress`). En C4/C5, ni balanced ni cobertura son
satisfactorios para ninguno de los dos modelos.

## 3.11 E10 — Jerarquía binaria Schmidt sin temperatura (detalle)

**Pregunta.** ¿Se sostiene el binario al alinear la tarea con el paper original
(`non-stress` = `baseline` + `amusement`) y al quitar la temperatura, que no se
degrada y correlaciona con el orden del protocolo (r = 0,801)?

**Diseño.** 2 modelos × 5 condiciones × **424 ventanas** de 12 sujetos de
evaluación (143 de `meditation` descartadas); variante `schmidt`, sin
temperatura; 2.835 respuestas por modelo; bootstrap agrupado por sujeto (2.000
remuestras). Artefactos en
`degradation_binary_schmidt_notemp/{qwen2_5_7b,qwen3_14b}/`. Tablas completas,
IC y sensibilidades en [`resultados_jerarquia.md`](resultados_jerarquia.md).

**Resultado.** Tasa base (siempre `non-stress`) = 0,700; balanced del azar = 0,500.

| Condición | 7b acc | 7b acc* | 7b bal | 14b acc | 14b acc* | 14b cov | 14b bal |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 limpia | 0,908 | 0,908 | 0,907 | 0,915 | 0,915 | 1,000 | 0,888 |
| C1 respiración ausente | 0,915 | 0,913 | 0,905 | 0,917 | 0,917 | 1,000 | 0,892 |
| C2 EDA parcial | 0,887 | 0,884 | 0,854 | 0,945 | 0,736 | 0,778 | 0,804 |
| C4 EDA a cero | 0,548 | 0,547 | 0,637 | 0,824 | 0,705 | 0,856 | 0,522 |
| C5 ambas degradadas | 0,682 | 0,682 | 0,723 | 0,826 | 0,696 | 0,842 | 0,508 |

`acc*` = abstención contada como error.

**Lectura.**

1. En limpio hay empate en corrección (IC solapados) y reparto de perfil: el 7b
   calibra mejor (ECE 0,068 vs 0,125), el 14b rankea mejor (AUROC 0,964 vs
   0,912).
2. El 14b **se abstiene de forma dependiente de la etiqueta**: 63,8 % de las
   ventanas de estrés en C2, 47,2 % en C4 y 50,4 % en C5, frente a 0,3–4,4 % de
   las de `non-stress`. Se retira justo cuando desaparece la señal que define la
   clase positiva.
3. Con EDA a cero, el 7b sobrerreclama estrés (accuracy 0,548, **por debajo** de
   la tasa base) y el 14b infra-reclama (1–3 predicciones de estrés de 424;
   balanced ≈ azar). La accuracy de 0,824/0,826 es un artefacto de cobertura.
4. El colapso del 14b es **estructural**: balanced 0,522 en las cuatro
   variantes.
5. Frente al corpus histórico (`all` + temperatura), el 7b mejora en C5
   (balanced 0,613 → 0,723): parte del daño atribuido al modelo era de la
   temperatura y de las clases ajenas a la tarea.

**Hipótesis (mismas ventanas y features que el LLM, sin temperatura).**

| Hipótesis | 7b | 14b |
|---|---|---|
| **H1** coste en exceso | Ref. 0,905 → 0,861 (C4); LLM 0,908 → 0,547: **exceso +0,317** (C5: +0,205) | Ref. igual; LLM `acc*` 0,915 → 0,705: **exceso +0,166** (C5: +0,198) |
| **H2a** AUROC de confianza | 0,50–0,54 (dos valores de confianza); sobreconfianza **+0,300** en C4 | **0,867 → 0,731**; sobreconfianza ≤ 0 |
| **H2b** índice de certeza | −0,58 → −0,31, **no monótono** | −0,25 → −0,85 sobre **decididas** (global −1,02 por el texto del veto, −2,71 en las abstenciones) |
| **H4** brecha error−abstención | 0 % de abstención en las 5 condiciones; brecha **+0,453** (C4) | Abstención 22,2 % / 14,4 % / 15,8 %; brecha +0,042 / **+0,151** / +0,146 |
| **H5** respaldo de claims | 0,768 → 0,811; `unsupported` 0; dirección 0,85–0,88 | 0,876 → 0,915; `unsupported` 0; dirección 0,92–0,95 |
| **H6** ΔSQI | **−0,139** (C0) … +0,119 (C2) … −0,042 (C5): inestable | +0,001 … **+0,112** (C4); Δgate = 0 en todo |

En limpio **el LLM iguala a la referencia supervisada** (0,908 / 0,915 frente a
0,905): el fallo bajo degradación no es falta de información. H1, H4 y H5 se
sostienen; H2a, H2b y H6 dependen del modelo, y el «SQI sustituye a la
confianza» **no se replica** en este corpus.

**Comparación pareada 7b vs 14b** (bootstrap por sujeto): Δ accuracy +0,007
[−0,024; +0,042] en C0 y +0,002 [−0,019; +0,021] en C1 (empate); en C2
+0,059 [+0,023; +0,101] pero **−0,151** si la abstención cuenta como error; en
C4/C5 las accuracy suben a favor del 14b (+0,277 y +0,145) mientras el
**balanced** cae (−0,114 y −0,215) y `acc*` casi se anula en C5 (+0,014).
La elección de modelo **depende de la métrica**: accuracy bruta → 14b; balanced
y calibración → 7b; respaldo de claims y ranking de confianza → 14b.

---

# 4. Qué es firme y qué es exploratorio

## Firme

- **E1 · Barrido de degradación** (numérico y textual) con split congelado,
  prompts y few-shot construidos solo con desarrollo, referencia por condición e
  intervalos bootstrap agrupados por sujeto.
- **E2 · Resolubilidad** como análisis (no como resultado del LLM).
- **E3 · Compuerta de consistencia** como mecanismo: determinista, 0 % de falsos
  positivos en limpio, 100 % en datos incoherentes. No usa prompt ni etiquetas.
- **E8 · PhysioNet 2017** como análisis: artefactos congelados, split
  determinista, detector de calidad entrenado solo con desarrollo.
- **E9 · Modelos × tarea bajo degradación**: matriz 11.340 respuestas, bootstrap
  agrupado por sujeto, resultados en [`resultados_binario.md`](resultados_binario.md).
- **E10 · Jerarquía binaria (diseño primario)**: 5.670 respuestas (424 ventanas ×
  5 condiciones × 2 modelos), variante `schmidt` sin temperatura, bootstrap por
  sujeto; clasificación e H1/H2a/H2b/H4/H5/H6 en
  [`resultados_jerarquia.md`](resultados_jerarquia.md).
- **Infraestructura** (E0): corpus, SQI, degradación, procedencia.

## Exploratorio

- **E4 · Prompt binario** y **E5 · calibración/conformal**: decisiones **post hoc**
  (el endpoint binario se eligió tras ver que el modelo nunca predice
  `amusement` ni `meditation`) y ejecutadas **antes** de corregir la fuga del
  few-shot. Deben re-ejecutarse con el diseño congelado antes de citarse como
  resultado.
- **E6 · Representación centrada (E12)**: post hoc y con normalización
  transductiva por sujeto (sin etiquetas, pero usando la distribución del sujeto
  que se evalúa).

## Negativo documentado

- **E7 · K-EmoCon**: no valida el endpoint; sus etiquetas afectivas no son
  predecibles con estas features.
- **E8 · PhysioNet** en su parte de corrección y abstención: el modelo falla; se
  documenta como límite de capacidad, no como error de implementación.

## La fuga que obligó a congelar el diseño

Los ejemplos few-shot iniciales usaban medias por clase de **las 705 ventanas
evaluadas** (HR baseline 72,4 bpm, EDA 3,9 µS…): estadísticos del test dentro del
prompt. Se corrigió tomando medianas de **S2–S4** (138 ventanas), con procedencia
declarada (`FEW_SHOT_PROVENANCE`) y dos tests que bloquean la reaparición de los
valores antiguos. Todo lo ejecutado antes de esa corrección es material de
desarrollo. Detalle y desviaciones declaradas:
[`split_dev_test.md`](split_dev_test.md).

---

# 5. Limitaciones transversales

1. **Dos modelos, familias distintas.** Las conclusiones comparan `qwen2.5:7b`
   y `qwen3:14b`; no se puede separar efecto de tamaño de efecto de familia.
   `qwen3:8b` está instalado y permitiría aislar el tamaño (pendiente, 2
   condiciones ≈ 2 h).
2. **Una sola semilla** en el experimento principal (`seeds=[42]`,
   `repetitions=1`): no se estima la varianza de muestreo del LLM.
3. **Etiquetas de WESAD y de PhysioNet** provienen de un único protocolo/experto;
   `amusement` es fisiológicamente casi idéntica a `baseline` (AUROC 0,621).
4. **Umbral SCR** (0,05 µS) calibrado empíricamente sobre S3, no con literatura ni
   anotación humana.
5. **SQI** sin calibrar contra anotación humana: es un indicador derivado, no una
   verdad de referencia.
6. **Ventanas de 60 s** sin análisis de sensibilidad a 30/120 s.
7. **Muestreo balanceado** en PhysioNet: las métricas miden discriminación, no
   rendimiento en despliegue.
8. **Validación humana de claims y hedges/boosters** pendiente: el verificador es
   determinista y correcto por construcción, pero nadie ha comprobado si un lector
   humano aceptaría sus etiquetas.

---

# 6. Pendientes

1. Re-ejecutar **E4** y **E5** con el split congelado y el few-shot corregido.
2. Reconstruir los ejemplares few-shot de PhysioNet con **registros prototípicos**
   (los actuales son medianas por clase con razonamiento fijo: funcionan como
   prior de etiqueta).
3. Validación humana de una muestra de claims y de hedges/boosters (idealmente
   con un segundo anotador).
4. Repeticiones con semillas distintas en el experimento principal
   (`repetitions > 1`, `seeds` múltiples) para estimar varianza del modelo.
5. Dataset externo con **inducción de estrés** (UBFC-Phys / SWELL-KW) para
   validez externa del endpoint.
6. Análisis de sensibilidad del tamaño de ventana (30/120 s).
7. Validar el umbral SCR y calibrar el SQI con anotación humana.

---

# 7. Mapa de documentos

| Necesitas… | Documento |
|---|---|
| Pregunta de tesis, hipótesis y objetivos | [`hipotesis.md`](hipotesis.md) |
| Cómo evolucionó el experimento (reorientaciones y descartes) | [`evolucion.md`](evolucion.md) |
| Experimento principal, tablas completas e IC | [`resultados_degradacion.md`](resultados_degradacion.md) |
| Preregistro y matriz de degradación | [`experiment_degradation.md`](experiment_degradation.md) |
| Diseño del barrido en el LLM | [`degradacion_llm.md`](degradacion_llm.md) |
| Pipeline de datos y contrato de artefactos | [`ingestion.md`](ingestion.md) |
| SQI: fórmulas y agregación | [`sqi.md`](sqi.md) |
| Diseño de Fase 2 (prompts, esquema, orquestador) | [`fase2_diseno.md`](fase2_diseno.md) |
| Compuerta de consistencia | [`consistencia.md`](consistencia.md) |
| Verificador de claims y certeza | [`claims_h5.md`](claims_h5.md) |
| Split desarrollo/evaluación y fuga corregida | [`split_dev_test.md`](split_dev_test.md) |
| Separabilidad y confusor temporal | [`resolubilidad.md`](resolubilidad.md) |
| Prompt binario (exploratorio) | [`p2_binario.md`](p2_binario.md) |
| Calibración y conformal (exploratorio) | [`p4_calibracion.md`](p4_calibracion.md) |
| Representación centrada (exploratorio) | [`e12_resultados.md`](e12_resultados.md) |
| Validación externa K-EmoCon | [`p3_kemocon.md`](p3_kemocon.md) |
| Validación externa PhysioNet 2017 | [`p3_physionet.md`](p3_physionet.md) |
| Literatura y oportunidades (E1–E14) | [`literatura_slip.md`](literatura_slip.md) |
| Índice completo de documentación | [`README.md`](README.md) |

Memoria operativa: [`AGENTS.md`](../AGENTS.md),
[`SESSION_STATE.md`](../SESSION_STATE.md), [`CHANGELOG.md`](../CHANGELOG.md).
