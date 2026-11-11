# Experimentos y resultados — visión general

**Actualizado:** 2026-11-11

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
  `groundtruth/`; el resto de artefactos no las contiene.
- **Representaciones**: numérica (23 features de pecho) y textual (mismo
  contenido con etiquetas legibles). Los embeddings quedan como material
  complementario: no se envían al LLM.
- **Degradación controlada**: `C0` limpia, `C1` respiración ausente, `C2` EDA
  parcial (sin SCR), `C4` EDA a cero, `C5` ambas. `C3` es idéntica a `C1` a nivel
  de features y se omite por diseño (declarado en el preregistro).
- **Instrumentos**: verificación de claims, índice de hedges/boosters,
  calibración (ECE/Brier), predicción selectiva (conformal), compuerta
  determinista de consistencia fisiológica y SQI.
- **Reproducibilidad**: fingerprints por artefacto, hashes de prompt como caché,
  procedencia completa por respuesta (modelo, semilla, temperatura, `num_ctx`,
  latencia), split desarrollo/evaluación congelado y determinista.

Detalle: [`ingestion.md`](ingestion.md), [`sqi.md`](sqi.md),
[`fase2_diseno.md`](fase2_diseno.md), [`degradacion_llm.md`](degradacion_llm.md).

## 1.2 Qué se midió

El experimento principal son **5.670 respuestas** del LLM (`qwen2.5:7b`, T=0,3,
semilla 42) sobre los **12 sujetos de evaluación** (S5–S17 salvo S12):
567 ventanas × 5 condiciones × 2 representaciones.

Las 200 ventanas de desarrollo (S2–S4) se usan **solo** para diseñar prompts,
ejemplos few-shot y la referencia de estandarización. El split es determinista y
está documentado en [`split_dev_test.md`](split_dev_test.md) (versión 1.0).

A eso se añaden: el corpus completo para los análisis exploratorios (705
ventanas, variantes de prompt), el barrido de controles negativos y dos
validaciones externas (K-EmoCon y PhysioNet/CinC 2017).

## 1.3 Resultados en una página (WESAD, experimento principal)

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| **H1** corrección bajo degradación | **Apoyada** | 0,611 → 0,263; la referencia cae menos (0,828 → 0,625): exceso de coste 0,15–0,20 |
| **H2a** confianza y corrección | Matizada | AUROC de confianza 0,51 en limpio → 0,76 en C5; la sobreconfianza pasa de +0,15 a +0,47 |
| **H2b** certeza lingüística | **Apoyada en dirección** | Índice +0,13 → −1,12; hedges 0,39 → 0,71 con boosters estables |
| **H3** representación × degradación | **Apoyada** | Texto −0,09 en accuracy; interacción +0,065 [0,025; 0,103]; en C5 el texto se vuelve más asertivo (+1,36) y el numérico más cauto (−1,12) |
| **H4** adecuación de la abstención | **Refutada** | Infra-abstención (brecha error−abstención hasta +0,58) y sobre-abstención en limpio (12,5 % con tarea resoluble) |
| **H5** respaldo de afirmaciones | **Hallazgo principal** | El respaldo **sube** a 0,828 en C4 mientras la accuracy cae a 0,289: claims localmente ciertos, inferencia global errónea. `unsupported = 0` |
| **H6** indicador externo de calidad | **Apoyada con SQI** | El SQI añade AUROC incremental (+0,05 a +0,09); la compuerta de consistencia no añade (Δ≈0, mide integridad) |

### Lectura en una frase

El LLM **describe correctamente** lo que ve (los claims son ciertos sobre los
valores), pero **no razona sobre el conjunto**: cuando la evidencia se degrada,
sigue afirmando con confianza y solo añade cautela en el lenguaje, sin abstenerse
ni corregir su inferencia.

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

# 2. Inventario de experimentos

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

# 3. Detalle por experimento

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

**Diseño.** 23 participantes; tabla de features de PSRI agregada a ventanas de
60 s; z intra-participante; etiquetas externas y autoinformadas de arousal y
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
CRC32: 197 desarrollo / 403 evaluación. Cinco features canónicas de ECG en z
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

1. **Un solo modelo.** Todas las conclusiones son de `qwen2.5:7b`. `qwen3:8b` se
   descartó porque su modo *thinking* vaciaba el campo `content`; otros modelos no
   se han probado.
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
