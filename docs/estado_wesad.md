# Estado del trabajo con WESAD — resumen ejecutivo

**Actualizado:** 2026-11-10

Este documento responde en una página a cuatro preguntas: **qué se construyó**,
**qué se midió**, **qué resultados hay** y **qué es firme y qué es exploratorio**.
Para el detalle, cada sección enlaza al documento correspondiente.

---

## 1. Qué se construyó

Un pipeline reproducible que convierte señales fisiológicas en **respuestas de un
LLM auditables**, y un conjunto de instrumentos para evaluar esas respuestas
frente a la evidencia que el modelo recibió.

```
WESAD (15 sujetos)
   └─ INGEST ──► canonical ──► QC ──► ventanas 60 s ──► degradación ──► features + SQI
                                                              └──► representaciones (numérica / textual)
                                                                        └──► prompts ──► LLM ──► respuestas + procedencia
                                                                                              └──► evaluación (corrección,
                                                                                                   confianza, claims,
                                                                                                   certeza, abstención)
```

- **Corpus**: 15 sujetos, **705 ventanas** de 60 s (283 baseline, 157 stress,
  86 amusement, 179 meditation). Etiquetas separadas del resto de artefactos.
- **Representaciones**: numérica (23 features de pecho) y textual (mismo
  contenido con etiquetas legibles). Los embeddings quedan como material
  complementario, no se envían al LLM.
- **Degradación controlada**: `C0` limpia, `C1` respiración ausente, `C2` EDA
  parcial (sin SCR), `C4` EDA a cero, `C5` ambas. `C3` es idéntica a `C1` a
  nivel de features y se omite.
- **Instrumentos**: verificación de claims, índice de hedges/boosters,
  calibración (ECE/Brier), predicción selectiva (conformal), compuerta de
  consistencia fisiológica y SQI.

Detalle: [`ingestion.md`](ingestion.md), [`sqi.md`](sqi.md),
[`fase2_diseno.md`](fase2_diseno.md), [`degradacion_llm.md`](degradacion_llm.md).

---

## 2. Qué se midió (experimento principal)

**5.670 respuestas** del LLM (`qwen2.5:7b`, T=0,3, semilla 42) sobre los
**12 sujetos de evaluación** (S5–S17 salvo S12), 567 ventanas × 5 condiciones ×
2 representaciones. Las 200 ventanas de desarrollo (S2–S4) solo se usan para
diseñar prompts y ejemplos.

Diseño congelado: [`split_dev_test.md`](split_dev_test.md).

### Resultados principales (representación numérica)

| Condición | Referencia (4 clases) | Llm accuracy | Abstención | ECE | Claims respaldados |
|---|---:|---:|---:|---:|---:|
| C0 limpia | 0,828 | 0,611 | 12,5 % | 0,190 | 0,781 |
| C1 sin respiración | 0,723 | 0,571 | 9,5 % | 0,250 | 0,697 |
| C2 EDA parcial | 0,826 | 0,601 | 20,5 % | 0,201 | 0,734 |
| C4 EDA a cero | 0,702 | **0,289** | 28,0 % | **0,489** | **0,828** |
| C5 ambas | 0,625 | **0,263** | 16,2 % | **0,512** | 0,762 |

Tablas completas, IC y contraste de representaciones:
[`resultados_degradacion.md`](resultados_degradacion.md).

---

## 3. Qué dicen los resultados

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| **H1** corrección bajo degradación | **Apoyada** | 0,611 → 0,263; la referencia cae menos (0,828 → 0,625): exceso de coste 0,15–0,20 |
| **H2a** confianza y corrección | Matizada | AUROC de confianza 0,51 en limpio → 0,76 en C5; la sobreconfianza pasa de +0,15 a +0,47 |
| **H2b** certeza lingüística | **Apoyada en dirección** | Índice +0,13 → −1,12; hedges 0,39 → 0,71 con boosters estables |
| **H3** representación × degradación | **Apoyada** | Texto −0,09 en accuracy; interacción +0,065 [0,025; 0,103]; en C5 el texto se vuelve más asertivo (+1,36) y el numérico más cauto (−1,12) |
| **H4** adecuación de la abstención | **Refutada** | Infra-abstención (brecha error−abstención hasta +0,58) y sobre-abstención en limpio (12,5 % con tarea resoluble) |
| **H5** respaldo de afirmaciones | **Hallazgo principal** | El respaldo **sube** a 0,828 en C4 mientras la accuracy cae a 0,29: claims localmente ciertos, inferencia global errónea. `unsupported = 0` (el modelo deja de citar canales ausentes) |
| **H6** indicador externo de calidad | **Apoyada con SQI** | La compuerta de consistencia no añade AUROC (Δ≈0, mide integridad); el SQI añade +0,05 a +0,09 |

### Lectura en una frase

El LLM **describe correctamente** lo que ve (los claims son ciertos sobre los
valores), pero **no razona sobre el conjunto**: cuando la evidencia se degrada,
sigue afirmando con confianza y solo añade cautela en el lenguaje, sin abstenerse
ni corregir su inferencia.

---

## 4. Qué es firme y qué es exploratorio

### Firme (diseño congelado, split dev/test, sin fuga)

- Barrido de degradación numérico y textual (secciones 2–3).
- Referencia features-only por condición.
- SQI como indicador externo (H6).
- Compuerta de consistencia como mecanismo: 0 % de abstención en limpio y
  100 % en datos estructuralmente incoherentes.

Documento: [`consistencia.md`](consistencia.md).

### Exploratorio (no usar como resultado final)

- **Corpus anterior** (`results/output/WESAD/PHASE2/llm_responses/`): generado
  con few-shot contaminado por estadísticos del test. Queda como material de
  desarrollo.
- **Endpoint binario stress/no-stress** (accuracy 0,904; AUROC 0,908) y
  **representación centrada E12** (AUROC stress 0,958): decisiones post hoc,
  declaradas como desviación. Documentos: [`p2_binario.md`](p2_binario.md),
  [`e12_resultados.md`](e12_resultados.md).
- **Calibración y predicción selectiva** (rúbrica graduada, cobertura 6 % con
  precisión selectiva 1,000): [`p4_calibracion.md`](p4_calibracion.md).
- **Resolubilidad de 4 clases** y confusor temporal:
  [`resolubilidad.md`](resolubilidad.md).

### Validación externa (resultado negativo, documentado)

- **K-EmoCon**: no tiene etiqueta de estrés y sus etiquetas de arousal/valencia
  no son predecibles con EDA/HR de muñeca (AUROC ≤ 0,61). El contraste
  `pre`/`debate` sí es detectable (0,785) pero mide tarea/actividad, no afecto.
  [`p3_kemocon.md`](p3_kemocon.md).
- **PhysioNet/CinC 2017** (ECG, 4 clases con `noisy`): el modelo colapsa a
  `normal` (balanced 0,25; referencia 0,55), 0 % de abstención y 0 % de
  detección de `noisy`; con few-shot en desarrollo detecta AF (recall 0,85) pero
  sobre-predice esa clase. [`p3_physionet.md`](p3_physionet.md).

---

## 5. Pendientes

1. Validación humana de una muestra de claims y hedges/boosters (instrumentos).
2. Commit/push del trabajo.
3. Decisión sobre corpus/datasets y tipos de representación (siguiente línea).

Memoria operativa: [`AGENTS.md`](../AGENTS.md), [`SESSION_STATE.md`](../SESSION_STATE.md),
[`CHANGELOG.md`](../CHANGELOG.md).
