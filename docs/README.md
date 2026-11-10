# Documentación del proyecto ACL2027

Esta carpeta contiene **toda** la documentación del proyecto. Este índice está
pensado para dos lectores:

- **Quien llega sin contexto técnico**: lee los tres documentos de «Empezar
  aquí» en orden; en ~30 minutos tendrá el qué, el cómo y el qué se encontró.
- **Quien quiere el detalle**: usa las tablas por área y entra directamente al
  documento que necesite.

---

## 1. Empezar aquí (lectura mínima)

| Orden | Documento | Qué responde |
|---:|---|---|
| 1 | [`hipotesis.md`](hipotesis.md) | Qué pregunta la tesis, con qué hipótesis y criterios de refutación |
| 2 | [`estado_wesad.md`](estado_wesad.md) | **Resumen ejecutivo**: qué se construyó, qué se midió, resultados y qué es firme |
| 3 | [`resultados_degradacion.md`](resultados_degradacion.md) | Resultados completos del experimento principal (tablas y lecturas) |

Si solo se lee un documento, que sea [`estado_wesad.md`](estado_wesad.md).

---

## 2. Documentos por área

### Datos e ingesta

| Documento | Contenido | Estado |
|---|---|---|
| [`ingestion.md`](ingestion.md) | Pipeline `INGEST` de WESAD: pasos, contrato de artefactos, ventanas, features, representaciones, empaquetado | Firme |
| [`dataIngestion.md`](dataIngestion.md) | Notas extensas de ingesta y estructura del dataset | Referencia |
| [`sqi.md`](sqi.md) | Indicador de calidad de señal (SQI): fórmulas por modalidad, agregación, ejemplos | Firme |
| [`experiment_degradation.md`](experiment_degradation.md) | Preregistro de la degradación controlada: canales mínimos y matriz de condiciones | Firme (diseño) |

### Diseño experimental

| Documento | Contenido | Estado |
|---|---|---|
| [`split_dev_test.md`](split_dev_test.md) | Split congelado (dev S2–S4, test S5–S17), corrección de la fuga del few-shot y desviaciones declaradas | Firme |
| [`fase2_diseno.md`](fase2_diseno.md) | Diseño de Fase 2: prompts, JSON de salida, configuración del modelo, controles negativos, compuerta | Firme |
| [`fase3_evaluacion.md`](fase3_evaluacion.md) | Diseño de Fase 3: métricas de corrección, calibración, abstención | Firme |
| [`degradacion_llm.md`](degradacion_llm.md) | Diseño del barrido de degradación con LLM (condiciones, referencia solo C0, coste) | Firme |

### Resultados

| Documento | Contenido | Estado |
|---|---|---|
| [`estado_wesad.md`](estado_wesad.md) | **Resumen ejecutivo de todo el trabajo con WESAD** | Firme |
| [`resultados_degradacion.md`](resultados_degradacion.md) | H1–H6 y H3: corrección, abstención, claims, certeza, indicadores de calidad | **Final** |
| [`consistencia.md`](consistencia.md) | Compuerta de consistencia fisiológica: 0 % de falsos vetos en limpio, 100 % en datos incoherentes, robustez | Firme |
| [`claims_h5.md`](claims_h5.md) | Verificador de claims y certeza lingüística: definiciones y resultados | Firme (método) |
| [`p2_binario.md`](p2_binario.md) | Prompt binario stress/no-stress: resultados | **Exploratorio** |
| [`e12_resultados.md`](e12_resultados.md) | Representación centrada por sujeto (E12) | **Exploratorio** |
| [`p4_calibracion.md`](p4_calibracion.md) | Calibración, temperatura, conformal y rúbrica graduada | **Exploratorio** |
| [`resolubilidad.md`](resolubilidad.md) | Separabilidad de clases, confusor temporal y modelos de referencia | Firme (análisis) |

### Validación externa

| Documento | Contenido | Resultado |
|---|---|---|
| [`p3_kemocon.md`](p3_kemocon.md) | K-EmoCon: por qué no sirve para estrés; arousal/valencia no predecibles | Negativo documentado |
| [`p3_physionet.md`](p3_physionet.md) | PhysioNet/CinC 2017: colapso a `normal`, 0 % de abstención, clase `noisy` no detectada | Negativo documentado |

### Literatura y material de apoyo

| Documento | Contenido |
|---|---|
| [`literatura_slip.md`](literatura_slip.md) | Análisis del corpus SLIP y 14 oportunidades experimentales (E1–E14) |
| [`SLIP/`](SLIP/) | Índices de literatura (`papers_zoned.json`, `papers_text.json`) |
| [`DimensionesAnalisis.md`](DimensionesAnalisis.md) | Dimensiones de análisis del proyecto (`.pdf` adjunto) |
| [`UsoJSON.md`](UsoJSON.md) | Guía de salidas estructuradas JSON (`.pdf` adjunto) |

### Memoria operativa

| Documento | Contenido |
|---|---|
| [`AGENTS.md`](../AGENTS.md) | Contexto y memoria del proyecto para agentes y nuevas sesiones |
| [`SESSION_STATE.md`](../SESSION_STATE.md) | Bitácora de la última sesión y punto de continuación |
| [`CHANGELOG.md`](../CHANGELOG.md) | Historial de cambios por versión |

---

## 3. Cómo interpretar los estados

- **Firme**: diseño congelado, sin fuga del conjunto de evaluación, con split
  dev/test declarado. Se puede citar como resultado.
- **Exploratorio**: se usó para decidir el diseño (o se detectó una fuga
  posteriormente). Sirve de contexto y de evidencia de desarrollo, **no** como
  resultado final.
- **Negativo documentado**: se intentó y no funcionó; se conserva porque acota
  el alcance de las conclusiones.

El detalle de qué quedó invalidado y por qué está en
[`split_dev_test.md`](split_dev_test.md) (fuga del few-shot, iteraciones y
desviaciones declaradas).

---

## 4. Dónde están los artefactos

| Artefacto | Ruta |
|---|---|
| Corpus WESAD (canónico, QC, ventanas, features, representaciones) | `results/output/WESAD/INGEST/` |
| Corpus degradado (C0–C5) | `results/output/WESAD/INGEST/degraded{,_features}/{condición}/` |
| Respuestas del LLM (degradación, numérico) | `results/output/WESAD/PHASE2/degradation/` |
| Respuestas del LLM (degradación, textual) | `results/output/WESAD/PHASE2/degradation_text/` |
| Reporte agregado de degradación | `results/output/WESAD/PHASE2/degradation_report.json` |
| Corpus anterior (exploratorio) | `results/output/WESAD/PHASE2/llm_responses/` |
| Validación externa | `results/output/KEMOCON/P3/`, `results/output/PHYSIONET2017/P3/` |

---

## 5. Convenciones técnicas

1. Las rutas se resuelven siempre a través de `processControl.env` (poblado por
   `paramsManager.manageEnv()`). Está prohibido hardcodear `"results"` en
   módulos; los scripts de experimentación usan rutas relativas al repositorio.
2. Los logs se escriben con `writeLog(nivel, logger, mensaje)`.
3. La configuración se lee de `processControl.defaults` y `processControl.env`,
   con `processControl.config` para el documento fusionado global + subject.
4. Los procesos se registran en `PROCESS_REGISTRY` de `main.py` y se importan
   de forma diferida.
5. Las etiquetas (`groundtruth/`) nunca se mezclan con features,
   representaciones ni prompts: hay comprobaciones anti-fuga
   (`assert_no_leakage`) y tests que las verifican.
