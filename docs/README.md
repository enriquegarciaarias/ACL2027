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
| 2 | [`experimentos.md`](experimentos.md) | **Resumen ejecutivo + inventario de experimentos**: qué se construyó, qué se midió, resultados y qué es firme |
| 3 | [`conclusiones.md`](conclusiones.md) | **Síntesis entre modelos y datasets**: qué replica y qué no (WESAD × DREAMT × PhysioNet), amenazas a la validez y las cinco afirmaciones que sostiene la evidencia |
| 4 | [`resultados_jerarquia.md`](resultados_jerarquia.md) | **Resultados del diseño primario**: binario `schmidt` sin temperatura (424 ventanas), dos modelos, con cobertura, tasa base e IC por sujeto |
| 5 | [`deriva.md`](deriva.md) | **Por qué** cambia (no solo cuánto): hacia dónde deriva la decisión, qué parte del error induce la degradación y si la deriva es de la información o de la política de decisión |
| 6 | [`resultados_degradacion.md`](resultados_degradacion.md) | Resultados completos del experimento principal (tablas y lecturas) |
| 7 | [`evolucion.md`](evolucion.md) | Cómo se llegó hasta aquí: qué se descartó y por qué (útil para escribir la tesis) |

Si solo se lee un documento, que sea [`experimentos.md`](experimentos.md).

---

## 2. Documentos por área

### Datos e ingesta

| Documento | Contenido | Estado |
|---|---|---|
| [`ingestion.md`](ingestion.md) | Pipeline `INGEST` de WESAD: pasos, contrato de artefactos, ventanas, features, representaciones, empaquetado | Firme |
| [`dataIngestion.md`](dataIngestion.md) | Notas extensas de ingesta y estructura del dataset | Referencia |
| [`sqi.md`](sqi.md) | Indicador de calidad de señal (SQI): fórmulas por modalidad, agregación, ejemplos | Firme |
| [`experiment_degradation.md`](experiment_degradation.md) | Preregistro de la degradación controlada: canales mínimos y matriz de condiciones | Firme (diseño) |
| [`datos_entrada.md`](datos_entrada.md) | Qué datos de entrada necesita cada experimento, dónde viven dentro del proyecto y cómo replicarlos en el servidor | Firme |
| [`verificacion_corpus.md`](verificacion_corpus.md) | Fingerprints por contenido de código y verificación de que el corpus reconstruido es el que alimentó las respuestas | Firme |
| [`ejecucion_slurm.md`](ejecucion_slurm.md) | Lanzamiento en el clúster: `launch_lsi.sh` (lanzador + monitor) y `run_ACL.sh` (trabajo Slurm), GPU, Ollama, logs y códigos de salida | Firme |

### Diseño experimental

| Documento | Contenido | Estado |
|---|---|---|
| [`split_dev_test.md`](split_dev_test.md) | Split congelado (dev S2–S4, test S5–S17), corrección de la fuga del few-shot y desviaciones declaradas | Firme |
| [`fase2_diseno.md`](fase2_diseno.md) | Diseño de Fase 2: prompts, JSON de salida, configuración del modelo, controles negativos, compuerta | Firme |
| [`fase3_evaluacion.md`](fase3_evaluacion.md) | Diseño de Fase 3: métricas de corrección, calibración, abstención | Firme |
| [`degradacion_llm.md`](degradacion_llm.md) | Diseño del barrido de degradación con LLM (condiciones, referencia solo C0, coste) | Firme |
| [`experimento_binario_jerarquia.md`](experimento_binario_jerarquia.md) | Preregistro de la jerarquía binaria: `schmidt` (baseline+amusement) como primaria, sensibilidades 355/498/567, exclusión de temperatura y bootstrap agrupado por sujeto | Firme (especificación) |
| [`repetitions.md`](repetitions.md) | Repeticiones de prompt: diseño, regla de decisión congelada (`R=3` para el agregado, `k≈20` para describir una ventana), resultados de los pilotos y **§9 plan completo pendiente** por brazo, `R` y coste | Firme (pilotos) · plan de campaña guardado |
| [`h5_relevancia.md`](h5_relevancia.md) | **Relevancia en H5**: tabla de firma por clase (12 features primarias, 10 sin firma, 1 en sensibilidad), regla de coherencia, métricas, predicciones P0–P4 y falsadores F1–F5. Implementado (`signatures.py`, `claims.py`, `--coherence-set`) | Firme (especificación congelada) |
| [`piloto_dreamt.md`](piloto_dreamt.md) | **Segunda tarea: estadios de sueño (DREAMT, wearable)**. Preregistro congelado (mapa de canales, ventana de 60 s, exclusión de N3, matriz ausencia/degeneración, tabla de firma con P0, falsadores), **resultados del piloto** (4 sujetos, 960 llamadas con `qwen2.5:7b`) y **comparación con dos modelos más** (960 más cada uno, `medgemma:4b` y `llama3.1:8b`: el único invariante es `NC_absent`, la detección de ausencia como propiedad del par modelo × canal × representación, y la corrección de medición de `verify_claims`) | Ejecutado 2026-11-22/24 (n=4, dirección) |
| [`escala_dreamt.md`](escala_dreamt.md) | **Escala de DREAMT**: cohorte de 12 sujetos congelada antes de generar (regla de disponibilidad ≥ 12 ventanas puras por clase, exclusión de desarrollo/piloto, reparto uniforme sobre 50 candidatos), 10 ventanas por clase, 5.760 llamadas por modelo en las dos vistas, y reglas de lectura con IC bootstrap **sobre sujetos**. Resultados de los tres modelos: la abstención es del par (modelo, canal, vista), `NC_absent` es el único invariante y ningún brazo alcanza la referencia | Final (17.280 respuestas) |

### Resultados

| Documento | Contenido | Estado |
|---|---|---|
| [`experimentos.md`](experimentos.md) | **Resumen ejecutivo, inventario (E0–E12) y detalle de todos los experimentos** (WESAD, K-EmoCon, PhysioNet, DREAMT); punto de entrada único | Firme |
| [`conclusiones.md`](conclusiones.md) | **Síntesis entre modelos y datasets**: seis resultados transversales (ausencia ≠ inutilidad, trampa de la accuracy, claims locales, representación como modo de fallo, confianza ≠ incertidumbre, calidad específica de la tarea), tabla de replicación cruzada, amenazas a la validez y cinco afirmaciones para el paper | Firme (síntesis) |
| [`evolucion.md`](evolucion.md) | **Sinopsis de la evolución**: cómo cambió el experimento con cada resultado, reorientaciones y registro por sesión | Firme (memoria) |
| [`resultados_degradacion.md`](resultados_degradacion.md) | H1–H6 y H3: corrección, abstención, claims, certeza, indicadores de calidad (4 clases, `qwen2.5:7b`) | **Final** |
| [`resultados_jerarquia.md`](resultados_jerarquia.md) | **Diseño primario**: binario `schmidt` sin temperatura, 424 ventanas, `qwen2.5:7b` vs `qwen3:14b`; empate en limpio, abstención dependiente de la etiqueta del 14b, colapso opuesto con EDA destruida y trampa de la accuracy sobre decididas | **Final (clasificación y H1–H6)** |
| [`resultados_binario.md`](resultados_binario.md) | Matriz 2 modelos × 2 tareas × 5 condiciones (**variante `all` con temperatura**): empate en limpio, fallos opuestos bajo destrucción, trampa accuracy/balanced, H2b dependiente de modelo×tarea y H6 como sustituto de la confianza. El primario `schmidt` sin temperatura está en [`resultados_jerarquia.md`](resultados_jerarquia.md) | **Final (variante `all`)** |
| [`consistencia.md`](consistencia.md) | Compuerta de consistencia fisiológica: 0 % de falsos vetos en limpio, 100 % en datos incoherentes, robustez | Firme |
| [`deriva.md`](deriva.md) | **Deriva de la decisión bajo degradación** (capa de mecanismo): marginal frente al prior con **tasa de reclamación**, matriz pareada `C0→Cx` con descomposición error inducido/preexistente, destino por clase real, flujos de abstención, cruce con claims y **control con la referencia sobre las mismas features**. Incluye el preliminar exploratorio (4 clases) donde el LLM mueve 220/220 ventanas hacia `stress` y la referencia deriva de forma difusa | Implementado · resultados definitivos pendientes (servidor) |
| [`claims_h5.md`](claims_h5.md) | Verificador de claims y certeza lingüística: definiciones y resultados; el eje de **relevancia** está preregistrado en [`h5_relevancia.md`](h5_relevancia.md) e implementado | Firme (método) |
| [`p2_binario.md`](p2_binario.md) | Prompt binario stress/no-stress: resultados | **Exploratorio** |
| [`e12_resultados.md`](e12_resultados.md) | Representación centrada por sujeto (E12) | **Exploratorio** |
| [`p4_calibracion.md`](p4_calibracion.md) | Calibración, temperatura, conformal y rúbrica graduada | **Exploratorio** |
| [`resolubilidad.md`](resolubilidad.md) | Separabilidad de clases, confusor temporal y modelos de referencia | Firme (análisis) |

### Validación externa

| Documento | Contenido | Resultado |
|---|---|---|
| [`p3_kemocon.md`](p3_kemocon.md) | K-EmoCon: por qué no sirve para estrés; arousal/valencia no predecibles | Negativo documentado |
| [`p3_physionet.md`](p3_physionet.md) | PhysioNet/CinC 2017: colapso a `normal` (balanced 0,25 vs 0,55 de la referencia), 0 % de abstención con 103 registros `noisy`, claims 47,6 % contradichos, certeza *mayor* en ruido; la calidad sí es detectable (AUROC 0,873) | Negativo documentado (con matices de H6) |

### Literatura y material de apoyo

| Documento | Contenido |
|---|---|
| [`literatura_slip.md`](literatura_slip.md) | Análisis del corpus SLIP y 14 oportunidades experimentales (E1–E14) |
| [`analisisDatasets.md`](analisisDatasets.md) | **Compendio consolidado de datasets** (117 papers), WESAD primero, competidores por hipótesis, papel de PhysioNet/K-EmoCon y justificación de WESAD frente a alternativas |
| [`lecturas.md`](lecturas.md) | **Lecturas prioritarias** (10 papers): por qué importan y versión en español por secciones; PDF en [`lecturas.pdf`](lecturas.pdf) |
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
