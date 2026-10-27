Actúa como un metodólogo de NLP especializado en evaluación de LLMs para ACL. Necesito literatura científica que justifique el uso de salidas JSON estructuradas (en lugar de texto libre) para evaluar respuestas de LLMs en las siguientes cinco dimensiones: (1) corrección de la respuesta, (2) confianza declarada por el modelo, (3) certeza transmitida mediante marcadores epistémicos (hedges y boosters), (4) respaldo que los datos proporcionan a las afirmaciones y (5) decisión de responder o abstenerse.
Busca y sintetiza evidencia empírica que responda a estas preguntas:
Robustez y parseabilidad del formato: ¿Existe evidencia de que JSON produce menores tasas de fallo estructural y mayor parseabilidad que YAML, XML o texto libre en tareas de extracción y evaluación con LLMs? (Busca benchmarks como JSONSchemaBench o estudios comparativos de serialización en ACL).
Elicitación de confianza calibrada: ¿La confianza verbalizada como campo numérico dentro de un JSON produce estimaciones mejor calibradas (menor ECE) que la confianza inferida de logits o de texto libre? (Busca trabajos sobre verbalized confidence, calibration in RLHF-LMs y structured confidence elicitation).
Anotación de marcadores epistémicos: ¿Se ha utilizado JSON para anotar sistemáticamente hedges y boosters, permitiendo separar la certeza lingüística de la confianza numérica declarada? (Busca papers sobre epistemic markers typology, hedge/booster annotation y su relación con la precisión).
Abstención y selective prediction: ¿El uso de un campo JSON explícito de abstain o decision facilita la evaluación de curvas riesgo-cobertura y la distinción entre abstención correcta e incorrecta? (Busca selective prediction, abstention metrics, risk-coverage).
Reproducibilidad en ACL: ¿Las guías de reproducibilidad de ACL (Responsible NLP Checklist) o estándares como ReproEvalCard recomiendan o se benefician de salidas estructuradas para auditar prompts, configuraciones y trazas intermedias?
Para cada punto, proporciona:
Al menos 2 referencias concretas (autores, año, venue si está disponible).
Una frase de takeaway que explique por qué esa evidencia respalda el uso de JSON en mi diseño experimental.
Si existe algún contraargumento (p. ej., que JSON no captura la riqueza pragmática del lenguaje natural), señálalo para poder anticiparlo en la discusión.

Sí: la evidencia disponible respalda un diseño con **JSON Schema estricto** para separar campos evaluables —`correct`, `declared_confidence`, `hedges`, `boosters`, `evidence_support`, `decision`— porque mejora la fiabilidad del parsing, permite medir calibración y abstención de forma reproducible, y facilita la auditoría experimental exigida por ACL. No obstante, JSON debe complementarse, no sustituir, la cita textual y la justificación en lenguaje natural.

## 1. Robustez y parseabilidad

- **Geng et al. (2025), *JSONSchemaBench*** (arXiv/EMNLP-track-style benchmark): evalúa 10.000 esquemas JSON reales y mide eficiencia, cobertura de restricciones y calidad de salidas de seis motores de decodificación restringida, incluidos Guidance, Outlines, XGrammar, OpenAI y Gemini. [arxiv](https://arxiv.org/html/2501.10868v3)
- **NVIDIA (2025), *Structured Outputs for Nemotron***: reporta que el entrenamiento con salidas válidas en JSON, YAML y XML mejora la conformidad en JSONSchemaBench de 80.2% a 86.9%, y StructEval-Text de 64.5% a 72.1%; también muestra que los modelos pueden alcanzar 92.8–95.8% de salidas JSON válidas. [docs.nvidia](https://docs.nvidia.com/nemo/datadesigner/v0.8.0/dev-notes/structured-outputs-from-nemotron)
- **Takeaway:** JSON Schema ofrece una especificación verificable de tipos, campos obligatorios y valores permitidos; por tanto, puedes registrar como métrica primaria la tasa de `jsonschema.validate == True`, en lugar de depender de expresiones regulares frágiles sobre texto libre.
- **Matiz importante:** JSONSchemaBench evalúa conformidad estructural, no necesariamente calidad semántica; un JSON válido puede contener una confianza mal calibrada o una justificación incorrecta.  Además, SO-Bench encuentra que incluso modelos fuertes pueden seguir el esquema con >95% de precisión pero producir contenidos estructurados completamente correctos en proporciones mucho menores. [arxiv](https://arxiv.org/html/2501.10868v3)

## 2. Confianza declarada y calibración

- **Tian et al. (2023), *Just Ask for Calibration*, EMNLP**: demuestra que pedir al modelo una probabilidad verbalizada, especialmente tras generar varias opciones candidatas, mejora la calibración medida con ECE frente a alternativas menos estructuradas. [aclanthology](https://aclanthology.org/2023.emnlp-main.330.pdf)
- **Xiong et al. (2024), *Can LLMs Express Their Uncertainty?***: compara estrategias de elicitación verbalizada, muestreo y agregación en cinco tareas y cinco LLMs; reporta mejoras medias de ECE de 0.250 a 0.187 y de AUROC de 56.4% a 73.0% para la predicción de fallos. [alphaxiv](https://www.alphaxiv.org/abs/2306.13063)
- **Takeaway:** solicitar `declared_confidence` como número acotado —por ejemplo, `0.00–1.00`— dentro del JSON convierte la confianza en una variable medible y comparable entre modelos, permitiendo calcular ECE, Brier score y AUROC de error sin extraer probabilidades de texto libre.
- **Contraargumento:** la confianza verbalizada no es automáticamente mejor que los logits o métodos basados en consistencia. Un estudio reciente de interacciones multi-turno encuentra que tanto la confianza verbalizada como \(P(\text{TRUE})\) basada en logits pueden estar pobremente calibradas, con InfoECE entre 40 y 80.  Por ello, tu JSON debe incluir no solo el número, sino también la escala, la definición operativa y una validación contra precisión empírica. [web3.arxiv](https://web3.arxiv.org/pdf/2601.02179)

## 3. Hedges, boosters y certeza lingüística

- **Hyland (1998, 2005)** establece la tipología clásica: los *hedges* mitigan el compromiso epistémico con una proposición —*may*, *might*, *perhaps*, *seems*— y los *boosters* lo refuerzan —*clearly*, *certainly*, *definitely*, *must*. [jlls](https://www.jlls.org/index.php/jlls/article/viewFile/2083/714)
- **“Do large language models keep the hedge?” (2026), Frontiers in Language Sciences**: usa un léxico determinista de marcadores epistémicos validado con anotación manual y dos jueces independientes; encuentra que los LLMs tienden a “endurecer” la incertidumbre al resumir, con retención de hedges explícitos de solo alrededor de 0.08. [frontiersin](https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2026.1953328/full)
- **“Critical or Compliant?” (2025/2026, arXiv)**: evalúa outputs de LLM mediante un LLM-as-judge que identifica boosters y hedges, y sitúa estos marcadores como señales centrales de postura epistémica. [arxiv](https://arxiv.org/html/2511.12001v2)
- **Takeaway:** un campo JSON como `"epistemic_markers": {"hedges": [{"span": "...", "type": "epistemic_adverb"}], "boosters": [...]}` permite separar analíticamente la **certeza lingüística** de la **confianza numérica declarada**, y testear si un modelo dice “70% seguro” mientras usa *clearly* o *definitely*.
- **Contraargumento:** los marcadores epistémicos son contextuales y polifuncionales; un léxico fijo confunde usos epistémicos y deónticos, como *should*, *could* o *can*. En el estudio de Frontiers, los modales epistémicos alcanzaron precisión de solo 0.23 frente a 0.90 para hedges y boosters léxicos.  Conviene exigir en el JSON el *span* textual, la categoría y una justificación breve, y validar una muestra con anotadores humanos. [frontiersin](https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2026.1953328/full)

## 4. Abstención y predicción selectiva

- **Xin et al. (2021), *The Art of Abstention*, ACL**: formaliza la predicción selectiva en NLP mediante un clasificador \(h=(f,g)\) que puede abstenerse, y propone evaluarlo con la curva riesgo-cobertura y su AUC. [cs.uwaterloo](https://cs.uwaterloo.ca/~y328yu/publication/xin-tyl-21-b/xin-tyl-21-b.pdf)
- **El-Yaniv y Wiener (2010)** y **Geifman y El-Yaniv (2017)** son las referencias fundacionales citadas por la literatura posterior para riesgo-cobertura y control selectivo del riesgo. [arxiv](https://arxiv.org/html/2608.14617v1)
- **Trabajos recientes sobre abstención en LLMs** definen explícitamente cobertura como proporción de preguntas respondidas y riesgo como tasa de error entre respuestas aceptadas; algunos añaden utilidad, tasa de compromiso erróneo y comparación con abstención aleatoria. [arxiv](https://arxiv.org/html/2610.16318v1)
- **Takeaway:** un campo `"decision": "answer" | "abstain"` más `"abstention_reason"` hace auditable la política de abstención y permite construir directamente curvas de riesgo-cobertura, AURC, riesgo a cobertura fija y una matriz de abstención correcta/incorrecta.
- **Diseño recomendable:** registra cuatro celdas: respuesta correcta, respuesta incorrecta, abstención adecuada —el modelo no sabía o la evidencia era insuficiente— y abstención indebida —el modelo podía responder correctamente pero se abstuvo—. Esto va más allá del riesgo-cobertura estándar y captura la calidad pragmática de la abstención. [cs.uwaterloo](https://cs.uwaterloo.ca/~y328yu/publication/xin-tyl-21-b/xin-tyl-21-b.pdf)

## 5. Reproducibilidad en ACL

- **ARR Responsible NLP Research Checklist**: pide documentar el montaje experimental, búsqueda y valores finales de hiperparámetros, selección de modelos, preprocesamiento y configuración de paquetes empleados. [aclrollingreview](https://aclrollingreview.org/responsibleNLPresearch/)
- **NLP Reproducibility For All (2023/2024, arXiv)** recoge los elementos del ACL Reproducibility Checklist: número de ejecuciones, límites de hiperparámetros, configuraciones de los mejores modelos y criterio de selección. [arxiv](https://arxiv.org/html/2305.16579v3)
- **Takeaway:** JSON facilita la trazabilidad máquina-legible: puedes versionar el esquema, el prompt, la temperatura, el modelo, la semilla, los campos extraídos y las métricas derivadas en un único artefacto por respuesta, en lugar de reconstruirlos desde logs de texto libre.
- **Matiz:** el checklist no prescribe JSON como formato; recomienda documentación completa y reproducible.  Tu argumento debe ser, por tanto, que JSON es un **medio** para satisfacer mejor esos requisitos de auditoría, no un requisito normativo de ACL. [aclrollingreview](https://aclrollingreview.org/responsibleNLPresearch/)

## Esquema mínimo sugerido

```json
{
  "item_id": "string",
  "model_id": "string",
  "prompt_hash": "string",
  "answer": "string",
  "correct": "boolean | null",
  "declared_confidence": 0.72,
  "confidence_scale": "P(answer_is_correct)",
  "epistemic_markers": {
    "hedges": [{"span": "may", "category": "epistemic_modal"}],
    "boosters": [{"span": "clearly", "category": "epistemic_adverb"}]
  },
  "evidence_support": {
    "rating": "supported | partially_supported | unsupported",
    "cited_evidence": ["string"]
  },
  "decision": "answer | abstain",
  "abstention_reason": "string | null"
}
```

## Contraargumentos que conviene anticipar

- **Pérdida de riqueza pragmática:** JSON fuerza una taxonomía; puede omitir matices como ironía, condicionales, cita indirecta o incertidumbre distribuida entre varias afirmaciones. Solución: conserva `answer` y `rationale` como texto libre, y usa JSON para los campos codificables.
- **Validez estructural no implica validez semántica:** un objeto JSON conforme puede tener una confianza arbitraria o una etiqueta de soporte errónea.  Reporta tanto *schema validity* como métricas de contenido. [arxiv](https://arxiv.org/html/2501.10868v3)
- **La confianza numérica autoinformada puede estar mal calibrada:** no la trates como probabilidad postcalibrada sin validación empírica; evalúa ECE, Brier, AUROC y, si procede, recalibración en un conjunto held-out. [aclanthology](https://aclanthology.org/2023.emnlp-main.330.pdf)
- **Los marcadores epistémicos no son proxies directos de exactitud:** un texto muy hedgeado puede ser correcto y un texto con boosters puede ser falso; su valor está en analizar la **coherencia entre certeza lingüística, confianza declarada y desempeño real**. [jlls](https://www.jlls.org/index.php/jlls/article/viewFile/2083/714)

Sí: la evidencia disponible respalda un diseño con **JSON Schema estricto** para separar campos evaluables —`correct`, `declared_confidence`, `hedges`, `boosters`, `evidence_support`, `decision`— porque mejora la fiabilidad del parsing, permite medir calibración y abstención de forma reproducible, y facilita la auditoría experimental exigida por ACL. No obstante, JSON debe complementarse, no sustituir, la cita textual y la justificación en lenguaje natural.

## 1. Robustez y parseabilidad

- **Geng et al. (2025), *JSONSchemaBench*** (arXiv/EMNLP-track-style benchmark): evalúa 10.000 esquemas JSON reales y mide eficiencia, cobertura de restricciones y calidad de salidas de seis motores de decodificación restringida, incluidos Guidance, Outlines, XGrammar, OpenAI y Gemini. [arxiv](https://arxiv.org/html/2501.10868v3)
- **NVIDIA (2025), *Structured Outputs for Nemotron***: reporta que el entrenamiento con salidas válidas en JSON, YAML y XML mejora la conformidad en JSONSchemaBench de 80.2% a 86.9%, y StructEval-Text de 64.5% a 72.1%; también muestra que los modelos pueden alcanzar 92.8–95.8% de salidas JSON válidas. [docs.nvidia](https://docs.nvidia.com/nemo/datadesigner/v0.8.0/dev-notes/structured-outputs-from-nemotron)
- **Takeaway:** JSON Schema ofrece una especificación verificable de tipos, campos obligatorios y valores permitidos; por tanto, puedes registrar como métrica primaria la tasa de `jsonschema.validate == True`, en lugar de depender de expresiones regulares frágiles sobre texto libre.
- **Matiz importante:** JSONSchemaBench evalúa conformidad estructural, no necesariamente calidad semántica; un JSON válido puede contener una confianza mal calibrada o una justificación incorrecta.  Además, SO-Bench encuentra que incluso modelos fuertes pueden seguir el esquema con >95% de precisión pero producir contenidos estructurados completamente correctos en proporciones mucho menores. [arxiv](https://arxiv.org/html/2501.10868v3)

## 2. Confianza declarada y calibración

- **Tian et al. (2023), *Just Ask for Calibration*, EMNLP**: demuestra que pedir al modelo una probabilidad verbalizada, especialmente tras generar varias opciones candidatas, mejora la calibración medida con ECE frente a alternativas menos estructuradas. [aclanthology](https://aclanthology.org/2023.emnlp-main.330.pdf)
- **Xiong et al. (2024), *Can LLMs Express Their Uncertainty?***: compara estrategias de elicitación verbalizada, muestreo y agregación en cinco tareas y cinco LLMs; reporta mejoras medias de ECE de 0.250 a 0.187 y de AUROC de 56.4% a 73.0% para la predicción de fallos. [alphaxiv](https://www.alphaxiv.org/abs/2306.13063)
- **Takeaway:** solicitar `declared_confidence` como número acotado —por ejemplo, `0.00–1.00`— dentro del JSON convierte la confianza en una variable medible y comparable entre modelos, permitiendo calcular ECE, Brier score y AUROC de error sin extraer probabilidades de texto libre.
- **Contraargumento:** la confianza verbalizada no es automáticamente mejor que los logits o métodos basados en consistencia. Un estudio reciente de interacciones multi-turno encuentra que tanto la confianza verbalizada como \(P(\text{TRUE})\) basada en logits pueden estar pobremente calibradas, con InfoECE entre 40 y 80.  Por ello, tu JSON debe incluir no solo el número, sino también la escala, la definición operativa y una validación contra precisión empírica. [web3.arxiv](https://web3.arxiv.org/pdf/2601.02179)

## 3. Hedges, boosters y certeza lingüística

- **Hyland (1998, 2005)** establece la tipología clásica: los *hedges* mitigan el compromiso epistémico con una proposición —*may*, *might*, *perhaps*, *seems*— y los *boosters* lo refuerzan —*clearly*, *certainly*, *definitely*, *must*. [jlls](https://www.jlls.org/index.php/jlls/article/viewFile/2083/714)
- **“Do large language models keep the hedge?” (2026), Frontiers in Language Sciences**: usa un léxico determinista de marcadores epistémicos validado con anotación manual y dos jueces independientes; encuentra que los LLMs tienden a “endurecer” la incertidumbre al resumir, con retención de hedges explícitos de solo alrededor de 0.08. [frontiersin](https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2026.1953328/full)
- **“Critical or Compliant?” (2025/2026, arXiv)**: evalúa outputs de LLM mediante un LLM-as-judge que identifica boosters y hedges, y sitúa estos marcadores como señales centrales de postura epistémica. [arxiv](https://arxiv.org/html/2511.12001v2)
- **Takeaway:** un campo JSON como `"epistemic_markers": {"hedges": [{"span": "...", "type": "epistemic_adverb"}], "boosters": [...]}` permite separar analíticamente la **certeza lingüística** de la **confianza numérica declarada**, y testear si un modelo dice “70% seguro” mientras usa *clearly* o *definitely*.
- **Contraargumento:** los marcadores epistémicos son contextuales y polifuncionales; un léxico fijo confunde usos epistémicos y deónticos, como *should*, *could* o *can*. En el estudio de Frontiers, los modales epistémicos alcanzaron precisión de solo 0.23 frente a 0.90 para hedges y boosters léxicos.  Conviene exigir en el JSON el *span* textual, la categoría y una justificación breve, y validar una muestra con anotadores humanos. [frontiersin](https://www.frontiersin.org/journals/language-sciences/articles/10.3389/flang.2026.1953328/full)

## 4. Abstención y predicción selectiva

- **Xin et al. (2021), *The Art of Abstention*, ACL**: formaliza la predicción selectiva en NLP mediante un clasificador \(h=(f,g)\) que puede abstenerse, y propone evaluarlo con la curva riesgo-cobertura y su AUC. [cs.uwaterloo](https://cs.uwaterloo.ca/~y328yu/publication/xin-tyl-21-b/xin-tyl-21-b.pdf)
- **El-Yaniv y Wiener (2010)** y **Geifman y El-Yaniv (2017)** son las referencias fundacionales citadas por la literatura posterior para riesgo-cobertura y control selectivo del riesgo. [arxiv](https://arxiv.org/html/2608.14617v1)
- **Trabajos recientes sobre abstención en LLMs** definen explícitamente cobertura como proporción de preguntas respondidas y riesgo como tasa de error entre respuestas aceptadas; algunos añaden utilidad, tasa de compromiso erróneo y comparación con abstención aleatoria. [arxiv](https://arxiv.org/html/2610.16318v1)
- **Takeaway:** un campo `"decision": "answer" | "abstain"` más `"abstention_reason"` hace auditable la política de abstención y permite construir directamente curvas de riesgo-cobertura, AURC, riesgo a cobertura fija y una matriz de abstención correcta/incorrecta.
- **Diseño recomendable:** registra cuatro celdas: respuesta correcta, respuesta incorrecta, abstención adecuada —el modelo no sabía o la evidencia era insuficiente— y abstención indebida —el modelo podía responder correctamente pero se abstuvo—. Esto va más allá del riesgo-cobertura estándar y captura la calidad pragmática de la abstención. [cs.uwaterloo](https://cs.uwaterloo.ca/~y328yu/publication/xin-tyl-21-b/xin-tyl-21-b.pdf)

## 5. Reproducibilidad en ACL

- **ARR Responsible NLP Research Checklist**: pide documentar el montaje experimental, búsqueda y valores finales de hiperparámetros, selección de modelos, preprocesamiento y configuración de paquetes empleados. [aclrollingreview](https://aclrollingreview.org/responsibleNLPresearch/)
- **NLP Reproducibility For All (2023/2024, arXiv)** recoge los elementos del ACL Reproducibility Checklist: número de ejecuciones, límites de hiperparámetros, configuraciones de los mejores modelos y criterio de selección. [arxiv](https://arxiv.org/html/2305.16579v3)
- **Takeaway:** JSON facilita la trazabilidad máquina-legible: puedes versionar el esquema, el prompt, la temperatura, el modelo, la semilla, los campos extraídos y las métricas derivadas en un único artefacto por respuesta, en lugar de reconstruirlos desde logs de texto libre.
- **Matiz:** el checklist no prescribe JSON como formato; recomienda documentación completa y reproducible.  Tu argumento debe ser, por tanto, que JSON es un **medio** para satisfacer mejor esos requisitos de auditoría, no un requisito normativo de ACL. [aclrollingreview](https://aclrollingreview.org/responsibleNLPresearch/)

## Esquema mínimo sugerido

```json
{
  "item_id": "string",
  "model_id": "string",
  "prompt_hash": "string",
  "answer": "string",
  "correct": "boolean | null",
  "declared_confidence": 0.72,
  "confidence_scale": "P(answer_is_correct)",
  "epistemic_markers": {
    "hedges": [{"span": "may", "category": "epistemic_modal"}],
    "boosters": [{"span": "clearly", "category": "epistemic_adverb"}]
  },
  "evidence_support": {
    "rating": "supported | partially_supported | unsupported",
    "cited_evidence": ["string"]
  },
  "decision": "answer | abstain",
  "abstention_reason": "string | null"
}
```

## Contraargumentos que conviene anticipar

- **Pérdida de riqueza pragmática:** JSON fuerza una taxonomía; puede omitir matices como ironía, condicionales, cita indirecta o incertidumbre distribuida entre varias afirmaciones. Solución: conserva `answer` y `rationale` como texto libre, y usa JSON para los campos codificables.
- **Validez estructural no implica validez semántica:** un objeto JSON conforme puede tener una confianza arbitraria o una etiqueta de soporte errónea.  Reporta tanto *schema validity* como métricas de contenido. [arxiv](https://arxiv.org/html/2501.10868v3)
- **La confianza numérica autoinformada puede estar mal calibrada:** no la trates como probabilidad postcalibrada sin validación empírica; evalúa ECE, Brier, AUROC y, si procede, recalibración en un conjunto held-out. [aclanthology](https://aclanthology.org/2023.emnlp-main.330.pdf)
- **Los marcadores epistémicos no son proxies directos de exactitud:** un texto muy hedgeado puede ser correcto y un texto con boosters puede ser falso; su valor está en analizar la **coherencia entre certeza lingüística, confianza declarada y desempeño real**. [jlls](https://www.jlls.org/index.php/jlls/article/viewFile/2083/714)

**Es una práctica adecuada y científicamente sustentada para los aspectos cuantificables y auditables de la evaluación, pero no debe presentarse como un método autosuficiente para analizar la calidad completa de una respuesta.** La postura metodológicamente más defendible es: **JSON estructurado como instrumento de medición primario para variables operacionalizadas**, complementado con análisis cualitativo del texto libre, validación humana y auditoría de sesgos del juez.

## Qué respalda el uso de JSON

La evidencia empírica más directa proviene de la evaluación con *LLM-as-a-judge*. En BioRE, exigir que el modelo generador produzca relaciones en JSON en lugar de texto libre permitió convertir la evaluación en una comparación estructurada y mejoró la precisión del juez en aproximadamente 15 puntos; el propio artículo de ACL 2025 describe el formato JSON como parte central de su enfoque de *Structured Output Formatting*. [en.papernotes](https://en.papernotes.org/ACL2025/medical_nlp/biore_llm_judge_evaluation/)

Una revisión de *LLM-as-a-judge* concluye que estandarizar la salida —mediante JSON, puntuaciones numéricas o respuestas binarias— reduce la fragilidad de la extracción de tokens y limita la propagación de sesgos estilísticos, mejorando la interpretabilidad y validez de la evaluación.  Esto respalda JSON para campos como `correct`, `declared_confidence`, `decision`, `evidence_support` y listas de marcadores epistémicos. [aclanthology](https://aclanthology.org/2025.acl-long.1238.pdf)

También hay soporte desde la fiabilidad estructural: JSONSchemaBench evalúa aproximadamente 10.000 esquemas JSON reales y mide cobertura de restricciones, eficiencia y calidad de generación bajo decodificación restringida.  ACL 2025 incluye trabajos específicos sobre generación JSON válida frente a un esquema, reconociendo que la integración de LLMs con programas depende de esa capacidad. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC13237853/)

**Takeaway:** JSON es adecuado cuando tus cinco dimensiones se han definido previamente como constructos medibles, con reglas de codificación explícitas. Permite calcular tasas de validez de esquema, exactitud, ECE, curvas riesgo-cobertura, acuerdo entre jueces y concordancia con anotación humana.

## Por qué no es suficiente por sí solo

JSON **no convierte automáticamente una evaluación en rigurosa**. Un objeto JSON válido puede contener una etiqueta incorrecta, una confianza mal calibrada o una justificación superficial. JSONSchemaBench mide conformidad estructural y eficiencia, no la corrección semántica del contenido. 

Además, los jueces LLM presentan sesgos documentados —posición, longitud/verbosidad, atajos de razonamiento y falta de fidelidad— incluso cuando se les obliga a emitir JSON.  Por tanto, la validez del resultado depende del diseño completo: rúbrica, prompt, esquema, validación, muestreo, orden de presentación, repetición y comparación con jueces humanos. [aclanthology](https://aclanthology.org/2025.acl-long.243.pdf)

**Takeaway:** JSON mejora la **fiabilidad de medición**, pero no garantiza la **validez de constructo**. Si tu variable es “confianza transmitida”, el campo numérico y los *hedges/boosters* son operacionalizaciones parciales, no la totalidad del fenómeno pragmático.

## Postura metodológica recomendada

| Componente | Rol del JSON | Límite |
|---|---|---|
| Corrección | Campo binario, categórico o con rúbrica acotada | Requiere clave de respuesta o validación humana |
| Confianza declarada | Número normalizado y escala definida | Puede estar mal calibrado; hay que medir ECE/Brier/AUROC |
| Hedges y boosters | Spans, categorías y conteos auditables | El léxico no capta todo el uso contextual; validar muestra humana |
| Respaldo evidencial | Etiqueta + citas textuales + justificación | La etiqueta sola no prueba *grounding*; exige verificar citas |
| Abstención | `answer` / `abstain` + razón | Necesita distinguir abstención correcta, incorrecta y evasiva |

## Literatura que sostiene esta posición

- **Geng et al. (2025), *JSONSchemaBench***: justifica JSON como formato evaluable y verificable mediante esquemas, pero su evidencia es sobre validez estructural, no calidad semántica. 
- **Tian et al. (2023), EMNLP, *Just Ask for Calibration***: respalda la elicitación explícita de probabilidades verbalizadas para mejorar calibración, condición necesaria para usar `declared_confidence` como medida científica. 
- **Xiong et al. (2024), *Can LLMs Express Their Uncertainty?***: muestra que la confianza verbalizada puede mejorar ECE y AUROC, pero también que su calidad depende de la estrategia de elicitación. 
- **Hyland (1998, 2005)** y **“Do large language models keep the hedge?” (2026), Frontiers in Language Sciences**: fundamentan la anotación de hedges y boosters, pero muestran que los marcadores son contextuales y que los LLMs pueden alterar o perder la incertidumbre original. 
- **Xin et al. (2021), ACL, *The Art of Abstention***: proporciona el marco formal de predicción selectiva, riesgo-cobertura y AUC de riesgo-cobertura para evaluar abstención. 
- **ARR Responsible NLP Research Checklist**: exige documentar configuración experimental, hiperparámetros y reproducibilidad; JSON es un mecanismo útil para satisfacer esa trazabilidad, aunque no un requisito prescriptivo. 
- **Survey de LLM-as-a-judge (2026)**: respalda la estandarización estructurada como buena práctica para reducir fragilidad y sesgos de extracción. [aclanthology](https://aclanthology.org/2025.acl-long.1238.pdf)
- **Estudios de sesgos en LLM-as-a-judge**: obligan a tratar JSON como señal medible sujeta a validación, no como verdad automática. [aclanthology](https://aclanthology.org/2025.acl-long.243.pdf)

## Formulación para tu artículo

Podrías defenderlo así:

> “We adopt schema-constrained JSON outputs as the primary measurement layer because they render each evaluation dimension explicitly operationalizable, machine-checkable, and reproducible. However, we do not treat JSON conformity or self-reported fields as sufficient evidence of response quality. Following selective-prediction and LLM-as-a-judge evaluation practice, structured fields are complemented by free-text rationales, source quotations, human validation of a stratified sample, and bias/reliability audits of the judge.”

En términos estrictos: **JSON es una buena práctica metodológica, no una moda ni un mero formato alternativo**, siempre que se use como parte de un protocolo de medición validado. Pero afirmar que “permite un análisis riguroso completo” sería sobredimensionarlo: permite un análisis **riguroso y reproducible de dimensiones operacionalizadas**, y debe complementarse con evidencia cualitativa y validación humana para captar la riqueza pragmática del lenguaje natural. [en.papernotes](https://en.papernotes.org/ACL2025/medical_nlp/biore_llm_judge_evaluation/)

Con tus hipótesis, la recomendación es más específica: utilizaría JSON como capa de registro y anotación, pero no obligaría a que toda la respuesta del LLM adopte un esquema cerrado. Tu diseño necesita conservar texto explicativo para estudiar la certeza lingüística y las afirmaciones que el modelo introduce; al mismo tiempo, necesita decisiones y probabilidades extraíbles para evaluar corrección, confianza y abstención. Esta combinación encaja mejor con tus RQ2 y RQ3 que una salida formada únicamente por etiquetas y números. 

La distinción central es la **reactividad**: una cosa es guardar y analizar una respuesta en JSON y otra pedir al modelo que genere bajo restricciones JSON. La primera organiza la medición; la segunda también interviene sobre el comportamiento que estás estudiando. Existe evidencia de que las restricciones de formato pueden modificar el rendimiento, con efectos distintos según la tarea y la implementación. [arxiv](https://arxiv.org/html/2408.02442)

## 1. Primero, una precisión respecto a mis respuestas anteriores

Debo matizar dos afirmaciones:

- La literatura sobre confianza verbalizada respalda solicitar probabilidades explícitas, pero no demuestra, por sí misma, que escribir esa probabilidad dentro de JSON mejore la calibración. Tian et al. comparan estrategias de elicitación y probabilidades del modelo; no aíslan un efecto causal de la serialización JSON. [arxiv](https://arxiv.org/abs/2305.14975)
- Un benchmark de conformidad con JSON Schema respalda la evaluación de la estructura, pero no demuestra que JSON sea universalmente superior al texto libre, YAML o XML para medir tus cinco dimensiones.

Por tanto, no defendería tu diseño diciendo «JSON mejora la validez científica de las respuestas». Lo defendería diciendo: «El registro estructurado hace explícitas y auditables las variables; su validez se establece mediante referencias externas y validación de los instrumentos».

## 2. Pertinencia de JSON para cada hipótesis

La siguiente tabla es una propuesta metodológica aplicada a las operacionalizaciones de tu documento.

| Hipótesis | Utilidad del JSON | Qué no debe sustituir |
|---|---|---|
| H1: corrección bajo degradación | Registrar predicción, condición, referencia y resultado de evaluación. | La comparación con las etiquetas de referencia. El campo `correct` no debería rellenarlo el modelo evaluado. |
| H2a: confianza numérica | Extraer una probabilidad con escala y objeto claramente definidos. | La comprobación empírica de calibración y discriminación. |
| H2b: certeza lingüística | Guardar anotaciones posteriores: spans, categorías, alcance y recuentos. | El texto original en el que aparecen —o no aparecen— hedges y boosters. |
| H3: efecto de representación | Mantener constante el contrato de salida entre representaciones. | El control del formato de salida como posible factor experimental. |
| H4: abstención | Registrar una decisión inequívoca y distinguirla de errores de generación o parsing. | El criterio externo de adecuación de la abstención y la evaluación riesgo–cobertura. |
| H5: respaldo de afirmaciones | Representar afirmaciones extraídas, operaciones, tolerancias y veredictos del verificador. | La extracción exhaustiva desde el texto y la verificación independiente. |
| H6 y HE | Unificar resultados, identificadores y trazabilidad. | Los análisis predictivos o de representaciones internas correspondientes. |

### H2b es donde más debes evitar una salida excesivamente cerrada

Tu hipótesis no pregunta si el modelo sabe seleccionar una etiqueta como `"certainty": "low"`. Pregunta cómo expresa certeza en el lenguaje y si esa expresión se desacopla de la corrección.

Por ello, no usaría como medición principal:

```json
{
  "certainty": "low",
  "hedges": ["posiblemente"],
  "boosters": []
}
```

Si lo produce el modelo evaluado, estarías midiendo su autodescripción metalingüística, no necesariamente los recursos lingüísticos que realmente utiliza.

Preferiría conservar su explicación y anotar después los marcadores presentes, incluyendo posiciones y alcance. Si el modelo escribe «posiblemente hay artefactos, pero el estado es claramente de estrés», no basta con contar un hedge y un booster: interesa saber qué proposición modifica cada uno. Esta exigencia es coherente con tu propia advertencia sobre la interpretación contextual de los marcadores. 

### H5 requiere separar generación, extracción y verificación

Tu propuesta de salida con «métrica, ventana, dirección y magnitud» es útil, pero necesita especificar quién produce esa estructura. 

Hay dos diseños diferentes:

1. El generador recibe la instrucción de emitir únicamente afirmaciones estructuradas.
2. El generador produce una explicación y un procedimiento posterior extrae sus afirmaciones.

El primero estudia generación restringida; el segundo permite estudiar qué afirma el modelo en una explicación menos restringida. Para tu pregunta sobre respaldo del lenguaje generado, daría prioridad al segundo.

Un precedente especialmente pertinente es FActScore, de Min et al. (2023, EMNLP): descompone generaciones en hechos atómicos y evalúa qué proporción está respaldada por una fuente fiable. Lo transferible a tu trabajo es la separación entre generación, descomposición y verificación, no la obligación de que el generador entregue JSON. Tu verificación numérica necesitaría, además, operaciones y tolerancias propias. [aclanthology](https://aclanthology.org/2023.emnlp-main.741/)

## 3. Diseño recomendado: respuesta textual y registro estructurado separados

Propongo distinguir tres capas.

### Capa A. Salida original del modelo

Conservar exactamente la respuesta recibida, junto con prompt, configuración y metadatos.

La respuesta puede contener una explicación textual y, según la condición experimental, una decisión y una probabilidad explícitas. No solicitaría al modelo que determine si su propia respuesta es correcta ni que certifique el respaldo de todas sus afirmaciones.

### Capa B. Extracción y anotación

Transformar la salida original en variables analizables:

- Predicción emitida.
- Decisión de responder o abstenerse.
- Confianza declarada.
- Marcadores epistémicos con spans y alcance.
- Afirmaciones atómicas extraídas.

Aquí JSON resulta particularmente útil como representación común entre anotadores, extractores y verificadores.

### Capa C. Evaluación externa

Añadir:

- Corrección frente a la referencia.
- Verificación numérica.
- Categoría de respaldo.
- Concordancia con anotación humana.
- Errores de extracción o formato.

Esta separación evita confundir autoinformes del modelo con resultados de evaluación. También se ajusta a tu OE5: validar los instrumentos de medida, en vez de asumir que sus etiquetas son correctas.

## 4. Alternativas al uso de JSON

No todas las alternativas son meros cambios de sintaxis: algunas cambian el procedimiento de medición.

| Alternativa | Aplicación en tu estudio | Principal limitación |
|---|---|---|
| Texto libre + anotación humana | Medir certeza lingüística y respaldo en una muestra de referencia. | Coste y necesidad de reglas de anotación y adjudicación. |
| Texto libre + extracción automática posterior | Analizar explicaciones sin exigir que el generador enumere sus propias afirmaciones. | El extractor puede omitir, dividir o reformular incorrectamente afirmaciones. |
| Formato híbrido con secciones delimitadas | Separar explicación, decisión y confianza sin exigir sintaxis JSON completa. | Necesita reglas de parsing y validación; tampoco es una condición completamente libre de intervención. |
| Etiquetas de conjunto cerrado | Clasificación y decisión `RESPONDER`/`ABSTENERSE`. | Insuficiente por sí sola para H2b y H5. |
| YAML o XML | Serializar campos y anotaciones con otra sintaxis. | Cambiar de serialización no elimina la posible influencia de las restricciones de formato. |
| Tablas CSV/TSV y anotaciones de spans | Guardar métricas por respuesta y anotaciones lingüísticas por separado. | Las relaciones anidadas requieren varias tablas e identificadores. |

La anotación humana y la extracción posterior pueden producir análisis rigurosos; no son alternativas «menos científicas» por no exigir JSON al generador. FActScore constituye precisamente un precedente de evaluación de texto mediante descomposición y comprobación posterior. [aclanthology](https://aclanthology.org/2023.emnlp-main.741/)

### Una alternativa híbrida concreta

Podrías solicitar:

```text
EXPLICACIÓN:
[Explicación en lenguaje natural]

DECISIÓN:
RESPONDER o ABSTENERSE

PREDICCIÓN:
ESTRÉS, REPOSO o NO_EMITIDA

CONFIANZA:
[Probabilidad de que la predicción emitida sea correcta]
```

Esto evita parte de la complejidad sintáctica de JSON, pero no elimina la intervención del prompt. Si solicitas una probabilidad y una decisión, estudias lenguaje generado bajo ese contrato de respuesta, no lenguaje espontáneo sin instrucciones.

Tian et al. (2023, EMNLP), *Just Ask for Calibration*, respalda la viabilidad de obtener probabilidades explícitas de modelos ajustados con feedback humano. No obliga a usar una serialización particular. [arxiv](https://arxiv.org/abs/2305.14975)

## 5. El control de formato que incorporaría al piloto

Tu documento ya contempla sensibilidad a prompts, temperatura y repeticiones. Añadiría un control pequeño de formato de salida, especialmente para H2b y H5. 

Compararía, sobre un subconjunto emparejado:

- Condición A: explicación textual con secciones mínimas.
- Condición B: JSON con los mismos campos y una explicación textual.
- Condición C, si el presupuesto lo permite: explicación sin solicitud de confianza, seguida de una segunda llamada que elicite la probabilidad manteniendo fija la respuesta original.

La tercera condición permite examinar si pedir confianza modifica el texto inicial, aunque introduce otra limitación: la probabilidad obtenida en una segunda llamada no equivale necesariamente a la confianza elicitada durante la generación.

Mantendría constante el contenido semántico de las instrucciones entre A y B, para no confundir sintaxis con diferencias de tarea.

Compararía:

- Exactitud y abstención.
- Longitud de la explicación.
- Densidad y distribución de hedges y boosters.
- Número y tipo de afirmaciones.
- Proporción de afirmaciones respaldadas.
- Fallos de extracción y de estructura.

La justificación empírica del control es Tam et al. (2024), *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models*: encuentran pérdidas de rendimiento en determinadas tareas de razonamiento bajo restricciones, pero resultados competitivos o favorables en algunas tareas de clasificación. Por tanto, el efecto no debe suponerse ni siempre perjudicial ni siempre inocuo. [arxiv](https://arxiv.org/html/2408.02442)

Si detectas diferencias relevantes, no significa que debas abandonar JSON. Significa que debes delimitar la conclusión: «sensibilidad a la evidencia bajo generación estructurada» o analizar el formato como moderador.

## 6. Dos precauciones adicionales para tus hipótesis

### No equiparar fallo de JSON con abstención

Preespecificaría estados diferentes:

- Respuesta emitida y parseable.
- Abstención explícita.
- Respuesta semánticamente interpretable, pero con formato inválido.
- Salida vacía, truncada o no interpretable.

Así evitas que la tasa de abstención absorba errores de infraestructura o cumplimiento del esquema.

### No etiquetar automáticamente una abstención individual como correcta por la condición experimental

Tu resolubilidad se define a nivel de condición y depende del modelo de referencia. Eso no establece necesariamente si una ventana concreta era respondible. El documento reconoce la dependencia respecto a la tarea y al modelo, pero H4 necesita mantener también esta distinción entre niveles. 

Por ello, presentaría «abstención en condiciones de baja resolubilidad estimada» como resultado principal. Reservaría «abstención individual correcta/incorrecta» para casos con un criterio adicional por ejemplo. JSON puede registrar ambas cosas, pero no resolver esa cuestión de validez.

## 7. Cómo lo escribiría en la metodología

> Se empleará JSON como formato de registro y anotación de las variables de evaluación, manteniendo las respuestas originales sin modificar. Se distinguirán los campos declarados por el modelo de las etiquetas obtenidas mediante referencias externas, verificación computacional y anotación humana. Las afirmaciones y los marcadores epistémicos se extraerán del texto generado, en lugar de asumir que la autoanotación del modelo constituye una medición válida. El piloto examinará si imponer un formato estructurado modifica la corrección, la abstención, la expresión de certeza o la composición de las afirmaciones.

En tu investigación, por tanto, JSON no debería ser ni el argumento de rigor ni un obstáculo para él: es una herramienta de instrumentación. La decisión sustantiva es conservar el lenguaje que quieres estudiar y validar cómo lo conviertes en variables. Para H1, H2a y H4, un contrato estructurado es especialmente práctico; para H2b y H5, priorizaría texto explicativo conservado y anotación estructurada posterior. 

# JSON **mejora la evaluabilidad** de respuestas LLM, pero **no resuelve** por sí solo calibración ni pragmática.

La evidencia se organiza en **robustez estructural**, **confianza y marcadores epistémicos**, y **abstención y reproducibilidad**. En conjunto, respalda usar JSON como interfaz de evaluación porque separa campos, reduce ambigüedad de parsing y facilita métricas por dimensión, aunque varios trabajos advierten que la validez semántica, la calibración y la riqueza pragmática siguen siendo problemas distintos del formato.

## Robustez Estructural

La justificación más fuerte para JSON es **parseabilidad y validación automática**. Los benchmarks recientes de salidas estructuradas se han estandarizado alrededor de JSON Schema, y evalúan explícitamente cumplimiento estructural, cobertura de restricciones y calidad end-to-end (Geng et al., 2025). JSONSchemaBench muestra además que el decoding restringido puede acelerar la generación en **50%** frente a decoding no restringido, y que el mejor framework soporta **el doble** de esquemas que el peor, lo que convierte la validez estructural en un eje medible y reproducible (Geng et al., 2025).

LLMStructBench y ExtractBench muestran por qué esto importa para evaluación: conviene medir por separado **JSON válido** y **corrección semántica**, porque un output puede ser parseable pero incorrecto, o correcto localmente pero inutilizable end-to-end (Koddenbrock & Rodner, 2026; Ferguson et al., 2026). En ExtractBench, solo **51%** de las extracciones fueron JSON válido, mientras el pass rate agregado fue **4.6%**; entre las salidas válidas, la exactitud de campos subió a **72.9%**, mostrando que la separación entre fallo estructural y fallo semántico no es opcional (Ferguson et al., 2026).

| Referencias | Hallazgo útil | Takeaway para tu diseño |
|---|---|---|
| Geng et al., 2025, JSONSchemaBench; Lu et al., 2025, Schema RL | JSON Schema ya es el estándar práctico de constrained decoding, y prompting simple falla con esquemas complejos (Geng et al., 2025; Lu et al., 2025) | Usa **JSON con schema** para aislar errores de estructura de errores de contenido. |
| Koddenbrock & Rodner, 2026, LLMStructBench; Ferguson et al., 2026, ExtractBench | Deben reportarse métricas separadas de validez documental y exactitud semántica (Koddenbrock & Rodner, 2026; Ferguson et al., 2026) | JSON permite una evaluación **bifásica**: parseo primero, corrección después. |
| Liu et al., 2024, SoEval; Geng et al., 2025 | Los LLMs aún tienen déficits notables en salidas estructuradas y el rendimiento varía por tipo de estructura (Liu et al., 2024; Geng et al., 2025) | JSON no elimina errores, pero los vuelve **auditablemente visibles**. |

Contraargumento: este corpus **no demuestra directamente** que JSON supere a YAML o XML en tasas de fallo cara a cara; más bien muestra que JSON es el formato dominante y mejor instrumentado para benchmark, validación y constrained decoding (Geng et al., 2025; Liu et al., 2024).

## Confianza y Marcadores

Para **confianza declarada**, la evidencia favorece pedir una probabilidad explícita como salida textual estructurada frente a depender solo de logits, sobre todo en modelos RLHF. Tian et al. encuentran que en ChatGPT, GPT-4 y Claude las confianzas verbalizadas suelen estar **mejor calibradas** que las probabilidades condicionales del modelo, con reducciones relativas de **ECE del 50%** en TriviaQA, SciQ y TruthfulQA (Tian et al., 2023). Xiong et al. confirman que estas verbalizaciones siguen siendo sobreconfiadas, pero son una vía negra-caja práctica cuando los logits no están disponibles (Xiong et al., 2023). ConfidenceBench refuerza que la calibración verbalizada es un eje distinto de la exactitud y evaluable sin acceso a logits (Ffrench-Constant et al., 2026).

Para **hedges y boosters**, varios trabajos justifican separarlos de la confianza numérica en campos distintos. Los modelos son sobreconfiados lingüísticamente y generan strengtheners incluso en respuestas incorrectas, además con fuerte variación entre lenguas (Rathi et al., 2025). La tipología de marcadores epistémicos cambia mucho la precisión y puede moverla en más de **80%**, lo que demuestra que la certeza lingüística no es intercambiable con una probabilidad numérica (Zhou et al., 2023). Estudios recientes de marker confidence muestran además que los marcadores son inestables fuera de distribución y que los LLMs fallan al alinear consistentemente hedge/booster con confianza real (Liu et al., 2025; Liu & Cohan, 2026).

- **Referencias**: Tian et al., 2023, arXiv; Xiong et al., 2023, arXiv.  
  **Takeaway**: un campo JSON `confidence: 0.xx` permite medir calibración con ECE/Brier en sistemas cerrados (Tian et al., 2023; Xiong et al., 2023).
- **Referencias**: Zhou et al., 2023; Rathi et al., 2025, arXiv.  
  **Takeaway**: un campo separado para `epistemic_markers` evita confundir **estilo asertivo** con confianza calibrada (Zhou et al., 2023; Rathi et al., 2025).
- **Referencias**: Liu & Cohan, 2026, arXiv; Liu et al., 2025, arXiv.  
  **Takeaway**: JSON facilita anotar hedges/boosters como objetos o spans independientes porque su relación con la confianza real es **inestable** y debe auditarse aparte (Liu & Cohan, 2026; Liu et al., 2025).

Contraargumento: verbalized confidence también es **protocol-sensitive**; el formato común no garantiza que la cifra mida una magnitud estable, porque la comparación con logits cambia según contexto, answer slot y readout (Kim & Kang, 2026).

## Abstención y Reproducibilidad

Para **selective prediction**, un campo JSON explícito como `decision: answer|abstain` o `abstain: true|false` facilita directamente curvas riesgo-cobertura, AURC/AUARC y el conteo de abstenciones correctas e incorrectas. La literatura de abstención define el problema precisamente como una decisión binaria de confiar o no en la salida bajo incertidumbre (Bai & Jin, 2026). En LLMs, métodos recientes optimizan AUARC o AURC y muestran mejoras sustantivas en generación selectiva y detección de alucinación, lo que encaja de forma natural con un campo estructurado de decisión (Tayebati et al., 2025; Luo et al., 2026). Los trabajos conformales también distinguen explícitamente cobertura, riesgo y garantías por grupo o bajo shift, algo difícil de recuperar de texto libre poshoc (Salem et al., 2026; Wang et al., 2025).

En **reproducibilidad ACL**, el corpus proporcionado no contiene una guía ACL o ReproEvalCard que recomiende explícitamente JSON. Aun así, varios benchmarks muestran un beneficio metodológico claro de salidas estructuradas: hacer que el schema sea una **especificación ejecutable**, fijar métricas por campo y registrar configuraciones comparables entre modelos y prompts (Ferguson et al., 2026; Koddenbrock & Rodner, 2026). Eso no sustituye un checklist de reproducibilidad, pero sí fortalece la auditabilidad de prompts, trazas intermedias y scoring.

- **Referencias**: Tayebati et al., 2025, arXiv; Luo et al., 2026, arXiv.  
  **Takeaway**: un campo JSON de decisión convierte la abstención en una variable **nativamente evaluable** con AUARC/AURC (Tayebati et al., 2025; Luo et al., 2026).
- **Referencias**: Bai & Jin, 2026, arXiv; Salem et al., 2026, arXiv.  
  **Takeaway**: JSON facilita auditar qué casos fueron “trusted” y bajo qué umbral o grupo, crucial para risk control (Bai & Jin, 2026; Salem et al., 2026).
- **Referencias**: Ferguson et al., 2026, arXiv; Koddenbrock & Rodner, 2026, arXiv.  
  **Takeaway**: aunque no sea una recomendación ACL explícita, el uso de schemas y métricas por campo mejora la **reproducibilidad operacional** (Ferguson et al., 2026; Koddenbrock & Rodner, 2026).

En síntesis, la mejor defensa metodológica de JSON no es que vuelva “mejor” al modelo, sino que vuelve **separables, parseables y auditables** las cinco dimensiones que quieres medir. La principal objeción es válida: JSON captura mal la riqueza pragmática y la variación contextual de la incertidumbre lingüística, así que conviene usarlo como **capa de instrumentación**, no como sustituto del análisis textual completo (Liu et al., 2025; Tao et al., 2025).
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Bai, T., & Jin, Y. (2026). Conformal Selective Prediction with General Risk Control. *ArXiv, abs/2603.24704*. https://doi.org/10.48550/arxiv.2603.24704
 
Ferguson, N., Pennington, J., Beghian, N., Mohan, A., Kiela, D., Agrawal, S., & Nguyen, T. (2026). ExtractBench: A Benchmark and Evaluation Methodology for Complex Structured Extraction. *ArXiv, abs/2602.12247*. https://doi.org/10.48550/arxiv.2602.12247
 
Ffrench-Constant, M., Yang, D., Huang, X., & Kapoor, S. (2026). ConfidenceBench: Evaluating Confidence Calibration in Large Language Models. *ArXiv, abs/2607.20526*. https://doi.org/10.48550/arxiv.2607.20526
 
Geng, S.-B., Cooper, H., Moskal, M., Jenkins, S., Berman, J., Ranchin, N., West, R., Horvitz, E., & Nori, H. (2025). JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models. https://doi.org/10.48550/arxiv.2501.10868
 
Kim, H., & Kang, P. (2026). Same Answer, Different Confidence: Protocol Sensitivity in LLM Confidence Calibration. https://doi.org/10.48550/arxiv.2605.27752
 
Koddenbrock, M., & Rodner, E. (2026). LLMStructBench: Benchmarking Large Language Model Structured Data Extraction. *ArXiv, abs/2602.14743*. https://doi.org/10.48550/arxiv.2602.14743
 
Liu, Y., Li, D., Wang, K., Xiong, Z., Shi, F., Wang, J., Li, B., & Hang, B. (2024). Are LLMs good at structured outputs? A benchmark for evaluating structured output capabilities in LLMs. *Inf. Process. Manag., 61*, 103809. https://doi.org/10.1016/j.ipm.2024.103809
 
Liu, J., Zong, Q., Wang, W., & Song, Y.-Q. (2025). Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?. *ArXiv, abs/2505.24778*. https://doi.org/10.48550/arxiv.2505.24778
 
Liu, G. K., & Cohan, A. (2026). Can LLMs Use Linguistic Uncertainty Markers to Reliably Reflect Intrinsic Confidence?. *ArXiv, abs/2605.28778*. https://doi.org/10.48550/arxiv.2605.28778
 
Lu, Y., Li, H., Cong, X., Zhang, Z., Wu, Y.-S., Lin, Y.-K., Liu, Z., Liu, F., & Sun, M. (2025). Learning to Generate Structured Output with Schema Reinforcement Learning. 4905-4918. https://doi.org/10.48550/arxiv.2502.18878
 
Luo, G., Wu, Y.-F., Zhang, S., Deshwal, A., & Sun, J. (2026). Aligning Language Models with Selective Prediction. *ArXiv, abs/2607.03528*. https://doi.org/10.48550/arxiv.2607.03528
 
Rathi, N., Jurafsky, D., & Zhou, K. (2025). Humans overrely on overconfident language models, across languages. *ArXiv, abs/2507.06306*. https://doi.org/10.48550/arxiv.2507.06306
 
Salem, M., Böhm, L., Pontes, D., & Ferrugem, A. (2026). Hierarchical Group-Conditional Conformal Risk Control for Selective Prediction in Language Models. *ArXiv, abs/2607.24562*. https://doi.org/10.48550/arxiv.2607.24562
 
Tao, L.-W., Yeh, Y.-F., Kai, B., Dong, M.-J., Huang, T., Lamb, T. A., Yu, J.-L., Torr, P., & Xu, C. (2025). Can Large Language Models Express Uncertainty Like Human?. *ArXiv, abs/2509.24202*. https://doi.org/10.48550/arxiv.2509.24202
 
Tayebati, S., Kumar, D., Darabi, N., Jayasuriya, D., Krishnan, R., & Trivedi, A. (2025). Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models. *ArXiv, abs/2502.06884*. https://doi.org/10.48550/arxiv.2502.06884
 
Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C. D. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *ArXiv, abs/2305.14975*. https://doi.org/10.48550/arxiv.2305.14975
 
Wang, Q.-N., Fan, Y., & Wang, X. (2025). SAFER: Risk-Constrained Sample-then-Filter in Large Language Models. *ArXiv, abs/2510.10193*. https://doi.org/10.48550/arxiv.2510.10193
 
Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J.-X., & Hooi, B. (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ArXiv, abs/2306.13063*. https://doi.org/10.48550/arxiv.2306.13063
 
Zhou, K., Jurafsky, D., & Hashimoto, T. (2023). Navigating the Grey Area: How Expressions of Uncertainty and Overconfidence Affect Language Models. 5506-5524. https://doi.org/10.18653/v1/2023.emnlp-main.335
 
