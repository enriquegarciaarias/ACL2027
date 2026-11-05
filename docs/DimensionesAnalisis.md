# Las **cinco dimensiones clave** interactúan fuertemente al evaluar respuestas de LLMs.

 el proyecto analiza de forma conjunta cinco propiedades críticas del comportamiento de los sistemas: la corrección de la respuesta, la confianza declarada por el modelo, la certeza transmitida por el lenguaje mediante marcadores epistémicos (hedges y boosters), el respaldo que los datos proporcionan a las afirmaciones y la decisión de responder o abstenerse.
El análisis del comportamiento de los LLMs requiere examinar tres facetas interconectadas: **la relación entre corrección y confianza declarada**, **la función de los marcadores epistémicos frente al respaldo empírico**, y **los mecanismos que gobiernan responder o abstenerse**.

## Corrección y Confianza Declarada

La confianza expresada numéricamente por los LLMs suele estar desalineada de la exactitud real de la respuesta. Los modelos tienden al exceso de confianza sistemático, asignando probabilidades altas incluso a afirmaciones incorrectas o alucinadas (Shorinwa et al., 2024; Vanhoyweghen et al., 2025). Además, la autoevaluación verbal no discrimina adecuadamente entre errores y aciertos en tareas complejas (Omar et al., 2025).

- La correlación entre la precisión global y la confianza media puede resultar **paradójicamente inversa** en dominios clínicos especializados (Omar et al., 2025).
- La confianza verbalizada predice el **compromiso con la decisión** mejor que la corrección objetiva subyacente de la salida (Kumaran, 2026).
- Los modelos ajustados no adaptan de forma consistente sus juicios de confianza tras cometer **errores en desempeño previo** (Cash et al., 2025).

## Marcadores Epistémicos y Respaldo

Los marcadores lingüísticos como atenuadores (*hedges*) y reforzadores (*boosters*) modifican la percepción del usuario, pero rara vez reflejan fielmente la incertidumbre interna del modelo. Los evaluadores automatizados y los usuarios humanos frecuentemente confunden la firmeza léxica con una verificación factual sólida (Kwon, 2026; Lee et al., 2024).

| Dimensión de Marcador | Comportamiento del Sistema | Impacto en la Evaluación |
| :--- | :--- | :--- |
| **Atenuadores (*hedges*)** | Reducen la decisividad percibida pero decaen fuertemente fuera de distribución (Liu et al., 2025; Yona et al., 2024). | Generan penalizaciones y sesgo negativo en jueces LLM automáticos (Lee et al., 2024). |
| **Reforzadores (*boosters*)** | Aparecen con frecuencia en afirmaciones no respaldadas o alucinaciones (Qizi et al., 2026; Shorinwa et al., 2024). | Inducen confianza injustificada y sobreconfianza en usuarios finales (Heersmink et al., 2024). |
| **Evidencialidad y fuentes** | Las fuentes clicables mitigan la confianza excesiva ante errores patentes (Kim et al., 2025). | La presencia de explicaciones incrementa la aceptación incluso sin respaldo (Kim et al., 2025). |
| **Consolidación en memoria** | Las afirmaciones tentativas previas se reescriben internamente como hechos definitivos (Kwon, 2026). | Convierten conjeturas no respaldadas en directrices aceptadas sin filtro (Kwon, 2026). |

**Figure 1:** Comparación del impacto de marcadores lingüísticos y fuentes en la credibilidad de salidas de LLM.

## Responder o Abstenerse

La decisión de abstenerse permite mitigar riesgos en dominios críticos, pero la mayoría de los modelos prefieren conjeturar antes que rechazar una consulta (Presacan et al., 2026; Wu et al., 2025). Convertir la incertidumbre interna en una abstención oportuna exige intervenciones algorítmicas o esquemas de penalización explícitos (Presacan et al., 2026; Zong et al., 2026; Wu et al., 2025).

- Los modelos estándar casi nunca se abstienen voluntariamente, sufriendo un **colapso de utilidad** ante penalizaciones por fallo (Wang et al., 2026).
- La incertidumbre estadística basada en entropía permite optimizar **decisiones de abstención selectiva** frente a entradas inseguras (Wen et al., 2024; Tomani et al., 2024; Abbasi-Yadkori et al., 2024).
- Los marcos de andamiaje pedagógico y metacognitivo elevan las **negativas apropiadas ante desconocimiento** factual (Qizi et al., 2026; Zong et al., 2026).

Integrar estas cinco dimensiones permite desacoplar la elocuencia superficial del modelo de su verdadera solidez epistemológica y su fiabilidad operativa.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Q., Chen, S., Zhang, Z., Li, J., & Zhang, Y. (2026). Knowing When to Answer: Prompting Epistemically Responsible Behaviors in Large Language Models. *IEEE Access, 14*, 95598-95610. https://doi.org/10.1109/access.2026.3705963
 
Abbasi-Yadkori, Y., Kuzborskij, I., Stutz, D., György, A., Fisch, A., Doucet, A., Beloshapka, I., Weng, W.-H., Yang, Y., Szepesvári, C., Cemgil, A. T., & Tomasev, N. (2024). Mitigating LLM Hallucinations via Conformal Abstention. *ArXiv, abs/2405.01563*. https://doi.org/10.48550/arxiv.2405.01563
 
Cash, T. N., Oppenheimer, D. M., Christie, S., & Devgan, M. (2025). Quantifying uncert-AI-nty: Testing the accuracy of LLMs’ confidence judgments. *Memory & Cognition, 54*, 375 - 400. https://doi.org/10.3758/s13421-025-01755-4
 
Heersmink, R., De Rooij, B., & Colombo, M. (2024). A phenomenology and epistemology of large language models: transparency, trust, and trustworthiness. *Ethics and Information Technology, 26*. https://doi.org/10.1007/s10676-024-09777-3
 
Kim, S. S. Y., Vaughan, J. W., Liao, Q., Lombrozo, T., & Russakovsky, O. (2025). Fostering Appropriate Reliance on Large Language Models: The Role of Explanations, Sources, and Inconsistencies. *Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems*. https://doi.org/10.1145/3706598.3714020
 
Kumaran, D. (2026). Reported Confidence in LLMs Tracks Commitment More Than Correctness. *ArXiv, abs/2606.29490*. https://doi.org/10.48550/arxiv.2606.29490
 
Kwon, A. (2026). Manufactured Confidence: How Memory Consolidation Turns Hearsay into Confident Facts. *ArXiv, abs/2606.29279*. https://doi.org/10.48550/arxiv.2606.29279
 
Lee, D., Hwang, Y., Kim, Y.-M., Park, J., & Jung, K. (2024). Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the effect of Epistemic Markers on LLM-based Evaluation. 8962-8984. https://doi.org/10.48550/arxiv.2410.20774
 
Liu, J., Zong, Q., Wang, W., & Song, Y.-Q. (2025). Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?. *ArXiv, abs/2505.24778*. https://doi.org/10.48550/arxiv.2505.24778
 
Omar, M., Agbareia, R., Glicksberg, B. S., Nadkarni, G., & Klang, E. (2025). Benchmarking the Confidence of Large Language Models in Answering Clinical Questions: Cross-Sectional Evaluation Study. *JMIR Medical Informatics, 13*. https://doi.org/10.2196/66917
 
Presacan, O., Nik, A., Ojha, J., Thambawita, V., Ionescu, B., & Riegler, M. A. (2026). When silence is safer: a review and decision-theoretic framework for LLM abstention in healthcare. *NPJ Digital Medicine, 9*. https://doi.org/10.1038/s41746-026-02882-1
 
Shorinwa, O., Mei, Z.-T., Lidard, J., Ren, A. Z., & Majumdar, A. (2024). A Survey on Uncertainty Quantification of Large Language Models: Taxonomy, Open Research Challenges, and Future Directions. *ACM Computing Surveys, 58*, 1 - 38. https://doi.org/10.1145/3744238
 
Tomani, C., Chaudhuri, K., Evtimov, I., Cremers, D., & Ibrahim, M. (2024). Uncertainty-Based Abstention in LLMs Improves Safety and Reduces Hallucinations. *ArXiv, abs/2404.10960*. https://doi.org/10.48550/arxiv.2404.10960
 
Vanhoyweghen, A., Verbeken, B., Algaba, A., & Ginis, V. (2025). Lexical hints of accuracy in LLM reasoning chains. *Scientific Reports, 16*. https://doi.org/10.1038/s41598-026-55000-2
 
Wang, J.-W., Zhou, Y.-F., Devic, S., & Fu, D. (2026). Are LLM Decisions Faithful to Verbal Confidence?. *ArXiv, abs/2601.07767*. https://doi.org/10.48550/arxiv.2601.07767
 
Wen, B.-B., Yao, J.-H., Feng, S.-B., Xu, C., Tsvetkov, Y., Howe, B., & Wang, L. L. (2024). Know Your Limits: A Survey of Abstention in Large Language Models. *Trans. Assoc. Comput. Linguistics, 13*, 529-556. https://doi.org/10.1162/tacl_a_00754
 
Wu, J.-Y., Liu, J.-S., Zeng, Z., Zhan, T., Cai, T., & Huang, W.-H. (2025). Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning. *ArXiv, abs/2512.19920*. https://doi.org/10.48550/arxiv.2512.19920
 
Yona, G., Aharoni, R., & Geva, M. (2024). Can Large Language Models Faithfully Express Their Intrinsic Uncertainty in Words?. *ArXiv, abs/2405.16908*. https://doi.org/10.48550/arxiv.2405.16908
 
Zong, H., Li, B., Long, Y., Chang, S., Wu, J.-L., & Hadfield, G. K. (2026). I-CALM: Incentivizing Confidence-Aware Abstention for LLM Selective Answering. https://doi.org/10.48550/arxiv.2604.03904
 

# Mapeo de papers por dimensión de análisis

Tras revisar sistemáticamente los documentos proporcionados, identifico los siguientes papers para cada dimensión:

---

## 1. Corrección de la respuesta

| Paper ID | Título | Evidencia |
|---|---|---|
| `arxiv_288` | When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust... | "correct repair rate (CRR)"; evalúa si las decisiones del LLM son correctas respecto al ECG de referencia |
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | Clasifica respuestas como "hallucinated/correct"; mide "Overall Hallucination Rate" y "Correctness Rate" |
| `scopus_111` | StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data | Reporta accuracy y MAE en predicción de estrés |
| `ieee_88` | An empirical study for effective LLM-based human state prediction from physiological data | Reporta accuracy en tres tareas (estrés, sueño, actividad) |
| `arxiv_190` | Towards a Personal Health Large Language Model | Reporta accuracy en exámenes profesionales (79% sueño, 88% fitness) |
| `scopus_80` | Improving Dietary Supplement Information Retrieval... | Reporta accuracy: 99% verdadero/falso, 95% selección múltiple |
| `ieee_196` | Reducing diagnostic uncertainty using large language models | Reporta F1 scores para 19 modelos de sistemas corporales |
| `scopus_341` | DiscBioSem-LM: A framework for anxiety detection... | Reporta Accuracy=0.8100, F1=0.8102 |
| `pubmed_39` | Cog-TiPRO: Iterative Prompt Refinement with LLMs... | Reporta accuracy 73.80%, F1 72.67% en detección de MCI |
| `scopus_401` | Reducing annotation burden in physical activity research... | Reporta F1-scores y Cohen's κ para clasificación de intensidad |
| `acl_anthology_73` | Gladiator at MEDIQA-SYNUR 2026 | Reporta precision 74.27%, recall 84.32%, F1 78.98% |
| `ieee_116` | Multi-task multi-emotion joint representation... | Reporta reducciones de MAE/RMSE para depresión, ansiedad, estrés |
| `ieee_886` | Counterfactual modeling with fine-tuned LLMs... | Reporta validity (hasta 0.99), plausibility (hasta 99%), F1 recovery |
| `scopus_77` | Explaining recovery trajectories of older adults... | Validación estadística con tests t, Mann-Whitney, Kruskal-Wallis |
| `scopus_85` | Large language models integrated into brain computer interfaces... | Menciona "semantic accuracy of 88%" en tarea de mensajes ALS |
| `scopus_293` | A survey on large language models for medical time series | Discute métricas de evaluación (AUROC, AUPRC, BLEU, ROUGE, MSE, MAE) |

**Total: 16 papers miden corrección.**

---

## 2. Confianza declarada por el modelo

| Paper ID | Título | Evidencia |
|---|---|---|
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | **Central al paper.** Define "Verbal Uncertainty (VU)" como característica lineal; descubre "Verbal Uncertainty Feature (VUF)" en el espacio de representación; mide correlación entre VU y Semantic Uncertainty (SU) |
| `arxiv_288` | When Derived Measurements Mislead... | Define "derived-feature over-trust (DFOT)": el LLM asigna a una medición derivada "el estatus epistémico de un hecho directo"; mide tasas de sobreconfianza en decisiones del LLM ante señales PPG contradictorias |
| `arxiv_190` | Towards a Personal Health Large Language Model | Usa AutoEval con modelos secundarios para evaluar respuestas; expertos califican calidad con rúbricas Likert; evalúa si el modelo genera "confabulations or unwarranted assumptions" |

**Total: 3 papers miden confianza declarada.**

---

## 3. Certeza transmitida por el lenguaje (hedges y boosters)

| Paper ID | Título | Evidencia |
|---|---|---|
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations | **El más directo.** Ejemplifica explícitamente: "Hmm, maybe Bournemouth?" (VU alta) vs. "Bournemouth" (VU baja); distingue "verbal uncertainty" de "semantic uncertainty"; interviene sobre VUF para inducir expresiones matizadas de incertidumbre |
| `wos_200` | The shrinking landscape of linguistic diversity in the age of large language models | Analiza cambios en marcadores lingüísticos tras reescritura con LLMs; documenta reducción en uso de pronombres, palabras conversacionales y variabilidad estilística; aunque no aborda hedges/boosters epistémicos específicamente, sí examina cómo los LLMs alteran marcadores lingüísticos de identidad |

**Total: 2 papers abordan certeza lingüística, pero solo 1 directamente con hedges/boosters.**

---

## 4. Respaldo de los datos a las afirmaciones (faithfulness/grounding)

| Paper ID | Título | Evidencia |
|---|---|---|
| `arxiv_288` | When Derived Measurements Mislead... | **Central al paper.** Define DFOT: el LLM usa mediciones derivadas "fuera de su scope válido"; evalúa si el LLM confía apropiadamente en mediciones PPG con evidencia de fiabilidad; usa intervención matched-vs-shuffled para verificar si la revisión depende de información específica de la instancia |
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature... | Define alucinaciones como afirmaciones "no apoyadas por su entrada o por sus datos de entrenamiento"; detecta alucinaciones mediante desajuste SU-VU |
| `scopus_80` | Improving Dietary Supplement Information Retrieval... | Sistema RAG que recupera subgrafos de knowledge graph para "grounding" de respuestas; "mitiga el problema de alucinación inherente a LLMs standalone"; verifica que las respuestas estén soportadas por iDISK2.0 |
| `acl_anthology_73` | Gladiator at MEDIQA-SYNUR 2026 | Filtro de "diagnóstico especulativo" que elimina extracciones con lenguaje incierto ('possibly', 'could be', 'might'); validación fisiológica de rangos; filtrado contextual para prevenir falsos positivos no soportados |
| `wos_386` | A scoping review of large language models in personal wellness | Discute riesgos de "hallucination" en LLMs para salud del sueño; señala que respuestas pueden ser "occasionally errors or omissions" |

**Total: 5 papers abordan respaldo de afirmaciones.**

---

## 5. Decisión de responder o abstenerse

| Paper ID | Título | Evidencia |
|---|---|---|
| `acl_anthology_118` | Calibrating Verbal Uncertainty as a Linear Feature... | **El más completo.** Define "Refusal Rate" como métrica; clasifica respuestas en "abstained" (consistentemente se niega) vs. "partly abstained" (cumple al menos una vez); evalúa trade-off entre alucinación y falsa abstención |
| `arxiv_288` | When Derived Measurements Mislead... | El LLM toma decisiones binarias: "Trustworthy" vs. "Suspect"; "Use-As-Is" vs. "Verify-First"; mide "Utility Harm Rate" (verificación innecesaria); evalúa si el modelo se abstiene apropiadamente ante señales no confiables |
| `scopus_80` | Improving Dietary Supplement Information Retrieval... | Cuando las entidades no pueden ser matcheadas en iDISK2.0, el sistema "informa al usuario que el conocimiento específico no está disponible" y recurre al conocimiento interno del LLM como fallback |

**Total: 3 papers abordan abstención.**

---

## Tabla resumen consolidada

| Paper (ID) | Corrección | Confianza | Certeza lingüística | Respaldo | Abstención |
|---|---|---|---|---|---|
| `arxiv_288` | ✅ | ✅ | — | ✅ | ✅ |
| `acl_anthology_118` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `arxiv_190` | ✅ | ✅ (parcial) | — | ✅ (parcial) | — |
| `scopus_80` | ✅ | — | — | ✅ | ✅ (parcial) |
| `acl_anthology_73` | ✅ | — | — | ✅ | — |
| `wos_386` | ✅ (parcial) | — | — | ✅ | — |
| `scopus_111` | ✅ | — | — | — | — |
| `ieee_88` | ✅ | — | — | — | — |
| `ieee_196` | ✅ | — | — | — | — |
| `scopus_341` | ✅ | — | — | — | — |
| `pubmed_39` | ✅ | — | — | — | — |
| `scopus_401` | ✅ | — | — | — | — |
| `ieee_116` | ✅ | — | — | — | — |
| `ieee_886` | ✅ | — | — | — | — |
| `scopus_77` | ✅ | — | — | — | — |
| `scopus_85` | ✅ | — | — | — | — |
| `scopus_293` | ✅ (parcial) | — | — | — | — |
| `wos_200` | ✅ (parcial) | — | ✅ (parcial) | — | — |

---

## Hallazgo clave para tu proyecto

**La dimensión menos cubierta sigue siendo la #3 (certeza lingüística mediante hedges/boosters)**, con solo 1 paper que la aborda directamente (`acl_anthology_118`). Esto confirma el gap identificado previamente:

- **`acl_anthology_118`** es el único paper que cubre las **5 dimensiones simultáneamente**, lo que lo convierte en tu referente metodológico principal.
- **`arxiv_288`** cubre 4 de 5 dimensiones (le falta certeza lingüística) y es el más cercano a tu dominio fisiológico, ya que trabaja con señales PPG/ECG.
- **Ningún paper en este corpus mide las cinco dimensiones sobre señales fisiológicas degradadas**, lo que confirma que tu proyecto llenaría un vacío único en la literatura.

# Analisis adicional

Sí: las cinco dimensiones que planteas están bien respaldadas por la literatura sobre fiabilidad de LLM y son especialmente pertinentes cuando el modelo interpreta señales fisiológicas. Constituyen un marco coherente de evaluación, pero conviene añadir al menos tres dimensiones complementarias: **consistencia/robustez**, **calidad de la percepción de la señal** y **seguridad/clínica utilidad de la salida**.

## Respaldo de las cinco dimensiones

| Dimensión | Respaldo científico | Relevancia para señales fisiológicas |
|---|---|---|
| Corrección de la respuesta | Es el núcleo de la evaluación de LLM: exactitud, factualidad y calidad del razonamiento. En salud, la literatura recomienda evaluar el proceso, no solo la respuesta final.  [arxiv](https://arxiv.org/html/2510.17764v1) | Fundamental: un error de interpretación de ECG, ritmo cardíaco, sueño, actividad o variabilidad puede traducirse en una recomendación incorrecta. |
| Confianza declarada | La “verbalized confidence” está ampliamente estudiada, pero se documenta que suele estar mal calibrada y ser sobreconfiada.  [arxiv](https://arxiv.org/html/2407.18418v1) | Permite comprobar si el modelo distingue entre una lectura clara y una señal ambigua, ruidosa o incompleta. |
| Certeza lingüística: hedges y boosters | Los marcadores epistémicos se estudian como señales de incertidumbre comunicada; los hedges reducen el compromiso y los boosters lo aumentan.  [dl.acm](https://dl.acm.org/doi/full/10.1145/3816046.3816231) | Es especialmente relevante porque el lenguaje puede transmitir seguridad clínica incluso cuando la evidencia de la señal es débil. |
| Respaldo de los datos a las afirmaciones | La literatura de *grounding*, atribución y *faithfulness* exige que las afirmaciones se deriven de evidencia verificable y no de priors del modelo.  [is.ijs](https://is.ijs.si/wp-content/uploads/2026/02/AI_IN_HEALTHCARE_2025_paper_2_ack.pdf) | Crítico: el modelo debe justificar sus conclusiones en rasgos concretos de la señal, no inventar patrones fisiológicos. |
| Responder o abstenerse | La abstención es un área consolidada, con marcos específicos en salud que evalúan abstención apropiada, sobre-rechazo y calidad de la justificación.  [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC13631354/) | Esencial: ante datos insuficientes, artefactos o ambigüedad clínica, abstenerse o escalar puede ser más seguro que responder. |

## Qué dice la literatura

### Corrección y confianza no bastan

La corrección mide si la salida es correcta; la confianza declarada mide cuánta seguridad expresa el modelo. La literatura muestra que ambas deben analizarse conjuntamente mediante calibración: una respuesta correcta con confianza baja y una incorrecta con confianza alta son fallos distintos, pero ambos degradan la fiabilidad. [arxiv](https://arxiv.org/html/2407.18418v1)

Las métricas habituales incluyen exactitud, *Expected Calibration Error* (ECE), *Brier score*, curvas de fiabilidad y análisis por intervalos de confianza.  En dominios clínicos se recomienda complementarlas con curvas riesgo-cobertura (*risk–coverage*), que muestran el error aceptado según el porcentaje de casos que el sistema decide responder. [arxiv](https://arxiv.org/html/2510.17764v1)

### La confianza verbalizada es útil pero no fiable por sí sola

Los LLM pueden expresar confianza numérica, verbal o mediante escalas, pero diversos estudios encuentran sobreconfianza y baja discriminación entre respuestas correctas e incorrectas.  Además, la confianza declarada puede predecir mejor la decisión posterior de responder o abstenerse que la probabilidad real de acierto, lo que la convierte en una señal conductual útil, pero no en una medida directa de corrección. [arxiv](https://arxiv.org/html/2407.18418v1)

Para tu proyecto, esto justifica tratar la **confianza declarada** como una variable observable independiente de la **corrección**, y no como un sustituto de ella.

### Hedges y boosters miden certeza comunicada

Los *hedges* —“posiblemente”, “parece”, “podría indicar”— y los *boosters* —“claramente”, “definitivamente”, “confirmado”— son marcadores lingüísticos de compromiso epistémico.  La literatura muestra que los modelos pueden endurecer la incertidumbre presente en un texto original, eliminando hedges y produciendo afirmaciones más categóricas. [dl.acm](https://dl.acm.org/doi/full/10.1145/3816046.3816231)

También se ha observado que los modelos usan marcadores epistémicos de forma imperfectamente alineada con su confianza interna: preservan cierto orden relativo entre marcadores, pero no diferencian de manera robusta sus niveles de certeza.  Por tanto, el análisis de hedges y boosters es pertinente, pero debe contrastarse con la corrección y con la evidencia disponible; un *booster* ante una señal ambigua es un indicador de riesgo. [arxiv](https://arxiv.org/html/2605.28778v1)

### El respaldo de los datos es una dimensión diferenciada

La *faithfulness* o fundamentación exige que las afirmaciones estén estrictamente apoyadas por la evidencia proporcionada, sin introducir afirmaciones no respaldadas.  En evaluación médica se emplean métricas como precisión de citas, tasa de respuestas fundamentadas, contradicción con la fuente y detección de alucinaciones. [arxiv](https://arxiv.org/html/2510.17764v1)

En señales fisiológicas, esta dimensión puede operacionalizarse preguntando: **¿qué segmento, valor, tendencia, evento o característica de la serie respalda cada afirmación?** La literatura reciente señala que aplicar LLM directamente a datos fisiológicos crudos puede producir consejos inconsistentes o alucinados, y propone separar procesamiento numérico, interpretación y generación textual para preservar trazabilidad. [scitepress](https://www.scitepress.org/Papers/2026/144802/144802.pdf)

### La abstención es un criterio de seguridad, no solo de humildad

La abstención se justifica por dos motivos: incertidumbre y riesgo de daño.  En contextos clínicos, los modelos tienden a no abstenerse incluso cuando falta información decisiva; estudios recientes encuentran que, al eliminarse la opción correcta, los modelos siguen eligiendo respuestas incorrectas con alta confianza. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC13631354/)

Los marcos más actuales proponen ir más allá del binomio “responder/abstener”: por ejemplo, **decidir, pedir más información o derivar a un clínico**.  Esto es muy relevante para sensores fisiológicos, donde a menudo la respuesta correcta no es un diagnóstico, sino solicitar un registro más largo, comprobar calidad de señal o remitir a supervisión humana. [arxiv](https://arxiv.org/pdf/2610.04542v2.pdf)

## Dimensiones que conviene añadir

Tu conjunto es sólido, pero para interpretar y validar respuestas sobre señales fisiológicas sugeriría incorporar:

- **Percepción y extracción correcta de la señal:** antes de valorar la interpretación, hay que comprobar si el modelo identifica correctamente tendencias, picos, arritmias, artefactos, valores fuera de rango o cambios temporales. El benchmark HEARTS muestra que los LLM tienen dificultades con el razonamiento temporal multicapa y rinden peor que modelos especializados en series de salud. [proceedings.mlr](https://proceedings.mlr.press/v306/li26hr.html)
- **Consistencia y robustez:** la misma señal, con ligeras variaciones de formato, ventana temporal, ruido o formulación de la pregunta, debería producir interpretaciones estables. La robustez es un pilar explícito en marcos de evaluación médica. [is.ijs](https://is.ijs.si/wp-content/uploads/2026/02/AI_IN_HEALTHCARE_2025_paper_2_ack.pdf)
- **Seguridad y gravedad del error:** no todos los errores son equivalentes. Un falso “todo normal” ante una posible arritmia puede ser más peligroso que un falso positivo benigno. La literatura clínica recomienda medir errores inseguros, violaciones de guías y tasa de error con alta confianza. [is.ijs](https://is.ijs.si/wp-content/uploads/2026/02/AI_IN_HEALTHCARE_2025_paper_2_ack.pdf)
- **Utilidad y accionabilidad:** la respuesta debe indicar si la conclusión es clínicamente accionable, si requiere confirmación y qué información adicional sería necesaria. Los marcos de autonomía clínica proponen evaluar la salida según el nivel de decisión que se permite al sistema. [arxiv](https://arxiv.org/html/2510.17764v1)
- **Trazabilidad de la evidencia:** conviene distinguir entre “el dato respalda la afirmación” y “el modelo cita un dato”. Una cita puede existir y aun así no justificar la conclusión; por eso la validación debe ser a nivel de afirmación, no solo de respuesta. [is.ijs](https://is.ijs.si/wp-content/uploads/2026/02/AI_IN_HEALTHCARE_2025_paper_2_ack.pdf)

## Propuesta de modelo integrado

Una formulación útil para tu tesis sería analizar cada respuesta como una secuencia de afirmaciones \(a_i\), cada una con:

\[
\text{Fiabilidad}(a_i) = f(\text{corrección}, \text{confianza declarada}, \text{certeza lingüística}, \text{soporte evidencial}, \text{decisión de compromiso})
\]

Donde la **decisión de compromiso** incluye responder, matizar, pedir datos, abstenerse o escalar. Esta estructura es coherente con la literatura de incertidumbre consciente del riesgo, que distingue entre fuentes de incertidumbre, señales observables, calibración y acción posterior. [arxiv](https://arxiv.org/html/2608.17084v1)

## Implicación principal

La literatura respalda plenamente tus cinco dimensiones, pero sugiere no tratarlas como cinco métricas paralelas. Deben analizarse como un **sistema de validación cruzada**:

- Alta confianza + bajo soporte evidencial → riesgo de alucinación.
- Lenguaje categórico + señal ambigua → riesgo de transmisión indebida de certeza.
- Respuesta correcta + confianza excesiva → mala calibración.
- Abstención frecuente sin justificación → pérdida de utilidad o sobre-rechazo.
- Respuesta incorrecta + ausencia de abstención → fallo crítico de seguridad.

En el contexto de señales fisiológicas, el criterio central no debería ser únicamente “¿acertó el LLM?”, sino: **“¿respondió solo cuando la señal, la evidencia y el riesgo lo justificaban, y comunicó una certeza proporcional al soporte disponible?”** [arxiv](https://arxiv.org/html/2510.17764v1)


# Analisis adicional 2

# Evaluación de respuestas de LLM sobre señales fisiológicas: dimensiones y marco de validación

## Síntesis

Las cinco dimensiones propuestas —corrección, confianza declarada, certeza lingüística, respaldo empírico de las afirmaciones y decisión de responder o abstenerse— son pertinentes y están respaldadas por líneas de investigación convergentes sobre calibración, incertidumbre verbal, factualidad/grounding y predicción selectiva. No son, sin embargo, cinco medidas intercambiables ni basta con analizarlas por separado: describen respectivamente el resultado, la metacognición comunicada, la forma retórica, la relación afirmación-evidencia y la política de decisión. La evaluación más sólida las vincula por respuesta y por afirmación, y las contrasta con referencia independiente. [^1][^2][^3][^4][^5]

Para una tesis centrada en datos de sensores fisiológicos, conviene añadir al menos: (1) validez de la interpretación del propio sensor y de sus transformaciones (calidad, artefactos, sincronización y datos faltantes); (2) calibración de la incertidumbre por subgrupo, tarea y condiciones de adquisición; (3) robustez ante perturbaciones y cambios de distribución; (4) utilidad/daño de los errores y evaluación clínica o funcional; y (5) trazabilidad y fidelidad de las explicaciones a la evidencia realmente usada. La literatura específica de señales fisiológicas y LLM todavía es emergente: la evidencia más directa evalúa tareas o benchmarks concretos, mientras que la evidencia de confianza verbal y abstención procede mayoritariamente de preguntas médicas o científicas, no de interpretación de ondas fisiológicas en flujo real. [^5][^2][^3][^4]

## ¿Están respaldadas y son relevantes las cinco dimensiones?

| Dimensión propuesta | Respaldo y valor | Cómo medirla y cautela principal |
|---|---|---|
| **Corrección de la respuesta** | Es indispensable, pero no representa por sí sola fiabilidad. Los trabajos de interpretación de datos fisiológicos evalúan desempeño en tareas específicas; también se documenta que los LLM pueden producir respuestas genéricas, inexactas o mal fundamentadas en series temporales. [^5][^6][^7] | Definir unidad de evaluación y respuesta de referencia: detección de evento, estimación numérica, clasificación, tendencia o explicación. Informar métricas adecuadas a cada tarea (p. ej., sensibilidad/especificidad, error de estimación, F1), además de exactitud global. Un texto plausible no prueba que el modelo haya leído correctamente la señal. |
| **Confianza declarada por el modelo** | Es una dimensión relevante de metacognición, pero debe tratarse como una predicción que requiere calibración, no como evidencia de fiabilidad intrínseca. En preguntas clínicas se observó confianza alta con independencia de la corrección y diferencias pequeñas entre confianza en respuestas correctas e incorrectas. [^1] En otro estudio médico, la calibración varió entre modelos y especialidades; la alta exactitud no garantizó incertidumbre fiable. [^2] | Solicitar una probabilidad numérica asociada a una respuesta bien definida y estimar calibración (reliability plots, Brier/log score o ECE, junto con discriminación). No equiparar “confianza” declarada con probabilidad objetiva, ni comparar escalas distintas sin normalización. Estimar por tarea, modelo y contexto. |
| **Certeza transmitida por el lenguaje (hedges/boosters)** | Justificada como dimensión comunicativa distinta de la confianza numérica. La investigación reciente encuentra que los LLM asignan significados probabilísticos a marcadores verbales que pueden diferir sustancialmente de los humanos, y propone calibrar la relación marcador-corrección. [^3] | Codificar marcadores lingüísticos (p. ej., “posible”, “probable”, “sin duda”) y su intensidad, manualmente con protocolo reproducible o con clasificador validado. Comprobar consistencia entre marcador, confianza numérica, evidencia y corrección. La expresión puede variar por idioma, estilo, prompt y audiencia; los marcadores no tienen probabilidades universales. |
| **Respaldo de los datos a las afirmaciones** | Es central para interpretación fisiológica: una respuesta puede ser correcta por casualidad o por conocimiento previo y, aun así, no estar sustentada por la señal concreta. En sensores temporales, añadir resúmenes estructurados, metadatos y contexto mejoró grounding y redujo alucinaciones/contradicciones en una evaluación específica. [^5] Los trabajos sobre salud también subrayan la necesidad de fundamentar el razonamiento textual en datos numéricos. [^6] | Descomponer la respuesta en afirmaciones verificables; evaluar si cada una está implicada por la señal, sus características derivadas o metadatos autorizados. Separar **apoyo** (la evidencia justifica la afirmación) de **fidelidad** (la explicación refleja el proceso/evidencia realmente usado) y de **corrección** (la afirmación es verdadera). Las métricas automáticas de juez-LLM necesitan validación humana o instrumental. |
| **Responder o abstenerse** | Pertinente para aplicaciones de riesgo y como decisión selectiva: el sistema debe poder no emitir una conclusión cuando la señal o la evidencia no bastan. Estudios de razonamiento científico muestran que la abstención puede ser inconsistente y sensible al modelo, al texto de la opción y al modo de razonamiento. [^4] | Medir cobertura (proporción respondida), riesgo/error entre las respondidas, curva riesgo-cobertura y desempeño de la regla de abstención. Definir qué cuenta como abstención y qué alternativas ofrece (solicitar señal mejor, derivar a experto, limitar la conclusión). Una alta abstención no es automáticamente segura: puede ser inútil o selectiva de forma injusta. | 

## Qué dice la literatura sobre analizarlas conjuntamente

La interpretación de señales debe evaluarse como una cadena, no como una respuesta textual aislada. Primero se determina si la entrada representa adecuadamente la fisiología; después, si el modelo extrajo una interpretación correcta; luego, si la formuló con un grado de certeza congruente con su desempeño y con el respaldo de la evidencia; finalmente, si la decisión de responder era apropiada. Los trabajos sobre sensores apuntan precisamente a que los LLM tienen dificultades con señales crudas y pueden mejorar al recibir representaciones estructuradas y contexto explícito. [^5][^7]

El análisis conjunto permite detectar perfiles que una métrica aislada ocultaría: respuesta correcta pero sin soporte de la señal; respuesta equivocada expresada con gran certeza; afirmación bien respaldada pero lenguaje excesivamente categórico; o abstención ante un caso resoluble. Es una inferencia metodológica coherente con la evidencia de desajuste entre confianza y corrección, con la divergencia semántica de los marcadores verbales, y con la inestabilidad de abstención. [^1][^2][^3][^4]

Una operacionalización útil conservaría un registro por caso y por afirmación con: resultado de referencia; salida y confianza numérica; marcadores epistémicos; evidencia de señal/segmento/metadatos que respalda cada afirmación; etiqueta de apoyo/fidelidad; y decisión responder/abstenerse. Deben analizarse tanto asociaciones (p. ej., confianza–corrección, hedging–error, soporte–corrección) como desacuerdos entre dimensiones. No conviene condensar inmediatamente todo en una puntuación compuesta: puede ocultar fallos críticos compensados por buen rendimiento en otra dimensión; la variación por especialidad y las diferencias entre confianza, exactitud y abstención justifican conservar resultados separados. [^2][^1][^4]

## Dimensiones adicionales recomendadas

1. **Validez de la señal y calidad de entrada.** Evaluar artefactos, pérdida de datos, frecuencia de muestreo, calibración del sensor, sincronización entre canales, segmentación y calidad de la conversión a tokens, gráficos o resúmenes. Distinguir error de adquisición/preprocesamiento de error de interpretación del LLM. La literatura de sensores señala dificultades con datos numéricos crudos y mejoras mediante codificación/guía métrica. [^6][^7]
2. **Robustez y generalización.** Probar ruido, artefactos plausibles, cambios de frecuencia/dispositivo, faltantes, variaciones poblacionales y cambios de contexto. Comprobar si se alteran simultáneamente corrección, confianza, lenguaje, soporte y abstención. Los resultados de sensores dependen del sistema de representación y de la tarea, por lo que la validez externa no debe asumirse a partir de un benchmark único. [^5][^7]
3. **Resolución y calibración por estratos.** Informar por tipo de señal (ECG, PPG, EEG, signos vitales), condición/tarea, calidad de señal, población, idioma, modelo y nivel de dificultad. Los promedios pueden ocultar variación importante: un estudio médico reportó diferencias de calibración entre especialidades pese a su promedio agregado. [^2]
4. **Fidelidad explicativa y trazabilidad.** Comprobar si la explicación identifica correctamente los intervalos, características y metadatos que justifican la conclusión, y si el razonamiento cambia cuando se retira o altera evidencia relevante. Citar o señalar la evidencia no garantiza por sí mismo que la explicación represente el proceso decisorio.
5. **Utilidad y consecuencias del error.** Ponderar errores por severidad clínica o funcional, tiempo de aviso y posibilidad de intervención. Una clasificación errónea de evento crítico no debe pesar igual que una descripción imprecisa de una tendencia no accionable. La decisión de responder/abstenerse debe evaluarse contra costes explícitos de falsos positivos, falsos negativos y abstenciones.
6. **Reproducibilidad y estabilidad.** Medir variabilidad por repetición, prompt, orden de contexto, versión del modelo y temperatura. En sistemas con herramientas, registrar qué análisis, transformaciones o recuperaciones se usaron para producir la respuesta.

## Propuesta de marco para el proyecto doctoral

**Nivel 1: validez de entrada.** Registrar características del sensor, calidad, artefactos, faltantes y transformaciones; preparar casos de referencia con anotación experta y desacuerdos documentados. Las limitaciones de interpretación de señales crudas y las mejoras observadas con codificación estructurada respaldan tratar la entrada como una fuente de error separada. [^5][^7]

**Nivel 2: desempeño de tarea.** Separar inferencias cuantitativas, detección/clasificación y explicación narrativa; puntuar cada objetivo con métricas apropiadas y evaluación clínica cuando corresponda. Los estudios de salud con señales de wearables evalúan tareas fisiológicas diferenciadas, lo que aconseja no tratarlas como un único constructo de exactitud. [^6]

**Nivel 3: evaluación epistémica y discursiva.** Por respuesta, contrastar confianza numérica y marcadores verbales con corrección y dificultad. Analizar calibración global y estratificada; documentar si los hedges/boosters expresan incertidumbre adecuada, no sólo su frecuencia. La evidencia muestra desajuste entre confianza y corrección, así como significados probabilísticos verbales distintos de los humanos. [^1][^2][^3]

**Nivel 4: soporte a nivel de afirmación.** Atomizar las respuestas; verificar cada afirmación contra la señal y el contexto permitido; calificar apoyo, contradicción, omisión y fidelidad de la explicación. Mantener separadas verdad, respaldo y fidelidad. El grounding y las contradicciones con los datos se han evaluado como modos de fallo diferenciados en interpretación temporal de sensores. [^5]

**Nivel 5: decisión selectiva.** Evaluar cobertura–riesgo y utilidad de la abstención bajo criterios preespecificados. Incluir casos sin respuesta determinable, con señal degradada y con evidencia contradictoria. La abstención observada puede depender de la disponibilidad de la respuesta y del modo de formular la opción, por lo que debe medirse explícitamente. [^4]

**Nivel 6: robustez y utilidad.** Repetir bajo perturbaciones y datos externos, evaluar subgrupos y estimar el impacto de los errores. Comparar LLM solo, LLM con herramientas de análisis de señal y flujo híbrido con especialista. Las diferencias de calibración entre especialidades y la dependencia del desempeño respecto de la representación de la señal apoyan el análisis estratificado y la validación externa. [^2][^7]

## Conclusión

Las cinco dimensiones constituyen una base conceptualmente sólida para estudiar cómo se interpretan y validan las respuestas de LLM sobre sensores fisiológicos; su principal fortaleza es que separan desempeño, autoconfianza, comunicación de certeza, soporte probatorio y conducta selectiva. Para que el marco sea válido, la corrección no debe funcionar como sustituto de grounding, la confianza verbal no debe tratarse como calibrada sin probarla, y la abstención debe valorarse junto con cobertura y coste del error. La tesis ganaría alcance si incorpora calidad de señal, robustez/generalización, fidelidad explicativa, estratificación y utilidad clínica. Las limitaciones con señales crudas, el mejor grounding al estructurar contexto y las variaciones observadas en calibración respaldan estas extensiones. [^5][^7][^2]

La brecha más importante es de evidencia directa: los estudios específicos de sensores fisiológicos aún no parecen integrar sistemáticamente las cinco dimensiones en un mismo protocolo y con validación frente a expertos. La evidencia de calibración lingüística y abstención procede en buena medida de tareas médicas o científicas de preguntas y respuestas, por lo que su transferencia a interpretación de ondas y series fisiológicas debe probarse empíricamente, no darse por sentada. [^5][^2][^3][^4]


[^1]: Omar et al., 2024. Overconfident AI? Benchmarking LLM Self-Assessment in Clinical Scenarios. medRxiv.

[^2]: Boie et al., 2026. Calibration of Self-Reported Confidence and Accuracy of Large Language Models in Medical Question Answering. Journal of medical systems.

[^3]: Duan et al., 2026. "very likely"Means"uncertain"? How LLMs Diverge from Humans in Linguistic Uncertainty Quantification.

[^4]: Summers et al., 2026. Exploring the Abstention Ability of LLMs using Scientific Reasoning in Critical Fields. Proceedings of the 26th ACM/IEEE Joint Conference on Digital Libraries.

[^5]: Rico & Larson, 2026. Making Sense of Sensors: Improving LLM Interpretation of Time-Series Data. AI Sensors.

[^6]: Liu et al., 2023. Large Language Models are Few-Shot Health Learners. arXiv.org.

[^7]: Hota et al., 2024. Evaluating Large Language Models as Virtual Annotators for Time-Series Physical Sensing Data. ACM Transactions on Intelligent Systems and Technology.