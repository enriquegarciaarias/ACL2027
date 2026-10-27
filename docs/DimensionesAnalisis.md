# Las **cinco dimensiones clave** interactúan fuertemente al evaluar respuestas de LLMs.

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


