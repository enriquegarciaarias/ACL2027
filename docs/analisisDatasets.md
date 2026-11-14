# Datasets estudios sensores fisiológicos


## Tabla datasets1
**Notas de alcance:**  
- `ieee_196`, `wos_403`, `acl_anthology_148`, `acl_anthology_231`, `acl_anthology_286`, `acl_anthology_169`, `pubmed_157`/`wos_327` y `wos_262` **no tratan una señal fisiológica** como ECG/EEG/EMG; trabajan con texto clínico, benchmarks, notas o revisiones.  
- `ieee_598`, `scopus_293`, `scopus_50`, `pubmed_157`, `wos_327` y `wos_262` son **revisiones/surveys**; los datasets listados son los mencionados/analizados en el texto, no datasets primarios únicos. 

| Paper ID | Título | Dataset(s) | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|---|---|
| `pubmed_128` | On the limits of LLM surprisal as a functional explanation of the N400 and P600 | Tres datasets ERP controlados de los Estudios 1–3 (no nombrados) | EEG (N400/P600) | Evaluar si el surprisal de LLM explica N400/P600 y si está contaminado por asociación | El surprisal LLM no captura consistentemente ambos componentes; está contaminado por asociación, sobre todo en modelos pequeños; propone P600 como mejor índice de surprisal verdadero |
| `scopus_85` | Large language models integrated into brain computer interfaces for communication | No dataset primario; revisión sistemática de 11 estudios; escenarios P300, SSVEP, cVEP, copy-spelling y ALS message-bank | EEG/BCI (P300, SSVEP, cVEP) | Revisar integración LLM-BCI para comunicación asistiva | Taxonomía de 5 patrones de integración; reporta ahorros >50% de tecleo y 42 caracteres/min en un estudio ALS; advierte falta de pacientes con discapacidad motora y poca latencia reportada |
| `scopus_341` | DiscBioSem-LM: a framework for anxiety detection via semantic fusion of multimodal physiological signals and LLMs | Dataset propio de escenarios de intervención psicológica en VR (180 participantes; EMG, EDA, HR, BVP, RESP, ACC; SAI) | EMG, EDA, HR, BVP, respiración, acelerometría | Detección de ansiedad mediante descripciones semánticas y fine-tuning discriminativo de LLM | Accuracy 0.8100; F1 0.8102; supera baselines multimodales |
| `pubmed_39` | Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands | Comandos de voz de 35 adultos mayores; 15 con interacción diaria en casa durante 18 meses | Voz/audio y transcripciones | Detectar deterioro cognitivo leve mediante comandos de voz longitudinales y refinamiento iterativo de prompts | Accuracy 73.80%; F1 72.67%; mejora 27.13% sobre baseline |
| `scopus_401` | Reducing annotation burden in physical activity research using vision language models | CAPTURE24 Oxfordshire (161 participantes) y CAPTURE24-CN Sichuan (111 participantes) | Imágenes de cámara wearable y acelerometría de muñeca | Clasificar intensidad de actividad física con VLMs y modelos discriminativos | SB bien predicho; LIPA/MVPA peor; caída fuerte al generalizar a Sichuan; modelos útiles para reducir anotación, no superan anotadores humanos |
| `ieee_88` | An empirical study for effective LLM based human state prediction from physiological data | WESAD; HHAR; DREAMT | ECG, datos de sensores fisiológicos/wearables | Evaluar combinaciones de representación de entrada, uso de LLM y forma de salida en predicción de estados humanos | Feature input + few-shot + label-with-explanation fue la configuración efectiva; diferencias individuales y temporales afectan el rendimiento |
| `scopus_77` | Explaining recovery trajectories of older adults post lower limb fracture using modality wise multiview clustering and LLMs | MAISON-LLF (560 días; 10 adultos mayores; aceleración, pasos, movimiento ambiental, GPS, HR, sueño; SIS, OHS, OKS, TUG, 30sCST) | Acelerometría, HR, sueño, GPS, movimiento ambiental | Interpretar clústeres multimodales con LLM en recuperación tras fractura | La mayoría de etiquetas de clúster fueron estadísticamente significativas frente a escalas clínicas; aceleración no significativa |
| `ieee_116` | Multi task multi emotion joint representation and LLM based posterior calibration for mental health risk assessment | Dataset propio multitarea/multiemoción con 237 participantes | Onda pupilar | Evaluar depresión, ansiedad y estrés con representación conjunta y calibración posterior con LLM | Reduce MAE/RMSE frente a modelos de una tarea; calibración LLM mejora errores adicionalmente |
| `arxiv_288` | When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence | MIMIC-III Matched Waveform Database v1.0: 50,000 registros PPG–ECG de 1,275 pacientes; test bloqueado de 187 pacientes | PPG y ECG | Definir y cuantificar sobreconfianza en medidas derivadas (DFOT) y evaluar reparación con evidencia de fiabilidad | K2 mejora CRR/ESRM y especificidad en 1.82–6.69 pp; UHR +0.67 pp; intervalos pareados excluyen cero |
| `scopus_111` | StressLLM: Large Language Models for Stress Prediction via Wearable Sensor Data | PMData; WESAD | Datos fisiológicos/wearables (HR, sueño, actividad, EDA, ECG, etc.) | Identificar biomarcadores óptimos para predicción de estrés con LLM y ablación de características | El tamaño del modelo no siempre correlaciona con accuracy; edad ayuda; género poco; biomarcadores óptimos varían por dataset/modelo |
| `ieee_886` | Counterfactual modeling with fine tuned LLMs for health intervention design and sensor data augmentation | AI-READI Flagship Dataset (1,067 participantes; CGM, actividad, sueño, estrés, sensores ambientales) | Datos fisiológicos, conductuales y ambientales de wearables | Generar contra factuales clínicamente plausibles y usarlos para augmentación bajo escasez de etiquetas | LLaMA-3.1-8B fine-tuned: plausibilidad hasta 99%, validez hasta 0.99; recuperación media ~20% F1 en escasez de etiquetas |
| `scopus_56` | Toward multimodal AI for dementia: a challenge driven survey of LLMs and VLMs | DementiaBank; ADReSS; TIHM; dataset multimodal ADL [13] (no nombrado); ADRD-Bench | Voz/audio, vídeo, sensores, datos fisiológicos, texto | Revisar LLM/VLM para demencia en grounding, multimodalidad y personalización | Taxonomía de tres retos; RAG y grafos de conocimiento ayudan; fusión multimodal y personalización siguen abiertas |
| `pubmed_150` | Seamless monitoring of stress levels leveraging a foundational model for time sequences | DREAMER; MAHNOB-HCI; WESAD; @HOME | ECG, BVP/PPG, HR/HRV, señales ambientales | Detección de estrés como anomalía personalizada con UniTS y explicaciones LLM | UniTS supera 12 métodos; F1 medio 0.821; en @HOME 30/32 anomalías confirmadas; robusto con wearables |
| `arxiv_365` | Daily and Weekly Periodicity in Large Language Model Performance and Its Implications for Research | Humanity’s Last Exam (tarea de física; consultas repetidas a GPT-4o) | N/A | Testar invariancia temporal del rendimiento de GPT-4o bajo condiciones fijas | ~20% de la varianza atribuible a componentes periódicos; variación pico-pico ~14% de la escala; no hay deriva lineal significativa |
| `arxiv_286` | Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data | Dreaddit; DepSeverity; SDCNL; CSSRS-Suicide; Red-Sam; Twt-60Users; SAD | N/A (texto online) | Evaluar zero-shot, few-shot e instruction fine-tuning en predicción de salud mental | Mental-Alpaca y Mental-FLAN-T5 superan GPT-3.5/GPT-4 y se acercan a Mental-RoBERTa; fine-tuning multi-dataset mejora generalización |
| `pubmed_161` | LLM powered dysphagia screening with multimodal physiological signal analysis and medically informed prompts | Dataset clínico propio: 217 participantes (109 con disfagia post-ictus, 108 sanos); 1,391 muestras disfágicas y 1,273 sanas | Vibración laríngea triaxial, flujo nasal, sonido de deglución | Screening de disfagia con señales multimodales y prompts médicos | Accuracy 96.3%; sensibilidad 95.2%; especificidad 97.4%; estable en few-shot |
| `ieee_676` | Multi modal integration analysis of Alzheimer’s disease using LLMs and knowledge graphs | Cinco datasets públicos independientes (MRI, EEG, biomarcadores, clínicos, expresión génica); tres cohortes de validación independientes (n=28–45) | MRI, EEG, biomarcadores, datos clínicos, expresión génica | Integrar datos fragmentados de Alzheimer sin matching de IDs mediante grafos y LLM | Encuentra relaciones cruzadas; ruta metabólico-inflamatoria-tau con r>0.6; correlaciones EEG-gen r=0.42–0.58; validación con efectos consistentes |
| `ieee_255` | Multimodal to semantics large language model framework for loneliness assessment | Dataset de soledad de trabajo previo [11]: Oura Ring, Samsung Gear Sport y smartphone AWARE; 16 participantes | HRV, HR, sueño, actividad, uso de smartphone | Evaluar soledad con representaciones semánticas relativas a baseline individual y LLM | No mejora el ajuste numérico global, pero reduce subestimación de soledad alta; falsos negativos 39 vs 48 del ML |
| `pubmed_105` | Multimodal large language models in health care: applications, challenges and future | No dataset primario; revisión. Menciona MIMIC-IV, PMC-VQA, LLaVA-Med y caso CONCH | Imágenes médicas, series temporales, audio, texto, vídeo, ómicas | Revisar principios, aplicaciones, retos y futuro de M-LLM en salud | Marco conceptual; destaca fusión de datos, benchmarks, interpretabilidad, privacidad y regulación |
| `ieee_378` | AI driven wearable sensor analytics for dementia monitoring: machine learning, deep learning and emerging foundation model opportunities | Revisión de 42 estudios; menciona SINEW (n=204); actigrafía research-grade 87.8%, wearables consumer <15% | IMU, PPG, ECG, actigrafía, sueño, señales oculares | Revisar biomarcadores digitales y paradigmas ML/DL/FM en demencia | ML es más maduro (TRL 6–7); DL limitado por datos (TRL 4–5); FM incipiente (TRL 2–3); propone arquitecturas híbridas edge-cloud |
| `pubmed_100` | Automated generation of discharge summaries leveraging large language models with clinical data | 25 casos de cirugía pancreática del Heidelberg University Hospital; EHR estructurado y resúmenes médicos en alemán | N/A (texto clínico) | Generar resúmenes de alta en alemán desde datos estructurados con LLM open-source | ~2.84 errores/resumen; ROUGE-1 0.25; BERTScore 0.64; valoraciones 3.72–3.96/5; solo 60% evaluó completitud como buena |
| `pubmed_12` | The role of multimodal generative AI in older adults’ health management: systematic scoping review | Revisión sistemática de 28 estudios; no dataset primario | Modalidades multimodales (texto, audio, imagen, sensores) | Revisar rol, impactos y tendencias de GenAI multimodal en salud de mayores | 82% de publicaciones 2024–2025; predominan chatbots; evidencia limitada a mayores cognitivamente normales; faltan estudios longitudinales y equidad |
| `pubmed_194` | Large language models to extract cancer staging data from clinical documentation | 700 notas anotadas de un sistema de salud integrado; 2.5M registros de estadificación de 217,768 pacientes desde >2M notas; cáncer vesical, cervical, colorrectal, mama y próstata | N/A (documentación clínica) | Extraer estadificación oncológica desde notas clínicas con LLM fine-tuned | Precisión a nivel relación 0.77–1.0 en tres cánceres y 0.83–1.0 sin entrenamiento adicional en dos cánceres |
| `pubmed_78` | Automating clinical phenotyping using natural language processing | MSDW/MSCCR: 49,572 notas clínicas y 2,204 informes de radiología de 584 pacientes con Crohn; test de 280 textos; 134 pacientes para nivel paciente | N/A (texto clínico y radiológico) | Comparar reglas spaCy vs GPT-4 para fenotipado de Crohn | GPT-4 similar o mejor; F1 nota ≥0.90 conducta y ≥0.82 edad; nivel paciente ≥0.66 y ≥0.71; sin diferencia significativa vs expertos |
| `scopus_223` | Large language models in sports cardiology: challenges, governance and human in the loop integration | No dataset primario; revisión narrativa | ECG, imagen cardiovascular, síntomas, historia familiar | Revisar rol de LLM en cardiología deportiva y gobernanza | Propone paradigma text-proxy; LLM como capa semántica supervisada, no diagnóstico autónomo; validación por etapas |
| `scopus_97` | A novel chain of thought reasoning approach for Alzheimer’s disease detection using LLM and vision-language models | ADReSS / DementiaBank Pitt Corpus (descripción de imagen “cookie theft”) | Voz/audio, imagen, texto | Detectar Alzheimer con ASR, VQA y Chain-of-Thought durante fine-tuning | Accuracy 86.67%; F1 86.34%; CoT mejora relativa 16.7% |
| `arxiv_280` | Benchmarking LLM Summaries of Multimodal Clinical Time Series for Remote Monitoring | TIHM-1.5 (56 personas con demencia; 2,803 días-paciente; test de 100 días-paciente) | HR, presión arterial, temperatura, actividad de habitaciones, sueño | Evaluar fidelidad clínica de resúmenes LLM de series temporales multimodales | Métricas NLP altas no implican corrección clínica; pipeline visual Gemini-2.5-Pro logra 45.7% recall de anomalías y 100% recall de duración/cobertura |
| `scopus_244` | Integrating large language models with formal planning to automate the design and validation of biosignal processing pipelines | Dataset novel de 78 descripciones de pipelines ECG/EEG de PubMed; corpus preliminar de escenarios | ECG y EEG (descripciones de procesamiento) | Traducir lenguaje natural a pipelines válidos de procesamiento biosignal con LLM + planificación HTN | Precisión 0.89–0.98 mapeando bloques; F1 0.97–0.98 generando pipelines válidos de 2–15 pasos |
| `scopus_197` | EmoSavior: depression recognition and intervention via multimodal physiological signals and LLMs | Dataset propio de 200 participantes; EEG, ECG, GSR; PHQ-9; estímulos IADS-2 | EEG, ECG, GSR | Reconocimiento de depresión e intervención cerrada con LLM | Accuracy 0.7667; F1 0.7666; supera baselines y SOTA; ablación confirma componentes |
| `pubmed_69` | Unveiling the potential of large language models in transforming chronic disease management: mixed methods systematic review | Revisión sistemática de 20 estudios; no dataset primario | N/A (revisión) | Sintetizar evidencia sobre LLM en manejo de enfermedades crónicas | Tasa de acierto agrupada 71%; RAG mejora precisión (OR 2.89); retos de privacidad, idioma, cultura y tareas avanzadas |
| `ieee_259` | A review of multimodal sensing enabled large language models for automated emotional regulation in mental health applications | IEMOCAP; DEAP; revisión de 120 publicaciones | EEG, HRV, GSR, facial, voz, IoT | Revisar y evaluar LLM multimodales para regulación emocional | Accuracy 92.7%; F1 0.91; +14.3% vs EEG unimodal; federated learning/differential privacy reduce riesgo de fuga |
| `ieee_59` | Large language model driven feature weighting for cross device harmonization of wearable data | Cohortes ETRI, MDCRC, LifeSnaps, Fitbit Versa, Actigraphy, Empatica E4; cohorte externa independiente | HR, actividad, sueño, GPS, EMA de ánimo/estrés | Armonizar datos wearable entre cohortes con pesos generados por LLM | GPT-OSS mejora F1 externo de forma más consistente; SHAP muestra que los pesos se usan y correlacionan con variables fisiológicas/conductuales |
| `scopus_484` | Graph representations for reading comprehension analysis using LLM and eye tracking biomarker | ZuCo 1.0 (12 sujetos; 21,629 palabras; 1,107 frases; 154,173 fijaciones) | EEG y eye-tracking | Analizar comprensión lectora con grafos generados por LLM y validar con fijaciones | PageRank obtiene las mejores AUC (hasta 0.651); nodos importantes reciben más fijaciones (diferencia media 0.291) |
| `arxiv_199` | Hindsight Bias in Clinical Temporal Reasoning: How Future Data Exposure Affects Large Language Model Judgment | 171 case reports de PMC OA: 40 sepsis y 131 GLP1/diabetes; TTS humano y generado por LLM | N/A (texto clínico/series temporales textuales) | Medir sesgo de retrospección en razonamiento clínico temporal | Exposición a timeline completo aumenta movimiento hacia trampas de retrospección; el enmascaramiento temporal reduce sesgo sin pérdida de accuracy |
| `pubmed_28` | Resting state fMRI foundation models enable robust and generalizable latent neural target discovery in cognitive aging interventions | ADNI (fine-tuning); ACT; CogTE; pretraining de FMs en UK Biobank/HCP | rsfMRI/BOLD | Evaluar foundation models de rsfMRI para clasificar respuesta de memoria episódica a intervenciones | BrainLM supera SVM/MLP/GCN; BrainLM-ADNI mejora ACC/F1; robusto a sitio y movimiento; RSA alinea embeddings con redes cerebrales |
| `pubmed_106` | Validating large language model assisted data extraction from clinical notes | 60 pacientes; 1,482 páginas; 29 categorías; 2,555 valores; cáncer de cabeza y cuello | N/A (notas clínicas) | Validar extracción de datos clínicos por LLM vs extracción manual | Accuracy 74–90%; tiempo 8.6 → 1.9 min; alucinaciones 0.16%; 68 errores de alto impacto |
| `ieee_651` | A communication robot for elderly persons using LLM to generate responses according to emotions estimated from physiological signals | Dos participantes mayores (80 y 90 años) en residencia de cuidado | EEG (Muse S Gen2) y HRV (Polar H10); voz | Robot conversacional que adapta respuestas a emociones estimadas | Robot propuesto reduce estrés y estimula función cerebral frente al robot previo; comunicación más activa |
| `scopus_263` | AWARE Narrator and the utilization of large language models to extract behavioral insights from smartphone sensing data | StudentSense: estudiantes universitarios durante 17 semanas; análisis de una semana de un participante; AWARE-Light | Sensado de smartphone (GPS, acelerómetro, uso de apps, teclado, etc.) | Convertir datos tabulares de smartphone en narrativas y extraer insights conductuales/psicológicos con LLM | Demuestra factibilidad; narrativas permiten resúmenes y predicción exploratoria de escalas DASS-21/I-PANAS-SF |
| `arxiv_367` | LLMs Remember First, Forget Last: Dual-Process Interference in Large Language Models | 46 categorías semánticas con 478–624 valores; 39/54 LLMs | N/A (texto/LLM) | Comparar interferencia retroactiva y proactiva en LLM | PI > RI en todos los modelos (d=1.73); tamaño predice RI pero no PI; errores distintos; alucinación <1% |
| `ieee_828` | Predicting fall risk in community dwelling older adults using a fine tuned quantized large language model | Cohorte prospectiva de 206 adultos mayores comunitarios (217 basal) a 6 meses; posturografía, cuestionarios, pruebas físicas, sensores inerciales/plataforma de presión | Posturografía, acelerometría/inercial, datos clínicos | Predecir riesgo de caída con LLM cuantizado fine-tuned (QLoRA) | Modelo I con LLM: accuracy 0.86; recall 0.92; especificidad 0.83; AUC 0.88; supera ML tradicional |
| `arxiv_317` | Neurocognitive Modeling for Text Generation: Deep Learning Architecture for EEG Data | ImageNet of the Brain (Emotiv Insight; 5 canales EEG; 3 s a 128 Hz; estímulos ImageNet ILSVRC2013) | EEG | Generar texto a partir de EEG con encoder RNN + classifier-LLM (Gemma 2B) | Accuracy 66.67% en 2 clases, 51.85% en 3 clases; perplexity 24.81–73.54 según clases; eficiente en datos |
| `ieee_449` | Validated multi agent simulation for dementia care: LLM based agents with biometric feedback | 120 interacciones simuladas cuidador-paciente; sin dataset real | HRV/EDA simulados | Simulación multiagente con LLM y biometría para validar razonamiento clínico sin circularidad | Estimación de confusión r=0.915; compliance predicha r=−0.933; asociaciones biométricas mediadas por acción (parcial r≈−0.009) |
| `pubmed_182` | Using large language models for chronic disease management tasks: scoping review | Revisión de 29 estudios; no dataset primario | N/A (revisión) | Revisar aplicaciones de LLM en manejo de enfermedades crónicas | Tareas centradas en paciente 62%; retos: respuestas inexactas 62%, ética/privacidad 35%, usabilidad 31% |
| `scopus_128` | Predicting momentary suicidal ideation from smartphone screenshots using vision language models: prospective machine learning study | 2.5 millones de capturas de pantalla de 70 participantes durante 28 días; EMA | N/A (capturas de pantalla; no fisiológica) | Predecir ideación suicida momentánea desde screenshots con VLM | Temporal holdout AUC 0.83/AUPRC 0.77; subject holdout ≈0.50; screening léxico AUC 0.62; modelos pequeños viables on-device |
| `arxiv_51` | Beyond Latent Personalization: Integrating HRV-Conditioned Personalization via LLM for Interpretable Stress Detection | TILES-2018; WESAD | HRV | Integrar HRV y atributos personales mediante interpretación semántica LLM para detección de estrés | MCC 0.261 en TILES-2018; +29.9% frente a baseline más fuerte; ablation confirma HRV, atributos y embeddings |
| `ieee_314` | MM-LLM for depression: towards an augmented intelligent diagnosis and intervention by fusing multimodal data and LLMs | Dataset 1, Dataset 2 y Dataset 3 (no nombrados en el extracto) | EEG, habla y texto | Reconocimiento de depresión, soporte de evaluación y generación de respuestas de intervención | Proof-of-concept; mejora discriminabilidad EEG; piloto pre-post con 20 participantes muestra reducción de síntomas (p<0.01), preliminar |
| `arxiv_7` | Large language models in medical time series analysis | Revisión; no dataset primario; menciona PTB-XL, MIMIC-III/IV, etc. | ECG, EEG, PPG, signos vitales, wearables | Revisar arquitecturas, prompting, aplicaciones y evaluación de MedTSLLM | Taxonomía y guías clínicas; retos de alucinación, datos, benchmarks, razonamiento mecanístico, cómputo y regulación |
| `scopus_355` | Mind the data: evaluating data quality sensitivity in medical LLMs | LQMR: Heart Disease, Heart Failure, PIMA, ILPD, Vertebral; MIMIC-IV-ED: Hospitalization outcome y Critical outcome | N/A (EHR estructurado/tabular) | Evaluar sensibilidad de LLM a datos médicos de baja calidad | AUC cae hasta 16.79% con anomalías de plausibilidad; LLM detecta mal anomalías de plausibilidad; prompting simple no mitiga |
| `ieee_401` | The impact of generative AI and large language models on telemedicine diagnostics: a systematic review | Revisión sistemática; no dataset primario | Texto, imagen, señal multimodal en telemedicina | Revisar evidencia de GenAI/LLM en diagnóstico telemedicine | Evidencia preliminar; alta accuracy en contextos estructurados y reducción de documentación; retos de alucinación, sesgo, interpretabilidad y regulación |
| `pubmed_187` | MedTsLLM: Medical Time Series Analysis Using Multimodal LLMs | No se nombran datasets en el extracto; tareas de ECG, monitoreo respiratorio y arritmias | ECG, señales respiratorias, series temporales fisiológicas | Integrar señales fisiológicas con contexto clínico textual mediante patch reprogramming y prompting contextual | Supera baselines en segmentación semántica, detección de límites, anomalías y clasificación; encoder-decoder y decoder-only tienen ventajas según tarea |

## Tabla datasets2

| Paper ID | Título | Dataset(s) | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|---|---|
| `ieee_196` | Reducing diagnostic uncertainty using large language models | **MIMIC-III** | No fisiológica; notas clínicas/EHR | Predecir categorías de sistema corporal a partir de notas clínicas de los primeros 3 días | ~80% modelos F1>0.7; varios sistemas >0.8; bajos en malformaciones congénitas, ojo y oído; con 1 día vs 3 días F1 baja pero la mayoría mantiene rendimiento satisfactorio |
| `ieee_598` | Artificial intelligence for speech classification and enhancement of speech and language disorders | Revisión sistemática: **TORGO, UA-Speech, PC-GITA, SVD/Saarbruecken, LAANA, Fluency Bank, SEP-28k, KSoF, MEEI, NEMOURS, TCC-300, NCMMSC2021, TIMIT, LibriSpeech** | Habla/audio (acústica, prosódica, lingüística); no señal fisiológica única | Revisar IA/ML/DL para clasificación y mejora de trastornos del habla y lenguaje | Síntesis de 231 estudios; progresión SVM→DL; brechas en datasets longitudinales, multilingües y explicabilidad; mejora del habla como preprocessing |
| `scopus_80` | Improving dietary supplement information retrieval: RAG with LLMs | **iDISK2.0**; fuentes **NMCD, MSKCC, DSLD, LNHPD**; preguntas derivadas de **MSKCC** | No fisiológica; texto/knowledge graph | RAG sobre knowledge base de suplementos dietéticos para QA | RAG >95% accuracy; GPT-4o 62% y 52% en true/false y MCQ de interacciones; reduce hallucinations |
| `scopus_293` | A survey on large language models for medical time series | Survey: **MIMIC-III, MIMIC-IV, eICU-CRD, HiRID, AmsterdamUMCdb, OhioT1DM, VitalDB, WESAD, PPG-DaLiA, TROIKA, QT Database, MIT-BIH Arrhythmia, Apnea-ECG, Sleep-EDF, CHB-MIT**; PhysioNet benchmark | ECG, EEG, PPG/BVP, signos vitales, wearables | Revisar LLMs para series temporales médicas | Taxonomía, estrategias de prompting/evaluación, retos de datos, multimodalidad, eficiencia, interpretabilidad y privacidad |
| `arxiv_190` | Towards a Personal Health Large Language Model | **857 case studies** sleep/fitness; **629 MCQs sueño**; **99 MCQs fitness**; **Google Digital Wellbeing Study** (4.759; filtrado 4.163) con **PROMIS Sleep Disturbance/Impairment** | Wearables: sueño, actividad, HR, HRV, respiración, pasos | Fine-tuning de Gemini para salud personal (sueño/fitness) | PH-LLM 79% sueño y 88% fitness; rendimiento cercano a expertos; PROs con adapter comparable a modelos especializados |
| `scopus_226` | Structural priming in humans and large language models | Estímulos dativos en inglés: **64 verbos, 256 oraciones, 128 pares prime-target, 396 fillers**; **EEG de 27 participantes**; **norming con 60 participantes**; suite de 13 LMs | EEG y eye-tracking | Comparar priming estructural humano vs LMs | Ambos muestran priming; LMs más fuerte en DO; predicción ítem-a-ítem limitada; ningún LM predice consistentemente el priming humano |
| `arxiv_70` | Large Language Models are Few-Shot Health Learners | **MIT-BIH Atrial Fibrillation Database; MIMIC-III; PAMAP2; dataset sintético metabólico; Fitbit longitudinal (stress EMA/PHQ-8)** | ECG/IBI, acelerómetro, HR/sueño/pasos de Fitbit | Few-shot prompt tuning para tareas de salud | Mejora sobre zero-shot y baselines supervisados; contexto en prompt mejora; zero-shot pobre; hasta 130% mejora en cardiovasculares |
| `ieee_271` | Developing a RAG agent for personalized fitness and dietary guidance | **10 PDFs** de dieta/ejercicio; datos de wearables “en tiempo real” previstos, sin dataset fisiológico nombrado | No fisiológica; voz/texto; wearable previsto | RAG para fitness/dieta personalizada | Relevancia ~80–90%; faithfulness 40% (F1 57%); latencia 2–3 s; voz 85%; memoria 95% |
| `arxiv_163` | Large language models surpass domain-specific architectures for antepartum electronic fetal monitoring analysis | **Oxford Maternity (OxMat)**: 23.127 segmentos train; 2.500 eval; 4.000 limited | CTG: frecuencia cardíaca fetal (FHR) y actividad uterina (UA) | Benchmark de modelos para clasificación CTG APO/NPO | Llama-3B fine-tuned mejor AUC 0.852; data-limited 0.800; sin UA, 1D SE-ResNet 0.815; LLMs requieren mucho más cómputo |
| `pubmed_25` | Low frequency neural activity tracks syntactic information through semantic mediation | **EEG Mandarin frequency-tagging**: 25 participantes; 4 condiciones; 30 trials/cond | EEG | Examinar tracking sintáctico mediado por semántica y simulación GPT-2 | Picos 1 Hz en antónimos, living-nonliving y NV-sensical; no en NV-nonsensical; 2 Hz en todas; GPT-2 con limitaciones |
| `scopus_180` | Guiding LLMs to decode text via aligning semantics in EEG signals and language | **ZuCo benchmark** (ZuCo 1.0 + 2.0) | EEG; eye-tracking para segmentar palabras | EEG-to-text como generación controlada con re-ranking por LLM | SOTA en multi-sujeto, single-sujeto y cross-subject; mejores BLEU/ROUGE/BERTScore que baselines |
| `scopus_271` | EEG-informed adaptive UX questionnaire generation with multi-agent LLMs | **EEG DSI-24**: 6 participantes; 4 condiciones visuo-hápticas; 24 trials | EEG | Generar cuestionarios UX adaptativos con multi-agent LLMs | G-Eval 80.96; +45.32% sobre baseline single-stage |
| `arxiv_276` | Integrating LLM, EEG and eye-tracking biomarker analysis for word-level neural state classification | **ZuCo Task 3**: 12 sujetos; 407 oraciones; 356 usadas | EEG + eye-tracking | Clasificar HRW vs LRW con etiquetas LLM | >60% validation accuracy; HRW más fijaciones; primer intento de clasificación word-level con LLM |
| `pubmed_178` | Sleep disorder analysis using large language models | **CAP Sleep Database; Sleep Accel Dataset; Sleep Bioradiolocation Database; Sleep-EDF Expanded; Sleep Heart Health Study PSG; HMC Sleep Database; Charité University Hospital Berlin; NCH Sleep DataBank** | EEG/PSG, stages de sueño | Fine-tuning BERT/Longformer para clasificar sueño sano/no sano | Accuracies 70–80%; BERT mejor en secuencias cortas, Longformer en largas; overfitting; no clasifica trastornos específicos |
| `scopus_69` | 2-D GAF-enhanced multimodal vision language model for breathing patterns analysis via ECG sensing | **Long-Term Infant Cardio-Respiratory Signals DB; Stress Recognition in Automobile Drivers DB; Fantasia DB; Combined ECG/breathing/seismocardiogram; ECG spider video clips; UoS ECG test** | ECG | Clasificar patrones respiratorios con VLLM sobre GAF 2D | Idefics BLEU 0.572 (2D) vs 0.549 (1D); Florence 0.626 en 1D; GAF mejora visualización |
| `scopus_50` | Effectiveness, safety, and workflow burden of LLM-based medical report generation: systematic review | Revisión: **101 estudios**; menciona **MIMIC-CXR** y corpus públicos | Imágenes médicas (X-ray, CT, MRI, US, endoscopia, patología, ECG); no señal fisiológica única | Revisar efectividad, seguridad y carga de trabajo | Sin estudios bajo riesgo bajo; 15 moderado, 72 alto, 14 serio; evidencia heterogénea/no poolable; certeza muy baja |
| `ieee_205` | Using generative AI for neurofeedback content personalization | **OpenNeuro ds002718, ds002778, ds003064** | EEG | Framework LLM para neurofeedback personalizado | Framework propuesto; experimentos preliminares muestran eficacia en selección de features y diferenciación de casos |
| `scopus_39` | Leveraging LLMs for automated feature extraction and model training in EMG-based motion decoding | **EMG in-hand manipulation dataset**: 11 participantes; 16 músculos; 3 objetos/movimientos | EMG | Automatizar extracción de features y entrenamiento con LLM | Rendimiento comparable a código manual; sin diferencia significativa (p=0.16); pequeñas desviaciones por precisión/parámetros |
| `pubmed_168` | Integrating large language model, EEG and eye tracking for word-level neural state classification | **ZuCo 1.0 Task 3** | EEG + eye-tracking | Clasificar relevancia de palabras con etiquetas LLM | >60% accuracy; HRW más fijaciones; primer uso de LLM para etiquetas word-level |
| `pubmed_41` | The effects of maturation and dyslexia risk on neural speech sound encoding and discrimination at preschool stage | **DyslexiaBaby project**: 146 a 28 meses; 143 a 4–5 años | EEG/ERP/MMR | Estudiar maduración y riesgo de dislexia | P1–N2; MMN+LDN; riesgo de dislexia con MMN reducida; sin diferencias en ERPs obligatorios |
| `scopus_500` | Lightweight vision language model guided gesture recognition based on electromyography | **EMG pretraining dataset** (33 sujetos); **4 nuevos participantes**; **Flickr30k** para VLM | EMG + visión | Pseudolabeling con VLM para decodificación EMG | 100% decoding accuracy en 4 gestos/4 participantes bajo distribution shift; VLM reliability 1 vs MVC 0.8 |
| `arxiv_77` | Fine-tuning LLMs using EEG microstate features for mental workload assessment | **Zyma et al.** (36 sujetos) + **Shin et al.** (29 sujetos); paper reporta 103 sujetos | EEG microstates | Fine-tuning LLM para carga cognitiva Rest/Load | Accuracy 97% tras fine-tuning vs 4.5% antes; ~24× mejora |
| `scopus_118` | Multimodal large language models in human-centered health: practical insights | **Ocho datasets públicos de sensores inerciales**; se nombran **mHealth, DLA, ComplexHAD, DSADS, FLAAP, RealWorld, PAMAP2**; el octavo no se nombra en el extracto | Acelerómetro/giroscopio wearable | Dual-view FM para forecasting de señales inerciales | Supera TimeGPT-1 hasta 36% MAE; fine-tuning >7% MAE; mejora clasificación downstream |
| `pubmed_157` | Integrating NLP into radiation oncology: a practical guide to transformer architecture and LLMs | No dataset primario; revisión | No fisiológica; texto clínico | Revisar NLP/LLM en oncología radioterápica y proponer framework | Framework de evaluación: propósito, fit clínico, rendimiento, sesgo/confianza, legal/ético, QA |
| `acl_anthology_65` | LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models | **LENS dataset**: 258 participantes; 50.957 EMAs; >100.000 pares sensor-texto; Garmin vivoactive 3 + Android | GPS, pasos, acelerómetro ZCR/energía, conversación, unlock, HR, sueño, estrés | Alinear sensores con LLM para narrativas clínicas de salud mental | LENS supera baselines en métricas NLP y alineación clínica; expertos valoran narrativas; eficiencia token ~94% menor que texto |
| `acl_anthology_245` | Speech disfluencies and LLM confidence: length bias and pragmatic insensitivity in Brazilian Portuguese | **Roda Viva corpus**: 344 turnos | Habla/transcripciones (no fisiológica) | Estudiar confianza LLM vs incertidumbre pragmática | Longitud de turno domina confianza (β=+14.47); marcadores pragmáticos no significativos; ECE 41.95/41.14 |
| `wos_308` | A personal health large language model for sleep and fitness coaching | **PH-LLM case studies**; **629 MCQs sueño**; **99 fitness**; **Google DWB/PROMIS**; 833 survey responses; ratings 4.265/5.049 y 2.606/3.335 | Wearables: sueño, actividad, HR, HRV, respiración | Evaluar PH-LLM en exámenes, coaching y PROs | 79% sueño, 88% fitness; case studies cercanos a expertos; PROs con adapter comparable a modelos especializados |
| `acl_anthology_34` | Can LLMs understand unvoiced speech? Exploring EMG-to-text conversion with LLMs | **Gaddy & Klein unvoiced EMG**: 1 hablante, 8 canales, 67 palabras, ~26 min, 500 utterances; **CSL-EMG_Array** para person ID | EMG no vocalizado | Convertir EMG no vocalizado a texto sin audio/voz | WER 0.49; con 6 min mejora ~20% a modelos especializados; person ID 96% |
| `wos_85` | Enabling dietary assessment with smart wearables: a passive, scalable approach for precision nutrition | **Estudio de viabilidad**: 30 participantes UK; Hammersmith y Reading; 8 días; cámaras wearable; menú de referencia dietista | Imágenes de cámara wearable (no fisiológica) | Evaluar pipeline LVLM para evaluación dietética pasiva | Recall de alimentos 82%; MAE porción 44.7 g alimentos y 70.4 mL bebidas |
| `wos_403` | Evaluation of LLMs for structured data extraction from interstitial lung disease clinical notes | **Stanford ILD clinic notes** 2018–2022: 10 prompt engineering + 100 validación; 12 LLMs | No fisiológica; notas clínicas | Extraer datos binarios y multiclase de notas ILD | 7/12 modelos a nivel humano (96.2%); gpt-oss-120b 97.5%; multiclase 91.1%/88.0% |
| `acl_anthology_148` | Do language models mirror human confidence? Exploring psychological insights to address overconfidence in LLMs | **MMLU** (high school/college; physics, chemistry, biology) y **GPQA** (expert) | No fisiológica | Analizar calibración/confianza y proponer AFCE | AFCE reduce ECE; LLMs menos sensibles a dificultad; sesgos por personas demográficas; GPT-4o más balanceado |
| `acl_anthology_303` | Can a remedy find a researcher? Exploring semantic knowledge in Italian BabyLMs | **Benchmark italiano de pares mínimos**: 272 AI, 332 AA, 282 AA-rev; validación con 232 participantes; modelos BAMBI y LMs italianos | No fisiológica | Evaluar competencia semántica de BabyLMs italianos | BabyLMs sensibles a violaciones fuertes; calibración baja ECE; datos child-directed algo mejores que web |
| `acl_anthology_231` | ClinDet-Bench: Beyond abstention, evaluating judgment determinability of LLMs in clinical decision-making | **ClinDet-Bench**: 16 sistemas de scoring; 188 casos; 8 LLMs | No fisiológica | Evaluar determinabilidad bajo información incompleta | LLMs fallan: conclusiones prematuras y abstención excesiva; imputan información faltante (75.9% errores) |
| `acl_anthology_286` | Vulnerability of LLMs’ stated belief? LLMs belief resistance check through strategic persuasive conversation interventions | **BoolQ; PubMedQA; LatentHatred** | No fisiológica | Evaluar vulnerabilidad a persuasión multi-turno | Vulnerabilidad alta y desigual; confianza verbalizada aumenta vulnerabilidad; fine-tuning adversarial dependiente del modelo |
| `acl_anthology_169` | Vers une évaluation rigoureuse des systèmes RAG : le défi de la due diligence | **Dataset RAG due diligence**: 300 documentos DataCorp; 121 preguntas; 20 respuestas/pregunta; anotaciones humanas/LLM; público `gmartinonQM/eval-rag` | No fisiológica | Evaluar RAG industrial con PPI/human+LLM judge | Métricas automáticas y PPI; LLM-Judge sobreestima pertinencia; acuerdo bajo hace marginal la ganancia de PPI |
| `wos_327` | Integrating NLP into radiation oncology: a practical guide to transformer architecture and LLMs | No dataset primario; duplicado de `pubmed_157` | No fisiológica; texto clínico | Revisar NLP/LLM en oncología radioterápica | Framework de evaluación clínica; mismos hallazgos que `pubmed_157` |
| `wos_262` | Large language models in mental health: a systematic review of applications, innovations and ethical challenges | Revisión: **PHQ-9**, datasets de redes sociales, notas clínicas; 92 estudios finales | Texto/social media; no fisiológica única | Revisar aplicaciones LLM en salud mental | Aplicaciones en detección, agentes conversacionales, apoyo terapéutico; retos de datos, sesgo, privacidad, validación clínica |

 

# Datasets utilizados en los papers de `gemini_notebook_03.md`

A continuación se presenta la extracción estructurada de **todos los datasets** utilizados en los papers contenidos exclusivamente en `gemini_notebook_03.md`. Se organiza por paper, indicando para cada dataset: nombre, tipo de señal fisiológica tratada, objetivo del paper y resultado principal.

> **Nota:** Varios papers de este documento **no utilizan señales fisiológicas** (son puramente texto/visión/lenguaje). Se indica explícitamente con "— (sin señal fisiológica)".

---

## 1. acl_anthology_73 — *Gladiator at MEDIQA-SYNUR 2026: Contextual Clinical Extraction Integrating Foundation Models with Domain-Specific Validation Rules*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **MEDIQA-SYNUR shared task dataset** (122 train, 101 dev, held-out test) | — (texto clínico: dictados de enfermería) | Extracción híbrida de información clínica guiada por esquema (193 conceptos, 4 tipos de valor) | **74.27% precision, 84.32% recall, 78.98% F1** en dev set |

---

## 2. wos_200 — *The shrinking landscape of linguistic diversity in the age of large language models*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Reddit r/WritingPrompts** (N=318,490 historias) | — (texto) | Medir homogenización lingüística por LLMs | Reducción significativa de varianza en complejidad de escritura |
| **Patch News** (N=379,583 artículos) | — (texto) | Ídem | Efecto más débil pero medible |
| **arXiv preprints** (N=80,238 abstracts CS) | — (texto) | Ídem | Declive sostenido post-ChatGPT (POST β=−0.0014, p<.001) |
| **Essays dataset** (N=2,348) | — (texto) | Evaluar traits de personalidad desde texto | Reducción predictiva tras reescritura por LLM |
| **YourMorals** (N=3,641) | — (texto) | Valores morales | Asociaciones léxicas erosionadas |
| **Empathic Conversations** (N=711) | — (texto) | Empatía | Ídem |
| **Congress** (N=710) | — (texto) | Demografía | Ídem |

**Resultado global:** reducción de 21–50% en varianza de complejidad de escritura (p≤.05).

---

## 3. wos_386 — *A scoping review of large language models in personal wellness*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **21 estudios revisados** (sleep education, OSA, CBT-I, coaching) | Señales de sueño/wearables (Fitbit, Oura, PSG) | Revisión scoping de LLMs en bienestar del sueño | Evidencia preliminar; evaluaciones a corto plazo, expert-rated o proxy-based |

> **Nota:** Es una revisión; no introduce dataset propio. Menciona WESAD, PSQI, FER-2013, etc., en los estudios revisados.

---

## 4. acl_anthology_118 — *Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **TriviaQA** | — (texto QA) | Detectar y mitigar hallucinations calibrando incertidumbre verbal | Reducción relativa media de ~30% en hallucinations confiadas |
| **NQ-Open** | — (texto QA) | Ídem | Mejora en detección (AUROC) y calibración |
| **PopQA** | — (texto QA) | Ídem | Ídem |

---

## 5. wos_230 — *Leveraging large language models for automated psychological analysis in electronic sandboxes*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Electronic sandbox data** (60 participantes planificados, 18–35 años) | — (imágenes de sandbox, logs de interacción, movimiento de ratón) | Análisis psicológico automatizado integrando LLMs + visión por computador | Sistema propuesto; validación experimental futura (sin resultados aún) |

---

## 6. acl_anthology_72 — *Multimodal Affective Modeling in an LLM-based Intelligent Tutoring System for Foreign Language Learning*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **EmoBot proof-of-concept** (sin dataset público nombrado) | Audio, vídeo, biosignales (HR, SpO₂, HRV desde smartwatch) | Reconocimiento multimodal de emociones en aprendizaje de alemán | Audio: 81.3% accuracy en habla alemana; Visual: ~66% en FER-2013 (nivel humano 65±5%) |

---

## 7. acl_anthology_2 — *Psycholinguistic Diagnosis of Language Models' Commonsense Reasoning*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Singh et al. (2016) stimuli** (82 items de presuposición) | — (behavioral/EEG-ERP referenciado, no recolectado) | Evaluar razonamiento pragmático de LMs (presuposición) | DistillBERT detecta Presp; fine-tuning no mejora |
| **Nieuwland et al. (2010) data** (168 items de implicatura escalar) | — (EEG/ERP, N400 referenciado) | Evaluar implicatura escalar | Fine-tuning mejora SI significativamente |
| **ImpPress dataset** (>25k pares; 14,100 Presp + 6,000 SI) | — (texto) | Fine-tuning de DistillBERT | Mejora en SI; sin mejora en Presp |

**Resultado global:** GPT-3 ≈ azar; DistillBERT con cierta comprensión de intenciones implícitas.

---

## 8. acl_anthology_288 — *When LLMs Disagree with Human Experts: Understanding LLM Annotation Failures in Nutrition Misinformation*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **169 posts de Instagram sobre seed oils** (anotados por 3 nutricionistas, Fleiss' κ=0.813) | — (texto de redes sociales) | Evaluar LLMs como anotadores de desinformación en salud | Taxonomía jerárquica de errores; sobreconfianza sistemática (149/169 errores con confianza ≥0.8) |

---

## 9. acl_anthology_57 — *Teaching LLMs to Abstain across Languages via Multilingual Feedback*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **M-MMLU** (Multilingual MMLU) | — (texto QA) | Abstención multilingüe vía feedback en lenguajes relacionados | Hasta **9.2% mejora** en lenguajes de bajos recursos; reducción de Gini (equidad) 12.9% |
| **M-Hellaswag** | — (texto QA) | Ídem | Ídem |
| **Belebele** (apéndice) | — (texto QA) | Ídem | Ídem |

---

## 10. acl_anthology_56 — *Decompose and Compare Consistency: Measuring VLMs' Answer Reliability via Task-Decomposition Consistency Comparison*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **SNLI-VE** | — (visión-lenguaje) | Medir fiabilidad de respuestas de VLMs | DeCC mejora Brier Score y Effective Reliability |
| **VCR** | — (visión-lenguaje) | Ídem | Ídem |
| **A-OKVQA** | — (visión-lenguaje) | Ídem | Ídem |
| **Winoground** | — (visión-lenguaje) | Ídem | Ídem |
| **MMMU** | — (visión-lenguaje) | Ídem | Ídem |
| **MathVista** | — (visión-lenguaje) | Ídem | Ídem |

**Resultado global:** reducción relativa media de Brier Score de 8.7% (LLaVA), 14.3% (Idefics2), 2.9% (InternVL); aumento de Effective Reliability 16.5%, 25.6%, 1.7% respectivamente.

---

## 11. acl_anthology_243 — *Beyond Toxic Positivity: Interpersonal Affect Regulation in LLM-Based Dialogue Agents using Discourse Politeness Theory*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Simulación longitudinal de 8 semanas** (18 observadores terceros) | — (diálogo texto; sin señal fisiológica) | Regulación afectiva interpersonal vía Discourse Politeness Theory | 72% seguridad psicológica vs 22% baseline; 72% preferencia por modelo propuesto |

---

## 12. pubmed_123 — *DataAtlas: automatic generation of data dictionaries using large language models*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **ORCHID** (64 columnas, PhysioNet) | — (datos tabulares clínicos) | Generación automática de diccionarios de datos | GPT-4 preferido sobre documentación oficial en 46.9% de comparaciones |
| **CarpeDiem** (101 columnas, PhysioNet) | — (datos tabulares UCI) | Ídem | GPT-4 preferido en 60.4% |
| **INSPIRE** (62 columnas, PhysioNet) | — (datos tabulares perioperatorios) | Ídem | GPT-4 preferido en 95.2% |

**Resultado global:** text-to-SQL execution accuracy de **0.52 → 0.88** (+35.7%).

---

## 13. ieee_430 — *Speech recognition in a smart home: some experiments for telemonitoring*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **HIS (Health Smart Home) Grenoble** (47 m², 8 micrófonos) | Audio/voz (no señal fisiológica corporal, pero señal acústica biomédica) | Reconocimiento de voz y sonidos para telemonitoreo de personas mayores | Global Error Rate **15.6%** (10 sujetos, 429 frases) |
| **Normal/Distress corpus** (126 frases en francés: 66 normales, 60 de distress) | Audio/voz | Detección de situaciones de distress por keywords | MAR 27% con echo-cancellation en tiempo real (4 hablantes) |

---

## 14. scopus_198 — *Multimodal LLMs for health grounded in individual-specific data*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **UK Biobank** (50,000 registros; 36,115 train / 6,464 val / 7,421 test de 187 pacientes) | Datos tabulares clínicos + **spirogram** (time-series de función pulmonar) | Estimación de riesgo de enfermedad con LLM multimodal (HeLM) | **AUROC 0.75 para asma** con tabular+spirogram vs 0.49 solo tabular; supera o iguala a LogReg/XGBoost en 8 traits |

---

## 15. pubmed_177 — *The first step is the hardest: pitfalls of representing and tokenizing temporal data*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Sample mobile sensing dataset** (case study con tokenizador OpenAI) | Time-series de wearables (genérico) | Analizar retos de tokenización de datos temporales en LLMs | Case study: tokenizadores dividen timestamps y valores en tokens no significativos; se aboga por prompt tuning, model grafting y tokenizadores mejorados |

---

## 16. scopus_207 — *Exploring the feasibility of generating physical prescriptions using large language models with motion tracking inputs*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **30 testers** (15 pacientes clínicos + 15 voluntarios sanos) | **Motion tracking / gait analysis** (Vicon: 24 cámaras ópticas, IMU, marker-less) | Generación de prescripciones físicas personalizadas con LLM + datos cinemáticos | Expertos puntúan LLM mejor que humano en 6/10 casos; MAE tiempo de generación 46s vs 35min humano; el LLM **no interpreta bien** datos cinemáticos sin fine-tuning |

---

## 17. scopus_272 — *LLM-assisted explainable daily stress recognition: physiologically grounded threshold rules from PPG features*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Can et al. (2020) multimodal physiological database** (14 estudiantes, TSST + 12h/día durante 1 semana, ~1000h, 388 EMA) | **PPG, HRV (RMSSD, HF, pNN50, LF/HF, stdV), EDA, temperatura cutánea, aceleración** (Empatica E4) | Detección de estrés explicable con reglas de umbral justificadas por LLM | **~75% accuracy binaria** (75.4%); comparable a CNN/LSTM/Transformer (75.8–78.1%) pero totalmente interpretable |

---

## 18. ieee_392 — *NarratingSleep: a language model based framework for sleep staging*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Sleep-EDF SC-EDF-20** (20 sujetos) | **EEG** (PSG) | Sleep staging usando LM preentrenado (cross-modal transfer) | **ACC 85.72%, MF1 81.65** |
| **Sleep-EDF SC-EDF-78** (78 sujetos) | **EEG** (PSG) | Ídem | **ACC 82.35%, MF1 77.11** |
| **ISRUC-Sleep** (100 sujetos) | **EEG** (PSG) | Ídem (dataset desbalanceado) | SOTA con solo 35 épocas (vs 100–300 de baselines) |

---

## 19. scopus_176 — *A review of the emergence of AI-driven cognitive digital phenotyping: multimodal sensing, foundation models and the path toward digital twins for human cognition*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Revisión** (sin dataset propio) | Behavioral, neurophysiológico (EEG, fNIRS, fMRI), smartphone, wearables | Revisión y hoja de ruta hacia gemelos digitales cognitivos (CDT) | Propone pipeline CDP→CDT con FMs/VLMs; identifica retos de estandarización, simulación en tiempo real y privacidad |

---

## 20. scopus_149 — *Agent-based large language model system for extracting structured data from breast cancer synoptic reports: a dual validation study*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **864 casos sintéticos** (validación controlada) | — (texto de patología) | Extracción estructurada de datos de informes de cáncer de mama (229 campos CAP) | Accuracy sintética **93.8–99.0%** |
| **90 informes de patología reales** (6,651 campos anotados) | — (texto de patología) | Ídem (validación real) | Recall real **61.8–87.7%** (gemini-2.5-pro el mejor con 87.7%); "reality gap" de 11–32 pp |

---

## 21. scopus_396 — *Understanding uncertainty in large language model predictions of early death in critically ill patients: a conformal prediction approach*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **MIMIC-III** (4,714 admisiones únicas; 910 muertes; cohorte balanceada final 1,354) | — (notas clínicas de texto de UCI) | Predicción de muerte intrahospitalaria con GPT-4o zero-shot + conformal prediction | Cobertura empírica **90.4%** (global), 99.7% muerte / 81.1% supervivencia; recall muerte 93%, supervivencia 13% |

---

## 22. scopus_68 — *Integrating cognitive processing signals into language models: a review of advances, applications, and future directions*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **Revisión** (ZUCO dataset: EEG+ET; múltiples corpora ET) | **Eye-tracking (ET), EEG, fMRI** | Revisión de integración de señales cognitivas en LMs/MLLMs | Identifica ET como proxy prometedor para data augmentation, alineación humana y mitigación de hallucinations |

---

## 23. acl_anthology_8 — *Evaluating Subjective Cognitive Appraisals of Emotions from Large Language Models*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **COVIDET-APPRAISALS** (241 posts de Reddit; 24 dimensiones de appraisal; 21 aplicables) | — (texto) | Evaluar capacidad de LLMs para predecir cognitive appraisals y generar rationales | ChatGPT: ρ=0.388, MAE=1.694; open-source LLMs (Alpaca, Dolly) con correlaciones negativas; ChatGPT ≈ anotadores humanos en calidad de rationales |

---

## 24. scopus_460 — *Harnessing LLM for noise-robust cognitive diagnosis in web-based intelligent education systems*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **ASSIST0910** (2,485 estudiantes, 16,818 ejercicios, 263,209 logs) | — (logs de respuesta, no fisiológico) | Diagnóstico cognitivo robusto a ruido con LLM + difusión | DLLM-NCDM: ACC 74.46%, AUC 77.92%, F1 82.06% |
| **Junyi1808** (2,619 estudiantes, 18,047 ejercicios) | — (logs) | Ídem | DLLM-NCDM: ACC 78.91%, AUC 77.52% |
| **Eedi50** (4,918 estudiantes, 948 ejercicios, 1.38M logs) | — (logs) | Ídem | DLLM-CDMFKC: ACC 72.58%, AUC 79.60%, F1 74.13% |

---

## 25. wos_399 — *Multimodal large language models in medical research and clinical practice: development, applications, challenges and future*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **MIMIC-CXR** (377k imágenes CXR + 227k informes) | Imagen médica (rayos X) | Revisión de MLLMs médicos (M³FM, Med-Gemini, MUSK, PathChat, GPT-4V(Med), Med-MLLM) | Revisión; M³FM: BLEU-4 38.8% zero-shot chino; Med-Gemini: 91.1% USMLE; MUSK: C-index 0.747 pancáncer |
| **COVID-19-CT-CXR** (1,000 imágenes) | Imagen médica | Ídem | Ídem |
| **IU-Xray, COVID19-CT, NIH Chest X-ray, CheXpert, Shenzhen TB** | Imagen médica | Ídem | Ídem |
| **MedQA (USMLE), NEJM CPC, GeneTuring, Slake-VQA, Path-VQA, ECG-VQA, MIMIC-III** | Imagen + **ECG** | Ídem | Med-Gemini: 57.7% en arrhythmia classification ECG |

---

## 26. manual_1 — *WearableQA: A Benchmark for Health Reasoning over Real-World Wearable Data*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **WearableQA benchmark** (200 usuarios reales, 4,084 preguntas MCQ de 10 opciones, hasta 500 días de datos) | **Wearable time-series** (16 métricas diarias: RHR, HRV, VO₂max, pasos, calorías, sueño, estrés), **blood biomarkers** (17), demographics | Benchmark de razonamiento en salud sobre datos wearables reales | 14 LLMs: accuracy **19.6%–72.9%** (baseline azar 10%); Gemini-3.1-Pro el mejor (72.9%); data reasoning más difícil que health reasoning |

---

## 27. manual_2 — *When Derived Measurements Mislead: Quantifying and Mitigating LLM Over-Trust with Privileged-Modality Reliability Evidence*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **MIMIC-III Matched Waveform Database v1.0** (50,000 registros PPG–ECG sincronizados de 1,275 pacientes; 36,115 train / 6,464 val / 7,421 test de 187 pacientes) | **PPG + ECG** (ritmo cardiaco) | Definir y cuantificar Derived-Feature Over-Trust (DFOT); 5 estimandos (COTR, CIR, CRR, ESRM, UHR) | K2 vs B2: mejoras de **1.82–6.69 pp** en 4 endpoints de repair/specificity; todos los IC pareados excluyen cero; UHR +0.67 pp (IC −0.4 a +1.7) |

---

## 28. manual_3 — *PhysioBench: A Unified Benchmark for Physiological Signal Question Answering*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **PhysioBench** (22 datasets públicos, **61,418,821 preguntas**, 30 tareas, 240,171 sujetos) — PSG 74.76%, ECG 23.19%, EEG 1.90%, PPG 0.15% | **ECG, PPG, EEG, PSG** | Benchmark unificado de QA sobre señales fisiológicas | 21 modelos evaluados: **ninguno robusto**; zero-shot LLMs/VLMs/TSLMs cerca o bajo azar; PSFMs con encoder congelado superan azar pero con alta heterogeneidad por tarea |

---

## 29. manual_4 — *An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **PPG–ECG dataset** (46 participantes; 12 held-out, 621 grabaciones de 15 min; Samsung Gear Sport + Shimmer3) | **PPG + ECG** (frecuencia cardiaca) | Agente LLM (OpenCHA, GPT-3.5-turbo) orquestando pipeline de estimación de HR desde PPG | **MAE 2.83, RMSE 5.47, MAPE 0.04, MAD 1.90** — vs GPT-4o (MAE 8.93) y GPT-4o-mini (MAE 9.41); outliers: 0.08% vs 6.20%/17.55% |

---

## 30. manual_5 — *Telemetry degradation and sequential stability in LLM-integrated IoT decision systems*

| Dataset | Señal fisiológica | Objetivo | Resultado |
|---|---|---|---|
| **AI-tennis coaching case study** (telemetría sintética de smartwatch, 40 features IMU, ventanas de 20 golpes) | **IoT telemetry / smartwatch IMU** (biomecánica) | Evaluar estabilidad conductual de LLM como capa de decisión bajo degradación de telemetría (dropout, downsampling, reliability signaling) | FlipRate: **0.144 (clean) → 0.207 (downsampling, +44%)** en régimen balanceado; **0.300** en régimen serve-dominant; ViolationRate=0 (modelo open-loop); structured prompting mitiga parcialmente |

---

# Resumen de cobertura por tipo de señal fisiológica

| Señal fisiológica | Papers que la utilizan |
|---|---|
| **PPG / BVP** | scopus_272, manual_2, manual_3, manual_4 |
| **ECG** | scopus_198 (no directa), manual_2, manual_3, manual_4, wos_399 (ECG-VQA) |
| **EEG / PSG** | ieee_392, manual_3, acl_anthology_2 (referencia), scopus_68 (revisión) |
| **HRV (derivada de PPG/ECG)** | scopus_272, manual_1 |
| **EDA / temperatura / aceleración** | scopus_272, manual_1 |
| **Audio / voz** | ieee_430, acl_anthology_72 |
| **Motion tracking / IMU / gait** | scopus_207, manual_5 |
| **Eye-tracking** | scopus_68 (revisión) |
| **Imagen médica** | wos_399 |
| **Sin señal fisiológica (texto/visión/tabular)** | acl_anthology_73, wos_200, wos_386, acl_anthology_118, wos_230, acl_anthology_72 (parcial), acl_anthology_288, acl_anthology_57, acl_anthology_56, acl_anthology_243, pubmed_123, pubmed_177, scopus_149, scopus_396, acl_anthology_8, scopus_460 |

