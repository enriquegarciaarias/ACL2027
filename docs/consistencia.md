# Comprobación de consistencia fisiológica (P1 de Fase 2)

**Fecha:** 2026-11-09
**Módulo:** `sources/phase2/consistency.py`
**Scripts:** `scripts/analyze_consistency.py`, `scripts/run_e12_pilot.py --consistency`
**Reporte:** `results/output/WESAD/PHASE2/e12/consistency_gate_report.json`

---

> **Aviso (2026-11-09):** el mecanismo de la compuerta no depende del prompt y
> sigue siendo válido; las métricas del LLM que lo acompañan proceden de un
> corpus con few-shot contaminado y son exploratorias (ver
> `docs/split_dev_test.md`).

## 1. Problema que resuelve

La vista E12 (z-scores intra-sujeto) es la mejor representación para el endpoint
binario limpio (accuracy 93,0 %, AUROC stress 0,958), pero pierde sensibilidad a
la corrupción estructural: en `NC_shuffled` la vista absoluta se abstiene el
95,3 % de las veces y la centrada solo el **23,6 %**, emitiendo 534 decisiones
confiadas sobre datos internamente incoherentes.

La causa es doble:

1. La estandarización por sujeto **destruye las relaciones aritméticas** entre
   features (las expresa como desviaciones independientes).
2. El LLM **aplica umbrales a valores individuales** y no comprueba la
   coherencia conjunta (EDA alta + SCR incoherente sigue leyéndose como estrés).

## 2. Idea: relaciones deterministas entre features

Varias features de una ventana no son independientes:

| Comprobación | Relación | Tipo |
|---|---|---|
| `ecg_hr_vs_rr_mean` | HR = 60000 / RR_mean | identidad exacta |
| `ecg_n_beats_vs_hr` | n_beats ≈ HR × duración | identidad exacta |
| `eda_scr_rate_vs_count` | SCR_rate ≈ SCR_count / duración | identidad exacta |
| `eda_scr_amplitude_mean_vs_max` | amplitud media ≤ amplitud máxima | desigualdad |
| `eda_mean_vs_tonic` | EDA media ≈ EDA tónica | identidad aproximada |
| `resp_n_cycles_vs_rate` | n_ciclos ≈ FR × duración | empírica (tol. amplia) |
| `temp_range_vs_std` | rango ≥ 2 × DE | empírica |
| `eda_std_vs_phasic_std` | DE total ≥ DE fásica | empírica |

Barajar valores dentro de una modalidad conserva la distribución marginal pero
**rompe estas relaciones**. Cinco comprobaciones son `hard` (un fallo es
imposible en datos reales) y tres son `soft` (un fallo aislado puede ser
legítimo; varios juntos indican corrupción).

La compuerta (`should_abstain`) se abstiene si:

- falla alguna comprobación `hard`, o
- fallan más de una `soft`, o
- falta alguna feature clave (`ecg_hr_bpm`, `ecg_n_beats`, `eda_scr_count`,
  `eda_scr_rate_per_min`, `resp_rate_bpm`, `resp_n_cycles`).

No usa etiquetas, ni SQI, ni cobertura: solo los valores de la propia ventana.

## 3. Resultados sobre el corpus (705 ventanas, 15 sujetos)

Compuerta por defecto (`max_hard=0`, `max_soft=1`, tolerancias estrictas):

| Condición | Abstención de la compuerta | Fallos `hard` (media) | Fallos `soft` (media) | Score |
|---|---:|---:|---:|---:|
| `C0_clean` | **0,0 %** | 0,00 | 0,00 | 1,000 |
| `NC_shuffled` | **100 %** | 2,06 | 2,96 | 0,213 |
| `NC_nan` | 100 % | — | — | 0,000 |
| `NC_absent` | 100 % | — | — | 0,000 |
| `NC_zero` | 0,0 % | 0,00 | 0,00 | 1,000 |

- En datos limpios solo falla `resp_n_cycles_vs_rate` en el **0,3 %** de las
  ventanas (y no dispara la abstención, porque se tolera un fallo `soft`).
- En `NC_shuffled` fallan `eda_scr_rate_vs_count` (100 %),
  `eda_mean_vs_tonic` (100 %), `resp_n_cycles_vs_rate` (100 %),
  `temp_range_vs_std` (100 %), `eda_std_vs_phasic_std` (95,7 %) y
  `ecg_n_beats_vs_hr` (85 %).
- `NC_zero` **no es detectable**: el vector de ceros es internamente coherente
  (0 = 0 en todas las identidades). Es la misma limitación ya documentada para
  la escala centrada (0 es la media del sujeto, no «sin señal»).

### 3.1. Sistema compuesto (LLM + compuerta)

La compuerta actúa como **veto previo**: si se abstiene, la predicción final es
`abstain` sin llamar al modelo. Sobre las respuestas E12 ya generadas:

| Sistema | Condición | Abstención | Accuracy | Balanced | Binario acc | Recall stress |
|---|---|---:|---:|---:|---:|---:|
| E12 solo | `C0_clean` | 19,3 % | 0,587 | 0,468 | 0,930 | 0,980 |
| E12 + compuerta | `C0_clean` | 19,3 % | 0,587 | 0,468 | 0,930 | 0,980 |
| E12 solo | `NC_shuffled` | 23,6 % | 0,596 | 0,452 | 0,891 | 0,963 |
| E12 + compuerta | `NC_shuffled` | **100 %** | — | — | — | — |
| Absoluta solo | `C0_clean` | 8,0 % | 0,485 | 0,370 | 0,787 | 0,583 |
| Absoluta + compuerta | `C0_clean` | 8,0 % | 0,485 | 0,370 | 0,787 | 0,583 |

**El resultado central de P1:** la compuerta recupera la abstención total en
`NC_shuffled` (23,6 % → 100 %) **sin ningún coste** en datos limpios (0 % de
abstenciones añadidas, métricas idénticas). El trade-off que documentaba E12
desaparece a nivel de sistema.

## 4. Robustez: no todo es `NC_shuffled`

`NC_shuffled` aplica una **permutación fija** (semilla 42), es decir, la
corrupción más favorable para un detector determinista. Se probaron
corrupciones alternativas sobre las 705 ventanas:

| Corrupción | Abstención | Score medio |
|---|---:|---:|
| `shuffle_modality_random` (permutación aleatoria por ventana) | 99,9 % | 0,303 |
| `shuffle_global` (permutación entre modalidades) | 100 % | 0,204 |
| `swap_one` (un único intercambio de valores) | 33,5 % | 0,903 |
| `noise_10pct` (ruido multiplicativo ±10 %, coherente) | 96,2 % | 0,714 |

Lectura:

- Las corrupciones **estructurales** (permutar valores) se detectan casi
  siempre, incluso con semilla aleatoria por ventana.
- Un **único intercambio** pasa desapercibido el 66,5 % de las veces: es una
  corrupción mínima, indistinguible de una ventana real con un valor atípico.
  La compuerta no pretende detectar errores puntuales.
- El **ruido coherente** del ±10 % dispara la compuerta (96,2 %) porque rompe
  las identidades exactas. Es un falso positivo *de diseño*: los datos reales
  del pipeline son exactamente coherentes (0 % en limpio), pero un pipeline con
  features derivadas de forma independiente podría necesitar más tolerancia.

### 4.1. Calibración de la laxitud

`check_consistency(..., tolerance_scale=s)` multiplica las tolerancias
(`s=2` ⇒ 5 % → 10 % en las identidades, `rango ≥ 1 × DE`, etc.):

| Laxitud | `C0_clean` | `NC_shuffled` | `swap_one` | `noise_10pct` |
|---|---:|---:|---:|---:|
| 1,0 (defecto) | 0,0 % | 100 % | 33,5 % | 96,2 % |
| 1,5 | 0,0 % | 100 % | 31,9 % | 87,8 % |
| 2,0 | 0,0 % | 100 % | 30,6 % | 76,5 % |
| 3,0 | 0,0 % | 100 % | 29,4 % | 52,9 % |

La detección de corrupción estructural es plana frente a la laxitud; la
sensibilidad al ruido cae rápido. Con datos limpios reales no hay falsos
positivos en ningún punto, así que la laxitud es un parámetro de diseño
(documentado) y no una calibración empírica sobre etiquetas.

## 5. Intervención solo-prompt (ablación)

Pregunta: ¿puede el LLM hacer la comprobación por sí mismo si se le instruye?
En la escala centrada las identidades exactas no son recalculables, pero sí lo
es la coherencia de **niveles estandarizados** de features ligadas. De hecho,
en datos limpios los z-scores de `eda_scr_count` y `eda_scr_rate_per_min` son
**idénticos** (la tasa es el conteo dividido por una constante), igual que
`ecg_n_beats` y `ecg_hr_bpm`, o `eda_mean_us` y `eda_tonic_mean_us`. Por tanto
la comprobación es posible en la representación: solo hay que comparar niveles.

Se añadió `CONSISTENCY_RULE_STANDARDIZED` al prompt (92 llamadas: 46 ventanas ×
`C0_clean` + `NC_shuffled`, S3, `qwen2.5:7b`, T=0,3, seed 42):

| Variante | Condición | Abstención | Accuracy | Balanced | Binario acc | Recall stress |
|---|---|---:|---:|---:|---:|---:|
| Base E12 | `C0_clean` | 21,7 % | 0,667 | 0,469 | 0,923 | 1,000 |
| + regla de consistencia | `C0_clean` | 10,9 % | 0,634 | 0,472 | 0,929 | 1,000 |
| Base E12 | `NC_shuffled` | 13,0 % | 0,575 | 0,441 | 0,852 | 1,000 |
| + regla de consistencia | `NC_shuffled` | **0,0 %** | 0,522 | 0,444 | 0,857 | 1,000 |

Acuerdo de decisión entre la misma ventana limpia y su versión corrupta
(0 = el modelo reacciona a la corrupción, 1 = ignora el cambio):

| Variante | Acuerdo `C0` vs `NC_shuffled` |
|---|---:|
| Base E12 | 35/46 = 0,761 |
| + regla de consistencia | **38/46 = 0,826** |

**La intervención solo-prompt falla.** La regla no aumenta la abstención sobre
datos incoherentes (baja a 0 %) y hace que el modelo decida *más* igual en
limpio y corrupto: no ejecuta la comparación entre features ligadas, sigue
aplicando umbrales a valores individuales y responde con confianza 75–85.
La comprobación es factible en la representación, pero el modelo no la realiza
a partir de una instrucción; por eso P1 se resuelve con la compuerta
determinista y no con prompting.

## 6. Integración en el orquestador (`run_phase2`)

La compuerta está integrada en `sources/phase2/orchestrator.py` y **activa por
defecto**:

- `build_consistency_reports(...)` calcula un veredicto por
  `(window_id, condition_id)` sobre las features **absolutas** de la vista
  numérica (las mismas que ve el LLM en ambas representaciones) y, en las
  condiciones `NC_*`, sobre la versión degradada.
- `apply_consistency_gate(...)` aplica el veredicto al registro:

| Campo | Contenido |
|---|---|
| `consistency.evaluated` | `true` si la ventana se pudo evaluar |
| `consistency.gate_abstained` | veto de la compuerta |
| `consistency.n_hard_failures` / `n_soft_failures` | fallos por severidad |
| `consistency.failed_checks` | nombres de las comprobaciones que fallan |
| `consistency.consistency_score` | fracción de comprobaciones que pasan |
| `decision_llm` / `confidence_llm` | **decisión original del modelo** (se conserva) |
| `decision` / `confidence` | decisión final del sistema (abstain / 0 si veta) |
| `status` | `gate_abstention` cuando la compuerta anula una decisión concreta |

- La caché guarda la respuesta **cruda** del LLM (sin compuerta), de modo que el
  veredicto se recalcula en cada ejecución: cambiar tolerancias o umbrales no
  requiere invalidar la caché ni volver a llamar al modelo.
- Desactivación (ablación): `python main.py --proc GENERATE --experiment
  controls ... --noGate`, o
  `defaults.llm.consistency_gate = false` en `config.json`. La laxitud se ajusta
  con `defaults.llm.consistency_tolerance_scale`.
- El resumen de ejecución (`run_orchestrator`) y el manifiesto por sujeto
  incluyen `n_gate_abstentions` y el bloque `consistency_gate`.

Validación sobre el **corpus completo** (15 sujetos, 705 ventanas × 2
representaciones = 1.410 respuestas + controles), refrescado desde caché con
0 llamadas al LLM:

| Condición (numeric / text) | Abstención del modelo (crudo) | Abstención del sistema (con compuerta) | Vetos |
|---|---:|---:|---:|
| `C0_clean` | 8,0 % / 17,3 % | **8,0 % / 17,3 %** (idéntica) | 0 |
| `NC_shuffled` | 95,3 % / 86,5 % | **100 % / 100 %** | 33 / 95 |
| `NC_zero` | 100 % | 100 % | 0 |
| `NC_nan` | 100 % | 100 % | 0 (ya se abstenía) |
| `NC_absent` | 100 % | 100 % | 0 (ya se abstenía) |

- Total: **128 decisiones anuladas** por la compuerta (sobre `NC_shuffled`).
- `C0_clean` no pierde ni una decisión: accuracy 0,485 (numeric) / 0,536
  (text), exactamente igual que sin compuerta.
- Reportes: `evaluation_report_gated.json` (sistema) y
  `evaluation_report_raw.json` (modelo crudo, `--raw-decisions`).

En S3, antes de refrescar el corpus, se verificó el mismo comportamiento
(460 respuestas desde caché, 0 llamadas): `NC_shuffled` 92/92 abstención con 8
vetos, `C0_clean` 0 vetos, `NC_nan`/`NC_absent` 92/92 vetadas y `NC_zero`
0 vetos.

## 7. Interpretación

- La comprobación de consistencia resuelve P1 **a nivel de sistema**, no de
  prompt: es una compuerta determinista y auditable antes del LLM.
- La ablación solo-prompt (§5) muestra que el modelo **no ejecuta** la
  comprobación aunque la instrucción sea explícita y la representación la
  permita. Es un resultado de RQ2/RQ3: el límite no está solo en la
  información disponible, sino en el razonamiento aplicado sobre ella.
- No aporta información de etiquetas ni de calidad del pipeline (SQI,
  cobertura): deriva la coherencia de los propios valores, lo que la mantiene
  dentro del contrato anti-leakage.
- Encaja con la taxonomía de anomalías del informe de literatura
  (`docs/literatura_slip.md`, E1: *missing / plausibility / conformance*) y con
  la idea de severidad + FlipRate (E2): la compuerta es un validador de
  *conformance* de las identidades del propio extractor.

## 7. Limitaciones

- **`NC_zero` no se detecta** (vector coherente); igual que en E12, no es un
  control válido en la escala centrada.
- **Un solo intercambio** pasa desapercibido; la compuerta detecta corrupción
  estructural, no outliers individuales.
- Las tolerancias asumen coherencia exacta del extractor; con features
  derivadas independientemente habría que relajar (`tolerance_scale`).
- Se evaluó con la permutación fija del control negativo; la robustez se
  documenta con corrupciones sintéticas adicionales, no con fallos reales de
  sensor (eso requiere el validador de calidad, p. ej. PhysioNet 2011).
- El veto es binario: no gradúa la confianza. Una versión probabilística
  (score de consistencia → probabilidad de abstención) queda como trabajo
  futuro.

## 8. Reproducción

```bash
# Análisis completo (compuerta + sistema compuesto + robustez + calibración)
python scripts/analyze_consistency.py

# Solo un sujeto
python scripts/analyze_consistency.py --subjects S3 --conditions C0_clean,NC_shuffled

# Piloto de la variante solo-prompt en S3
python scripts/run_e12_pilot.py --subjects S3 \
  --conditions C0_clean,NC_shuffled --scaling robust --consistency
```

Salidas:

- `results/output/WESAD/PHASE2/e12/consistency_gate_report.json`
- `results/output/WESAD/PHASE2/e12/S3.numeric_centered_consistency.responses.jsonl`
- `results/output/WESAD/PHASE2/e12/e12_summary_consistency.json`
