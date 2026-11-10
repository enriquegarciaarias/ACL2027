# Análisis de resolubilidad estadística (WESAD, Fase 3)

**Fecha:** 2026-11-08
**Script:** `scripts/resolvability_analysis.py`
**Reporte:** `results/output/WESAD/PHASE2/resolvability_report.json`
**Pregunta:** ¿se puede justificar el paso a clasificación binaria porque las
clases no son estadísticamente separables con las features que ve el LLM?

**Respuesta corta:** no. Las clases **sí son separables** con las mismas 23
features, incluso tras un control temporal aproximado. El LLM está por debajo
de un modelo de referencia simple. La decisión binaria puede defenderse como
decisión **pragmática** (espacio de decisión efectivo del LLM, relevancia de
stress, potencia estadística), pero **no** como consecuencia de que la
información no permita distinguir las clases.

---

## 1. Diseño

- **Datos:** 705 ventanas de 60 s, 15 sujetos, 4 clases
  (`baseline`, `stress`, `amusement`, `meditation`).
- **Features:** las 23 de pecho de `LLM_ALLOWED_FEATURES` (5 ECG, 9 EDA,
  5 resp, 4 temp). Son exactamente las que recibe el LLM en `numeric_llm` y
  `text_llm`.
- **Validación:** `GroupKFold(5)` por sujeto (sin fuga entre sujetos).
- **Modelos:** regresión logística (lineal) y Random Forest (no lineal), con
  imputación de mediana, estandarización y `class_weight='balanced'`.
- **Métricas:** balanced accuracy, macro-F1, AUROC; IC 95 % bootstrap
  agrupado por sujeto (1000 remuestreos); p-valor de permutación (200).
- **Margen preespecificado:** balanced accuracy 0,60 para pares y 0,35 para
  4 clases (azar 0,25 + 0,10).
- **Regla:** `separable` si IC95 inferior > margen; `equivalente al azar en
  el margen` si IC95 superior < margen; `inconcluso` en otro caso.

**Controles temporales.** WESAD es un protocolo secuencial. Se comparan:

1. `features_only`: referencia justa (misma información que el LLM).
2. `time_only`: fase del protocolo (0-1 por sujeto) como único predictor.
3. `features_plus_time`: features + fase (valor incremental).
4. `time_residualized`: features residualizadas sobre fase² por sujeto.
5. `subject_centered`: z-score intra-sujeto (personalización).

---

## 2. Resultados

### 2.1. Referencia con las mismas features que el LLM

| Par | Balanced acc. | IC 95 % | p perm. | RF | Máx. univariante | Lectura |
|---|---:|---:|---:|---:|---:|---|
| baseline vs stress | **0,869** | [0,785; 0,939] | 0,005 | 0,923 | 0,897 | separable |
| baseline vs amusement | **0,621** | [0,549; 0,699] | 0,005 | 0,594 | 0,696 | inconcluso |
| baseline vs meditation | **0,862** | [0,774; 0,935] | 0,005 | 0,825 | 0,942 | separable |
| stress vs amusement | **0,860** | [0,758; 0,944] | 0,005 | 0,879 | 0,880 | separable |
| stress vs meditation | **0,912** | [0,843; 0,963] | 0,005 | 0,936 | 0,947 | separable |
| amusement vs meditation | **0,926** | [0,876; 0,966] | 0,005 | 0,849 | 0,948 | separable |

4 clases: logreg **0,644** [0,568; 0,720]; RF **0,677** [0,603; 0,744].

**Conclusión:** con la misma información que recibe el LLM, un modelo simple
separa casi todos los pares muy por encima del azar. Solo `baseline vs
amusement` queda en zona inconclusa (0,621), y aun así su IC no es compatible
con equivalencia al azar en el margen de 0,60.

### 2.2. Confusor de orden temporal

| Modelo | 4 clases |
|---|---:|
| Solo fase temporal (RF) | **0,857** [0,820; 0,890] |
| Features solas | 0,644–0,677 |
| Features + fase | **0,890** [0,829; 0,940] |

Pares con solo fase: `baseline` vs cualquier clase = **1,000**;
`stress vs amusement` = 0,323; `stress vs meditation` = 0,689;
`amusement vs meditation` = 0,634.

**Conclusión:** el orden del protocolo (baseline siempre primero, meditación
en bloques tardíos) explica una parte enorme de la separabilidad aparente.
Cualquier modelo de referencia “naive” que use features absolutas está
inflado por este confusor. El LLM **no ve la fase**, así que su referencia
justa es `features_only`, no `features_plus_time`.

### 2.3. Control temporal

| Par | Residualizado en fase² (logreg) | IC 95 % | RF | Lectura |
|---|---:|---:|---:|---|
| baseline vs stress | **0,854** | [0,798; 0,903] | 0,899 | separable |
| baseline vs amusement | **0,793** | [0,731; 0,852] | 0,803 | separable |
| baseline vs meditation | **0,720** | [0,661; 0,776] | 0,777 | separable |
| stress vs amusement | **0,882** | [0,767; 0,981] | 0,944 | separable |
| stress vs meditation | **0,894** | [0,840; 0,946] | 0,894 | separable |
| amusement vs meditation | **0,858** | [0,809; 0,908] | 0,862 | separable |

4 clases residualizado: logreg **0,648** [0,580; 0,716]; RF **0,711**
[0,638; 0,786].

**Conclusión:** incluso eliminando tendencias temporales cuadráticas por
sujeto, todas las clases siguen siendo separables por encima del margen. La
señal fisiológica existe; no es solo orden del protocolo.

### 2.4. Personalización

Features centradas por sujeto (z-score intra-sujeto), 4 clases:
**0,823** [0,774; 0,878].

**Conclusión:** gran parte de la dificultad del LLM con valores absolutos es
la variabilidad inter-sujeto. Una representación relativa al sujeto recupera
mucha separabilidad. Es la intervención E12 más prometedora.

### 2.5. LLM vs referencia (binario stress vs baseline)

| Sistema | Balanced acc. | Accuracy | Recall stress | Cobertura |
|---|---:|---:|---:|---:|
| Referencia features-only | **0,869** | — | — | — |
| LLM `numeric_llm` | 0,739 | 0,787 | 0,583 | 0,907 |
| LLM `text_llm` | 0,805 | 0,843 | 0,652 | 0,841 |

**Coste en exceso** (referencia − LLM): 0,129 (numeric) y 0,064 (text).

El LLM es conservador con `stress`: alta precisión (75–90 %), recall
moderado (58–65 %). En 4 clases nunca predice `amusement` ni `meditation`
(0/705), mientras que la referencia sí las separa. Por tanto, el fallo del
LLM es de **representación/razonamiento**, no de información disponible.

---

## 3. Implicación para la decisión binaria

### Lo que NO se puede afirmar

- ❌ “Las distribuciones no permiten diferenciar las clases.”
- ❌ “El LLM falla porque la información es insuficiente.”
- ❌ “El 4 clases es irresoluble, por eso pasamos a binario.”

### Lo que SÍ se puede afirmar

- ✅ El LLM colapsa su espacio de decisión a `baseline`/`stress`/`abstain`;
  nunca usa `amusement` ni `meditation`.
- ✅ `stress vs baseline` es la distinción más robusta y la que el LLM
  reproduce mejor (balanced acc. 0,74–0,81 frente a 0,87 de la referencia).
- ✅ El LLM está por debajo de un modelo de referencia simple con la misma
  información, sobre todo en 4 clases.
- ✅ El orden del protocolo WESAD es un confusor fuerte; cualquier análisis
  de separabilidad debe controlarlo.
- ✅ La personalización (features relativas al sujeto) mejora mucho la
  referencia (0,823 en 4 clases) y es una vía concreta para mejorar el LLM.

### Decisión recomendada

1. **Endpoint primario:** binario `stress` vs `no-stress` (o vs `baseline`),
   pre-registrado. Justificación: espacio de decisión efectivo del LLM,
   relevancia práctica de stress, mejor potencia/precisión y menor
   ambigüedad de etiquetas. **No** justificarlo por falta de separabilidad.
2. **Secundario:** 4 clases con modelo de referencia y control temporal;
   reportar el gap LLM-referencia y el colapso de predicciones.
3. **Antes de cerrar el diseño:** probar la intervención E12
   (representación centrada por sujeto / basada en desviación respecto al
   baseline) en el LLM. Si recupera parte del 4 clases, la tesis gana un
   resultado más fuerte que “abandonamos 4 clases”.
4. **Reportar el confusor temporal** como hallazgo metodológico: los
   benchmarks de WESAD a nivel de ventana pueden sobreestimar la
   separabilidad si no controlan el orden del protocolo.

---

## 4. Limitaciones

- La residualización temporal es una normalización no supervisada por sujeto;
  usa la distribución de la sesión (sin etiquetas), pero asume que en
  despliegue se dispone de la sesión del sujeto.
- Los huecos entre condiciones son grandes (mediana 601 s; mínimo 237 s), así
  que no hay ventanas fisiológicamente adyacentes para un análisis de
  transición estricto.
- 15 sujetos y ventanas correlacionadas dentro de sujeto: los IC agrupados
  son la unidad correcta, pero la potencia es limitada.
- No se han ajustado modelos mixtos (`feature ~ clase + fase + (1|sujeto)`)
  porque `statsmodels` no está instalado; es la extensión natural.
- El margen (0,60) es una convención; debe pre-registrarse y justificarse.

---

## 5. Reproducción

```bash
python scripts/resolvability_analysis.py --permutations 200
```

El reporte incluye `features_only`, `time_only`, `features_plus_time`,
`time_residualized`, `subject_centered`, `transitions` y `llm_comparison`.
