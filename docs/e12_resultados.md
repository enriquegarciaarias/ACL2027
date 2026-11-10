# E12 — Representación centrada por sujeto: resultados del corpus completo

**Fecha:** 2026-11-08
**Script:** `scripts/run_e12_pilot.py`
**Datos:** 705 ventanas, 15 sujetos, condición `C0_clean`
**Modelo:** `qwen2.5:7b`, T=0,3, seed=42
**Reportes:** `e12_summary_c0.json` (C0) y `e12_summary_nc.json` (controles negativos)

---

## 1. Qué se probó

- Features de pecho (23) estandarizadas **dentro de cada sujeto** con escalado
  robusto: `(x − mediana) / IQR`, recortado a `[−3, +3]`.
- Prompt específico para la escala: nota de z-scores, few-shot en la misma
  escala y regla de “patrón dominante”.
- Sin rúbrica fisiológica explícita (la versión con rúbrica se probó en S3 y
  empeoró la calibración).
- Normalización **no supervisada** (no usa etiquetas); es transductiva:
  usa la distribución de la sesión del sujeto. En despliegue se usaría una
  referencia inicial o una ventana móvil.

## 2. Resultados agregados (C0_clean)

| Métrica | Absoluto | Centrado robusto |
|---|---:|---:|
| Ventanas | 705 | 705 |
| Válidas | 704 | 696 |
| Inválidas | 1 | 9 (1,3 %) |
| Abstención | 8,0 % | 19,3 % |
| Decididas | 648 | 562 |
| Accuracy (decididas) | 48,5 % | **58,7 %** |
| Balanced accuracy (decididas) | 37,0 % | **46,9 %** |
| ECE | 0,299 | **0,232** |
| Brier | 0,215 | **0,175** |
| AUROC baseline | 0,550 | **0,722** |
| AUROC stress | 0,744 | **0,958** |
| Decisiones | 515 baseline / 133 stress | 349 baseline / 214 stress |

### Binario stress vs baseline (decididas)

| Sistema | n | Accuracy | Precision | Recall | Balanced | Cobertura |
|---|---:|---:|---:|---:|---:|---:|
| Absoluto | 399 | 78,7 % | 75,0 % | 58,3 % | 0,739 | 90,7 % |
| Centrado robusto | 355 | **93,0 %** | **86,7 %** | **98,0 %** | **0,937** | 80,7 % |

El centrado robusto **sube el recall de stress del 58 % al 98 %** y la
precisión del 75 % al 87 %. El coste es más abstención (19,3 % global) y algo
menos de cobertura en el subconjunto binario (80,7 % vs 90,7 %).

## 3. El 4 clases sigue colapsado

| Real | Predicho (centrado) |  |  |  |  |
|---|---|---:|---:|---:|---:|
|  | baseline | stress | amusement | meditation | abstain |
| baseline | 186 | 22 | 0 | 0 | 72 |
| stress | 3 | 144 | 0 | 0 | 8 |
| amusement | 61 | 10 | 0 | 0 | 13 |
| meditation | 99 | 38 | 0 | 0 | 40 |

- `amusement`: **0 predicciones**; 61 ventanas van a baseline.
- `meditation`: **0 predicciones**; 99 van a baseline, 38 a stress, 40 se
  abstienen.
- La balanced accuracy de 0,468 es “solo decididas”; si se cuenta la
  abstención como error, baja a ≈0,398.

**Interpretación:** E12 mejora la distinción binaria y la calibración, pero
el LLM sigue sin usar los patrones de `meditation` (respiración baja + HRV
alta) ni de `amusement`. La referencia centrada por sujeto obtiene 0,823 en
4 clases, así que la información está; el límite es del modelo/prompt.

## 4. Comparación con la referencia

| Referencia (mismas features) | 4 clases | stress vs baseline |
|---|---:|---:|
| Absoluta | 0,644–0,677 | 0,869 |
| Centrada por sujeto | **0,823** | **0,978** |
| LLM absoluto | 0,485 | 0,739 |
| LLM centrado | 0,586 | **0,937** |

El LLM centrado se acerca a la referencia centrada en el binario
(0,937 vs 0,978, con menor cobertura), pero sigue muy lejos en 4 clases
(0,468 vs 0,823).

## 5. Lectura para el diseño experimental

- **E12 es un éxito para el endpoint binario:** representación relativa al
  sujeto + prompt en la misma escala produce un detector de stress con
  recall 98 % y balanced accuracy 0,937.
- **No resuelve el 4 clases.** El modelo sigue siendo, en la práctica, un
  clasificador binario. Eso refuerza la decisión de usar el binario como
  endpoint primario, pero por **capacidad del modelo**, no por falta de
  información.
- **La calibración mejora** (ECE 0,232 vs 0,299; Brier 0,175 vs 0,215), pero
  sigue habiendo sobreconfianza.
- **La abstención aumenta** (19,3 % vs 8,0 %), concentrada en baseline (72) y
  meditation (40); es una abstención fisiológicamente razonable, pero hay que
  reportarla junto con la cobertura.

## 6. Limitaciones

- La normalización es transductiva (usa la sesión del sujeto).
- El cambio incluye representación **y** prompt específico de escala; no es
  una ablación pura de features.
- Solo `C0_clean`: faltan los controles negativos (`NC_*`) para verificar que
  el modelo se abstiene con datos degradados.
- Los 9 inválidos son errores de formato del modelo
  (`magnitude: "stable"`, `direction: "high"`, claims incompletos); 1,3 %.
- La comparación con la referencia no está ajustada por cobertura.

## 7. Controles negativos y dependencia de la representación

### 7.1. `NC_zero` no es válido en la escala centrada

En la vista centrada, **0 ya no significa “sin señal”**, sino “en la media del
sujeto”. El modelo interpretó un vector de ceros como un baseline plausible
(46/46 en S3) y **no se abstuvo**; la vista absoluta sí se abstiene porque
HR=0/EDA=0 es fisiológicamente imposible. Por eso `NC_zero` se excluyó del
barrido y no debe interpretarse como un fallo del modelo.

### 7.2. Resultados de los controles válidos (15 sujetos, 705 ventanas)

| Condición | Vista | Abstención | Decisiones | Inválidos |
|---|---|---:|---|---:|
| `NC_nan` | centrada | **100 %** | 705 abstain | 0 |
| `NC_nan` | absoluta | **100 %** | 705 abstain | 0 |
| `NC_absent` | centrada | **100 %** | 705 abstain | 0 |
| `NC_absent` | absoluta | **100 %** | 705 abstain | 0 |
| `NC_shuffled` | absoluta | **95,3 %** | 666 abstain / 18 baseline / 15 stress | 0 |
| `NC_shuffled` | centrada | **23,6 %** | 338 baseline / 202 stress / 165 abstain | 6 |

### 7.3. El hallazgo: la vista centrada pierde la detección de estructura corrupta

En `NC_shuffled` los valores se barajan dentro de cada modalidad: la
distribución se mantiene, pero se destruye la correspondencia
feature→valor y las dependencias fisiológicas (HR↔RR, EDA media↔SCR).
La vista absoluta se abstiene en el 95,3 % de las ventanas. La centrada
**solo se abstiene en el 23,6 %** y emite 534 decisiones confiadas
(confianza 75–85), con 59,6 % de accuracy y balanced accuracy 0,452.

El razonamiento típico es: “EDA y SCR están muy por encima de la referencia,
indica estrés”, sin comprobar que el resto de features no es coherente con
esa lectura. Es decir, el modelo aplica **umbrales sobre valores
individuales**, no una comprobación de estructura fisiológica.

Por sujeto, la abstención en `NC_shuffled` centrado va del 11 % al 36 %:
el fallo es sistemático, no de un sujeto aislado.

### 7.4. Implicación

- E12 centrado es excelente para el binario limpio (93,0 % accuracy,
  recall 98,0 %, balanced 0,937) y para detectar datos ausentes
  (`NC_nan`/`NC_absent` 100 % abstención).
- Pero **empeora la sensibilidad a la corrupción estructural** frente a la
  vista absoluta. Es un trade-off explícito que debe reportarse: más
  precisión en limpio, menos robustez ante evidencia incoherente.
- Este resultado es, en sí mismo, un hallazgo de la tesis (RQ2/RQ3): la
  representación no solo cambia el acierto, también la capacidad de detectar
  que la evidencia no es fiable. Sugiere la necesidad de una **comprobación
  de consistencia** o de una señal de calidad explícita (PSRI) antes del LLM.
- **Resuelto en P1:** la compuerta determinista de consistencia
  (`sources/phase2/consistency.py`) recupera el 100 % de abstención en
  `NC_shuffled` sin coste en limpio. Ver `docs/consistencia.md`.

## 8. Próximos pasos

1. **Comprobación de consistencia** en el prompt (HR↔RR, EDA media↔SCR,
   amplitudes) o compuerta de calidad previa; probar en S3 `NC_shuffled`.
2. **Prompt binario específico** (stress vs no-stress) sobre E12; probablemente
   mejore aún más la precisión y reduzca la abstención.
3. **Intervención para meditation**: regla explícita de respiración o
   subpregunta descompuesta; si no mejora, dejar 4 clases como secundario.
4. **Integrar E12** en `represent.py`/orquestador o mantenerlo como
   experimento separado con su propio manifiesto.
5. **Calibración/conformal** para el umbral de abstención del binario.
6. Comparar con la referencia centrada **a igual cobertura**.
7. **K-EmoCon como validación externa** (`pre` vs `debate`), reutilizando
   la tabla de features de PSRI.

## 9. Reproducción

```bash
python scripts/run_e12_pilot.py \
  --subjects S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S13,S14,S15,S16,S17 \
  --scaling robust
```

Salidas:

- `results/output/WESAD/PHASE2/e12/{subject}.numeric_centered.responses.jsonl`
- `results/output/WESAD/PHASE2/e12/all.numeric_centered.responses.jsonl`
- `results/output/WESAD/PHASE2/e12/e12_summary_c0.json`
- `results/output/WESAD/PHASE2/e12/e12_summary_nc.json`
- `results/output/WESAD/PHASE2/e12/all.numeric_centered.responses.jsonl` (C0)
- `results/output/WESAD/PHASE2/e12/all.numeric_centered_nc.responses.jsonl` (NC)
