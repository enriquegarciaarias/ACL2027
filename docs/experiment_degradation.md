# Documento de Preregistro — Degradación de Canales Mínimos (Piloto ACL2027)

**Proyecto:** ACL2027 — Fase 1 (H0–H6) + Experimento de degradación  
**Fecha:** 2026-11-04  
**Estado:** Borrador para revisión antes de ejecución

---

## 1. Objetivo

Validar que el pipeline puede generar condiciones experimentales donde la
señal fisiológica pierde información de forma controlada, permitiendo evaluar:

- **H1:** Degradación de señal → caída de rendimiento del modelo de referencia y del LLM.
- **H4:** Cuando la evidencia es insuficiente (condición irresoluble), el LLM debe abstenerse en lugar de forzar una clasificación.

---

## 2. Conjunto mínimo de canales

Se selecciona un subconjunto de **2 canales no redundantes** en WESAD, fisiológicamente ortogonales y sin respaldo directo de otro sensor:

| Canal | Fenómeno | ¿Tiene respaldo en WESAD? |
|-------|----------|---------------------------|
| `chest_Resp` | Respiración (frecuencia, variabilidad, amplitud) | **No** — único canal de respiración |
| `chest_EDA` | Actividad electrodérmica (tono tónico, picos SCR) | `wrist_EDA` existe (redundante parcial) |

**Justificación:** La respiración es el único fenómeno medido por un solo canal en WESAD. La EDA del pecho es el canal electrodérmico de mayor calidad (700 Hz vs 4 Hz de muñeca). Aunque `wrist_EDA` existe como respaldo parcial, el conjunto de 2 canales reduce drásticamente la información redundante comparado con las 10 modalidades actuales.

---

## 3. Matriz de condiciones de degradación

Cada condición especifica qué canales se degradan y con qué severidad. La degradación **total** se define como **canal plano (ceros)**, simulando sensor desconectado o señal inutilizable.

| ID | Condición | Canales degradados | Severidad | Definición de severidad | Esperado (modelo de referencia) |
|----|-----------|-------------------|-----------|------------------------|----------------------------------|
| C0 | Limpieza (baseline) | Ninguno | 0.0 | Sin degradación | Resoluble (F1 alto) |
| C1 | Degradación parcial Resp | `chest_Resp` | 0.5 | 50 % de muestras sustituidas por NaN / ruido | Parcialmente resoluble (F1 medio) |
| C2 | Degradación parcial EDA | `chest_EDA` | 0.5 | 50 % de muestras sustituidas por NaN / ruido | Parcialmente resoluble |
| C3 | Degradación total Resp | `chest_Resp` | 1.0 | Canal plano (todos los valores = 0) | Irresoluble o cercano al azar |
| C4 | Degradación total EDA | `chest_EDA` | 1.0 | Canal plano (todos los valores = 0) | Irresoluble o cercano al azar |
| C5 | Degradación total ambos | `chest_Resp` + `chest_EDA` | 1.0 | Ambos canales planos | Irresoluble (debería abstenerse) |

**Definición de severidad:**
- **0.0:** Sin degradación (señal original)
- **0.5:** Parcial — 50 % de las muestras del canal sustituidas por NaN (dropout aleatorio) o ruido blanco con misma varianza
- **1.0:** Total — todo el canal sustituido por ceros (plano), haciendo imposible cualquier extracción de feature fisiológica

---

## 4. Criterios de aceptación del modelo de referencia

Antes de pasar cualquier ventana al LLM, se verificará con un **modelo de referencia simple** (regresión logística o Random Forest sobre las features del conjunto mínimo):

| Criterio | Umbral | Acción si no se cumple |
|----------|--------|------------------------|
| Al menos 1 condición resoluble | F1 ≥ 0.80 en C0 | Reducir aún más canales (a 1) |
| Al menos 1 condición parcialmente resoluble | 0.40 ≤ F1 ≤ 0.80 en C1/C2 | Ajustar severidad parcial |
| Al menos 1 condición irresoluble | F1 ≤ 0.55 en C3/C4/C5 | Aumentar severidad o reducir canales |
| Abstención esperada en C5 | Tasa de abstención ≥ 0.80 (si el modelo la soporta) | Revisar diseño de umbral de abstención |

**Regla de parada:** Si todas las condiciones son resolubles (F1 > 0.80 en todas), el diseño **no sirve** y hay que reducir a 1 canal o aumentar la severidad antes de continuar.

---

## 5. Cambios al pipeline

### Nuevo paso: `degrade`

Se inserta entre `window` y `features` en el orden del pipeline:

```
discover → ingest → qc → window → **degrade** → features → represent → package
```

El paso `degrade`:
1. Lee el canónico ``.npz`` del sujeto
2. Aplica la degradación especificada en la configuración sobre los canales indicados
3. Genera una versión degradada por condición en ``results/output/{subject}/INGEST/degraded/{condition_id}/{subject}.npz``
4. Guarda metadata con fingerprint, canales degradados, severidad y checksum

### Vista LLM filtrada por canales mínimos

Cuando `experiment.minimal_channels` está definido, la vista LLM (`numeric_llm`, `text_llm`) solo incluye features de esos canales, no de las 10 modalidades. Esto evita que el LLM use canales no degradados como respaldo.

---

## 6. Configuración (`config.json`)

Bloque opcional a añadir:

```json
"experiment": {
  "minimal_channels": ["chest_Resp", "chest_EDA"],
  "degradation_enabled": true,
  "degradation_conditions": [
    {"id": "C0_clean", "channels": [], "severity": 0.0},
    {"id": "C1_partial_resp", "channels": ["chest_Resp"], "severity": 0.5},
    {"id": "C2_partial_eda", "channels": ["chest_EDA"], "severity": 0.5},
    {"id": "C3_total_resp", "channels": ["chest_Resp"], "severity": 1.0},
    {"id": "C4_total_eda", "channels": ["chest_EDA"], "severity": 1.0},
    {"id": "C5_total_both", "channels": ["chest_Resp", "chest_EDA"], "severity": 1.0}
  ]
}
```

---

## 7. Hipótesis y criterios de aceptación (H1, H4)

| Hipótesis | Predicción | Criterio de aceptación |
|-----------|------------|------------------------|
| **H1:** Degradación → caída de rendimiento | F1 cae de C0 (alto) a C5 (cercano al azar) | F1(C5) ≤ 0.55 y F1(C0) ≥ 0.80 |
| **H4:** Evidencia insuficiente → abstención | En C5, el LLM se abstiene en la mayoría de ventanas | Tasa de abstención ≥ 0.80 en C5 (si el modelo de evaluación la soporta) |

---

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| `chest_EDA` tiene respaldo en `wrist_EDA` | La vista LLM filtra a solo los 2 canales mínimos. Si `wrist_EDA` sigue dando información, el modelo de referencia lo revelará en C4 y se ajustará el diseño |
| El modelo de referencia es demasiado robusto | Reducir a 1 canal (`chest_Resp` solo) o degradar también `wrist_EDA` en C4/C5 |
| Las ventanas degradadas no se distinguen en el índice | Cada ventana degradada lleva `degradation_id` y `degradation_condition` en su metadata |

---

## 9. Próximos pasos

1. [ ] Revisar y aprobar este documento de preregistro
2. [ ] Implementar el paso `degrade` en `sources/ingestion/degrade.py` (hecho)
3. [ ] Registrar `degrade` en `sources/ingestion/pipeline.py` entre `window` y `features` (hecho)
4. [ ] Añadir bloque `experiment` a `config_example.json` y `config.json` (hecho)
5. [ ] Ejecutar piloto sobre S2 con todas las condiciones C0–C5
6. [ ] Verificar con modelo de referencia que se cumplen los criterios de aceptación
7. [ ] Si se cumplen, pasar a Fase 2 (prompts + LLM) con las condiciones degradadas
