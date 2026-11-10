# ACL2027

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20prototype-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-227-passing-brightgreen)

---

## 🎯 Qué pregunta esta tesis (en claro)

**¿Dicen los LLM lo que la evidencia respalda?**

Cuando le das a un modelo de lenguaje señales fisiológicas resumidas —frecuencia cardiaca, actividad electrodérmica, respiración, temperatura— y le pides que clasifique un estado afectivo (baseline, estrés, diversión, meditación), cuatro cosas importan más que el acierto bruto:

1. **Corrección:** ¿acierta el estado?
2. **Confianza numérica:** ¿expresa una certeza adecuada a lo que sabe?
3. **Certeza lingüística:** ¿usa marcadores de cautela (*hedges*) o de seguridad (*boosters*) acordes con la evidencia?
4. **Respaldo de afirmaciones:** ¿sus justificaciones están respaldadas por los datos que realmente recibió?
5. **Abstención:** ¿sabe cuándo no hay suficiente información y se abstiene?

ACL2027 no se conforma con medir el acierto. Mide **cómo cambia todo lo anterior cuando se degrada la señal de entrada**: se elimina la respiración, se degrada la actividad electrodérmica, se anulan canales, con parámetros conocidos y controlados. Así se comprueba si el LLM modula su respuesta en función de la calidad de la evidencia o si sigue hablando con la misma fluidez y seguridad aunque la información se haya desvanecido.

**Banco de pruebas principal:** WESAD (15 sujetos, estados afectivos inducidos, señales de pecho y muñeca).
**Controles de generalidad:** K-EmoCon y PhysioNet/CinC 2017.

---

## 📊 Resultados en una página (qué encontramos)

| Hipótesis | Resultado | Qué significa |
|---|---|---|
| **H1** Corrección bajo degradación | ✅ Apoyada | Accuracy cae de 0,61 → 0,29; EDA es la modalidad crítica (exceso de coste 0,15–0,20 sobre la referencia). |
| **H2a** Confianza numérica | ⚠️ Matizada | AUROC de confianza 0,51 → 0,76; sobreconfianza crece de +0,15 a +0,47. |
| **H2b** Certeza lingüística | ✅ Apoyada | Índice de certeza +0,13 → −1,12; los *hedges* suben y los *boosters* caen con la degradación. |
| **H3** Efecto de la representación | ✅ Apoyada | El texto acierta menos que el numérico (−0,09) pero degrada menos (+0,065 de interacción); en degradación severa se vuelve **más asertivo** (+1,36 de certeza). |
| **H4** Adecuación de la abstención | ❌ Refutada (forma fuerte) | Infra-abstención en degradación (falla 74 %, se abstiene 16 %) y sobre-abstención en limpio (12,5 % con tarea resoluble). |
| **H5** Respaldo de afirmaciones | ✅ Hallazgo principal | El respaldo de claims *sube* a 0,83 mientras la accuracy cae a 0,29: el modelo cita bien los valores pero falla al combinarlos. |
| **H6** Indicador externo de calidad | ⚠️ Apoyado (SQI) | El SQI añade AUROC incremental (+0,05 a +0,09); el gate de consistencia no añade (Δ≈0). |

Detalle y tablas completas: [`docs/estado_wesad.md`](docs/estado_wesad.md) y [`docs/resultados_degradacion.md`](docs/resultados_degradacion.md).

---

## 📚 Documentación — empieza por la hipótesis

Toda la documentación vive en **`docs/`**, con un índice pensado para dos lectores en [`docs/README.md`](docs/README.md).

| Si quieres… | Empieza por |
|---|---|
| **Entender la pregunta y las hipótesis de la tesis** | [`docs/hipotesis.md`](docs/hipotesis.md) |
| **Ver qué se construyó, qué se midió y qué es firme** | [`docs/estado_wesad.md`](docs/estado_wesad.md) |
| **Revisar los resultados completos del experimento principal** | [`docs/resultados_degradacion.md`](docs/resultados_degradacion.md) |
| **Saber qué es firme y qué es exploratorio** | [`docs/split_dev_test.md`](docs/split_dev_test.md) |
| **Conocer el pipeline de datos (Fase 1)** | [`docs/ingestion.md`](docs/ingestion.md) |
| **Ver el mapa completo de documentos** | [`docs/README.md`](docs/README.md) |

El README de [`docs/`](docs/README.md) describe la arquitectura por componentes (datos, diseño experimental, resultados, validación externa, literatura y memoria operativa) y enlaza cada documento con su estado (firme / exploratorio / negativo documentado). Es el punto de entrada para quien quiera entrar en detalle.

---

## 🚀 Cómo replicar el experimento

### 1. Entorno

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuración

```sh
cp config_example.json config.json   # editar rutas, tokens y ajustes de LLM
```

`config.json` está gitignorado. Admite merge profundo por subject (`results/input/{subject}/config.json`).

### 3. Fase 1 — ingesta del corpus (`INGEST`)

```sh
python main.py --proc INGEST --subject WESAD --dry-run
python main.py --proc INGEST --subject WESAD --subjects S2 --step package
python main.py --proc INGEST --subject WESAD          # 15 sujetos, corpus completo
```

Pasos: `discover → ingest → qc → window → degrade → features → represent → package`.
`--step X` ejecuta hasta X inclusive. Genera el corpus canónico, las ventanas de 60 s, las features con SQI y las representaciones numérica y textual.

### 4. Fase 2 — llamadas al LLM (`PHASE2`)

```sh
python main.py --proc PHASE2 --subject WESAD --subjects S5,S6
python main.py --proc PHASE2 --subject WESAD --subjects S5 --noGate   # ablación (sin compuerta)
```

Requiere Ollama con `qwen2.5:7b`. Cada respuesta guarda decisión, confianza, claims, razonamiento, prompt completo y veredicto de la compuerta de consistencia.

### 5. Experimento principal — degradación controlada (H1–H6)

```sh
# Representación numérica (2.835 llamadas) y textual (2.835)
python scripts/run_degradation_sweep.py
python scripts/run_degradation_sweep.py --representation text

# Análisis: corrección, abstención, calibración, claims, certeza, contraste numérico vs texto
python scripts/analyze_degradation.py \
  --compare-with results/output/WESAD/PHASE2/degradation_text
```

Salidas: `results/output/WESAD/PHASE2/degradation/` y `degradation_text/`, más el reporte agregado.

### 6. Validación externa (control de generalidad)

```sh
# K-EmoCon (arousal/valencia; no tiene etiqueta de estrés — resultado negativo documentado)
python scripts/run_p3_kemocon.py --reference --reference-only

# PhysioNet/CinC 2017 (clase `noisy` real como control de generalidad)
python scripts/run_p3_physionet.py --per-class 150
```

### Flags de CLI

| Flag | Descripción |
|---|---|
| `--subject` | Carpeta en `results/input/` (default: `default`). |
| `--proc` | Proceso: `INGEST` o `PHASE2`. |
| `--step` | Inicia el pipeline desde un paso con nombre. |
| `--subjects` | Subconjunto de sujetos (`S5,S6`). |
| `--force` | Recalcula artefactos aunque el fingerprint coincida. |
| `--dry-run` | Resuelve configuración y rutas sin ejecutar. |
| `--noGate` | Desactiva la compuerta de consistencia (ablación). |
| `--allowDev` | Permite sujetos de desarrollo (S2–S4) en `PHASE2`. |

---

## 📁 Estructura del proyecto

```
ACL2027/
├── main.py                      # Punto de entrada (PROCESS_REGISTRY)
├── config.json / config_example.json
├── docs/                        # Documentación (empezar por docs/README.md)
│   ├── hipotesis.md             # Pregunta de tesis, hipótesis H1–H6, objetivos
│   ├── estado_wesad.md          # Resumen ejecutivo de resultados
│   ├── resultados_degradacion.md# Resultados completos del experimento principal
│   ├── ingestion.md             # Pipeline INGEST de WESAD
│   ├── fase2_diseno.md          # Diseño de prompts, esquema JSON, modelo, controles, gate
│   ├── fase3_evaluacion.md      # Diseño de evaluación (métricas, calibración, abstención)
│   ├── degradacion_llm.md       # Diseño del barrido de degradación con LLM
│   ├── resolubilidad.md         # Separabilidad de clases y confusor temporal
│   ├── consistencia.md          # Compuerta de consistencia fisiológica (P1)
│   ├── claims_h5.md             # Verificador de claims y certeza lingüística
│   ├── p2_binario.md            # Prompt binario stress/no-stress (exploratorio)
│   ├── e12_resultados.md        # Representación centrada por sujeto (exploratorio)
│   ├── p4_calibracion.md        # Calibración, temperatura, conformal (exploratorio)
│   ├── p3_kemocon.md            # Validación externa K-EmoCon (negativo documentado)
│   ├── p3_physionet.md          # Validación externa PhysioNet 2017 (negativo documentado)
│   ├── literatura_slip.md       # Análisis del corpus SLIP (E1–E14)
│   └── README.md                # Índice completo de documentos por área
├── scripts/                     # Experimentos y análisis
│   ├── run_degradation_sweep.py     # barrido C0–C5 (numérico/textual)
│   ├── analyze_degradation.py       # H1–H6 + contraste de representaciones
│   ├── run_p3_kemocon.py            # validación externa K-EmoCon
│   ├── run_p3_physionet.py          # validación externa PhysioNet 2017
│   ├── run_e12_pilot.py             # representación centrada (exploratorio)
│   ├── resolvability_analysis.py    # separabilidad y confusor temporal
│   ├── analyze_consistency.py       # compuerta de consistencia
│   ├── analyze_calibration.py       # calibración y conformal
│   └── evaluate_phase3.py           # evaluación agregada del corpus
├── sources/
│   ├── common/                  # logging, config, rutas, procedencia, validación, LLM client
│   ├── ingestion/               # pipeline INGEST (WESAD): discover, ingest, qc, window, degrade, features, represent, package
│   ├── phase2/                  # prompts, LLM, esquema de respuesta, controles negativos, gate de consistencia
│   ├── phase3/                  # evaluación, claims, certeza, calibración, degradación
│   └── external/                # adaptadores K-EmoCon y PhysioNet 2017
├── tests/                       # 227 tests (sin dependencia de los datasets reales)
└── results/                     # NO versionado: input/ y output/ por subject
```

---

## 🧪 Tests

```sh
python -m pytest -q
```

**227 tests**: infraestructura, ingesta, features, representaciones, prompts, esquema de respuesta, orquestador, consistencia, calibración, claims/certeza y adaptadores externos. No requieren descargar WESAD ni Ollama.

---

## 🧭 Filosofía del código

- **Anti-leakage por diseño:** las etiquetas viven solo en `groundtruth/`; nunca se mezclan con features, representaciones ni prompts. Hay comprobaciones (`assert_no_leakage`) y tests que lo verifican.
- **Reproducibilidad:** versiones fijadas, caché de llamadas al LLM, split dev/test congelado (dev S2–S4, test S5–S17), few-shot con medianas de desarrollo (sin fuga del test al prompt).
- **Trazabilidad:** cada respuesta del LLM guarda el prompt completo, parámetros del modelo, semilla, hash de caché y veredicto de la compuerta de consistencia para auditoría posterior.
- **Rutas sin hardcodeo:** todas las rutas se resuelven a través de `processControl.env` (prohibido hardcodear `"results"` en módulos).

---

## ⚠️ Limitaciones y ética

- **No es un estudio clínico:** no evalúa utilidad diagnóstica ni debe usarse para decisiones médicas.
- **Resultados acotados:** no se extrapolan automáticamente a otros dominios, datasets o modelos.
- **Datos sujetos a licencia:** respetar las licencias de WESAD y los demás datasets utilizados.
- Los modelos de API pueden cambiar o retirarse; se usan modelos de pesos abiertos con versiones fijadas siempre que es posible.

---

La memoria operativa para retomar el trabajo está en [`AGENTS.md`](AGENTS.md) y [`SESSION_STATE.md`](SESSION_STATE.md).