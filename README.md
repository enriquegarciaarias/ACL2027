# ACL2027

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20prototype-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Qué es este proyecto (en claro)

El proyecto pregunta **si un LLM dice lo que la evidencia respalda**. Se le dan
señales fisiológicas resumidas (frecuencia cardiaca, actividad electrodérmica,
respiración, temperatura) y debe (a) clasificar el estado, (b) expresar su
confianza, (c) justificar con afirmaciones concretas y (d) abstenerse cuando la
evidencia no basta.

Lo diferencial no es el acierto, sino **cómo cambia la respuesta cuando se
degrada la señal**: se elimina la respiración, se degrada la actividad
electrodérmica, etc., con parámetros conocidos. Así se comprueba si el modelo
modula su corrección, su confianza, su lenguaje y sus afirmaciones en función de
la calidad de la evidencia.

WESAD (15 sujetos, estados inducidos) es el banco de pruebas principal. Se
incluyen dos validaciones externas (K-EmoCon y PhysioNet/CinC 2017) que acotan
la generalidad de los hallazgos.

## Resultados en una página

- **La corrección se hunde más que la información disponible.** Con la EDA
  anulada, el LLM baja de 0,61 a 0,29 de accuracy mientras una referencia
  features-only baja de 0,83 a 0,70 (exceso de coste 0,15–0,20).
- **La abstención no sigue a la resolubilidad.** Hay infra-abstención en las
  condiciones degradadas (falla el 74 % y se abstiene el 16 %) y sobre-abstención
  en limpio (12,5 % con una tarea resoluble).
- **Las afirmaciones son localmente ciertas pero la inferencia es errónea.**
  El respaldo de claims *sube* a 0,83 en la condición degradada mientras la
  accuracy cae a 0,29; el modelo cita bien los valores y falla al combinarlos.
- **Confianza, lenguaje y evidencia se desacoplan.** La confianza apenas
  distingue aciertos en limpio (AUROC 0,51) y la sobreconfianza crece hasta
  +0,47; el lenguaje sí añade cautela (índice de certeza +0,13 → −1,12).
- **La representación importa.** El resumen textual acierta menos que el JSON
  numérico (−0,09), pero se degrada menos (+0,065 de interacción) y en
  degradación severa se vuelve **más asertivo** (+1,36) mientras el numérico se
  vuelve más cauto.

Detalle y tablas: [`docs/estado_wesad.md`](docs/estado_wesad.md) y
[`docs/resultados_degradacion.md`](docs/resultados_degradacion.md).

## 📚 Documentación

Toda la documentación está en **`docs/`**, con un índice pensado para dos
lectores en [`docs/README.md`](docs/README.md):

| Si quieres… | Empieza por |
|---|---|
| Entender la pregunta y las hipótesis | [`docs/hipotesis.md`](docs/hipotesis.md) |
| Ver qué se hizo y qué se encontró | [`docs/estado_wesad.md`](docs/estado_wesad.md) |
| Revisar los resultados completos | [`docs/resultados_degradacion.md`](docs/resultados_degradacion.md) |
| Saber qué es firme y qué exploratorio | [`docs/split_dev_test.md`](docs/split_dev_test.md) |
| Conocer el pipeline de datos | [`docs/ingestion.md`](docs/ingestion.md) |
| Ver el mapa completo de documentos | [`docs/README.md`](docs/README.md) |

La memoria operativa (para retomar el trabajo) está en
[`AGENTS.md`](AGENTS.md) y [`SESSION_STATE.md`](SESSION_STATE.md).

---

## 🚀 Operativa

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

`config.json` está gitignorado. Admite merge profundo por subject
(`results/input/{subject}/config.json`).

### 3. Fase 1 — ingesta del corpus (`INGEST`)

```sh
python main.py --proc INGEST --subject WESAD --dry-run
python main.py --proc INGEST --subject WESAD --subjects S2 --step package
python main.py --proc INGEST --subject WESAD          # 15 sujetos
```

Pasos: `discover → ingest → qc → window → degrade → features → represent → package`.
`--step X` ejecuta hasta X inclusive.

### 4. Fase 2 — llamadas al LLM (`PHASE2`)

```sh
python main.py --proc PHASE2 --subject WESAD --subjects S5,S6
python main.py --proc PHASE2 --subject WESAD --subjects S5 --noGate   # ablación
```

Requiere Ollama con `qwen2.5:7b`. Cada respuesta guarda la decisión, la confianza,
los claims, el razonamiento, el prompt completo y el veredicto de la compuerta de
consistencia.

### 5. Experimento principal — degradación controlada

```sh
# Representación numérica (2.835 llamadas) y textual (2.835)
python scripts/run_degradation_sweep.py
python scripts/run_degradation_sweep.py --representation text

# Análisis (incluye H1–H6 y el contraste de representaciones)
python scripts/analyze_degradation.py \
  --compare-with results/output/WESAD/PHASE2/degradation_text
```

### 6. Validación externa

```sh
python scripts/run_p3_kemocon.py --reference --reference-only   # K-EmoCon
python scripts/run_p3_physionet.py --per-class 150             # PhysioNet 2017
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

## 📁 Estructura

```
ACL2027/
├── main.py                      # Punto de entrada (PROCESS_REGISTRY)
├── config.json / config_example.json
├── docs/                        # Documentación (empezar por docs/README.md)
├── scripts/                     # Experimentos y análisis
│   ├── run_degradation_sweep.py     # barrido C0–C5 (numérico/textual)
│   ├── analyze_degradation.py       # H1–H6 + contraste de representaciones
│   ├── run_p3_kemocon.py            # validación externa K-EmoCon
│   ├── run_p3_physionet.py          # validación externa PhysioNet 2017
│   ├── analyze_consistency.py       # compuerta de consistencia
│   ├── analyze_calibration.py       # calibración y conformal
│   ├── run_e12_pilot.py             # representación centrada (exploratorio)
│   ├── resolvability_analysis.py    # separabilidad y confusor temporal
│   └── evaluate_phase3.py           # evaluación agregada del corpus
├── sources/
│   ├── common/                  # logging, config, rutas, procedencia, validación
│   ├── ingestion/               # pipeline INGEST (WESAD)
│   ├── phase2/                  # prompts, LLM, esquema, controles, gate
│   ├── phase3/                  # evaluación, claims, certeza, calibración
│   └── external/                # adaptadores K-EmoCon y PhysioNet 2017
├── tests/                       # 227 tests (sin dependencia de los dataset)
└── results/                     # NO versionado: input/ y output/ por subject
```

---

## 🧪 Tests

```sh
python -m pytest -q
```

227 tests: infraestructura, ingesta, features, representaciones, prompts,
esquema de respuesta, orquestador, consistencia, calibración, claims/certeza y
adaptadores externos. No requieren descargar WESAD ni Ollama.

---

## Notas

- `results/` y `config.json` están gitignorados: contienen datos y credenciales.
- Las etiquetas viven solo en `results/output/{subject}/INGEST/groundtruth/`; no
  se mezclan con features, representaciones ni prompts.
- El detalle de decisiones, desviaciones declaradas y estado de cada
  experimento está en [`docs/README.md`](docs/README.md).
