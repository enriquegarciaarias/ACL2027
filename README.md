# ACL2027

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20prototype-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-323-passing-brightgreen)

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

Detalle y tablas completas: [`docs/experimentos.md`](docs/experimentos.md) (resumen, inventario y detalle de todos los experimentos) y [`docs/resultados_degradacion.md`](docs/resultados_degradacion.md) (experimento principal).

---

## 📚 Documentación — empieza por la hipótesis

Toda la documentación vive en **`docs/`**, con un índice pensado para dos lectores en [`docs/README.md`](docs/README.md).

| Si quieres… | Empieza por |
|---|---|
| **Entender la pregunta y las hipótesis de la tesis** | [`docs/hipotesis.md`](docs/hipotesis.md) |
| **Ver qué se construyó, qué se midió y qué es firme** | [`docs/experimentos.md`](docs/experimentos.md) |
| **Revisar los resultados completos del experimento principal** | [`docs/resultados_degradacion.md`](docs/resultados_degradacion.md) |
| **Saber qué es firme y qué es exploratorio** | [`docs/split_dev_test.md`](docs/split_dev_test.md) |
| **Conocer el pipeline de datos (Fase 1)** | [`docs/ingestion.md`](docs/ingestion.md) |
| **Saber qué datos hay que copiar al servidor** | [`docs/datos_entrada.md`](docs/datos_entrada.md) |
| **Ver el mapa completo de documentos** | [`docs/README.md`](docs/README.md) |

El README de [`docs/`](docs/README.md) describe la arquitectura por componentes (datos, diseño experimental, resultados, validación externa, literatura y memoria operativa) y enlaza cada documento con su estado (firme / exploratorio / negativo documentado). Es el punto de entrada para quien quiera entrar en detalle.

---

## 🚀 Cómo replicar el experimento

### 1. Entorno

**Python 3.12** (la versión del servidor de GPU y del entorno de referencia).
Los pinneos de `requirements.txt` tienen wheels `cp312` para todas las
dependencias, incluidas `numpy 2.5.4`, `scipy 1.18.1`, `torch 2.15.0` y
`triton 3.9.0`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest (opcional, para tests)
python -m pytest -q                   # 323 tests
```

Comprobación rápida de versiones:

```sh
python -V                              # Python 3.12.x
python -c "import numpy, scipy, neurokit2, sklearn, torch; print(numpy.__version__, torch.__version__)"
```

Si prefieres no descargar las ruedas CUDA 13 de `torch` (la inferencia la hace
Ollama; `torch` solo detecta GPU), instala torch CPU después de los requisitos:

```sh
grep -v -E '^(torch|triton|nvidia-|cuda-)' requirements.txt > /tmp/req-notorch.txt
pip install -r /tmp/req-notorch.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
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

### 4. Fase 2 — respuestas del LLM (`GENERATE`)

Tres fases, un verbo cada una: `INGEST` (corpus) → `GENERATE` (respuestas) → `EVALUATE` (métricas). El **experimento** es un parámetro (`--experiment`), no una fase: `controls`, `degradation`, `centering`, `physionet`, `kemocon`.

```sh
# Experimento `controls`: C0 + 4 controles negativos (el antiguo PHASE2)
python main.py --proc GENERATE --experiment controls --subject WESAD --subjects S5,S6
python main.py --proc GENERATE --experiment controls --subject WESAD --subjects S5 --noGate

# Experimento principal: degradación controlada C0–C5 (H1–H6)
python main.py --proc GENERATE --experiment degradation --subject WESAD --representation numeric
python main.py --proc GENERATE --experiment degradation --subject WESAD --representation text

# Validación externa
python main.py --proc GENERATE --experiment physionet --per-class 150
python main.py --proc GENERATE --experiment kemocon --task arousal
```

`PHASE2` sigue funcionando como alias deprecado de `GENERATE --experiment controls`. Cada respuesta guarda decisión, confianza, claims, razonamiento, prompt completo y veredicto de la compuerta de consistencia.

### 5. Fase 3 — evaluación (`EVALUATE`)

```sh
# Métricas del experimento (según cuál sea)
python main.py --proc EVALUATE --experiment degradation --representation numeric
python main.py --proc EVALUATE --experiment controls
python main.py --proc EVALUATE --experiment physionet

# Comparar modelos: tabla por condición + diferencia de accuracy (bootstrap por sujeto)
python main.py --proc EVALUATE --experiment degradation \
  --compare-models qwen2.5:7b,qwen3:14b
python main.py --proc EVALUATE --experiment controls \
  --compare-models qwen2.5:7b,qwen3:14b
```

Salidas: `results/output/WESAD/PHASE2/degradation{,_text}/{modelo}/`, los reportes agregados y `model_comparison.json`.

### 6. Elegir modelo

```sh
python main.py --proc GENERATE --experiment degradation --subject WESAD --model qwen3:14b
```

Cada modelo escribe en su propio subdirectorio (`qwen3:14b` → `qwen3_14b`), así que no sobrescribe al anterior. El modelo de referencia del corpus publicado es `qwen2.5:7b`; el modo *thinking* de `qwen3` está desactivado por defecto (`--think` lo activa).

### Flags de CLI

| Flag | Descripción |
|---|---|
| `--proc` | Fase: `INGEST`, `GENERATE` o `EVALUATE` (alias: `PHASE2`, `PHASE3`). |
| `--experiment` | `controls` (def.), `degradation`, `centering`, `physionet`, `kemocon`. |
| `--subject` | Carpeta en `results/input/` (default: `default`). |
| `--subjects` | Subconjunto de sujetos (`S5,S6`). |
| `--step` | Inicia el pipeline de ingesta desde un paso con nombre. |
| `--force` | Recalcula artefactos aunque el fingerprint coincida. |
| `--dry-run` | Imprime el plan (fase, modelo, sujetos, llamadas estimadas, salidas) y termina. |
| `--model` | Modelo LLM (def. `defaults.llm.primary_model`). Aísla salidas por modelo. |
| `--think` | Activa el modo *thinking* (solo `qwen3`); desactivado por defecto. |
| `--compare-models` | `EVALUATE`: modelos a comparar (`a,b`); el primero es la referencia. |
| `--representation` | `numeric` o `text` (experimento `degradation`). |
| `--conditions` | Condiciones separadas por comas. |
| `--split` | `evaluation`, `development` o `all` (al evaluar). |
| `--max-windows` | Límite de ventanas por sujeto (pilotos). |
| `--max-records` | Límite de registros (`physionet`). |
| `--per-class` | Registros por clase (`physionet`). |
| `--task` | Tarea afectiva (`kemocon`). |
| `--tag` | Sufijo de ficheros de salida (`physionet`, `kemocon`, `centering`). |
| `--raw-decisions` | Evalúa la decisión cruda del modelo (sin compuerta). |
| `--noGate` | Desactiva la compuerta de consistencia (ablación). |
| `--allowDev` | Permite sujetos de desarrollo (S2–S4). |

### Si el pipeline termina en segundos y el log parece vacío

El proceso **resuelve todas las rutas contra el directorio de trabajo** y
`config.json` está gitignorado, así que un clon reciente suele fallar antes de
escribir nada. Comprobación rápida:

```bash
cd /ruta/al/ACL2027            # 1. cwd = raíz del repo
ls config.json                 # 2. si no está: cp config_example.json config.json
ls results/input/WESAD         # 3. dataset: results/input/WESAD/S*/S*.pkl
python main.py --proc GENERATE --experiment controls --dry-run   # 4. preflight
echo $?                        # 5. 0 = ok · 2 = falta config.json · 1 = otro fallo
```

- El log real es **`ProcessLog.txt` en el directorio de trabajo** (no el del
  wrapper); si está vacío, el proceso murió antes de configurar el logging
  (caso típico: falta `config.json`).
- `--dry-run` imprime el plan con las rutas y un bloque `warnings` que indica si
  falta el dataset, además del número estimado de llamadas.
- `results/input` y `results/output` se crean automáticamente; el dataset no.
- El proceso devuelve **código de salida** (0/1/2). Si el wrapper no lo
  comprueba, un fallo parece una finalización normal:

```bash
set -euo pipefail
python main.py --proc GENERATE --experiment controls --subject WESAD 2>&1 | tee run.log
```

---

## 📁 Estructura del proyecto

```
ACL2027/
├── main.py                      # Punto de entrada (PROCESS_REGISTRY)
├── config.json / config_example.json
├── docs/                        # Documentación (empezar por docs/README.md)
│   ├── hipotesis.md             # Pregunta de tesis, hipótesis H1–H6, objetivos
│   ├── experimentos.md          # Resumen ejecutivo + inventario de experimentos
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
│   ├── analyze_p3_physionet.py      # análisis de PhysioNet 2017 (claims, calidad, ablación)
│   ├── compare_models.py            # comparación de modelos (tabla + bootstrap por sujeto)
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
├── tests/                       # 323 tests (sin dependencia de los datasets reales)
└── results/                     # NO versionado: input/ y output/ por subject
```

---

## 🧪 Tests

```sh
python -m pytest -q
```

**323 tests**: infraestructura, ingesta, features, representaciones, prompts, esquema de respuesta, orquestador, consistencia, calibración, claims/certeza y adaptadores externos. No requieren descargar WESAD ni Ollama.

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