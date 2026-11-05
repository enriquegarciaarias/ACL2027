# ACL2027

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Skeleton-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Esqueleto base del proyecto **ACL2027**, construido sobre el patrón común
heredado de los proyectos **PSRI** y **SLIP**.

Este esqueleto constituye la base reutilizable para iniciar cualquier nuevo
proyecto del grupo. Incluye:

- `main.py` — punto de entrada con despacho por `--proc`, soporte a `--step`
  y `--serverGPU`.
- `sources/common/paramsManager.py` — carga de configuración (`config.json`),
  argumentos de CLI, detección de GPU, entorno y `processControl`.
- `sources/common/common.py` — logging con colores, iconos en logs sin color y
  el singleton `processControl`.
- `sources/common/utils.py` — utilidades genéricas (JSON, hashing, rutas,
  `configLoader`, `inicioModulo`).
- `sources/common/llm_client.py` — cliente LLM con timeout, reintentos y
  fallback en cascada (mejora propuesta).
- `sources/common/paths.py` — resolución centralizada de rutas y escritura
  atómica de artefactos.
- `sources/common/provenance.py` — hashes, fingerprints, git y versiones.
- `sources/common/validation.py` — validaciones de esquemas y rangos.
- `sources/ingestion/` — pipeline `INGEST` de WESAD (ver
  [`docs/ingestion.md`](docs/ingestion.md)).
- `config.json` (gitignorado) — configuración global con rutas, tokens y
  ajustes de LLM (crear a partir de `config_example.json`).
- `results/input/{subject}/` — entrada por asunto de investigación.
- `results/output/{subject}/INGEST/` — salidas del pipeline: `canonical/`,
  `qc/`, `windows/`, `features/`, `representations/{numeric,text,embeddings}/`,
  `groundtruth/`, `corpus_*.jsonl`, `corpus_manifest.json`,
  `corpus_checksums.sha256`.
- `docs/` — documentación del proyecto.
- `tests/` — tests unitarios.

---

## 🚀 Operativa

### 1. Entorno

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuración

Copia `config_example.json` en `config.json` (gitignorado) y edítalo con tus
rutas, tokens y ajustes:

```sh
cp config_example.json config.json
```

### 3. Ejecución

```sh
source .venv/bin/activate
python main.py --proc INGEST --subject WESAD
```

Flags disponibles:

| Flag | Descripción |
|------|-------------|
| `--subject` | Carpeta en `results/input/` (default: `default`). |
| `--proc` | Tipo de proceso; dispara el pipeline registrado. |
| `--step` | Inicia el pipeline desde un paso con nombre (case-insensitive). |
| `--serverGPU` | `start` = parte interactiva local; `server` = parte GPU. |
| `--subjects` | Subconjunto de sujetos del dataset (p. ej. `S2,S3`). |
| `--force` | Recalcula artefactos aunque el fingerprint coincida. |
| `--dry-run` | Resuelve configuración y rutas sin ejecutar. |

Pipeline de ingesta (Fase 1 completa: `discover → ingest → qc → window →
features → represent → package`):

```sh
python main.py --proc INGEST --subject WESAD --dry-run
python main.py --proc INGEST --subject WESAD --step discover
python main.py --proc INGEST --subject WESAD --subjects S2 --step package
```

`--step X` ejecuta el pipeline hasta X inclusive. Ver
[`docs/ingestion.md`](docs/ingestion.md) para el contrato de artefactos y las
decisiones metodológicas.

Los resultados se escriben en `results/output/{SUBJECT}/{PROC}/` y los logs en
`ProcessLog.txt` / `Process.txt`.

---

## 📁 Estructura de directorios

```
ACL2027/
├── config.json                  # Config global (gitignorado; crear desde config_example.json)
├── config_example.json          # Ejemplo de configuración
├── main.py                      # Orquestador / punto de entrada
├── requirements.txt             # Dependencias Python
├── README.md                    # Este archivo
├── AGENTS.md                    # Contexto para agentes (memoria del proyecto)
├── SESSION_STATE.md             # Bitácora operativa de la última sesión
├── CHANGELOG.md                 # Historial de cambios
├── .gitignore
├── docs/                        # Documentación
│   └── README.md
├── sources/                     # Todos los módulos del proyecto
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── common.py            # logging + processControl (singleton)
│   │   ├── paramsManager.py     # configuración, args, GPU, entorno
│   │   ├── utils.py             # utilidades genéricas + configLoader
│   │   ├── llm_client.py        # cliente LLM con timeout/fallback
│   │   ├── paths.py             # rutas centralizadas + escritura atómica
│   │   ├── provenance.py        # hashes, fingerprints, git, versiones
│   │   └── validation.py        # validaciones de esquemas/rangos
│   └── ingestion/               # pipeline INGEST (WESAD)
│       ├── pipeline.py          # orquestador y registro de pasos
│       ├── config.py            # configuración resuelta
│       ├── schemas.py           # contratos de artefactos
│       ├── discover.py          # manifiesto + hashes
│       ├── loader.py            # carga/validación/canonicalización
│       ├── qc.py                # control de calidad por canal
│       ├── window.py            # segmentación temporal
│       ├── units.py             # conversión y verificación de unidades
│       ├── represent.py         # representaciones numérica, textual y embebidos
│       ├── package.py           # índice, manifiesto y checksums
│       └── features/            # extracción de features + SQI
│           ├── registry.py      # registro de modalidades
│           ├── ecg.py  eda.py  resp.py  temp.py
│           ├── acc.py  emg.py  bvp.py
│           ├── sqi.py           # indicadores de calidad
│           └── common.py        # utilidades y FeatureBuilder
├── results/
│   ├── input/{subject}/         # Entrada por investigación
│   │   ├── config.json          # Config específica del subject (merge profundo)
│   │   └── ...
│   └── output/{subject}/INGEST/
│       ├── canonical/           # arrays .npz + .meta.json
│       ├── qc/                # control de calidad por canal
│       ├── windows/           # segmentación temporal (JSONL)
│       ├── features/          # features por ventana (JSONL)
│       ├── representations/
│       │   ├── numeric/       # JSON autodescriptivo por ventana
│       │   ├── text/          # texto estructurado por ventana
│       │   └── embeddings/    # vectores densos vía Ollama
│       ├── groundtruth/       # etiquetas separadas (anti-leakage)
│       ├── corpus_index.jsonl
│       ├── corpus_manifest.json
│       └── corpus_checksums.sha256
└── tests/                       # Tests unitarios
    ├── __init__.py
    └── test_*.py
└── tests/                       # Tests unitarios
    ├── __init__.py
    └── test_*.py
```

> Regla clave: todos los módulos deben resolver rutas a través de
> `processControl.env` (poblado por `paramsManager.manageEnv()`). Está
> prohibido hardcodear `"results"` o `"results/input"` en las rutas de archivo.

---

## 📖 Citar

```
@misc{acl2027,
  title  = {ACL2027 Project Skeleton},
  author = {},
  year   = {2027},
  note   = {Esqueleto base reutilizable (patrón PSRI/SLIP)}
}
```
