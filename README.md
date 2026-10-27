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
- `config.json` (gitignorado) — configuración global con rutas, tokens y
  ajustes de LLM (crear a partir de `config_example.json`).
- `results/input/{subject}/` — entrada por asunto de investigación.
- `results/output/{subject}/` — salidas intermedias y finales.
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
python main.py --proc ACL2027 --subject mi_investigacion
```

Flags disponibles:

| Flag | Descripción |
|------|-------------|
| `--subject` | Carpeta en `results/input/` (default: `default`). |
| `--proc` | Tipo de proceso; dispara el pipeline registrado. |
| `--step` | Inicia el pipeline desde un paso con nombre (case-insensitive). |
| `--serverGPU` | `start` = parte interactiva local; `server` = parte GPU. |

Los resultados se escriben en `results/output/{SUBJECT}/` y los logs en
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
│   └── common/
│       ├── __init__.py
│       ├── common.py            # logging + processControl (singleton)
│       ├── paramsManager.py     # configuración, args, GPU, entorno
│       ├── utils.py             # utilidades genéricas + configLoader
│       └── llm_client.py        # cliente LLM con timeout/fallback
├── results/
│   ├── input/{subject}/         # Entrada por investigación
│   │   ├── config.json          # Config específica del subject (merge profundo)
│   │   └── ...
│   └── output/{subject}/        # Salidas generadas
└── tests/                       # Tests unitarios
    └── __init__.py
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
