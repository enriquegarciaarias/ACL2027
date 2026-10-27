# AGENTS.md — Contexto del proyecto ACL2027

Este archivo resume el esqueleto del proyecto **ACL2027** para que cualquier
agente pueda comenzar a trabajar sin reaprender la estructura. Úsalo como
contexto inicial en cada nueva sesión.

---

## 1. Qué es este esqueleto

ACL2027 es la base reutilizable para iniciar proyectos del grupo, construida
sobre el patrón común heredado de **PSRI** y **SLIP**. Proporciona:

- Punto de entrada (`main.py`) con despacho por `--proc`.
- Control de proceso global (`processControl`, singleton) en
  `sources/common/common.py`.
- Gestión de configuración, argumentos CLI, detección de GPU y entorno en
  `sources/common/paramsManager.py`.
- Utilidades genéricas (`configLoader`, `inicioModulo`, JSON, hashing) en
  `sources/common/utils.py`.
- Cliente LLM con timeout, reintentos y fallback en cascada en
  `sources/common/llm_client.py`.
- Configuración global en `config.json` (gitignorado; crear desde
  `config_example.json`) con opción de merge profundo de configuración por
  subject (`results/input/{subject}/config.json`).

**Punto de entrada:** `main.py` → bloque `if __name__ == '__main__'`.

---

## 2. Estructura de directorios

```
results/input/{subject}/              # Entrada por asunto de investigación
├── config.json                       # Config específica del subject (merge profundo sobre la global)
├── studyDescription.json             # (opcional) Descripción del estudio
└── ...

results/output/{subject}/             # Salidas intermedias y finales
└── ...

sources/                              # todos los módulos del proyecto
├── common/
│   ├── common.py                     # logging + processControl
│   ├── paramsManager.py              # configuración, args, GPU, entorno
│   ├── utils.py                      # utilidades genéricas + configLoader
│   └── llm_client.py                 # cliente LLM (timeout, fallback)
config.json (raíz, gitignorado)       # config global; crear a partir de config_example.json
config_example.json
requirements.txt
main.py                               # orquestador
SESSION_STATE.md                      # bitácora operativa de la última sesión
AGENTS.md                             # este archivo
```

Regla clave: todos los módulos deben resolver rutas a través de
`processControl.env` (poblado por `paramsManager.manageEnv()`). Está
prohibido hardcodear `"results"` o `"results/input"` en las rutas de archivos.

---

## 3. Configuración (`config.json`)

El repositorio **no incluye** `config.json` real (contiene claves privadas).
Crearlo con:

```bash
cp config_example.json config.json
```

Cada subject puede tener su propio `results/input/{subject}/config.json` que
hace **merge profundo** sobre el global (claves del subject sobrescriben las
globales).

---

## 4. Ejecución

```bash
source .venv/bin/activate
python main.py --proc ACL2027 --subject default
```

Flags:
- `--subject`: carpeta en `results/input/` (default `default`).
- `--proc`: tipo de proceso; dispara el pipeline registrado en `PROCESS_REGISTRY`.
- `--step`: iniciar el pipeline directamente desde un paso con nombre.
- `--serverGPU`: `start` (parte interactiva local) o `server` (parte GPU).

---

## 5. processControl (singleton)

Objeto global accesible desde cualquier módulo:

```python
from sources.common.common import processControl, logger, writeLog

processControl.args       # argumentos de CLI (--subject, --proc, --step, --serverGPU)
processControl.defaults   # valores por defecto de config.json
processControl.env        # rutas resueltas del entorno
processControl.datasetVars# variables del dataset (con timestamp)
processControl.parms      # parámetros adicionales
```

---

## 6. Cómo usar este archivo en futuras sesiones

Al iniciar una sesión nueva, pega el contenido de este archivo (`AGENTS.md`)
junto con `SESSION_STATE.md` como contexto inicial al agente.

Cuando modifiques código, configuración o flujos, **actualiza `AGENTS.md` y
`SESSION_STATE.md`** al terminar la sesión para mantener la memoria
sincronizada.

---

## 7. Estado de esta sesión (cierre — 2026-10-27)

### Qué salió bien en esta sesión
- Repositorio git inicializado localmente con `git init` y rama `main`.
- Primer commit realizado (`60ca167`): 22 archivos, 2869 líneas, incluyendo
  esqueleto base, módulos comunes, documentación y `.gitignore`.
- `.gitignore` verificado: excluye correctamente `config.json`, `.venv/`,
  `results/`, `paper/`, `__pycache__/`, `.idea/` y otros archivos no versionables.
- Remote configurado: `origin -> https://github.com/enriquegarciaarias/ACL2027.git`

### Qué falló en esta sesión
- El `git push -u origin main` falló por falta de autenticación:
  `could not read Username for 'https://github.com': No existe el dispositivo o la dirección`.
  El commit existe localmente pero aún no se subió a GitHub.
- (Previo, ya documentado) Al ejecutar `python main.py` directamente en el
  entorno base del sistema falló por falta de `colorama`; no es un fallo del
  esqueleto, se resuelve instalando dependencias en el `.venv`.

### Cambios en estructura de directorios, módulos, configuración o flujos
- No hubo cambios en la estructura de archivos; solo se inicializó el control
  de versiones git y se preparó el remoto hacia GitHub.

### Próximos pasos sugeridos (pendientes)

1. Completar el push a GitHub. Opciones:
   - Token PAT:
     ```bash
     git remote set-url origin https://<TOKEN>@github.com/enriquegarciaarias/ACL2027.git
     git push -u origin main
     ```
   - SSH:
     ```bash
     git remote set-url origin git@github.com:enriquegarciaarias/ACL2027.git
     git push -u origin main
     ```
   - GitHub CLI: `gh auth login && git push -u origin main`
2. Crear `config.json` a partir de `config_example.json` y editarlo con rutas y tokens.
3. Crear el entorno e instalar dependencias:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Registrar los procesos del proyecto en `PROCESS_REGISTRY` de `main.py`.
5. Implementar los módulos del pipeline bajo `sources/`.
6. Añadir tests unitarios en `tests/`.
7. Documentar el pipeline en `docs/`.
