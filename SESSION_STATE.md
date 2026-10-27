# ACL2027 - Estado de Sesión
**Fecha:** 2026-10-26
**Última actualización:** Cierre de sesión — repositorio git inicializado y preparado para GitHub

---

## ✅ ESTADO ACTUAL: ESQUELETO BASE CREADO + GIT INICIALIZADO (CIERRE DE SESIÓN)

El proyecto ACL2027 se ha inicializado con el esqueleto común heredado de
PSRI y SLIP, incorporando las mejoras propuestas. La estructura está completa
y todos los archivos Python compilan correctamente. Además, se ha inicializado
el repositorio git local con su primer commit y el remote apuntando a GitHub.

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
- Al ejecutar `python main.py` directamente en el entorno base del sistema
  falló por falta de `colorama` (y `pip` no está disponible en ese entorno).
  **No es un fallo del esqueleto**: `colorama` es una dependencia de runtime
  que se instala en el `.venv` del proyecto mediante
  `pip install -r requirements.txt`, exactamente como en PSRI y SLIP.

### Cambios en estructura de directorios, módulos, configuración o flujos

**Nuevos archivos y directorios creados (estado anterior):**

```
ACL2027/
├── config_example.json          # Config global de ejemplo (gitignorado el real)
├── main.py                      # Orquestador: PROCESS_REGISTRY, --step, --serverGPU
├── README.md                    # Operativa y estructura
├── AGENTS.md                    # Memoria para agentes
├── SESSION_STATE.md             # Bitácora operativa
├── CHANGELOG.md                 # Historial de cambios
├── requirements.txt             # Dependencias base
├── .gitignore                   # config.json, .venv, results/, logs, __pycache__
├── docs/
│   └── README.md
├── sources/
│   ├── __init__.py
│   └── common/
│       ├── __init__.py
│       ├── common.py            # logging + processControl (singleton)
│       ├── paramsManager.py     # config, args CLI, GPU, entorno, deep_merge, results_root
│       ├── utils.py             # configLoader, inicioModulo, JSON, hashing sha1/sha256
│       └── llm_client.py        # cliente LLM: timeout, reintentos, fallback en cascada
├── results/
│   └── .gitkeep                 # input/{subject}/ y output/{subject}/ (no versionados)
└── tests/
    └── __init__.py
```

**Mejoras propuestas incorporadas al esqueleto común:**
- `--step`: iniciar el pipeline desde un paso con nombre (reanudar tras suspensión).
- `--serverGPU` (`start`/`server`): dividir ejecución local ↔ servidor GPU.
- `llm_client.py`: cliente LLM con timeout, reintentos y fallback en cascada.
- `results_root`: reubicar `results/` entero vía `config.json > environment.results_root`.
- `deep_merge`: configuración por subject fusionada profundamente sobre la global.
- Iconos visuales (`⚠️`, `❌`) en logs warning/error para `sbatch out` sin color.
- Hashing sha256 además de sha1 (útil para checksums de corpus).
- Regla clave: rutas resueltas vía `processControl.env`; prohibido hardcodear `"results"`.

---

## 🔄 CÓMO CONTINUAR EN PRÓXIMA SESIÓN

```bash
# 1. Verificar estado
cd /home/vant/Documentos/proyectos/LSI/ACL2027
cat SESSION_STATE.md
cat AGENTS.md

# 2. Completar el push a GitHub (opción Token PAT)
git remote set-url origin https://<TOKEN>@github.com/enriquegarciaarias/ACL2027.git
git push -u origin main

# Opción SSH:
# git remote set-url origin git@github.com:enriquegarciaarias/ACL2027.git
# git push -u origin main

# Opción GitHub CLI:
# gh auth login && git push -u origin main

# 3. Crear configuración real (gitignorado)
cp config_example.json config.json
# editar config.json con rutas, tokens y ajustes de LLM

# 4. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. (Cuando existan procesos) Ejecutar
python main.py --proc ACL2027 --subject default
```

---

### Pendiente

- [ ] Completar el push a GitHub (autenticación con Token PAT, SSH o GitHub CLI).
- [ ] Crear `config.json` a partir de `config_example.json`.
- [ ] Crear `.venv` e instalar `pip install -r requirements.txt`.
- [ ] Registrar procesos concretos en `PROCESS_REGISTRY` de `main.py`.
- [ ] Implementar módulos del pipeline bajo `sources/`.
- [ ] Añadir tests unitarios en `tests/`.
- [ ] Documentar el pipeline en `docs/`.

---

### Notas importantes

- `config.json` está gitignorado (contiene claves privadas). Nunca versionar
  el archivo real.
- Todas las rutas deben resolverse vía `processControl.env`; prohibido
  hardcodear `"results"` o `"results/input"`.
- Cuando se añadan procesos, módulos, configuración o flujos, documentarlos
  en `AGENTS.md` y `SESSION_STATE.md` al terminar la sesión.

