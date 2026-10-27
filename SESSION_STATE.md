# ACL2027 - Estado de Sesión
**Fecha:** 2026-10-26
**Última actualización:** Cierre de sesión — esqueleto base creado y validado

---

## ✅ ESTADO ACTUAL: ESQUELETO BASE CREADO (CIERRE DE SESIÓN)

El proyecto ACL2027 se ha inicializado con el esqueleto común heredado de
PSRI y SLIP, incorporando las mejoras propuestas. La estructura está completa
y todos los archivos Python compilan correctamente.

### Qué salió bien en esta sesión

- Esqueleto base de ACL2027 creado completo, siguiendo el patrón común de
  PSRI y SLIP.
- Todos los archivos Python compilan correctamente (`py_compile` OK).
- `config_example.json` validado como JSON correcto.
- Estructura de directorios, módulos comunes, documentación y tests creados.
- `requirements.txt` con dependencias base definidas.

### Qué falló en esta sesión

- Al ejecutar `python main.py` directamente en el entorno base del sistema
  falló por falta de `colorama` (y `pip` no está disponible en ese entorno).
  **No es un fallo del esqueleto**: `colorama` es una dependencia de runtime
  que se instala en el `.venv` del proyecto mediante
  `pip install -r requirements.txt`, exactamente como en PSRI y SLIP.

### Cambios en estructura de directorios, módulos, configuración o flujos

**Nuevos archivos y directorios creados:**

```
ACL2027/
├── config_example.json          # Config global de ejemplo (gitignorado el real)
├── main.py                      # Orquestador: PROCESS_REGISTRY, --step, --serverGPU
├── README.md                    # Operativa y estructura
├── AGENTS.md                    # Memoria para agentes (este archivo)
├── SESSION_STATE.md             # Bitácora operativa (este archivo)
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

# 2. Crear configuración real (gitignorado)
cp config_example.json config.json
# editar config.json con rutas, tokens y ajustes de LLM

# 3. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. (Cuando existan procesos) Ejecutar
python main.py --proc ACL2027 --subject default
```

---

### Pendiente

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

