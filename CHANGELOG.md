# Changelog — ACL2027

## [0.1.0] — 2026-10-26

### Añadido
- Esqueleto base del proyecto ACL2027 (patrón común heredado de PSRI y SLIP).
- `main.py`: punto de entrada con despacho por `--proc`, `PROCESS_REGISTRY`,
  soporte a `--step` y `--serverGPU`, `customProcess()`.
- `sources/common/common.py`: logging con colores e iconos, singleton
  `processControl`.
- `sources/common/paramsManager.py`: gestión de configuración, argumentos CLI,
  detección de GPU, `deep_merge` para configuración por subject, `results_root`.
- `sources/common/utils.py`: utilidades genéricas, `configLoader`, `inicioModulo`,
  `read_json`/`write_json`, hashing (sha1/sha256).
- `sources/common/llm_client.py`: cliente LLM con timeout, reintentos y
  fallback en cascada (mejora propuesta).
- `config_example.json`: ejemplo de configuración global.
- `requirements.txt`, `.gitignore`, `README.md`, `AGENTS.md`,
  `SESSION_STATE.md`, `CHANGELOG.md`.
- `docs/README.md`, `tests/__init__.py`, `results/.gitkeep`.

### Mejoras propuestas incorporadas al esqueleto común
- Timeout por llamada LLM para evitar bloqueos indefinidos.
- Fallback en cascada entre modelos (primary → fallbacks).
- Soporte a `results_root` para reubicar `results/`.
- Prohibición de hardcodear `"results"` en rutas (uso de `processControl.env`).
- Merge profundo de configuración por subject sobre la global.
- Iconos visuales en logs warning/error para lectura en servidores sin color.
- `--step` para reanudar desde un paso específico.
- `--serverGPU` para dividir ejecución local ↔ servidor GPU.
