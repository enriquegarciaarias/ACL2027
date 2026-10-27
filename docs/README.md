# Documentación — ACL2027

Este directorio contiene la documentación del proyecto.

## Índice

- [README](../README.md) — descripción general, operativa y estructura.
- [AGENTS.md](../AGENTS.md) — contexto para agentes (memoria del proyecto).
- [SESSION_STATE.md](../SESSION_STATE.md) — bitácora operativa.
- [CHANGELOG.md](../CHANGELOG.md) — historial de cambios.

## Convenciones

Todos los módulos del proyecto deben seguir el patrón común del esqueleto:

1. Resolver rutas a través de `processControl.env` (poblado por
   `paramsManager.manageEnv()`). Está prohibido hardcodear `"results"` o
   `"results/input"`.
2. Usar `writeLog(nivel, logger, mensaje)` para todos los mensajes de log.
3. Acceder a la configuración a través de `processControl.defaults` y
   `processControl.env`.
4. Los procesos se registran en `PROCESS_REGISTRY` de `main.py` y se
   importan de forma diferida.
