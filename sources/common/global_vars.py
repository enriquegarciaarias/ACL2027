"""
Variables globales de configuración y rutas del proyecto.

Este módulo declara los contenedores globales (argumentos, rutas de directorios
y estado del control de proceso) que comparten todos los módulos.
`procCtrl` se instancia en `sources.common.common` y se referencia aquí como
almacén compartido.

Este es un esqueleto reutilizable para cualquier proyecto basado en este
patrón de construcción (ACL2027 y futuros).
"""

args = None
dirs = None
dirs = {
    'BASE_ROOT': '../',
    'OUT_DIR': "results",
    'TEXT_DIR': "txt",
    'HTML_DIR': "html",
}
proc_ctrl = {}
