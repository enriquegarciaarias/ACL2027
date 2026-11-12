# Ejecución en el clúster (Slurm)

Cómo se lanza el pipeline en el servidor. Dos scripts, en `scripts/`:

| Script | Qué es | Se usa con |
|---|---|---|
| [`run_ACL.sh`](../scripts/run_ACL.sh) | trabajo Slurm (lleva directivas `#SBATCH`) | `sbatch` o a través del lanzador |
| [`launch_lsi.sh`](../scripts/launch_lsi.sh) | lanzador + monitor en vivo | desde el shell de login |

---

## 1. Lanzador `launch_lsi.sh`

Envía el trabajo y sigue su estado hasta que termina: cola, nodo asignado,
últimas líneas del log y resumen final con `sacct`.

No hay valores por defecto: **`--proc` es obligatorio** y la fase se indica
siempre detrás de `--`. El comando dice exactamente qué se lanza.

```bash
# corpus
./scripts/launch_lsi.sh --proc run_ACL.sh -- --proc INGEST --subject WESAD

# mismo, refresco cada 15 s
./scripts/launch_lsi.sh --proc run_ACL.sh --interval 15 -- --proc INGEST --subject WESAD

# respuestas del LLM
./scripts/launch_lsi.sh --proc run_ACL.sh -- \
    --proc GENERATE --experiment degradation --subject WESAD

# empezar de cero
./scripts/launch_lsi.sh --proc run_ACL.sh -- \
    --proc INGEST --subject WESAD --fresh --yes

# otro script (p. ej. SLIP)
./scripts/launch_lsi.sh --proc ~/data/run_slip_gpu.sh --interval 15

./scripts/launch_lsi.sh --help
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `--proc SCRIPT` | **obligatorio** | Script Slurm a enviar |
| `--interval SEG` | `30` | Refresco del monitor |
| `-- ARGS...` | — | Argumentos para el script Slurm (y de ahí a `main.py`) |
| `-h`, `--help` | — | Ayuda |

Sin `--proc` el lanzador sale con código **2** y muestra el uso. La separación
es deliberada: `--proc` antes de `--` es *qué script*; lo que va después de `--`
es *qué fase del pipeline*.

**Resolución del script** (en este orden): el directorio del lanzador
(`scripts/`), el directorio actual, `~/data` y `~`; las rutas absolutas se
respetan. Por eso funciona desde cualquier `cwd` y también con los scripts de
SLIP en `~/data`.

**Qué hace durante la espera:**

- `PENDING`: cada 4 ciclos muestra la cola de la partición, tu posición
  aproximada, el inicio estimado (`squeue --start`) y el estado de la partición.
- `RUNNING`: refresca en la misma línea y muestra las últimas 3 líneas de
  `.out` y `.err`.
- Al salir de la cola: `sacct` con estado y `ExitCode`, y 20 líneas de `stderr`
  si falló.
- `Ctrl-C` **no cancela** el trabajo: avisa de que sigue vivo y cómo verlo o
  cancelarlo.

---

## 2. Trabajo `run_ACL.sh`

```bash
sbatch scripts/run_ACL.sh --proc INGEST   --subject WESAD
sbatch scripts/run_ACL.sh --proc INGEST   --subject WESAD --noEmbeddings
sbatch scripts/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD
sbatch scripts/run_ACL.sh --proc EVALUATE --experiment degradation --subject WESAD
```

**Los argumentos son obligatorios** y se pasan tal cual a `main.py`. Si faltan
argumentos, `--proc` o `--subject`, el trabajo **no arranca**: sale con código 2
y muestra el uso. Es un fallo temprano a propósito, para no ejecutar una fase
que nadie pidió ni escribir en `results/output/default/` por descuido.

Con el lanzador, esos argumentos van detrás de `--`.

Secuencia:

1. **Localiza el proyecto a partir de sí mismo** (`scripts/` → raíz) y activa
   `.venv`. Si el entorno no existe, sale con mensaje, sin traza de Python.
2. **Comprueba la GPU** y lo deja escrito en el `.out`:
   `nvidia-smi`, `/dev/nvidia*`, `CUDA_VISIBLE_DEVICES` y `gpu_report()`. Ninguna
   comprobación es fatal: la fase 1 es CPU pura.
3. **Arranca Ollama** en segundo plano, con log propio
   (`~/data/logs/ollama_<jobid>.log`), y espera hasta 60 s (12 × 5 s).
4. **Ejecuta el pipeline** y propaga su código de salida.
5. **Apaga Ollama** con un `trap ... EXIT`, también si el pipeline falla.

### Ollama: cuándo es obligatorio

| Fase | ¿Necesita Ollama? | Si no arranca |
|---|---|---|
| `INGEST` con embeddings | sí (complementarios) | continúa; `represent` marca `embeddings: skipped` |
| `INGEST --noEmbeddings` | no | continúa |
| `GENERATE` | **sí** | aborta con código 1 y la ruta del log de Ollama |
| `EVALUATE` | no | continúa |

La espera está acotada a propósito: un `until curl` sin límite podía consumir
las 4 h de pared esperando a un Ollama que no arranca.

### Códigos de salida

Los del pipeline (`0`, `1`, `2`, `3`, `130`) más:

- `1` si falta el `.venv`, si no existe `sbatch` o si `GENERATE` no tiene Ollama.

Slurm marcará el trabajo como `FAILED` cuando el pipeline falle, en lugar de
`COMPLETED` con un error dentro del log.

---

## 3. Qué lanzar y cuándo

```bash
# Corpus (una vez; reutiliza por fingerprint en las siguientes)
sbatch scripts/run_ACL.sh --proc INGEST --subject WESAD

# Empezar de cero (servidor nuevo, artefactos dudosos)
sbatch scripts/run_ACL.sh --proc INGEST --subject WESAD --fresh --yes

# Respuestas del LLM (lo caro: horas)
sbatch scripts/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD

# Métricas, sin LLM
sbatch scripts/run_ACL.sh --proc EVALUATE --experiment degradation --subject WESAD
```

`--fresh` **no** va en el lanzamiento habitual: borra los 15 GB derivados y
recalcula. Sin él, `INGEST` reutiliza en segundos lo que no ha cambiado
(ver [`ingestion.md`](ingestion.md#empezar-de-cero)).

---

## 4. Logs

| Log | Ruta |
|---|---|
| Job (`stdout` / `stderr`) | `/home/egarcia/data/logs/ACL2027-<jobid>.{out,err}` |
| Ollama | `~/data/logs/ollama_<jobid>.log` |
| Pipeline (`ProcessLog.txt`) | raíz del proyecto (directorio de trabajo) |

La ruta de los `#SBATCH --output/--error` es la única que hay que cambiar al
mover los scripts a otra máquina o usuario. El lanzador crea el directorio antes
de enviar el trabajo.

---

## 5. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Faltan dependencias ... neurokit2` (código 3) | se lanzó con otro intérprete | el script activa el `.venv` del proyecto; si se llama a `python` a mano, usa `.venv/bin/python` |
| `gpu_report` con `devices: []` en el nodo de login | la GPU se asigna al trabajo, no al shell | normal: comprueba el bloque de GPU del `.out` |
| `nvidia-smi` no encontrado en el job | binario fuera del PATH | `gpu_report` busca en `/usr/bin`, `/usr/local/bin`, `/usr/local/nvidia/bin` y `/opt/nvidia/bin`, y lee el driver de `/proc/driver/nvidia/version` |
| `Ollama no arrancó` | servicio no levantado o modelo no descargado | mira `~/data/logs/ollama_<jobid>.log`; en `GENERATE` el trabajo se aborta |
| El job termina en segundos y el log parece vacío | falta `config.json` o el dataset | `python main.py ... --dry-run` y revisa `paths` y `warnings` |
| `Job completado` pero sin resultados | el pipeline devolvió error y el wrapper no lo miraba | ahora `run_ACL.sh` propaga el código y Slurm marca `FAILED` |

---

## 6. Reproducir la validación local de los scripts

Sin Slurm se pueden validar sintaxis y flujo:

```bash
bash -n scripts/run_ACL.sh
bash -n scripts/launch_lsi.sh
./scripts/launch_lsi.sh --help
```

Para probar el pipeline dentro del script sin lanzar nada pesado:

```bash
bash scripts/run_ACL.sh --help          # imprime la ayuda de main.py
```
