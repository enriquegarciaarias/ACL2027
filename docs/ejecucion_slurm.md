# Ejecución en el clúster (Slurm)

Cómo se lanza el pipeline en el servidor. Dos scripts, en `scripts/`:

| Script | Qué es | Se usa con |
|---|---|---|
| [`run_ACL.sh`](../scripts/run_ACL.sh) | trabajo Slurm (lleva directivas `#SBATCH`) | `sbatch` o a través del lanzador |
| [`launch_lsi.sh`](../scripts/launch_lsi.sh) | lanzador + monitor en vivo | desde el shell de login |

---

## 0. Dónde vive cada cosa

**El repositorio es la fuente; `~/data` es el despliegue.** Los dos scripts viven
en `scripts/` (ahí se editan y se versionan), pero en el servidor **se ejecutan
desde `~/data`**, junto a los de los demás proyectos. El proyecto y su entorno
están en otro sitio.

| Qué | Ruta en el servidor |
|---|---|
| Lanzador y trabajo Slurm (**despliegue**) | `~/data/launch_lsi.sh`, `~/data/run_ACL.sh` (nombres confirmados en el servidor, 2026-11-13) |
| Fuente versionada de esos scripts | `~/data/proyectos/ACL2027/scripts/` |
| Proyecto (código, `config.json`, `results/`) | `~/data/proyectos/ACL2027` |
| Entorno virtual | `~/data/proyectos/ACL2027/.venv` |
| Logs de todos los proyectos | `~/data/logs` |

Los comandos de este documento usan **`~/data/…`**: es la ruta que existe en el
servidor. `./scripts/launch_lsi.sh` solo sirve en el repo de desarrollo y
**falla o ejecuta una copia distinta** en el servidor. Tras editar los del repo,
se despliegan así:

```bash
cp scripts/run_ACL.sh scripts/launch_lsi.sh ~/data/
```

Por eso `run_ACL.sh` **no deduce el proyecto de su propia ruta**: Slurm copia el
trabajo a `/var/spool/slurmd/jobXXXX/slurm_script`, así que `$BASH_SOURCE`
apunta al spool y no al directorio desde el que se envió. Se resuelve así:

```bash
PROJECT_DIR="${ACL2027_PROJECT:-$HOME/data/proyectos/ACL2027}"
```

Se puede fijar explícitamente con `--project` en el lanzador (lo propaga a Slurm
con `sbatch --export=ALL,ACL2027_PROJECT=...`) o con la variable de entorno
`ACL2027_PROJECT` al usar `sbatch` directamente:

```bash
ACL2027_PROJECT=~/data/proyectos/ACL2027 sbatch ~/data/run_ACL.sh --proc INGEST --subject WESAD
```

> **Los ejemplos de este documento van en una sola línea a propósito.** Si se
> parte un comando con `\` y al pegar queda un espacio detrás de la barra, el
> `\` escapa el espacio en lugar del salto de línea: la primera parte se ejecuta
> truncada y el resto se interpreta como comandos sueltos
> (`--chdir=...: No such file or directory`). Vale también para los `--` del
> lanzador: si el `\` falla, el trabajo se queda sin argumentos.

## 1. Lanzador `launch_lsi.sh`

Envía el trabajo y sigue su estado hasta que termina: cola, nodo asignado,
últimas líneas del log y resumen final con `sacct`.

No hay valores por defecto: **`--proc` es obligatorio** y la fase se indica
siempre detrás de `--`. El comando dice exactamente qué se lanza.

```bash
# corpus
~/data/launch_lsi.sh --proc run_ACL.sh -- --proc INGEST --subject WESAD

# mismo, refresco cada 15 s
~/data/launch_lsi.sh --proc run_ACL.sh --interval 15 -- --proc INGEST --subject WESAD

# respuestas del LLM (12 h: el barrido completo supera las 4 h del script)
~/data/launch_lsi.sh --proc run_ACL.sh --time 12:00:00 -- --proc GENERATE --experiment degradation --subject WESAD

# empezar de cero
~/data/launch_lsi.sh --proc run_ACL.sh -- --proc INGEST --subject WESAD --fresh --yes

# otro script (p. ej. SLIP)
~/data/launch_lsi.sh --proc ~/data/run_slip_gpu.sh --interval 15

~/data/launch_lsi.sh --help
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `--proc SCRIPT` | **obligatorio** | Script Slurm a enviar |
| `--time LIMITE` | el `#SBATCH --time` del script | Tiempo máximo del job (`12:00:00`); imprescindible en barridos largos |
| `--project DIR` | `~/data/proyectos/ACL2027` | Proyecto (venv, config, results); se exporta como `ACL2027_PROJECT` |
| `--interval SEG` | `30` | Refresco del monitor |
| `-- ARGS...` | — | Argumentos para el script Slurm (y de ahí a `main.py`) |
| `-h`, `--help` | — | Ayuda |

Sin `--proc` el lanzador sale con código **2** y muestra el uso. La separación
es deliberada: `--proc` antes de `--` es *qué script*; lo que va después de `--`
es *qué fase del pipeline*.

> **Pon el comando en una sola línea.** Si lo partes con `\` al final, un espacio
> detrás de la barra la anula y bash ejecuta cada línea por separado
> (`--proc: no se encontró la orden`). El lanzador detecta el caso de `--` sin
> argumentos y se detiene con un mensaje, pero mejor evitarlo.
>
> **Cuidado al copiar comandos largos.** Si el terminal los parte por ancho,
> bash ejecuta el fragmento sobrante como comando aparte y verás líneas como
> `14b: no se encontró la orden` o `representation: no se encontró la orden`
> mientras `main.py` recibe un argumento truncado (`qwen3:` en lugar de
> `qwen3:14b`). Comprueba con `echo` antes de enviar, o usa comandos cortos
> (p. ej. `python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b`).

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
sbatch ~/data/run_ACL.sh --proc INGEST   --subject WESAD
sbatch ~/data/run_ACL.sh --proc INGEST   --subject WESAD --noEmbeddings
sbatch ~/data/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD
sbatch ~/data/run_ACL.sh --proc EVALUATE --experiment degradation --subject WESAD
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

### Ollama: arranque, reutilización y limpieza

El arranque es **el mismo que ya funcionaba en el servidor** (`run_slip_gpu.sh`):
`ollama serve` en segundo plano, 5 s de cortesía y sondeo con `curl -s` a
`/api/tags`. Lo único que se añade son tres cosas, ninguna cambia el arranque:

1. **Tope de espera**: 24 × 5 s = **2 min** (`OLLAMA_ATTEMPTS`). En el servidor
   el arranque en frío medido es de **~35 s**, así que 60 s se quedaban cortos y
   un servidor sano parecía caído. Sin tope, un `until` sin límite podía
   consumir las 4 h de pared.
   El `curl` lleva `--max-time 10`: el original no tenía timeout y una primera
   respuesta lenta (NFS frío, muchos modelos) con un tope corto contaría como
   "no listo" y alargaría el arranque artificialmente.
2. **Reutilización**: si ya hay un Ollama respondiendo en `127.0.0.1:11434`
   (habitual tras `scancel`: el servidor del trabajo anterior sigue vivo en el
   nodo), se reutiliza y se avisa. No se apaga al terminar, porque no lo arrancó
   este trabajo.
3. **Diagnóstico y limpieza**: si no llega a responder, el `.out` incluye el
   final de `~/data/logs/ollama_<jobid>.log`, que es donde está el motivo real.
   Al salir se cierran los hijos (`pkill -P`) y el servidor, con traps
   `EXIT/INT/TERM/HUP`, de modo que `scancel` también limpia.

### Ollama: dónde están los modelos y cómo añadirlos

El store **no** es `~/.ollama/models`. `run_ACL.sh` exporta
`OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/data/.ollama/models}"`, así que los
modelos viven en `~/data/.ollama/models` y ese es el directorio que hay que
mirar (y el que comparten todos los trabajos, porque está en `$HOME`).

```bash
# inventario sin necesidad de que Ollama esté corriendo
for d in ~/data/.ollama/models/manifests/registry.ollama.ai/library/*; do
  for t in "$d"/*; do printf '%s:%s\n' "$(basename "$d")" "$(basename "$t")"; done
done
du -sh ~/data/.ollama/models
```

**Consulta el inventario antes de hacer un `pull`.** Un `pull` puede fallar por
DNS (`dial tcp: lookup registry.ollama.ai: i/o timeout`) aunque el modelo **ya
esté instalado**; en ese caso el fallo no significa nada. Solo hay que descargar
lo que no aparezca en la lista de arriba.

Para añadir uno, el `pull` va **donde haya red** (los nodos de cálculo y a veces
el propio login dan `no resuelve DNS`). En `sirius` el binario no está en el
`PATH` del shell: lo añade el propio trabajo (`run_ACL.sh`), así que la receta
incluye los tres `export` para que el pull escriba en el store compartido.

```bash
export PATH="$HOME/data/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/data/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="$HOME/data/.ollama/models"
mkdir -p "$OLLAMA_MODELS" ~/data/logs

command -v ollama && ollama --version     # ~/data/.local/bin/ollama, cliente 0.40.6

ollama serve > ~/data/logs/ollama_pull.log 2>&1 &
sleep 5
ollama pull <modelo:tag>                  # solo si NO aparece en el inventario
ollama list                               # antes del kill: necesita el serve vivo
kill %1
```

Sin `OLLAMA_MODELS` exportado, la descarga cae en `~/.ollama/models` (otro
store) y el trabajo no la ve. La sugerencia de `snap install ollama` que imprime
el shell no sirve: requiere administrador y crearía una instalación distinta.

Si el modelo no está, `GENERATE` aborta al empezar con el `ollama pull` en el
mensaje; no se escriben `timeout` que luego se reutilizarían al reanudar.

**Inventario del store (2026-11-26)**: `qwen2.5:7b` (4,7 GB), `qwen3:8b`
(5,2 GB), `qwen3:14b` (9,3 GB), `llama3.1:8b` (4,9 GB), `llama3.2:3b` (2,0 GB),
`medgemma:4b` (3,3 GB) y `mxbai-embed-large` (0,7 GB). Es decir, los cuatro
brazos de WESAD y los tres de DREAMT se pueden ejecutar sin descargar nada; el
cliente de Ollama del servidor es **0.40.6** (motor de los resultados citables).

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
sbatch ~/data/run_ACL.sh --proc INGEST --subject WESAD

# Empezar de cero (servidor nuevo, artefactos dudosos)
sbatch ~/data/run_ACL.sh --proc INGEST --subject WESAD --fresh --yes

# Respuestas del LLM (lo caro: horas)
sbatch ~/data/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD

# Métricas, sin LLM
sbatch ~/data/run_ACL.sh --proc EVALUATE --experiment degradation --subject WESAD
```

`--fresh` **no** va en el lanzamiento habitual: borra los 15 GB derivados y
recalcula. Sin él, `INGEST` reutiliza en segundos lo que no ha cambiado
(ver [`ingestion.md`](ingestion.md#empezar-de-cero)).

### `--subject` es obligatorio también con los datasets externos

`run_ACL.sh` exige `--subject` (código 2 sin él) porque es el destino por
defecto en `results/output/{subject}/`. Los runners externos **no lo usan**
—escriben en una ruta fija bajo `results/output/`—, pero el argumento hay que
pasarlo igualmente. Valor recomendado, el mismo que su carpeta de salida:

| Experimento | `--subject` | Salida |
|---|---|---|
| WESAD (cualquier experimento) | `WESAD` | `results/output/WESAD/` |
| `--experiment dreamt` | `DREAMT` | `results/output/DREAMT/PHASE2/{pilot,scale}/` |
| `--experiment physionet` | `PHYSIONET2017` | `results/output/PHYSIONET2017/P3/` |
| `--experiment kemocon` | `KEMOCON` | `results/output/KEMOCON/P3/` |

```bash
# DREAMT, cohorte de escala (5.760 llamadas por modelo; reanudable)
sbatch ~/data/run_ACL.sh --proc GENERATE --experiment dreamt --subject DREAMT --cohort scale --windows-per-class 10 --model qwen2.5:7b

# PhysioNet 2017 (400 llamadas)
sbatch ~/data/run_ACL.sh --proc GENERATE --experiment physionet --subject PHYSIONET2017
```

Sin `--subject`, el trabajo muere en el preflight con `[ERROR] falta --subject`
y `ExitCode: 2:0`: es correcto, no un fallo del entorno.

`EVALUATE` no usa GPU ni Ollama (solo lee artefactos) y tarda minutos, pero **no
puede correrse en el nodo de login**: importa numpy y `sirius` aborta con
`Illegal instruction` (ver §5). Va en un nodo de cálculo, sin cola larga:

```bash
srun --partition=A5000 --gres=gpu:1 --cpus-per-task=4 --mem=8G --time=00:15:00 --pty bash
cd ~/data/proyectos/ACL2027 && source .venv/bin/activate
python scripts/compare_models.py --models qwen2.5:7b,qwen3:14b
```

### Qué se lanza con Slurm y qué se corre en la terminal

Regla: **en el servidor, por defecto se lanza con `~/data/launch_lsi.sh`**.
En la terminal SSH (`sirius`) solo se ejecuta lo que **no importa Python del
proyecto** y tarda segundos. Todo lo que importe numpy —y eso incluye
`--dry-run`, `scripts/analyze_*.py`, `scripts/compare_models.py` y
`main.py --proc EVALUATE`— va a un nodo de cálculo con `srun --pty` (corto) o
con el lanzador (largo). El nodo de login es compartido: nada largo ni
intensivo allí.

| Tarea | Cómo |
|---|---|
| `INGEST`, `GENERATE` (cualquier experimento) | lanzador (Slurm) |
| `GENERATE --think`, barridos de degradación, pilotos con LLM | lanzador (Slurm) |
| `EVALUATE` (cualquier experimento, también los análisis) | `srun --pty` (minutos) o lanzador (horas) |
| `--dry-run` de cualquier fase | `srun --pty` |
| `scripts/compare_models.py`, `scripts/analyze_*.py` | `srun --pty` |
| `bash -n`, `ls`, `grep`, `du`, `sha256sum -c`, `rsync`/`scp` | terminal SSH |

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
| `no existe el entorno en /var/spool/slurmd/.venv` | copia antigua del script que deducía el proyecto de su ruta | actualiza `~/data/run_ACL.sh`; el proyecto se resuelve con `ACL2027_PROJECT` |
| `Faltan dependencias ... neurokit2` (código 3) | se lanzó con otro intérprete | el script activa el `.venv` del proyecto; si se llama a `python` a mano, usa `.venv/bin/python` |
| `gpu_report` con `devices: []` en el nodo de login | la GPU se asigna al trabajo, no al shell | normal: comprueba el bloque de GPU del `.out` |
| `nvidia-smi` no encontrado en el job | binario fuera del PATH | `gpu_report` busca en `/usr/bin`, `/usr/local/bin`, `/usr/local/nvidia/bin` y `/opt/nvidia/bin`, y lee el driver de `/proc/driver/nvidia/version` |
| Ollama tarda ~30-40 s | **arranque en frío** en un nodo recién asignado (medido: ~35 s) | normal; la ventana es de 2 min |
| `Ollama no arrancó` | servicio no levantado, modelo no descargado o puerto ocupado | el `.out` incluye el final de `~/data/logs/ollama_<jobid>.log`; en `GENERATE` el trabajo se aborta |
| `bind: address already in use` | quedó un Ollama de un job anterior | el script lo detecta y lo reutiliza; si es de otro usuario, `scancel` el job o espera |
| Lanzas `ollama serve` a mano y el job "se arregla" | el manual ocupa el puerto; el del job falla al bindear | no lances uno en paralelo: espera los 2 min del script |
| El job muere a las 4 h en mitad de un barrido | el `#SBATCH --time` del script es de 4 h y 2.835 llamadas rondan las 5 | `--time 12:00:00`; el barrido reanuda por `(ventana, condición)` |
| `Illegal instruction (core dumped)` al importar numpy en el nodo de login | la CPU de `sirius` no soporta las instrucciones del wheel (el mismo `.venv` funciona en los nodos de cálculo) | no es el entorno: ejecuta desde `srun --pty` (o con el lanzador); en el login solo shell puro |
| `ls ~/.ollama/models` sale vacío pero los modelos existen | el store del trabajo es **`$HOME/data/.ollama/models`** (`OLLAMA_MODELS` en `run_ACL.sh`), no `~/.ollama` | lista con `ls ~/data/.ollama/models/manifests/registry.ollama.ai/library/*/` (y evita `2>/dev/null`, que oculta el «No such file») |
| `main.py: error: unrecognized arguments: --` (código 2) | se coló un `--` **entre** los argumentos de la fase al copiar el comando; el separador va una sola vez, antes de la fase | desde 2026-11-13 `main.py` lo ignora con aviso y el lanzador avisa antes de enviar; aun así, revisa el comando |
| El job termina en segundos y el log parece vacío | falta `config.json` o el dataset | `python main.py ... --dry-run` en un nodo de cálculo (no en `sirius`) y revisa `paths` y `warnings` |
| `Job completado` pero sin resultados | el pipeline devolvió error y el wrapper no lo miraba | ahora `run_ACL.sh` propaga el código y Slurm marca `FAILED` |
| Un modelo de ~27-32B va a 2-5 tok/s y `ollama ps` dice *CPU/GPU split* | 17-20 GB de pesos **más** la caché KV de 32k tokens no caben en 24 GB | baja `defaults.llm.num_ctx` en `config.json` (los prompts rondan 1.000 tokens: 8192 sobra) y **restáuralo** después; el valor queda en `run_provenance.json`, así que la desviación es auditable |

---

## 6. Reproducir la validación local de los scripts

Sin Slurm se pueden validar sintaxis y flujo. En el servidor, sobre las copias
**desplegadas**:

```bash
bash -n ~/data/run_ACL.sh
bash -n ~/data/launch_lsi.sh
~/data/launch_lsi.sh --help
```

En el repo de desarrollo, sobre las **fuentes**:

```bash
bash -n scripts/run_ACL.sh
bash -n scripts/launch_lsi.sh
./scripts/launch_lsi.sh --help
```

Para ver el fallo temprano de argumentos sin lanzar nada pesado:

```bash
bash ~/data/run_ACL.sh                  # uso + código 2 (faltan --proc y --subject)
```
