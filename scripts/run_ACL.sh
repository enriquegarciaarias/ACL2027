#!/bin/bash
#SBATCH --job-name=ACL2027
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --error=/home/egarcia/data/logs/ACL2027-%j.err
#SBATCH --output=/home/egarcia/data/logs/ACL2027-%j.out
#
# run_ACL.sh — trabajo Slurm del proyecto ACL2027.
#
# UBICACIÓN: este fichero es la FUENTE (repo: scripts/run_ACL.sh). En el
# servidor se envía la copia DESPLEGADA en ~/data/run_ACL.sh. Tras editar aquí,
# hay que desplegar: cp scripts/run_ACL.sh scripts/launch_lsi.sh ~/data/
#
# Hace, en este orden: comprueba el entorno y la GPU, arranca Ollama, ejecuta
# una fase del pipeline y lo deja todo limpio (Ollama incluido) al salir.
#
# Uso (los argumentos son OBLIGATORIOS; se pasan tal cual a `main.py`):
#   sbatch scripts/run_ACL.sh --proc INGEST   --subject WESAD
#   sbatch scripts/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD
#   sbatch scripts/run_ACL.sh --proc EVALUATE --experiment degradation --subject WESAD
#
# Sin argumentos, o sin `--proc`/`--subject`, el trabajo no arranca: sale con
# código 2 y muestra el uso. Se prefiere un fallo temprano a ejecutar una fase
# o escribir en results/output/default/ sin querer.
#
# Nota sobre Ollama: es imprescindible para `GENERATE` y para los embeddings.
# Si no arranca y la fase no es `GENERATE`, el trabajo continúa y `represent`
# omite los embeddings (material complementario). El pipeline nunca se cae por
# un Ollama caído.

set -euo pipefail

# ---------- 0. Argumentos obligatorios (fallo temprano) ----------
if (( $# == 0 )); then
    echo "[ERROR] faltan argumentos: hay que indicar la fase del pipeline."
    echo "        uso: sbatch $0 --proc INGEST --subject WESAD"
    echo "             sbatch $0 --proc GENERATE --experiment degradation --subject WESAD"
    echo "             sbatch $0 --proc EVALUATE --experiment degradation --subject WESAD"
    exit 2
fi

HAS_PROC=0
HAS_SUBJECT=0
for arg in "$@"; do
    case "$arg" in
        --proc|--proc=*)       HAS_PROC=1 ;;
        --subject|--subject=*) HAS_SUBJECT=1 ;;
    esac
done

if (( HAS_PROC == 0 )); then
    echo "[ERROR] falta --proc (fase del pipeline: INGEST, GENERATE o EVALUATE)."
    echo "        recibido: $*"
    exit 2
fi

if (( HAS_SUBJECT == 0 )); then
    echo "[ERROR] falta --subject (destino en results/output/{subject}/)."
    echo "        para WESAD: --subject WESAD"
    echo "        recibido: $*"
    exit 2
fi

# ---------- Localización del proyecto ----------
# OJO: NO se deduce de la ruta de este script. Slurm copia el trabajo a
# /var/spool/slurmd/jobXXXX/slurm_script, así que $BASH_SOURCE apunta al spool.
# Este script vive en ~/data (junto al lanzador), mientras que el proyecto y su
# entorno están en otro sitio. Se resuelve por variable de entorno, con la ruta
# del servidor como valor por defecto.
PROJECT_DIR="${ACL2027_PROJECT:-$HOME/data/proyectos/ACL2027}"

echo "=== Iniciando trabajo en $(hostname) a las $(date) ==="
echo "job: ${SLURM_JOB_ID:-?} | proyecto: $PROJECT_DIR"

# Sin buffer: los logs aparecen al instante (clave para el monitor en vivo)
export PYTHONUNBUFFERED=1

# ---------- 1. Entorno virtual ----------
VENV="$PROJECT_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[ERROR] no existe el entorno en $VENV/bin/python"
    echo "        proyecto resuelto: $PROJECT_DIR"
    echo "        se puede cambiar con la variable ACL2027_PROJECT o con"
    echo "        'launch_lsi.sh --project <ruta>' (sbatch --export)"
    echo "        crear el entorno: python3.12 -m venv $VENV"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python: $(command -v python) ($(python -V 2>&1))"

# El proyecto se usa como directorio de trabajo: así `sources` es importable y
# `config.json`/`results` se resuelven igual que en una ejecución manual.
cd "$PROJECT_DIR"

# ---------- 2. GPU: qué reservó Slurm y qué ve el proyecto ----------
# Este bloque responde a la pregunta "¿la GPU existe y es usable?" y deja la
# respuesta en el .out del trabajo. Ninguna comprobación es fatal: la fase 1
# es CPU pura y la inferencia la hace Ollama.
# Se captura antes de recortar: `cmd | head` con `pipefail` puede devolver 141
# (SIGPIPE) y hacer creer que el comando falló.
echo "--- nvidia-smi ---"
if NVIDIA_OUT=$(nvidia-smi 2>&1); then
    head -15 <<<"$NVIDIA_OUT"
else
    echo "  (nvidia-smi no disponible en este nodo)"
fi

echo "--- dispositivos CUDA ---"
if NVIDIA_DEV=$(ls -l /dev/nvidia* 2>&1); then
    head -5 <<<"$NVIDIA_DEV"
else
    echo "  (sin /dev/nvidia*)"
fi

echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-<sin definir>}]"

echo "--- gpu_report (paramsManager) ---"
python -c "from sources.common.paramsManager import gpu_report; import json; print(json.dumps(gpu_report(), indent=1))" \
    || echo "[WARN] gpu_report falló; revisa la instalación del entorno"

# ---------- 3. Ollama ----------
# Se replica el arranque que ya funcionaba en el servidor: `ollama serve` en
# segundo plano, 5 s de cortesía y sondeo con `curl -s`. Lo único que se añade
# es un tope de espera, la reutilización de un servidor ya existente y, si no
# arranca, el final de su log para saber por qué.
export PATH="$HOME/data/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/data/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/data/.ollama/models}"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_LOG_DIR="$HOME/data/logs"
mkdir -p "$OLLAMA_LOG_DIR"
OLLAMA_LOG="$OLLAMA_LOG_DIR/ollama_${SLURM_JOB_ID:-$$}.log"
OLLAMA_URL="http://127.0.0.1:11434"
# Arranque en frío observado en el servidor: ~35 s. 24 × 5 s = 2 min de margen.
OLLAMA_ATTEMPTS="${OLLAMA_ATTEMPTS:-24}"

# `curl -s` como en el script de SLIP, con un tope amplio: el original no tenía
# timeout, y con `--max-time 2` una primera respuesta lenta (NFS frío, muchos
# modelos) contaba como "no listo" y alargaba el arranque artificialmente.
ollama_up() { curl -s --max-time 10 "$OLLAMA_URL/api/tags" > /dev/null; }

OLLAMA_PID=""
cleanup() {
    if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Apagando Ollama (pid $OLLAMA_PID)..."
        # Los runners de GPU son hijos del servidor: se intentan cerrar antes.
        command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$OLLAMA_PID" 2>/dev/null || true
        kill -TERM "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM HUP

OLLAMA_READY=0
if ollama_up; then
    # Caso típico tras cancelar un job: su Ollama sigue vivo en el nodo. Se
    # reutiliza, pero NO se apaga al salir (no lo arrancó este trabajo).
    echo "[WARN] ya hay un Ollama respondiendo en $OLLAMA_URL (¿job anterior cancelado?)."
    echo "       Se reutiliza y NO se apagará al terminar este trabajo."
    OLLAMA_READY=1
elif command -v ollama >/dev/null 2>&1; then
    echo "Arrancando Ollama (log: $OLLAMA_LOG)..."
    ollama serve > "$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!

    sleep 5
    for attempt in $(seq 1 "$OLLAMA_ATTEMPTS"); do
        if ollama_up; then
            echo "Ollama listo (intento $attempt, ~$(( attempt * 5 )) s)."
            OLLAMA_READY=1
            break
        fi
        echo "  ... Ollama aún no responde ($attempt/$OLLAMA_ATTEMPTS)"
        sleep 5
    done

    if (( OLLAMA_READY == 0 )); then
        echo "[ERROR] Ollama no respondió en $(( OLLAMA_ATTEMPTS * 5 )) s."
        echo "--- últimas 20 líneas de $OLLAMA_LOG ---"
        tail -n 20 "$OLLAMA_LOG" 2>/dev/null || echo "  (sin log)"
        echo "----------------------------------------"
    fi
else
    echo "[WARN] ollama no está en PATH; no se arranca"
fi

# ---------- 4. Fase del pipeline ----------
PIPELINE_ARGS=("$@")

if (( OLLAMA_READY == 0 )); then
    case " ${PIPELINE_ARGS[*]} " in
        *" GENERATE "*)
            echo "[ERROR] Ollama no arrancó y la fase GENERATE lo necesita."
            echo "        revisa $OLLAMA_LOG"
            exit 1
            ;;
        *)
            echo "[WARN] Ollama no arrancó; se continúa (los embeddings se omitirán)."
            ;;
    esac
fi

echo "Lanzando: python main.py ${PIPELINE_ARGS[*]}"
set +e
python main.py "${PIPELINE_ARGS[@]}" 2>&1
PIPELINE_EXIT=$?
set -e

echo "Pipeline finalizado con código $PIPELINE_EXIT"
echo "=== Trabajo finalizado a las $(date) ==="

# `trap cleanup EXIT` apaga Ollama también en este camino.
exit "$PIPELINE_EXIT"
