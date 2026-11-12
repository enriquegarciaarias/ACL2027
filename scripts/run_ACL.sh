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
# Hace, en este orden: comprueba el entorno y la GPU, arranca Ollama, ejecuta
# una fase del pipeline y lo deja todo limpio (Ollama incluido) al salir.
#
# Uso:
#   sbatch scripts/run_ACL.sh
#   sbatch scripts/run_ACL.sh --proc INGEST --subject WESAD
#   sbatch scripts/run_ACL.sh --proc GENERATE --experiment degradation --subject WESAD
#   sbatch scripts/run_ACL.sh --proc EVALUATE --experiment degradation
#
# Sin argumentos ejecuta la fase 1 (`INGEST`) del corpus completo. Cualquier
# argumento se pasa tal cual a `main.py`.
#
# Nota sobre Ollama: es imprescindible para `GENERATE` y para los embeddings.
# Si no arranca y la fase no es `GENERATE`, el trabajo continúa y `represent`
# omite los embeddings (material complementario). El pipeline nunca se cae por
# un Ollama caído.

set -euo pipefail

# ---------- Localización del proyecto a partir del propio script ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Iniciando trabajo en $(hostname) a las $(date) ==="
echo "job: ${SLURM_JOB_ID:-?} | proyecto: $PROJECT_DIR"

# Sin buffer: los logs aparecen al instante (clave para el monitor en vivo)
export PYTHONUNBUFFERED=1

# ---------- 1. Entorno virtual ----------
VENV="$PROJECT_DIR/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[ERROR] no existe el entorno en $VENV/bin/python"
    echo "        crea el entorno con: python3.12 -m venv $VENV"
    echo "        o revisa la ruta del proyecto en este script"
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
export PATH="$HOME/data/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/data/.local/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/data/.ollama/models}"
mkdir -p "$OLLAMA_MODELS"

OLLAMA_LOG_DIR="$HOME/data/logs"
mkdir -p "$OLLAMA_LOG_DIR"
OLLAMA_LOG="$OLLAMA_LOG_DIR/ollama_${SLURM_JOB_ID:-$$}.log"

OLLAMA_PID=""
# Limpieza garantizada aunque el pipeline falle (set -e)
cleanup() {
    if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Apagando Ollama (pid $OLLAMA_PID)..."
        kill "$OLLAMA_PID" 2>/dev/null || true
        wait "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

OLLAMA_READY=0
if command -v ollama >/dev/null 2>&1; then
    echo "Arrancando Ollama (log: $OLLAMA_LOG)..."
    ollama serve > "$OLLAMA_LOG" 2>&1 &
    OLLAMA_PID=$!

    # Espera acotada: 12 intentos × 5 s. Antes, un `until` sin límite podía
    # consumir las 4 h de pared esperando a un Ollama que no arranca.
    for attempt in $(seq 1 12); do
        if curl -sf --max-time 2 http://127.0.0.1:11434/api/tags > /dev/null; then
            echo "Ollama listo (intento $attempt)."
            OLLAMA_READY=1
            break
        fi
        echo "  ... Ollama aún no responde ($attempt/12)"
        sleep 5
    done
else
    echo "[WARN] ollama no está en PATH; no se arranca"
fi

# ---------- 4. Fase del pipeline ----------
if (( $# > 0 )); then
    PIPELINE_ARGS=("$@")
else
    PIPELINE_ARGS=(--proc INGEST --subject WESAD)
fi

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
