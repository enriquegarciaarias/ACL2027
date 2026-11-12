#!/bin/bash
# launch_lsi.sh - Lanza un job Slurm y monitoriza su estado hasta el final.
#
# Uso:
#   ./launch_lsi.sh [--proc SCRIPT] [--interval SEG] [-h|--help]
#
# Ejemplos:
#   ./launch_lsi.sh                                   # scripts/run_ACL.sh, refresco 30 s
#   ./launch_lsi.sh --interval 15
#   ./launch_lsi.sh --proc run_ACL.sh                 # mismo script, explícito
#   ./launch_lsi.sh --proc ~/data/run_ACL.sh --interval 15
#   ./launch_lsi.sh --proc run_slip_gpu.sh            # compatibilidad con SLIP
#
# El script se busca, en este orden: el directorio de este lanzador
# (`scripts/`), el directorio actual, `~/data` y `~`.

# ---------- Si nos invocan con sh (dash), re-ejecutar con bash ----------
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- Colores ----------
if [[ -t 1 ]]; then
    C_RESET=$'\e[0m'; C_BOLD=$'\e[1m'; C_GREEN=$'\e[32m'
    C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_CYAN=$'\e[36m'; C_DIM=$'\e[2m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_DIM=""
fi

log()  { printf '%s[%s]%s %s\n' "$C_DIM" "$(date +%H:%M:%S)" "$C_RESET" "$*"; }
info() { printf '%s%s%s\n' "$C_CYAN" "$*" "$C_RESET"; }
ok()   { printf '%s%s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s%s%s\n' "$C_YELLOW" "$*" "$C_RESET"; }
err()  { printf '%s%s%s\n' "$C_RED" "$*" "$C_RESET"; }

usage() {
    cat <<EOF
Uso: $(basename "$0") [opciones]

Opciones:
  --proc SCRIPT        Script Slurm a enviar (por defecto: run_ACL.sh).
                       Se busca en $SCRIPT_DIR, \$PWD, \$HOME/data y \$HOME,
                       además de rutas absolutas.
  --interval SEG       Intervalo de refresco en segundos (por defecto: 30).
  -h, --help           Muestra esta ayuda.

Ejemplos:
  $(basename "$0")
  $(basename "$0") --proc run_ACL.sh
  $(basename "$0") --proc run_ACL.sh --interval 15
  $(basename "$0") --proc ~/data/run_ACL.sh --interval 15
EOF
}

# ---------- Parseo de argumentos ----------
PROC=""
INTERVAL=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --proc)
            [[ $# -ge 2 && "$2" != -* ]] || { err "ERROR: --proc requiere un argumento"; exit 2; }
            PROC="$2"; shift 2 ;;
        --proc=*)
            PROC="${1#*=}"
            [[ -n "$PROC" ]] || { err "ERROR: --proc requiere un argumento"; exit 2; }
            shift ;;
        --interval)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { err "ERROR: --interval requiere un entero"; exit 2; }
            INTERVAL="$2"; shift 2 ;;
        --interval=*)
            INTERVAL="${1#*=}"
            [[ "$INTERVAL" =~ ^[0-9]+$ ]] || { err "ERROR: --interval requiere un entero"; exit 2; }
            shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            err "ERROR: opción desconocida: $1"; usage; exit 2 ;;
    esac
done

PROC="${PROC:-run_ACL.sh}"

# ---------- Resolución del script ----------
resolve_script() {
    local name="$1"
    local candidates=()

    if [[ "$name" = /* ]]; then
        candidates=("$name")
    elif [[ "$name" == */* ]]; then
        # Ruta relativa con directorio: se respeta tal cual y luego en ~/data
        candidates=("$name" "$HOME/data/$name")
    else
        # Solo nombre: primero junto a este lanzador, luego el cwd y ~/data
        candidates=("$SCRIPT_DIR/$name" "$PWD/$name" "$HOME/data/$name" "$HOME/$name")
    fi

    local c
    for c in "${candidates[@]}"; do
        c="${c/#\~/$HOME}"
        if [[ -f "$c" ]]; then
            (cd "$(dirname "$c")" && printf '%s/%s\n' "$(pwd)" "$(basename "$c")")
            return 0
        fi
    done
    return 1
}

if ! SCRIPT=$(resolve_script "$PROC"); then
    err "ERROR: no encuentro el script '$PROC'."
    err "  Rutas buscadas:"
    err "    - $SCRIPT_DIR/$PROC"
    err "    - $PWD/$PROC"
    err "    - $HOME/data/$PROC"
    err "    - $HOME/$PROC"
    exit 1
fi

# ---------- Comprobaciones ----------
command -v sbatch >/dev/null 2>&1 || { err "ERROR: sbatch no está en PATH"; exit 1; }
[[ -f "$SCRIPT" ]] || { err "ERROR: no existe '$SCRIPT'"; exit 1; }

# ---------- Lectura de directivas #SBATCH ----------
sbatch_directive() {
    local script="$1" key="$2"
    grep -E "^#SBATCH[[:space:]]+--${key}([=[:space:]]|$)" "$script" 2>/dev/null \
        | head -n1 \
        | sed -E "s/^#SBATCH[[:space:]]+--${key}[=[:space:]]+//" \
        | sed -E 's/[[:space:]]+#.*$//' \
        | sed -E 's/^"//; s/"$//' \
        | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
        || true
}

JOBNAME=$(sbatch_directive "$SCRIPT" "job-name");   JOBNAME="${JOBNAME:-unknown}"
PARTITION=$(sbatch_directive "$SCRIPT" "partition")
OUT_TPL=$(sbatch_directive "$SCRIPT" "output")
ERR_TPL=$(sbatch_directive "$SCRIPT" "error")

# ---------- Expandir plantillas de Slurm ----------
expand_slurm_pattern() {
    local pat="$1" jid="$2" jname="$3"
    pat="${pat/#\~/$HOME}"
    pat="${pat//%j/$jid}"
    pat="${pat//%A/$jid}"
    pat="${pat//%x/$jname}"
    pat="${pat//%u/$USER}"
    # %a (array task) y %N (nodo) se dejan literales; no soportados
    printf '%s' "$pat"
}

# Crear directorios de salida/error antes del submit
for tpl in "$OUT_TPL" "$ERR_TPL"; do
    [[ -z "$tpl" ]] && continue
    tpl="${tpl/#\~/$HOME}"
    mkdir -p "$(dirname "$tpl")" 2>/dev/null || true
done
mkdir -p "$HOME/data/logs" 2>/dev/null || true   # fallback

# ---------- Envío ----------
info "== Enviando job =="
info "   Script:    $SCRIPT"
info "   JobName:   $JOBNAME"
[[ -n "$PARTITION" ]] && info "   Partition: $PARTITION"
echo

SUBMIT_OUT=$(sbatch "$SCRIPT" 2>&1) || { err "Fallo al enviar: $SUBMIT_OUT"; exit 1; }
echo "$SUBMIT_OUT"

JOBID=$(printf '%s' "$SUBMIT_OUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+' | tail -n1 || true)
[[ -n "${JOBID:-}" ]] || { err "ERROR: no pude extraer el JobID de: $SUBMIT_OUT"; exit 1; }
ok "JobID: $JOBID"

# ---------- Resolución de rutas de log ----------
if [[ -n "$OUT_TPL" ]]; then
    OUT_FILE=$(expand_slurm_pattern "$OUT_TPL" "$JOBID" "$JOBNAME")
else
    OUT_FILE="$HOME/data/logs/${JOBNAME}-${JOBID}.out"
fi
if [[ -n "$ERR_TPL" ]]; then
    ERR_FILE=$(expand_slurm_pattern "$ERR_TPL" "$JOBID" "$JOBNAME")
else
    ERR_FILE="$HOME/data/logs/${JOBNAME}-${JOBID}.err"
fi

info "   OUT: $OUT_FILE"
info "   ERR: $ERR_FILE"
echo

# ---------- Trap ----------
cleanup() {
    echo
    warn "Monitorización interrumpida. El job $JOBID SIGUE EN COLA/EJECUCIÓN."
    warn "  Ver estado:  squeue -j $JOBID"
    warn "  Cancelar:    scancel $JOBID"
    exit 0
}
trap cleanup INT TERM

# ---------- Partición para el monitor ----------
MON_PART="$PARTITION"
if [[ -z "$MON_PART" ]]; then
    # %P puede devolver "A5000*" (partición por defecto): el asterisco rompe squeue -p
    MON_PART=$(squeue -h -j "$JOBID" -o "%P" 2>/dev/null | head -n1 | tr -d ' *' || true)
fi

# ---------- Helper: mostrar tail de un fichero si existe y tiene contenido ----------
show_tail() {
    local file="$1" label="$2" n="${3:-3}"
    if [[ -f "$file" && -s "$file" ]]; then
        printf '\n%s--- últimas %d líneas de %s [%s] ---%s\n' \
            "$C_DIM" "$n" "$(basename "$file")" "$label" "$C_RESET"
        tail -n "$n" "$file"
        return 0
    fi
    return 1
}

# ---------- Bucle de monitorización ----------
TRIES=0
LAST_STATE=""
EMPTY_STREAK=0
while true; do
    LINE=$(squeue -h -j "$JOBID" -o "%T|%R|%M|%N" 2>/dev/null || true)

    if [[ -z "$LINE" ]]; then
        # Un vacío transitorio no debe darse por terminado.
        EMPTY_STREAK=$((EMPTY_STREAK + 1))
        if (( EMPTY_STREAK < 3 )); then
            sleep "$INTERVAL"
            continue
        fi
        break
    fi
    EMPTY_STREAK=0

    STATE=$(cut -d'|' -f1 <<<"$LINE")
    REASON=$(cut -d'|' -f2 <<<"$LINE")
    ELAPSED=$(cut -d'|' -f3 <<<"$LINE")
    NODES=$(cut -d'|' -f4 <<<"$LINE")

    case "$STATE" in
        PENDING|CONFIGURING)
            if [[ "$LAST_STATE" != "$STATE" ]]; then
                echo
                warn "== Estado: $STATE (en cola) =="
                TRIES=0
            fi
            TRIES=$((TRIES+1))
            printf '\r%s[%s] %-11s motivo: %-20s ciclo: %d%s' \
                "$C_YELLOW" "$(date +%H:%M:%S)" "$STATE" "$REASON" "$TRIES" "$C_RESET"

            if (( TRIES % 4 == 1 )); then
                echo
                if [[ -n "$MON_PART" ]]; then
                    echo "  --- Cola de la partición $MON_PART ---"
                    squeue -p "$MON_PART" -o "%.10i %.9u %.2t %.10M %.6D %R" 2>/dev/null | head -n 15
                    POS=$(squeue -p "$MON_PART" -h -o "%i %u" 2>/dev/null | grep -n " $USER$" | head -n1 | cut -d: -f1 || true)
                    [[ -n "${POS:-}" ]] && echo "  Puesto aproximado en cola: $POS"
                else
                    echo "  --- Cola (sin partición conocida) ---"
                    squeue -o "%.10i %.9u %.2t %.10M %.6D %R" 2>/dev/null | head -n 15
                fi
                START=$(squeue -h -j "$JOBID" --start 2>/dev/null || true)
                if [[ -n "${START:-}" && "$START" != "N/A" ]]; then
                    echo "  Inicio estimado: $START"
                else
                    echo "  Inicio estimado: N/A (sin estimación todavía)"
                fi
                if [[ -n "$MON_PART" ]]; then
                    echo "  --- Estado de la partición ---"
                    sinfo -p "$MON_PART" -o "%20P %5a %.10l %16F %N" 2>/dev/null | head -n 5
                fi
            fi
            ;;
        RUNNING|COMPLETING)
            if [[ "$LAST_STATE" != "$STATE" ]]; then
                echo
                ok "== Estado: $STATE en nodo $NODES =="
            fi
            printf '\r%s[%s] %-11s nodo: %-15s tiempo: %s%s' \
                "$C_GREEN" "$(date +%H:%M:%S)" "$STATE" "$NODES" "$ELAPSED" "$C_RESET"

            SHOWED=0
            show_tail "$OUT_FILE" "stdout" 3 && SHOWED=1
            show_tail "$ERR_FILE" "stderr" 3 && SHOWED=1
            if (( SHOWED == 0 )); then
                printf '\n%s(esperando salida del job...)%s\n' "$C_DIM" "$C_RESET"
            fi
            ;;
        *)
            printf '\r[%s] %s %s' "$(date +%H:%M:%S)" "$STATE" "$REASON"
            ;;
    esac

    LAST_STATE="$STATE"
    sleep "$INTERVAL"
done

# ---------- Post-mortem ----------
echo
echo "==============================================="
info "El job $JOBID ya no está en la cola. Resumen:"

if command -v sacct >/dev/null 2>&1; then
    sacct -j "$JOBID" -X --format=JobID,JobName,State,ExitCode,Elapsed,NodeList 2>/dev/null || true
fi

FINAL=$(sacct -j "$JOBID" -X -n -o State 2>/dev/null | head -n1 | tr -d ' ' || true)
EXITCODE=$(sacct -j "$JOBID" -X -n -o ExitCode 2>/dev/null | head -n1 | tr -d ' ' || true)

if [[ "$FINAL" == "COMPLETED" && "$EXITCODE" == "0:0" ]]; then
    ok "== Job completado con éxito ($JOBNAME) =="
else
    warn "== Job finalizado con estado: ${FINAL:-desconocido} (ExitCode: ${EXITCODE:-?}) =="
    if [[ -f "$ERR_FILE" ]]; then
        echo "--- últimas 20 líneas de stderr ---"
        tail -n 20 "$ERR_FILE"
    fi
fi

if [[ -f "$OUT_FILE" ]]; then
    echo "--- últimas 30 líneas de $OUT_FILE ---"
    tail -n 30 "$OUT_FILE"
else
    err "No se generó el fichero de log: $OUT_FILE"
    echo "Posibles ubicaciones alternativas:"
    echo "  - $PWD/slurm-${JOBID}.out (fallback de Slurm)"
    echo "  - sacct -j $JOBID --format=JobID,State,ExitCode,Reason"
fi
