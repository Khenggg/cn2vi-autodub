#!/usr/bin/env bash
set -euo pipefail

# Scripts to manage the CN2VI AutoDub Web UI server in background
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${AUTODUB_DATA_ROOT:-/data}}"
PID_FILE="${DATA_DIR}/run/web.pid"
LOG_FILE="${DATA_DIR}/logs/web.log"
PYTHON_BIN="${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}/core/bin/python"
export DATA_DIR
# Keep uploads/review available while fresh-model validation is still pending.
export ENABLE_PIPELINE="${ENABLE_PIPELINE:-false}"
export MODELS_DIR="${MODELS_DIR:-${DATA_DIR}/models}"
export VENVS_DIR="${VENVS_DIR:-${AUTODUB_VENV_ROOT:-/opt/autodub/venvs}}"
umask 077

mkdir -p "${DATA_DIR}/run" "${DATA_DIR}/logs"

start_server() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            echo "[INFO] Web UI is already running with PID ${PID}."
            echo "[INFO] Access locally via: http://127.0.0.1:8080"
            return 0
        else
            rm -f "${PID_FILE}"
        fi
    fi

    if [[ -z "${ADMIN_TOKEN:-}" ]]; then
        # Generate a fresh token; legacy launchers stored a shared public default.
        ADMIN_TOKEN="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
    fi
    export ADMIN_TOKEN
    printf '%s\n' "${ADMIN_TOKEN}" > "${DATA_DIR}/run/admin_token.txt"
    chmod 600 "${DATA_DIR}/run/admin_token.txt"

    echo "[INFO] Starting CN2VI AutoDub Web UI..."
    cd "${REPO_DIR}"
    # CTranslate2/ONNX use the pinned CUDA libraries from these isolated environments.
    local cuda_dirs=""
    for env_name in asr separation tts; do
        for library_dir in "$VENVS_DIR/$env_name"/lib/python3.12/site-packages/nvidia/*/lib; do
            if [[ -d "$library_dir" ]]; then cuda_dirs="${cuda_dirs:+$cuda_dirs:}$library_dir"; fi
        done
    done
    export LD_LIBRARY_PATH="${cuda_dirs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    nohup "${PYTHON_BIN}" -m uvicorn autodub.main:create_app --factory \
        --host 127.0.0.1 --port 8080 --workers 1 \
        >> "${LOG_FILE}" 2>&1 &
    
    PID=$!
    echo "${PID}" > "${PID_FILE}"
    sleep 2

    if kill -0 "${PID}" 2>/dev/null; then
        echo "[SUCCESS] Web UI started successfully (PID: ${PID})!"
        echo "[INFO] Admin token is stored in ${DATA_DIR}/run/admin_token.txt (owner only)."
        echo "[INFO] Logs: ${LOG_FILE}"
        echo "[INFO] Forward port from local PC:"
        echo "       ssh -L 8080:127.0.0.1:8080 <username>@<server_ip>"
        echo "[INFO] Then open browser at: http://localhost:8080"
    else
        echo "[ERROR] Failed to start Web UI. Check logs:"
        tail -n 20 "${LOG_FILE}"
        exit 1
    fi
}

stop_server() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            echo "[INFO] Stopping Web UI (PID: ${PID})..."
            kill "${PID}" || true
            # Allow the scheduler to cancel children and persist its checkpoint.
            for ((attempt=0; attempt<70; attempt++)); do
                if ! kill -0 "${PID}" 2>/dev/null; then break; fi
                sleep 1
            done
            if kill -0 "${PID}" 2>/dev/null; then
                echo "[ERROR] Shutdown is still pending; preserving PID file and process." >&2
                return 1
            fi
            rm -f "${PID_FILE}"
            echo "[SUCCESS] Web UI stopped."
        else
            echo "[INFO] Process ${PID} not running. Removing stale PID file."
            rm -f "${PID_FILE}"
        fi
    else
        echo "[INFO] Web UI is not running."
    fi
}

status_server() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if kill -0 "${PID}" 2>/dev/null; then
            echo "[STATUS] Web UI is RUNNING (PID: ${PID})."
            echo "[STATUS] Listening on: 127.0.0.1:8080"
            if [ -f "${DATA_DIR}/run/admin_token.txt" ]; then
                echo "[INFO] Admin token file: ${DATA_DIR}/run/admin_token.txt"
            fi
            return 0
        fi
    fi
    echo "[STATUS] Web UI is STOPPED."
    return 1
}

logs_server() {
    if [ -f "${LOG_FILE}" ]; then
        tail -f -n 50 "${LOG_FILE}"
    else
        echo "[WARN] Log file ${LOG_FILE} does not exist yet."
    fi
}

case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        start_server
        ;;
    status)
        status_server
        ;;
    logs)
        logs_server
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
