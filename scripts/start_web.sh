#!/usr/bin/env bash
set -euo pipefail

# Scripts to manage the CN2VI AutoDub Web UI server in background
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${DATA_DIR:-/data}"
PID_FILE="${DATA_DIR}/run/web.pid"
LOG_FILE="${DATA_DIR}/logs/web.log"
PYTHON_BIN="/opt/autodub/venvs/core/bin/python"

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

    export ADMIN_TOKEN="${ADMIN_TOKEN:-p6DGUlHVj9PrTQ3WWJOxIIY5WhMVItTm6bW9h2_WfA0}"
    echo "${ADMIN_TOKEN}" > "${DATA_DIR}/run/admin_token.txt"

    echo "[INFO] Starting CN2VI AutoDub Web UI..."
    cd "${REPO_DIR}"
    nohup "${PYTHON_BIN}" -m uvicorn autodub.main:create_app --factory \
        --host 127.0.0.1 --port 8080 --workers 1 \
        >> "${LOG_FILE}" 2>&1 &
    
    PID=$!
    echo "${PID}" > "${PID_FILE}"
    sleep 2

    if kill -0 "${PID}" 2>/dev/null; then
        echo "[SUCCESS] Web UI started successfully (PID: ${PID})!"
        echo "[KEY] Admin Token: ${ADMIN_TOKEN}"
        echo "[INFO] Logs: ${LOG_FILE}"
        echo "[INFO] Forward port from local Windows PC:"
        echo "       ssh -p 58431 -L 8080:127.0.0.1:8080 ezycloudx-admin@14.169.113.144"
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
            sleep 1
            if kill -0 "${PID}" 2>/dev/null; then
                kill -9 "${PID}" || true
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
                echo "[KEY] Admin Token: $(cat "${DATA_DIR}/run/admin_token.txt")"
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
