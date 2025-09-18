#!/usr/bin/env bash

set -euo pipefail

# Defaults
DEBUGPY_ENABLED="${DEBUGPY:-0}"
DEBUG_PORT="${DEBUG_PORT:-5678}"

APP_PATH="/sandbox/agent_runner.py"

if [ "$DEBUGPY_ENABLED" = "1" ] || [ "$DEBUGPY_ENABLED" = "true" ]; then
  echo "[entrypoint] Starting with debugpy on 0.0.0.0:${DEBUG_PORT} (waiting for client)"
  exec python -m debugpy --wait-for-client --listen 0.0.0.0:"${DEBUG_PORT}" "${APP_PATH}"
else
  echo "[entrypoint] Starting without debugpy"
  exec python "${APP_PATH}"
fi


