#!/usr/bin/env bash
# Launcher for the SmallWebRTC voice agent server.
#
# Usage:
#   ./server/run.sh                # bind localhost:7860 with --reload
#   HOST=0.0.0.0 PORT=7860 ./server/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-localhost}"
PORT="${PORT:-7860}"
RELOAD_FLAG="${RELOAD_FLAG:---reload}"

exec uv run uvicorn server.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    ${RELOAD_FLAG}
