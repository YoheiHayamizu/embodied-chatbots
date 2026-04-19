#!/usr/bin/env bash
# Launcher for the SmallWebRTC voice agent server.
#
# Usage:
#   ./smallwebrtc_bot/run.sh                # bind localhost:7860 with --reload
#   HOST=0.0.0.0 PORT=7860 ./smallwebrtc_bot/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-localhost}"
PORT="${PORT:-7860}"
RELOAD_FLAG="${RELOAD_FLAG:---reload}"

exec uv run uvicorn smallwebrtc_bot.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    ${RELOAD_FLAG}
