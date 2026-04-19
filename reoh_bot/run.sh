#!/usr/bin/env bash
# Launcher for the REOH Daily voice bot.
#
# Usage:
#   ./reoh_bot/run.sh                     # bind localhost:7861 with --reload
#   HOST=0.0.0.0 PORT=7861 ./reoh_bot/run.sh
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${HOST:-localhost}"
PORT="${PORT:-7861}"
RELOAD_FLAG="${RELOAD_FLAG:---reload}"

exec uv run uvicorn reoh_bot.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    ${RELOAD_FLAG}
