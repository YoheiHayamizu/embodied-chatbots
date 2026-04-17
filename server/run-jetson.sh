#!/usr/bin/env bash
# Launcher for the SmallWebRTC voice agent server on Jetson (Orin NX,
# JetPack 5, aarch64).
#
# JetPack-built torch wheels ship libc10.so / libtorch.so with symbols the
# linker fails to resolve when some dependency (e.g. torchaudio) is loaded
# before torch itself. Pre-loading libc10.so makes those symbols globally
# visible.
#
# Usage:
#   ./server/run-jetson.sh                  # bind 0.0.0.0:7860
#   HOST=0.0.0.0 PORT=7860 ./server/run-jetson.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-7860}"

# Jetson-friendly service defaults — override in the environment if the
# model choice or device placement needs to change.
export LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
export STT_DEVICE="${STT_DEVICE:-cuda}"
export STT_COMPUTE_TYPE="${STT_COMPUTE_TYPE:-int8}"
export STT_MODEL="${STT_MODEL:-small}"
export PIPER_VOICE="${PIPER_VOICE:-en_US-ryan-high}"
export PIPER_MODEL_DIR="${PIPER_MODEL_DIR:-$PWD/models/piper}"

if [[ "$(uname -m)" == "aarch64" ]]; then
  TORCH_LIB="$(uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  export LD_PRELOAD="${LD_PRELOAD:-}${LD_PRELOAD:+:}${TORCH_LIB}/libc10.so"
  echo "[run-jetson] aarch64 detected; LD_PRELOAD=${LD_PRELOAD}"
fi

exec uv run python -m server.app
