#!/usr/bin/env bash
# Launcher for the REOH Daily voice bot on Jetson (Orin NX, JetPack 5,
# aarch64). Mirrors smallwebrtc_bot/run-jetson.sh — same LD_PRELOAD dance for
# the same two aarch64-specific issues:
#
#   1. ctranslate2 ships a bundled libgomp
#      (ctranslate2.libs/libgomp-<hash>.so.1.0.0). On aarch64 glibc, loading
#      it lazily fails with "cannot allocate memory in static TLS block"
#      because earlier-loaded libraries have already consumed the surge
#      capacity. Preloading libgomp first claims its TLS slot up front.
#
#   2. JetPack-built torch wheels ship libc10.so / libtorch.so with symbols
#      the linker fails to resolve when some dependency (e.g. torchaudio)
#      is loaded before torch itself. Preloading libc10.so makes those
#      symbols globally visible.
#
# Order matters: libgomp must be listed before libc10.so so it gets its
# TLS allocation before anything else pulls torch in.
#
# Usage:
#   ./reoh_bot/run-jetson.sh                  # bind localhost:7861
#   HOST=0.0.0.0 PORT=7861 ./reoh_bot/run-jetson.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export HOST="${HOST:-localhost}"
export PORT="${PORT:-7861}"

# Jetson-friendly service defaults — override in the environment if the
# model choice or device placement needs to change.
#
# STT_DEVICE defaults to cpu on aarch64 because the PyPI ctranslate2
# wheels for aarch64 are built without CUDA support. Running on GPU
# requires building ctranslate2 from source against JetPack's cuDNN.
if [[ "$(uname -m)" == "aarch64" ]]; then
  export STT_DEVICE="${STT_DEVICE:-cpu}"
else
  export STT_DEVICE="${STT_DEVICE:-cuda}"
fi
# `int8_float32` keeps ctranslate2 weights in int8 (fast on CPU) but does
# math in float32 (better numerical accuracy than pure int8). Drop to plain
# `int8` if Whisper is the dominant latency on your hardware.
export STT_COMPUTE_TYPE="${STT_COMPUTE_TYPE:-int8_float32}"
# Model accuracy ladder for Jetson Orin NX (CPU, no CUDA):
#   tiny.en          ~5x faster than small.en, very low accuracy
#   base.en          ~3x faster than small.en, usable accuracy   ← default
#   distil-small.en  ~base.en speed, ~small.en accuracy (recommended)
#   small.en         best accuracy, ~3x slower than base.en
# Raise STT_MODEL if transcripts are wrong; lower it if STT is the bottleneck.
export STT_MODEL="${STT_MODEL:-base.en}"
# Borderline segments are still common on CPU; keep the silence threshold
# permissive so noisy utterances still yield a transcript.
export STT_NO_SPEECH_PROB="${STT_NO_SPEECH_PROB:-0.7}"
export PIPER_VOICE="${PIPER_VOICE:-en_US-ryan-high}"
export PIPER_MODEL_DIR="${PIPER_MODEL_DIR:-$PWD/models/piper}"
export LLM_MODEL="${LLM_MODEL:-claude-haiku-4-5}"
# Bump the user-turn fallback timeout well above the Jetson STT latency
# budget. If STT finishes after this timeout, the transcript is silently
# dropped and the bot appears unresponsive.
export USER_TURN_STOP_TIMEOUT="${USER_TURN_STOP_TIMEOUT:-30}"

if [[ "$(uname -m)" == "aarch64" ]]; then
  PRELOADS=()

  CT2_GOMP="$(uv run python - <<'PY'
import glob
import os
import ctranslate2

libs_dir = os.path.join(os.path.dirname(ctranslate2.__file__), os.pardir, "ctranslate2.libs")
libs_dir = os.path.abspath(libs_dir)
matches = sorted(glob.glob(os.path.join(libs_dir, "libgomp-*.so*")))
print(matches[0] if matches else "")
PY
)"
  if [[ -n "${CT2_GOMP}" && -f "${CT2_GOMP}" ]]; then
    PRELOADS+=("${CT2_GOMP}")
  else
    echo "[run-jetson] warning: ctranslate2 bundled libgomp not found; STT may fail to load" >&2
  fi

  TORCH_LIB="$(uv run python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  if [[ -f "${TORCH_LIB}/libc10.so" ]]; then
    PRELOADS+=("${TORCH_LIB}/libc10.so")
  fi

  if (( ${#PRELOADS[@]} > 0 )); then
    JOINED="$(IFS=:; echo "${PRELOADS[*]}")"
    export LD_PRELOAD="${LD_PRELOAD:-}${LD_PRELOAD:+:}${JOINED}"
  fi
  echo "[run-jetson] aarch64 detected; LD_PRELOAD=${LD_PRELOAD:-}"
fi

exec uv run python -m reoh_bot.app
