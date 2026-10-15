#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_9B_ROOT:?Set OPENGLASS_9B_ROOT to the native OpenGlass runtime root}"
MODEL="${OPENGLASS_9B_MODEL:-${ROOT}/models/MiniCPM-o-4_5-Q4_0.gguf}"
SERVER="${OPENGLASS_9B_SERVER:-${ROOT}/llama.cpp-omni/build/bin/llama-omni-server}"
DEMO="${OPENGLASS_9B_DEMO:-${ROOT}/MiniCPM-o-Demo}"
GPU_LAYERS="${OPENGLASS_9B_GPU_LAYERS:-0}"

for path in "$MODEL" "$SERVER" "$DEMO/worker.py" "$DEMO/gateway.py"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing OpenGlass 9B artifact: $path" >&2
    echo "See OPENGLASS_9B.md for the expected runtime layout." >&2
    exit 2
  fi
done

CHAIN="${OPENGLASS_9B_CHAIN:-${ROOT}/start_openglass_9b.sh}"
if [[ ! -x "$CHAIN" ]]; then
  echo "Missing OpenGlass 9B process-chain launcher: $CHAIN" >&2
  echo "The model file alone is not enough; install MiniCPM-o-Demo and its native chain first." >&2
  exit 2
fi

if [[ "$GPU_LAYERS" == "0" ]]; then
  echo "Starting MiniCPM-o 4.5 (~9B) in CPU mode (-ngl 0)." >&2
else
  echo "Starting MiniCPM-o 4.5 (~9B) with $GPU_LAYERS GPU layers." >&2
fi

exec "$CHAIN" \
  --server "$SERVER" \
  --model "$MODEL" \
  --gpu-layers "$GPU_LAYERS"
