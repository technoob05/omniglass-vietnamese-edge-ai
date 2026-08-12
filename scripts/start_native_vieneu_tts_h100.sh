#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
VENV_ROOT="${OPENGLASS_VIENEU_VENV:-${ROOT}/.venv-vieneu}"
PYTHON="${VENV_ROOT}/bin/python"
SERVICE="${OPENGLASS_VIENEU_SERVICE:-${ROOT}/native_vieneu_tts_service.py}"
RUN_ROOT="${ROOT}/run"
LOG_ROOT="${ROOT}/logs"
PORT="${OPENGLASS_VIENEU_PORT:-18782}"
BACKBONE_ROOT="${OPENGLASS_VIENEU_BACKBONE:-${ROOT}/models/VieNeu-TTS-v3-Turbo}"
MOSS_ROOT="${OPENGLASS_MOSS_TOKENIZER:-${ROOT}/models/MOSS-Audio-Tokenizer-Nano}"

[[ -x "${PYTHON}" ]] || { echo "Missing VieNeu Python: ${PYTHON}" >&2; exit 1; }
[[ -f "${SERVICE}" ]] || { echo "Missing VieNeu service: ${SERVICE}" >&2; exit 1; }
[[ -d "${BACKBONE_ROOT}" ]] || { echo "Missing VieNeu model: ${BACKBONE_ROOT}" >&2; exit 1; }
[[ -d "${MOSS_ROOT}" ]] || { echo "Missing MOSS tokenizer: ${MOSS_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if ! "${PYTHON}" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', ${PORT})); s.close()"; then
  echo "Port ${PORT} is already in use; refusing to replace an existing service." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${PYTHON}" "${SERVICE}" \
  --host 127.0.0.1 --port "${PORT}" \
  --device cuda --backend pytorch --dtype auto --max-batch-size 1 \
  --backbone-repo "${BACKBONE_ROOT}" --moss-tokenizer "${MOSS_ROOT}" \
  >"${LOG_ROOT}/vi_tts_vieneu.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/vi_tts_vieneu.pid"

for _ in {1..900}; do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${PORT}/health"
    echo
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for VieNeu on port ${PORT}. See ${LOG_ROOT}/vi_tts_vieneu.log" >&2
exit 1
