#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOWHISPER_BENCH_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
VENV="${PHOWHISPER_ASR_VENV:-${ROOT}/.venv}"
SERVICE_SCRIPT="${PHOWHISPER_ASR_SCRIPT:-${ROOT}/scripts/phowhisper_asr_service.py}"
MODEL_ID="${PHOWHISPER_ASR_MODEL_ID:-vinai/PhoWhisper-large}"
MODEL_REVISION="${PHOWHISPER_ASR_MODEL_REVISION:-b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb}"
MODEL_DIR="${PHOWHISPER_ASR_MODEL_DIR:-${ROOT}/models/PhoWhisper-large-${MODEL_REVISION}}"
ENGLISH_MODEL_DIR="${WHISPER_ENGLISH_MODEL_DIR:-/network-volume/icse27/edge-ai/vi-asr-candidates/models/whisper-large-v3-turbo-41f01f3fe87f28c78e2fbf8b568835947dd65ed9}"
HOST="${PHOWHISPER_ASR_HOST:-127.0.0.1}"
PORT="${PHOWHISPER_ASR_PORT:-18783}"
INSTANCE="${PHOWHISPER_ASR_INSTANCE:-$(hostname -s)}"
INSTANCE="${INSTANCE//[^a-zA-Z0-9_.-]/_}"
RUN_DIR="${PHOWHISPER_ASR_RUN_DIR:-${ROOT}/run/${INSTANCE}}"
PID_FILE="${RUN_DIR}/phowhisper-asr-${PORT}.pid"
LOG_FILE="${RUN_DIR}/phowhisper-asr-${PORT}.log"

mkdir -p "${RUN_DIR}"

for required in "${VENV}/bin/python" "${SERVICE_SCRIPT}" "${MODEL_DIR}/pytorch_model.bin" "${ENGLISH_MODEL_DIR}/model.safetensors"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "PhoWhisper ASR already healthy at http://${HOST}:${PORT}"
  exit 0
fi

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "PID ${old_pid} is alive but health failed; refusing to replace it" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

nohup "${VENV}/bin/python" "${SERVICE_SCRIPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-dir "${MODEL_DIR}" \
  --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --english-model-dir "${ENGLISH_MODEL_DIR}" \
  >"${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" >"${PID_FILE}"

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "PhoWhisper ASR ready instance=${INSTANCE} pid=${pid} url=http://${HOST}:${PORT} log=${LOG_FILE}"
    exit 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "PhoWhisper ASR exited during startup" >&2
    tail -80 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "PhoWhisper ASR did not become healthy within 120 seconds" >&2
tail -80 "${LOG_FILE}" >&2 || true
exit 1
