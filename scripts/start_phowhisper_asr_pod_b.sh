#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOWHISPER_BENCH_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
VENV="${ROOT}/.venv"
SERVICE_SCRIPT="${ROOT}/scripts/phowhisper_asr_service.py"
MODEL_REVISION="55a7e3eb6c906de891f8f06a107754427dd3be79"
MODEL_DIR="${ROOT}/models/PhoWhisper-medium-${MODEL_REVISION}"
HOST="${PHOWHISPER_ASR_HOST:-127.0.0.1}"
PORT="${PHOWHISPER_ASR_PORT:-18783}"
RUN_DIR="${ROOT}/run"
PID_FILE="${RUN_DIR}/phowhisper-asr-${PORT}.pid"
LOG_FILE="${RUN_DIR}/phowhisper-asr-${PORT}.log"

mkdir -p "${RUN_DIR}"

for required in "${VENV}/bin/python" "${SERVICE_SCRIPT}" "${MODEL_DIR}/pytorch_model.bin"; do
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
    echo "PID ${old_pid} is alive but health failed; refusing to restart automatically" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

nohup "${VENV}/bin/python" "${SERVICE_SCRIPT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-dir "${MODEL_DIR}" \
  >"${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" >"${PID_FILE}"

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "PhoWhisper ASR ready pid=${pid} url=http://${HOST}:${PORT} log=${LOG_FILE}"
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
