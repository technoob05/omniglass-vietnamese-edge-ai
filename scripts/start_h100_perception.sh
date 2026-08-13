#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
SOURCE="${OPENGLASS_PERCEPTION_SOURCE:-${ROOT}/h100_perception_service.py}"
MODEL="${OPENGLASS_PERCEPTION_MODEL:-/network-volume/icse27/edge-ai/omniglass/yolo11n.pt}"
PYTHON="${OPENGLASS_PERCEPTION_PYTHON:-python3}"
PORT="${OPENGLASS_PERCEPTION_PORT:-18784}"
DEPTH_MODEL="${OPENGLASS_DEPTH_MODEL:-depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf}"
RUN_ROOT="${ROOT}/run"
LOG_ROOT="${ROOT}/logs"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
for required in "${SOURCE}" "${MODEL}"; do
  [[ -f "${required}" ]] || { echo "Missing perception artifact: ${required}" >&2; exit 66; }
done

pid_file="${RUN_ROOT}/vi_perception.pid"
if [[ -f "${pid_file}" ]]; then
  old_pid="$(cat "${pid_file}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    kill "${old_pid}"
    for _ in {1..50}; do
      kill -0 "${old_pid}" 2>/dev/null || break
      sleep 0.1
    done
  fi
fi

nohup "${PYTHON}" "${SOURCE}" \
  --host 127.0.0.1 --port "${PORT}" --model "${MODEL}" --device cuda:0 \
  --depth-model "${DEPTH_MODEL}" --depth-every 5 --history-size 800 \
  >"${LOG_ROOT}/vi_perception.log" 2>&1 </dev/null &
echo $! >"${pid_file}"

for _ in {1..900}; do
  if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:${PORT}/health"
    echo
    exit 0
  fi
  if ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    tail -n 100 "${LOG_ROOT}/vi_perception.log" >&2
    exit 1
  fi
  sleep 0.2
done

echo "Timed out waiting for H100 perception" >&2
tail -n 100 "${LOG_ROOT}/vi_perception.log" >&2
exit 1
