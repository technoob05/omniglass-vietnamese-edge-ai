#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOWHISPER_BENCH_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
PORT="${PHOWHISPER_ASR_PORT:-18783}"
INSTANCE="${PHOWHISPER_ASR_INSTANCE:-$(hostname -s)}"
INSTANCE="${INSTANCE//[^a-zA-Z0-9_.-]/_}"
RUN_DIR="${PHOWHISPER_ASR_RUN_DIR:-${ROOT}/run/${INSTANCE}}"
PID_FILE="${RUN_DIR}/phowhisper-asr-${PORT}.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "No PhoWhisper PID file for instance=${INSTANCE} port=${PORT}"
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.2
  done
fi
rm -f "${PID_FILE}"
echo "PhoWhisper ASR stopped instance=${INSTANCE} pid=${pid}"
