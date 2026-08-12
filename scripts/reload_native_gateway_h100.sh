#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
DEMO_ROOT="${ROOT}/MiniCPM-o-Demo"
RUN_ROOT="${ROOT}/run"
LOG_ROOT="${ROOT}/logs"
PYTHON="${OPENGLASS_NATIVE_VENV:-${DEMO_ROOT}/.venv-native}/bin/python"
HTTP_KEEPALIVE_SECONDS="${OPENGLASS_HTTP_KEEPALIVE_SECONDS:-60}"

if [[ ! "${HTTP_KEEPALIVE_SECONDS}" =~ ^[0-9]+$ ]] || (( HTTP_KEEPALIVE_SECONDS < 5 || HTTP_KEEPALIVE_SECONDS > 600 )); then
  echo "OPENGLASS_HTTP_KEEPALIVE_SECONDS must be an integer from 5 to 600." >&2
  exit 1
fi
export OPENGLASS_HTTP_KEEPALIVE_SECONDS="${HTTP_KEEPALIVE_SECONDS}"

if [[ -f "${RUN_ROOT}/gateway.pid" ]]; then
  gateway_pid="$(cat "${RUN_ROOT}/gateway.pid")"
  if kill -0 "${gateway_pid}" 2>/dev/null; then
    kill "${gateway_pid}"
    for _ in {1..50}; do
      kill -0 "${gateway_pid}" 2>/dev/null || break
      sleep 0.1
    done
  fi
fi

cd "${DEMO_ROOT}"
nohup "${PYTHON}" gateway.py \
  --host 127.0.0.1 --port 8006 --internal-port 8007 \
  --https --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem \
  >"${LOG_ROOT}/gateway.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/gateway.pid"

for _ in {1..100}; do
  curl -kfsS https://127.0.0.1:8006/health >/dev/null 2>&1 && break
  sleep 0.1
done
curl -kfsS https://127.0.0.1:8006/health >/dev/null

curl -fsS -X PUT \
  -H 'content-type: application/json' \
  --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}' \
  http://127.0.0.1:8007/internal/workers/openglass-native-h100 >/dev/null

echo "Gateway reloaded with HTTPS keep-alive ${HTTP_KEEPALIVE_SECONDS}s; backend and worker stayed warm."
