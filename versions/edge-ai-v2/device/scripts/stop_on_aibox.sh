#!/usr/bin/env bash
set -euo pipefail

config="/data/local/tmp/aibox-eye/config/production.json"
mapfile -t pids < <(pgrep -f "aibox_eye --config $config" || true)
if [[ ${#pids[@]} -eq 0 ]]; then
  echo "AIBOX-eye coordinator is not running"
  exit 0
fi
for pid in "${pids[@]}"; do
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command_line" == *" -m aibox_eye --config $config"* ]] || {
    echo "Refusing to stop unexpected process $pid: $command_line" >&2
    exit 65
  }
done
kill -TERM "${pids[@]}"
for _ in $(seq 1 10); do
  remaining=()
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && remaining+=("$pid")
  done
  [[ ${#remaining[@]} -eq 0 ]] && { echo "STOPPED ${pids[*]}"; exit 0; }
  sleep 1
done
kill -KILL "${remaining[@]}"
echo "STOPPED ${pids[*]} (forced after graceful timeout)"
