#!/usr/bin/env bash
# Starts only the coordinator.  The validated QNN camera process remains the
# sole owner of /dev/video2 and the only qtimlqnn/HTP engine.
set -euo pipefail

base="/data/local/tmp/aibox-eye"
log="$base/aibox-eye.log"
config="$base/config/production.json"

for required in "$base/aibox_eye/__main__.py" "$config"; do
  test -f "$required" || { echo "Missing required file: $required" >&2; exit 66; }
done

if ! curl -fsS --max-time 1 http://127.0.0.1:8080/health >/dev/null; then
  echo "The single-engine QNN perception service on :8080 is unavailable" >&2
  exit 69
fi
if pgrep -f "aibox_eye --config $config" >/dev/null; then
  echo "AIBOX-eye coordinator is already running"
  exit 0
fi

export PYTHONPATH="$base:$base/vendor/VieNeu-TTS/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$base/cache"
export HF_HOME="$base/cache/huggingface"
export TORCH_HOME="$base/cache/torch"
python_bin="python3"
test -x "$base/venv/bin/python" && python_bin="$base/venv/bin/python"
nohup "$python_bin" -m aibox_eye --config "$config" >"$log" 2>&1 &
pid=$!
for _ in $(seq 1 20); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Coordinator exited during startup:" >&2
    tail -n 80 "$log" >&2
    exit 1
  fi
  if curl -fsS --max-time 1 http://127.0.0.1:8090/health >/dev/null; then
    echo "STARTED pid=$pid url=http://127.0.0.1:8090/ log=$log"
    exit 0
  fi
  sleep 1
done
echo "Timed out waiting for coordinator port 8090; process remains pid=$pid" >&2
exit 1
