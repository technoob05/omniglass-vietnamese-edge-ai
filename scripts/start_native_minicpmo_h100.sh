#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
DEMO_ROOT="${ROOT}/MiniCPM-o-Demo"
MODEL_ROOT="${ROOT}/models/MiniCPM-o-4_5"
RUN_ROOT="${ROOT}/run"
LOG_ROOT="${ROOT}/logs"
VENV_ROOT="${OPENGLASS_NATIVE_VENV:-${DEMO_ROOT}/.venv-native}"
PYTHON="${VENV_ROOT}/bin/python"
VIENEU_VENV_ROOT="${OPENGLASS_VIENEU_VENV:-${ROOT}/.venv-vieneu}"
VIENEU_PYTHON="${VIENEU_VENV_ROOT}/bin/python"
VIENEU_SERVICE="${OPENGLASS_VIENEU_SERVICE:-${ROOT}/native_vieneu_tts_service.py}"
VIENEU_ENABLED="${OPENGLASS_VIENEU_ENABLED:-true}"
VIENEU_REQUIRED="${OPENGLASS_VIENEU_REQUIRED:-false}"
VIENEU_BACKBONE_ROOT="${OPENGLASS_VIENEU_BACKBONE:-${ROOT}/models/VieNeu-TTS-v3-Turbo}"
MOSS_TOKENIZER_ROOT="${OPENGLASS_MOSS_TOKENIZER:-${ROOT}/models/MOSS-Audio-Tokenizer-Nano}"
ASR_ROOT="${OPENGLASS_PHOWHISPER_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
ASR_PYTHON="${OPENGLASS_PHOWHISPER_PYTHON:-${ASR_ROOT}/.venv/bin/python}"
ASR_SERVICE="${OPENGLASS_PHOWHISPER_SERVICE:-${ASR_ROOT}/scripts/phowhisper_asr_service.py}"
ASR_MODEL_ID="${OPENGLASS_PHOWHISPER_MODEL_ID:-vinai/PhoWhisper-large}"
ASR_MODEL_REVISION="${OPENGLASS_PHOWHISPER_MODEL_REVISION:-b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb}"
ASR_MODEL_ROOT="${OPENGLASS_PHOWHISPER_MODEL:-${ASR_ROOT}/models/PhoWhisper-large-${ASR_MODEL_REVISION}}"
ASR_ENGLISH_MODEL_ROOT="${OPENGLASS_ENGLISH_WHISPER_MODEL:-/network-volume/icse27/edge-ai/vi-asr-candidates/models/whisper-large-v3-turbo-41f01f3fe87f28c78e2fbf8b568835947dd65ed9}"
ASR_ENABLED="${OPENGLASS_PHOWHISPER_ENABLED:-true}"
ASR_REQUIRED="${OPENGLASS_PHOWHISPER_REQUIRED:-false}"
HTTP_KEEPALIVE_SECONDS="${OPENGLASS_HTTP_KEEPALIVE_SECONDS:-60}"

if [[ ! "${HTTP_KEEPALIVE_SECONDS}" =~ ^[0-9]+$ ]] || (( HTTP_KEEPALIVE_SECONDS < 5 || HTTP_KEEPALIVE_SECONDS > 600 )); then
  echo "OPENGLASS_HTTP_KEEPALIVE_SECONDS must be an integer from 5 to 600." >&2
  exit 1
fi
export OPENGLASS_HTTP_KEEPALIVE_SECONDS="${HTTP_KEEPALIVE_SECONDS}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${DEMO_ROOT}/data"

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

require_file "${PYTHON}"
require_file "${MODEL_ROOT}/model.safetensors.index.json"
require_file "${DEMO_ROOT}/certs/cert.pem"
require_file "${DEMO_ROOT}/certs/key.pem"
require_file "${ROOT}/native_vietnamese_tts_service.py"

vieneu_available=false
if [[ "${VIENEU_ENABLED}" == "true" && -x "${VIENEU_PYTHON}" && -f "${VIENEU_SERVICE}" && -d "${VIENEU_BACKBONE_ROOT}" && -d "${MOSS_TOKENIZER_ROOT}" ]]; then
  vieneu_available=true
elif [[ "${VIENEU_REQUIRED}" == "true" ]]; then
  echo "VieNeu is required but its venv/service is missing." >&2
  echo "Expected: ${VIENEU_PYTHON} and ${VIENEU_SERVICE}" >&2
  exit 1
elif [[ "${VIENEU_ENABLED}" == "true" ]]; then
  echo "VieNeu is not installed; continuing with MMS-TTS fallback on port 18781." >&2
fi

asr_available=false
if [[ "${ASR_ENABLED}" == "true" && -x "${ASR_PYTHON}" && -f "${ASR_SERVICE}" && -f "${ASR_MODEL_ROOT}/pytorch_model.bin" ]]; then
  asr_available=true
elif [[ "${ASR_REQUIRED}" == "true" ]]; then
  echo "PhoWhisper is required but its venv/service/model is missing." >&2
  echo "Expected: ${ASR_PYTHON}, ${ASR_SERVICE}, ${ASR_MODEL_ROOT}" >&2
  exit 1
elif [[ "${ASR_ENABLED}" == "true" ]]; then
  echo "PhoWhisper is not installed; /vi ASR will be unavailable." >&2
fi

ports=(8006 8007 18781 22400 22500)
[[ "${vieneu_available}" == "true" ]] && ports+=(18782)
[[ "${asr_available}" == "true" ]] && ports+=(18783)
for port in "${ports[@]}"; do
  if ! "${PYTHON}" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', ${port})); s.close()" 2>/dev/null; then
    echo "Port ${port} is already in use; refusing to replace an existing service." >&2
    exit 1
  fi
done

wait_http() {
  local url="$1"
  local seconds="$2"
  local insecure="${3:-false}"
  local curl_args=(-fsS)
  [[ "${insecure}" == "true" ]] && curl_args+=(-k)
  for ((i = 0; i < seconds; i += 2)); do
    if curl "${curl_args[@]}" "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for ${url}" >&2
  return 1
}

cd "${DEMO_ROOT}"

nohup "${PYTHON}" gateway.py \
  --host 127.0.0.1 --port 8006 --internal-port 8007 \
  --https --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem \
  >"${LOG_ROOT}/gateway.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/gateway.pid"
wait_http "https://127.0.0.1:8006/health" 60 true

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${PYTHON}" -m py_backend.server \
  --host 127.0.0.1 --port 22500 --gpu-id 0 --model-path "${MODEL_ROOT}" \
  >"${LOG_ROOT}/backend.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/backend.pid"
wait_http "http://127.0.0.1:22500/health" 600 false

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${PYTHON}" \
  "${ROOT}/native_vietnamese_tts_service.py" \
  --host 127.0.0.1 --port 18781 \
  >"${LOG_ROOT}/vi_tts.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/vi_tts.pid"
wait_http "http://127.0.0.1:18781/health" 120 false

vieneu_ready=false
if [[ "${vieneu_available}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${VIENEU_PYTHON}" \
    "${VIENEU_SERVICE}" \
    --host 127.0.0.1 --port 18782 \
    --device cuda --backend pytorch --dtype auto --max-batch-size 1 \
    --backbone-repo "${VIENEU_BACKBONE_ROOT}" --moss-tokenizer "${MOSS_TOKENIZER_ROOT}" \
    >"${LOG_ROOT}/vi_tts_vieneu.log" 2>&1 </dev/null &
  echo $! >"${RUN_ROOT}/vi_tts_vieneu.pid"
  if wait_http "http://127.0.0.1:18782/health" 900 false; then
    if curl -fsS "http://127.0.0.1:18782/health" | "${VIENEU_PYTHON}" -c '
import json, sys
health = json.load(sys.stdin)
assert health["status"] == "ready", health
assert health["backend"] == "pytorch", health
assert health["device"].startswith("cuda"), health
assert health["sample_rate"] == 48000, health
print("VieNeu readiness validated:", health)
'; then
      vieneu_ready=true
    fi
  fi
  if [[ "${vieneu_ready}" != "true" ]]; then
    vieneu_pid="$(tr -cd '0-9' <"${RUN_ROOT}/vi_tts_vieneu.pid")"
    [[ -n "${vieneu_pid}" ]] && kill "${vieneu_pid}" 2>/dev/null || true
    rm -f "${RUN_ROOT}/vi_tts_vieneu.pid"
    if [[ "${VIENEU_REQUIRED}" == "true" ]]; then
      echo "VieNeu failed readiness and is required. See ${LOG_ROOT}/vi_tts_vieneu.log" >&2
      exit 1
    fi
    echo "VieNeu failed readiness; continuing with MMS-TTS fallback. See ${LOG_ROOT}/vi_tts_vieneu.log" >&2
  fi
fi

asr_ready=false
if [[ "${asr_available}" == "true" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup "${ASR_PYTHON}" \
    "${ASR_SERVICE}" \
    --host 127.0.0.1 --port 18783 --model-dir "${ASR_MODEL_ROOT}" \
    --model-id "${ASR_MODEL_ID}" --model-revision "${ASR_MODEL_REVISION}" \
    --english-model-dir "${ASR_ENGLISH_MODEL_ROOT}" \
    >"${LOG_ROOT}/vi_asr_phowhisper.log" 2>&1 </dev/null &
  echo $! >"${RUN_ROOT}/vi_asr.pid"
  if wait_http "http://127.0.0.1:18783/health" 180 false; then
    asr_ready=true
  else
    asr_pid="$(tr -cd '0-9' <"${RUN_ROOT}/vi_asr.pid")"
    [[ -n "${asr_pid}" ]] && kill "${asr_pid}" 2>/dev/null || true
    rm -f "${RUN_ROOT}/vi_asr.pid"
    if [[ "${ASR_REQUIRED}" == "true" ]]; then
      echo "PhoWhisper failed readiness and is required. See ${LOG_ROOT}/vi_asr_phowhisper.log" >&2
      exit 1
    fi
    echo "PhoWhisper failed readiness; /vi ASR will be unavailable." >&2
  fi
fi

nohup "${PYTHON}" worker.py \
  --host 127.0.0.1 --port 22400 --gpu-id 0 \
  --backend-server-url http://127.0.0.1:22500 \
  >"${LOG_ROOT}/worker.log" 2>&1 </dev/null &
echo $! >"${RUN_ROOT}/worker.pid"
wait_http "http://127.0.0.1:22400/health" 60 false

curl -fsS -X PUT \
  -H 'content-type: application/json' \
  --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}' \
  http://127.0.0.1:8007/internal/workers/openglass-native-h100 >/dev/null

echo "MiniCPM-o native baseline is ready: https://127.0.0.1:8006/"
echo "Vietnamese TTS: MMS fallback=ready, VieNeu streaming=${vieneu_ready}"
echo "Vietnamese ASR: PhoWhisper=${asr_ready}; dedicated UI=https://127.0.0.1:8006/vi"
echo "Gateway HTTPS keep-alive: ${HTTP_KEEPALIVE_SECONDS}s"
echo "Logs: ${LOG_ROOT}/{gateway,backend,worker,vi_tts,vi_tts_vieneu,vi_asr_phowhisper}.log"
