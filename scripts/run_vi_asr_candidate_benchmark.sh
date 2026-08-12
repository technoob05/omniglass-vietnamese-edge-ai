#!/usr/bin/env bash
set -euo pipefail

# Run one candidate without touching the production ASR process. This script
# fails closed when the H100 does not have enough free VRAM.
ROOT="${OMNIGLASS_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
BENCH_ROOT="${VI_ASR_BENCH_ROOT:-/network-volume/icse27/edge-ai/vi-asr-candidates}"
MANIFEST="${VI_ASR_MANIFEST:-/network-volume/icse27/edge-ai/phowhisper-benchmark/datasets/fleurs-vi_vn-test-50-rev70bb2e84/manifest.json}"
MODEL_KEY="${1:-qwen3-asr-0.6b}"

case "${MODEL_KEY}" in
  qwen3-asr-0.6b)
    BACKEND=qwen3_asr
    MODEL_ID=Qwen/Qwen3-ASR-0.6B
    MODEL_REVISION=5eb144179a02acc5e5ba31e748d22b0cf3e303b0
    MODEL_LICENSE=Apache-2.0
    MIN_FREE_GIB=5
    ;;
  qwen3-asr-1.7b)
    BACKEND=qwen3_asr
    MODEL_ID=Qwen/Qwen3-ASR-1.7B
    MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
    MODEL_LICENSE=Apache-2.0
    MIN_FREE_GIB=8
    ;;
  whisper-large-v3-turbo)
    BACKEND=whisper
    MODEL_ID=openai/whisper-large-v3-turbo
    MODEL_REVISION=41f01f3fe87f28c78e2fbf8b568835947dd65ed9
    MODEL_LICENSE=MIT
    MIN_FREE_GIB=5
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

mkdir -p "${BENCH_ROOT}/models" "${BENCH_ROOT}/artifacts" "${BENCH_ROOT}/predictions"
VENV="${BENCH_ROOT}/venv-qwen-asr-0.0.6"
if [[ ! -x "${VENV}/bin/python" ]]; then
  # Reuse the pod's CUDA-enabled Torch wheel without modifying it. Candidate
  # packages are still isolated and cannot perturb the production venv.
  VIRTUALENV_ROOT="${VI_ASR_VIRTUALENV_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark/bootstrap-tools}"
  if [[ -d "${VIRTUALENV_ROOT}/virtualenv" ]]; then
    PYTHONPATH="${VIRTUALENV_ROOT}" python3 -m virtualenv --system-site-packages "${VENV}"
  else
    python3 -m venv --system-site-packages "${VENV}"
  fi
  "${VENV}/bin/pip" install --upgrade pip
  "${VENV}/bin/pip" install \
    'transformers==4.57.6' 'accelerate==1.12.0' 'qwen-asr==0.0.6' \
    'huggingface-hub>=0.34' 'soundfile>=0.13' 'scipy>=1.15'
fi

# Parent-GPU memory queries can report N/A inside a MIG slice. CUDA's allocator
# reports the actual visible slice and is therefore the authoritative preflight.
FREE_MIB="$(python3 -c 'import torch; print(torch.cuda.mem_get_info()[0] // (1024 * 1024))')"
REQUIRED_MIB="$((MIN_FREE_GIB * 1024))"
if (( FREE_MIB < REQUIRED_MIB )); then
  echo "Refusing ${MODEL_KEY}: ${FREE_MIB} MiB free < ${REQUIRED_MIB} MiB; production remains untouched" >&2
  exit 3
fi

MODEL_DIR="${BENCH_ROOT}/models/${MODEL_KEY}-${MODEL_REVISION}"
if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  "${VENV}/bin/python" -c \
    'from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])' \
    "${MODEL_ID}" "${MODEL_REVISION}" "${MODEL_DIR}"
fi

SCRIPT="${ROOT}/scripts/evaluate_asr_candidate_manifest.py"
OUTPUT="${BENCH_ROOT}/artifacts/${MODEL_KEY}-fleurs-vi-test50.json"
PREDICTIONS="${BENCH_ROOT}/predictions/${MODEL_KEY}-fleurs-vi-test50.jsonl"
"${VENV}/bin/python" "${SCRIPT}" \
  --backend "${BACKEND}" \
  --manifest "${MANIFEST}" \
  --model-dir "${MODEL_DIR}" \
  --model-id "${MODEL_ID}" \
  --model-revision "${MODEL_REVISION}" \
  --model-license "${MODEL_LICENSE}" \
  --minimum-free-vram-gib "${MIN_FREE_GIB}" \
  --output "${OUTPUT}" \
  --predictions-jsonl "${PREDICTIONS}"

echo "Wrote ${OUTPUT}"
