#!/usr/bin/env bash
set -euo pipefail

# Reproducible VieNeu v3 Turbo GPU environment. The H100 pod currently uses
# driver 550/CUDA 12.4, so this pins the exact PyTorch stack used by the local
# 15-call benchmark instead of assuming the upstream CUDA 12.8 development
# environment is compatible with the deployed driver.
ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
REPO_ROOT="${OPENGLASS_VIENEU_REPO:-${ROOT}/dependencies/VieNeu-TTS}"
VENV_ROOT="${OPENGLASS_VIENEU_VENV:-${ROOT}/.venv-vieneu}"
VIENEU_COMMIT="${OPENGLASS_VIENEU_COMMIT:-a8c9fbf99749d5ce45c89111f71558d6ceef3424}"
VIENEU_MODEL_REVISION="${OPENGLASS_VIENEU_MODEL_REVISION:-75ff82a72f54d55ed389e1eeb12041d3c4bac7d4}"
MOSS_MODEL_REVISION="${OPENGLASS_MOSS_MODEL_REVISION:-6aa02b01e445cc585582cf0ba480bc3ea6c8dd68}"
MODEL_ROOT="${ROOT}/models"

command -v git >/dev/null
command -v python3 >/dev/null

if [[ ! -d "${REPO_ROOT}/.git" ]]; then
  git clone https://github.com/pnnbao97/VieNeu-TTS.git "${REPO_ROOT}"
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  echo "VieNeu repository has local changes; refusing to change revisions." >&2
  exit 1
fi

git -C "${REPO_ROOT}" fetch --depth 1 origin "${VIENEU_COMMIT}"
git -C "${REPO_ROOT}" checkout --detach "${VIENEU_COMMIT}"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  python3 -m venv "${VENV_ROOT}"
fi
PYTHON="${VENV_ROOT}/bin/python"

"${PYTHON}" -m pip install -U pip setuptools wheel
"${PYTHON}" -m pip install \
  torch==2.4.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
"${PYTHON}" -m pip install \
  transformers==4.51.0 safetensors accelerate \
  sea-g2p==0.8.4 onnxruntime==1.23.2 numpy==2.2.6 soundfile==0.14.0 \
  soxr==1.1.0 huggingface-hub==0.36.2 PyYAML==6.0.3 \
  librosa==0.11.0 perth==1.0.0 fastapi uvicorn
"${PYTHON}" -m pip install --no-deps -e "${REPO_ROOT}"

mkdir -p "${MODEL_ROOT}/VieNeu-TTS-v3-Turbo" "${MODEL_ROOT}/MOSS-Audio-Tokenizer-Nano"
"${VENV_ROOT}/bin/hf" download \
  pnnbao-ump/VieNeu-TTS-v3-Turbo --revision "${VIENEU_MODEL_REVISION}" \
  --local-dir "${MODEL_ROOT}/VieNeu-TTS-v3-Turbo"
"${VENV_ROOT}/bin/hf" download \
  OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano --revision "${MOSS_MODEL_REVISION}" \
  --local-dir "${MODEL_ROOT}/MOSS-Audio-Tokenizer-Nano"

"${PYTHON}" - <<'PY'
import torch
from vieneu import Vieneu

assert torch.cuda.is_available(), "CUDA is not available in the VieNeu environment"
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name())
print("vieneu", Vieneu)
PY

echo "VieNeu GPU environment ready: ${VENV_ROOT}"
echo "VieNeu model revision: ${VIENEU_MODEL_REVISION}"
echo "MOSS tokenizer revision: ${MOSS_MODEL_REVISION}"
