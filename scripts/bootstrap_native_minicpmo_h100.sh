#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
DEMO_ROOT="${ROOT}/MiniCPM-o-Demo"
VENV_ROOT="${OPENGLASS_NATIVE_VENV:-${DEMO_ROOT}/.venv-native}"
LOG_ROOT="${ROOT}/logs"

mkdir -p "${LOG_ROOT}"
if [[ -e "${VENV_ROOT}" ]]; then
  echo "Virtualenv already exists: ${VENV_ROOT}" >&2
  echo "Remove that generated directory explicitly before rebuilding it." >&2
  exit 1
fi

virtualenv "${VENV_ROOT}"
PYTHON="${VENV_ROOT}/bin/python"
PIP="${VENV_ROOT}/bin/pip"

"${PIP}" install --upgrade pip setuptools wheel
"${PIP}" install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

"${PIP}" install --no-deps 'minicpmo-utils[all]==1.0.6'
"${PIP}" install \
  transformers==4.51.0 accelerate==1.12.0 'safetensors>=0.7.0' \
  'fastapi>=0.128.0' 'uvicorn>=0.40.0' 'httpx>=0.28.0' \
  'websockets>=16.0' python-multipart 'markdown>=3.6' 'Pygments>=2.18.0' \
  'PyYAML>=6.0' librosa==0.11.0 soundfile==0.12.1 \
  'pydantic>=2.11.0' numpy==2.2.6 'tqdm>=4.67.0' \
  decord==0.6.0 moviepy==2.1.2 pillow==10.4.0 \
  'onnxruntime>=1.18.0,<=1.21.0' onnx hyperpyyaml einops==0.8.1 \
  'scipy>=1.15.0' 'numba>=0.61.0'

cd "${DEMO_ROOT}"
"${PYTHON}" -c 'import torch, torchaudio, torchvision, transformers, librosa, fastapi; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, transformers.__version__, librosa.__version__)'
"${PYTHON}" -m py_backend.server --help >/dev/null
echo "Clean MiniCPM-o environment is ready: ${VENV_ROOT}"
