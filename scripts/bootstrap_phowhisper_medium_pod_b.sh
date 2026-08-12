#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOWHISPER_BENCH_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
MODEL_ID="vinai/PhoWhisper-medium"
MODEL_REVISION="55a7e3eb6c906de891f8f06a107754427dd3be79"
VENV="${ROOT}/.venv"
COLD_BOOTSTRAP_TOOLS="${ROOT}/bootstrap-tools"
CACHE="${ROOT}/hf-cache"
MODEL_DIR="${ROOT}/models/PhoWhisper-medium-${MODEL_REVISION}"

mkdir -p "${ROOT}/models" "${ROOT}/results" "${CACHE}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  if ! python3 -m venv --system-site-packages "${VENV}"; then
    if [[ -e "${VENV}" ]]; then
      mv "${VENV}" "${VENV}.incomplete.$$.bak"
    fi
    python3 -m pip install \
      --target "${COLD_BOOTSTRAP_TOOLS}" \
      "virtualenv==20.26.6"
    PYTHONPATH="${COLD_BOOTSTRAP_TOOLS}" python3 -m virtualenv \
      --system-site-packages "${VENV}"
  fi
fi

if ! "${VENV}/bin/python" -c "import accelerate, scipy, soundfile, torch, transformers" 2>/dev/null; then
  mv "${VENV}" "${VENV}.invalid.$$.bak"
  python3 -m pip install \
    --target "${COLD_BOOTSTRAP_TOOLS}" \
    "virtualenv==20.26.6"
  PYTHONPATH="${COLD_BOOTSTRAP_TOOLS}" python3 -m virtualenv \
    --system-site-packages "${VENV}"
fi

# Transformers 4.57 refuses legacy pytorch_model.bin files on torch<2.6
# because of CVE-2025-32434. Install the official CUDA 12.4 wheel locally in
# this benchmark venv; never bypass the safety check or mutate the live env.
if ! "${VENV}/bin/python" -c "import torch; raise SystemExit(0 if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6) else 1)"; then
  "${VENV}/bin/python" -m pip install \
    "torch==2.6.0" \
    --index-url https://download.pytorch.org/whl/cu124
fi

if ! "${VENV}/bin/python" -c "import torchvision; raise SystemExit(0 if torchvision.__version__.split('+')[0] == '0.21.0' else 1)" 2>/dev/null; then
  "${VENV}/bin/python" -m pip install \
    "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124
fi

"${VENV}/bin/python" - <<'PY'
import accelerate
import numpy
import scipy
import soundfile
import torch
import transformers

print(
    "runtime_ok",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"accelerate={accelerate.__version__}",
    f"numpy={numpy.__version__}",
    f"scipy={scipy.__version__}",
    f"soundfile={soundfile.__version__}",
)
PY

if [[ ! -f "${MODEL_DIR}/pytorch_model.bin" ]]; then
  HF_HOME="${CACHE}" hf download "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"
fi

remote_head="$(git ls-remote "https://huggingface.co/${MODEL_ID}" HEAD | awk '{print $1}')"

"${VENV}/bin/python" -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('${MODEL_DIR}', local_files_only=True); print('model_ok', c.model_type, c.d_model)"

echo "PhoWhisper benchmark environment ready"
echo "root=${ROOT}"
echo "model=${MODEL_DIR}"
echo "revision=${MODEL_REVISION}"
echo "remote_head_at_setup=${remote_head}"
