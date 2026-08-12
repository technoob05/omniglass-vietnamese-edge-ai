#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOWHISPER_BENCH_ROOT:-/network-volume/icse27/edge-ai/phowhisper-benchmark}"
VENV="${ROOT}/.venv"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Run bootstrap_phowhisper_medium_pod_b.sh first" >&2
  exit 1
fi

# Silero's official Python package imports torchaudio even when its ONNX
# backend is selected.  Keep the audio wheel exactly aligned with torch.
"${VENV}/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torchaudio==2.6.0+cu124"
"${VENV}/bin/python" -m pip install --disable-pip-version-check \
  "silero-vad==6.2.1" \
  "uvicorn==0.30.6" \
  "websockets==15.0.1" \
  "python-multipart==0.0.20"

"${VENV}/bin/python" - <<'PY'
import fastapi
import onnxruntime
import silero_vad
import soundfile
import torch
import torchaudio
import uvicorn
import websockets

model = silero_vad.load_silero_vad(onnx=True)
print(
    "asr_service_runtime_ok",
    f"torch={torch.__version__}",
    f"torchaudio={torchaudio.__version__}",
    f"silero_vad={silero_vad.__version__}",
    f"fastapi={fastapi.__version__}",
    f"uvicorn={uvicorn.__version__}",
    f"websockets={websockets.__version__}",
    f"onnxruntime={onnxruntime.__version__}",
    f"vad_backend={type(model).__name__}",
)
PY
