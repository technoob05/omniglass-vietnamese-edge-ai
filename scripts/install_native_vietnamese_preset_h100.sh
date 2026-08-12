#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
DEMO_ROOT="${ROOT}/MiniCPM-o-Demo"
OVERRIDE_ROOT="${OPENGLASS_NATIVE_OVERRIDE_ROOT:-${ROOT}/native-overrides}"

install -m 0644 \
  "${OVERRIDE_ROOT}/vietnamese_call.yaml" \
  "${DEMO_ROOT}/assets/presets/omni/vietnamese_call.yaml"
install -m 0644 \
  "${OVERRIDE_ROOT}/ref_vi_mms.wav" \
  "${DEMO_ROOT}/assets/ref_audio/ref_vi_mms.wav"

DEMO_ROOT="${DEMO_ROOT}" "${DEMO_ROOT}/.venv-native/bin/python" - <<'PY'
import os
from pathlib import Path
import wave
import yaml

root = Path(os.environ["DEMO_ROOT"])
preset = yaml.safe_load(
    (root / "assets/presets/omni/vietnamese_call.yaml").read_text(encoding="utf-8")
)
assert preset["id"] == "vietnamese_call"
with wave.open(str(root / "assets/ref_audio/ref_vi_mms.wav"), "rb") as wav_file:
    assert wav_file.getnchannels() == 1
    assert wav_file.getsampwidth() == 2
    assert wav_file.getframerate() == 16000
print("Vietnamese preset and reference audio validated.")
PY

echo "Run ${ROOT}/reload_native_gateway_h100.sh to make the gateway reload presets."
