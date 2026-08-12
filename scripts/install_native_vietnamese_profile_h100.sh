#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
DEMO_ROOT="${ROOT}/MiniCPM-o-Demo"
PYTHON="${OPENGLASS_NATIVE_VENV:-${DEMO_ROOT}/.venv-native}/bin/python"
PATCH_SCRIPT="${OPENGLASS_VI_PROFILE_PATCH:-${ROOT}/native-overrides/patch_native_vietnamese_profile.py}"
ASSETS_ROOT="${OPENGLASS_VI_PROFILE_ASSETS:-${ROOT}/native-overrides/vi-profile}"
RELOAD="${OPENGLASS_VI_PROFILE_RELOAD:-false}"

for required in "${PYTHON}" "${DEMO_ROOT}/gateway.py" "${PATCH_SCRIPT}" \
  "${ASSETS_ROOT}/vi-chat.html" "${ASSETS_ROOT}/vi-chat.js"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 1
  fi
done

"${PYTHON}" "${PATCH_SCRIPT}" "${DEMO_ROOT}" --assets-root "${ASSETS_ROOT}"
"${PYTHON}" -m py_compile "${DEMO_ROOT}/gateway.py"

grep -q 'OPENGLASS_VI_ASSISTANT_PROFILE_V1' "${DEMO_ROOT}/gateway.py"
grep -q 'speech_pad_ms=300' "${DEMO_ROOT}/static/vi/vi-chat.js"
grep -q 'value="Trúc Ly" selected' "${DEMO_ROOT}/static/vi/vi-chat.html"

if [[ "${RELOAD}" == "true" ]]; then
  "${ROOT}/reload_native_gateway_h100.sh"
fi

echo "Vietnamese profile installed: https://127.0.0.1:8006/vi (reload=${RELOAD})"
