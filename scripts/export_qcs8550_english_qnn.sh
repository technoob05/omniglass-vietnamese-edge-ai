#!/usr/bin/env bash
set -euo pipefail

# Host-side export planner. It never installs packages and defaults to a dry run.
AI_HUB_MODELS_ROOT="${AI_HUB_MODELS_ROOT:?Set AI_HUB_MODELS_ROOT to a pinned qualcomm/ai-hub-models checkout}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to an isolated artifact directory}"
DEVICE="${QCS8550_EXPORT_DEVICE:-QCS8550 (Proxy)}"
RUN_EXPORT="${RUN_EXPORT:-0}"
REVISION="f413a03dc8845739afce27cd3e691d6a5a7339a3"

for tool in git python3; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "Missing required tool: ${tool}" >&2; exit 2; }
done
[[ -d "${AI_HUB_MODELS_ROOT}/.git" ]] || { echo "Not a git checkout: ${AI_HUB_MODELS_ROOT}" >&2; exit 2; }
[[ "$(git -C "${AI_HUB_MODELS_ROOT}" rev-parse HEAD)" == "${REVISION}" ]] || {
  echo "ai-hub-models revision mismatch; expected ${REVISION}" >&2
  exit 2
}
python3 -c 'import qai_hub, qai_hub_models' >/dev/null 2>&1 || {
  echo "qai_hub and qai_hub_models must already be installed in this environment." >&2
  exit 2
}

mkdir -p "${OUT_ROOT}"
DETECTOR_CMD=(qai-hub-models export yolov11_det
  --target-runtime qnn_context_binary --precision w8a16
  --device "${DEVICE}" --output-dir "${OUT_ROOT}/yolov11_det_w8a16")
DEPTH_CMD=(qai-hub-models export depth_anything_v2
  --target-runtime qnn_context_binary --precision w8a16
  --device "${DEVICE}" --output-dir "${OUT_ROOT}/depth_anything_v2_w8a16")

{
  echo "ai_hub_models_revision=${REVISION}"
  echo "device=${DEVICE}"
  printf 'detector_command='; printf '%q ' "${DETECTOR_CMD[@]}"; echo
  printf 'depth_command='; printf '%q ' "${DEPTH_CMD[@]}"; echo
  echo "claim_boundary=Proxy export is not physical HSPTEK execution evidence."
} > "${OUT_ROOT}/EXPORT_PLAN.txt"

if [[ "${RUN_EXPORT}" != "1" ]]; then
  cat "${OUT_ROOT}/EXPORT_PLAN.txt"
  echo "Dry run only. Set RUN_EXPORT=1 after configuring a private Qualcomm AI Hub token."
  exit 0
fi

command -v qai-hub-models >/dev/null 2>&1 || { echo "Missing qai-hub-models CLI" >&2; exit 2; }
"${DETECTOR_CMD[@]}"
"${DEPTH_CMD[@]}"
(
  cd "${OUT_ROOT}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "Export finished under ${OUT_ROOT}. Inspect job status and exact target metadata before staging."
