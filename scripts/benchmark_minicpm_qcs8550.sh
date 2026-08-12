#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT to the installed pkg-snapdragon directory}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to a downloaded MiniCPM-V lane directory}"
IMAGE="${IMAGE:?Set IMAGE to a representative glasses-camera JPEG or PNG}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a writable benchmark result directory}"
MODEL_FILE="${MODEL_FILE:-MiniCPM-V-4_6-Q4_0.gguf}"
PROJECTOR_FILE="${PROJECTOR_FILE:-mmproj-model-f16.gguf}"
PROMPT="${PROMPT:-Describe obstacles, people, readable text, and safe next actions in one concise sentence.}"
CTX="${CTX:-2048}"
TOKENS="${TOKENS:-64}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

BIN="${RUNTIME_ROOT}/bin"
MODEL="${MODEL_ROOT}/${MODEL_FILE}"
PROJECTOR="${MODEL_ROOT}/${PROJECTOR_FILE}"
MTMD="${BIN}/llama-mtmd-cli"
BENCH="${BIN}/llama-bench"
CLI="${BIN}/llama-cli"

for file in "${MTMD}" "${BENCH}" "${CLI}" "${MODEL}" "${PROJECTOR}" "${IMAGE}"; do
  [[ -e "${file}" ]] || { echo "Missing: ${file}" >&2; exit 2; }
done
[[ "$(uname -m)" == "aarch64" ]] || {
  [[ "${ALLOW_NON_ARM64:-0}" == "1" ]] || { echo "This harness must run on the ARM64 target." >&2; exit 2; }
}
command -v timeout >/dev/null 2>&1 || { echo "GNU timeout is required" >&2; exit 2; }
TIME_BIN="/usr/bin/time"
[[ -x "${TIME_BIN}" ]] || { echo "/usr/bin/time is required for peak RSS evidence" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_ROOT%/}/${STAMP}"
mkdir -p "${OUT}"
export LD_LIBRARY_PATH="${RUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ADSP_LIBRARY_PATH="${RUNTIME_ROOT}/lib${ADSP_LIBRARY_PATH:+:${ADSP_LIBRARY_PATH}}"

thermal_snapshot() {
  local label="$1"
  {
    echo "label=${label}"
    date -u +%FT%TZ
    for zone in /sys/class/thermal/thermal_zone*; do
      [[ -r "${zone}/temp" ]] || continue
      printf '%s type=%s temp=%s\n' "${zone}" "$(cat "${zone}/type" 2>/dev/null || true)" "$(cat "${zone}/temp")"
    done
  } > "${OUT}/thermal_${label}.txt"
}

{
  date -u +%FT%TZ
  uname -a
  command -v lscpu >/dev/null 2>&1 && lscpu || true
  command -v free >/dev/null 2>&1 && free -b || true
  [[ -r /etc/os-release ]] && cat /etc/os-release
  [[ -r "${RUNTIME_ROOT}/BUILD_EVIDENCE.txt" ]] && cat "${RUNTIME_ROOT}/BUILD_EVIDENCE.txt"
  sha256sum "${MODEL}" "${PROJECTOR}" "${IMAGE}"
} > "${OUT}/inventory.txt"

"${CLI}" --list-devices > "${OUT}/devices.txt" 2>&1
mapfile -t DISCOVERED_HTP < <(grep -oE 'HTP[0-9]+' "${OUT}/devices.txt" | sort -Vu)
if [[ -n "${HTP_DEVICES:-}" ]]; then
  IFS=',' read -r -a DEVICES <<< "${HTP_DEVICES}"
else
  DEVICES=("${DISCOVERED_HTP[@]}")
fi
[[ "${#DEVICES[@]}" -gt 0 ]] || {
  echo "No HTP device discovered. See ${OUT}/devices.txt" >&2
  exit 3
}

thermal_snapshot start

# CPU reference: keeps both LLM and multimodal projector off accelerator.
timeout "${TIMEOUT_SECONDS}" "${TIME_BIN}" -v \
  "${MTMD}" \
  -m "${MODEL}" --mmproj "${PROJECTOR}" --image "${IMAGE}" \
  -p "${PROMPT}" -c "${CTX}" -n "${TOKENS}" -ngl 0 --no-mmproj-offload \
  > "${OUT}/vision_cpu.log" 2>&1

for device in "${DEVICES[@]}"; do
  thermal_snapshot "before_${device}"
  GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 \
    timeout "${TIMEOUT_SECONDS}" "${TIME_BIN}" -v \
    "${BENCH}" -m "${MODEL}" -p 128 -n "${TOKENS}" -r 3 \
    --device "${device}" -ngl 99 --no-mmap \
    > "${OUT}/text_${device}.log" 2>&1

  GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 \
    timeout "${TIMEOUT_SECONDS}" "${TIME_BIN}" -v \
    "${MTMD}" \
    -m "${MODEL}" --mmproj "${PROJECTOR}" --image "${IMAGE}" \
    -p "${PROMPT}" -c "${CTX}" -n "${TOKENS}" \
    --device "${device}" -ngl 99 --no-mmap \
    > "${OUT}/vision_${device}.log" 2>&1

  grep -Eq "${device}.*(model buffer|REPACK)|ggml-hex.*${device}" "${OUT}/vision_${device}.log" || {
    echo "No evidence that ${device} was used by the VLM run; rejecting silent fallback." >&2
    exit 4
  }
  thermal_snapshot "after_${device}"
done

thermal_snapshot end
echo "Raw, auditable result bundle: ${OUT}"
echo "Do not promote from one run; use the 100-turn and 30-minute acceptance suite next."
