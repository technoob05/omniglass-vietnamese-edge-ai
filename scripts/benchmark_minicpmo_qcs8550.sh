#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${RUNTIME_ROOT:?Set RUNTIME_ROOT to pkg-snapdragon}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to MiniCPM-o-4_5-Q4}"
TEST_PREFIX="${TEST_PREFIX:?Set TEST_PREFIX to the chunk prefix ending before 0000.wav/.jpg}"
REF_AUDIO="${REF_AUDIO:?Set REF_AUDIO to a consented reference WAV}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to a writable result directory}"
COUNT="${COUNT:-3}"
CTX="${CTX:-2048}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
MODEL="${MODEL_ROOT}/MiniCPM-o-4_5-Q4_0.gguf"
BIN="${RUNTIME_ROOT}/bin/llama-omni-single-test-omni"

for file in "${BIN}" "${MODEL}" "${REF_AUDIO}" "${TEST_PREFIX}0000.wav" "${TEST_PREFIX}0000.jpg"; do
  [[ -e "${file}" ]] || { echo "Missing: ${file}" >&2; exit 2; }
done
[[ "$(uname -m)" == "aarch64" ]] || { echo "Run this only on the ARM64 target." >&2; exit 2; }
[[ -x /usr/bin/time ]] || { echo "/usr/bin/time is required" >&2; exit 2; }
command -v timeout >/dev/null 2>&1 || { echo "GNU timeout is required" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_ROOT%/}/omni_${STAMP}"
mkdir -p "${OUT}"
export LD_LIBRARY_PATH="${RUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ADSP_LIBRARY_PATH="${RUNTIME_ROOT}/lib${ADSP_LIBRARY_PATH:+:${ADSP_LIBRARY_PATH}}"
export GGML_HEXAGON_NDEV="${GGML_HEXAGON_NDEV:-1}"

{
  date -u +%FT%TZ
  uname -a
  free -b 2>/dev/null || true
  sha256sum "${MODEL}" "${REF_AUDIO}" "${TEST_PREFIX}0000.wav" "${TEST_PREFIX}0000.jpg"
  [[ -r "${RUNTIME_ROOT}/BUILD_EVIDENCE.txt" ]] && cat "${RUNTIME_ROOT}/BUILD_EVIDENCE.txt"
} > "${OUT}/inventory.txt"

for zone in /sys/class/thermal/thermal_zone*; do
  [[ -r "${zone}/temp" ]] || continue
  printf '%s type=%s temp=%s\n' "${zone}" "$(cat "${zone}/type" 2>/dev/null || true)" "$(cat "${zone}/temp")"
done > "${OUT}/thermal_before.txt"

# First isolate native audio/vision understanding without Talker/Token2Wav.
GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 \
  timeout "${TIMEOUT_SECONDS}" /usr/bin/time -v \
  "${BIN}" -m "${MODEL}" --test "${TEST_PREFIX}" "${COUNT}" \
  --ref-audio "${REF_AUDIO}" -c "${CTX}" -ngl 99 --no-tts \
  > "${OUT}/omni_no_tts.log" 2>&1

# Then exercise the complete native English Talker and Token2Wav path.
GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 \
  timeout "${TIMEOUT_SECONDS}" /usr/bin/time -v \
  "${BIN}" -m "${MODEL}" --test "${TEST_PREFIX}" "${COUNT}" \
  --ref-audio "${REF_AUDIO}" -c "${CTX}" -ngl 99 \
  > "${OUT}/omni_with_tts.log" 2>&1

grep -Eq 'HTP[0-9].*(model buffer|REPACK)|ggml-hex.*HTP[0-9]' "${OUT}/omni_with_tts.log" || {
  echo "No HTP execution evidence; rejecting possible CPU-only fallback." >&2
  exit 4
}
for zone in /sys/class/thermal/thermal_zone*; do
  [[ -r "${zone}/temp" ]] || continue
  printf '%s type=%s temp=%s\n' "${zone}" "$(cat "${zone}/type" 2>/dev/null || true)" "$(cat "${zone}/temp")"
done > "${OUT}/thermal_after.txt"

echo "Full-Omni raw evidence: ${OUT}"
echo "This bounded test does not establish realtime or release readiness."
