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
IMAGE_MIN_TOKENS="${IMAGE_MIN_TOKENS:-64}"
IMAGE_MAX_TOKENS="${IMAGE_MAX_TOKENS:-64}"

BIN="${RUNTIME_ROOT}/bin"
MODEL="${MODEL_ROOT}/${MODEL_FILE}"
PROJECTOR="${MODEL_ROOT}/${PROJECTOR_FILE}"
MTMD="${BIN}/llama-mtmd-cli"
BENCH="${BIN}/llama-bench"
CLI="${BIN}/llama-cli"
DEVICE_PROBE="${CLI}"
[[ -x "${DEVICE_PROBE}" ]] || DEVICE_PROBE="${BENCH}"

for file in "${MTMD}" "${BENCH}" "${DEVICE_PROBE}" "${MODEL}" "${PROJECTOR}" "${IMAGE}"; do
  [[ -e "${file}" ]] || { echo "Missing: ${file}" >&2; exit 2; }
done
[[ "$(uname -m)" == "aarch64" ]] || {
  [[ "${ALLOW_NON_ARM64:-0}" == "1" ]] || { echo "This harness must run on the ARM64 target." >&2; exit 2; }
}
command -v timeout >/dev/null 2>&1 || { echo "GNU timeout is required" >&2; exit 2; }
if [[ ! -x /usr/bin/time ]]; then
  command -v python3 >/dev/null 2>&1 || {
    echo "Either /usr/bin/time or Python 3 is required for peak RSS evidence" >&2
    exit 2
  }
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUT_ROOT%/}/${STAMP}"
mkdir -p "${OUT}"
export LD_LIBRARY_PATH="${RUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ADSP_LIBRARY_PATH="${RUNTIME_ROOT}/lib${ADSP_LIBRARY_PATH:+:${ADSP_LIBRARY_PATH}}"

run_timed() {
  if [[ -x /usr/bin/time ]]; then
    timeout --kill-after=5s "${TIMEOUT_SECONDS}" /usr/bin/time -v "$@"
    return
  fi

  # Ubuntu board images may omit GNU time. wait4 supplies the same auditable
  # child-process peak-RSS evidence without installing or changing the target.
  timeout --kill-after=5s "${TIMEOUT_SECONDS}" python3 -c '
import os
import sys
import time

started = time.monotonic()
pid = os.fork()
if pid == 0:
    os.execvp(sys.argv[1], sys.argv[1:])

_, status, usage = os.wait4(pid, 0)
elapsed = time.monotonic() - started
print(f"Elapsed (wall clock) time (seconds): {elapsed:.6f}", file=sys.stderr)
print(f"User time (seconds): {usage.ru_utime:.6f}", file=sys.stderr)
print(f"System time (seconds): {usage.ru_stime:.6f}", file=sys.stderr)
print(f"Maximum resident set size (kbytes): {usage.ru_maxrss}", file=sys.stderr)
print(f"Major page faults: {usage.ru_majflt}", file=sys.stderr)
print(f"Minor page faults: {usage.ru_minflt}", file=sys.stderr)
code = os.waitstatus_to_exitcode(status)
raise SystemExit(code if code >= 0 else 128 - code)
' "$@"
}

require_log() {
  local log="$1"
  local pattern="$2"
  local reason="$3"
  grep -Eq "${pattern}" "${log}" || {
    echo "${reason}. See ${log}" >&2
    exit 4
  }
}

reject_log() {
  local log="$1"
  local pattern="$2"
  local reason="$3"
  if grep -Eiq "${pattern}" "${log}"; then
    echo "${reason}. See ${log}" >&2
    exit 4
  fi
}

verify_htp_log() {
  local log="$1"
  local device="$2"
  local require_clip="$3"

  echo "Verifying ${device} execution evidence and rejecting silent fallback."

  require_log "${log}" "ggml-hex: ${device} new session" \
    "${device} session was not opened"
  require_log "${log}" "${device}-REPACK model buffer size = +[1-9][0-9.]* (MiB|GiB)" \
    "No non-zero ${device} repacked model buffer was allocated"
  require_log "${log}" "offloaded +[1-9][0-9]*/[1-9][0-9]* layers to GPU" \
    "No LLM layers were offloaded to ${device}"
  require_log "${log}" "ggml-hex: ${device} graph-compute n_nodes +[1-9][0-9]*" \
    "No compute graph executed on ${device}"
  require_log "${log}" "ggml-hex: ${device} profile-op .*usec +[0-9]+ cycles +[0-9]+" \
    "No hardware operator profile was emitted by ${device}"

  if [[ "${require_clip}" == "1" ]]; then
    require_log "${log}" "CLIP using ${device} backend" \
      "The multimodal projector did not select ${device}"
  fi

  reject_log "${log}" "Failed to initialize .*backend, falling back|failed to create device/session|failed to open session|remote_session_control\(unsign\) failed|dsp-rsp: (NO-SUPPORT|INVAL-PARAMS|VTCM-TOO-SMALL|INTERNAL-ERROR|UNKNOWN)" \
    "HTP initialization/operator failure or explicit fallback was observed"
}

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

if command -v ldd >/dev/null 2>&1; then
  : > "${OUT}/runtime_dependencies.txt"
  for binary in "${DEVICE_PROBE}" "${BENCH}" "${MTMD}"; do
    echo "### ${binary}" >> "${OUT}/runtime_dependencies.txt"
    ldd "${binary}" >> "${OUT}/runtime_dependencies.txt" 2>&1
  done
  reject_log "${OUT}/runtime_dependencies.txt" "not found" \
    "An ARM64 runtime dependency is missing"
fi

"${DEVICE_PROBE}" --list-devices > "${OUT}/devices.txt" 2>&1
mapfile -t DISCOVERED_HTP < <(sed -nE 's/^  (HTP[0-9]+): Hexagon.*/\1/p' "${OUT}/devices.txt" | sort -Vu)
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
run_timed \
  "${MTMD}" \
  -m "${MODEL}" --mmproj "${PROJECTOR}" --image "${IMAGE}" \
  -p "${PROMPT}" -c "${CTX}" -n "${TOKENS}" \
  --image-min-tokens "${IMAGE_MIN_TOKENS}" --image-max-tokens "${IMAGE_MAX_TOKENS}" \
  -ngl 0 --no-mmproj-offload \
  > "${OUT}/vision_cpu.log" 2>&1

for device in "${DEVICES[@]}"; do
  thermal_snapshot "before_${device}"
  GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=1 \
    run_timed \
    "${BENCH}" -m "${MODEL}" -p 128 -n "${TOKENS}" -r 3 \
    --device "${device}" -ngl 99 --no-mmap -v \
    > "${OUT}/text_${device}.log" 2>&1

  MTMD_BACKEND_DEVICE="${device}" \
  GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=1 \
    run_timed \
    "${MTMD}" \
    -m "${MODEL}" --mmproj "${PROJECTOR}" --image "${IMAGE}" \
    -p "${PROMPT}" -c "${CTX}" -n "${TOKENS}" \
    --image-min-tokens "${IMAGE_MIN_TOKENS}" --image-max-tokens "${IMAGE_MAX_TOKENS}" \
    --device "${device}" -ngl 99 --no-mmap --mmproj-offload -v \
    > "${OUT}/vision_${device}.log" 2>&1

  verify_htp_log "${OUT}/text_${device}.log" "${device}" 0
  verify_htp_log "${OUT}/vision_${device}.log" "${device}" 1
  thermal_snapshot "after_${device}"
done

thermal_snapshot end
echo "Raw, auditable result bundle: ${OUT}"
echo "Do not promote from one run; use the 100-turn and 30-minute acceptance suite next."
