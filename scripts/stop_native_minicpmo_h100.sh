#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENGLASS_NATIVE_ROOT:-/network-volume/icse27/edge-ai/openglass-native}"
RUN_ROOT="${ROOT}/run"

for name in vi_asr vi_tts_vieneu vi_tts worker backend gateway; do
  pid_file="${RUN_ROOT}/${name}.pid"
  [[ -f "${pid_file}" ]] || continue
  pid="$(tr -cd '0-9' <"${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    cmd="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${cmd}" == *"MiniCPM-o-Demo"* || "${cmd}" == *"py_backend.server"* || "${cmd}" == *"gateway.py"* || "${cmd}" == *"worker.py"* || "${cmd}" == *"native_vietnamese_tts_service.py"* || "${cmd}" == *"native_vieneu_tts_service.py"* || "${cmd}" == *"phowhisper_asr_service.py"* ]]; then
      kill "${pid}"
      echo "Stopped ${name} (PID ${pid})."
    else
      echo "Refusing to stop PID ${pid}: it is not an owned MiniCPM-o process." >&2
    fi
  fi
  rm -f "${pid_file}"
done
