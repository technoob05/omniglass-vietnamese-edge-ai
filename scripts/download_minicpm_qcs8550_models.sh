#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
MODEL_ROOT="${MODEL_ROOT:?Set MODEL_ROOT to a private model directory}"
DRY_RUN="${DOWNLOAD_DRY_RUN:-0}"
command -v hf >/dev/null 2>&1 || { echo "Hugging Face hf CLI is required" >&2; exit 2; }
mkdir -p "${MODEL_ROOT}"

V46_REV="78e02f066e9819a60573b78a4275df8a0c27f698"
V45_REV="cefe1580fe9402b06c4e1b8ed7343809377b8147"
OMNI_REV="f706cc65f45288ef13f18d60834a9141c8e40b8f"
HF_EXTRA=()
if [[ "${DRY_RUN}" == "1" ]]; then
  HF_EXTRA+=(--dry-run)
fi

download_v46() {
  hf download openbmb/MiniCPM-V-4.6-gguf \
    MiniCPM-V-4_6-Q4_0.gguf \
    mmproj-model-f16.gguf \
    --revision "${V46_REV}" \
    --local-dir "${MODEL_ROOT}/MiniCPM-V-4.6-Q4" \
    "${HF_EXTRA[@]}"
}

download_v45() {
  hf download openbmb/MiniCPM-V-4_5-gguf \
    MiniCPM-V-4_5-Q4_0.gguf \
    mmproj-model-f16.gguf \
    --revision "${V45_REV}" \
    --local-dir "${MODEL_ROOT}/MiniCPM-V-4_5-Q4" \
    "${HF_EXTRA[@]}"
}

download_omni() {
  hf download openbmb/MiniCPM-o-4_5-gguf \
    MiniCPM-o-4_5-Q4_0.gguf \
    vision/MiniCPM-o-4_5-vision-F16.gguf \
    audio/MiniCPM-o-4_5-audio-F16.gguf \
    tts/MiniCPM-o-4_5-tts-F16.gguf \
    tts/MiniCPM-o-4_5-projector-F16.gguf \
    token2wav-gguf/encoder.gguf \
    token2wav-gguf/flow_matching.gguf \
    token2wav-gguf/flow_extra.gguf \
    token2wav-gguf/hifigan2.gguf \
    token2wav-gguf/prompt_cache.gguf \
    --revision "${OMNI_REV}" \
    --local-dir "${MODEL_ROOT}/MiniCPM-o-4_5-Q4" \
    "${HF_EXTRA[@]}"
}

case "${MODE}" in
  vision46) download_v46 ;;
  vision45) download_v45 ;;
  omni) download_omni ;;
  *)
    echo "Usage: MODEL_ROOT=/private/models $0 vision46|vision45|omni" >&2
    echo "Set DOWNLOAD_DRY_RUN=1 to audit filenames and sizes without downloading." >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete; no weights were downloaded."
  exit 0
fi

TARGET="${MODEL_ROOT}/MiniCPM-V-4.6-Q4"
[[ "${MODE}" == "vision45" ]] && TARGET="${MODEL_ROOT}/MiniCPM-V-4_5-Q4"
[[ "${MODE}" == "omni" ]] && TARGET="${MODEL_ROOT}/MiniCPM-o-4_5-Q4"
(
  cd "${TARGET}"
  find . -type f ! -path './.cache/*' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)
echo "Downloaded pinned ${MODE} lane to ${TARGET}; hashes are in SHA256SUMS."
