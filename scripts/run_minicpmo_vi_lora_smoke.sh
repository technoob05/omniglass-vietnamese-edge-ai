#!/usr/bin/env bash
# Isolated 10-step MiniCPM-o 4.5 Vietnamese thinker-LoRA smoke.
# This launcher deliberately refuses a busy GPU so it cannot steal memory from
# the production MiniCPM-o/VieNeu/PhoWhisper services.
set -euo pipefail

MS_SWIFT_REVISION="ca937fbaf8e0c3dc4ea34358889430e36475463b"
MODEL_REVISION="073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1"
MODEL_ID="openbmb/MiniCPM-o-4_5"
MIN_FREE_MIB="${MIN_FREE_MIB:-36000}"
MAX_STEPS="${MAX_STEPS:-10}"
WORK_ROOT="${WORK_ROOT:?Set WORK_ROOT to an isolated private training directory}"

usage() {
  echo "Usage: TRAIN_GPU=<idle GPU index/UUID> $0 TRAIN.jsonl [VALIDATION.jsonl]" >&2
  echo "The selected GPU must be idle and expose at least ${MIN_FREE_MIB} MiB free." >&2
}

die() {
  echo "minicpmo LoRA smoke refused: $*" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
[[ -n "${TRAIN_GPU:-}" ]] || die "TRAIN_GPU is required; the script never guesses a production GPU"
[[ "${MIN_FREE_MIB}" =~ ^[0-9]+$ ]] || die "MIN_FREE_MIB must be an integer"
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_STEPS must be a positive integer"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
command -v python3 >/dev/null 2>&1 || die "python3 is unavailable"

TRAIN_JSONL="$(realpath "$1")"
[[ -f "${TRAIN_JSONL}" ]] || die "training JSONL does not exist: ${TRAIN_JSONL}"
VAL_JSONL=""
if [[ $# -eq 2 ]]; then
  VAL_JSONL="$(realpath "$2")"
  [[ -f "${VAL_JSONL}" ]] || die "validation JSONL does not exist: ${VAL_JSONL}"
fi

GPU_CSV="$(nvidia-smi -i "${TRAIN_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)" \
  || die "TRAIN_GPU is not a visible GPU index/UUID: ${TRAIN_GPU}"
FREE_MIB="$(printf '%s\n' "${GPU_CSV}" | head -n 1 | tr -d '[:space:]')"
[[ "${FREE_MIB}" =~ ^[0-9]+$ ]] || die "could not parse free memory for TRAIN_GPU=${TRAIN_GPU}"
(( FREE_MIB >= MIN_FREE_MIB )) \
  || die "only ${FREE_MIB} MiB is free; ${MIN_FREE_MIB} MiB is required"

if ! PROCESS_CSV="$(nvidia-smi -i "${TRAIN_GPU}" \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null)"; then
  die "could not inspect compute processes on TRAIN_GPU=${TRAIN_GPU}"
fi
if [[ -n "$(printf '%s' "${PROCESS_CSV}" | tr -d '[:space:]')" ]]; then
  printf '%s\n' "${PROCESS_CSV}" >&2
  die "compute processes already exist on TRAIN_GPU=${TRAIN_GPU}; choose a dedicated GPU/MIG instance"
fi

# Validate the ms-swift schema and keep the smoke bounded. This reads media but
# never copies or edits the source dataset.
python3 - "${TRAIN_JSONL}" "${VAL_JSONL}" <<'PY'
import json
import pathlib
import sys

for raw_path in (item for item in sys.argv[1:] if item):
    path = pathlib.Path(raw_path)
    rows = 0
    for line_no, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SystemExit(f"{path}:{line_no}: messages must be a non-empty list")
        if not any(m.get("role") == "assistant" and m.get("content") for m in messages):
            raise SystemExit(f"{path}:{line_no}: a non-empty assistant answer is required")
        content = "\n".join(str(m.get("content", "")) for m in messages)
        for tag, key in (("<image>", "images"), ("<audio>", "audios")):
            expected = content.count(tag)
            actual = len(row.get(key, []))
            if expected != actual:
                raise SystemExit(
                    f"{path}:{line_no}: {expected} {tag} tags but {actual} entries in {key}"
                )
        for key in ("images", "audios"):
            for media in row.get(key, []):
                if not pathlib.Path(media).is_file():
                    raise SystemExit(f"{path}:{line_no}: missing media: {media}")
    if rows == 0:
        raise SystemExit(f"{path}: no records")
    print(f"validated {rows} rows: {path}")
PY

VENV_ROOT="${WORK_ROOT}/venv-ms-swift-${MS_SWIFT_REVISION:0:12}"
OUTPUT_ROOT="${WORK_ROOT}/output"
mkdir -p "${WORK_ROOT}" "${OUTPUT_ROOT}"

if [[ ! -x "${VENV_ROOT}/bin/swift" ]]; then
  python3 -m venv "${VENV_ROOT}"
  "${VENV_ROOT}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_ROOT}/bin/python" -m pip install \
    "ms_swift @ git+https://github.com/modelscope/ms-swift.git@${MS_SWIFT_REVISION}" \
    "transformers==4.51.3" \
    "minicpmo-utils==1.0.6" \
    timm decord soundfile
fi

SWIFT_ARGS=(
  --model "${MODEL_ID}"
  --model_revision "${MODEL_REVISION}"
  --dataset "${TRAIN_JSONL}"
  --split_dataset_ratio 0
  --tuner_type lora
  --target_modules all-linear
  --freeze_llm false
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  --attn_impl sdpa
  --gradient_checkpointing true
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 4
  --learning_rate 1e-4
  --lora_rank 8
  --lora_alpha 32
  --lora_dropout 0.05
  --max_length 1024
  --max_steps "${MAX_STEPS}"
  --logging_steps 1
  --save_strategy no
  --eval_strategy no
  --report_to none
  --dataloader_num_workers 2
  --output_dir "${OUTPUT_ROOT}"
  --add_version true
)
if [[ -n "${VAL_JSONL}" ]]; then
  SWIFT_ARGS+=(--val_dataset "${VAL_JSONL}" --eval_strategy no)
fi

echo "Launching isolated thinker-LoRA smoke on TRAIN_GPU=${TRAIN_GPU}; production services are untouched."
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
INIT_TTS=false \
INIT_AUDIO=true \
USE_AUDIO_IN_VIDEO=false \
MAX_SLICE_NUMS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"${VENV_ROOT}/bin/swift" sft "${SWIFT_ARGS[@]}"

echo "Smoke training completed under ${OUTPUT_ROOT}."
echo "This proves the trainer contract only; run held-out inference before any quality claim."
