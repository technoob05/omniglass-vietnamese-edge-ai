# Vietnamese adaptation and QCS8550 plan

## Decision

Use two complementary tracks instead of trying to turn one model into the whole product:

1. **H100 teacher and quality track:** keep the verified PhoWhisper -> MiniCPM-o -> VieNeu
   pipeline, then evaluate an LLM-side LoRA for better Vietnamese visual answers and tool plans.
2. **QCS8550 product track:** preserve the same typed turn/tool contract, but run small dedicated
   ASR, detector/tracker, VLM and TTS components on the device.  The physical board decides which
   components use HTP, GPU or CPU.

The LoRA experiment is now technically plausible because current ms-swift lists
`OpenBMB/MiniCPM-o-4_5` with the `minicpmo4_5` template and image, video, omni and audio
modalities.  It initializes audio by default and TTS only when `INIT_TTS=true`.  This is evidence
for supervised multimodal adaptation of the Hugging Face model; it is **not** evidence that the
native full-duplex Talker/CosyVoice path can be fine-tuned or exported to the existing GGUF
runtime.  OpenBMB's full-model and audio-duplex fine-tuning questions remain open.

## Experiments in priority order

### E0 - prompt and router baseline

Retain the current measured `/vi` profile.  Freeze its model revisions and run every later
candidate against the same questions, frames, ASR finals and latency harness.  A candidate must
not regress abstention, exact-once tool execution, barge-in, or response latency.

### E1 - MiniCPM-o thinker LoRA

Train an LLM-only LoRA on Vietnamese visual-assistance examples:

- freeze vision, audio encoder and aligner;
- do not initialize TTS (`INIT_TTS=false`);
- apply rank-16 LoRA to LLM linear layers;
- train short Vietnamese answers, OCR transcription, object-finding language, safe abstention and
  strict typed tool plans;
- validate on speaker/session/scene-disjoint data;
- load the adapter through ms-swift inference first.  Do not patch production or merge/export to
  GGUF until output parity is proven.

This experiment improves Vietnamese reasoning and response style.  It does not replace
PhoWhisper or VieNeu.

### E2 - PhoWhisper adaptation

Fine-tune PhoWhisper-small first, then medium only if necessary, on consented glasses/phone-mic
speech.  Include North, Central and South speakers, commands, names, units, code-switching,
street/cafe noise and room impulse responses.  Compare against the pinned medium checkpoint on
raw WER, normalized WER, number/name accuracy, intent accuracy and endpoint-to-final latency.

This is likely to deliver more user-visible value than trying to teach MiniCPM-o native ASR
Vietnamese.  Keep the current FLEURS test sample evaluation-only; never train on it.

### E3 - preference tuning after SFT

Only after E1 passes, collect paired Vietnamese answers and run DPO/ORPO on concise, grounded,
non-alarming responses.  Do not start GRPO or autonomous tool rewards before deterministic SFT
and tool-schema validation pass; reward hacking is especially risky in assistive outputs.

### E4 - native Talker research spike

Treat native MiniCPM-o Vietnamese speech generation as research-only.  It requires an official or
reproducible recipe for Talker/audio-duplex training, suitable Vietnamese speech data with speaker
consent, and proof that the trained components work in the deployed streaming backend.  Until
then, VieNeu remains the primary Vietnamese voice and native audio stays disabled in `/vi`.

## Dataset contract

Run `scripts/prepare_vietnamese_omni_dataset.py` on a private source JSONL.  Every row must have:

- a unique `id` and stable `group_id`/`speaker_id`;
- `consent.training=true`, a license/provenance label and private raw-media policy;
- a supported task, `question_vi`, `answer_vi`, and an image and/or audio path;
- optional explicit `train`, `validation`, or `test` split.

The builder writes ms-swift JSONL and fails on group leakage or identical media crossing splits.
Raw frames/audio are referenced and are ignored by Git.  Example source row:

```json
{"id":"scene-001-q1","group_id":"speaker-001-session-01","task":"describe_scene","question_vi":"Trước mặt tôi có gì?","answer_vi":"Phía trước có một chiếc bàn và hai chiếc ghế.","images":["private/scene-001.jpg"],"consent":{"training":true,"raw_media_public":false},"license":"project-consent-v1"}
```

Public datasets are candidates, not a single unrestricted pool:

| Data | Intended use | Status |
|---|---|---|
| FLEURS Vietnamese | ASR read-speech evaluation | CC-BY-4.0; keep current test subset evaluation-only |
| Mozilla Common Voice Vietnamese | ASR training/evaluation | obtain from Mozilla Data Collective; record exact release/license |
| MUSAN + OpenSLR RIR | noise/reverberation augmentation | preserve attribution and source split |
| ViTextVQA / ViOCRVQA / ViSignVQA | research VQA/OCR evaluation | license/redistribution permission must be resolved before training |
| own glasses-domain recordings | primary adaptation/evaluation | explicit participant consent, speaker-disjoint splits, no raw public upload |

Synthetic teacher answers may expand prompts, but every synthetic row must retain its source frame,
teacher revision and human verification status.  Synthetic-only validation is prohibited.

## H100 LoRA smoke

The checked-in launcher is intentionally a smoke-first command.  It requires an isolated ms-swift
environment and private train/validation JSONL:

```bash
TRAIN_GPU=0 \
WORK_ROOT=/private/runs/minicpmo-vi-lora \
bash scripts/run_minicpmo_vi_lora_smoke.sh \
  /private/omniglass-vi/train.jsonl \
  /private/omniglass-vi/validation.jsonl
```

Start with 32-100 reviewed rows and `MAX_STEPS=2` to prove loading, encoding and backward pass.
The fail-closed launcher requires an explicitly selected idle GPU with at least 36 GB free and
creates a revision-pinned isolated environment. A quality run requires a frozen dataset manifest,
a held-out test set and an explicit experiment ID. Never train inside the production MiniCPM
environment or while a shared MIG lacks memory headroom.

## QCS8550 target architecture

The HSPTEK box lists QCS8550, 16 GB memory, Android/Ubuntu/Linux and a 16 W TDP.  Qualcomm's current
QAI AppBuilder documents Linux QCS8550, QNN context/DLC execution, Genie LLM/VLM service, Whisper,
and dynamic LoRA support.  These facts establish a deployment route, not measured performance on
the physical HSPTEK image.

Recommended first board stack:

| Loop | Candidate | Initial processor |
|---|---|---|
| camera + resize | QIM/GStreamer/ISP | hardware/zero-copy where supported |
| always-on detection | YOLO nano INT8 | HTP |
| prompt tracking | EdgeTAM QNN pieces + CPU memory attention | HTP + CPU |
| Vietnamese VAD/ASR | Zipformer VI INT8 baseline; custom PhoWhisper-small later | CPU first, HTP after validated conversion |
| open visual QA | Qwen2.5-VL-3B through Genie/QAI AppBuilder | NPU feasibility gate |
| planning | deterministic router; small local Qwen only for ambiguity | CPU/NPU |
| Vietnamese speech | prerecorded critical phrases; Piper/VieNeu CPU experiment | CPU/audio subsystem |
| complex fallback | H100 MiniCPM-o teacher | optional network path |

Do not put critical alerts through the VLM.  Detector/tracker/depth events use deterministic rules;
the VLM describes or disambiguates on demand.  If the local VLM misses its memory/thermal/latency
gate, the product remains a hybrid edge system rather than forcing MiniCPM-o 9B onto the box.

## Go/no-go gates

- Dataset: consent and license present; zero speaker/session/media leakage; at least three regions
  and realistic glasses noise represented before a quality claim.
- LoRA: visual task success improves by at least 5 points or Vietnamese correction turns fall by
  at least 20%, while English control degrades by no more than 2 points.
- ASR: intent >=95%, names/numbers >=90%, spontaneous accent WER <=15%; endpoint-final P95 <=1 s.
- Device: 100 turns and 30 minutes; no overlap/leak/restart; >=25% RAM headroom; no thermal latency
  regression above 10%.
- Interaction: local critical cue P95 <=250 ms; open VQA first audio P95 <=3 s in hybrid mode.
- Safety: calibrated uncertainty required for metres; otherwise report only coarse near/mid/far.

All thresholds are targets until measured on the delivered physical QCS8550 box.

## Primary references

- [ms-swift supported models](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)
- [ms-swift custom multimodal datasets](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md)
- [ms-swift MiniCPM-o audio-support release](https://github.com/modelscope/ms-swift/releases)
- [OpenBMB MiniCPM-o fine-tuning issue](https://github.com/OpenBMB/MiniCPM-V/issues/1071)
- [OpenBMB audio-duplex fine-tuning issue list](https://github.com/OpenBMB/MiniCPM-o/issues)
- [QAI AppBuilder](https://github.com/qualcomm/qai-appbuilder)
- [HSPTEK AIBOX 8550](https://hsptek.com/vi/san-pham/aibox-8550/)
