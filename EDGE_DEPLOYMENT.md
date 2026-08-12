# OmniGlass Edge deployment contract

This document separates the working H100 prototype from claims that require physical-board
validation. The immediate loop must remain useful when the network or H100 is unavailable.

## Runtime split

| Layer | H100 prototype | QCS8550/QCS6490 target |
|---|---|---|
| Camera/audio | Browser `getUserMedia` | ISP/QIM/GStreamer, calibrated camera ID |
| Always-on perception | Optional YOLO/tracker workers | Detector on HTP, tracker/policy on CPU |
| Depth | Metric DAV2 research worker | DAV2 Small candidate on HTP; coarse only until calibrated |
| Conversation | MiniCPM-o 4.5 chat mode with fresh frame + rolling text history | H100 hybrid first; local VLM only after board profiling |
| Vietnamese ASR | Silero VAD + PhoWhisper-medium on second H100 | sherpa Zipformer VI INT8 CPU baseline; custom PhoWhisper QNN only after validation |
| Vietnamese TTS | VieNeu v3 Turbo streaming on H100; MMS-TTS fallback | Local critical phrases/Piper baseline; VieNeu/QNN is unverified |
| Safety cue | Prototype only | Deterministic local policy; never wait for an LLM |

Do not send a continuous camera stream to the VLM. Capture one fresh frame or target ROI per
finalized utterance/event. Keep a RAM-only ring buffer of at most two seconds when continuous local
perception needs recent frames.

## Versioned tool envelope

Every request carries `api_version`, `session_id`, `turn_id`, a monotonic capture timestamp and a
deadline. Every response echoes the IDs and identifies the exact model artifact.

```json
{
  "api_version": "1.0",
  "session_id": "...",
  "turn_id": "...",
  "frame_ref": {
    "frame_id": 42,
    "captured_monotonic_ns": 123456789,
    "width": 1280,
    "height": 720,
    "rotation_deg": 0,
    "camera_id": "rear-wide",
    "calibration_id": "rear-wide-1280x720-v1",
    "sha256": "..."
  },
  "deadline_ms": 2500
}
```

Required tools are `describe`, `detect`, `track`, `depth`, `stop` and `help`. The planner may emit
only this allowlisted enum and validated arguments; model text never directly executes code. A
depth result returns `distance_m: null` unless the camera/model pair is calibrated and quality
checks pass. Otherwise it returns only `near`, `mid` or `far` plus a warning.

The conversation FSM is `LISTENING -> CAPTURED -> THINKING -> SPEAKING -> LISTENING`. There is one
active turn, explicit cancellation, latest-wins stale-response rejection and a `finally` transition
back to listening on every error. Typed events use monotonically increasing sequence IDs and are
idempotent.

## Model manifest

Every deployed model artifact records:

- upstream repository and commit;
- model/checkpoint license and SHA-256;
- input shape, color order, layout and normalization;
- target SoC, precision, QAIRT version and artifact format;
- calibration dataset hash and task-level accuracy report;
- actual-board latency, memory, power and thermal report.

Proxy or AI Hub profile numbers are screening evidence, not physical HSPTEK/RB3 performance.

The machine-readable Vietnamese ASR inventory is
[`manifests/edge_vi_asr_baselines.json`](manifests/edge_vi_asr_baselines.json). Run
`scripts/smoke_edge_vi_asr.py` to verify every frozen asset hash before inference. A claim marked
`planned`, `planned_blocked`, or `upstream_verified` is not board evidence.

## Vietnamese edge ASR baseline

The current edge recommendation is `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` on the ARM64
CPU, with Silero VAD/endpointing in front. This is an offline RNN-Transducer utterance model; the
upstream “simulated streaming” example means VAD-cut utterances, not a stateful incremental ASR
decoder. The official archive contains an approximately 26 MiB int8 encoder, 4.9 MiB decoder and
1 MiB int8 joiner. Our pinned archive and every runtime asset have SHA-256 values in the manifest.

What is verified now:

- the official archive hash and all extracted model hashes;
- `sherpa-onnx==1.13.2`, provider `cpu`, one thread, and `CUDA_VISIBLE_DEVICES` empty;
- successful qualitative decoding of the three bundled Vietnamese WAVs on Linux x86_64 CPU;
- a reproducible JSON smoke result in `artifacts/edge-vi-sherpa-smoke-pod-b.json`.

What remains planned:

- Linux ARM64 loading and inference on each delivered QCS8550/QCS6490 device;
- WER/CER on a versioned Vietnamese golden set;
- device P50/P95 latency, peak RSS, power, temperature, throttling and 30-minute stability;
- microphone/VAD integration under real room noise.

The model card declares `CC-BY-NC-ND-4.0`, while sherpa-onnx itself is Apache-2.0. The official
model archive does not include a license file. For the non-commercial school demonstration, keep
the upstream model unmodified, preserve attribution and do not redistribute a converted model.
Commercial use or distribution of a quantized/converted derivative requires separate permission
and legal review. This restriction is a release blocker even if runtime performance is acceptable.

### CPU smoke command

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/path/to/sherpa-runtime \
python scripts/smoke_edge_vi_asr.py \
  --manifest manifests/edge_vi_asr_baselines.json \
  --asset-root /path/to/edge-vi-baseline \
  --output artifacts/edge-vi-sherpa-smoke.json
```

The harness is CPU-only by construction and reports artifact integrity, runtime version, load
time, per-file latency/RTF, transcripts and peak process RSS. Transcripts without a gold reference
are qualitative smoke evidence, not an accuracy result.

## PhoWhisper QNN feasibility boundary

`vinai/PhoWhisper-medium` is pinned at revision
`55a7e3eb6c906de891f8f06a107754427dd3be79` and its PyTorch checkpoint and processor files are
hashed in the manifest. It is a 769M-parameter, 24-layer encoder plus 24-layer autoregressive
decoder. The working H100 service does not demonstrate QNN compatibility.

There is currently no PhoWhisper ONNX encoder/decoder export, calibration-set hash, QNN conversion
report, context binary, exact-board runtime result or accuracy comparison. QNN work must split the
encoder and one-token decoder with explicit KV cache, freeze supported shapes, validate the host
token loop, calibrate Vietnamese audio and compile separately for each SoC. QCS6490 should begin
with fully quantized INT8 I/O; QCS8550 can screen FP16/W8A16/W8A8, but the choice is not accepted
until task-level accuracy and physical-board measurements pass.

The official sherpa-onnx QNN documentation demonstrates device-specific context binaries for
selected fixed-duration Zipformer-CTC and SenseVoice models on another Snapdragon target. It does
not supply a Vietnamese PhoWhisper-medium binary or prove that a binary is portable to QCS8550 or
QCS6490. An official Qualcomm AI Hub issue also documents a historical Whisper-medium out-of-memory
conversion and support only through Whisper-small at that time; treat this as feasibility risk,
not a claim about every newer QAIRT release.

Decision for the summer-school build: ship the int8 Zipformer CPU path first. Keep PhoWhisper on
H100 as the measured hybrid fallback and treat PhoWhisper/QNN as a separate research spike.

## QAIRT/QNN release path

1. Freeze the exact board, BSP/OS, camera/audio path and installed QAIRT version.
2. Export a static ONNX graph with locked preprocessing and compare it numerically with PyTorch.
3. Compile/profile through Qualcomm AI Hub Workbench for the exact target where available. Start
   with W8A8 for small detection models and test W8A16/FP16 for accuracy-sensitive depth.
4. Validate on a held-out golden set using task metrics: recall/mAP, IDF1/lost rate, Vietnamese
   WER/CER and calibrated depth error. Cosine similarity alone is not an acceptance test.
5. Integrate a QNN context binary or QNN delegate through native QAIRT/QIM. Warm models at boot and
   verify HTP placement; treat CPU fallbacks as visible performance failures.
6. Run camera-to-output profiling on the physical board for at least 30 minutes, including power,
   temperature, peak RSS and throttling.
7. Sign the model manifest/artifacts, add startup self-test, A/B update and rollback.

## Provisional acceptance targets

These are targets, not achieved device measurements:

- detector/tracker: at least 15 FPS sustained on QCS6490 and 20 FPS on QCS8550;
- local critical cue: camera-to-audio P95 at most 250 ms;
- captured turn frame visible within 100 ms of end-of-speech;
- hybrid VQA first text P95 at most 2.5 s and first audio at most 3.0 s on controlled LAN;
- local depth update P95 at most 250 ms, with metric values only after calibration;
- 100 consecutive turns and a 30-minute run with at least 99% return to `LISTENING`, no overlapping
  turns and no latency regression above 10% from thermal throttling;
- no raw image/audio in logs, no persistent recording by default, encrypted transport and an
  explicit cloud/microphone indicator.

## Staged migration

1. Stabilize the atomic H100 turn, IDs, deadlines and timings.
2. Bring up camera/mic/speaker and a power/thermal harness on the delivered board.
3. Move YOLO + tracking + deterministic alerts to the device; validate network-loss fallback.
4. Move coarse depth, calibrate each camera mode and keep meters disabled until validation passes.
5. Add local Vietnamese VAD/ASR and prerecorded critical phrases; migrate TTS independently.
6. Use a deterministic tool router locally and H100 Qwen-VL only for open-ended descriptions.
7. Evaluate local VLM/GenieX only as a measured QCS8550 feasibility spike.
8. Complete security, OTA, accessibility field testing and product-claim review.

## Current claim boundary

The current demo is not fully offline, not a certified navigation aid and does not provide trusted
metric distance. Do not claim MiniCPM-o, PhoWhisper, VieNeu or Vietnamese MMS-TTS runs on Qualcomm NPU, do not quote
QCS8550 Proxy latency as HSPTEK board latency, and do not infer application speed from TOPS.

Official references: [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/),
[RB3 Gen 2](https://www.qualcomm.com/developer/hardware/rb3-gen-2-development-kit),
[QCS6490](https://www.qualcomm.com/internet-of-things/products/q6-series/qcs6490),
[Qualcomm YOLOv8](https://huggingface.co/qualcomm/YOLOv8-Detection), and
[Qualcomm Depth Anything V2](https://huggingface.co/qualcomm/Depth-Anything-V2),
[PhoWhisper](https://github.com/VinAIResearch/PhoWhisper),
[VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS), and
[sherpa-onnx Vietnamese Zipformer](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html),
[sherpa-onnx QNN models](https://k2-fsa.github.io/sherpa/onnx/qnn/models.html),
[Zipformer VI 30M model card](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h),
[PhoWhisper-medium model card](https://huggingface.co/vinai/PhoWhisper-medium), and
[Qualcomm Whisper-medium conversion issue](https://github.com/quic/ai-hub-models/issues/95).
