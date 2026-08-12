# HSPTEK AIBOX 8550 deployment gate

This is the board-specific release contract for the Vietnamese OmniGlass edge build. It is a
deployment plan and an evidence checklist, not a claim that the current H100 demo or any model has
already run on a physical HSPTEK device.

## What is known upstream

HSPTEK advertises the AIBOX 8550 with a Qualcomm QCS8550-family SoC, 16 GB uMCP memory, 128 GB
storage, Android/Ubuntu/Linux support, 48 TOPS INT8, 12 TFLOPS FP16 and a 16 W TDP. These are vendor
specifications, not application benchmarks. The exact delivered SKU, BSP, camera/audio devices and
thermal design still have to be inventoried.

The Qualcomm `qai-appbuilder` repository explicitly lists QCS8550 Linux support. Its current
documentation and samples cover QNN context binaries, HTP execution, a Genie OpenAI-compatible
service, Qwen2.5-VL-3B, Whisper Tiny/Base English, Zipformer Chinese, Piper English and MeloTTS
Chinese. This demonstrates available integration patterns. It does **not** provide a Vietnamese
QNN artifact and it does not prove compatibility or latency on the HSPTEK board.

Pinned evidence and every acceptance gate are machine-readable in
[`manifests/qcs8550_deployment.json`](manifests/qcs8550_deployment.json).

## Recommended product split

Keep the always-on safety loop deterministic and local:

1. QIM/GStreamer camera capture produces timestamped frames and a two-second RAM-only ring buffer.
2. A small QNN detector plus a CPU tracker updates at 15--20 FPS. Safety alerts come from typed,
   allowlisted rules and never wait for an LLM.
3. Silero-style endpointing plus the Vietnamese Zipformer INT8 CPU baseline handles finalized
   utterances first. A PhoWhisper QNN port is a separate feasibility experiment.
4. A local tool router chooses `describe`, `detect`, `track`, `depth`, `stop` or `help`. Invalid,
   late or stale actions are rejected.
5. Qwen2.5-VL-3B through Genie/QNN is the first local VLM candidate. Until physical profiling and
   Vietnamese quality tests pass, use the measured H100 VLM as a hybrid fallback.
6. Critical warnings use prerecorded Vietnamese phrases. General speech uses a CPU Vietnamese TTS
   candidate or the H100 VieNeu service until a licensed, accurate QNN artifact is produced.

MiniCPM-o 4.5 full duplex is not the first on-device target. Its memory footprint, multimodal
runtime support, audio talker path and thermals are not yet proven on QCS8550. Port the modular
pipeline first, then compare a compact unified model against the same acceptance suite.

## Runtime paths

### QNN / QAI AppBuilder

Use this for fixed-shape CV, depth and ASR graphs:

- freeze preprocessing, tensor names/shapes, quantization and calibration-set hash;
- compile a context binary for the exact DSP architecture and installed QAIRT version;
- load through `QNNContext`/native I/O and verify HTP placement in profiling output;
- treat silent CPU fallback as a failed test;
- compare task-level accuracy against the frozen PyTorch/ONNX reference before latency testing.

### Genie

Use Genie for a supported local LLM/VLM artifact. The upstream service accepts OpenAI-compatible
requests and documents Qwen2.5-VL-3B plus LoRA adapter arguments. A Hugging Face LoRA file is not
automatically a deployable Genie adapter: base revision, tensor mapping, quantization, adapter
format and runtime version must all match and be recorded. First prove the unmodified base model,
then the Vietnamese adapter.

### CPU fallback

The CPU path is a release feature, not an accident. It must keep VAD, final Vietnamese ASR,
critical prerecorded speech, tool routing and stop/help functional if HTP or the network fails.
Surface fallback state in telemetry and the UI.

## Bring-up sequence

1. Copy this repository to the board and run:

   ```bash
   python3 scripts/qcs8550_preflight.py --output artifacts/qcs8550-preflight.json
   ```

   The collector is read-only and does not capture camera frames or microphone audio. Re-run with
   `--require-qcs8550` in CI; it returns non-zero until the target identity is detected.

2. Fill every `required_inventory` value in the manifest from the report and vendor BSP package:
   board/SKU/serial pseudonym, OS image and kernel, QAIRT/QNN versions, DSP architecture, firmware,
   camera modes/calibration, audio devices, memory/storage and thermal/power sensors.

3. Validate camera timestamps, rotation and dropped frames before loading models. Validate the
   microphone sample rate, channel map, echo cancellation and speaker routing before ASR/TTS.

4. Bring up one model at a time: detector, tracker, Vietnamese ASR, prerecorded TTS, depth, then
   VLM. Record SHA-256 and a signed manifest for every binary.

5. Run accuracy, latency and failure tests on a versioned Vietnamese golden set. Proxy profiling
   may reject bad candidates, but only physical-board results can promote a gate to `verified`.

6. Put the raw scalar results and report hashes in a JSON evidence file, then run:

   ```bash
   python3 scripts/qcs8550_acceptance.py \
     --preflight artifacts/qcs8550-preflight.json \
     --metrics artifacts/qcs8550-physical-metrics.json \
     --output artifacts/qcs8550-acceptance.json
   ```

   A zero exit code means “ready for human release review”, not “certified” or automatically
   `verified`.

## Physical acceptance

No product claim is allowed until all mandatory gates pass on the delivered HSPTEK unit:

- cold boot, model integrity check and warm-up complete without network access;
- detector/tracker sustains at least 20 FPS and never silently falls back from HTP;
- local critical camera-to-audio warning P95 is at most 250 ms;
- Vietnamese ASR reports speaker-disjoint WER/CER and P50/P95 endpoint-to-final latency;
- local or hybrid VQA reports fresh-frame age and speech-end-to-first-text/audio latency;
- 100 consecutive turns and a 30-minute camera/mic soak return to listening at least 99% of the
  time, with no overlapping turns, memory growth or latency regression above 10%;
- power, temperature and frequency telemetry show no unreported thermal throttling;
- unplugging the network preserves detect/track/stop/help and critical Vietnamese warnings;
- raw camera/audio is RAM-only by default, logs contain no media, transport is encrypted and
  microphone/cloud state is visible.

The final report must include the physical board identifier pseudonym, ambient temperature,
camera/mic configuration, QAIRT version, artifact hashes and raw percentile samples. A statement
such as “QCS8550 Proxy passed” is never acceptable board evidence.

## Current claim boundary

As of this manifest, QCS8550 support is `upstream_verified`, while all OmniGlass physical-board
results are `planned` or `blocked`. In particular, Qwen2.5-VL-3B, Vietnamese Zipformer,
PhoWhisper, VieNeu-TTS and MiniCPM-o have not been reproduced on an HSPTEK AIBOX 8550 here.

Sources: [HSPTEK AIBOX 8550](https://hsptek.com/vi/san-pham/aibox-8550/),
[Qualcomm QAI AppBuilder](https://github.com/qualcomm/qai-appbuilder),
[QAI AppBuilder guide](https://github.com/qualcomm/qai-appbuilder/blob/main/docs/guide_en.md),
[Genie API guide](https://github.com/qualcomm/qai-appbuilder/blob/main/docs/genie_guide_en.md), and
[Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/).
