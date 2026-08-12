# HSPTEK AIBOX 8550 deployment gate

This is the board-specific release contract for the OmniGlass edge build. On 2026-08-12 the
connected Kalama board completed a real QNN/HTP smoke test. That proves the physical V73 HTP
runtime works; it does **not** yet prove the complete English or Vietnamese interaction stack.

## Physical bring-up verified on 2026-08-12

- Linux identifies the board as `QCS_KALAMAP`, Snapdragon SoC ID 603, revision 2.0.
- Ubuntu 22.04.2 aarch64 exposes 15,552,844 KiB RAM and six online CPU cores.
- QAIRT/QNN and SNPE are `2.36.0.250627101419_123260`.
- `qnn-platform-validator` reports DSP prerequisites present, libraries found and Hexagon V73.
- A cached YOLOv5n V73 context ran 20/20 inferences through `libQnnHtp.so`. Detailed profiling
  reports four HVX threads, per-operation accelerator cycles and an average accelerator execution
  time of about 22.43 ms. This is a graph microbenchmark, not camera-to-detection latency.
- The debug BSP required `adb root` for unsigned HTP protection-domain access. The USB transport
  detached after that session and must be reconnected before the next physical step.

The frozen machine-readable evidence is in
[`manifests/qcs8550_english_experiment.json`](manifests/qcs8550_english_experiment.json). MiniCPM,
Whisper, EdgeTAM, depth, TTS, power and sustained thermal behavior are still separate gates.

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

## Additive language profiles

English/Chinese and Vietnamese are not competing deployments:

- **`/omni` native EN/ZH:** preserve MiniCPM-o full-duplex audio/vision/Talker behavior unchanged.
- **`/vi` additive profile:** reuse the MiniCPM visual/conversation brain, but feed an immutable
  Vietnamese ASR final and synthesize the resulting Vietnamese text with a Vietnamese TTS.
- Selecting Vietnamese must never overwrite, merge into, or disable the English model/runtime.
  The UI selects a language profile; the service supervisor may share model weights when the
  runtime supports it, but session and audio state remain isolated.

For QCS8550, the same product behavior is preserved. Native English is the compatibility control;
Vietnamese adds ASR/TTS adapters around the vision brain. Every device test runs both profiles so
an optimization cannot silently regress English.

## Recommended product split

Keep the always-on safety loop deterministic and local:

1. QIM/GStreamer camera capture produces timestamped frames and a two-second RAM-only ring buffer.
2. A small QNN detector plus a CPU tracker updates at 15--20 FPS. Safety alerts come from typed,
   allowlisted rules and never wait for an LLM.
3. Silero-style endpointing plus the Vietnamese Zipformer INT8 CPU baseline handles finalized
   utterances first. A PhoWhisper QNN port is a separate feasibility experiment.
4. A local tool router chooses `describe`, `detect`, `track`, `depth`, `stop` or `help`. Invalid,
   late or stale actions are rejected.
5. MiniCPM is the primary visual-family candidate. Test MiniCPM-o 4.5 Q4_0 as the full native
   compatibility lane, then MiniCPM-V 4.6 1.3B Q4_0 plus modular speech as the memory-efficient product
   lane. Qwen3-VL-4B-Instruct is a current-generation reference candidate only; do not replace
   MiniCPM unless the same Vietnamese/English visual suite proves it better on the physical box.
6. Critical warnings use prerecorded Vietnamese phrases. General speech uses a CPU Vietnamese TTS
   candidate or the H100 VieNeu service until a licensed, accurate QNN artifact is produced.

### First physical English experiment

The smallest useful experiment does not try to move the entire Omni model on day one. Preserve the
known-good native-English MiniCPM-o conversation on H100 and move only the deterministic perception
loop to the box:

```text
QCS8550 camera -> YOLO11-N W8A16/QNN HTP -> CPU tracker -> local critical warning
              -> Depth-Anything-V2 Small W8A16/HTP only when requested
              -> newest frame/ROI + microphone -> existing H100 MiniCPM-o English workflow
```

This gives a deployable offline detect/track/warn path while retaining the interaction quality the
current demo already has. English Whisper Tiny on QNN is phase two; it must be exported for the
actual QCS8550 instead of reusing an artifact named for Snapdragon X Elite. The installed GenieX
service with Qwen3-0.6B is a **text-planner control only** until its NPU placement is profiled. It is
not a VLM. Qualcomm's current Qwen2.5-VL and Qwen3-VL AI Hub cards do not list QCS8550, so local
free-form visual QA remains a BYOM research lane rather than an official deployment path.

MiniCPM-V 4.6 remains an important research lane, but its 1.61 GB GGUF bundle is not a QNN context
binary. Memory fit on CUDA or CPU cannot prove that the language graph and F16 projector remain on
HTP. Keep it off the critical path until the physical Hexagon build reports HTP placement for both
parts with no fallback. The complete frozen experiment is in
[`manifests/qcs8550_english_experiment.json`](manifests/qcs8550_english_experiment.json).

Prepare exports on a configured host without installing or changing anything automatically:

```bash
AI_HUB_MODELS_ROOT=/private/src/ai-hub-models \
OUT_ROOT=/private/artifacts/qcs8550-english \
bash scripts/export_qcs8550_english_qnn.sh

# Review EXPORT_PLAN.txt, then explicitly authorize Workbench jobs:
RUN_EXPORT=1 AI_HUB_MODELS_ROOT=/private/src/ai-hub-models \
OUT_ROOT=/private/artifacts/qcs8550-english \
bash scripts/export_qcs8550_english_qnn.sh
```

Stage selected context binaries plus portable board tools in a checksum-pinned directory:

```bash
python3 scripts/stage_qcs8550_english_bundle.py \
  --destination /private/stage/qcs8550-english-v1 \
  --artifact /private/artifacts/qcs8550-english/yolov11_detector.bin \
  --artifact /private/artifacts/qcs8550-english/depth_anything_v2.bin
```

After copying the bundle to the physical box, run `qcs8550_preflight.py --require-qcs8550`, then
benchmark one context at a time. The benchmark uses `qnn-net-run --retrieve_context`, captures raw
logs and thermal snapshots, and fails when it cannot find HTP evidence:

```bash
python3 benchmark_qcs8550_english_stack.py \
  --qnn-net-run /opt/qairt/bin/aarch64-ubuntu-gcc9.4/qnn-net-run \
  --htp-backend /opt/qairt/lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so \
  --detector-context /opt/omniglass/models/yolov11_detector.bin \
  --detector-input-list /opt/omniglass/eval/yolo_input_list.txt \
  --depth-context /opt/omniglass/models/depth_anything_v2.bin \
  --depth-input-list /opt/omniglass/eval/depth_input_list.txt \
  --output /opt/omniglass/results/english-stack-001
```

Process-wall percentiles from this microbenchmark include process startup and are deliberately
conservative. They do not replace resident-context camera-to-warning or full conversation metrics.

MiniCPM-o 4.5 full duplex remains the desired compatibility target, but its memory footprint,
multimodal runtime, Talker path and thermals are not yet proven on QCS8550. The first usable edge
candidate is therefore MiniCPM-V 4.6 1.3B Q4_0 (501,256,896-byte language GGUF plus a
1,108,746,944-byte projector) with separate Vietnamese ASR/TTS. This uses the newer 4.6 visual
model rather than falling back to MiniCPM-o 2.6. The full MiniCPM-o Q4_0 bundle is still tested as a
bounded lane, not discarded.

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

Physical QNN/HTP bring-up is now verified on the connected Kalama development image: QAIRT 2.36,
Hexagon V73 and one YOLO context completed 20/20 inferences with accelerator profiling. The full
OmniGlass product remains blocked: selected detector/depth exports, English ASR/TTS, local VLM,
MiniCPM, camera-to-speech latency, power/thermal and long-run reliability are not yet accepted.

Sources: [HSPTEK AIBOX 8550](https://hsptek.com/vi/san-pham/aibox-8550/),
[Qualcomm QAI AppBuilder](https://github.com/qualcomm/qai-appbuilder),
[QAI AppBuilder guide](https://github.com/qualcomm/qai-appbuilder/blob/main/docs/guide_en.md),
[Genie API guide](https://github.com/qualcomm/qai-appbuilder/blob/main/docs/genie_guide_en.md), and
[Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com/docs/),
[Qualcomm YOLOv11-Detection](https://aihub.qualcomm.com/models/yolov11_det), and
[Qualcomm Depth-Anything-V2](https://aihub.qualcomm.com/models/depth_anything_v2).
