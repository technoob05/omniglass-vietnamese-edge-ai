# MiniCPM-first QCS8550 port

## Decision

Keep native English/Chinese MiniCPM-o intact and add Vietnamese as a separate profile:

```text
/omni  camera + microphone -> MiniCPM-o 4.5 native duplex -> native English/Chinese speech
/vi    camera + Vietnamese ASR -> MiniCPM vision -> Vietnamese TTS
```

H100 remains the known-good `/omni` target. The edge port has four explicitly ranked lanes; none is
called QCS8550-verified until it runs on the physical HSPTEK box.

## Lane A - MiniCPM-V 4.6 modular edge (primary)

This is the best current MiniCPM edge candidate, superseding 4.5 for the first bring-up:

- official `openbmb/MiniCPM-V-4.6` has 1,300,428,016 parameters and is described as an on-device
  model;
- official `MiniCPM-V-4_6-Q4_0.gguf` is 501,256,896 bytes;
- official F16 projector is 1,108,746,944 bytes;
- total weight bundle is 1,610,003,840 bytes before runtime buffers and KV.

Use Q4_0 for the HTP experiment. The pinned Hexagon backend explicitly documents/repackages Q4_0,
Q8_0 and MXFP4; its source does not establish a comparable K-quant fast path. Q4_K_M remains a CPU
quality reference only. The 1.3B lane leaves far more of the box's 16 GB RAM for camera buffers,
ASR, TTS and tracking than MiniCPM-V 4.5.

This lane is event-driven rather than a wasteful continuous VLM loop. Detector/tracker runs on every
frame; MiniCPM receives a fresh frame/ROI on final speech, a state change, or a low-rate scene pulse.

## Lane B - MiniCPM-V 4.5 modular edge (quality fallback)

Keep the known 8.2B visual model only if 4.6 loses the held-out English/Vietnamese glasses suite:

- official Q4_0: 4,773,679,808 bytes;
- official projector: 1,095,113,184 bytes;
- bundle: 5,868,792,992 bytes before runtime buffers and KV.

It may fit 16 GB, but fit is not realtime behavior. It has much higher memory-bandwidth and thermal
risk than 4.6.

## Lane C - MiniCPM-o 4.5 full Omni (native English experiment)

This is the requested unified native English experience. Its pinned Q4_0 bundle is 8,619,262,592
bytes across the LLM, vision, audio, TTS and Token2Wav modules. Start in isolation with:

- one session and one camera;
- context 2,048 before 4,096;
- standard-definition vision;
- native English preset;
- Vietnamese ASR/TTS processes stopped for this bounded device test only.

The cross-build can compile the Omni targets, but that is not execution proof. In the current C++
engine Token2Wav selects a generic `GPU` backend when CUDA is absent, and the Hexagon backend reports
itself as a GPU device. Every TTS/vision/audio operation must therefore be checked for supported HTP
placement and CPU fallback. LLM offload alone is insufficient.

Abort this lane if peak RSS exceeds 75% of physical RAM, an unsupported HTP operation appears,
fallback is silent, 30-minute latency regresses by more than 10%, or the 100-turn native session is
unstable. In that case, keep `/omni` on H100 and ship Lane A locally; do not downgrade to the old
MiniCPM-o 2.6 merely to claim one-model edge execution.

## Lane D - Qwen3-VL-4B control

Use Qualcomm's newer `Qwen3-VL-4B-Instruct` as a controlled comparison, not the product default.
Its published release assets are not measurements from this HSPTEK QCS8550. It can replace MiniCPM
only after winning the identical visual-quality, memory, latency, thermal and license gates.

## Audited runtime facts

The build pins `tc-mb/llama.cpp-omni` master at
`09f5c3f1b484759f17b06fc63574f749c89c8761`. That advertised commit contains both MiniCPM-V 4.6
multimodal support and the Snapdragon Linux backend.

The pinned Linux CMake preset actually enables ARM64 CPU plus Hexagon HTP and explicitly sets
`GGML_OPENCL=OFF`. Upstream prose currently says CPU/OpenCL/Hexagon, but the preset is the executable
source of truth. Do not report Adreno OpenCL for this Linux package.

The Hexagon package includes several HTP architecture libraries. Do not guess the QCS8550 DSP
architecture or number of sessions: discover the physical device using `llama-cli --list-devices`,
the preflight report, and its QAIRT/BSP information.

## Reproducible build and artifact audit

On a Linux x86 host with Docker:

```bash
WORK_ROOT=/private/build/minicpm-qcs8550 \
BUILD_OMNI=1 \
bash scripts/build_minicpm_qcs8550_runtime.sh
```

The build checks the advertised branch against the pinned commit, builds ARM64 CPU/HTP vision and
Omni targets, and writes `BUILD_EVIDENCE.txt`. The default public toolchain image is version-tagged,
not immutable; retain the resolved image ID/digest from that evidence in the board report.

Audit downloads without fetching weights:

```bash
MODEL_ROOT=/private/models/minicpm-edge DOWNLOAD_DRY_RUN=1 \
bash scripts/download_minicpm_qcs8550_models.sh vision46
```

Then fetch only the selected lane. Repository revisions are pinned and `SHA256SUMS` is generated:

```bash
MODEL_ROOT=/private/models/minicpm-edge \
bash scripts/download_minicpm_qcs8550_models.sh vision46

# Bounded full-Omni experiment only (about 8.62 GB):
MODEL_ROOT=/private/models/minicpm-edge \
bash scripts/download_minicpm_qcs8550_models.sh omni
```

## Physical-board benchmark

After copying the package, model and a representative camera frame to the box:

```bash
RUNTIME_ROOT=/opt/omniglass/pkg-snapdragon \
MODEL_ROOT=/opt/omniglass/models/MiniCPM-V-4.6-Q4 \
IMAGE=/opt/omniglass/eval/scene_001.jpg \
OUT_ROOT=/opt/omniglass/results \
bash scripts/benchmark_minicpm_qcs8550.sh
```

The harness records device discovery, package/model hashes, OS/CPU/RAM, CPU reference, Q4_0 HTP
text/VLM runs, `/usr/bin/time -v` peak RSS, Hexagon profiling and thermal snapshots. It rejects an
HTP run with no accelerator evidence. Raw logs are inputs to `scripts/qcs8550_acceptance.py`; one
successful prompt is not a release decision.

For the bounded native-English full-Omni lane, copy the upstream 1-second WAV/JPEG chunk fixture (or
an equivalent consented English fixture) and run `scripts/benchmark_minicpmo_qcs8550.sh`. It first
isolates audio/vision understanding with TTS disabled, then enables Talker/Token2Wav, while recording
peak RSS, HTP logs and thermal snapshots. This is a bring-up test, not a realtime claim.

Bring-up order:

1. Run `qcs8550_preflight.py --require-qcs8550`; save QAIRT/BSP/DSP/camera/audio inventory.
2. Run MiniCPM-V 4.6 Q4_0 CPU, then HTP. Compare exact output and detect fallback.
3. Add detector/tracker, Vietnamese ASR, and Vietnamese TTS one at a time.
4. Compare MiniCPM-V 4.5 only if 4.6 misses the held-out visual gate.
5. Run full MiniCPM-o native English in isolation.
6. Execute 100 turns and a 30-minute thermal soak before promotion.

## Primary sources

- [MiniCPM-V 4.6 model](https://huggingface.co/openbmb/MiniCPM-V-4.6)
- [MiniCPM-V 4.6 official GGUF](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf)
- [MiniCPM-V 4.5 official GGUF](https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf)
- [MiniCPM-o 4.5 official GGUF](https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf)
- [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni)
- [Snapdragon Linux backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md)
- [Qwen3-VL-4B Qualcomm control](https://huggingface.co/qualcomm/Qwen3-VL-4B-Instruct)
