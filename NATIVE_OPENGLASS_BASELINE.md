# Native OpenGlass / MiniCPM-o baseline

This keeps the MiniCPM-o 4.5 model/runtime and browser protocol used by the
current OpenGlass Omni path. The browser talks directly to the official
MiniCPM-o gateway; there is no Gradio or browser SpeechRecognition in the
realtime path. A local `Vietnamese Call` preset adds a Vietnamese system prompt
and preserves MiniCPM-o's native audio/vision reasoning. For this preset only,
the unsupported English/Chinese CosyVoice output is disabled and the completed
Vietnamese text is spoken by `facebook/mms-tts-vie`. English and Chinese
presets still use the original native TTS path. The Vietnamese adapter now
streams VieNeu v3 Turbo PCM to a browser-owned `AudioContext`; MMS-TTS remains
the automatic fallback when VieNeu fails before emitting its first chunk.

## Running deployment

- Local desktop URL: `https://127.0.0.1:8006/omni`
- Dedicated Vietnamese URL: `https://127.0.0.1:8006/vi`
- Mobile-oriented page: `https://127.0.0.1:8006/mobile-omni/`
- Remote runtime root: `$OPENGLASS_NATIVE_ROOT`
- GPU: H100 MIG 4g.40gb reached through `$SSH_USER@$H100_HOST:$SSH_PORT`
- Runtime topology: browser -> HTTPS/WSS gateway `8006` -> worker `22400` ->
  PyTorch backend `22500`
- Vietnamese speech adapter: gateway `/api/tts/vi/stream` -> VieNeu service
  `18782`; MMS-TTS service `18781` is the fallback
- Vietnamese ASR adapter: same-origin gateway `/v1/asr/vi` -> persistent
  PhoWhisper-medium + Silero VAD service `18783`

The certificate is self-signed. Accept the local certificate warning once,
then grant camera and microphone permission. Select **Vietnamese Call** and
press **Start**. The English and Chinese upstream presets remain available.

For the best Vietnamese path, use `/vi` and press **Bắt đầu hội thoại** once.
After that, finalized Vietnamese utterances automatically capture a fresh frame,
query MiniCPM-o and stream a VieNeu answer; no per-question button is required.
The default voice is **Trúc Ly** (natural Northern female); the page also offers
natural Central/Southern and male presets without restarting the model.

The upstream `/mobile` route currently redirects to a frontend bundle that is
not included in the clean repository checkout. Use `/mobile-omni/` directly.

## Provenance

- OpenGlass: `1ffe701e808cc67aa6462bc6071c61413302f17d`
- MiniCPM-o-Demo: `d0a002093615b7f1d4d0f87a03fc01cb39bef3f6`
- llama.cpp-omni (cloned for the later Q4 path):
  `09f5c3f1b484759f17b06fc63574f749c89c8761`
- Model: `openbmb/MiniCPM-o-4_5`, downloaded snapshot metadata starts at
  `073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1`
- Torch: `2.4.0+cu124`
- Transformers: `4.51.0`
- Model directory size: `20,048,682,892` bytes
- Loaded GPU footprint observed: approximately `22.47 GiB`

## Re-run

On the H100 pod:

```bash
cd "$OPENGLASS_NATIVE_ROOT"
./start_native_minicpmo_h100.sh
```

On the Windows laptop:

```powershell
ssh -N `
  -L 8006:127.0.0.1:8006 `
  -p "$SSH_PORT" "$SSH_USER@$H100_HOST"
```

Stop only the processes owned by this baseline:

```bash
cd "$OPENGLASS_NATIVE_ROOT"
./stop_native_minicpmo_h100.sh
```

Reload only the gateway after adding or changing presets, while keeping the
H100 model and worker warm:

```bash
./reload_native_gateway_h100.sh
```

Reinstall/update the dedicated Vietnamese profile idempotently and optionally
reload only the gateway:

```bash
OPENGLASS_VI_PROFILE_RELOAD=true ./scripts/install_native_vietnamese_profile_h100.sh
```

The environment bootstrap is captured in
`scripts/bootstrap_native_minicpmo_h100.sh`. It intentionally uses a clean
virtualenv because the pod's preinstalled FlashAttention binary is ABI-bound
to a different Torch build.

## Verified

- Model loaded all four checkpoint shards.
- LLM, vision encoder, audio encoder, native TTS and vocoder are on CUDA.
- Backend, worker and gateway health checks return 200.
- Gateway registered one healthy worker.
- Browser page rendered with webcam and microphone permissions.
- A real full-duplex session was created and entered `LIVE`; the backend ran
  its first streaming generation unit.
- `Vietnamese Call` loaded its UTF-8 prompt and 16 kHz Vietnamese reference
  audio through the native preset API.
- The backend now honors `DuplexConfig.generate_audio` per session. A verified
  Vietnamese turn emitted zero native audio events and zero native TTS tokens;
  an English control turn re-enabled CosyVoice and emitted native audio.
- MMS-TTS generated a valid Vietnamese WAV for the Vietnamese turn. On 15 warm
  calls covering five short visual-assistance sentences, first playable audio
  was 72.8 ms median / 118.3 ms P95 and RTF was 0.022 median.
- VieNeu-TTS v3 Turbo was also tested with its official ONNX INT8 CPU path. At
  eight threads it reached 905.6 ms median first chunk but RTF 3.27, so this
  measured configuration is not used in the live path. Thirty-two threads was
  slower (RTF 3.79).
- VieNeu's PyTorch/H100 path reached 127.5 ms median first chunk, 162.6 ms P95,
  and RTF 0.385 median across 15 warm calls. Its streaming adapter is deployed:
  the same-origin gateway smoke returned six PCM chunks with 163.9 ms first
  chunk, and the browser schedules chunks directly with an `AudioContext`.
- The post-deploy browser soak passed every gate across five Vietnamese turns:
  five streaming speech starts/ends, zero native audio, mic suppression during
  playback, no self-triggered extra turn, a fresh 640x480 camera stream, and a
  final return to Live with no page/request error.
- The dedicated `/vi` E2E passed three consecutive real protocol turns in 36.7
  seconds fixture wall time: three unique immutable ASR finals, three exactly-once
  chat inputs, three distinct JPEG hashes, three assistant answers and three
  VieNeu HTTP 200 streams. It returned to Listening, then Stop ended the session;
  camera time advanced from 0.11 s to 34.90 s with no request/page/console error.
- A separate exact-service protocol soak passed 10/10 turns with zero duplicate
  final/input/frame or protocol error. P95 was 496.1 ms from speech end to ASR
  final, 15.3 ms frame age, 536.7 ms MiniCPM TTFT, and 4.20 s from speech end to
  first audio. This is the recorded full-answer-TTS baseline. `/vi` now queues
  each complete MiniCPM sentence into VieNeu before `response.done`; FIFO,
  fallback, barge-in and Stop acceptance passed, with `?early_tts=0` retained
  as an instant rollback. Post-deploy latency is reported separately rather
  than overwriting this baseline.
- The post-deploy early-TTS soak passed 10/10 turns with no duplicate/error.
  Speech-end to first decodable audio was 2.25 s median / 3.11 s P95, improving
  the full-answer baseline P95 by 26%. This excludes physical speaker latency
  and still misses the provisional two-second P95 target.
- The native gateway's supported `mode=chat` contract was exercised for five
  external-transcript turns with a fresh JPEG and rolling text history: 5/5
  completed, frame age P95 3.1 ms, input-to-first-text P95 496.7 ms, and
  input-to-done P95 1703.5 ms.
- The persistent PhoWhisper-medium service on the second H100 combines Silero
  VAD with immutable `asr.final` IDs. Its persistent two-turn WebSocket test
  passed; a 50-utterance FLEURS Vietnamese subset measured 8.97% normalized WER,
  6.06% CER, 979 ms inference P95 and RTF 0.053.
- PhoWhisper-large was evaluated on the exact same 50 IDs. It improved WER only
  to 8.62% while P95 rose to 1.147 s and model memory increased by about 1.68
  GiB, so medium remains the realtime default.

Evidence:

- `results/native_minicpmo_browser_report.json`
- `results/native_minicpmo_omni.png`
- `results/native_vietnamese_call.png`
- `results/native_vietnamese_reply.wav`
- `results/mms_cuda/benchmark.json`
- `results/vieneu_cpu_int8/benchmark.json`
- `results/vieneu_gpu_fp32/benchmark.json`
- `results/vieneu_gpu_fp32_r3/benchmark.json`
- `results/native_external_asr_5turn_report.json`
- `results/native_vi_soak/report.json`
- `results/native_vi_soak/native_vietnamese_soak.png`
- `results/vi_profile_e2e/report.json`
- `results/vi_profile_e2e/vietnamese_profile_e2e.png`
- `artifacts/vietnamese-protocol-soak-10turn.json`
- `results/vieneu_gateway_stream_smoke.ndjson`
- `artifacts/phowhisper-asr-service-pod-b.json`
- `artifacts/phowhisper-medium-fleurs-vi-test50.json`
- `artifacts/phowhisper-medium-vs-large-fleurs-vi-test50.json`
- `scripts/verify_native_vietnamese_duplex.py`

## Honest limitations

- MiniCPM-o 4.5 officially advertises native speech conversation in English
  and Chinese. Its Vietnamese recognition/text quality is not an upstream-
  guaranteed capability. The verified PhoWhisper service must still be wired
  into the final browser profile; the present `/omni` page hears through the
  native audio path.
- VieNeu PCM streaming is functional, while MMS fallback still returns a
  complete WAV. End-to-end ASR -> chat -> streaming-TTS latency and barge-in
  remain acceptance gates for the dedicated Vietnamese browser profile.
- OpenGlass's current public panel, its pinned dependency lock, and the latest
  MiniCPM gateway describe different runtime generations. The clean panel is
  therefore not used to supervise this verified browser baseline.
- The OpenGlass ESP32 bridge remains the next layer. It is not needed for this
  webcam baseline and its tracked firmware/audio endpoint currently differs
  from the tracked host bridge protocol.
