# Edge AI v2 — Vietnamese assistive conversation

This directory is a source snapshot of the connected QCS8550 AIBOX deployment,
captured before switching the VLM profile to Qwen3.5 2B.

## End-to-end flow

```text
QNN camera/depth on Hexagon HTP
        ↓
scene + fresh JPEG
        ↓
Whisper.cpp Vietnamese STT
        ↓
GenieX OpenAI-compatible server → local/Qwen3.5-2B-GGUF:Q4_0
        ↓
VieNeu ONNX INT8 Vietnamese TTS → ALSA speaker
```

The deterministic detector/risk lane remains independent of the VLM. VLM
requests are single-flight and admitted only when the frame is fresh, detector
FPS is healthy, and NPU temperature is below the configured guardrail.

The deployed coordinator uses two concurrent lanes:

```text
/dev/video2 -> QNN YOLO + depth -> tracked SceneSnapshot -> risk/audio alerts
                         |                    |
                         |                    +-> compact deduplicated context
                         +-> latest raw JPEG -------> Qwen3.5 2B VL (on demand)
Whisper final transcript --------------------------> short conversation history
                                                     |
                                                     +-> streamed text -> VieNeu -> ALSA
```

Qwen receives a 512 px JPEG plus at most eight stable, deduplicated detections
with label, confidence, zone, and normalized bounding box. Uncalibrated depth
values are never exposed as metres. The last two text turns are retained only
to resolve follow-up wording; old visual facts are never reused.

VieNeu speaks the exact semantic response produced by Qwen: the first streamed
phrase is queued immediately, then the remaining text is synthesized in order
by one serial TTS worker. A speech-only normalization layer expands English
object labels and units (for example `laptop`, `FPS`, `NPU`, and `°C`) into
Vietnamese pronunciations without replacing the answer. Rendered phrases are
cached by voice, tempo, and normalized text. Common fallback phrases are
pre-rendered at startup, while P0/P1 safety WAV assets retain playback priority.

The web demo exposes `POST /tts/speak` and a **Phát TTS** button for direct
speaker verification. On the connected box, a persisted cache hit reached the
ALSA queue in 139 ms; a new dynamic sentence reached it in 3.77 s before being
cached for later turns.

For development from a PC, the web UI enables **loa máy tính** by default. It
posts the exact `answer_vi` to `POST /tts/stream`; the box decodes short VieNeu
phrases and sends 48 kHz PCM16 NDJSON to the browser's Web Audio API. Phrase
streaming is the QCS8550 mode that passed Vietnamese Whisper loopback; native
ONNX frame streaming remains unqualified. This keeps the selected voice even when Windows
has no Vietnamese system voice. Use **Test loa máy tính** once after opening
the page to unlock browser audio. Web Speech is only a last-resort fallback if
the neural stream fails; VieNeu/ALSA remains available directly on the box.

On the connected QCS8550 box, the validated warm turn reached first VLM text
in 4.12 s and completed VLM decoding in 8.31 s while QNN perception remained at
23.7 FPS. Total request time was 12.88 s including a 4.1 s thermal admission
wait. Cold model loading remains substantially slower, so this profile is
interactive/on-demand rather than a per-frame VLM.

The `device/` tree contains only source/config/scripts pulled from the box;
model weights, recordings, generated audio, logs, virtual environments, and
credentials are intentionally excluded.

Deploy the Qwen3.5 profile with:

```powershell
.\scripts\deploy_edge_ai_v2.ps1 -Serial 17513b4
```

Then verify:

```powershell
adb -s 17513b4 shell curl -sS http://127.0.0.1:8090/health
adb -s 17513b4 shell curl -sS -X POST http://127.0.0.1:8090/ask `
  -H 'Content-Type: application/json' `
  -d '{"text":"Trước mặt tôi có gì?"}'
```

This is an assistive research prototype, not a certified navigation aid.

Open `http://localhost:8090` after `adb forward tcp:8090 tcp:8090` for the
full box demo. The page shows the QNN camera stream and its hold-to-talk button
plus a dedicated **Speech-to-Text Lab**. Hold the STT Lab button to record the
box microphone and run Whisper/VAD without invoking Qwen or TTS; the page shows
the Vietnamese transcript, captured-audio duration, STT latency, RTF, sample
rate, VAD state, and detector FPS. Full assistant turns include the same STT
measurements in the end-to-end performance panel.

The production STT profile uses multilingual Whisper Base Q8 on three CPU NEON
threads. Install the pinned, checksum-verified model from the host before first
deployment:

```powershell
.\scripts\install_whisper_base_q8.ps1 -Serial 17513b4
```

On the live QCS8550 workload, Base Q8 reduced the 3.44 s loopback case from
14.82 s (Small Q8, RTF 4.31) to 4.76 s (RTF 1.38). Bounding the encoder audio
context to 512 frames reduced an 8.75 s real box-mic recording from 4.53 s to
2.48 s (RTF 0.28) while preserving all three spoken questions and keeping QNN
perception live. This profile is intended for short push-to-talk questions.
Tiny Q8 was faster on the short clip but hallucinated repeated spatial
directions on real microphone audio, so it is not the safety-facing default.

Direct `whisper-stream` capture was also built and exercised on the box. It is
not enabled in the safety profile: with 1.5 s chunks the full Base encoder took
about 2.7 s per chunk and produced Vietnamese hallucinations on ambient/silent
input. The production path therefore records continuously while the button is
held, applies Silero VAD, and returns a bounded final transcript after release.
Moving Whisper to HTP requires a separately exported QNN Whisper context and a
detector-concurrency soak test; the installed `whisper.cpp` model cannot be
routed to HTP merely by changing a runtime flag.

drives box microphone → STT → VLM/router → TTS → box ALSA speaker.
