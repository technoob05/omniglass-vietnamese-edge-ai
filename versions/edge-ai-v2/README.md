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
drives box microphone → STT → VLM/router → TTS → box ALSA speaker.
