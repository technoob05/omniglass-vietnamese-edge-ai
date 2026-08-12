# OmniGlass Vietnamese realtime architecture

## Decision

Keep MiniCPM-o 4.5 as the visual reasoner and conversation model, but do not
fine-tune or depend on its unsupported Vietnamese speech output. The production
candidate is a modular two-H100 pipeline:

1. Browser/glasses microphone -> VAD/endpointer -> PhoWhisper-medium.
2. Immutable Vietnamese transcript + a frame captured after the worker is
   ready -> MiniCPM-o native gateway `mode=chat`.
3. Streaming Vietnamese text -> VieNeu-TTS v3 Turbo -> browser PCM playback.
4. MMS-TTS remains a small, fast fallback if VieNeu or its GPU is unavailable.

The original MiniCPM-o full-duplex English/Chinese path stays available as the
control baseline. A Vietnamese preset must set `generate_audio=false`; the
backend patch makes that setting effective without disabling native audio for
other sessions.

## Why this path

- MiniCPM-o audio/TTS is designed primarily for English and Chinese. There is
  no supported end-to-end audio fine-tuning recipe for the current full-duplex
  stack.
- PhoWhisper provides a Vietnamese-specialized ASR checkpoint while retaining
  the standard Whisper architecture. It is utterance ASR, not a persistent
  streaming recognizer, so VAD and stable-final semantics are mandatory.
- VieNeu offers Vietnamese regional voices and a native chunk generator.
- The gateway already supports text + image + rolling history in `mode=chat`.
  Re-synthesizing a transcript into audio just to re-enter `mode=video` would
  add latency and errors.

## Verified measurements

All numbers below are measurements from this workspace, not vendor claims.

| Component | Configuration | Result |
|---|---|---|
| MiniCPM-o chat | H100 MIG 4g.40GB, five turns | 5/5 complete; frame age P95 3.1 ms; TTFT P95 496.7 ms; done P95 1703.5 ms |
| Native Vietnamese browser | animated 640x480 fake webcam + real-time fake mic, five turns | 5/5 text/listen and VieNeu stream start/end; native audio 0; mic suppressed during playback; returned to Live |
| Dedicated `/vi` pipeline | PhoWhisper -> fresh JPEG -> MiniCPM chat -> VieNeu, three turns | 3 unique immutable finals; 3 exactly-once inputs and distinct frame hashes; 3 answers/TTS 200; returned to Listening; Stop final |
| Dedicated `/vi` protocol soak | exact H100 services, 10 sequential turns | 10/10; no duplicate/error; frame age P95 15.3 ms; ASR final P95 496.1 ms; VLM TTFT P95 536.7 ms; post-speech first audio P95 4.20 s |
| Production early-TTS soak | exact deployed `/vi` flow, 10 sequential turns | 10/10; no duplicate/error; speech end to first decodable audio P50/P95 2.25/3.11 s, 26% lower P95 than the 4.20 s baseline |
| MMS-TTS | H100, 15 warm calls | first playable WAV median 72.8 ms, P95 118.3 ms; RTF 0.022 |
| VieNeu v3 | H100 PyTorch, 15 warm calls | first chunk median 127.5 ms, P95 162.6 ms; RTF 0.385 |
| VieNeu v3 | CPU ONNX INT8, 8 threads | first chunk median 905.6 ms; RTF 3.27; no-go for live |
| PhoWhisper-medium | H100 MIG 2g.20GB, FP16; FLEURS `vi_vn/test`, 50 utterances | normalized WER 8.97%, CER 6.06%; inference P95 979 ms; RTF 0.053; peak 1.60 GiB |
| PhoWhisper-large | same 50 utterances, FP16 | WER 8.62%, CER 5.31%; P95 1147 ms; RTF 0.069; model VRAM +1.68 GiB vs medium |

The native browser soak used an animated synthetic camera and therefore verifies
transport/state/audio lifecycle, frame freshness and echo suppression, not visual
accuracy. The external-ASR chat soak used a real saved image and verified rolling
context. FLEURS contains read speech and no regional-accent labels, so its result
does not replace a consented glasses-microphone evaluation.

Large removes only five additional word errors out of 1,416 while raising
median latency by 30.8%, P95 by 17.1% and model memory by about 1.68 GiB. Both
miss the provisional 8% WER gate, so medium remains the realtime default; the
next quality work is endpoint/normalization and a representative microphone
command set, not paying the large-model cost.

The 10-turn baseline soak meets the current ASR, frame-freshness and VLM-TTFT
gates but does not meet the provisional two-second end-of-speech-to-first-audio
gate. The deployed `/vi` profile now starts VieNeu as soon as MiniCPM completes
the first sentence, preserves FIFO playback, and supports immediate rollback at
`/vi?early_tts=0`. Its post-deploy latency is measured separately so the older
4.20-second full-answer baseline is not misreported as the early-TTS result.
The production 10-turn soak measured 3.11 seconds P95 to first decodable audio;
this is not yet the two-second goal and does not include physical speaker delay.

## Realtime contracts

### ASR

Client audio frames:

```json
{
  "type": "audio.append",
  "session_id": "...",
  "seq": 42,
  "capture_ms": 123456.7,
  "pcm_f32_b64": "..."
}
```

Only an immutable final may trigger MiniCPM-o:

```json
{
  "type": "asr.final",
  "utterance_id": "utt_...",
  "final_id": "final_...",
  "text": "Mô tả vật trước mặt tôi.",
  "speech_start_ms": 1000,
  "speech_end_ms": 2700,
  "endpoint_ms": 3100
}
```

Partial hypotheses are display-only. `final_id` is the idempotency key.

### MiniCPM turn

For every final transcript:

1. Connect to `/v1/realtime?mode=chat`.
2. Wait for `session.queue_done`.
3. Send `session.init` and wait for `session.created`.
4. Capture the next decoded webcam frame.
5. Send an `input.append` message containing rolling text history, the current
   JPEG, and the final transcript.
6. Stream `response.output.delta/text` to the UI and flush complete sentences
   into the ordered VieNeu queue. Flush any trailing fragment at `response.done`.
7. Close the chat turn after `response.done`; never replay text already queued.

Keep the last five text turns and omit old images. Retry a pre-input worker
HTTP 403 once after 300–500 ms because the gateway can release a pool slot just
before the worker finishes returning to `idle`. Never retry after first text.

### TTS

VieNeu streams 48 kHz mono chunks. The browser owns one `AudioContext`, converts
PCM16 chunks into `AudioBuffer`s, and schedules them contiguously. Every chunk
and callback carries `turn_id` and an epoch. MMS returns a complete 16 kHz WAV
and is the fallback path.

## State machine and cancellation

```text
IDLE -> LISTENING -> FINALIZING -> WAITING_WORKER
     -> CAPTURING_FRAME -> THINKING -> SPEAKING -> LISTENING
```

- Every asynchronous continuation checks the active `epoch` and `turn_id`.
- Stop increments the epoch, clears timers, aborts ASR/fetch/WebSocket, stops
  browser audio, and prevents an old `finally` from re-entering LISTENING.
- Speech start during THINKING closes the current chat turn; speech start during
  SPEAKING stops scheduled audio. The new utterance receives a new epoch.
- State-changing agent tools execute only from a unique ASR final.

## Acceptance gates

These are engineering targets, not achieved claims:

- Five and then 100 consecutive turns, exactly one action per `final_id`.
- ASR endpoint-to-final P95 <= 1.0 s.
- Current-frame age P95 <= 250 ms.
- MiniCPM input-to-first-text P95 <= 1.0 s.
- End-of-speech to first audible response P95 <= 2.0 s.
- Barge-in stops stale speech/output P95 <= 300 ms.
- Returns to LISTENING after every success, timeout, playback rejection, and
  network error.
- Vietnamese ASR: clean read WER <= 8%, spontaneous accent WER <= 15%, intent
  exact >= 95%, names/numbers exact >= 90% on a consented evaluation set.
- Thirty-minute soak with no growing queue, leaked session, thermal failure, or
  raw audio/image retention by default.

## Data and fine-tuning decision

Build the evaluation set before training:

- consented North/Central/South speakers;
- assistive commands, scene questions, names, numbers, distances, and
  Vietnamese-English code-switching;
- quiet, room, street, and café conditions at multiple microphone distances;
- clean augmentation from Common Voice Vietnamese, MUSAN noise, and OpenSLR
  room impulse responses where their licenses permit the intended use.

Fine-tune PhoWhisper only if it fails the measured WER/intent gates. Current
ms-swift now recognizes MiniCPM-o 4.5 image/video/audio data and LoRA, so an
LLM-side Vietnamese thinker adapter is a valid isolated experiment. It is not
evidence of a supported native Talker/full-duplex TTS recipe, nor of adapter
compatibility with the deployed GGUF/C++ streaming runtime. Keep that adapter
off the demo critical path until held-out quality, English-control, latency and
runtime-parity gates pass. VieNeu voice adaptation is optional after latency
and intelligibility pass with preset voices. See
`VIETNAMESE_FINETUNING_PLAN.md` for the experiment and dataset contract.

## Edge AI migration

1. QCS8550 first, QCS6490 second. Inventory exact BSP, QAIRT/QNN version,
   camera/audio path, thermal envelope, and secure-boot state.
2. Run VAD/wake locally. Start Vietnamese ASR with sherpa-onnx Zipformer INT8
   on CPU ARM64; separately test PhoWhisper tiny/base/small through Qualcomm AI
   Hub/QNN. Do not infer PhoWhisper performance from stock OpenAI Whisper jobs.
3. Keep MiniCPM-o/VLM on H100 initially; upload only a stable transcript and a
   selected frame/ROI. Local detector/tracker/depth handles time-critical cues.
4. Keep short prerecorded safety phrases locally. Evaluate a Vietnamese Piper
   or platform TTS fallback; VieNeu does not currently have a turnkey Android or
   QNN path.
5. For every artifact, lock source revision, model revision, license, input
   preprocessing, precision, QAIRT version, target SoC, calibration hash, and
   artifact SHA-256. Validate task metrics on the physical board before making
   realtime, offline, privacy, distance, or safety claims.

This remains an assistive research prototype and must not be presented as a
certified navigation aid or a replacement for a cane, guide dog, or trained
orientation and mobility support.
