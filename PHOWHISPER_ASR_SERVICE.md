# PhoWhisper persistent ASR service

This service keeps `vinai/PhoWhisper-medium` loaded persistently. Pod B hosts
the benchmark/reference instance; Pod A hosts the same pinned model for the
same-origin `/vi` production demo. It exposes:

- `GET /health`
- `POST /transcribe` for a WAV body, multipart `file`, or raw mono 16 kHz PCM16LE
- `WS /v1/asr` for endpointed microphone audio

PhoWhisper is not a true incremental decoder. The WebSocket uses the official
Silero VAD ONNX backend to find utterance boundaries, then runs PhoWhisper once
per finalized utterance. Text partials are speculative and carry
`ui_only=true`, `vlm_eligible=false`. Only the immutable `asr.final` event may
be routed to a VLM/agent, deduplicated by `final_id`.

## Deployment

The deployed root is configured with `PHOWHISPER_BENCH_ROOT` and
the service binds only to `127.0.0.1:18783`, so it does not expose a new public
port or replace ports 7870, 7871, 7873, 8780, or 18781.

```bash
bash scripts/bootstrap_phowhisper_asr_pod_b.sh
bash scripts/start_phowhisper_asr_pod_b.sh
curl -s http://127.0.0.1:18783/health
curl -s -H 'Content-Type: audio/wav' --data-binary @sample.wav \
  http://127.0.0.1:18783/transcribe
```

The dedicated `/vi` browser runs another instance on Pod A so the gateway can
proxy audio over localhost. The generic launcher isolates PID/log files by host
or an explicit instance name and avoids collisions on the shared network volume:

```bash
PHOWHISPER_ASR_INSTANCE=pod-a bash scripts/start_phowhisper_asr_h100.sh
curl -s http://127.0.0.1:18783/health
PHOWHISPER_ASR_INSTANCE=pod-a bash scripts/stop_phowhisper_asr_h100.sh
```

The main `start_native_minicpmo_h100.sh`/`stop_native_minicpmo_h100.sh` lifecycle
also owns this Pod A service for clean full-stack restarts.

## WebSocket wire contract

The strongest client contract is a JSON chunk with capture sequence and
timestamp. Timestamps are passed through and interpolated; the server never
subtracts its wall clock from an unrelated client clock.

```json
{
  "type": "audio.chunk",
  "sequence": 42,
  "timestamp_ms": 123456.0,
  "sample_rate": 16000,
  "channels": 1,
  "encoding": "pcm_s16le",
  "pcm_s16le_base64": "..."
}
```

Sequences must increase and timestamps must not decrease. Raw binary PCM is
also accepted for simple clients, but the server then assigns receive-time
sequence/timestamps. Send `{"type":"audio.end"}` to flush active speech.

Server events include `asr.ready`, `asr.speech_start`, `asr.ack`, revisable
`asr.partial`, immutable `asr.final`, `asr.no_speech`, and `asr.error`.
Default endpoint settings are threshold 0.5, 500 ms trailing silence, 150 ms
speech padding, 2 s partial interval, and 20 s maximum utterance. They can be
changed with WebSocket query parameters. The `/vi` profile uses 300 ms speech
padding after E2E testing showed better preservation of Vietnamese command
prefixes.

`endpoint.algorithmic_delay_ms` is derived in the audio sample clock. Client
capture timestamps and server inference timestamps are reported separately to
avoid false latency calculations across unsynchronized clocks.

## Verification

```bash
pytest -q tests/test_phowhisper_asr_service.py
python tests/integration_phowhisper_asr_service.py \
  --wav /path/ref_vi_mms.wav \
  --session-jsonl /path/session/stream.jsonl \
  --output results/phowhisper-asr-service.json
```

The integration test sends two utterances over one persistent WebSocket,
checks unique immutable final IDs and VLM routing, and also benchmarks the HTTP
WAV endpoint.
