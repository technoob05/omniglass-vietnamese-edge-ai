from __future__ import annotations

import importlib.util
import io
import sys
import wave
import asyncio
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "phowhisper_asr_service.py"
SPEC = importlib.util.spec_from_file_location("phowhisper_asr_service", SCRIPT)
assert SPEC and SPEC.loader
service = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = service
SPEC.loader.exec_module(service)


class FakeWebSocket:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def send_json(self, event: dict) -> None:
        self.events.append(event)


class FakeEngine:
    async def transcribe(self, audio: np.ndarray, language: str = "vi") -> dict:
        duration_ms = len(audio) * 1000 / service.SAMPLE_RATE
        return {"text": "xin chào", "duration_ms": duration_ms, "inference_ms": 12.5, "rtf": 0.01}


class FakeVad:
    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0
        self.current_sample = 0

    def __call__(self, window, return_seconds=False):
        self.calls += 1
        self.current_sample += len(window)
        if self.calls == 1:
            return {"start": 0}
        if self.calls == 8:
            return {"end": 4096}
        return None

    def reset_states(self) -> None:
        self.calls = 0
        self.current_sample = 0


def test_decode_wav_and_raw_pcm() -> None:
    pcm = (np.sin(np.linspace(0, 20, 1600)) * 10_000).astype("<i2").tobytes()
    raw = service.decode_audio_payload(pcm, "application/octet-stream", 16000, 1)
    assert raw.dtype == np.float32
    assert len(raw) == 1600

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    decoded = service.decode_audio_payload(buffer.getvalue(), "audio/wav", 16000, 1)
    assert len(decoded) == 1600

    with pytest.raises(ValueError, match="16 kHz"):
        service.decode_audio_payload(pcm, "application/octet-stream", 8000, 1)


def test_final_is_immutable_and_only_final_is_vlm_eligible(monkeypatch) -> None:
    monkeypatch.setattr(service, "create_silero_vad_iterator", lambda **kwargs: FakeVad())
    session = service.StreamSession(
        FakeEngine(),
        vad_threshold=0.5,
        min_silence_ms=500,
        speech_pad_ms=150,
        partial_interval_ms=5000,
        max_utterance_ms=20000,
    )
    websocket = FakeWebSocket()
    pcm = (np.ones(4096, dtype="<i2") * 1000).tobytes()
    asyncio.run(session.add_pcm(websocket, pcm, sequence=7, timestamp_ms=1234.0))

    final = next(event for event in websocket.events if event["type"] == "asr.final")
    assert final["immutable"] is True
    assert final["routing"]["vlm_eligible"] is True
    assert final["sequence"] == {"first": 7, "last": 7}
    assert final["final_id"]
    assert len([event for event in websocket.events if event["type"] == "asr.final"]) == 1

    with pytest.raises(ValueError, match="sequence must increase"):
        asyncio.run(session.add_pcm(websocket, pcm, sequence=7, timestamp_ms=1300.0))
