from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
from fastapi.testclient import TestClient

from scripts.native_vieneu_tts_service import STREAM_SCHEMA, create_app


class FakeVieNeu:
    sample_rate = 48_000
    backend = "pytorch"
    device = "cuda"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def infer_stream(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        yield np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        yield np.asarray([0.25, -0.25], dtype=np.float32)


def parse_ndjson(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_health_and_stream_protocol() -> None:
    engine = FakeVieNeu()
    client = TestClient(create_app(engine))

    health = client.get("/health")
    assert health.status_code == 200
    health_body = health.json()
    assert health_body["status"] == "ready"
    assert health_body["backend"] == "pytorch"
    assert health_body["device"] == "cuda"
    assert health_body["sample_rate"] == 48_000
    assert health_body["stream_schema"] == STREAM_SCHEMA

    response = client.post(
        "/stream",
        json={"text": "  <b>Xin chào</b> **bạn**  ", "voice": "Hà", "style": "tu_nhien"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_ndjson(response.text)
    assert [event["type"] for event in events] == ["meta", "audio", "audio", "done"]
    assert events[0] == {
        "type": "meta",
        "schema": STREAM_SCHEMA,
        "text": "Xin chào bạn",
        "model": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
        "sample_rate": 48_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
    }
    assert [event["seq"] for event in events[1:3]] == [0, 1]
    pcm = b"".join(base64.b64decode(event["pcm_s16le_base64"]) for event in events[1:3])
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [
        -32767,
        -16384,
        0,
        16384,
        32767,
        8192,
        -8192,
    ]
    done = events[-1]
    assert done["chunks"] == 2
    assert done["samples"] == 7
    assert done["first_chunk_ms"] >= 0
    assert done["total_ms"] >= done["first_chunk_ms"]
    assert engine.calls == [
        (
            "Xin chào bạn",
            {"voice": "Hà", "style": "tu_nhien", "apply_watermark": False},
        )
    ]


def test_speak_returns_valid_pcm16_wav() -> None:
    client = TestClient(create_app(FakeVieNeu()))
    response = client.post("/speak", json={"text": "Xin chào."})
    assert response.status_code == 200
    payload = response.json()
    with wave.open(io.BytesIO(base64.b64decode(payload["audio_wav_base64"])), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 7
    assert payload["duration_seconds"] == round(7 / 48_000, 6)


def test_stream_reports_generator_failure_as_protocol_error() -> None:
    class BrokenVieNeu(FakeVieNeu):
        def infer_stream(self, text: str, **kwargs):
            yield np.ones(4, dtype=np.float32)
            raise RuntimeError("decoder failed")

    client = TestClient(create_app(BrokenVieNeu()))
    response = client.post("/stream", json={"text": "Xin chào"})
    events = parse_ndjson(response.text)
    assert [event["type"] for event in events] == ["meta", "audio", "error"]
    assert events[-1] == {
        "type": "error",
        "code": "synthesis_failed",
        "message": "decoder failed",
        "chunks_emitted": 1,
    }


def test_blank_cleaned_text_is_rejected() -> None:
    client = TestClient(create_app(FakeVieNeu()))
    response = client.post("/stream", json={"text": "<i></i> ***"})
    assert response.status_code == 422
