#!/usr/bin/env python3
"""VieNeu-TTS v3 Turbo streaming service for OpenGlass.

The wire format is newline-delimited JSON so a browser can schedule PCM chunks
as soon as VieNeu produces them without waiting for a complete WAV container.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import threading
import time
import unicodedata
import wave
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


LOGGER = logging.getLogger("openglass.vieneu_tts")
MODEL_ID = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
STREAM_SCHEMA = "omniglass.vi-tts-stream.v1"


class VieNeuEngine(Protocol):
    sample_rate: int
    backend: str

    def infer_stream(self, text: str, **kwargs: Any) -> Iterator[np.ndarray]: ...


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    voice: str | None = Field(default=None, max_length=128)
    style: str = Field(default="tu_nhien", min_length=1, max_length=64)


def clean_vietnamese(text: str) -> str:
    """Keep readable Unicode while removing markup and control characters."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`#|]", " ", text)
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def float_audio_to_pcm16(chunk: Any) -> tuple[bytes, int]:
    audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b"", 0
    if not np.isfinite(audio).all():
        raise ValueError("VieNeu produced non-finite audio")
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    return pcm.tobytes(), int(pcm.size)


def ndjson(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _engine_info(engine: VieNeuEngine, loaded_seconds: float) -> dict[str, Any]:
    device = str(getattr(engine, "device", "cuda"))
    return {
        "status": "ready",
        "model": MODEL_ID,
        "mode": "v3turbo",
        "backend": str(getattr(engine, "backend", "pytorch")),
        "device": device,
        "sample_rate": int(engine.sample_rate),
        "sample_format": "pcm_s16le",
        "loaded_seconds": round(loaded_seconds, 3),
        "stream_schema": STREAM_SCHEMA,
    }


def create_app(
    engine: VieNeuEngine | None = None,
    *,
    engine_factory: Callable[[], VieNeuEngine] | None = None,
) -> FastAPI:
    load_started = time.perf_counter()
    if engine is None:
        if engine_factory is None:
            raise ValueError("create_app requires an engine or engine_factory")
        engine = engine_factory()
    loaded_seconds = time.perf_counter() - load_started
    sample_rate = int(engine.sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"Invalid VieNeu sample rate: {sample_rate}")

    # The underlying autoregressive engine and its CUDA state are not safe to
    # interleave. Serialize synthesis while health checks remain responsive.
    synthesis_lock = threading.Lock()
    app = FastAPI(title="OpenGlass VieNeu Streaming TTS", version="1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return _engine_info(engine, loaded_seconds)

    def inference_kwargs(request: SpeakRequest) -> dict[str, Any]:
        return {
            "voice": request.voice,
            "style": request.style,
            "apply_watermark": False,
        }

    @app.post("/stream")
    def stream(request: SpeakRequest) -> StreamingResponse:
        spoken = clean_vietnamese(request.text)
        if not spoken:
            raise HTTPException(status_code=422, detail="Không có nội dung để đọc")

        def generate() -> Iterator[bytes]:
            request_started = time.perf_counter()
            lock_started = request_started
            sequence = 0
            total_samples = 0
            first_chunk_ms: float | None = None
            yield ndjson(
                {
                    "type": "meta",
                    "schema": STREAM_SCHEMA,
                    "text": spoken,
                    "model": MODEL_ID,
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "sample_format": "pcm_s16le",
                }
            )
            try:
                with synthesis_lock:
                    inference_started = time.perf_counter()
                    queue_ms = (inference_started - lock_started) * 1000.0
                    for chunk in engine.infer_stream(spoken, **inference_kwargs(request)):
                        pcm, samples = float_audio_to_pcm16(chunk)
                        if not pcm:
                            continue
                        elapsed_ms = (time.perf_counter() - request_started) * 1000.0
                        if first_chunk_ms is None:
                            first_chunk_ms = elapsed_ms
                        total_samples += samples
                        yield ndjson(
                            {
                                "type": "audio",
                                "seq": sequence,
                                "pcm_s16le_base64": base64.b64encode(pcm).decode("ascii"),
                                "samples": samples,
                                "duration_ms": round(samples / sample_rate * 1000.0, 3),
                                "elapsed_ms": round(elapsed_ms, 3),
                            }
                        )
                        sequence += 1
                    inference_ms = (time.perf_counter() - inference_started) * 1000.0

                if total_samples == 0 or first_chunk_ms is None:
                    raise RuntimeError("VieNeu returned no audio")
                total_ms = (time.perf_counter() - request_started) * 1000.0
                audio_seconds = total_samples / sample_rate
                yield ndjson(
                    {
                        "type": "done",
                        "chunks": sequence,
                        "samples": total_samples,
                        "audio_duration_seconds": round(audio_seconds, 6),
                        "queue_ms": round(queue_ms, 3),
                        "first_chunk_ms": round(first_chunk_ms, 3),
                        "inference_ms": round(inference_ms, 3),
                        "total_ms": round(total_ms, 3),
                        "rtf": round(inference_ms / 1000.0 / audio_seconds, 6),
                    }
                )
                LOGGER.info(
                    "stream chars=%s chunks=%s first_chunk_ms=%.1f inference_ms=%.1f audio_s=%.2f",
                    len(spoken),
                    sequence,
                    first_chunk_ms,
                    inference_ms,
                    audio_seconds,
                )
            except Exception as exc:
                LOGGER.exception("VieNeu streaming synthesis failed")
                yield ndjson(
                    {
                        "type": "error",
                        "code": "synthesis_failed",
                        "message": str(exc),
                        "chunks_emitted": sequence,
                    }
                )

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/speak")
    def speak(request: SpeakRequest) -> dict[str, Any]:
        spoken = clean_vietnamese(request.text)
        if not spoken:
            raise HTTPException(status_code=422, detail="Không có nội dung để đọc")
        started = time.perf_counter()
        pcm_parts: list[bytes] = []
        total_samples = 0
        first_chunk_ms: float | None = None
        try:
            with synthesis_lock:
                for chunk in engine.infer_stream(spoken, **inference_kwargs(request)):
                    pcm, samples = float_audio_to_pcm16(chunk)
                    if not pcm:
                        continue
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - started) * 1000.0
                    pcm_parts.append(pcm)
                    total_samples += samples
        except Exception as exc:
            LOGGER.exception("VieNeu full synthesis failed")
            raise HTTPException(status_code=503, detail=f"VieNeu unavailable: {exc}") from exc
        if total_samples == 0 or first_chunk_ms is None:
            raise HTTPException(status_code=500, detail="VieNeu returned no audio")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        pcm = b"".join(pcm_parts)
        duration = total_samples / sample_rate
        return {
            "text": spoken,
            "audio_wav_base64": base64.b64encode(wav_bytes(pcm, sample_rate)).decode("ascii"),
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 6),
            "first_chunk_ms": round(first_chunk_ms, 3),
            "inference_ms": round(elapsed_ms, 3),
            "model": MODEL_ID,
            "backend": str(getattr(engine, "backend", "pytorch")),
        }

    return app


def build_vieneu_engine(args: argparse.Namespace) -> VieNeuEngine:
    from vieneu import Vieneu

    engine = Vieneu(
        mode="v3turbo",
        backbone_repo=args.backbone_repo,
        moss_tokenizer=args.moss_tokenizer,
        device=args.device,
        backend=args.backend,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
    )
    # Some upstream wrappers do not expose device even though health metadata
    # benefits from reporting the selected deployment target.
    if not hasattr(engine, "device"):
        engine.device = args.device
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18782)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="pytorch", choices=("pytorch", "onnx", "auto"))
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument(
        "--backbone-repo",
        default="pnnbao-ump/VieNeu-TTS-v3-Turbo",
        help="Pinned local snapshot directory or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--moss-tokenizer",
        default="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        help="Pinned local MOSS Audio Tokenizer snapshot or repository ID.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = create_app(engine_factory=lambda: build_vieneu_engine(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
