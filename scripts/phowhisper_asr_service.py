#!/usr/bin/env python3
"""Persistent PhoWhisper ASR with HTTP and endpointed WebSocket audio.

PhoWhisper is an utterance model, not a stateful streaming decoder.  The
WebSocket endpoint therefore uses Silero VAD to cut an incoming PCM stream
into immutable utterances.  Partial transcripts are explicitly UI-only;
only ``asr.final`` events are eligible for downstream VLM/agent routing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


LOGGER = logging.getLogger("omniglass.phowhisper_asr")
MODEL_ID = "vinai/PhoWhisper-large"
MODEL_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"
ENGLISH_MODEL_ID = "openai/whisper-large-v3-turbo"
ENGLISH_MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SAMPLE_RATE = 16_000
PCM_WIDTH_BYTES = 2
SILERO_WINDOW_SAMPLES = 512
MAX_HTTP_SECONDS = 60.0
MAX_WS_CHUNK_SECONDS = 1.0


def now_ms() -> float:
    return time.time_ns() / 1_000_000


def pcm16_to_float32(data: bytes) -> np.ndarray:
    if not data or len(data) % PCM_WIDTH_BYTES:
        raise ValueError("PCM must contain a non-empty, even number of bytes")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def decode_audio_payload(data: bytes, content_type: str, sample_rate: int, channels: int) -> np.ndarray:
    if not data:
        raise ValueError("Empty audio payload")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"audio/wav", "audio/x-wav", "audio/wave"} or data[:4] == b"RIFF":
        audio, source_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if source_rate != SAMPLE_RATE:
            divisor = np.gcd(int(source_rate), SAMPLE_RATE)
            audio = scipy.signal.resample_poly(
                audio,
                SAMPLE_RATE // divisor,
                int(source_rate) // divisor,
            ).astype(np.float32)
        return np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE or channels != 1:
        raise ValueError("Raw PCM must be 16 kHz, mono, signed 16-bit little-endian")
    return pcm16_to_float32(data)


def create_silero_vad_iterator(
    *,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
):
    """Load the optional production VAD only when a WebSocket is opened."""
    try:
        from silero_vad import VADIterator, load_silero_vad
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WebSocket ASR requires silero-vad==6.2.1; run the ASR bootstrap script"
        ) from exc
    return VADIterator(
        load_silero_vad(onnx=True),
        threshold=threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )


class PhoWhisperEngine:
    """One persistent GPU model; serialization prevents concurrent OOM spikes."""

    def __init__(self, model_dir: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self._decode_lock = threading.Lock()

    def _transcribe_sync(self, audio: np.ndarray, language: str = "vi") -> dict[str, Any]:
        duration = len(audio) / SAMPLE_RATE
        if not 0 < duration <= MAX_HTTP_SECONDS:
            raise ValueError(f"Audio duration must be within (0, {MAX_HTTP_SECONDS}] seconds")
        with self._decode_lock:
            inputs = self.processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(device="cuda", dtype=torch.float16)
            attention_mask = inputs.attention_mask.to(device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    language=language,
                    task="transcribe",
                    max_new_tokens=224,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {
            "text": text,
            "duration_ms": round(duration * 1000, 2),
            "inference_ms": round(elapsed * 1000, 2),
            "rtf": round(elapsed / duration, 4),
        }

    async def transcribe(self, audio: np.ndarray, language: str = "vi") -> dict[str, Any]:
        return await asyncio.to_thread(self._transcribe_sync, audio, language)


@dataclass
class ChunkMeta:
    start_sample: int
    end_sample: int
    sequence: int
    timestamp_ms: float


class StreamSession:
    def __init__(
        self,
        engine: PhoWhisperEngine,
        *,
        vad_threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
        partial_interval_ms: int,
        max_utterance_ms: int,
        language: str = "vi",
    ) -> None:
        self.engine = engine
        self.vad = create_silero_vad_iterator(
            threshold=vad_threshold,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.vad_threshold = vad_threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.partial_interval_samples = int(partial_interval_ms * SAMPLE_RATE / 1000)
        self.max_utterance_samples = int(max_utterance_ms * SAMPLE_RATE / 1000)
        self.language = language
        self.raw = bytearray()
        self.raw_base_sample = 0
        self.received_samples = 0
        self.processed_samples = 0
        self.vad_pending = np.empty(0, dtype=np.float32)
        self.chunk_meta: list[ChunkMeta] = []
        self.last_sequence = -1
        self.last_timestamp_ms = -1.0
        self.speech_start_sample: int | None = None
        self.partial_revision = 0
        self.next_partial_sample = 0

    def _reset_vad_at_current_position(self) -> None:
        self.vad.reset_states()
        self.vad.current_sample = self.processed_samples

    def _trim_before(self, absolute_sample: int) -> None:
        count = max(0, min(absolute_sample - self.raw_base_sample, len(self.raw) // 2))
        if count:
            del self.raw[: count * PCM_WIDTH_BYTES]
            self.raw_base_sample += count
        self.chunk_meta = [meta for meta in self.chunk_meta if meta.end_sample > self.raw_base_sample]

    def _audio_slice(self, start_sample: int, end_sample: int) -> np.ndarray:
        start = max(0, start_sample - self.raw_base_sample) * PCM_WIDTH_BYTES
        end = max(start, end_sample - self.raw_base_sample) * PCM_WIDTH_BYTES
        return pcm16_to_float32(bytes(self.raw[start:end]))

    def _meta_at(self, sample: int) -> ChunkMeta | None:
        for meta in self.chunk_meta:
            if meta.start_sample <= sample < meta.end_sample:
                return meta
        return self.chunk_meta[-1] if self.chunk_meta else None

    def _client_timestamp(self, sample: int) -> float | None:
        meta = self._meta_at(sample)
        if meta is None:
            return None
        return round(meta.timestamp_ms + (sample - meta.start_sample) * 1000 / SAMPLE_RATE, 3)

    def _sequence_range(self, start_sample: int, end_sample: int) -> tuple[int | None, int | None]:
        selected = [
            meta.sequence
            for meta in self.chunk_meta
            if meta.end_sample > start_sample and meta.start_sample < end_sample
        ]
        return (min(selected), max(selected)) if selected else (None, None)

    async def _partial(self, websocket: WebSocket) -> None:
        if self.speech_start_sample is None:
            return
        end_sample = self.processed_samples
        audio = self._audio_slice(self.speech_start_sample, end_sample)
        if len(audio) < int(0.8 * SAMPLE_RATE):
            return
        result = await self.engine.transcribe(audio, self.language)
        self.partial_revision += 1
        await websocket.send_json({
            "type": "asr.partial",
            "revision": self.partial_revision,
            "text": result["text"],
            "stable": False,
            "ui_only": True,
            "vlm_eligible": False,
            "audio_end_timestamp_ms": self._client_timestamp(end_sample - 1),
            "inference_ms": result["inference_ms"],
        })

    async def _finalize(self, websocket: WebSocket, end_sample: int, reason: str) -> None:
        start_sample = self.speech_start_sample
        if start_sample is None or end_sample <= start_sample:
            return
        audio = self._audio_slice(start_sample, end_sample)
        if len(audio) < int(0.2 * SAMPLE_RATE):
            self.speech_start_sample = None
            self._trim_before(end_sample)
            return
        first_sequence, last_sequence = self._sequence_range(start_sample, end_sample)
        detected_sample = self.processed_samples
        detected_at_ms = now_ms()
        inference_started_ms = now_ms()
        result = await self.engine.transcribe(audio, self.language)
        inference_completed_ms = now_ms()
        final_id = str(uuid.uuid4())
        event = {
            "type": "asr.final",
            "final_id": final_id,
            "immutable": True,
            "text": result["text"],
            "routing": {"vlm_eligible": True, "consume_event": "asr.final"},
            "sequence": {"first": first_sequence, "last": last_sequence},
            "audio": {
                "sample_rate": SAMPLE_RATE,
                "start_timestamp_ms": self._client_timestamp(start_sample),
                "end_timestamp_ms": self._client_timestamp(max(start_sample, end_sample - 1)),
                "duration_ms": result["duration_ms"],
            },
            "endpoint": {
                "reason": reason,
                "vad": "silero-vad",
                "vad_version": "6.2.1",
                "threshold": self.vad_threshold,
                "min_silence_ms": self.min_silence_ms,
                "speech_pad_ms": self.speech_pad_ms,
                "detected_at_server_ms": round(detected_at_ms, 3),
                "algorithmic_delay_ms": round(max(0, detected_sample - end_sample) * 1000 / SAMPLE_RATE, 2),
            },
            "inference": {
                "started_at_server_ms": round(inference_started_ms, 3),
                "completed_at_server_ms": round(inference_completed_ms, 3),
                "latency_ms": result["inference_ms"],
                "rtf": result["rtf"],
            },
        }
        await websocket.send_json(event)
        LOGGER.info(
            "final id=%s reason=%s duration_ms=%.1f inference_ms=%.1f seq=%s-%s text=%r",
            final_id,
            reason,
            result["duration_ms"],
            result["inference_ms"],
            first_sequence,
            last_sequence,
            result["text"],
        )
        self.speech_start_sample = None
        self.partial_revision = 0
        self.next_partial_sample = 0
        self._trim_before(end_sample)

    async def add_pcm(
        self,
        websocket: WebSocket,
        pcm: bytes,
        sequence: int,
        timestamp_ms: float,
    ) -> None:
        if sequence <= self.last_sequence:
            raise ValueError(f"sequence must increase monotonically; got {sequence} after {self.last_sequence}")
        if timestamp_ms < self.last_timestamp_ms:
            raise ValueError("timestamp_ms must be monotonically non-decreasing")
        audio = pcm16_to_float32(pcm)
        if len(audio) > int(MAX_WS_CHUNK_SECONDS * SAMPLE_RATE):
            raise ValueError(f"chunk exceeds {MAX_WS_CHUNK_SECONDS:.1f} second limit")
        start_sample = self.received_samples
        end_sample = start_sample + len(audio)
        self.raw.extend(pcm)
        self.chunk_meta.append(ChunkMeta(start_sample, end_sample, sequence, timestamp_ms))
        self.received_samples = end_sample
        self.last_sequence = sequence
        self.last_timestamp_ms = timestamp_ms
        self.vad_pending = np.concatenate((self.vad_pending, audio))

        while len(self.vad_pending) >= SILERO_WINDOW_SAMPLES:
            window = self.vad_pending[:SILERO_WINDOW_SAMPLES]
            self.vad_pending = self.vad_pending[SILERO_WINDOW_SAMPLES:]
            event = self.vad(torch.from_numpy(window), return_seconds=False)
            self.processed_samples += SILERO_WINDOW_SAMPLES
            if event and "start" in event:
                self.speech_start_sample = int(event["start"])
                self.next_partial_sample = self.speech_start_sample + self.partial_interval_samples
                await websocket.send_json({
                    "type": "asr.speech_start",
                    "sequence": sequence,
                    "audio_start_timestamp_ms": self._client_timestamp(self.speech_start_sample),
                })
            if event and "end" in event:
                await self._finalize(websocket, int(event["end"]), "silence")
            elif (
                self.speech_start_sample is not None
                and self.processed_samples - self.speech_start_sample >= self.max_utterance_samples
            ):
                await self._finalize(websocket, self.processed_samples, "max_utterance")
                self._reset_vad_at_current_position()
            elif (
                self.speech_start_sample is not None
                and self.processed_samples >= self.next_partial_sample
            ):
                await self._partial(websocket)
                self.next_partial_sample = self.processed_samples + self.partial_interval_samples

        if self.speech_start_sample is None and len(self.raw) > 2 * SAMPLE_RATE * PCM_WIDTH_BYTES:
            self._trim_before(self.received_samples - 2 * SAMPLE_RATE)

        await websocket.send_json({
            "type": "asr.ack",
            "sequence": sequence,
            "timestamp_ms": timestamp_ms,
            "received_samples": len(audio),
        })

    async def flush(self, websocket: WebSocket) -> None:
        if self.speech_start_sample is not None:
            await self._finalize(websocket, self.received_samples, "client_flush")
            self._reset_vad_at_current_position()
        else:
            await websocket.send_json({"type": "asr.no_speech", "reason": "client_flush"})


def create_app(
    model_dir: str,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    english_model_dir: str | None = None,
) -> FastAPI:
    engine = PhoWhisperEngine(model_dir)
    english_engine = PhoWhisperEngine(english_model_dir) if english_model_dir else engine
    app = FastAPI(title="OmniGlass PhoWhisper ASR", version="1.0.0")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "ok": True,
            "model": model_id,
            "revision": model_revision,
            "model_dir": model_dir,
            "load_seconds": round(engine.load_seconds, 3),
            "gpu": torch.cuda.get_device_name(),
            "gpu_free_gib": round(free_bytes / 2**30, 3),
            "gpu_total_gib": round(total_bytes / 2**30, 3),
            "vad": "silero-vad",
            "vad_version": "6.2.1",
            "streaming_semantics": "VAD-endpointed utterance ASR; not a stateful streaming decoder",
            "routing_contract": "Only immutable asr.final events may be sent to the VLM",
            "languages": {"vi": model_id, "en": ENGLISH_MODEL_ID if english_model_dir else model_id},
        }

    @app.post("/transcribe")
    async def transcribe(
        request: Request,
        sample_rate: int = Query(SAMPLE_RATE),
        channels: int = Query(1),
        language: str = Query("vi", pattern="^(vi|en)$"),
    ) -> dict[str, Any]:
        content_type = request.headers.get("content-type", "application/octet-stream")
        if content_type.lower().startswith("multipart/form-data"):
            form = await request.form()
            uploaded = form.get("file")
            if uploaded is None or not hasattr(uploaded, "read"):
                raise HTTPException(status_code=422, detail="multipart field 'file' is required")
            data = await uploaded.read()
            media_type = getattr(uploaded, "content_type", None) or "application/octet-stream"
        else:
            data = await request.body()
            media_type = content_type
        try:
            audio = decode_audio_payload(data, media_type, sample_rate, channels)
            result = await (english_engine if language == "en" else engine).transcribe(audio, language)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        final_id = str(uuid.uuid4())
        return {
            "type": "asr.final",
            "final_id": final_id,
            "immutable": True,
            "text": result["text"],
            "routing": {"vlm_eligible": True, "consume_event": "asr.final"},
            "audio": {"sample_rate": SAMPLE_RATE, "duration_ms": result["duration_ms"]},
            "endpoint": {"reason": "http_complete", "vad": None},
            "inference": {"latency_ms": result["inference_ms"], "rtf": result["rtf"]},
        }

    @app.websocket("/v1/asr")
    async def websocket_asr(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            vad_threshold = min(0.9, max(0.1, float(websocket.query_params.get("vad_threshold", "0.5"))))
            min_silence_ms = min(2000, max(100, int(websocket.query_params.get("min_silence_ms", "500"))))
            speech_pad_ms = min(500, max(0, int(websocket.query_params.get("speech_pad_ms", "150"))))
            partial_interval_ms = min(5000, max(800, int(websocket.query_params.get("partial_interval_ms", "2000"))))
            max_utterance_ms = min(60000, max(3000, int(websocket.query_params.get("max_utterance_ms", "20000"))))
            language = websocket.query_params.get("language", "vi").casefold()
            if language not in {"vi", "en"}:
                raise ValueError("language must be vi or en")
        except ValueError:
            await websocket.send_json({"type": "asr.error", "code": "invalid_query_parameters"})
            await websocket.close(code=1008)
            return

        session = StreamSession(
            english_engine if language == "en" else engine,
            vad_threshold=vad_threshold,
            min_silence_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            partial_interval_ms=partial_interval_ms,
            max_utterance_ms=max_utterance_ms,
            language=language,
        )
        await websocket.send_json({
            "type": "asr.ready",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "encoding": "pcm_s16le",
            "vad": "silero-vad",
            "partial_contract": "revisable UI-only; never route to VLM",
            "final_contract": "immutable; route to VLM exactly once by final_id",
        })

        binary_sequence = 0
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                try:
                    if message.get("bytes") is not None:
                        pcm = message["bytes"]
                        sequence = max(binary_sequence, session.last_sequence + 1)
                        binary_sequence = sequence + 1
                        timestamp_ms = now_ms()
                    else:
                        payload = message.get("text")
                        if payload is None:
                            continue
                        import json

                        event = json.loads(payload)
                        event_type = event.get("type")
                        if event_type == "audio.end":
                            await session.flush(websocket)
                            continue
                        if event_type == "ping":
                            await websocket.send_json({"type": "pong", "timestamp_ms": event.get("timestamp_ms")})
                            continue
                        if event_type != "audio.chunk":
                            raise ValueError("Expected audio.chunk, audio.end, ping, or binary PCM")
                        if event.get("sample_rate", SAMPLE_RATE) != SAMPLE_RATE or event.get("channels", 1) != 1:
                            raise ValueError("WebSocket audio must be 16 kHz mono")
                        if event.get("encoding", "pcm_s16le") != "pcm_s16le":
                            raise ValueError("WebSocket encoding must be pcm_s16le")
                        sequence = int(event["sequence"])
                        timestamp_ms = float(event["timestamp_ms"])
                        try:
                            pcm = base64.b64decode(event["pcm_s16le_base64"], validate=True)
                        except (KeyError, binascii.Error) as exc:
                            raise ValueError("Invalid pcm_s16le_base64") from exc
                    await session.add_pcm(websocket, pcm, sequence, timestamp_ms)
                except (KeyError, TypeError, ValueError) as exc:
                    await websocket.send_json({"type": "asr.error", "code": "invalid_audio_chunk", "detail": str(exc)})
        except WebSocketDisconnect:
            pass
        except Exception:
            LOGGER.exception("WebSocket ASR session failed")
            try:
                await websocket.send_json({"type": "asr.error", "code": "internal_error"})
            except Exception:
                pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18783)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "PHOWHISPER_MODEL_DIR",
            f"models/PhoWhisper-large-{MODEL_REVISION}",
        ),
    )
    parser.add_argument("--model-id", default=os.environ.get("PHOWHISPER_MODEL_ID", MODEL_ID))
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("PHOWHISPER_MODEL_REVISION", MODEL_REVISION),
    )
    parser.add_argument(
        "--english-model-dir",
        default=os.environ.get("WHISPER_ENGLISH_MODEL_DIR"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(
        create_app(args.model_dir, args.model_id, args.model_revision, args.english_model_dir),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
