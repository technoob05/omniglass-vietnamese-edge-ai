#!/usr/bin/env python3
"""Ten-turn timing soak for the real dedicated Vietnamese service chain.

The harness deliberately goes through the native gateway routes used by /vi:
MMS creates deterministic microphone fixtures, PhoWhisper emits one immutable
final, the native MiniCPM gateway receives a freshly read JPEG, and VieNeu
streams the answer.  Audio is received but not played, so the report measures
service completion rather than loudspeaker playback duration.

No service lifecycle operation is performed by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import ssl
import statistics
import sys
import time
import uuid
import wave
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import ClientSession, ClientTimeout, WSMsgType


QUESTIONS = (
    "Bạn thấy gì trước mặt?",
    "Có người trong ảnh không?",
    "Có vật gì trên bàn?",
    "Vật nổi bật nằm ở đâu?",
    "Màu sắc nổi bật là gì?",
    "Có máy tính trong ảnh không?",
    "Phía trước có nguy hiểm rõ ràng không?",
    "Hãy mô tả phần bên trái.",
    "Hãy mô tả phần bên phải.",
    "Tóm tắt cảnh trước mặt bằng một câu.",
)
SYSTEM_PROMPT = (
    "Bạn là trợ lý thị giác cho người Việt. Trả lời bằng tiếng Việt, ngắn gọn, "
    "chính xác và chỉ dựa vào ảnh mới nhất. Nếu không chắc, hãy nói rõ."
)


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def _ws_url(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}{path}"


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "p50": None, "p95_nearest_rank": None, "max": None}
    return {
        "count": len(values),
        "p50": round(statistics.median(values), 3),
        "p95_nearest_rank": _nearest_rank(values, 0.95),
        "max": round(max(values), 3),
    }


async def _receive_json(ws: Any, timeout_s: float) -> dict[str, Any]:
    message = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if message.type == WSMsgType.TEXT:
        value = json.loads(message.data)
        if isinstance(value, dict):
            return value
        raise RuntimeError(f"Expected JSON object, got {type(value).__name__}")
    if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING}:
        raise RuntimeError(f"WebSocket closed: {message.data} {message.extra}")
    if message.type == WSMsgType.ERROR:
        raise RuntimeError(f"WebSocket error: {ws.exception()}")
    raise RuntimeError(f"Unexpected WebSocket message type: {message.type}")


def _decode_fixture(wav_bytes: bytes) -> tuple[bytes, dict[str, Any]]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_in:
        channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        sample_rate = wav_in.getframerate()
        frames = wav_in.getnframes()
        if (channels, sample_width, sample_rate) != (1, 2, 16_000):
            raise ValueError(
                "Fixture must be mono PCM16 16 kHz, got "
                f"channels={channels}, width={sample_width}, rate={sample_rate}"
            )
        pcm = wav_in.readframes(frames)
    return pcm, {
        "sample_rate": sample_rate,
        "samples": frames,
        "duration_ms": round(frames / sample_rate * 1000, 3),
        "sha256_12": hashlib.sha256(pcm).hexdigest()[:12],
    }


async def _prepare_fixtures(
    session: ClientSession,
    *,
    base_url: str,
    ssl_context: ssl.SSLContext,
    questions: tuple[str, ...],
) -> list[tuple[bytes, dict[str, Any]]]:
    fixtures: list[tuple[bytes, dict[str, Any]]] = []
    for question in questions:
        async with session.post(
            f"{base_url}/api/tts/vi",
            json={"text": question},
            ssl=ssl_context,
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        pcm, metadata = _decode_fixture(base64.b64decode(payload["audio_wav_base64"]))
        fixtures.append((pcm, metadata | {"text": question}))
    return fixtures


async def _run_asr(
    session: ClientSession,
    *,
    ws_url: str,
    ssl_context: ssl.SSLContext,
    pcm: bytes,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    chunk_samples = 3_200
    chunk_bytes = chunk_samples * 2
    silence_chunks = 4
    started = time.perf_counter()
    event_types: list[str] = []
    finals: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    speech_end_at: float | None = None
    final_received_at: float | None = None

    async with session.ws_connect(
        ws_url,
        ssl=ssl_context,
        heartbeat=20,
        max_msg_size=16 * 1024 * 1024,
    ) as ws:
        ready = await _receive_json(ws, timeout_s)
        if ready.get("type") != "asr.ready":
            raise RuntimeError(f"Expected asr.ready, got {ready}")
        event_types.append("asr.ready")
        sequence = 0
        timestamp_origin_ms = time.time() * 1000
        loop = asyncio.get_running_loop()
        final_future: asyncio.Future[tuple[dict[str, Any], float]] = loop.create_future()

        async def receive_events() -> None:
            try:
                while True:
                    event = await _receive_json(ws, timeout_s)
                    event_type = str(event.get("type") or "")
                    event_types.append(event_type)
                    if event_type == "asr.final":
                        finals.append(event)
                        if not final_future.done():
                            final_future.set_result((event, time.perf_counter()))
                    elif event_type == "asr.error":
                        errors.append(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not final_future.done():
                    final_future.set_exception(exc)

        receiver = asyncio.create_task(receive_events())

        async def send_chunk(chunk: bytes) -> None:
            nonlocal sequence
            await ws.send_json(
                {
                    "type": "audio.chunk",
                    "sequence": sequence,
                    "timestamp_ms": timestamp_origin_ms + sequence * 200,
                    "sample_rate": 16_000,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "pcm_s16le_base64": base64.b64encode(chunk).decode("ascii"),
                }
            )
            sequence += 1
            await asyncio.sleep(len(chunk) / 2 / 16_000)

        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            if len(chunk) < chunk_bytes:
                chunk += b"\x00" * (chunk_bytes - len(chunk))
            await send_chunk(chunk)
        speech_end_at = time.perf_counter()
        for _ in range(silence_chunks):
            if final_future.done():
                break
            await send_chunk(b"\x00" * chunk_bytes)
        if not final_future.done():
            await ws.send_json({"type": "audio.end"})
        _, final_received_at = await asyncio.wait_for(final_future, timeout=timeout_s)
        # Keep receiving briefly to detect accidental duplicate finals.
        await asyncio.sleep(0.15)
        receiver.cancel()
        with suppress(asyncio.CancelledError):
            await receiver
        await ws.close()

    if len(finals) != 1:
        raise RuntimeError(f"Expected exactly one asr.final, got {len(finals)}; errors={errors}")
    final = finals[0]
    if not final.get("immutable") or not final.get("final_id"):
        raise RuntimeError(f"Invalid final contract: {final}")
    if speech_end_at is None or final_received_at is None:
        raise RuntimeError("ASR timing markers were not captured")
    return str(final.get("text") or "").strip(), {
        "final_id": final["final_id"],
        "immutable": final["immutable"],
        "final_count": len(finals),
        "text": final.get("text"),
        "event_types": event_types,
        "errors": errors,
        "audio": final.get("audio"),
        "endpoint": final.get("endpoint"),
        "inference": final.get("inference"),
        "wall_ms": {
            "connect_to_final": round((final_received_at - started) * 1000, 3),
            "speech_end_to_final": round((final_received_at - speech_end_at) * 1000, 3),
        },
    }


async def _run_vlm(
    session: ClientSession,
    *,
    ws_url: str,
    ssl_context: ssl.SSLContext,
    transcript: str,
    frame_path: Path,
    history: list[dict[str, Any]],
    timeout_s: float,
    turn_number: int,
) -> tuple[str, dict[str, Any]]:
    connected_at = time.perf_counter()
    queue_done_at: float | None = None
    session_created_at: float | None = None
    frame_captured_at: float | None = None
    input_sent_at: float | None = None
    first_text_at: float | None = None
    done_at: float | None = None
    response_parts: list[str] = []
    event_types: list[str] = []
    errors: list[dict[str, Any]] = []
    done_count = 0
    input_id = f"soak_{turn_number}_{uuid.uuid4().hex}"
    frame_meta: dict[str, Any] = {}
    server_metrics: dict[str, Any] = {}

    async with session.ws_connect(
        ws_url,
        ssl=ssl_context,
        heartbeat=20,
        max_msg_size=128 * 1024 * 1024,
    ) as ws:
        init_sent = False
        while True:
            event = await _receive_json(ws, timeout_s)
            event_type = str(event.get("type") or "")
            kind = str(event.get("kind") or "")
            event_types.append(f"{event_type}/{kind}" if kind else event_type)
            if event_type in {"session.queued", "session.queue_update"}:
                continue
            if event_type in {"session.queue_done", "queue_done"}:
                queue_done_at = time.perf_counter()
                if not init_sent:
                    await ws.send_json({"type": "session.init", "payload": {}})
                    init_sent = True
                continue
            if event_type == "session.created":
                session_created_at = time.perf_counter()
                if not init_sent:
                    raise RuntimeError("session.created arrived before session.init")
                frame_captured_at = time.perf_counter()
                frame = frame_path.read_bytes()
                if not frame.startswith(b"\xff\xd8"):
                    raise ValueError(f"Expected JPEG frame: {frame_path}")
                frame_base64 = base64.b64encode(frame).decode("ascii")
                frame_meta = {
                    "path": str(frame_path),
                    "bytes": len(frame),
                    "sha256": hashlib.sha256(frame).hexdigest(),
                }
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *history[-10:],
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "data": frame_base64},
                            {"type": "text", "text": transcript},
                        ],
                    },
                ]
                await ws.send_json(
                    {
                        "type": "input.append",
                        "input": {
                            "input_id": input_id,
                            "messages": messages,
                            "streaming": True,
                            "generation": {
                                "max_new_tokens": 64,
                                "do_sample": False,
                                "length_penalty": 1.0,
                            },
                            "image": {"max_slice_nums": 1},
                            "tts": {"enabled": False},
                            "use_tts_template": False,
                            "omni_mode": False,
                            "enable_thinking": False,
                        },
                    }
                )
                input_sent_at = time.perf_counter()
                frame_meta["age_ms_at_send"] = round(
                    (input_sent_at - frame_captured_at) * 1000, 3
                )
                continue
            if event_type == "response.output.delta" and kind == "text":
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                response_parts.append(str(event.get("text") or ""))
                continue
            if event_type == "response.done":
                done_count += 1
                done_at = time.perf_counter()
                server_metrics = dict(event.get("metrics") or {})
                if not response_parts and event.get("text"):
                    response_parts.append(str(event["text"]))
                await ws.send_json({"type": "session.close", "reason": "turn_done"})
                continue
            if event_type == "session.closed":
                break
            if event_type == "error":
                errors.append(event)
                raise RuntimeError(str(event.get("error") or event))

    answer = "".join(response_parts).strip()
    if not answer or input_sent_at is None or first_text_at is None or done_at is None:
        raise RuntimeError("VLM response did not complete with text")
    return answer, {
        "input_id": input_id,
        "response_done_count": done_count,
        "answer": answer,
        "frame": frame_meta,
        "event_types": event_types,
        "errors": errors,
        "server_metrics": server_metrics,
        "timing_ms": {
            "connect_to_queue_done": round((queue_done_at - connected_at) * 1000, 3)
            if queue_done_at is not None
            else None,
            "queue_done_to_session_created": round(
                (session_created_at - queue_done_at) * 1000, 3
            )
            if session_created_at is not None and queue_done_at is not None
            else None,
            "connect_to_session_created": round(
                (session_created_at - connected_at) * 1000, 3
            )
            if session_created_at is not None
            else None,
            "input_to_first_text": round((first_text_at - input_sent_at) * 1000, 3),
            "input_to_done": round((done_at - input_sent_at) * 1000, 3),
        },
    }


async def _run_tts(
    session: ClientSession,
    *,
    url: str,
    ssl_context: ssl.SSLContext,
    text: str,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    first_audio_at: float | None = None
    done_at: float | None = None
    pending = ""
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    async with session.post(url, json={"text": text}, ssl=ssl_context) as response:
        response.raise_for_status()
        async for chunk in response.content.iter_any():
            pending += chunk.decode("utf-8")
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                event = json.loads(line)
                events.append(event)
                if event.get("type") == "audio" and first_audio_at is None:
                    first_audio_at = time.perf_counter()
                if event.get("type") == "done":
                    done_at = time.perf_counter()
                if event.get("type") == "error":
                    errors.append(event)
        if pending.strip():
            event = json.loads(pending)
            events.append(event)
            if event.get("type") == "audio" and first_audio_at is None:
                first_audio_at = time.perf_counter()
            if event.get("type") == "done":
                done_at = time.perf_counter()
            if event.get("type") == "error":
                errors.append(event)

    done_events = [event for event in events if event.get("type") == "done"]
    audio_events = [event for event in events if event.get("type") == "audio"]
    if errors or len(done_events) != 1 or not audio_events or first_audio_at is None or done_at is None:
        raise RuntimeError(
            f"Incomplete VieNeu stream: audio={len(audio_events)}, done={len(done_events)}, errors={errors}"
        )
    return {
        "http_status": response.status,
        "audio_chunk_count": len(audio_events),
        "done_count": len(done_events),
        "errors": errors,
        "meta": next((event for event in events if event.get("type") == "meta"), None),
        "server_done": done_events[0],
        "client_wall_ms": {
            "request_to_first_audio": round((first_audio_at - request_started) * 1000, 3),
            "request_to_done": round((done_at - request_started) * 1000, 3),
        },
    }


async def _health(
    session: ClientSession, base_url: str, ssl_context: ssl.SSLContext
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in (("gateway", "/health"), ("asr", "/api/asr/vi/health")):
        started = time.perf_counter()
        try:
            async with session.get(f"{base_url}{path}", ssl=ssl_context) as response:
                payload = await response.json(content_type=None)
                result[name] = {
                    "ok": response.status == 200,
                    "status": response.status,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "payload": payload,
                }
        except Exception as exc:
            result[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.frame) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} --frame arguments")
    for frame in args.frame:
        if not frame.is_file():
            raise FileNotFoundError(frame)

    ssl_context = _ssl_context(args.insecure)
    timeout = ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=args.timeout)
    turns: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    async with ClientSession(timeout=timeout) as session:
        health_before = await _health(session, args.base_url, ssl_context)
        fixtures_started = time.perf_counter()
        fixtures = await _prepare_fixtures(
            session,
            base_url=args.base_url,
            ssl_context=ssl_context,
            questions=QUESTIONS,
        )
        fixture_generation_ms = round((time.perf_counter() - fixtures_started) * 1000, 3)

        for index, (question, frame_path, fixture) in enumerate(
            zip(QUESTIONS, args.frame, fixtures), 1
        ):
            pcm, fixture_meta = fixture
            turn_started = time.perf_counter()
            try:
                transcript, asr = await _run_asr(
                    session,
                    ws_url=_ws_url(
                        args.base_url,
                        f"/v1/asr/vi?min_silence_ms={args.min_silence_ms}"
                        "&speech_pad_ms=300&partial_interval_ms=1600",
                    ),
                    ssl_context=ssl_context,
                    pcm=pcm,
                    timeout_s=args.timeout,
                )
                asr_final_at = time.perf_counter()
                answer, vlm = await _run_vlm(
                    session,
                    ws_url=_ws_url(args.base_url, "/v1/realtime?mode=chat"),
                    ssl_context=ssl_context,
                    transcript=transcript,
                    frame_path=frame_path,
                    history=history,
                    timeout_s=args.timeout,
                    turn_number=index,
                )
                vlm_done_at = time.perf_counter()
                tts = await _run_tts(
                    session,
                    url=f"{args.base_url}/api/tts/vi/stream",
                    ssl_context=ssl_context,
                    text=answer,
                )
                turn_done_at = time.perf_counter()
                history.extend(
                    [
                        {"role": "user", "content": transcript},
                        {"role": "assistant", "content": answer},
                    ]
                )
                history = history[-10:]
                turns.append(
                    {
                        "turn": index,
                        "requested_text": question,
                        "fixture": fixture_meta,
                        "asr": asr,
                        "vlm": vlm,
                        "tts": tts,
                        "stage_wall_ms": {
                            "asr_stage": round((asr_final_at - turn_started) * 1000, 3),
                            "gateway_vlm_stage": round((vlm_done_at - asr_final_at) * 1000, 3),
                            "tts_stage": round((turn_done_at - vlm_done_at) * 1000, 3),
                            "total_turn": round((turn_done_at - turn_started) * 1000, 3),
                            "speech_end_to_tts_first_audio": round(
                                asr["wall_ms"]["speech_end_to_final"]
                                + (vlm_done_at - asr_final_at) * 1000
                                + tts["client_wall_ms"]["request_to_first_audio"],
                                3,
                            ),
                            "speech_end_to_tts_done": round(
                                asr["wall_ms"]["speech_end_to_final"]
                                + (turn_done_at - asr_final_at) * 1000,
                                3,
                            ),
                        },
                        "error": None,
                    }
                )
            except Exception as exc:
                turns.append(
                    {
                        "turn": index,
                        "requested_text": question,
                        "fixture": fixture_meta,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_ms": round((time.perf_counter() - turn_started) * 1000, 3),
                    }
                )

        health_after = await _health(session, args.base_url, ssl_context)

    completed = [turn for turn in turns if turn.get("error") is None]
    final_ids = [turn["asr"]["final_id"] for turn in completed]
    input_ids = [turn["vlm"]["input_id"] for turn in completed]
    transcripts = [str(turn["asr"]["text"]) for turn in completed]
    frame_hashes = [turn["vlm"]["frame"]["sha256"] for turn in completed]
    metric_paths: dict[str, list[float]] = {
        "asr_speech_end_to_final": [turn["asr"]["wall_ms"]["speech_end_to_final"] for turn in completed],
        "asr_inference": [turn["asr"]["inference"]["latency_ms"] for turn in completed],
        "asr_algorithmic_endpoint_delay": [turn["asr"]["endpoint"]["algorithmic_delay_ms"] for turn in completed],
        "gateway_connect_to_queue_done": [turn["vlm"]["timing_ms"]["connect_to_queue_done"] for turn in completed],
        "gateway_queue_done_to_session_created": [turn["vlm"]["timing_ms"]["queue_done_to_session_created"] for turn in completed],
        "gateway_connect_to_session_created": [turn["vlm"]["timing_ms"]["connect_to_session_created"] for turn in completed],
        "vlm_input_to_first_text": [turn["vlm"]["timing_ms"]["input_to_first_text"] for turn in completed],
        "vlm_input_to_done": [turn["vlm"]["timing_ms"]["input_to_done"] for turn in completed],
        "vieneu_client_request_to_first_audio": [turn["tts"]["client_wall_ms"]["request_to_first_audio"] for turn in completed],
        "vieneu_client_request_to_done": [turn["tts"]["client_wall_ms"]["request_to_done"] for turn in completed],
        "vieneu_server_queue": [turn["tts"]["server_done"]["queue_ms"] for turn in completed],
        "vieneu_server_first_chunk": [turn["tts"]["server_done"]["first_chunk_ms"] for turn in completed],
        "vieneu_server_total": [turn["tts"]["server_done"]["total_ms"] for turn in completed],
        "frame_age_at_send": [turn["vlm"]["frame"]["age_ms_at_send"] for turn in completed],
        "total_turn": [turn["stage_wall_ms"]["total_turn"] for turn in completed],
        "speech_end_to_tts_first_audio": [turn["stage_wall_ms"]["speech_end_to_tts_first_audio"] for turn in completed],
        "speech_end_to_tts_done": [turn["stage_wall_ms"]["speech_end_to_tts_done"] for turn in completed],
    }
    protocol_errors = [turn for turn in turns if turn.get("error")]
    duplicate_report = {
        "final_id_duplicates": len(final_ids) - len(set(final_ids)),
        "input_id_duplicates": len(input_ids) - len(set(input_ids)),
        "transcript_duplicates": len(transcripts) - len(set(transcripts)),
        "frame_hash_duplicates": len(frame_hashes) - len(set(frame_hashes)),
        "asr_non_single_final_turns": sum(turn["asr"]["final_count"] != 1 for turn in completed),
        "vlm_non_single_done_turns": sum(turn["vlm"]["response_done_count"] != 1 for turn in completed),
        "tts_non_single_done_turns": sum(turn["tts"]["done_count"] != 1 for turn in completed),
    }
    health_ok = all(
        health_after.get(service, {}).get("ok") for service in ("gateway", "asr")
    )
    passed = (
        len(completed) == len(QUESTIONS)
        and not protocol_errors
        and duplicate_report["final_id_duplicates"] == 0
        and duplicate_report["input_id_duplicates"] == 0
        and duplicate_report["asr_non_single_final_turns"] == 0
        and duplicate_report["vlm_non_single_done_turns"] == 0
        and duplicate_report["tts_non_single_done_turns"] == 0
        and health_ok
    )
    return {
        "schema_version": 2,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "real-time PCM -> /v1/asr/vi -> native MiniCPM chat + fresh JPEG -> /api/tts/vi/stream",
        "measurement_note": "Fixtures are streamed in real time while ASR events are received concurrently. TTS audio is fully received but not played. Total turn includes spoken fixture duration and excludes fixture generation.",
        "base_url": args.base_url,
        "asr_session_config": {
            "min_silence_ms": args.min_silence_ms,
            "speech_pad_ms": 300,
            "partial_interval_ms": 1600,
        },
        "turns_requested": len(QUESTIONS),
        "turns_completed": len(completed),
        "passed": passed,
        "fixture_generation_ms_excluded": fixture_generation_ms,
        "health_before": health_before,
        "health_after": health_after,
        "tts_health_after": {
            "explicit_endpoint_available": False,
            "operational_evidence": f"{sum(turn.get('tts', {}).get('done_count') == 1 for turn in completed)}/{len(QUESTIONS)} real VieNeu streams emitted exactly one done event",
        },
        "duplicates": duplicate_report,
        "errors": [
            {"turn": turn["turn"], "error": turn["error"]} for turn in protocol_errors
        ],
        "summary_ms": {name: _summary(values) for name, values in metric_paths.items()},
        "turns": turns,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://127.0.0.1:8006")
    parser.add_argument("--frame", type=Path, action="append")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--min-silence-ms", type=int, default=500)
    parser.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "vietnamese-protocol-soak-10turn.json",
    )
    args = parser.parse_args()
    if not args.frame:
        frames = root / "results" / "indoor_laptop" / "frames"
        args.frame = [frames / f"frame_{index:05d}.jpg" for index in range(30, 40)]
    if not 100 <= args.min_silence_ms <= 2000:
        parser.error("--min-silence-ms must be between 100 and 2000")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
