#!/usr/bin/env python3
"""Ten-turn production-service soak for the /vi early-sentence TTS path.

This harness uses the exact deployed service boundaries and the same sentence
boundary contract as vi-chat.js. It receives real MiniCPM text deltas and
starts real VieNeu streaming through the public gateway as soon as a sentence
is ready. Audio is received, not played, so first-audio timing is comparable to
the prior protocol soak's first decodable audio event.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import re
import ssl
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout

from e2e_vietnamese_protocol_soak import (
    QUESTIONS,
    SYSTEM_PROMPT,
    _health,
    _prepare_fixtures,
    _receive_json,
    _run_asr,
    _run_tts,
    _ssl_context,
    _ws_url,
)


BOUNDARY = re.compile(r'''[.!?…;]+(?:["'”’)\]]*)?(?=\s)|\n+''')


class SentenceBuffer:
    """Python port of native-overrides/vi-profile/vi-chat.js SentenceBuffer."""

    def __init__(self) -> None:
        self.pending = ""

    def push(self, delta: str, flush: bool = False) -> list[str]:
        self.pending += str(delta or "")
        ready: list[str] = []
        cut = 0
        for match in BOUNDARY.finditer(self.pending):
            end = match.end()
            sentence = " ".join(self.pending[cut:end].strip().split())
            if sentence:
                ready.append(sentence)
            cut = end
        self.pending = self.pending[cut:]
        if flush:
            tail = " ".join(self.pending.strip().split())
            if tail:
                ready.append(tail)
            self.pending = ""
        return ready


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)], 3)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95_nearest_rank": None, "max": None}
    return {
        "count": len(values),
        "p50": round(statistics.median(values), 3),
        "p95_nearest_rank": _nearest_rank(values, 0.95),
        "max": round(max(values), 3),
    }


async def _run_vlm_with_early_tts(
    session: ClientSession,
    *,
    base_url: str,
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
    input_sent_at: float | None = None
    first_text_at: float | None = None
    first_segment_ready_at: float | None = None
    response_done_at: float | None = None
    first_audio_at: float | None = None
    tts_all_done_at: float | None = None
    response_parts: list[str] = []
    event_types: list[str] = []
    server_metrics: dict[str, Any] = {}
    response_done_count = 0
    input_id = f"early_soak_{turn_number}_{uuid.uuid4().hex}"
    frame_meta: dict[str, Any] = {}
    buffer = SentenceBuffer()
    segment_queue: asyncio.Queue[str | None] = asyncio.Queue()
    segments: list[dict[str, Any]] = []
    segment_errors: list[str] = []

    async def tts_worker() -> None:
        nonlocal first_audio_at, tts_all_done_at
        while True:
            segment = await segment_queue.get()
            if segment is None:
                break
            request_started = time.perf_counter()
            try:
                result = await _run_tts(
                    session,
                    url=f"{base_url}/api/tts/vi/stream",
                    ssl_context=ssl_context,
                    text=segment,
                )
                reconstructed_first = request_started + (
                    result["client_wall_ms"]["request_to_first_audio"] / 1000
                )
                if first_audio_at is None:
                    first_audio_at = reconstructed_first
                segments.append(
                    {
                        "index": len(segments),
                        "text": segment,
                        "request_started_after_connect_ms": round(
                            (request_started - connected_at) * 1000, 3
                        ),
                        "client_wall_ms": result["client_wall_ms"],
                        "server_done": result["server_done"],
                        "audio_chunk_count": result["audio_chunk_count"],
                        "done_count": result["done_count"],
                        "errors": result["errors"],
                    }
                )
            except Exception as exc:
                segment_errors.append(f"{type(exc).__name__}: {exc}")
                break
        tts_all_done_at = time.perf_counter()

    worker = asyncio.create_task(tts_worker())
    try:
        async with session.ws_connect(
            _ws_url(base_url, "/v1/realtime?mode=chat"),
            ssl=ssl_context,
            heartbeat=20,
            max_msg_size=128 * 1024 * 1024,
        ) as ws:
            init_sent = False
            close_sent = False
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
                    captured_at = time.perf_counter()
                    frame = frame_path.read_bytes()
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
                                {"type": "image", "data": base64.b64encode(frame).decode("ascii")},
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
                        (input_sent_at - captured_at) * 1000, 3
                    )
                    continue
                if event_type == "response.output.delta" and kind == "text":
                    now = time.perf_counter()
                    if first_text_at is None:
                        first_text_at = now
                    delta = str(event.get("text") or "")
                    response_parts.append(delta)
                    for segment in buffer.push(delta):
                        if first_segment_ready_at is None:
                            first_segment_ready_at = now
                        segment_queue.put_nowait(segment)
                    continue
                if event_type == "response.done":
                    response_done_count += 1
                    response_done_at = time.perf_counter()
                    server_metrics = dict(event.get("metrics") or {})
                    if not response_parts and event.get("text"):
                        text = str(event["text"])
                        response_parts.append(text)
                        for segment in buffer.push(text):
                            if first_segment_ready_at is None:
                                first_segment_ready_at = response_done_at
                            segment_queue.put_nowait(segment)
                    for segment in buffer.push("", flush=True):
                        if first_segment_ready_at is None:
                            first_segment_ready_at = response_done_at
                        segment_queue.put_nowait(segment)
                    segment_queue.put_nowait(None)
                    await ws.send_json({"type": "session.close", "reason": "turn_done"})
                    close_sent = True
                    continue
                if event_type == "session.closed":
                    if not close_sent:
                        raise RuntimeError("Chat session closed before response.done")
                    break
                if event_type == "error":
                    raise RuntimeError(str(event.get("error") or event))
        await asyncio.wait_for(worker, timeout=timeout_s)
    finally:
        if not worker.done():
            worker.cancel()

    answer = "".join(response_parts).strip()
    if (
        not answer
        or input_sent_at is None
        or first_text_at is None
        or first_segment_ready_at is None
        or response_done_at is None
        or first_audio_at is None
        or tts_all_done_at is None
    ):
        raise RuntimeError("Early-TTS turn did not produce complete timing markers")
    if segment_errors:
        raise RuntimeError(f"Early-TTS segment failed: {segment_errors}")

    spoken_joined = " ".join(segment["text"] for segment in segments)
    normalized_answer = " ".join(answer.split())
    return answer, {
        "input_id": input_id,
        "answer": answer,
        "response_done_count": response_done_count,
        "event_types": event_types,
        "server_metrics": server_metrics,
        "frame": frame_meta,
        "segments": segments,
        "segment_count": len(segments),
        "segment_errors": segment_errors,
        "segment_text_exact": spoken_joined == normalized_answer,
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
            "input_to_first_segment_ready": round(
                (first_segment_ready_at - input_sent_at) * 1000, 3
            ),
            "input_to_response_done": round((response_done_at - input_sent_at) * 1000, 3),
            "input_to_first_audio": round((first_audio_at - input_sent_at) * 1000, 3),
            "first_audio_minus_response_done": round(
                (first_audio_at - response_done_at) * 1000, 3
            ),
            "input_to_all_tts_done": round((tts_all_done_at - input_sent_at) * 1000, 3),
        },
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    ssl_context = _ssl_context(args.insecure)
    timeout = ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=args.timeout)
    turns: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    async with ClientSession(timeout=timeout) as session:
        health_before = await _health(session, args.base_url, ssl_context)
        fixtures = await _prepare_fixtures(
            session,
            base_url=args.base_url,
            ssl_context=ssl_context,
            questions=QUESTIONS,
        )
        for index, (question, frame_path, fixture) in enumerate(
            zip(QUESTIONS, args.frame, fixtures), 1
        ):
            pcm, fixture_meta = fixture
            try:
                transcript, asr = await _run_asr(
                    session,
                    ws_url=_ws_url(
                        args.base_url,
                        "/v1/asr/vi?min_silence_ms=500&speech_pad_ms=300&partial_interval_ms=1600",
                    ),
                    ssl_context=ssl_context,
                    pcm=pcm,
                    timeout_s=args.timeout,
                )
                asr_final_at = time.perf_counter()
                answer, pipeline = await _run_vlm_with_early_tts(
                    session,
                    base_url=args.base_url,
                    ssl_context=ssl_context,
                    transcript=transcript,
                    frame_path=frame_path,
                    history=history,
                    timeout_s=args.timeout,
                    turn_number=index,
                )
                eos_to_first_audio = (
                    asr["wall_ms"]["speech_end_to_final"]
                    + pipeline["timing_ms"]["input_to_first_audio"]
                    + pipeline["timing_ms"]["connect_to_session_created"]
                )
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
                        "pipeline": pipeline,
                        "eos_to_first_audio_ms": round(eos_to_first_audio, 3),
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
                    }
                )
        health_after = await _health(session, args.base_url, ssl_context)

    completed = [turn for turn in turns if turn.get("error") is None]
    final_ids = [turn["asr"]["final_id"] for turn in completed]
    input_ids = [turn["pipeline"]["input_id"] for turn in completed]
    frame_hashes = [turn["pipeline"]["frame"]["sha256"] for turn in completed]
    duplicate_report = {
        "asr_final_id_duplicates": len(final_ids) - len(set(final_ids)),
        "chat_input_id_duplicates": len(input_ids) - len(set(input_ids)),
        "frame_hash_duplicates": len(frame_hashes) - len(set(frame_hashes)),
        "non_single_response_done_turns": sum(
            turn["pipeline"]["response_done_count"] != 1 for turn in completed
        ),
        "segment_text_mismatch_turns": sum(
            not turn["pipeline"]["segment_text_exact"] for turn in completed
        ),
        "tts_non_single_done_segments": sum(
            segment["done_count"] != 1
            for turn in completed
            for segment in turn["pipeline"]["segments"]
        ),
    }
    errors = [
        {"turn": turn["turn"], "error": turn["error"]}
        for turn in turns
        if turn.get("error")
    ]
    eos_values = [turn["eos_to_first_audio_ms"] for turn in completed]
    before_done = [
        turn["pipeline"]["timing_ms"]["first_audio_minus_response_done"] < 0
        for turn in completed
    ]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_summary = baseline["summary_ms"]["speech_end_to_tts_first_audio"]
    summary = {
        "eos_to_first_audio": _summary(eos_values),
        "asr_speech_end_to_final": _summary(
            [turn["asr"]["wall_ms"]["speech_end_to_final"] for turn in completed]
        ),
        "gateway_connect_to_session_created": _summary(
            [turn["pipeline"]["timing_ms"]["connect_to_session_created"] for turn in completed]
        ),
        "vlm_input_to_first_text": _summary(
            [turn["pipeline"]["timing_ms"]["input_to_first_text"] for turn in completed]
        ),
        "vlm_input_to_first_segment_ready": _summary(
            [turn["pipeline"]["timing_ms"]["input_to_first_segment_ready"] for turn in completed]
        ),
        "vlm_input_to_response_done": _summary(
            [turn["pipeline"]["timing_ms"]["input_to_response_done"] for turn in completed]
        ),
        "early_tts_input_to_first_audio": _summary(
            [turn["pipeline"]["timing_ms"]["input_to_first_audio"] for turn in completed]
        ),
        "frame_age_at_send": _summary(
            [turn["pipeline"]["frame"]["age_ms_at_send"] for turn in completed]
        ),
    }
    health_ok = all(
        health_after.get(name, {}).get("ok") for name in ("gateway", "asr")
    )
    passed = (
        len(completed) == len(QUESTIONS)
        and not errors
        and not any(duplicate_report.values())
        and health_ok
    )
    return {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "production /vi sentence buffer + real MiniCPM deltas + sequential VieNeu streams",
        "measurement_note": "Audio is fully received but not played; first decodable audio event is directly comparable to the prior protocol artifact.",
        "base_url": args.base_url,
        "turns_requested": len(QUESTIONS),
        "turns_completed": len(completed),
        "passed": passed,
        "health_before": health_before,
        "health_after": health_after,
        "duplicates": duplicate_report,
        "errors": errors,
        "early_audio_before_response_done_turns": sum(before_done),
        "summary_ms": summary,
        "baseline_comparison": {
            "artifact": str(args.baseline),
            "old_p50_ms": baseline_summary["p50"],
            "old_p95_ms": baseline_summary["p95_nearest_rank"],
            "new_p50_ms": summary["eos_to_first_audio"]["p50"],
            "new_p95_ms": summary["eos_to_first_audio"]["p95_nearest_rank"],
            "p50_delta_ms": round(
                summary["eos_to_first_audio"]["p50"] - baseline_summary["p50"], 3
            ),
            "p95_delta_ms": round(
                summary["eos_to_first_audio"]["p95_nearest_rank"]
                - baseline_summary["p95_nearest_rank"],
                3,
            ),
        },
        "turns": turns,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://127.0.0.1:8006")
    parser.add_argument("--frame", action="append", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=root / "artifacts" / "vietnamese-protocol-soak-10turn.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "vietnamese-early-tts-production-soak-10turn.json",
    )
    args = parser.parse_args()
    if not args.frame:
        frames = root / "results" / "indoor_laptop" / "frames"
        args.frame = [frames / f"frame_{index:05d}.jpg" for index in range(30, 40)]
    if len(args.frame) != len(QUESTIONS):
        parser.error(f"expected {len(QUESTIONS)} frames")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    report = asyncio.run(run(args))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
