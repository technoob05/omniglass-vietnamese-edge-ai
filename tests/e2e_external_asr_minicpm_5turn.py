#!/usr/bin/env python3
"""Five-turn contract soak for external-ASR text into MiniCPM-o chat.

This test does not capture or transcribe audio.  It starts at the boundary an
external ASR service must provide: one immutable, finalized Vietnamese
transcript per utterance.  Each turn reads a frame immediately before sending
it, carries the rolling text history explicitly, and records queue/first-token/
completion latency from the native MiniCPM gateway.

The gateway is not modified or restarted.  By default the test only prints a
report; pass ``--output`` to persist it.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import ssl
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout, WSMsgType


DEFAULT_TURNS = (
    "Bạn đang nhìn thấy gì trước mặt tôi?",
    "Vật nổi bật nhất nằm ở phía nào?",
    "Hãy nhớ vật đó và mô tả màu của nó.",
    "Trong ảnh hiện tại có thay đổi gì đáng chú ý?",
    "Tóm tắt ngắn gọn những gì bạn đã quan sát qua năm lượt.",
)


def _ssl_context(insecure: bool) -> ssl.SSLContext | bool:
    if not insecure:
        return ssl.create_default_context()
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _frame_payload(path: Path) -> tuple[str, dict[str, Any]]:
    read_started = time.perf_counter()
    payload = path.read_bytes()
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError(f"Expected a JPEG frame: {path}")
    return base64.b64encode(payload).decode("ascii"), {
        "path": str(path),
        "bytes": len(payload),
        "sha256_12": hashlib.sha256(payload).hexdigest()[:12],
        "read_ms": round((time.perf_counter() - read_started) * 1000, 1),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1))
    return round(ordered[index], 1)


async def _receive_json(ws: Any, timeout_s: float) -> dict[str, Any]:
    message = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
    if message.type == WSMsgType.TEXT:
        value = json.loads(message.data)
        if isinstance(value, dict):
            return value
        raise RuntimeError(f"Gateway returned non-object JSON: {value!r}")
    if message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING}:
        raise RuntimeError(f"Gateway closed WebSocket: {message.extra or message.data}")
    if message.type == WSMsgType.ERROR:
        raise RuntimeError(f"Gateway WebSocket error: {ws.exception()}")
    raise RuntimeError(f"Unexpected WebSocket message type: {message.type}")


async def _one_turn(
    session: ClientSession,
    *,
    ws_url: str,
    ssl_context: ssl.SSLContext | bool,
    transcript: str,
    frame_path: Path,
    history: list[dict[str, Any]],
    turn_number: int,
    timeout_s: float,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    turn_id = f"turn_{turn_number}_{uuid.uuid4().hex[:10]}"
    connected_at = time.perf_counter()
    queue_done_at: float | None = None
    session_created_at: float | None = None
    input_sent_at: float | None = None
    first_token_at: float | None = None
    done_at: float | None = None
    frame_captured_at: float | None = None
    frame_age_ms_at_send: float | None = None
    frame_meta: dict[str, Any] = {}
    response_parts: list[str] = []
    event_types: list[str] = []
    server_metrics: dict[str, Any] = {}

    async with session.ws_connect(
        ws_url,
        ssl=ssl_context,
        max_msg_size=128 * 1024 * 1024,
        heartbeat=20,
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
                if not init_sent:
                    raise RuntimeError("session.created arrived before session.init")
                # Queueing can take hundreds of milliseconds.  Capture only
                # after the worker session is ready so the VLM never receives
                # the frame that happened to be current before the queue.
                frame_captured_at = time.perf_counter()
                frame_base64, frame_meta = _frame_payload(frame_path)
                turn_messages = [
                    *history,
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "data": frame_base64},
                            {"type": "text", "text": transcript},
                        ],
                    },
                ]
                input_payload = {
                    "input_id": turn_id,
                    "messages": turn_messages,
                    "streaming": True,
                    "generation": {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": False,
                        "length_penalty": 1.0,
                    },
                    "image": {"max_slice_nums": 1},
                    "tts": {"enabled": False},
                    "use_tts_template": False,
                    "omni_mode": False,
                    "enable_thinking": False,
                }
                frame_age_ms_at_send = round(
                    (time.perf_counter() - frame_captured_at) * 1000, 1
                )
                await ws.send_json({"type": "input.append", "input": input_payload})
                input_sent_at = time.perf_counter()
                continue
            if event_type == "response.output.delta" and kind == "text":
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                response_parts.append(str(event.get("text") or ""))
                continue
            if event_type == "response.done":
                done_at = time.perf_counter()
                server_metrics = dict(event.get("metrics") or {})
                if not response_parts and event.get("text"):
                    response_parts.append(str(event["text"]))
                await ws.send_json({"type": "session.close", "reason": "turn_done"})
                close_sent = True
                continue
            if event_type == "session.closed":
                if not close_sent:
                    raise RuntimeError(f"Session closed before response.done: {event}")
                break
            if event_type == "error":
                raise RuntimeError(str(event.get("error") or event))

    answer = "".join(response_parts).strip()
    if not answer:
        raise RuntimeError(f"Turn {turn_number} completed without text")
    if input_sent_at is None or done_at is None:
        raise RuntimeError(f"Turn {turn_number} did not complete the gateway contract")

    report = {
        "turn": turn_number,
        "turn_id": turn_id,
        "stable_final": transcript,
        "answer": answer,
        "frame": {
            **frame_meta,
            "age_ms_at_send": frame_age_ms_at_send,
        },
        "latency_ms": {
            "connect_to_queue_done": round((queue_done_at - connected_at) * 1000, 1)
            if queue_done_at is not None
            else None,
            "queue_done_to_session_created": round(
                (session_created_at - queue_done_at) * 1000, 1
            )
            if queue_done_at is not None and session_created_at is not None
            else None,
            "input_to_first_text": round((first_token_at - input_sent_at) * 1000, 1)
            if first_token_at is not None
            else None,
            "input_to_done": round((done_at - input_sent_at) * 1000, 1),
        },
        "server_metrics": server_metrics,
        "event_types": event_types,
    }
    return answer, report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    frame_paths = args.frame
    if len(frame_paths) not in {1, len(args.turn)}:
        raise ValueError("Provide either one --frame or exactly one --frame per --turn")
    for frame_path in frame_paths:
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)

    system_message = {
        "role": "system",
        "content": (
            "Bạn là trợ lý thị giác cho người Việt. Trả lời bằng tiếng Việt, "
            "ngắn gọn và dựa đúng vào ảnh mới nhất. Nếu không chắc, hãy nói rõ."
        ),
    }
    history: list[dict[str, Any]] = [system_message]
    reports: list[dict[str, Any]] = []
    timeout = ClientTimeout(total=None, connect=15, sock_connect=15, sock_read=args.timeout)

    async with ClientSession(timeout=timeout) as session:
        for index, transcript in enumerate(args.turn, 1):
            frame_path = frame_paths[0] if len(frame_paths) == 1 else frame_paths[index - 1]
            answer, report = await _one_turn(
                session,
                ws_url=args.gateway,
                ssl_context=_ssl_context(args.insecure),
                transcript=transcript,
                frame_path=frame_path,
                history=history,
                turn_number=index,
                timeout_s=args.timeout,
                max_new_tokens=args.max_new_tokens,
            )
            reports.append(report)
            # Do not retain old images.  Text history keeps conversational
            # memory bounded while every new question is grounded in a fresh
            # frame from its own turn.
            history.extend(
                [
                    {"role": "user", "content": transcript},
                    {"role": "assistant", "content": answer},
                ]
            )

    first_text_values = [
        turn["latency_ms"]["input_to_first_text"]
        for turn in reports
        if turn["latency_ms"]["input_to_first_text"] is not None
    ]
    done_values = [turn["latency_ms"]["input_to_done"] for turn in reports]
    frame_age_values = [turn["frame"]["age_ms_at_send"] for turn in reports]
    queue_values = [
        turn["latency_ms"]["connect_to_queue_done"]
        for turn in reports
        if turn["latency_ms"]["connect_to_queue_done"] is not None
    ]
    return {
        "contract": "external stable transcript + fresh JPEG -> native MiniCPM chat",
        "gateway": args.gateway,
        "turns_requested": len(args.turn),
        "turns_completed": len(reports),
        "passed": len(reports) == len(args.turn) and all(turn["answer"] for turn in reports),
        "summary_ms": {
            "queue_p50": _percentile(queue_values, 0.50),
            "queue_p95": _percentile(queue_values, 0.95),
            "frame_age_p50": _percentile(frame_age_values, 0.50),
            "frame_age_p95": _percentile(frame_age_values, 0.95),
            "first_text_p50": _percentile(first_text_values, 0.50),
            "first_text_p95": _percentile(first_text_values, 0.95),
            "done_p50": _percentile(done_values, 0.50),
            "done_p95": _percentile(done_values, 0.95),
        },
        "turns": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gateway",
        default="wss://127.0.0.1:8006/v1/realtime?mode=chat",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        action="append",
        required=True,
        help="Repeat five times for changing frames, or pass once for a protocol-only soak.",
    )
    parser.add_argument("--turn", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.turn:
        args.turn = list(DEFAULT_TURNS)
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
