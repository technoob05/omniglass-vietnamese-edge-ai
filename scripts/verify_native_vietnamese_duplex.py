#!/usr/bin/env python3
"""Protocol smoke test for per-session MiniCPM-o native-audio control."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets


def pcm_base64(path: Path) -> str:
    audio, _ = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return base64.b64encode(np.asarray(audio, dtype=np.float32).tobytes()).decode("ascii")


def one_second_chunks(audio: np.ndarray, sample_rate: int, count: int) -> list[str]:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz input, got {sample_rate}")
    chunk_samples = sample_rate
    if len(audio) == 0:
        raise ValueError("Input audio is empty")
    # A real utterance is followed by silence; repeating the utterance makes
    # the duplex policy correctly keep listening forever.
    padded = np.pad(audio, (0, max(0, count * chunk_samples - len(audio))))
    return [
        base64.b64encode(
            np.asarray(padded[i * chunk_samples : (i + 1) * chunk_samples], dtype=np.float32).tobytes()
        ).decode("ascii")
        for i in range(count)
    ]


async def load_input_audio(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    if args.input_text:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(args.mms_url, json={"text": args.input_text})
            response.raise_for_status()
            wav_bytes = base64.b64decode(response.json()["audio_wav_base64"])
        return sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if args.input_wav is None:
        raise ValueError("Provide --input-wav or --input-text")
    return sf.read(args.input_wav, dtype="float32", always_2d=False)


async def receive_event(ws: websockets.ClientConnection, timeout: float) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise RuntimeError(f"Non-object event: {event!r}")
    return event


async def run_session(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    events: list[dict] = []
    text_parts: list[str] = []
    audio_events = 0
    chunks_sent = 0
    turn_ended_after_text = False

    init_payload = {
        "mode": "full_duplex",
        "system_prompt": args.system_prompt,
        "config": {
            "generate_audio": args.generate_audio,
            "force_listen_count": 0,
            "max_new_speak_tokens_per_chunk": 20,
            "length_penalty": 1.0,
            "listen_prob_scale": args.listen_prob_scale,
        },
        "ref_audio_base64": pcm_base64(args.ref_audio),
        "max_slice_nums": 1,
    }

    input_audio, input_sample_rate = await load_input_audio(args)
    chunks = one_second_chunks(input_audio, input_sample_rate, args.max_chunks)

    async with websockets.connect(args.backend_ws, max_size=128 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": init_payload}))
        created = await receive_event(ws, args.event_timeout)
        events.append(created)
        if created.get("type") != "session.created":
            raise RuntimeError(f"Unexpected init event: {created}")
        session_id = str(created.get("session_id"))

        for index, chunk in enumerate(chunks, 1):
            await ws.send(json.dumps({
                "type": "input.append",
                "input": {"audio": chunk, "input_id": f"chunk_{index}"},
            }))
            chunks_sent = index
            first = await receive_event(ws, args.event_timeout)
            batch = [first]
            while True:
                try:
                    batch.append(await receive_event(ws, args.drain_timeout))
                except asyncio.TimeoutError:
                    break

            for event in batch:
                events.append(event)
                if event.get("type") == "response.output.delta":
                    kind = event.get("kind")
                    if kind == "text" and event.get("text"):
                        text_parts.append(str(event["text"]))
                    elif kind == "audio" and event.get("audio"):
                        audio_events += 1
                    elif kind == "listen" and text_parts:
                        turn_ended_after_text = True

            if (not args.generate_audio and turn_ended_after_text) or (
                args.generate_audio and text_parts and audio_events > 0
            ):
                break

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{args.backend_http}/sessions/{session_id}/close",
                json={"reason": "verification_complete"},
            )
            response.raise_for_status()

    text = "".join(text_parts).strip()
    metrics = [
        event.get("metrics", {})
        for event in events
        if event.get("type") == "response.output.delta"
    ]
    result = {
        "session_id": session_id,
        "requested_generate_audio": args.generate_audio,
        "chunks_sent": chunks_sent,
        "text": text,
        "text_events": len(text_parts),
        "native_audio_events": audio_events,
        "turn_ended_after_text": turn_ended_after_text,
        "event_types": [
            f"{event.get('type')}/{event.get('kind', '')}" for event in events
        ],
        "metrics": metrics,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }

    if args.mms_url and text and not args.generate_audio:
        tts_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(args.mms_url, json={"text": text})
            response.raise_for_status()
            tts = response.json()
        result["mms_tts"] = {
            "http_elapsed_ms": round((time.perf_counter() - tts_started) * 1000, 1),
            "inference_ms": tts.get("inference_ms"),
            "duration_seconds": tts.get("duration_seconds"),
            "audio_base64_chars": len(tts.get("audio_wav_base64", "")),
            "model": tts.get("model"),
        }

    contract_errors: list[str] = []
    if not text:
        contract_errors.append("the session never exercised the speak branch")
    if args.generate_audio:
        if audio_events < 1:
            contract_errors.append("generate_audio=true emitted no native audio event")
    else:
        if audio_events != 0:
            contract_errors.append(
                f"generate_audio=false emitted {audio_events} native audio event(s)"
            )
        if not turn_ended_after_text:
            contract_errors.append("the Vietnamese text turn did not reach its final listen event")
        if args.mms_url and result.get("mms_tts", {}).get("audio_base64_chars", 0) <= 0:
            contract_errors.append("MMS-TTS returned no WAV payload")
    result["contract_passed"] = not contract_errors
    result["contract_errors"] = contract_errors
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-ws", default="ws://127.0.0.1:22500/backend")
    parser.add_argument("--backend-http", default="http://127.0.0.1:22500")
    parser.add_argument("--mms-url", default="http://127.0.0.1:18781/speak")
    parser.add_argument("--input-wav", type=Path)
    parser.add_argument("--input-text")
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--event-timeout", type=float, default=90.0)
    parser.add_argument("--drain-timeout", type=float, default=0.5)
    parser.add_argument(
        "--listen-prob-scale",
        type=float,
        default=0.0,
        help="Use 0 in verification to deterministically exercise the speak branch.",
    )
    parser.add_argument(
        "--system-prompt",
        default="Bạn là trợ lý tiếng Việt. Hãy trả lời thật ngắn gọn bằng tiếng Việt.",
    )
    args = parser.parse_args()
    result = asyncio.run(run_session(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["contract_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
