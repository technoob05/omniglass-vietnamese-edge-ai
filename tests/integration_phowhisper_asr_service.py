#!/usr/bin/env python3
"""Exercise HTTP plus a persistent two-utterance WebSocket session."""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import websockets


def load_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = np.gcd(int(rate), 16000)
        audio = scipy.signal.resample_poly(audio, 16000 // divisor, int(rate) // divisor)
    return np.asarray(audio, dtype=np.float32)


def load_native_session(path: Path, max_seconds: float) -> np.ndarray:
    chunks: list[np.ndarray] = []
    remaining = int(max_seconds * 16000)
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        frame = event.get("frame") or {}
        ref = (frame.get("input") or {}).get("audio")
        if event.get("dir") != "up" or frame.get("type") != "input.append" or not str(ref).startswith("@blob/"):
            continue
        audio = load_audio(path.parent / str(ref).removeprefix("@"))[:remaining]
        chunks.append(audio)
        remaining -= len(audio)
        if remaining <= 0:
            break
    if not chunks:
        raise RuntimeError(f"No input audio blobs in {path}")
    return np.concatenate(chunks)


def float_to_pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).round().astype("<i2").tobytes()


def http_post_wav(base_url: str, path: Path) -> dict:
    request = urllib.request.Request(
        f"{base_url}/transcribe",
        data=path.read_bytes(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    payload["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


def http_post_pcm(base_url: str, audio: np.ndarray) -> dict:
    request = urllib.request.Request(
        f"{base_url}/transcribe?sample_rate=16000&channels=1",
        data=float_to_pcm(audio),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)
    payload["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload


async def send_utterance(websocket, audio: np.ndarray, sequence: int, timestamp_ms: float) -> tuple[int, float, list[dict]]:
    # One second of silence lets Silero close the utterance naturally.
    audio = np.concatenate((audio, np.zeros(16000, dtype=np.float32)))
    pcm = float_to_pcm(audio)
    chunk_bytes = 1600 * 2  # 100 ms
    events: list[dict] = []
    final: dict | None = None
    started = time.perf_counter()
    for offset in range(0, len(pcm), chunk_bytes):
        chunk = pcm[offset : offset + chunk_bytes]
        event = {
            "type": "audio.chunk",
            "sequence": sequence,
            "timestamp_ms": timestamp_ms,
            "sample_rate": 16000,
            "channels": 1,
            "encoding": "pcm_s16le",
            "pcm_s16le_base64": base64.b64encode(chunk).decode("ascii"),
        }
        await websocket.send(json.dumps(event))
        while True:
            received = json.loads(await asyncio.wait_for(websocket.recv(), timeout=90))
            events.append(received)
            if received.get("type") == "asr.final":
                final = received
            if received.get("type") == "asr.ack" and received.get("sequence") == sequence:
                break
        sequence += 1
        timestamp_ms += len(chunk) / 2 / 16000 * 1000
        if final is not None:
            break
    if final is None:
        await websocket.send(json.dumps({"type": "audio.end"}))
        while final is None:
            received = json.loads(await asyncio.wait_for(websocket.recv(), timeout=90))
            events.append(received)
            if received.get("type") == "asr.final":
                final = received
            elif received.get("type") == "asr.no_speech":
                raise RuntimeError("VAD found no speech")
    assert final["immutable"] is True
    assert final["routing"]["vlm_eligible"] is True
    assert all(event.get("vlm_eligible") is False for event in events if event.get("type") == "asr.partial")
    final["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return sequence, timestamp_ms + 500, events


async def main_async(args) -> None:
    with urllib.request.urlopen(f"{args.base_url}/health", timeout=5) as response:
        health = json.load(response)
    native_audio = load_native_session(args.session_jsonl, args.session_max_seconds)
    samples = [
        {"name": args.wav.stem, "audio": load_audio(args.wav), "http_wav": await asyncio.to_thread(http_post_wav, args.base_url, args.wav)},
        {
            "name": args.session_jsonl.parent.name,
            "audio": native_audio,
            "http_pcm": await asyncio.to_thread(http_post_pcm, args.base_url, native_audio),
        },
    ]
    ws_url = args.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/asr?partial_interval_ms=1000"
    sequence = 0
    timestamp_ms = 1_000_000.0
    final_ids: list[str] = []
    async with websockets.connect(ws_url, max_size=4 * 1024 * 1024, ping_interval=20) as websocket:
        ready = json.loads(await websocket.recv())
        assert ready["type"] == "asr.ready"
        for sample in samples:
            sequence, timestamp_ms, events = await send_utterance(websocket, sample["audio"], sequence, timestamp_ms)
            final = next(event for event in events if event["type"] == "asr.final")
            final_ids.append(final["final_id"])
            sample["websocket"] = final
            sample["websocket_event_counts"] = dict(collections.Counter(event["type"] for event in events))
            sample.pop("audio")
    assert len(final_ids) == len(set(final_ids)), "final_id must be unique per utterance"
    result = {
        "health": health,
        "samples": samples,
        "contract": {
            "persistent_two_turn_session": True,
            "unique_final_ids": True,
            "only_final_for_vlm": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18783")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--session-jsonl", type=Path, required=True)
    parser.add_argument("--session-max-seconds", type=float, default=12.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
