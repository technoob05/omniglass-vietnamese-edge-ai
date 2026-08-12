#!/usr/bin/env python3
"""Benchmark the running Vietnamese MMS-TTS HTTP adapter."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import platform
import statistics
import time
import wave
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np


DEFAULT_SENTENCES = (
    "Xin chào, tôi đang quan sát cảnh vật trước mặt bạn.",
    "Phía trước có một người đang đứng cạnh chiếc bàn.",
    "Chiếc chai nằm bên trái, cách bạn một khoảng ngắn.",
    "Tôi chưa đo được khoảng cách đáng tin cậy.",
    "Hãy dừng lại một chút để tôi nhìn rõ hơn.",
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wav_metadata(blob: bytes) -> tuple[int, int, float]:
    with wave.open(io.BytesIO(blob), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        duration = frames / float(sample_rate)
    return sample_rate, frames, duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18781/speak")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Warmup keeps model loading out of steady-state latency.
    post_json(args.url, {"text": "Xin chào bạn."}, args.timeout)

    runs: list[dict[str, object]] = []
    for repeat_index in range(args.repeat):
        for sentence_index, text in enumerate(DEFAULT_SENTENCES):
            started = time.perf_counter()
            response = post_json(args.url, {"text": text}, args.timeout)
            total_seconds = time.perf_counter() - started
            audio_blob = base64.b64decode(
                str(response["audio_wav_base64"]),
                validate=True,
            )
            sample_rate, frames, duration = wav_metadata(audio_blob)
            output_name = f"r{repeat_index:02d}_s{sentence_index:02d}.wav"
            output_path = args.output_dir / output_name
            output_path.write_bytes(audio_blob)
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "sentence_index": sentence_index,
                    "text": text,
                    # The service returns a complete WAV; wall time is therefore
                    # both response latency and time to first playable audio.
                    "first_audio_ms": round(total_seconds * 1000.0, 3),
                    "total_ms": round(total_seconds * 1000.0, 3),
                    "server_inference_ms": response.get("inference_ms"),
                    "audio_duration_seconds": round(duration, 3),
                    "rtf": round(total_seconds / duration, 4),
                    "sample_rate": sample_rate,
                    "samples": frames,
                    "wav": output_name,
                    "wav_sha256": hashlib.sha256(audio_blob).hexdigest(),
                }
            )

    first_values = [float(run["first_audio_ms"]) for run in runs]
    total_values = [float(run["total_ms"]) for run in runs]
    rtf_values = [float(run["rtf"]) for run in runs]
    server_values = [float(run["server_inference_ms"]) for run in runs]
    report = {
        "schema": "omniglass.vietnamese-tts-benchmark.v1",
        "engine": "facebook/mms-tts-vie",
        "backend": "pytorch-cuda-http-full-wav",
        "endpoint": args.url,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": runs,
        "summary": {
            "count": len(runs),
            "first_audio_ms_median": round(statistics.median(first_values), 3),
            "first_audio_ms_p95": round(percentile(first_values, 95), 3),
            "server_inference_ms_median": round(statistics.median(server_values), 3),
            "server_inference_ms_p95": round(percentile(server_values, 95), 3),
            "total_ms_median": round(statistics.median(total_values), 3),
            "total_ms_p95": round(percentile(total_values, 95), 3),
            "rtf_median": round(statistics.median(rtf_values), 4),
            "rtf_p95": round(percentile(rtf_values, 95), 4),
        },
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(report_path)


if __name__ == "__main__":
    main()
