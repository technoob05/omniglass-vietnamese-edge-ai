#!/usr/bin/env python3
"""Benchmark VieNeu-TTS v3 Turbo's native streaming path.

The script deliberately runs the CPU/ONNX backend so its measurements remain
independent from the H100 that serves MiniCPM-o.  It records the first-chunk
latency, total synthesis time, audio duration, real-time factor, and a WAV for
each sentence so latency and Vietnamese intelligibility can be reviewed
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vieneu import Vieneu


DEFAULT_SENTENCES = (
    "Xin chào, tôi đang quan sát cảnh vật trước mặt bạn.",
    "Phía trước có một người đang đứng cạnh chiếc bàn.",
    "Chiếc chai nằm bên trái, cách bạn một khoảng ngắn.",
    "Tôi chưa đo được khoảng cách đáng tin cậy.",
    "Hãy dừng lại một chút để tôi nhìn rõ hơn.",
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--backend", default=None, choices=("onnx", "pytorch"))
    parser.add_argument("--precision", default="int8", choices=("int8", "fp32"))
    parser.add_argument("--voice", default=None)
    parser.add_argument("--warmup-text", default="Xin chào bạn.")
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    backend = args.backend or ("pytorch" if args.device == "cuda" else "onnx")
    tts = Vieneu(
        mode="v3turbo",
        device=args.device,
        backend=backend,
        precision=args.precision,
        threads=args.threads,
    )
    load_seconds = time.perf_counter() - load_started

    # Warm every graph and the phonemizer before recording latency.
    list(
        tts.infer_stream(
            args.warmup_text,
            voice=args.voice,
            apply_watermark=False,
        )
    )

    runs: list[dict[str, object]] = []
    for repeat_index in range(args.repeat):
        for sentence_index, text in enumerate(DEFAULT_SENTENCES):
            started = time.perf_counter()
            first_chunk_seconds: float | None = None
            chunks: list[np.ndarray] = []
            for chunk in tts.infer_stream(
                text,
                voice=args.voice,
                apply_watermark=False,
            ):
                now = time.perf_counter()
                if first_chunk_seconds is None:
                    first_chunk_seconds = now - started
                chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

            total_seconds = time.perf_counter() - started
            if not chunks or first_chunk_seconds is None:
                raise RuntimeError(f"VieNeu returned no audio for: {text!r}")

            audio = np.concatenate(chunks)
            audio_duration_seconds = len(audio) / float(tts.sample_rate)
            output_name = f"r{repeat_index:02d}_s{sentence_index:02d}.wav"
            output_path = args.output_dir / output_name
            sf.write(output_path, audio, tts.sample_rate, subtype="PCM_16")
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "sentence_index": sentence_index,
                    "text": text,
                    "first_chunk_ms": round(first_chunk_seconds * 1000.0, 3),
                    "total_ms": round(total_seconds * 1000.0, 3),
                    "audio_duration_seconds": round(audio_duration_seconds, 3),
                    "rtf": round(total_seconds / audio_duration_seconds, 4),
                    "chunks": len(chunks),
                    "sample_rate": int(tts.sample_rate),
                    "samples": int(len(audio)),
                    "wav": output_name,
                    "wav_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                }
            )

    first_chunk_values = [float(run["first_chunk_ms"]) for run in runs]
    total_values = [float(run["total_ms"]) for run in runs]
    rtf_values = [float(run["rtf"]) for run in runs]
    report = {
        "schema": "omniglass.vietnamese-tts-benchmark.v1",
        "engine": "VieNeu-TTS-v3-Turbo",
        "backend": f"{backend}-{args.precision}-{args.device}",
        "load_seconds": round(load_seconds, 3),
        "threads": args.threads,
        "voice": args.voice or "default",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": runs,
        "summary": {
            "count": len(runs),
            "first_chunk_ms_median": round(statistics.median(first_chunk_values), 3),
            "first_chunk_ms_p95": round(percentile(first_chunk_values, 95), 3),
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
