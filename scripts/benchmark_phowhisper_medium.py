#!/usr/bin/env python3
"""Reproducible PhoWhisper-medium cold/warm benchmark for the H100 pod."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == 16000:
        return audio
    divisor = np.gcd(sample_rate, 16000)
    return scipy.signal.resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)


def load_session_input(stream_jsonl: Path, max_seconds: float) -> tuple[np.ndarray, int, list[str]]:
    session_dir = stream_jsonl.parent
    paths: list[Path] = []
    for line in stream_jsonl.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("dir") != "up":
            continue
        frame = record.get("frame") or {}
        if frame.get("type") != "input.append":
            continue
        audio_ref = (frame.get("input") or {}).get("audio")
        if isinstance(audio_ref, str) and audio_ref.startswith("@blob/"):
            paths.append(session_dir / audio_ref.removeprefix("@"))

    chunks: list[np.ndarray] = []
    used_paths: list[str] = []
    total = 0
    limit = int(max_seconds * 16000)
    for path in paths:
        if not path.is_file() or total >= limit:
            continue
        audio, rate = load_wav(path)
        audio = resample_16k(audio, rate)
        remaining = limit - total
        chunks.append(audio[:remaining])
        used_paths.append(str(path))
        total += min(len(audio), remaining)
    if not chunks:
        raise RuntimeError(f"No valid upstream audio blobs found in {stream_jsonl}")
    return np.concatenate(chunks), 16000, used_paths


def gpu_snapshot() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "free_gib": round(free / 2**30, 3),
        "total_gib": round(total / 2**30, 3),
        "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


def transcribe_once(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    audio: np.ndarray,
    sample_rate: int,
) -> tuple[str, float, dict[str, float]]:
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device="cuda", dtype=torch.float16)
    attention_mask = inputs.attention_mask.to(device="cuda")
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="vi",
            task="transcribe",
            max_new_tokens=224,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    memory = {
        "baseline_allocated_gib": round(baseline_allocated / 2**30, 3),
        "baseline_reserved_gib": round(baseline_reserved / 2**30, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }
    return text, elapsed, memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="vinai/PhoWhisper-medium")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--wav", action="append", type=Path, default=[])
    parser.add_argument("--session-jsonl", action="append", type=Path, default=[])
    parser.add_argument("--session-max-seconds", type=float, default=12.0)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    samples: list[dict[str, Any]] = []
    for path in args.wav:
        audio, rate = load_wav(path)
        audio = resample_16k(audio, rate)
        samples.append({"name": path.stem, "source": str(path), "audio": audio, "rate": 16000})
    for path in args.session_jsonl:
        audio, rate, blobs = load_session_input(path, args.session_max_seconds)
        samples.append({
            "name": path.parent.name + "_input",
            "source": str(path),
            "audio": audio,
            "rate": rate,
            "source_blobs": blobs,
        })
    if not samples:
        raise RuntimeError("At least one --wav or --session-jsonl is required")

    started_wall = datetime.now(timezone.utc).isoformat()
    gpu_before = gpu_snapshot()
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    gpu_after_load = gpu_snapshot()

    sample_results: list[dict[str, Any]] = []
    for sample in samples:
        duration = len(sample["audio"]) / sample["rate"]
        runs: list[dict[str, Any]] = []
        for index in range(args.warm_runs + 1):
            transcript, elapsed, memory = transcribe_once(
                model,
                processor,
                sample["audio"],
                sample["rate"],
            )
            runs.append({
                "index": index,
                "phase": "first" if index == 0 else "warm",
                "seconds": round(elapsed, 4),
                "rtf": round(elapsed / duration, 4),
                "transcript": transcript,
                "memory": memory,
            })
        warm_seconds = [run["seconds"] for run in runs[1:]]
        warm_rtfs = [run["rtf"] for run in runs[1:]]
        sample_results.append({
            "name": sample["name"],
            "source": sample["source"],
            "source_blobs": sample.get("source_blobs"),
            "duration_seconds": round(duration, 4),
            "runs": runs,
            "warm_summary": {
                "median_seconds": round(statistics.median(warm_seconds), 4),
                "mean_seconds": round(statistics.mean(warm_seconds), 4),
                "median_rtf": round(statistics.median(warm_rtfs), 4),
                "mean_rtf": round(statistics.mean(warm_rtfs), 4),
            },
        })

    result = {
        "schema_version": 1,
        "started_at_utc": started_wall,
        "model": {
            "id": args.model_id,
            "revision": args.revision,
            "local_dir": str(args.model_dir),
            "dtype": "float16",
            "load_seconds": round(load_seconds, 4),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpu_before": gpu_before,
        "gpu_after_load": gpu_after_load,
        "samples": sample_results,
        "streaming_semantics": {
            "native_true_streaming": False,
            "mode": "utterance/file transcription",
            "note": "Use an external VAD/endpointing layer for pseudo-streaming; each finalized utterance is decoded independently.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
