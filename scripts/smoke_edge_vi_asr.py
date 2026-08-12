#!/usr/bin/env python3
"""Verify edge-ASR artifacts and run the sherpa-onnx CPU smoke baseline."""

from __future__ import annotations

import argparse
import hashlib
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


BASELINE_ID = "sherpa-onnx-zipformer-vi-30m-int8-cpu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported manifest schema")
    matches = [item for item in manifest["baselines"] if item["id"] == BASELINE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {BASELINE_ID} baseline")
    return manifest, matches[0]


def verify_model_files(model_dir: Path, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for artifact in baseline["artifacts"]:
        path = model_dir / artifact["path"]
        actual_size = path.stat().st_size if path.is_file() else None
        actual_hash = sha256_file(path) if path.is_file() else None
        ok = actual_size == artifact["bytes"] and actual_hash == artifact["sha256"]
        checks.append({
            "path": str(path),
            "expected_bytes": artifact["bytes"],
            "actual_bytes": actual_size,
            "expected_sha256": artifact["sha256"],
            "actual_sha256": actual_hash,
            "ok": ok,
        })
    if not all(item["ok"] for item in checks):
        raise RuntimeError("One or more model artifacts failed size/SHA-256 verification")
    return checks


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = np.gcd(int(sample_rate), 16000)
        audio = scipy.signal.resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor)
        sample_rate = 16000
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def load_native_session(path: Path, max_seconds: float) -> tuple[np.ndarray, int, list[str]]:
    chunks: list[np.ndarray] = []
    sources: list[str] = []
    remaining = int(max_seconds * 16000)
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        frame = event.get("frame") or {}
        ref = (frame.get("input") or {}).get("audio")
        if event.get("dir") != "up" or frame.get("type") != "input.append" or not str(ref).startswith("@blob/"):
            continue
        wav = path.parent / str(ref).removeprefix("@")
        audio, _ = load_wav(wav)
        audio = audio[:remaining]
        chunks.append(audio)
        sources.append(str(wav))
        remaining -= len(audio)
        if remaining <= 0:
            break
    if not chunks:
        raise RuntimeError(f"No native input audio found in {path}")
    return np.concatenate(chunks), 16000, sources


def peak_rss_mib() -> float:
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        divisor = 1024 if platform.system() != "Darwin" else 1024 * 1024
        return round(value / divisor, 3)
    except ImportError:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wav", type=Path, action="append", default=[])
    parser.add_argument("--session-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--session-max-seconds", type=float, default=12.0)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest, baseline = load_manifest(args.manifest)
    model_dir = args.asset_root / baseline["download"]["directory"]
    checks = verify_model_files(model_dir, baseline)
    base_result: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "baseline_id": BASELINE_ID,
        "artifact_checks": checks,
        "cpu_only": True,
    }
    if args.verify_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(base_result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(base_result, ensure_ascii=False, indent=2))
        return

    # Set before importing sherpa-onnx so this harness cannot claim/use CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import sherpa_onnx

    samples: list[dict[str, Any]] = []
    for wav in sorted((model_dir / "test_wavs").glob("*.wav")) + args.wav:
        audio, rate = load_wav(wav)
        samples.append({"name": wav.stem, "source": str(wav), "audio": audio, "rate": rate, "source_blobs": None})
    for session in args.session_jsonl:
        audio, rate, blobs = load_native_session(session, args.session_max_seconds)
        samples.append({"name": session.parent.name, "source": str(session), "audio": audio, "rate": rate, "source_blobs": blobs})

    load_started = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        dither=0.0,
        decoding_method="greedy_search",
        provider="cpu",
    )
    load_seconds = time.perf_counter() - load_started

    results: list[dict[str, Any]] = []
    for sample in samples:
        duration = len(sample["audio"]) / sample["rate"]
        runs: list[dict[str, Any]] = []
        for index in range(args.warm_runs + 1):
            stream = recognizer.create_stream()
            stream.accept_waveform(sample["rate"], sample["audio"])
            started = time.perf_counter()
            recognizer.decode_stream(stream)
            elapsed = time.perf_counter() - started
            runs.append({
                "phase": "first" if index == 0 else "warm",
                "seconds": round(elapsed, 6),
                "rtf": round(elapsed / duration, 6),
                "transcript": stream.result.text.strip(),
            })
        warm = runs[1:]
        results.append({
            "name": sample["name"],
            "source": sample["source"],
            "source_blobs": sample["source_blobs"],
            "duration_seconds": round(duration, 4),
            "runs": runs,
            "warm_median_seconds": round(statistics.median(item["seconds"] for item in warm), 6),
            "warm_median_rtf": round(statistics.median(item["rtf"] for item in warm), 6),
        })

    base_result.update({
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "sherpa_onnx": sherpa_onnx.__version__,
            "provider": "cpu",
            "num_threads": 1,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "model_load_seconds": round(load_seconds, 6),
        "peak_rss_mib": peak_rss_mib(),
        "samples": results,
        "claim_boundary": {
            "verified_here": "Linux x86_64 CPU artifact integrity and qualitative transcription smoke",
            "not_verified_here": ["WER/CER", "Linux ARM64", "QCS8550", "QCS6490", "QNN/HTP", "power", "thermal behavior"],
        },
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base_result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(base_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
