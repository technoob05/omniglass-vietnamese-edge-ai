#!/usr/bin/env python3
"""Evaluate a warm PhoWhisper checkpoint on a revision-locked ASR manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    normalized = []
    for character in text:
        category = unicodedata.category(character)
        normalized.append(character if category[0] in {"L", "N"} else " ")
    return " ".join("".join(normalized).split())


def levenshtein(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 4)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16_000:
        divisor = np.gcd(sample_rate, 16_000)
        audio = scipy.signal.resample_poly(
            audio, 16_000 // divisor, sample_rate // divisor
        ).astype(np.float32)
        sample_rate = 16_000
    return np.asarray(audio, dtype=np.float32), sample_rate


def transcribe(
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    audio: np.ndarray,
    sample_rate: int,
    max_new_tokens: int,
) -> tuple[str, float]:
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    features = inputs.input_features.to(device="cuda", dtype=torch.float16)
    attention_mask = inputs.attention_mask.to(device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            features,
            attention_mask=attention_mask,
            language="vi",
            task="transcribe",
            max_new_tokens=max_new_tokens,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip(), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="vinai/PhoWhisper-medium")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--warmup-count", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = list(manifest.get("samples") or [])
    if not 30 <= len(samples) <= 100:
        raise RuntimeError(f"Manifest must contain 30-100 samples, found {len(samples)}")

    gpu_free_before, gpu_total = torch.cuda.mem_get_info()
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
    gpu_free_after, _ = torch.cuda.mem_get_info()

    warmup: list[float] = []
    first_audio, first_rate = load_audio(Path(samples[0]["audio_path"]))
    for _ in range(args.warmup_count):
        _, elapsed = transcribe(
            model, processor, first_audio, first_rate, args.max_new_tokens
        )
        warmup.append(elapsed)

    results: list[dict[str, Any]] = []
    word_edits = word_total = 0
    character_edits = character_total = 0
    latency_values: list[float] = []
    rtf_values: list[float] = []
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    for index, sample in enumerate(samples):
        path = Path(sample["audio_path"])
        audio, sample_rate = load_audio(path)
        duration = len(audio) / sample_rate
        hypothesis, elapsed = transcribe(
            model, processor, audio, sample_rate, args.max_new_tokens
        )
        reference_normalized = normalize_vi(str(sample["raw_transcription"]))
        hypothesis_normalized = normalize_vi(hypothesis)
        ref_words = reference_normalized.split()
        hyp_words = hypothesis_normalized.split()
        ref_characters = list(reference_normalized.replace(" ", ""))
        hyp_characters = list(hypothesis_normalized.replace(" ", ""))
        sample_word_edits = levenshtein(ref_words, hyp_words)
        sample_character_edits = levenshtein(ref_characters, hyp_characters)
        word_edits += sample_word_edits
        word_total += len(ref_words)
        character_edits += sample_character_edits
        character_total += len(ref_characters)
        latency_values.append(elapsed)
        rtf_values.append(elapsed / duration)
        total_audio_seconds += duration
        total_inference_seconds += elapsed
        result = {
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "sentence_id": sample["sentence_id"],
            "filename": sample["filename"],
            "gender": sample["gender"],
            "duration_seconds": round(duration, 4),
            "reference": sample["raw_transcription"],
            "reference_normalized": reference_normalized,
            "hypothesis": hypothesis,
            "hypothesis_normalized": hypothesis_normalized,
            "word_edits": sample_word_edits,
            "reference_words": len(ref_words),
            "wer": round(sample_word_edits / max(1, len(ref_words)), 6),
            "character_edits": sample_character_edits,
            "reference_characters_no_space": len(ref_characters),
            "cer": round(sample_character_edits / max(1, len(ref_characters)), 6),
            "latency_seconds": round(elapsed, 4),
            "rtf": round(elapsed / duration, 4),
        }
        results.append(result)
        print(
            f"[{index + 1:03d}/{len(samples):03d}] "
            f"wer={result['wer']:.3f} cer={result['cer']:.3f} "
            f"latency={elapsed:.3f}s rtf={result['rtf']:.3f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "local_dir": str(args.model_dir),
            "dtype": "float16",
            "load_seconds": round(load_seconds, 4),
            "persistent_for_all_samples": True,
            "warmup_count": args.warmup_count,
            "warmup_seconds": [round(value, 4) for value in warmup],
        },
        "dataset": {
            **manifest["dataset"],
            "manifest_path": str(args.manifest),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "selection": manifest["selection"],
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_total_gib": round(gpu_total / 2**30, 3),
            "gpu_free_before_load_gib": round(gpu_free_before / 2**30, 3),
            "gpu_free_after_load_gib": round(gpu_free_after / 2**30, 3),
        },
        "normalization": {
            "unicode": "NFC",
            "case": "casefold",
            "kept": "Unicode letters and numbers",
            "replaced_with_space": "punctuation, symbols, separators and controls",
            "whitespace": "collapsed",
            "cer_spaces": "excluded",
            "diacritics": "preserved",
        },
        "aggregate": {
            "samples": len(results),
            "unique_sentence_ids": len({row["sentence_id"] for row in results}),
            "gender_counts": dict(Counter(str(row["gender"]) for row in results)),
            "accent_breakdown": None,
            "accent_note": "FLEURS vi_vn exposes no regional-accent metadata.",
            "word_edits": word_edits,
            "reference_words": word_total,
            "normalized_wer": round(word_edits / word_total, 6),
            "character_edits": character_edits,
            "reference_characters_no_space": character_total,
            "normalized_cer": round(character_edits / character_total, 6),
            "total_audio_seconds": round(total_audio_seconds, 4),
            "total_inference_seconds": round(total_inference_seconds, 4),
            "throughput_rtf": round(total_inference_seconds / total_audio_seconds, 6),
            "latency_seconds": {
                "mean": round(statistics.mean(latency_values), 4),
                "median": round(statistics.median(latency_values), 4),
                "p95_nearest_rank": percentile(latency_values, 0.95),
                "max": round(max(latency_values), 4),
            },
            "per_utterance_rtf": {
                "mean": round(statistics.mean(rtf_values), 4),
                "median": round(statistics.median(rtf_values), 4),
                "p95_nearest_rank": percentile(rtf_values, 0.95),
                "max": round(max(rtf_values), 4),
            },
        },
        "scope": {
            "quality_claim": "small deterministic FLEURS read-speech smoke benchmark",
            "not_measured": [
                "regional Vietnamese accents",
                "spontaneous assistive commands",
                "glasses microphone acoustics",
                "noise robustness",
                "true streaming or endpoint latency",
            ],
        },
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions_jsonl:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
