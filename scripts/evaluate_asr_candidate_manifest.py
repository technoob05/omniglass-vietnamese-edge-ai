#!/usr/bin/env python3
"""Evaluate one ASR candidate on the frozen OmniGlass Vietnamese manifest.

The runner deliberately uses batch size one and times only warm inference.  It
supports Qwen3-ASR's official package and Whisper-compatible Transformers
checkpoints while keeping the text normalization identical across candidates.
"""

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
from typing import Any, Callable, Sequence

import numpy as np


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    characters = []
    for character in text:
        category = unicodedata.category(character)
        characters.append(character if category[0] in {"L", "N"} else " ")
    return " ".join("".join(characters).split())


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
    import scipy.signal
    import soundfile as sf

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


def _load_qwen(
    model_dir: Path,
    max_new_tokens: int,
) -> tuple[Callable[[np.ndarray, int], str], dict[str, Any]]:
    import torch
    import qwen_asr
    from qwen_asr import Qwen3ASRModel

    model = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
    )

    def transcribe(audio: np.ndarray, sample_rate: int) -> str:
        result = model.transcribe(
            audio=(audio, sample_rate),
            language="Vietnamese",
        )[0]
        return str(result.text).strip()

    return transcribe, {
        "package": "qwen-asr",
        "package_version": getattr(qwen_asr, "__version__", "unknown"),
        "dtype": "bfloat16",
        "language_argument": "Vietnamese",
        "batch_size": 1,
    }


def _load_whisper(
    model_dir: Path,
    max_new_tokens: int,
) -> tuple[Callable[[np.ndarray, int], str], dict[str, Any]]:
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    def transcribe(audio: np.ndarray, sample_rate: int) -> str:
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs.input_features.to(device="cuda", dtype=torch.float16)
        attention_mask = inputs.attention_mask.to(device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                features,
                attention_mask=attention_mask,
                language="vi",
                task="transcribe",
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return transcribe, {
        "package": "transformers",
        "package_version": transformers.__version__,
        "dtype": "float16",
        "language_argument": "vi",
        "batch_size": 1,
    }


def validate_manifest(manifest: dict[str, Any], minimum_samples: int) -> list[dict[str, Any]]:
    samples = list(manifest.get("samples") or [])
    if len(samples) < minimum_samples:
        raise RuntimeError(
            f"Manifest requires at least {minimum_samples} samples, found {len(samples)}"
        )
    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Manifest contains duplicate sample IDs")
    for sample in samples:
        path = Path(sample["audio_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = sample.get("audio_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"Audio hash mismatch: {path}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["qwen3_asr", "whisper"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-license", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--minimum-free-vram-gib", type=float, default=5.0)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparable H100 benchmark")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = validate_manifest(manifest, args.minimum_samples)

    gpu_free_before, gpu_total = torch.cuda.mem_get_info()
    free_gib = gpu_free_before / 2**30
    if free_gib < args.minimum_free_vram_gib:
        raise RuntimeError(
            f"Refusing model load: free VRAM {free_gib:.3f} GiB is below "
            f"{args.minimum_free_vram_gib:.3f} GiB; production services remain untouched"
        )

    loader = _load_qwen if args.backend == "qwen3_asr" else _load_whisper
    load_started = time.perf_counter()
    transcribe, backend_metadata = loader(args.model_dir, args.max_new_tokens)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    gpu_free_after, _ = torch.cuda.mem_get_info()

    first_audio, first_rate = load_audio(Path(samples[0]["audio_path"]))
    warmup_seconds = []
    for _ in range(args.warmup_count):
        torch.cuda.synchronize()
        started = time.perf_counter()
        transcribe(first_audio, first_rate)
        torch.cuda.synchronize()
        warmup_seconds.append(time.perf_counter() - started)

    results: list[dict[str, Any]] = []
    word_edits = word_total = character_edits = character_total = 0
    latency_values: list[float] = []
    rtf_values: list[float] = []
    total_audio_seconds = total_inference_seconds = 0.0
    language_detection_failures = 0

    for index, sample in enumerate(samples):
        audio, sample_rate = load_audio(Path(sample["audio_path"]))
        duration = len(audio) / sample_rate
        torch.cuda.synchronize()
        started = time.perf_counter()
        hypothesis = transcribe(audio, sample_rate)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        reference_normalized = normalize_vi(str(sample["raw_transcription"]))
        hypothesis_normalized = normalize_vi(hypothesis)
        ref_words, hyp_words = reference_normalized.split(), hypothesis_normalized.split()
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
        row = {
            "sample_index": index,
            "sample_id": sample["sample_id"],
            "sentence_id": sample["sentence_id"],
            "filename": sample["filename"],
            "gender": sample.get("gender"),
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
        results.append(row)
        print(
            f"[{index + 1:03d}/{len(samples):03d}] wer={row['wer']:.3f} "
            f"cer={row['cer']:.3f} latency={elapsed:.3f}s rtf={row['rtf']:.3f}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "local_dir": str(args.model_dir),
            "license": args.model_license,
            "backend": args.backend,
            "load_seconds": round(load_seconds, 4),
            "persistent_for_all_samples": True,
            "warmup_count": args.warmup_count,
            "warmup_seconds": [round(value, 4) for value in warmup_seconds],
            **backend_metadata,
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
            "language_detection_failures": language_detection_failures,
        },
        "scope": {
            "quality_claim": "same deterministic FLEURS Vietnamese read-speech smoke benchmark",
            "not_measured": [
                "regional Vietnamese accents",
                "spontaneous assistive commands",
                "glasses microphone acoustics",
                "noise robustness",
                "true streaming or endpoint latency",
                "QCS8550 execution, power or thermal behavior",
            ],
        },
        "samples": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.predictions_jsonl:
        args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_jsonl.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
