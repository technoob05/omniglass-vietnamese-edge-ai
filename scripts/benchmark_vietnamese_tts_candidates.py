#!/usr/bin/env python3
"""Reproducible Vietnamese TTS latency and ASR-back intelligibility benchmark.

The ASR-back score is only a proxy: it measures the transcription error of a
fixed recognizer on synthetic speech, not listener naturalness or MOS.  The
script intentionally records both the synthesis measurement and this proxy so
they cannot be accidentally reported as the same metric.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import platform
import re
import statistics
import time
import unicodedata
import wave
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf


SENTENCES = (
    "Xin chào, tôi là trợ lý thị giác và đang quan sát khung cảnh trước mặt bạn.",
    "Phía trước có một người đang đứng cạnh chiếc bàn gỗ màu nâu.",
    "Chiếc chai nằm bên trái, nhưng tôi chưa đo được khoảng cách đáng tin cậy.",
    "Hãy dừng lại một chút để tôi nhìn rõ biển báo và kiểm tra lối đi.",
    "Cẩn thận, có một bậc thềm thấp ngay phía trước chân bạn.",
    "Bây giờ là 8 giờ 15 phút, pin còn 37 phần trăm.",
    "Địa chỉ là số 25 đường Nguyễn Thị Minh Khai, Thành phố Hồ Chí Minh.",
    "Camera AI đang chạy ở 24 khung hình mỗi giây trên thiết bị QCS8550.",
    "Bạn có muốn tôi tiếp tục theo dõi chiếc ba lô màu xanh không?",
    "Tôi nghe thấy tiếng xe máy ở bên phải; hãy chú ý trước khi sang đường.",
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^0-9a-zà-ỹđ]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def wav_blob(audio: np.ndarray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(output, np.asarray(audio, dtype=np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def asr_back(url: str, blob: bytes, timeout: float) -> tuple[str, float]:
    request = Request(url, data=blob, headers={"Content-Type": "audio/wav"}, method="POST")
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return str(result["text"]), (time.perf_counter() - started) * 1000.0


class VieNeuHTTP:
    name = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
    sample_rate = 48_000
    streaming = True

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout

    def synthesize(self, text: str) -> tuple[np.ndarray, float, float, int]:
        request = Request(
            self.url,
            data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        started = time.perf_counter()
        chunks: list[np.ndarray] = []
        first_audio_ms: float | None = None
        with urlopen(request, timeout=self.timeout) as response:
            for line in response:
                event = json.loads(line.decode("utf-8"))
                if event.get("type") == "audio":
                    if first_audio_ms is None:
                        first_audio_ms = (time.perf_counter() - started) * 1000.0
                    pcm = base64.b64decode(event["pcm_s16le_base64"], validate=True)
                    chunks.append(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
                elif event.get("type") == "error":
                    raise RuntimeError(str(event.get("message", "VieNeu stream failed")))
        total_ms = (time.perf_counter() - started) * 1000.0
        if not chunks or first_audio_ms is None:
            raise RuntimeError("VieNeu returned no audio")
        return np.concatenate(chunks), first_audio_ms, total_ms, len(chunks)


class KokoroVI:
    name = "anphunl/Kokoro-Vietnamese"
    sample_rate = 24_000
    streaming = False

    def __init__(self, device: str, voice: str) -> None:
        import torch
        import transformers
        import transformers.utils.import_utils as import_utils
        from huggingface_hub import hf_hub_download
        from kokoro_vietnamese._kokoro import KModel
        from kokoro_vietnamese.core import VOICES, merge_audio_chunks, phonemize, split_text

        # The pinned HF demo snapshot contains the inference core but no SDK
        # wrapper.  Mirror its minimal loading path without importing Gradio or
        # its ZeroGPU decorator.
        for flag in ("_torchvision_available", "_librosa_available", "_cv2_available"):
            if hasattr(import_utils, flag):
                setattr(import_utils, flag, False)
        if hasattr(import_utils, "_torchvision_version"):
            import_utils._torchvision_version = "N/A"
        self.torch = torch
        self.phonemize = phonemize
        self.split_text = split_text
        self.merge_audio_chunks = merge_audio_chunks
        repo_id = "anphunl/Kokoro-Vietnamese"
        config_path = hf_hub_download(repo_id, "config.json")
        model_path = hf_hub_download(repo_id, "kokoro_vi.pth")
        voice_path = hf_hub_download(repo_id, VOICES[voice]["filename"])
        self.model = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path).to(device).eval()
        self.voicepack = torch.load(voice_path, map_location="cpu", weights_only=True)

    def synthesize(self, text: str) -> tuple[np.ndarray, float, float, int]:
        started = time.perf_counter()
        parts: list[np.ndarray] = []
        for text_chunk in self.split_text(text):
            phonemes = self.phonemize(text_chunk)
            with self.torch.inference_mode():
                style = self.voicepack[len(phonemes) - 1]
                audio = self.model(phonemes, style, 1.0)
            parts.append(audio.detach().cpu().numpy())
        audio = self.merge_audio_chunks(parts, round(self.sample_rate * 0.05))
        total_ms = (time.perf_counter() - started) * 1000.0
        return np.asarray(audio, dtype=np.float32), total_ms, total_ms, 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("vieneu-http", "kokoro"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--voice", default="diem_trinh")
    parser.add_argument("--vieneu-url", default="http://127.0.0.1:18782/stream")
    parser.add_argument("--asr-url", default="http://127.0.0.1:18783/transcribe")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-asr-back", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_started = time.perf_counter()
    engine = (
        VieNeuHTTP(args.vieneu_url, args.timeout)
        if args.engine == "vieneu-http"
        else KokoroVI(args.device, args.voice)
    )
    load_seconds = time.perf_counter() - load_started

    # Warmup is deliberately excluded from steady-state results.
    engine.synthesize("Xin chào bạn.")
    runs: list[dict[str, object]] = []
    for repeat_index in range(args.repeat):
        for sentence_index, text in enumerate(SENTENCES):
            error: str | None = None
            try:
                audio, first_audio_ms, total_ms, chunks = engine.synthesize(text)
                duration = len(audio) / engine.sample_rate
                output_name = f"r{repeat_index:02d}_s{sentence_index:02d}.wav"
                output_path = args.output_dir / output_name
                blob = wav_blob(audio, engine.sample_rate)
                output_path.write_bytes(blob)
                transcript = None
                asr_ms = None
                cer = None
                wer = None
                cer_edits = None
                cer_reference_units = None
                wer_edits = None
                wer_reference_units = None
                if not args.skip_asr_back:
                    transcript, asr_ms = asr_back(args.asr_url, blob, args.timeout)
                    expected = normalize_vi(text)
                    observed = normalize_vi(transcript)
                    expected_chars = list(expected.replace(" ", ""))
                    observed_chars = list(observed.replace(" ", ""))
                    expected_words = expected.split()
                    observed_words = observed.split()
                    cer_edits = edit_distance(expected_chars, observed_chars)
                    cer_reference_units = len(expected_chars)
                    wer_edits = edit_distance(expected_words, observed_words)
                    wer_reference_units = len(expected_words)
                    cer = cer_edits / max(1, cer_reference_units)
                    wer = wer_edits / max(1, wer_reference_units)
                runs.append(
                    {
                        "repeat_index": repeat_index,
                        "sentence_index": sentence_index,
                        "text": text,
                        "first_audio_ms": round(first_audio_ms, 3),
                        "total_ms": round(total_ms, 3),
                        "audio_duration_seconds": round(duration, 3),
                        "rtf": round(total_ms / 1000.0 / duration, 5),
                        "chunks": chunks,
                        "sample_rate": engine.sample_rate,
                        "asr_back_text": transcript,
                        "asr_back_ms": None if asr_ms is None else round(asr_ms, 3),
                        "asr_back_cer": None if cer is None else round(cer, 6),
                        "asr_back_wer": None if wer is None else round(wer, 6),
                        "asr_back_cer_edits": cer_edits,
                        "asr_back_cer_reference_units": cer_reference_units,
                        "asr_back_wer_edits": wer_edits,
                        "asr_back_wer_reference_units": wer_reference_units,
                        "wav": output_name,
                        "wav_sha256": hashlib.sha256(blob).hexdigest(),
                        "error": error,
                    }
                )
            except Exception as exc:  # Continue the fixed suite and report failures.
                runs.append(
                    {
                        "repeat_index": repeat_index,
                        "sentence_index": sentence_index,
                        "text": text,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    successful = [run for run in runs if run.get("error") is None]
    if not successful:
        raise RuntimeError("All synthesis cases failed")
    summary: dict[str, object] = {
        "count": len(runs),
        "successful": len(successful),
        "errors": len(runs) - len(successful),
    }
    for key in ("first_audio_ms", "total_ms", "rtf", "asr_back_cer", "asr_back_wer"):
        values = [float(run[key]) for run in successful if run.get(key) is not None]
        if values:
            summary[f"{key}_median"] = round(statistics.median(values), 5)
            summary[f"{key}_p95"] = round(percentile(values, 95), 5)
    for metric in ("cer", "wer"):
        edits = sum(int(run[f"asr_back_{metric}_edits"]) for run in successful if run.get(f"asr_back_{metric}_edits") is not None)
        units = sum(int(run[f"asr_back_{metric}_reference_units"]) for run in successful if run.get(f"asr_back_{metric}_reference_units") is not None)
        if units:
            summary[f"asr_back_{metric}_micro"] = round(edits / units, 6)

    report = {
        "schema": "omniglass.vietnamese-tts-candidate-benchmark.v1",
        "engine": engine.name,
        "backend": f"{args.engine}-{args.device}",
        "streaming": engine.streaming,
        "load_seconds": round(load_seconds, 3),
        "voice": args.voice if args.engine == "kokoro" else "service-default",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "measurement_note": "ASR-back CER/WER is an intelligibility proxy, not MOS or human preference.",
        "runs": runs,
        "summary": summary,
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(report_path)


if __name__ == "__main__":
    main()
