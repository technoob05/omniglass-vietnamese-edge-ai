#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import time
import wave

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, VitsModel


LOGGER = logging.getLogger("openglass.vi_tts")
MODEL_ID = "facebook/mms-tts-vie"


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)


def clean_vietnamese(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_`#|]", " ", text)
    text = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ.,;:!?%\-–—()/'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_app() -> FastAPI:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID).eval().to("cuda")
    loaded_seconds = time.perf_counter() - started
    sample_rate = int(model.config.sampling_rate)
    LOGGER.info(
        "loaded model=%s seconds=%.2f sample_rate=%s gpu=%s",
        MODEL_ID,
        loaded_seconds,
        sample_rate,
        torch.cuda.get_device_name(),
    )

    app = FastAPI(title="OpenGlass Vietnamese TTS", version="1.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ready",
            "model": MODEL_ID,
            "sample_rate": sample_rate,
            "gpu": torch.cuda.get_device_name(),
        }

    @app.post("/speak")
    def speak(request: SpeakRequest) -> dict:
        spoken = clean_vietnamese(request.text)
        if not spoken:
            raise HTTPException(status_code=422, detail="Không có nội dung để đọc")

        inputs = tokenizer(spoken, return_tensors="pt").to("cuda")
        started_tts = time.perf_counter()
        with torch.inference_mode():
            waveform = model(**inputs).waveform[0].float().cpu()
        elapsed = time.perf_counter() - started_tts
        if not torch.isfinite(waveform).all() or float(waveform.abs().max()) <= 0:
            raise HTTPException(status_code=500, detail="Waveform không hợp lệ")

        pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)

        duration = len(pcm) / 2 / sample_rate
        LOGGER.info(
            "speak chars=%s inference_ms=%.1f audio_seconds=%.2f",
            len(spoken),
            elapsed * 1000,
            duration,
        )
        return {
            "text": spoken,
            "audio_wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "inference_ms": round(elapsed * 1000, 2),
            "model": MODEL_ID,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18781)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
