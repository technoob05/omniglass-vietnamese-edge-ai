#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import re
import threading
import time
import wave

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    VitsModel,
)

LOGGER = logging.getLogger("omniglass.vlm_service")
CJK_PATTERN = re.compile(
    r"[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u4dbf"
    r"\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)
VIETNAMESE_MARKS_PATTERN = re.compile(
    r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệ"
    r"ìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]",
    flags=re.IGNORECASE,
)
VIETNAMESE_COMMON_WORDS = {
    "ảnh", "bạn", "bên", "cảm", "chào", "có", "của", "đang", "đây", "giữa", "không",
    "một", "người", "này", "ơn", "phía", "thấy", "tôi", "trái", "trên", "và", "vật", "xin",
}


def contains_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def strip_cjk(text: str) -> str:
    cleaned = CJK_PATTERN.sub("", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s+", " ", cleaned)).strip()


def looks_vietnamese(text: str) -> bool:
    words = set(re.findall(r"[A-Za-zÀ-ỹĐđ]+", text.casefold()))
    common_hits = len(words & VIETNAMESE_COMMON_WORDS)
    return bool(VIETNAMESE_MARKS_PATTERN.search(text)) and common_hits >= 1


class AnalyzeRequest(BaseModel):
    image_jpeg_base64: str
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=128, ge=16, le=256)


class DistanceRequest(BaseModel):
    image_jpeg_base64: str
    bbox_xyxy: list[float] | None = Field(default=None, min_length=4, max_length=4)
    target_name: str = "đối tượng"


class PlanRequest(BaseModel):
    command: str = Field(min_length=1, max_length=500)


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent Qwen VLM service for OmniGlass")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    return parser.parse_args()


def create_app(model_id: str) -> FastAPI:
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    depth_model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    depth_processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id).eval().to("cuda")
    tts_model_id = "facebook/mms-tts-vie"
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_id)
    tts_model = VitsModel.from_pretrained(tts_model_id).eval().to("cuda")
    load_seconds = time.perf_counter() - started
    lock = threading.Lock()
    LOGGER.info("VLM loaded model=%s seconds=%.2f gpu=%s", model_id, load_seconds, torch.cuda.get_device_name())

    app = FastAPI(title="OmniGlass VLM Agent", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": model_id,
            "gpu": torch.cuda.get_device_name(),
            "load_seconds": load_seconds,
            "depth_model": depth_model_id,
            "tts_model": tts_model_id,
        }

    @app.post("/speak")
    def speak(request: SpeakRequest):
        # Markdown and emoji are useful on screen but make the synthesizer spell noise.
        spoken = re.sub(r"<[^>]+>", " ", request.text)
        spoken = re.sub(r"[*_`#|]", " ", spoken)
        spoken = re.sub(r"[^0-9A-Za-zÀ-ỹĐđ.,;:!?%\-–—()/'\s]", " ", spoken)
        spoken = re.sub(r"\s+", " ", spoken).strip()
        if not spoken:
            raise HTTPException(status_code=422, detail="Không có nội dung tiếng Việt để đọc")

        inputs = tts_tokenizer(spoken, return_tensors="pt").to("cuda")
        started_tts = time.perf_counter()
        with lock, torch.inference_mode():
            waveform = tts_model(**inputs).waveform[0].float().cpu()
        peak = float(waveform.abs().max())
        if not torch.isfinite(waveform).all() or peak <= 0:
            raise HTTPException(status_code=500, detail="Mô hình TTS tạo waveform không hợp lệ")
        pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
        sample_rate = int(tts_model.config.sampling_rate)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        elapsed = time.perf_counter() - started_tts
        duration = len(pcm) / 2 / sample_rate
        LOGGER.info("speak complete seconds=%.3f audio_seconds=%.2f chars=%s", elapsed, duration, len(spoken))
        return {
            "audio_wav_base64": base64.b64encode(wav_buffer.getvalue()).decode("ascii"),
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "inference_seconds": elapsed,
            "model": tts_model_id,
        }

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        try:
            data = base64.b64decode(request.image_jpeg_base64, validate=True)
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JPEG payload: {exc}") from exc

        safety = (
            "Bạn là trợ lý thị giác tiếng Việt cho người khiếm thị. Chỉ trả lời bằng tiếng Việt; "
            "tuyệt đối không dùng chữ Hán, tiếng Trung, tiếng Nhật, tiếng Hàn hoặc trộn ngôn ngữ. "
            "Trả lời ngắn, cụ thể, "
            "ưu tiên vật cản/người/vị trí trái-giữa-phải. Nói rõ khi không chắc chắn. "
            "Không tuyên bố đường đi an toàn và không bịa khoảng cách mét từ một ảnh RGB. "
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": safety}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Yêu cầu: " + request.prompt},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to("cuda")
        started_inference = time.perf_counter()
        with lock, torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if contains_cjk(answer) or not looks_vietnamese(answer):
            LOGGER.warning("Non-Vietnamese answer detected; rewriting in Vietnamese")
            rewrite_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Translate the content below into natural Vietnamese with full Vietnamese diacritics. "
                                "Output only the Vietnamese translation, with no explanation and no English sentence. "
                                "Không thêm thông tin mới; không dùng chữ Hán, Trung, Nhật hoặc Hàn.\n"
                                "Ví dụ: The image shows a red cup on a table.\n"
                                "Bản dịch: Ảnh cho thấy một chiếc cốc màu đỏ ở trên bàn.\n\n"
                                "Nội dung cần dịch:\n" + answer
                            ),
                        }
                    ],
                }
            ]
            rewrite_prompt = processor.apply_chat_template(
                rewrite_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            rewrite_inputs = processor(text=[rewrite_prompt], return_tensors="pt").to("cuda")
            with lock, torch.inference_mode():
                rewrite_ids = model.generate(
                    **rewrite_inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=False,
                )
            rewritten = processor.batch_decode(
                rewrite_ids[:, rewrite_inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            LOGGER.info("language rewrite original=%r rewritten=%r", answer[:500], rewritten[:500])
            if contains_cjk(rewritten) or not looks_vietnamese(rewritten):
                LOGGER.error("Vietnamese rewrite still failed language validation; returning safe fallback")
                answer = (
                    "Tôi chưa thể diễn đạt kết quả hoàn toàn bằng tiếng Việt. "
                    "Vui lòng hỏi lại hoặc hướng camera rõ hơn."
                )
            else:
                answer = rewritten
        inference_seconds = time.perf_counter() - started_inference
        LOGGER.info("analyze complete seconds=%.3f prompt_chars=%s", inference_seconds, len(request.prompt))
        return {"answer": answer, "inference_seconds": inference_seconds, "model": model_id}

    @app.post("/plan")
    def plan(request: PlanRequest):
        instruction = (
            "Bạn là command planner cho kính trợ năng. Chỉ trả về đúng một JSON object, không markdown. "
            "Schema: {\"action\":\"track|describe|distance|memory|stop|help|chat\","
            "\"target\":string|null,\"question\":string|null}. "
            "track=theo dõi/tìm một vật; describe=mô tả hoặc đọc chữ; distance=hỏi bao xa; "
            "memory=hỏi lần cuối/đã thấy; stop=dừng theo dõi; help=hỏi cách dùng; "
            "chat=câu hỏi thị giác khác. Không tạo thêm action và không tự trả lời câu hỏi. "
            "Ví dụ: 'để ý giúp tôi cái cốc đỏ' => track/cốc đỏ; "
            "'quanh đây có gì' => describe; 'nó ở đâu lần cuối' => memory; "
            "'nó bao xa' => distance.\n"
            f"Lệnh người dùng: {request.command}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], return_tensors="pt").to("cuda")
        started_plan = time.perf_counter()
        with lock, torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise HTTPException(status_code=422, detail="Planner did not return JSON")
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail="Planner returned invalid JSON") from exc
        allowed = {"track", "describe", "distance", "memory", "stop", "help", "chat"}
        action = str(result.get("action", "chat")).casefold()
        if action not in allowed:
            action = "chat"
        target = result.get("target")
        target = str(target).strip()[:120] if target else None
        question = result.get("question")
        question = str(question).strip()[:500] if question else request.command
        elapsed = time.perf_counter() - started_plan
        LOGGER.info("plan complete seconds=%.3f action=%s", elapsed, action)
        return {"action": action, "target": target, "question": question, "inference_seconds": elapsed}

    @app.post("/distance")
    def distance(request: DistanceRequest):
        try:
            data = base64.b64decode(request.image_jpeg_base64, validate=True)
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JPEG payload: {exc}") from exc

        inputs = depth_processor(images=image, return_tensors="pt").to("cuda")
        started_depth = time.perf_counter()
        with lock, torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = depth_model(**inputs).predicted_depth
            depth = F.interpolate(
                prediction.unsqueeze(1),
                size=(image.height, image.width),
                mode="bicubic",
                align_corners=False,
            )[0, 0].float()

        if request.bbox_xyxy:
            x1, y1, x2, y2 = request.bbox_xyxy
            x1, x2 = sorted((max(0, int(x1)), min(image.width, int(x2))))
            y1, y2 = sorted((max(0, int(y1)), min(image.height, int(y2))))
            margin_x = max(1, int((x2 - x1) * 0.10))
            margin_y = max(1, int((y2 - y1) * 0.10))
            x1, x2 = x1 + margin_x, x2 - margin_x
            y1, y2 = y1 + margin_y, y2 - margin_y
        else:
            x1, x2 = int(image.width * 0.35), int(image.width * 0.65)
            y1, y2 = int(image.height * 0.35), int(image.height * 0.75)
        region = depth[y1:y2, x1:x2]
        finite = region[torch.isfinite(region) & (region > 0)]
        if finite.numel() < 25:
            raise HTTPException(status_code=422, detail="Not enough valid metric-depth pixels")
        quantiles = torch.quantile(
            finite,
            torch.tensor([0.1, 0.5, 0.9], device=finite.device),
        ).tolist()
        near_m, median_m, far_m = (float(value) for value in quantiles)
        if median_m < 1.0:
            coarse = "dưới 1 mét"
        elif median_m < 2.0:
            coarse = "khoảng 1 đến 2 mét"
        else:
            coarse = "trên 2 mét"
        elapsed = time.perf_counter() - started_depth
        answer = (
            f"{request.target_name} được ước lượng {coarse}, trung vị khoảng {median_m:.1f} mét. "
            "Đây là depth đơn mắt chưa hiệu chỉnh camera, không dùng làm đảm bảo an toàn."
        )
        LOGGER.info("distance complete seconds=%.3f median_m=%.3f", elapsed, median_m)
        return {
            "answer": answer,
            "target": request.target_name,
            "median_m": median_m,
            "near_p10_m": near_m,
            "far_p90_m": far_m,
            "coarse": coarse,
            "calibrated": False,
            "model": depth_model_id,
            "inference_seconds": elapsed,
        }

    return app


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(create_app(args.model), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
