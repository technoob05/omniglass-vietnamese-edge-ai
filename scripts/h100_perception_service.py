#!/usr/bin/env python3
"""Bounded live YOLO/depth/memory service for the native Vietnamese demo."""

from __future__ import annotations

import argparse
import base64
import io
import re
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field


SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
VI_LABELS = {
    "person": "người", "bicycle": "xe đạp", "car": "ô tô", "motorcycle": "xe máy",
    "bus": "xe buýt", "truck": "xe tải", "traffic light": "đèn giao thông",
    "backpack": "ba lô", "handbag": "túi xách", "suitcase": "va li", "bottle": "chai",
    "cup": "cốc", "chair": "ghế", "couch": "ghế sofa", "dining table": "bàn",
    "tv": "màn hình", "laptop": "máy tính xách tay", "cell phone": "điện thoại",
    "book": "sách", "clock": "đồng hồ", "scissors": "kéo", "umbrella": "ô",
}
ALIASES = {
    "người": "person", "xe đạp": "bicycle", "ô tô": "car", "xe máy": "motorcycle",
    "ba lô": "backpack", "balo": "backpack", "túi": "handbag", "chai": "bottle",
    "cốc": "cup", "ly": "cup", "ghế": "chair", "bàn": "dining table",
    "màn hình": "tv", "máy tính": "laptop", "điện thoại": "cell phone", "sách": "book",
}
SAFETY_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack",
    "handbag", "suitcase", "bottle", "chair", "couch", "dining table",
}
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def screen_zone(box: tuple[float, float, float, float], width: int) -> str:
    center = (box[0] + box[2]) / 2 / max(width, 1)
    return "bên trái" if center < 0.36 else "bên phải" if center > 0.64 else "ở giữa"


def normalize_target(text: str, available: set[str]) -> str | None:
    value = text.casefold().strip()
    for alias, label in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in value:
            return label
    return next((label for label in sorted(available, key=len, reverse=True) if label in value), None)


@dataclass(slots=True)
class Observation:
    label: str
    label_vi: str
    confidence: float
    bbox_norm: tuple[float, float, float, float]
    location: str
    track_id: int | None
    seen_at_ms: int
    depth_m: float | None


def evaluate_safety(
    detections: list[Observation],
    previous_geometry: dict[str, tuple[float, int]],
    now_ms: int,
    last_alert_ms: dict[str, int],
) -> tuple[dict[str, Any], dict[str, tuple[float, int]]]:
    """Deterministic collision-risk rules; never calls a VLM."""
    hazards: list[dict[str, Any]] = []
    next_geometry: dict[str, tuple[float, int]] = {}
    for item in detections:
        if item.label not in SAFETY_CLASSES or item.confidence < 0.42:
            continue
        x1, y1, x2, y2 = item.bbox_norm
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        bottom = y2
        corridor = x2 >= 0.31 and x1 <= 0.69
        key = f"{item.label}:{item.track_id if item.track_id is not None else item.location}"
        next_geometry[key] = (area, now_ms)
        previous = previous_geometry.get(key)
        approaching = bool(
            previous and now_ms - previous[1] <= 2200 and previous[0] >= 0.025
            and area >= previous[0] * 1.22 and area - previous[0] >= 0.012
        )
        depth = item.depth_m
        very_close = (depth is not None and depth <= 0.85) or area >= 0.30 or (bottom >= 0.96 and area >= 0.16)
        close = (depth is not None and depth <= (2.4 if item.label in DYNAMIC_CLASSES else 1.65)) or area >= 0.12 or (bottom >= 0.86 and area >= 0.075)
        if not corridor and area < 0.22:
            continue
        if very_close and corridor:
            severity, score = "danger", 3
            message = f"Cẩn thận! {item.label_vi.capitalize()} rất gần {item.location}."
            message_en = f"Caution! {item.label.capitalize()} is very close in the travel path."
            rule = "center_path_very_close"
        elif close and corridor:
            severity, score = "warning", 2
            motion = " đang tiến lại gần" if approaching else " ở gần"
            message = f"Chú ý, {item.label_vi}{motion} {item.location}."
            message_en = f"Warning, {item.label} is {'approaching' if approaching else 'nearby'} in the travel path."
            rule = "center_path_close"
        elif approaching and corridor:
            severity, score = "warning", 2
            message = f"Chú ý, {item.label_vi} đang tiến lại gần {item.location}."
            message_en = f"Warning, {item.label} is approaching in the travel path."
            rule = "approaching"
        elif area >= 0.22:
            severity, score = "caution", 1
            message = f"Có {item.label_vi} khá gần {item.location}."
            message_en = f"There is a {item.label} fairly close {item.location}."
            rule = "large_lateral_object"
        else:
            continue
        alert_key = f"{severity}:{key}"
        cooldown_ms = 3000 if severity == "danger" else 7000 if severity == "warning" else 12000
        should_announce = now_ms - last_alert_ms.get(alert_key, -cooldown_ms) >= cooldown_ms
        if should_announce:
            last_alert_ms[alert_key] = now_ms
        hazards.append({
            "id": alert_key, "severity": severity, "score": score,
            "message_vi": message, "message_en": message_en, "label": item.label, "label_vi": item.label_vi,
            "location": item.location, "track_id": item.track_id,
            "confidence": item.confidence, "depth_m_advisory": depth,
            "area_ratio": round(area, 4), "approaching": approaching,
            "rule": rule, "should_announce": should_announce,
            "cooldown_ms": cooldown_ms,
        })
    hazards.sort(key=lambda row: (row["score"], row["area_ratio"], row["confidence"]), reverse=True)
    primary = hazards[0] if hazards else None
    return {
        "state": primary["severity"] if primary else "clear",
        "primary_alert": primary,
        "active_hazards": hazards[:5],
        "rules_engine": "geometry+depth+tracking-v1",
        "vlm_used": False,
        "depth_is_advisory": True,
    }, next_geometry


class FrameRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    image_jpeg_base64: str = Field(min_length=100)
    confidence: float = Field(default=0.35, ge=0.1, le=0.9)
    watch_target: str = Field(default="", max_length=80)


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


class QueryRequest(SessionRequest):
    question: str = Field(min_length=1, max_length=500)


class SessionState:
    def __init__(self, history_size: int) -> None:
        self.started = time.monotonic()
        self.history: deque[Observation] = deque(maxlen=history_size)
        self.current: list[Observation] = []
        self.frames = 0
        self.last_frame_at = 0.0
        self.inference_ms_ema: float | None = None
        self.depth: np.ndarray | None = None
        self.depth_frame = -1
        self.previous_geometry: dict[str, tuple[float, int]] = {}
        self.last_alert_ms: dict[str, int] = {}

    def memory_rows(self, now_ms: int) -> list[dict[str, Any]]:
        grouped: dict[str, list[Observation]] = {}
        for item in self.history:
            grouped.setdefault(item.label, []).append(item)
        rows = []
        for label, items in grouped.items():
            latest = max(items, key=lambda item: item.seen_at_ms)
            rows.append({
                "label": label,
                "label_vi": latest.label_vi,
                "observations": len(items),
                "last_seen_ms_ago": max(0, now_ms - latest.seen_at_ms),
                "location": latest.location,
                "confidence": latest.confidence,
                "depth_m": latest.depth_m,
            })
        return sorted(rows, key=lambda row: row["last_seen_ms_ago"])[:20]


class PerceptionEngine:
    def __init__(self, model_path: str, device: str, depth_model_id: str, history_size: int, depth_every: int) -> None:
        from ultralytics import YOLO

        self.detector = YOLO(model_path)
        self.device = device
        self.depth_model_id = depth_model_id
        self.history_size = history_size
        self.depth_every = max(1, depth_every)
        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.Lock()
        self.depth_processor = None
        self.depth_model = None
        self.loaded_at = time.monotonic()

    def warmup(self) -> None:
        sample = np.zeros((360, 640, 3), dtype=np.uint8)
        self.detector.predict(sample, conf=0.25, device=self.device, verbose=False)
        if self.depth_model_id:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            self.depth_processor = AutoImageProcessor.from_pretrained(self.depth_model_id)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_model_id).eval().to(self.device)

    def state(self, session_id: str) -> SessionState:
        if not SESSION_RE.fullmatch(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id")
        return self.sessions.setdefault(session_id, SessionState(self.history_size))

    def reset(self, session_id: str) -> dict[str, Any]:
        with self.lock:
            self.sessions[session_id] = SessionState(self.history_size)
        return {"ok": True, "session_id": session_id}

    @staticmethod
    def decode_image(payload: str) -> np.ndarray:
        try:
            raw = base64.b64decode(payload, validate=True)
            return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JPEG: {exc}") from exc

    def infer_depth(self, image: np.ndarray) -> np.ndarray | None:
        if self.depth_model is None or self.depth_processor is None:
            return None
        inputs = self.depth_processor(images=Image.fromarray(image), return_tensors="pt").to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = self.depth_model(**inputs).predicted_depth
            depth = F.interpolate(
                prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False,
            )[0, 0].float().cpu().numpy()
        return depth

    @staticmethod
    def box_depth(depth: np.ndarray | None, box: tuple[float, float, float, float]) -> float | None:
        if depth is None:
            return None
        height, width = depth.shape
        x1, y1, x2, y2 = (int(value) for value in box)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        mx, my = max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)
        values = depth[y1 + my:y2 - my, x1 + mx:x2 - mx]
        values = values[np.isfinite(values) & (values > 0)]
        return round(float(np.median(values)), 2) if values.size >= 25 else None

    def process(self, request: FrameRequest) -> dict[str, Any]:
        image = self.decode_image(request.image_jpeg_base64)
        height, width = image.shape[:2]
        started = time.perf_counter()
        with self.lock:
            state = self.state(request.session_id)
            result = self.detector.track(
                image, persist=True, tracker="bytetrack.yaml", conf=request.confidence,
                imgsz=640, device=self.device, verbose=False,
            )[0]
            if self.depth_model is not None and state.frames % self.depth_every == 0:
                state.depth = self.infer_depth(image)
                state.depth_frame = state.frames
            now_ms = int((time.monotonic() - state.started) * 1000)
            current: list[Observation] = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)
                for raw_box, score, class_id, track_id in zip(xyxy, scores, classes, ids):
                    box = tuple(float(value) for value in raw_box)
                    label = str(result.names[int(class_id)]).casefold()
                    item = Observation(
                        label=label, label_vi=VI_LABELS.get(label, label), confidence=round(float(score), 4),
                        bbox_norm=tuple(round(value, 5) for value in (box[0] / width, box[1] / height, box[2] / width, box[3] / height)),
                        location=screen_zone(box, width), track_id=None if int(track_id) < 0 else int(track_id),
                        seen_at_ms=now_ms, depth_m=self.box_depth(state.depth, box),
                    )
                    state.history.append(item)
                    current.append(item)
            state.current = current
            safety, state.previous_geometry = evaluate_safety(
                current, state.previous_geometry, now_ms, state.last_alert_ms,
            )
            state.frames += 1
            state.last_frame_at = time.monotonic()
            elapsed_ms = (time.perf_counter() - started) * 1000
            state.inference_ms_ema = elapsed_ms if state.inference_ms_ema is None else 0.2 * elapsed_ms + 0.8 * state.inference_ms_ema
            detector_fps = 1000 / max(state.inference_ms_ema, 1e-6)
            counts = Counter(item.label_vi for item in current)
            scene = ", ".join(f"{count} {label}" for label, count in counts.most_common(8)) or "chưa nhận ra vật rõ ràng"
            watch = normalize_target(request.watch_target, {item.label for item in state.history}) if request.watch_target else None
            watch_status = None
            if watch:
                visible = [item for item in current if item.label == watch]
                latest = next((row for row in state.memory_rows(now_ms) if row["label"] == watch), None)
                watch_status = "visible" if visible else "missing" if latest else "never_seen"
            return {
                "schema": "omniglass.live-perception.v2", "session_id": request.session_id,
                "frame_id": state.frames, "image_width": width, "image_height": height,
                "detections": [asdict(item) for item in current], "scene_vi": scene,
                "memory": state.memory_rows(now_ms), "watch": {"target": watch, "status": watch_status},
                "safety": safety,
                "metrics": {
                    "detector": "YOLO11n + ByteTrack", "detector_fps_capacity": round(detector_fps, 2),
                    "inference_ms": round(elapsed_ms, 2), "depth_enabled": self.depth_model is not None,
                    "depth_age_frames": None if state.depth_frame < 0 else state.frames - 1 - state.depth_frame,
                    "depth_calibrated": False, "history_size": len(state.history),
                },
            }

    def query(self, request: QueryRequest) -> dict[str, Any]:
        with self.lock:
            state = self.state(request.session_id)
            now_ms = int((time.monotonic() - state.started) * 1000)
            rows = state.memory_rows(now_ms)
            labels = {row["label"] for row in rows}
            target = normalize_target(request.question, labels)
            text = request.question.casefold()
            if target:
                row = next((item for item in rows if item["label"] == target), None)
                distance_intent = any(token in text for token in ("bao xa", "khoảng cách", "mấy mét", "mét"))
                if row and distance_intent and row["depth_m"] is not None:
                    answer = (
                        f"{row['label_vi']} {row['location']}, depth đơn mắt ước lượng khoảng {row['depth_m']:.1f} mét. "
                        "Khoảng cách này chưa hiệu chỉnh và không dùng để bảo đảm đường đi an toàn."
                    )
                elif row and distance_intent:
                    answer = f"Tôi đã thấy {row['label_vi']} {row['location']} nhưng chưa có depth hợp lệ."
                else:
                    answer = (f"Lần cuối thấy {row['label_vi']} {row['location']}, cách đây {row['last_seen_ms_ago']/1000:.1f} giây."
                              if row else f"Tôi chưa thấy {VI_LABELS.get(target, target)} trong phiên này.")
            elif any(token in text for token in ("hiện tại", "đang thấy", "trước mặt")):
                counts = Counter(item.label_vi for item in state.current)
                answer = "Hiện tại tôi thấy " + ", ".join(f"{n} {label}" for label, n in counts.items()) + "." if counts else "Hiện tại detector chưa nhận ra vật rõ ràng."
            else:
                answer = "Trong phiên này tôi đã thấy: " + ", ".join(row["label_vi"] for row in rows[:12]) + "." if rows else "Visual memory chưa có dữ liệu."
            return {"answer_vi": answer, "memory": rows}


def create_app(engine: PerceptionEngine) -> FastAPI:
    app = FastAPI(title="OmniGlass H100 Perception", version="2.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True, "detector": "YOLO11n + ByteTrack", "device": engine.device,
            "depth_model": engine.depth_model_id or None, "sessions": len(engine.sessions),
            "uptime_seconds": round(time.monotonic() - engine.loaded_at, 1),
        }

    @app.post("/frame")
    def frame(request: FrameRequest) -> dict[str, Any]:
        return engine.process(request)

    @app.post("/reset")
    def reset(request: SessionRequest) -> dict[str, Any]:
        return engine.reset(request.session_id)

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, Any]:
        return engine.query(request)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18784)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
    parser.add_argument("--depth-every", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=800)
    args = parser.parse_args()
    engine = PerceptionEngine(args.model, args.device, args.depth_model, args.history_size, args.depth_every)
    engine.warmup()
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
