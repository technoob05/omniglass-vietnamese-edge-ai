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
    "toilet": "bồn cầu", "sink": "bồn rửa", "bed": "giường", "refrigerator": "tủ lạnh",
    "microwave": "lò vi sóng", "oven": "lò nướng", "toaster": "máy nướng bánh",
}
ALIASES = {
    "người": "person", "xe đạp": "bicycle", "ô tô": "car", "xe máy": "motorcycle",
    "ba lô": "backpack", "balo": "backpack", "túi": "handbag", "chai": "bottle",
    "cốc": "cup", "ly": "cup", "ghế": "chair", "bàn": "dining table",
    "màn hình": "tv", "máy tính": "laptop", "điện thoại": "cell phone", "sách": "book",
    "bồn cầu": "toilet", "nhà vệ sinh": "toilet", "bồn rửa": "sink", "lavabo": "sink",
    "giường": "bed", "tủ lạnh": "refrigerator",
}
SAFETY_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack",
    "handbag", "suitcase", "bottle", "chair", "couch", "dining table",
}
DYNAMIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
ASSISTIVE_LANDMARKS = {"toilet", "sink", "bed", "refrigerator", "microwave", "oven"}
EN_LOCATIONS = {"bên trái": "on the left", "ở giữa": "in the center", "bên phải": "on the right"}


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
        distance_vi = f" Ước lượng khoảng {depth:.1f} mét." if depth is not None else ""
        distance_en = f" Estimated distance: {depth:.1f} meters." if depth is not None else ""
        if very_close and corridor:
            severity, score = "danger", 3
            message = f"Cẩn thận! {item.label_vi.capitalize()} rất gần {item.location}.{distance_vi}"
            message_en = f"Caution! {item.label.capitalize()} is very close in the travel path.{distance_en}"
            rule = "center_path_very_close"
        elif close and corridor:
            severity, score = "warning", 2
            motion = " đang tiến lại gần" if approaching else " ở gần"
            message = f"Chú ý, {item.label_vi}{motion} {item.location}.{distance_vi}"
            message_en = f"Warning, {item.label} is {'approaching' if approaching else 'nearby'} in the travel path.{distance_en}"
            rule = "center_path_close"
        elif approaching and corridor:
            severity, score = "warning", 2
            message = f"Chú ý, {item.label_vi} đang tiến lại gần {item.location}.{distance_vi}"
            message_en = f"Warning, {item.label} is approaching in the travel path.{distance_en}"
            rule = "approaching"
        elif area >= 0.22:
            severity, score = "caution", 1
            message = f"Có {item.label_vi} khá gần {item.location}.{distance_vi}"
            message_en = f"There is a {item.label} fairly close {item.location}.{distance_en}"
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


def depth_sector_summary(depth: np.ndarray | None) -> dict[str, float | None]:
    """Robust lower-field depth for coarse left/centre/right guidance."""
    if depth is None or depth.ndim != 2:
        return {"left": None, "center": None, "right": None}
    height, width = depth.shape
    y1, y2 = int(height * 0.38), int(height * 0.88)
    sectors = {"left": (0.05, 0.32), "center": (0.36, 0.64), "right": (0.68, 0.95)}
    result: dict[str, float | None] = {}
    for name, (start, end) in sectors.items():
        values = depth[y1:y2, int(width * start):int(width * end)]
        values = values[np.isfinite(values) & (values > 0.05) & (values < 20)]
        result[name] = round(float(np.median(values)), 2) if values.size >= 100 else None
    return result


def estimate_front_wall(
    depth: np.ndarray | None,
    detections: list[Observation],
) -> dict[str, Any]:
    """Estimate a broad front-facing planar surface directly from metric depth."""
    empty = {
        "distance_m_advisory": None, "wall_likely": False, "confidence": 0.0,
        "valid_coverage": 0.0, "planar_inlier_ratio": 0.0,
        "method": "metric_depth_broad_plane_v1",
    }
    if depth is None or depth.ndim != 2:
        return empty

    height, width = depth.shape
    x1, x2 = int(width * 0.20), int(width * 0.80)
    y1, y2 = int(height * 0.18), int(height * 0.72)
    roi = depth[y1:y2, x1:x2].astype(np.float32, copy=False)
    valid = np.isfinite(roi) & (roi > 0.10) & (roi < 12.0)
    coverage = float(valid.mean()) if valid.size else 0.0
    if coverage < 0.75 or int(valid.sum()) < 500:
        return {**empty, "valid_coverage": round(coverage, 3)}

    # A large recognised object in the measurement window is a strong reason
    # not to call the broad surface behind it a wall.
    occlusion = 0.0
    for item in detections:
        bx1, by1, bx2, by2 = item.bbox_norm
        overlap_w = max(0.0, min(bx2, .80) - max(bx1, .20))
        overlap_h = max(0.0, min(by2, .72) - max(by1, .18))
        occlusion += overlap_w * overlap_h / (.60 * .54)

    yy, xx = np.indices(roi.shape, dtype=np.float32)
    sample = valid & ((yy.astype(np.int32) % 4 == 0) & (xx.astype(np.int32) % 4 == 0))
    z = roi[sample]
    xn = xx[sample] / max(roi.shape[1] - 1, 1) - 0.5
    yn = yy[sample] / max(roi.shape[0] - 1, 1) - 0.5
    design = np.column_stack((xn, yn, np.ones_like(xn)))
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    residual = np.abs(z - design @ coefficients)
    center_distance = float(coefficients[2])
    tolerance = max(0.12, center_distance * 0.10)
    inlier_ratio = float(np.mean(residual <= tolerance))
    slope = float(np.hypot(coefficients[0], coefficients[1])) / max(center_distance, .1)
    wall_likely = 0.35 <= center_distance <= 6.0 and inlier_ratio >= 0.78 and slope <= 1.0 and occlusion < 0.28
    confidence = min(0.95, max(0.0, inlier_ratio * coverage * (1.0 - min(occlusion, 1.0))))
    return {
        "distance_m_advisory": round(center_distance, 2),
        "wall_likely": bool(wall_likely),
        "confidence": round(confidence, 3),
        "valid_coverage": round(coverage, 3),
        "planar_inlier_ratio": round(inlier_ratio, 3),
        "relative_plane_slope": round(slope, 3),
        "detector_occlusion_ratio": round(min(occlusion, 1.0), 3),
        "method": "metric_depth_broad_plane_v1",
    }


def evaluate_surface_safety(
    front_wall: dict[str, Any],
    previous: tuple[float, int] | None,
    now_ms: int,
    last_alert_ms: dict[str, int],
) -> tuple[dict[str, Any] | None, tuple[float, int] | None]:
    distance = front_wall.get("distance_m_advisory")
    if distance is None or not front_wall.get("wall_likely"):
        return None, None
    current = (float(distance), now_ms)
    approaching = bool(previous and now_ms - previous[1] <= 2500 and previous[0] - distance >= 0.16)
    if distance <= 0.75:
        severity, score, cooldown_ms = "danger", 4, 2500
        lead_vi = "Cẩn thận! Bạn đang tới rất gần tường phía trước" if approaching else "Cẩn thận! Tường ở ngay phía trước"
        lead_en = "Caution! You are moving very close to the wall ahead" if approaching else "Caution! A wall is immediately ahead"
    elif distance <= 1.45:
        severity, score, cooldown_ms = "warning", 3, 5500
        lead_vi = "Chú ý, bạn đang tới gần tường phía trước" if approaching else "Chú ý, có tường gần phía trước"
        lead_en = "Warning, you are approaching the wall ahead" if approaching else "Warning, there is a wall close ahead"
    else:
        return None, current
    key = f"{severity}:front_wall"
    should_announce = now_ms - last_alert_ms.get(key, -cooldown_ms) >= cooldown_ms
    if should_announce:
        last_alert_ms[key] = now_ms
    return {
        "id": key, "severity": severity, "score": score, "rule": "front_wall_depth_plane",
        "message_vi": f"{lead_vi}. Ước lượng khoảng {distance:.1f} mét.",
        "message_en": f"{lead_en}. Estimated distance: {distance:.1f} meters.",
        "label": "wall", "label_vi": "tường", "location": "ở giữa", "track_id": None,
        "confidence": front_wall.get("confidence", 0.0), "depth_m_advisory": distance,
        "area_ratio": front_wall.get("planar_inlier_ratio", 0.0), "approaching": approaching,
        "should_announce": should_announce, "cooldown_ms": cooldown_ms,
    }, current


def evaluate_assistive_guidance(
    detections: list[Observation],
    depth: np.ndarray | None,
    now_ms: int,
    last_alert_ms: dict[str, int],
) -> dict[str, Any]:
    """Conservative landmarks and coarse free-space hints; never calls a VLM."""
    candidates: list[dict[str, Any]] = []
    landmarks = [item for item in detections if item.label in ASSISTIVE_LANDMARKS and item.confidence >= 0.45]
    toilet = next((item for item in landmarks if item.label == "toilet"), None)
    sink = next((item for item in landmarks if item.label == "sink"), None)
    if toilet:
        distance_vi = f", ước lượng khoảng {toilet.depth_m:.1f} mét" if toilet.depth_m is not None else ""
        message_vi = f"Bồn cầu ở {toilet.location}{distance_vi}."
        message_en = f"A toilet is {EN_LOCATIONS.get(toilet.location, toilet.location)}" + (f", approximately {toilet.depth_m:.1f} meters away." if toilet.depth_m is not None else ".")
        if sink:
            message_vi += " Có bồn rửa trong cùng khung hình; đây có thể là khu vệ sinh."
            message_en += " A sink is also visible; this may be a restroom area."
        candidates.append({"rule":"toilet_landmark", "message_vi":message_vi, "message_en":message_en,
                           "confidence":toilet.confidence, "depth_m_advisory":toilet.depth_m, "score":4})
    elif sink:
        distance_vi = f", ước lượng khoảng {sink.depth_m:.1f} mét" if sink.depth_m is not None else ""
        candidates.append({"rule":"sink_landmark", "message_vi":f"Bồn rửa ở {sink.location}{distance_vi}.",
                           "message_en":f"A sink is {EN_LOCATIONS.get(sink.location, sink.location)}.", "confidence":sink.confidence,
                           "depth_m_advisory":sink.depth_m, "score":1})

    sectors = depth_sector_summary(depth)
    front_wall = estimate_front_wall(depth, detections)
    center, left, right = sectors["center"], sectors["left"], sectors["right"]
    if center is not None and center <= 1.25 and left is not None and right is not None:
        best_name, best_value = ("left", left) if left >= right else ("right", right)
        if best_value >= center + 0.55:
            direction_vi = "bên trái" if best_name == "left" else "bên phải"
            candidates.append({
                "rule":"center_surface_side_opening", "score":3, "confidence":0.7,
                "depth_m_advisory":center,
                "message_vi":f"Phía trước có {'tường' if front_wall['wall_likely'] else 'bề mặt hoặc vật cản'} gần, ước lượng khoảng {center:.1f} mét. {direction_vi.capitalize()} có vẻ thoáng hơn; hãy kiểm tra bằng gậy trước khi đổi hướng.",
                "message_en":f"A {'wall' if front_wall['wall_likely'] else 'surface or obstacle'} appears close ahead, approximately {center:.1f} meters. The {best_name} side appears more open; check with your cane before changing direction.",
            })

    candidates.sort(key=lambda row: row["score"], reverse=True)
    primary = candidates[0] if candidates else None
    if primary:
        key = f"guidance:{primary['rule']}"
        cooldown_ms = 15000 if "landmark" in primary["rule"] else 8000
        primary.update({"id":key, "severity":"guidance", "cooldown_ms":cooldown_ms})
        primary["should_announce"] = now_ms - last_alert_ms.get(key, -cooldown_ms) >= cooldown_ms
        if primary["should_announce"]:
            last_alert_ms[key] = now_ms
    return {"primary_alert":primary, "candidates":candidates[:5], "depth_sectors_m_advisory":sectors,
            "front_wall":front_wall,
            "vlm_used":False, "navigation_certified":False}


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
        self.previous_wall: tuple[float, int] | None = None
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
            guidance = evaluate_assistive_guidance(current, state.depth, now_ms, state.last_alert_ms)
            wall_alert, state.previous_wall = evaluate_surface_safety(
                guidance["front_wall"], state.previous_wall, now_ms, state.last_alert_ms,
            )
            if wall_alert is not None:
                hazards = [*safety["active_hazards"], wall_alert]
                hazards.sort(key=lambda row: (row["score"], row["area_ratio"], row["confidence"]), reverse=True)
                safety["active_hazards"] = hazards[:5]
                safety["primary_alert"] = hazards[0]
                safety["state"] = hazards[0]["severity"]
            # Collision alerts always win. Landmark/free-space guidance is only
            # spoken when the immediate safety channel is clear.
            if safety["primary_alert"] is None and guidance["primary_alert"] is not None:
                safety["state"] = "guidance"
                safety["primary_alert"] = guidance["primary_alert"]
            state.frames += 1
            state.last_frame_at = time.monotonic()
            elapsed_ms = (time.perf_counter() - started) * 1000
            state.inference_ms_ema = elapsed_ms if state.inference_ms_ema is None else 0.2 * elapsed_ms + 0.8 * state.inference_ms_ema
            detector_fps = 1000 / max(state.inference_ms_ema, 1e-6)
            counts = Counter(item.label_vi for item in current)
            scene = ", ".join(f"{count} {label}" for label, count in counts.most_common(8)) or "chưa nhận ra vật rõ ràng"
            watch = normalize_target(request.watch_target, {item.label for item in current}) if request.watch_target else None
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
                "safety": safety, "assistive_guidance": guidance,
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
