from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .memory import load_memory, normalize_query_label, save_memory
from .schema import Observation, SessionMemory, WatchEvent

LOGGER = logging.getLogger("omniglass.perception")


def normalize_video_value(value) -> str:
    if isinstance(value, dict):
        value = value.get("video", value.get("path"))
    if hasattr(value, "path"):
        value = value.path
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not value:
        raise ValueError("Chưa nhận được video.")
    return str(value)


def screen_location(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / max(width, 1)
    cy = (y1 + y2) / 2 / max(height, 1)
    horizontal = "bên trái" if cx < 0.36 else "bên phải" if cx > 0.64 else "chính giữa"
    vertical = "phía trên" if cy < 0.34 else "phía dưới" if cy > 0.70 else ""
    return f"{vertical} {horizontal}".strip()


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, (ax2 - ax1) * (ay2 - ay1)) + max(
        0.0, (bx2 - bx1) * (by2 - by1)
    ) - intersection
    return intersection / union if union > 0 else 0.0


class SharedPerceptionPipeline:
    """One detector/tracker pass shared by See, Find, Remember, and Watch."""

    def __init__(self, model_name: str = "yolo11n.pt", device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.lock = threading.Lock()

    def process_video(
        self,
        video_path: str,
        output_root: str | Path,
        sampled_fps: float = 3.0,
        max_seconds: float = 20.0,
        watch_target: str = "",
        confidence: float = 0.25,
    ) -> tuple[SessionMemory, str, str, str | None, dict[str, float]]:
        started = time.perf_counter()
        source = Path(video_path)
        session_id = uuid.uuid4().hex[:10]
        run_dir = Path(output_root) / f"run_{session_id}"
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=False)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Không thể mở video đã tải lên.")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_duration = source_count / source_fps if source_count > 0 else 0.0
        frame_step = max(1, int(round(source_fps / max(sampled_fps, 0.1))))
        max_source_frame = max(1, int(round(max_seconds * source_fps)))

        observations: list[Observation] = []
        watch_events: list[WatchEvent] = []
        sampled_timestamps: list[float] = []
        latest_keyframe: str | None = None
        available_labels = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        watch_requested = watch_target.strip().lower()
        watch_name = normalize_query_label(watch_requested, available_labels) or watch_requested
        initial_watch_center: np.ndarray | None = None
        selected_track_id: str | None = None
        previous_watch_box: tuple[float, float, float, float] | None = None
        watch_seen = False
        missing_since_timestamp: float | None = None
        region_change_reported = False
        missing_reported = False
        writer = None
        output_video = str(run_dir / "annotated_memory.mp4")
        effective_fps = source_fps / frame_step

        try:
            with self.lock:
                self.model.predictor = None
                source_index = 0
                sample_index = 0
                while source_index < max_source_frame:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if source_index % frame_step:
                        source_index += 1
                        continue

                    timestamp = source_index / source_fps
                    result = self.model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=confidence,
                        device=self.device,
                        verbose=False,
                    )[0]
                    annotated = result.plot()
                    keyframe_name = f"frame_{sample_index:05d}.jpg"
                    keyframe_path = frames_dir / keyframe_name
                    if not cv2.imwrite(str(keyframe_path), annotated):
                        raise RuntimeError(f"Không thể ghi keyframe: {keyframe_path}")
                    latest_keyframe = str(keyframe_path)
                    sampled_timestamps.append(timestamp)
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            output_video,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            max(float(effective_fps), 1.0),
                            (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Không thể mở video writer: {output_video}")
                    writer.write(annotated)

                    frame_watch = []
                    boxes = result.boxes
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        scores = boxes.conf.detach().cpu().numpy()
                        classes = boxes.cls.detach().cpu().numpy().astype(int)
                        track_ids = (
                            boxes.id.detach().cpu().numpy().astype(int)
                            if boxes.id is not None
                            else np.full(len(boxes), -1, dtype=int)
                        )
                        for box, score, class_id, track_id in zip(xyxy, scores, classes, track_ids):
                            label = str(result.names[int(class_id)]).lower()
                            box_tuple = tuple(float(x) for x in box)
                            observation = Observation(
                                timestamp=timestamp,
                                frame_index=sample_index,
                                label=label,
                                confidence=float(score),
                                bbox_xyxy=box_tuple,
                                track_id=None if track_id < 0 else str(track_id),
                                location=screen_location(box_tuple, frame.shape[1], frame.shape[0]),
                                keyframe_path=str(Path("frames") / keyframe_name),
                                attributes={"source": "ultralytics-yolo+bytetrack"},
                            )
                            observations.append(observation)
                            if watch_name and (watch_name == label or watch_name in label or label in watch_name):
                                frame_watch.append(observation)

                    if watch_name:
                        same_track = (
                            [item for item in frame_watch if item.track_id == selected_track_id]
                            if selected_track_id
                            else []
                        )
                        if same_track:
                            candidates = same_track
                        elif selected_track_id and previous_watch_box:
                            candidates = [
                                item for item in frame_watch
                                if box_iou(item.bbox_xyxy, previous_watch_box) >= 0.25
                            ]
                        else:
                            candidates = frame_watch
                        best = max(candidates, key=lambda item: item.confidence) if candidates else None
                        if best:
                            x1, y1, x2, y2 = best.bbox_xyxy
                            center = np.array(
                                [(x1 + x2) / 2 / frame.shape[1], (y1 + y2) / 2 / frame.shape[0]]
                            )
                            if selected_track_id is None:
                                selected_track_id = best.track_id
                            if initial_watch_center is None:
                                initial_watch_center = center
                            displacement = float(np.linalg.norm(center - initial_watch_center))
                            if displacement > 0.18 and not region_change_reported:
                                watch_events.append(
                                    WatchEvent(
                                        timestamp,
                                        sample_index,
                                        watch_name,
                                        "screen_region_changed",
                                        f"{watch_name} đã đổi vùng trong khung hình; có thể do vật hoặc camera di chuyển",
                                    )
                                )
                                region_change_reported = True
                            if watch_seen and missing_reported:
                                watch_events.append(
                                    WatchEvent(
                                        timestamp,
                                        sample_index,
                                        watch_name,
                                        "reappeared",
                                        f"{watch_name} đã xuất hiện trở lại ở {best.location}",
                                    )
                                )
                            watch_seen = True
                            previous_watch_box = best.bbox_xyxy
                            missing_since_timestamp = None
                            missing_reported = False
                        elif watch_seen:
                            missing_since_timestamp = missing_since_timestamp or timestamp
                            if timestamp - missing_since_timestamp >= 1.5 and not missing_reported:
                                watch_events.append(
                                    WatchEvent(
                                        timestamp,
                                        sample_index,
                                        watch_name,
                                        "missing",
                                        f"Không còn thấy {watch_name} trong khung hình",
                                    )
                                )
                                missing_reported = True

                    sample_index += 1
                    source_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if not sampled_timestamps:
            raise ValueError("Video không có frame hợp lệ.")

        video_seconds = 0.0

        actual_duration = sampled_timestamps[-1] if sampled_timestamps else 0.0
        memory = SessionMemory(
            session_id=session_id,
            source_name=source.name,
            source_fps=source_fps,
            sampled_fps=float(effective_fps),
            processed_frames=len(sampled_timestamps),
            duration_seconds=actual_duration,
            observations=observations,
            watch_events=watch_events,
            metadata={
                "watch_target": watch_name or None,
                "watch_target_requested": watch_requested or None,
                "source_duration_seconds": source_duration,
                "processed_until_seconds": actual_duration,
                "truncated": bool(source_duration and source_duration > max_seconds),
                "perception_model": str(getattr(self.model, "model_name", "yolo11n.pt")),
                "coordinate_mode": "2D screen-relative; not metric 3D",
                "directme_adapter_status": "planned; flat observations are not a DirectMe scene graph",
                "requested_sampled_fps": float(sampled_fps),
            },
        )
        memory_path = save_memory(memory, run_dir / "memory.json")
        memory = load_memory(memory_path)
        timings = {
            "total": time.perf_counter() - started,
            "video_encode": video_seconds,
        }
        return memory, output_video, memory_path, latest_keyframe, timings
