"""Calibration, scene construction, and advisory dense-depth corridor state."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from dataclasses import dataclass
from typing import Any

from .models import RawDetection, SceneObject, SceneSnapshot, WalkingCorridor
from .tracker import IoUTracker, Track


@dataclass(frozen=True)
class DepthCalibration:
    enabled: bool
    scale: float
    exponent: float
    offset_m: float
    dangerous_overestimate_p99_m: float
    valid_min_m: float
    valid_max_m: float

    @classmethod
    def from_config(cls, value: dict[str, Any]) -> "DepthCalibration":
        return cls(
            enabled=bool(value["enabled"]),
            scale=float(value["scale"]),
            exponent=float(value["exponent"]),
            offset_m=float(value["offset_m"]),
            dangerous_overestimate_p99_m=float(value["dangerous_overestimate_p99_m"]),
            valid_min_m=float(value["valid_min_m"]),
            valid_max_m=float(value["valid_max_m"]),
        )

    def calibrated(self, raw_m: float | None) -> float | None:
        if raw_m is None or raw_m <= 0:
            return None
        value = self.scale * (raw_m ** self.exponent) + self.offset_m
        if not self.enabled or value < self.valid_min_m or value > self.valid_max_m:
            return None
        return value

    def conservative(self, raw_m: float | None) -> float | None:
        calibrated = self.calibrated(raw_m)
        if calibrated is None:
            return None
        return max(0.0, calibrated - self.dangerous_overestimate_p99_m)


class SceneBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        tracking = config["tracking"]
        self.tracker = IoUTracker(float(tracking["iou_threshold"]), int(tracking["max_missed_frames"]))
        self.calibration = DepthCalibration.from_config(config["calibration"])
        self._frame_id = 0
        self._last_scene: SceneSnapshot | None = None
        self._last_source_sequence: int | None = None

    @property
    def latest(self) -> SceneSnapshot | None:
        return self._last_scene

    def update(
        self,
        detections: list[RawDetection],
        health: dict[str, Any],
        depth_grid: dict[str, Any] | None = None,
        now_ns: int | None = None,
    ) -> SceneSnapshot:
        now_ns = now_ns or time.monotonic_ns()
        camera_ok, depth_ok, age_ms = self._perception_health(health, now_ns)
        source_sequence = _as_nonnegative_int(health.get("publish_sequence"))
        if (
            source_sequence is not None
            and source_sequence == self._last_source_sequence
            and self._last_scene is not None
        ):
            # Polling must not make one QNN frame appear as several tracker
            # frames.  Preserve objects but let liveness age independently.
            scene = replace(
                self._last_scene,
                published_at_ns=now_ns,
                frame_age_ms=age_ms,
                camera_ok=camera_ok,
                depth_ok=depth_ok,
            )
            self._last_scene = scene
            return scene
        self._frame_id += 1
        if source_sequence is not None:
            self._last_source_sequence = source_sequence
        matched = self.tracker.update(detections, now_ns)
        objects = [self._scene_object(track, detection) for track, detection in matched]
        objects.sort(key=lambda item: (item.conservative_depth_m is None, item.conservative_depth_m or math.inf))
        scene = SceneSnapshot(
            frame_id=self._frame_id,
            captured_at_ns=now_ns - int(age_ms * 1e6),
            published_at_ns=now_ns,
            frame_age_ms=age_ms,
            camera_ok=camera_ok,
            depth_ok=depth_ok,
            depth_calibrated=self.calibration.enabled,
            objects=objects,
            walking_corridor=self._corridor(depth_grid),
        )
        self._last_scene = scene
        return scene

    def _perception_health(self, health: dict[str, Any], now_ns: int) -> tuple[bool, bool, float]:
        status_ok = health.get("status") == "ok"
        rates = health.get("rates", {})
        camera_ok = status_ok and float(rates.get("camera_fps", 0.0)) > 0.1
        depth_ok = status_ok and float(rates.get("depth_fps", 0.0)) > 0.1
        # The legacy service has no publish timestamp.  Its adapter supplies an
        # age measurement; missing data is conservative and becomes stale.
        adapter_age = health.get("adapter_age_ms")
        published_ns = _as_nonnegative_int(health.get("published_monotonic_ns"))
        if published_ns is not None and published_ns <= now_ns:
            age_ms = (now_ns - published_ns) / 1e6
        else:
            age_ms = float(adapter_age) if adapter_age is not None else float("inf")
        if age_ms > float(self.config["perception"]["stale_after_ms"]):
            camera_ok = False
            depth_ok = False
        return camera_ok, depth_ok, age_ms

    def _scene_object(self, track: Track, detection: RawDetection) -> SceneObject:
        width = float(self.config["perception"]["source_width"])
        intrinsics = self.config["calibration"]["camera_intrinsics"]
        cx = float(intrinsics.get("cx") or width / 2.0)
        fx = intrinsics.get("fx")
        center_x = (detection.x1 + detection.x2) / 2.0
        bearing = None if not fx else math.degrees(math.atan2(center_x - cx, float(fx)))
        zone = "left" if center_x < width / 3 else "right" if center_x > 2 * width / 3 else "center"
        conservative = self.calibration.conservative(track.filtered_depth_m)
        calibrated = self.calibration.calibrated(track.filtered_depth_m)
        ttc = None
        if conservative is not None and track.relative_velocity_mps is not None and track.relative_velocity_mps < -0.03:
            ttc = conservative / max(0.03, -track.relative_velocity_mps)
        return SceneObject(
            track_id=track.track_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox_xyxy=track.bbox_xyxy,
            zone=zone,
            bearing_deg=bearing,
            raw_depth_m=track.filtered_depth_m,
            calibrated_depth_m=calibrated,
            conservative_depth_m=conservative,
            relative_velocity_mps=track.relative_velocity_mps,
            ttc_s=ttc,
            depth_valid_ratio=None,
            track_age_frames=track.age_frames,
            stable=track.age_frames >= int(self.config["tracking"]["min_stable_frames"]),
            last_seen_ns=track.last_seen_ns,
        )

    @staticmethod
    def _corridor(depth_grid: dict[str, Any] | None) -> WalkingCorridor:
        if not depth_grid:
            return WalkingCorridor(None, None, None, 0.0)
        clearance = depth_grid.get("clearance_m", {})
        return WalkingCorridor(
            left_clearance_m=_as_positive_float(clearance.get("left")),
            center_clearance_m=_as_positive_float(clearance.get("center")),
            right_clearance_m=_as_positive_float(clearance.get("right")),
            confidence=float(depth_grid.get("confidence", 0.0)),
            advisory_only=bool(depth_grid.get("advisory_only", True)),
            floor_mode=str(depth_grid.get("floor_mode", "disabled")),
        )


def _as_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if converted > 0 else None


def _as_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None
