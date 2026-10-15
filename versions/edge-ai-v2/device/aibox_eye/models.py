"""Typed, JSON-serialisable contracts shared by all AIBOX-eye processes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SystemMode(str, Enum):
    BOOTING = "booting"
    SELF_TEST = "self_test"
    READY = "ready"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    ROUTING = "routing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ALERTING = "alerting"
    SYSTEM_BLIND = "system_blind"
    DEGRADED = "degraded"


class AlertPriority(int, Enum):
    INFORMATION = 20
    ACKNOWLEDGEMENT = 70
    EARCON = 90
    WARNING = 95
    EMERGENCY = 100


class HazardLevel(str, Enum):
    NONE = "none"
    P3_INFORMATION = "p3_information"
    P2_WARNING = "p2_warning"
    P1_IMMINENT = "p1_imminent"
    P0_SYSTEM_BLIND = "p0_system_blind"


@dataclass(frozen=True)
class RawDetection:
    """A detection exactly as reported by the established QNN service."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    estimated_distance_m: float | None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RawDetection":
        return cls(
            label=str(value.get("label", "unknown")),
            confidence=float(value.get("confidence", 0.0)),
            x1=int(value.get("x1", 0)),
            y1=int(value.get("y1", 0)),
            x2=int(value.get("x2", 0)),
            y2=int(value.get("y2", 0)),
            estimated_distance_m=(
                None
                if value.get("estimated_distance_m") is None
                else float(value["estimated_distance_m"])
            ),
        )


@dataclass
class SceneObject:
    track_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    zone: str
    bearing_deg: float | None
    raw_depth_m: float | None
    calibrated_depth_m: float | None
    conservative_depth_m: float | None
    relative_velocity_mps: float | None
    ttc_s: float | None
    depth_valid_ratio: float | None
    track_age_frames: int
    stable: bool
    last_seen_ns: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox_xyxy"] = list(self.bbox_xyxy)
        return value


@dataclass(frozen=True)
class WalkingCorridor:
    left_clearance_m: float | None
    center_clearance_m: float | None
    right_clearance_m: float | None
    confidence: float
    advisory_only: bool = True
    floor_mode: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSnapshot:
    frame_id: int
    captured_at_ns: int
    published_at_ns: int
    frame_age_ms: float
    camera_ok: bool
    depth_ok: bool
    depth_calibrated: bool
    objects: list[SceneObject] = field(default_factory=list)
    walking_corridor: WalkingCorridor = field(
        default_factory=lambda: WalkingCorridor(None, None, None, 0.0)
    )
    source: str = "qnn_adapter"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "captured_at_ns": self.captured_at_ns,
            "published_at_ns": self.published_at_ns,
            "frame_age_ms": round(self.frame_age_ms, 3),
            "camera_ok": self.camera_ok,
            "depth_ok": self.depth_ok,
            "depth_calibrated": self.depth_calibrated,
            "objects": [item.to_dict() for item in self.objects],
            "walking_corridor": self.walking_corridor.to_dict(),
            "source": self.source,
        }


@dataclass(frozen=True)
class HazardEvent:
    event_id: int
    created_ns: int
    level: HazardLevel
    priority: AlertPriority
    text_vi: str
    reason: str
    track_id: int | None = None
    interrupt_audio: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["level"] = self.level.value
        value["priority"] = int(self.priority)
        return value


@dataclass(frozen=True)
class AudioRequest:
    request_id: int
    priority: AlertPriority
    text_vi: str | None
    cached_asset: str | None
    interrupt: bool
    source: str
    created_ns: int


@dataclass(frozen=True)
class Intent:
    name: str
    requires_vlm: bool
    object_label: str | None = None


@dataclass
class ComponentHealth:
    name: str
    ok: bool
    detail: str = ""
    updated_ns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
