from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Observation:
    timestamp: float
    frame_index: int
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: str | None
    location: str
    keyframe_path: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WatchEvent:
    timestamp: float
    frame_index: int
    target: str
    event: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionMemory:
    session_id: str
    source_name: str
    source_fps: float
    sampled_fps: float
    processed_frames: int
    duration_seconds: float
    observations: list[Observation] = field(default_factory=list)
    watch_events: list[WatchEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "omniglass.observation-memory.v1",
            "session_id": self.session_id,
            "source_name": self.source_name,
            "source_fps": self.source_fps,
            "sampled_fps": self.sampled_fps,
            "processed_frames": self.processed_frames,
            "duration_seconds": self.duration_seconds,
            "observations": [item.to_dict() for item in self.observations],
            "watch_events": [item.to_dict() for item in self.watch_events],
            "metadata": self.metadata,
        }
