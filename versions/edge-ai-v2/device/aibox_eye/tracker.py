"""Small deterministic IoU tracker; no extra neural model is allowed here."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RawDetection


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


@dataclass
class Track:
    track_id: int
    label: str
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    age_frames: int
    missed_frames: int
    last_seen_ns: int
    previous_depth_m: float | None = None
    previous_depth_ns: int | None = None
    filtered_depth_m: float | None = None
    depth_samples: tuple[float, ...] = ()
    relative_velocity_mps: float | None = None


class IoUTracker:
    """Greedy label-aware matcher suitable for the small detection count here."""

    def __init__(self, iou_threshold: float, max_missed_frames: int) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def update(self, detections: list[RawDetection], now_ns: int) -> list[tuple[Track, RawDetection]]:
        unmatched_track_ids = set(self._tracks)
        unmatched_detection_ids = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if track.label != detection.label:
                    continue
                score = iou(track.bbox_xyxy, (detection.x1, detection.y1, detection.x2, detection.y2))
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, index))
        matches: list[tuple[Track, RawDetection]] = []
        for _score, track_id, index in sorted(candidates, reverse=True):
            if track_id not in unmatched_track_ids or index not in unmatched_detection_ids:
                continue
            track = self._tracks[track_id]
            detection = detections[index]
            track.bbox_xyxy = (detection.x1, detection.y1, detection.x2, detection.y2)
            track.confidence = detection.confidence
            track.age_frames += 1
            track.missed_frames = 0
            track.last_seen_ns = now_ns
            self._update_depth(track, detection.estimated_distance_m, now_ns)
            matches.append((track, detection))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_ids.remove(index)
        for index in sorted(unmatched_detection_ids):
            detection = detections[index]
            track = Track(
                track_id=self._next_track_id,
                label=detection.label,
                bbox_xyxy=(detection.x1, detection.y1, detection.x2, detection.y2),
                confidence=detection.confidence,
                age_frames=1,
                missed_frames=0,
                last_seen_ns=now_ns,
            )
            self._next_track_id += 1
            self._update_depth(track, detection.estimated_distance_m, now_ns)
            self._tracks[track.track_id] = track
            matches.append((track, detection))
        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]
        return matches

    @staticmethod
    def _update_depth(track: Track, depth_m: float | None, now_ns: int) -> None:
        if depth_m is None or depth_m <= 0:
            return
        samples = (*track.depth_samples[-6:], float(depth_m))
        ordered = sorted(samples)
        median = ordered[len(ordered) // 2]
        if track.filtered_depth_m is not None and track.previous_depth_ns is not None:
            elapsed_s = (now_ns - track.previous_depth_ns) / 1e9
            if elapsed_s > 0:
                track.relative_velocity_mps = (median - track.filtered_depth_m) / elapsed_s
        track.previous_depth_m = track.filtered_depth_m
        track.previous_depth_ns = now_ns
        track.filtered_depth_m = median
        track.depth_samples = samples
