from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Observation, SessionMemory, WatchEvent


LABEL_ALIASES = {
    "điện thoại": "cell phone",
    "dien thoai": "cell phone",
    "điện thoại của tôi": "cell phone",
    "chìa khóa": "keys",
    "chia khoa": "keys",
    "chai nước": "bottle",
    "chai nuoc": "bottle",
    "balo": "backpack",
    "ba lô": "backpack",
    "máy tính": "laptop",
    "may tinh": "laptop",
    "ghế": "chair",
    "ghe": "chair",
    "cốc": "cup",
    "ly": "cup",
    "người": "person",
    "nguoi": "person",
}

KNOWN_LABELS = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat",
    "traffic light", "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "chair", "couch", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "keys", "door",
}


def normalize_query_label(query: str, available_labels: Iterable[str]) -> str | None:
    text = query.lower().strip()
    for alias, canonical in sorted(LABEL_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in text:
            return canonical
    labels = sorted(set(available_labels) | KNOWN_LABELS, key=len, reverse=True)
    for label in labels:
        if re.search(rf"\b{re.escape(label.lower())}\b", text):
            return label
    return None


class MemoryIndex:
    def __init__(self, memory: SessionMemory):
        self.memory = memory
        self.by_label: dict[str, list[Observation]] = defaultdict(list)
        for observation in memory.observations:
            self.by_label[observation.label].append(observation)

    @property
    def labels(self) -> list[str]:
        return sorted(self.by_label)

    def latest(self, label: str) -> Observation | None:
        items = self.by_label.get(label, [])
        return max(items, key=lambda item: (item.timestamp, item.confidence)) if items else None

    def answer(self, question: str) -> tuple[str, str | None]:
        text = question.lower().strip()
        label = normalize_query_label(text, self.labels)

        moved_intent = any(token in text for token in ("di chuyển", "di chuyen", "moved", "move"))
        missing_intent = any(token in text for token in ("mất dấu", "mat dau", "missing", "không còn thấy"))
        watch_intent = any(token in text for token in ("canh", "watch"))

        if any(token in text for token in ("thay đổi", "thay doi", "changed")):
            before, after = self._boundary_labels()
            added = sorted(after - before)
            removed = sorted(before - after)
            chunks = []
            if added:
                chunks.append("xuất hiện: " + ", ".join(added))
            if removed:
                chunks.append("không còn thấy: " + ", ".join(removed))
            return ("Thay đổi chính: " + "; ".join(chunks), None) if chunks else (
                "Không phát hiện thay đổi ổn định giữa khung đầu và khung cuối video.", None
            )

        if moved_intent or missing_intent or watch_intent:
            events = [event for event in self.memory.watch_events if label is None or event.target == label]
            if moved_intent:
                events = [event for event in events if event.event == "screen_region_changed"]
            elif missing_intent:
                events = [event for event in events if event.event == "missing"]
            if not events:
                target = label or self.memory.metadata.get("watch_target") or "đối tượng"
                event_name = "đổi vùng màn hình" if moved_intent else "mất khỏi khung hình" if missing_intent else "watch"
                return f"Không phát hiện sự kiện {event_name} rõ ràng cho {target}.", None
            event = events[-1]
            return f"{event.detail} tại {format_time(event.timestamp)}.", None

        if label:
            observation = self.latest(label)
            if observation is None:
                return f"Tôi chưa thấy {label} trong video này.", None
            return (
                f"Lần cuối tôi thấy {label} ở {observation.location}, vào {format_time(observation.timestamp)} "
                f"(độ tin cậy {observation.confidence:.0%}).",
                observation.keyframe_path,
            )

        latest_time = max((item.timestamp for item in self.memory.observations), default=0.0)
        if any(token in text for token in ("trong video", "đã thấy", "da thay", "ever seen")):
            visible = self.memory.observations
            prefix = "Trong video tôi đã thấy"
        else:
            visible = [item for item in self.memory.observations if item.timestamp >= latest_time - 1.5]
            prefix = "Trong các khung hình gần nhất tôi thấy"
        counts = Counter(item.label for item in visible)
        latest_labels = sorted(
            counts,
            key=lambda name: max(item.confidence for item in visible if item.label == name),
            reverse=True,
        )
        if not latest_labels:
            return "Tôi chưa ghi nhận được đối tượng nào.", None
        description = ", ".join(latest_labels[:8])
        return f"{prefix}: {description}.", None

    def _boundary_labels(self) -> tuple[set[str], set[str]]:
        if not self.memory.observations:
            return set(), set()
        cutoff = max(self.memory.duration_seconds * 0.2, 0.5)
        end_start = max(self.memory.duration_seconds - cutoff, 0)
        before_counts = Counter(x.label for x in self.memory.observations if x.timestamp <= cutoff)
        after_counts = Counter(x.label for x in self.memory.observations if x.timestamp >= end_start)
        before = {label for label, count in before_counts.items() if count >= 2}
        after = {label for label, count in after_counts.items() if count >= 2}
        return before, after


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, remaining = divmod(total, 60)
    return f"{minutes:02d}:{remaining:02d}"


def save_memory(memory: SessionMemory, path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(memory.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination)


def load_memory(path: str | Path) -> SessionMemory:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema") != "omniglass.observation-memory.v1":
        raise ValueError(f"Unsupported memory schema: {raw.get('schema')}")
    observations = []
    for item in raw.get("observations", []):
        item = dict(item)
        item["bbox_xyxy"] = tuple(item["bbox_xyxy"])
        keyframe = item.get("keyframe_path")
        if keyframe and not Path(keyframe).is_absolute():
            item["keyframe_path"] = str((source.parent / keyframe).resolve())
        observations.append(Observation(**item))
    return SessionMemory(
        session_id=raw["session_id"],
        source_name=raw["source_name"],
        source_fps=float(raw["source_fps"]),
        sampled_fps=float(raw["sampled_fps"]),
        processed_frames=int(raw["processed_frames"]),
        duration_seconds=float(raw["duration_seconds"]),
        observations=observations,
        watch_events=[WatchEvent(**item) for item in raw.get("watch_events", [])],
        metadata=raw.get("metadata", {}),
    )
