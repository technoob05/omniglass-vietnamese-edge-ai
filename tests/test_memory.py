import json

import pytest

from omniglass.memory import MemoryIndex, format_time, load_memory, save_memory
from omniglass.schema import Observation, SessionMemory, WatchEvent


def build_memory():
    return SessionMemory(
        session_id="demo",
        source_name="demo.mp4",
        source_fps=30,
        sampled_fps=2,
        processed_frames=10,
        duration_seconds=5,
        observations=[
            Observation(0, 0, "laptop", 0.9, (0, 0, 10, 10), "1", "bên trái", "a.jpg"),
            Observation(4, 8, "laptop", 0.8, (20, 0, 30, 10), "1", "bên phải", "b.jpg"),
            Observation(0, 0, "bottle", 0.7, (0, 0, 10, 10), "2", "chính giữa", "a.jpg"),
        ],
        watch_events=[
            WatchEvent(
                4,
                8,
                "laptop",
                "screen_region_changed",
                "laptop đã đổi vùng trong khung hình; có thể do vật hoặc camera di chuyển",
            ),
            WatchEvent(5, 9, "laptop", "missing", "Không còn thấy laptop trong khung hình"),
        ],
    )


def test_last_seen_query():
    answer, evidence = MemoryIndex(build_memory()).answer("Laptop lần cuối ở đâu?")
    assert "bên phải" in answer
    assert "00:04" in answer
    assert evidence == "b.jpg"


def test_watch_query():
    answer, _ = MemoryIndex(build_memory()).answer("Laptop có bị di chuyển không?")
    assert "đổi vùng" in answer


def test_missing_intent_does_not_use_scene_change():
    answer, _ = MemoryIndex(build_memory()).answer("Laptop có bị mất dấu không?")
    assert "Không còn thấy laptop" in answer


def test_time_rounding_does_not_emit_second_60():
    assert format_time(59.6) == "01:00"


def test_memory_roundtrip_resolves_relative_evidence(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    evidence = frames / "a.jpg"
    evidence.write_bytes(b"image")
    memory = build_memory()
    path = save_memory(memory, tmp_path / "memory.json")
    raw = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    assert raw["observations"][0]["keyframe_path"] == "a.jpg"
    raw["observations"][0]["keyframe_path"] = "frames/a.jpg"
    (tmp_path / "memory.json").write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_memory(path)
    assert loaded.observations[0].bbox_xyxy == (0, 0, 10, 10)
    assert loaded.observations[0].keyframe_path == str(evidence.resolve())


def test_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text('{"schema": "future.v9"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported memory schema"):
        load_memory(path)


def test_known_but_unseen_object():
    answer, evidence = MemoryIndex(build_memory()).answer("Where was the chair last seen?")
    assert "chưa thấy chair" in answer
    assert evidence is None
