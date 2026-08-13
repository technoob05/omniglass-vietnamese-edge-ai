import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_perception_helpers_and_bounded_memory() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_perception_service")
    assert module.screen_zone((0, 0, 20, 20), 100) == "bên trái"
    assert module.screen_zone((40, 0, 60, 20), 100) == "ở giữa"
    assert module.screen_zone((80, 0, 100, 20), 100) == "bên phải"
    assert module.normalize_target("điện thoại của tôi", {"cell phone"}) == "cell phone"
    state = module.SessionState(history_size=2)
    for index in range(3):
        state.history.append(module.Observation(
            label="person", label_vi="người", confidence=0.9,
            bbox_norm=(0.1, 0.1, 0.2, 0.3), location="bên trái",
            track_id=1, seen_at_ms=index * 100, depth_m=None,
        ))
    assert len(state.history) == 2
    assert state.memory_rows(300)[0]["observations"] == 2


def test_gateway_perception_patch_is_idempotent(tmp_path: Path) -> None:
    module = load(ROOT / "scripts" / "patch_native_h100_perception.py", "patch_h100_perception")
    gateway = tmp_path / "gateway.py"
    gateway.write_text(
        'import os, httpx\nfrom fastapi import FastAPI, Request\n'
        'from fastapi.responses import JSONResponse\napp=FastAPI()\n'
        '@app.get("/api/presets")\nasync def get_presets():\n    return []\n',
        encoding="utf-8",
    )
    assert module.patch_gateway(gateway) is True
    assert module.patch_gateway(gateway) is False
    source = gateway.read_text(encoding="utf-8")
    assert source.count(module.MARKER) == 1
    assert '/api/perception/vi/frame' in source


def test_safety_rules_prioritize_close_center_hazard_and_cool_down() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_perception_safety")
    person = module.Observation(
        label="person", label_vi="người", confidence=0.91,
        bbox_norm=(0.3, 0.2, 0.72, 0.98), location="ở giữa",
        track_id=7, seen_at_ms=1000, depth_m=0.72,
    )
    announced: dict[str, int] = {}
    safety, geometry = module.evaluate_safety([person], {}, 1000, announced)
    assert safety["state"] == "danger"
    assert safety["vlm_used"] is False
    assert safety["primary_alert"]["should_announce"] is True
    assert "rất gần" in safety["primary_alert"]["message_vi"]
    safety_again, _ = module.evaluate_safety([person], geometry, 1800, announced)
    assert safety_again["primary_alert"]["should_announce"] is False


def test_safety_rules_detect_approach_without_depth() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_perception_approach")
    chair = module.Observation(
        label="chair", label_vi="ghế", confidence=0.88,
        bbox_norm=(0.38, 0.45, 0.62, 0.88), location="ở giữa",
        track_id=9, seen_at_ms=2000, depth_m=None,
    )
    safety, _ = module.evaluate_safety([chair], {"chair:9": (0.06, 1200)}, 2000, {})
    assert safety["state"] == "warning"
    assert safety["primary_alert"]["approaching"] is True


def test_native_ui_injects_bounded_perception_context() -> None:
    source = (ROOT / "native-overrides" / "vi-profile" / "vi-chat.js").read_text(encoding="utf-8")
    assert "PERCEPTION_INTERVAL_MS = 650" in source
    assert "detections = (data.detections || []).slice(0,10)" in source
    assert "memory = (data.memory || []).slice(0,12)" in source
    assert "depth_calibrated:false" in source
    assert "{role:'system', content:this.perceptionContext()}" in source
    assert "enqueueSafetyAlert(data.safety?.primary_alert)" in source
    assert "TTS_PLAYBACK_RATE = 1.5" in source
