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


def test_gateway_patch_removes_legacy_face_routes(tmp_path: Path) -> None:
    module = load(ROOT / "scripts" / "patch_native_h100_perception.py", "patch_h100_face_rollback")
    gateway = tmp_path / "gateway.py"
    gateway.write_text(
        f'''{module.MARKER}\nasync def _vi_perception_proxy(path, payload=None):\n    pass\n\n'''
        f'''{module.LEGACY_FACE_MARKER}\n@app.post("/api/perception/vi/faces/enroll")\n'''
        '''async def vietnamese_perception_face_enroll(request: Request):\n'''
        '''    return await _vi_perception_proxy("/faces/enroll", await request.json())\n\n'''
        '''@app.get("/api/presets")\nasync def get_presets():\n    return []\n''',
        encoding="utf-8",
    )
    assert module.patch_gateway(gateway) is True
    source = gateway.read_text(encoding="utf-8")
    assert module.LEGACY_FACE_MARKER not in source
    assert "/faces/enroll" not in source


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
    assert "0.7 mét" in safety["primary_alert"]["message_vi"]
    assert "0.7 meters" in safety["primary_alert"]["message_en"]
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


def test_depth_sectors_and_open_side_guidance_are_conservative() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_perception_guidance")
    depth = __import__("numpy").full((100, 120), 2.0, dtype="float32")
    depth[38:88, 43:76] = 0.8
    depth[38:88, 6:38] = 2.6
    depth[38:88, 81:114] = 1.6
    sectors = module.depth_sector_summary(depth)
    assert sectors == {"left": 2.6, "center": 0.8, "right": 1.6}
    guidance = module.evaluate_assistive_guidance([], depth, 1000, {})
    alert = guidance["primary_alert"]
    assert alert["rule"] == "center_surface_side_opening"
    assert "bên trái có vẻ thoáng hơn" in alert["message_vi"].lower()
    assert "kiểm tra bằng gậy" in alert["message_vi"].lower()
    assert guidance["navigation_certified"] is False


def test_wall_distance_uses_depth_plane_and_rejects_large_object_occlusion() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_wall_distance")
    np = __import__("numpy")
    height, width = 120, 160
    yy, xx = np.indices((height, width), dtype="float32")
    depth = 1.8 + 0.15 * (xx / width - .5) + 0.08 * (yy / height - .5)
    wall = module.estimate_front_wall(depth, [])
    assert wall["wall_likely"] is True
    assert 1.7 <= wall["distance_m_advisory"] <= 1.9
    person = module.Observation("person", "người", .9, (.22,.2,.78,.7), "ở giữa", 1, 1000, 1.0)
    occluded = module.estimate_front_wall(depth, [person])
    assert occluded["wall_likely"] is False


def test_wall_safety_warns_even_when_neither_side_is_open() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_wall_safety")
    wall = {"wall_likely": True, "distance_m_advisory": 1.1, "confidence": .86, "planar_inlier_ratio": .91}
    alert, previous = module.evaluate_surface_safety(wall, None, 1000, {})
    assert alert["severity"] == "warning"
    assert alert["rule"] == "front_wall_depth_plane"
    assert "1.1 mét" in alert["message_vi"]
    closer, _ = module.evaluate_surface_safety({**wall, "distance_m_advisory": .7}, previous, 1800, {})
    assert closer["severity"] == "danger"
    assert closer["approaching"] is True
    assert "đang tới rất gần tường" in closer["message_vi"]


def test_toilet_and_sink_create_bathroom_landmark_with_distance() -> None:
    module = load(ROOT / "scripts" / "h100_perception_service.py", "h100_perception_bathroom")
    toilet = module.Observation("toilet", "bồn cầu", .88, (.35,.3,.62,.9), "ở giữa", 1, 1000, 1.4)
    sink = module.Observation("sink", "bồn rửa", .77, (.65,.2,.9,.7), "bên phải", 2, 1000, 1.8)
    guidance = module.evaluate_assistive_guidance([toilet, sink], None, 1000, {})
    alert = guidance["primary_alert"]
    assert alert["rule"] == "toilet_landmark"
    assert "1.4 mét" in alert["message_vi"]
    assert "có thể là khu vệ sinh" in alert["message_vi"]
    assert alert["severity"] == "guidance"


def test_native_ui_injects_bounded_perception_context() -> None:
    source = (ROOT / "native-overrides" / "vi-profile" / "vi-chat.js").read_text(encoding="utf-8")
    assert "PERCEPTION_INTERVAL_MS = 650" in source
    assert "detections = (data.detections || []).slice(0,10)" in source
    assert "memory = (data.memory || []).slice(0,12)" in source
    assert "depth_calibrated:false" in source
    assert "{role:'system', content:this.perceptionContext()}" in source
    assert "enqueueSafetyAlert(data.safety?.primary_alert)" in source
    assert "assistiveGuidance" in source
    assert "TTS_PLAYBACK_RATE = 1.5" in source
