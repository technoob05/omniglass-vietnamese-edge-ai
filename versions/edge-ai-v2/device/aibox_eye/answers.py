"""Grounded Vietnamese answers generated only from SceneSnapshot facts."""

from __future__ import annotations

from .models import Intent, SceneObject, SceneSnapshot
from .risk import _speak_distance, _vi_label


def answer(intent: Intent, scene: SceneSnapshot) -> str:
    if not scene.camera_ok or not scene.depth_ok:
        return "Hệ thống hiện không nhìn thấy. Bạn hãy dừng lại."
    if intent.name == "system_status":
        calibration = "đã hiệu chuẩn" if scene.depth_calibrated else "chưa hiệu chuẩn khoảng cách"
        return f"Camera đang hoạt động, {calibration}."
    if intent.name == "count":
        items = [item for item in scene.objects if intent.object_label is None or item.label == intent.object_label]
        label = _vi_label(intent.object_label or "person")
        return f"Mình thấy {len(items)} {label}."
    if intent.name == "clearance":
        return _clearance(scene)
    candidates = _candidates(scene, intent.object_label)
    if not candidates:
        return "Mình chưa thấy vật phù hợp trong khung hình hiện tại."
    item = candidates[0]
    position = {"left": "bên trái", "right": "bên phải", "center": "phía trước"}[item.zone]
    label = _vi_label(item.label)
    if item.calibrated_depth_m is None:
        return f"Mình thấy {label} ở {position}. Khoảng cách chưa được hiệu chuẩn."
    return f"Có {label} ở {position}, cách ước lượng khoảng {_speak_distance(item.calibrated_depth_m)}."


def scene_facts(scene: SceneSnapshot) -> dict:
    return {
        "frame_age_ms": round(scene.frame_age_ms, 1),
        "depth_calibrated": scene.depth_calibrated,
        "objects": [
            {
                "label": item.label,
                "zone": item.zone,
                "confidence": round(item.confidence, 2),
                "distance_m": None if item.calibrated_depth_m is None else round(item.calibrated_depth_m, 2),
                "stable": item.stable,
            }
            for item in scene.objects
        ],
        "walking_corridor": scene.walking_corridor.to_dict(),
    }


def _candidates(scene: SceneSnapshot, label: str | None) -> list[SceneObject]:
    values = [item for item in scene.objects if label is None or item.label == label]
    return sorted(values, key=lambda item: (item.calibrated_depth_m is None, item.calibrated_depth_m or 1e9))


def _clearance(scene: SceneSnapshot) -> str:
    corridor = scene.walking_corridor
    if corridor.advisory_only or corridor.center_clearance_m is None:
        return "Khoảng trống đi lại chưa được hiệu chuẩn an toàn."
    return (
        f"Khoảng trống ước lượng: bên trái {_speak_distance(corridor.left_clearance_m)}, "
        f"phía trước {_speak_distance(corridor.center_clearance_m)}, "
        f"bên phải {_speak_distance(corridor.right_clearance_m)}."
    )
