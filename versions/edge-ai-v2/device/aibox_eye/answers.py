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
    ordered = _deduplicated_context_objects(scene.objects)[:8]
    corridor = scene.walking_corridor
    return {
        "frame_id": scene.frame_id,
        "frame_age_ms": round(scene.frame_age_ms, 1),
        "depth_calibrated": scene.depth_calibrated,
        "objects": [
            {
                "label": item.label,
                "zone": item.zone,
                "confidence": round(item.confidence, 2),
                "bbox_percent": [round(100.0 * value / size) for value, size in zip(item.bbox_xyxy, (1280, 720, 1280, 720))],
                "distance_m": None if item.calibrated_depth_m is None else round(item.calibrated_depth_m, 2),
                "stable": item.stable,
            }
            for item in ordered
        ],
        # Raw monocular-depth magnitudes are useful internally for ordering,
        # but must never become spoken metres before device calibration.
        "walking_corridor": {
            "left_m": corridor.left_clearance_m if scene.depth_calibrated and not corridor.advisory_only else None,
            "center_m": corridor.center_clearance_m if scene.depth_calibrated and not corridor.advisory_only else None,
            "right_m": corridor.right_clearance_m if scene.depth_calibrated and not corridor.advisory_only else None,
            "advisory_only": corridor.advisory_only,
            "floor_mode": corridor.floor_mode,
        },
    }


def detection_summary(scene: SceneSnapshot, limit: int = 3) -> str:
    objects = _deduplicated_context_objects(scene.objects)[:limit]
    if not objects:
        return "YOLO chưa phát hiện vật thể rõ ràng trong khung hình hiện tại."
    zones = {"left": "bên trái", "center": "ở giữa", "right": "bên phải"}
    phrases = [f"{_vi_label(item.label)} {zones[item.zone]}" for item in objects]
    return "YOLO hiện thấy " + ", ".join(phrases) + "."


def _deduplicated_context_objects(objects: list[SceneObject]) -> list[SceneObject]:
    ordered = sorted(objects, key=lambda item: (not item.stable, -item.confidence))
    kept: list[SceneObject] = []
    for candidate in ordered:
        duplicate = any(
            candidate.label == existing.label
            and _intersection_over_smaller(candidate.bbox_xyxy, existing.bbox_xyxy) >= 0.80
            for existing in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def _intersection_over_smaller(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return 0.0 if smaller <= 0 else intersection / smaller


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
