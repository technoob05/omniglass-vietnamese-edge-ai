"""Deterministic safety policy.  Generative models never enter this module."""

from __future__ import annotations

import time

from .models import AlertPriority, HazardEvent, HazardLevel, SceneObject, SceneSnapshot


class RiskEngine:
    def __init__(self, config: dict) -> None:
        self.config = config
        self._next_event_id = 1
        self._last_emitted_ns: dict[tuple[str, int | None], int] = {}

    def evaluate(self, scene: SceneSnapshot, now_ns: int | None = None) -> list[HazardEvent]:
        now_ns = now_ns or time.monotonic_ns()
        if not scene.camera_ok or not scene.depth_ok:
            return self._emit_if_due(
                now_ns, HazardLevel.P0_SYSTEM_BLIND, AlertPriority.EMERGENCY,
                "Dừng lại, hệ thống không nhìn thấy.", "camera_or_depth_unhealthy", None,
                True, float(self.config["system_blind_repeat_s"]),
            )
        if not scene.depth_calibrated and not bool(self.config["enable_uncalibrated_motion_alerts"]):
            return []
        events: list[HazardEvent] = []
        for item in scene.objects:
            if not item.stable:
                continue
            if item.zone != "center":
                continue
            level = self._object_level(item)
            if level is None:
                continue
            if level == HazardLevel.P1_IMMINENT:
                events.extend(self._emit_if_due(
                    now_ns, level, AlertPriority.EMERGENCY, "Dừng. Vật cản phía trước.",
                    "object_imminent", item.track_id, True,
                    float(self.config["event_cooldown_s"]),
                ))
            else:
                distance = _speak_distance(item.conservative_depth_m)
                label = _vi_label(item.label)
                events.extend(self._emit_if_due(
                    now_ns, level, AlertPriority.WARNING,
                    f"Có {label} phía trước, khoảng {distance}.", "object_warning", item.track_id,
                    False, float(self.config["event_cooldown_s"]),
                ))
        return events

    def _object_level(self, item: SceneObject) -> HazardLevel | None:
        distance = item.conservative_depth_m
        ttc = item.ttc_s
        if distance is None:
            return None
        if distance <= float(self.config["p1_distance_m"]) or (ttc is not None and ttc <= float(self.config["p1_ttc_s"])):
            return HazardLevel.P1_IMMINENT
        if distance <= float(self.config["p2_distance_m"]) or (ttc is not None and ttc <= float(self.config["p2_ttc_s"])):
            return HazardLevel.P2_WARNING
        return None

    def _emit_if_due(
        self,
        now_ns: int,
        level: HazardLevel,
        priority: AlertPriority,
        text: str,
        reason: str,
        track_id: int | None,
        interrupt: bool,
        cooldown_s: float,
    ) -> list[HazardEvent]:
        key = (level.value, track_id)
        previous = self._last_emitted_ns.get(key, 0)
        if now_ns - previous < int(cooldown_s * 1e9):
            return []
        self._last_emitted_ns[key] = now_ns
        event = HazardEvent(
            event_id=self._next_event_id,
            created_ns=now_ns,
            level=level,
            priority=priority,
            text_vi=text,
            reason=reason,
            track_id=track_id,
            interrupt_audio=interrupt,
        )
        self._next_event_id += 1
        return [event]


def _speak_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "gần"
    if distance_m < 1.0:
        # Keep this phrasing unambiguous for a safety message.  In particular,
        # "8 mươi" is not valid Vietnamese for 0.8 m.
        centimeters = max(1, int(round(distance_m * 100)))
        return f"{centimeters} xen ti mét"
    return f"{distance_m:.1f} mét".replace(".", " phẩy ")


def _vi_label(label: str) -> str:
    return {
        "person": "người",
        "chair": "ghế",
        "bottle": "chai",
        "backpack": "ba lô",
        "dining table": "bàn",
        "car": "xe hơi",
        "bicycle": "xe đạp",
    }.get(label, "vật cản")
