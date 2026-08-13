"""Vietnamese command routing that keeps simple questions away from the VLM."""

from __future__ import annotations

import re
import unicodedata

from .models import Intent


_OBJECT_WORDS = {
    "người": "person", "nguoi": "person", "chai": "bottle", "ba lô": "backpack",
    "balo": "backpack", "ghế": "chair", "ghe": "chair", "bàn": "dining table",
    "ban": "dining table", "xe": "car",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def route(text: str) -> Intent:
    value = normalize(text)
    object_label = next((label for word, label in _OBJECT_WORDS.items() if word in value), None)
    if any(phrase in value for phrase in ("hệ thống", "camera", "có nhìn", "trạng thái")):
        return Intent("system_status", False)
    if any(phrase in value for phrase in ("bao nhiêu", "mấy người", "số người")):
        return Intent("count", False, object_label or "person")
    if any(phrase in value for phrase in ("bên trái", "bên phải", "có trống", "trống không")):
        return Intent("clearance", False)
    if any(phrase in value for phrase in ("bao xa", "khoảng cách", "cách bao nhiêu", "gần nhất")):
        return Intent("nearest", False, object_label)
    if any(phrase in value for phrase in ("phía trước có gì", "trước mặt có gì", "có gì phía trước")):
        return Intent("front", False, object_label)
    # Natural push-to-talk phrasing is commonly "có người phía trước không?".
    # It is answerable from the current scene and must not wake the VLM.
    if ("phía trước" in value or "trước mặt" in value) and ("có" in value or object_label is not None):
        return Intent("front", False, object_label)
    if any(phrase in value for phrase in ("đọc chữ", "đọc biển", "màu gì", "mô tả", "đây là gì", "đang làm gì", "trên bàn")):
        return Intent("vlm", True, object_label)
    return Intent("vlm", True, object_label)
