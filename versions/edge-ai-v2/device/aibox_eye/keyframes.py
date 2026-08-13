"""Select one fresh, sharp raw frame for an on-demand VLM request."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class KeyframeScore:
    sharpness: float
    brightness: float
    clipped_fraction: float
    score: float


def score_jpeg(data: bytes) -> KeyframeScore:
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG cannot be decoded")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float(np.mean((gray <= 4) | (gray >= 251)))
    # Favour readable exposure, but do not pretend this proves semantic quality.
    exposure = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    score = min(sharpness / 300.0, 1.0) * 0.65 + exposure * 0.25 + (1.0 - clipped) * 0.10
    return KeyframeScore(sharpness, brightness, clipped, score)


def accepts_for_vlm(data: bytes, minimum_sharpness: float = 12.0, maximum_clipped_fraction: float = 0.75) -> tuple[bool, KeyframeScore]:
    metrics = score_jpeg(data)
    return metrics.sharpness >= minimum_sharpness and metrics.clipped_fraction <= maximum_clipped_fraction, metrics
