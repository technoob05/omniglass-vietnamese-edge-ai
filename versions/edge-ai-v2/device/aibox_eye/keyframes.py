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


def prepare_for_vlm(data: bytes, max_width: int = 768, jpeg_quality: int = 88) -> tuple[bytes, dict[str, int]]:
    """Resize a raw camera frame once before base64/API encoding.

    768 px preserves substantially more text detail than a thumbnail while
    avoiding the visual-token and transfer cost of the 1280 px camera frame.
    """
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG cannot be decoded")
    source_height, source_width = image.shape[:2]
    if source_width > max_width:
        scale = max_width / float(source_width)
        image = cv2.resize(
            image,
            (max_width, max(1, int(round(source_height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    output_height, output_width = image.shape[:2]
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("JPEG cannot be encoded")
    payload = encoded.tobytes()
    return payload, {
        "source_width": source_width,
        "source_height": source_height,
        "width": output_width,
        "height": output_height,
        "jpeg_bytes": len(payload),
    }
