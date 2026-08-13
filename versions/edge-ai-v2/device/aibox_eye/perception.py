"""Read-only adapter for the established single-engine QNN HTTP service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import RawDetection


@dataclass(frozen=True)
class PerceptionUpdate:
    detections: list[RawDetection]
    health: dict[str, Any]
    depth_grid: dict[str, Any] | None
    received_ns: int


class QnnPerceptionAdapter:
    """Polls QNN outputs; it never opens the camera or creates a QNN engine."""

    def __init__(self, base_url: str, timeout_seconds: float = 0.35) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._last_success_ns: int | None = None
        self._last_error = "not polled"

    @property
    def last_error(self) -> str:
        return self._last_error

    def poll(self, include_depth_grid: bool = True) -> PerceptionUpdate:
        received_ns = time.monotonic_ns()
        try:
            try:
                # One atomic response means the camera service only acquires
                # the Python GIL once per safety tick.  Three 10 Hz requests
                # measurably degraded its 21 FPS QNN postprocess on QCS8550.
                payload = self._get_json("/perception")
                health = dict(payload["health"])
                detections = [RawDetection.from_mapping(item) for item in payload["detections"]]
                depth_grid = payload.get("depth_grid") if include_depth_grid else None
            except RuntimeError as error:
                if "HTTP 404" not in str(error):
                    raise
                # Compatibility for an old baseline while it is being staged.
                health = self._get_json("/health")
                detections = [RawDetection.from_mapping(item) for item in self._get_json("/detections")]
                depth_grid = None
                if include_depth_grid:
                    try:
                        depth_grid = self._get_json("/depth-grid")
                    except RuntimeError as grid_error:
                        if "HTTP 404" not in str(grid_error):
                            raise
            self._last_success_ns = received_ns
            self._last_error = ""
            published_ns = health.get("published_monotonic_ns")
            if isinstance(published_ns, (int, float)) and 0 <= published_ns <= received_ns:
                health["adapter_age_ms"] = (received_ns - int(published_ns)) / 1e6
            else:
                # Compatibility with the pre-extension baseline.  The actual
                # first deployment exposes a monotonic publish timestamp.
                health["adapter_age_ms"] = 0.0
            return PerceptionUpdate(detections, health, depth_grid, received_ns)
        except Exception as error:
            self._last_error = str(error)
            age_ms = float("inf") if self._last_success_ns is None else (received_ns - self._last_success_ns) / 1e6
            return PerceptionUpdate(
                [],
                {"status": "error", "error": self._last_error, "adapter_age_ms": age_ms, "rates": {}},
                None,
                received_ns,
            )

    def latest_raw_jpeg(self) -> bytes:
        """Get a raw camera JPEG, never an overlay, for a VLM request."""
        return self._get_bytes("/snapshot.jpg")

    def _get_json(self, path: str) -> Any:
        payload = self._get_bytes(path)
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid JSON from perception {path}: {error}") from error

    def _get_bytes(self, path: str) -> bytes:
        request = urllib.request.Request(self.base_url + path, headers={"Cache-Control": "no-store"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} from perception {path}")
                return response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code} from perception {path}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Perception unavailable: {error.reason}") from error
