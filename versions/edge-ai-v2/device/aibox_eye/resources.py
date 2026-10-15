"""Read-only resource telemetry and admission checks for non-safety work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import threading
import time
from typing import Any


@dataclass(frozen=True)
class ResourceSnapshot:
    npu_temperature_c: float | None
    npu_zone_temperatures_c: dict[str, float]
    process_rss_bytes: int | None
    gpu_busy_percent: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceGuard:
    """Never blocks the safety loop; only declines optional VLM work."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.thermal_root = Path(str(config["thermal_root"]))
        self.npu_zone_prefix = str(config["npu_zone_prefix"])
        self.min_detector_fps = float(config["min_detector_fps_for_vlm"])
        self.max_npu_temperature_c = float(config["max_npu_temperature_c_for_vlm"])
        self._lock = threading.Lock()
        self._updated_at = 0.0
        self._cached = ResourceSnapshot(None, {}, None, None)

    def snapshot(self) -> ResourceSnapshot:
        # Thermal sysfs has many zones on this platform. Re-reading them at
        # the 5 Hz safety cadence wastes CPU; one fresh sample per second is
        # enough for admission of an optional VLM request.
        with self._lock:
            if time.monotonic() - self._updated_at < 1.0:
                return self._cached
            snapshot = self._read_snapshot()
            self._cached = snapshot
            self._updated_at = time.monotonic()
            return snapshot

    def _read_snapshot(self) -> ResourceSnapshot:
        temperatures: dict[str, float] = {}
        for zone in sorted(self.thermal_root.glob("thermal_zone*")):
            try:
                name = (zone / "type").read_text(encoding="utf-8").strip()
                if not name.startswith(self.npu_zone_prefix):
                    continue
                raw = int((zone / "temp").read_text(encoding="utf-8").strip())
                # Qualcomm thermal values are normally milli-degrees C.  Keep
                # the guard usable on platforms which report integral C.
                value = raw / 1000.0 if abs(raw) > 1000 else float(raw)
                if -20.0 <= value <= 150.0:
                    temperatures[name] = value
            except (OSError, ValueError):
                continue
        return ResourceSnapshot(
            npu_temperature_c=max(temperatures.values(), default=None),
            npu_zone_temperatures_c=temperatures,
            process_rss_bytes=_self_rss_bytes(),
            gpu_busy_percent=_gpu_busy_percent(),
        )

    def vlm_admission(self, detector_fps: float | None, snapshot: ResourceSnapshot) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if detector_fps is None or detector_fps < self.min_detector_fps:
            shown = "unknown" if detector_fps is None else f"{detector_fps:.1f}"
            reasons.append(f"detector_fps={shown} below {self.min_detector_fps:.1f}")
        if snapshot.npu_temperature_c is not None and snapshot.npu_temperature_c > self.max_npu_temperature_c:
            reasons.append(
                f"npu_temperature_c={snapshot.npu_temperature_c:.1f} above {self.max_npu_temperature_c:.1f}"
            )
        return not reasons, reasons


def _self_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _gpu_busy_percent() -> float | None:
    """Read Qualcomm KGSL's ``busy total`` counter when that driver exposes it."""
    try:
        values = Path("/sys/class/kgsl/kgsl-3d0/gpubusy").read_text(encoding="utf-8").split()
        busy, total = int(values[0]), int(values[1])
        return None if total <= 0 else max(0.0, min(100.0, 100.0 * busy / total))
    except (OSError, ValueError, IndexError):
        return None
