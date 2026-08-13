"""Strict configuration loading for the local, no-system-write deployment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "api": {"host": "0.0.0.0", "port": 8090},
    "perception": {
        "base_url": "http://127.0.0.1:8080",
        # 5 Hz is a 200 ms safety-state cadence; combined with the already
        # running 21 FPS detector it stays below the 250 ms alert budget while
        # avoiding Python-GIL contention in the QNN HTTP service.
        "poll_hz": 5.0,
        "stale_after_ms": 500.0,
        "snapshot_path": "/snapshot.jpg",
        "depth_grid_path": "/depth-grid",
        "source_width": 1280,
        "source_height": 720,
    },
    "calibration": {
        "enabled": False,
        "scale": 1.0,
        "exponent": 1.0,
        "offset_m": 0.0,
        "dangerous_overestimate_p99_m": 0.0,
        "valid_min_m": 0.30,
        "valid_max_m": 5.0,
        "camera_intrinsics": {"fx": None, "fy": None, "cx": 640.0, "cy": 360.0},
    },
    "tracking": {"iou_threshold": 0.30, "max_missed_frames": 8, "min_stable_frames": 3},
    "risk": {
        "enable_uncalibrated_motion_alerts": False,
        "enable_dense_grid_hazards": False,
        "p1_distance_m": 0.80,
        "p2_distance_m": 1.60,
        "p1_ttc_s": 1.20,
        "p2_ttc_s": 2.50,
        "event_cooldown_s": 2.50,
        "system_blind_repeat_s": 4.0,
    },
    "audio": {
        "capture_device": "",
        "playback_device": "",
        "playback_backend": "auto",
        "sample_rate": 48000,
        "tempo": 1.5,
        "asset_dir": "assets",
        "half_duplex": True,
    },
    "stt": {
        "backend": "disabled",
        "command": "",
        "model": "",
        "language": "vi",
        "vad_enabled": True,
        "vad_model": "",
        "vad_threshold": 0.50,
        "vad_min_speech_ms": 250,
        "vad_min_silence_ms": 400,
        "initial_prompt": "vật cản, khoảng cách, phía trước, bên trái, bên phải",
    },
    "tts": {
        "backend": "disabled",
        "voice": "Thái Sơn",
        "tempo": 1.5,
        "threads": 3,
        "model_dir": "models/vieneu",
        "asset_dir": "assets/cached_alerts",
    },
    "vlm": {
        "backend": "disabled",
        "base_url": "http://127.0.0.1:8081",
        "model": "Vintern-1B-v3_5-Q8_0",
        "timeout_seconds": 30.0,
        "max_tokens": 48,
        "temperature": 0.1,
        "max_frame_age_ms": 1000.0,
    },
    "resource_guard": {
        # Interactive VLM work is best-effort. It is rejected before it can
        # compete with safety perception on an already-hot HTP subsystem.
        "min_detector_fps_for_vlm": 18.0,
        "max_npu_temperature_c_for_vlm": 75.0,
        "thermal_root": "/sys/class/thermal",
        "npu_zone_prefix": "nspss",
    },
    "storage": {"root": "/data/local/tmp/aibox-eye", "max_event_history": 256},
}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON only, avoiding a new YAML dependency on the target."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        patch = json.load(handle)
    if not isinstance(patch, dict):
        raise ValueError("Configuration root must be an object")
    config = _merge(config, patch)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not 1 <= int(config["api"]["port"]) <= 65535:
        raise ValueError("api.port must be 1..65535")
    if float(config["perception"]["poll_hz"]) <= 0:
        raise ValueError("perception.poll_hz must be positive")
    if float(config["perception"]["stale_after_ms"]) <= 0:
        raise ValueError("perception.stale_after_ms must be positive")
    if not 0.5 <= float(config["audio"]["tempo"]) <= 2.0:
        raise ValueError("audio.tempo must be in 0.5..2.0")
    if not 0.0 <= float(config["stt"]["vad_threshold"]) <= 1.0:
        raise ValueError("stt.vad_threshold must be in 0..1")
    if int(config["stt"]["vad_min_speech_ms"]) < 0 or int(config["stt"]["vad_min_silence_ms"]) < 0:
        raise ValueError("stt VAD duration values must be non-negative")
    resource_guard = config["resource_guard"]
    if float(resource_guard["min_detector_fps_for_vlm"]) <= 0:
        raise ValueError("resource_guard.min_detector_fps_for_vlm must be positive")
    if not 40.0 <= float(resource_guard["max_npu_temperature_c_for_vlm"]) <= 110.0:
        raise ValueError("resource_guard.max_npu_temperature_c_for_vlm must be in 40..110")
    calibration = config["calibration"]
    if float(calibration["valid_min_m"]) <= 0 or float(calibration["valid_max_m"]) <= float(calibration["valid_min_m"]):
        raise ValueError("Invalid calibration distance range")
