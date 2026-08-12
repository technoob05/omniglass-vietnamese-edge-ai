#!/usr/bin/env python3
"""Collect a read-only QCS8550 deployment inventory.

The report intentionally makes no camera or microphone capture and cannot promote a deployment
gate to verified. It records enough host facts to start a reproducible board bring-up.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "qcs8550_deployment.json"


def read_text(path: str | Path, limit: int = 64_000) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[:limit].strip()
    except (OSError, PermissionError):
        return None


def run(command: list[str], timeout: int = 8) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if not executable:
        return {"command": command, "available": False}
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": command,
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout[:64_000].strip(),
            "stderr": result.stderr[:16_000].strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "available": True, "error": type(exc).__name__}


def hash_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(target), "bytes": target.stat().st_size, "sha256": digest.hexdigest()}
    except (OSError, PermissionError) as exc:
        return {"path": str(target), "error": type(exc).__name__}


def glob_text(pattern: str, value_limit: int = 4_000) -> list[dict[str, Any]]:
    values = []
    for path in sorted(glob.glob(pattern)):
        values.append({"path": path, "value": read_text(path, value_limit)})
    return values


def android_properties() -> dict[str, Any]:
    names = (
        "ro.board.platform",
        "ro.hardware",
        "ro.product.manufacturer",
        "ro.product.model",
        "ro.product.name",
        "ro.build.version.release",
        "ro.build.version.security_patch",
        "ro.build.fingerprint",
    )
    if not shutil.which("getprop"):
        return {"available": False, "values": {}}
    values = {}
    for name in names:
        result = run(["getprop", name])
        values[name] = result.get("stdout") if result.get("returncode") == 0 else None
    return {"available": True, "values": values}


def qnn_libraries(environment: dict[str, str | None]) -> list[dict[str, Any]]:
    """Hash only relevant runtime libraries under explicit SDK/runtime paths."""
    sdk_roots: set[Path] = set()
    runtime_roots: set[Path] = set()
    for name in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT"):
        if environment.get(name):
            sdk_roots.add(Path(str(environment[name])))
    for name in ("ADSP_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        for value in str(environment.get(name) or "").split(os.pathsep):
            if value:
                runtime_roots.add(Path(value))
    candidates: set[Path] = set()
    patterns = ("libQnnHtp*.so", "libQnnSystem.so", "libqai_appbuilder*.so")
    for root in sdk_roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.update(root.glob(f"**/{pattern}"))
    for root in runtime_roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.update(root.glob(pattern))
    return [hash_file(path) for path in sorted(candidates)[:100]]


def detect_qcs8550(evidence: list[str]) -> bool:
    joined = "\n".join(evidence).lower()
    return any(token in joined for token in ("qcs8550", "qcs-8550", "kimq 8550"))


def collect(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    os_release = read_text("/etc/os-release")
    soc_files = []
    if os.name == "posix":
        for name in ("machine", "family", "soc_id", "revision"):
            path = f"/sys/devices/soc0/{name}"
            if Path(path).exists():
                soc_files.append({"path": path, "value": read_text(path, 2_000)})
    android_props = android_properties()
    identity_evidence = [platform.platform(), platform.machine(), os_release or ""]
    identity_evidence.extend(str(item.get("value") or "") for item in soc_files)
    identity_evidence.extend(str(value or "") for value in android_props["values"].values())
    target_detected = detect_qcs8550(identity_evidence)

    qnn_env = {
        name: os.environ.get(name)
        for name in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT", "ADSP_LIBRARY_PATH", "LD_LIBRARY_PATH")
    }
    qnn_tools = {
        name: shutil.which(name)
        for name in ("qnn-net-run", "qnn-context-binary-generator", "qnn-platform-validator")
    }
    qnn_hashes = [hash_file(path) for path in qnn_tools.values() if path]

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "collector": {
            "script": str(Path(__file__).resolve()),
            "python": sys.version,
            "privacy": "read-only inventory; no camera frames or microphone audio captured",
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "schema_version": manifest["schema_version"],
            "expected_product": manifest["target"]["product"],
        },
        "target_detection": {
            "qcs8550_detected": target_detected,
            "status": "inventory_candidate" if target_detected else "not_detected",
            "claim_boundary": "Detection is not deployment verification.",
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uname": list(platform.uname()),
            "os_release": os_release,
            "soc_sysfs": soc_files,
            "android_getprop": android_props,
            "meminfo": read_text("/proc/meminfo"),
            "cpuinfo": read_text("/proc/cpuinfo"),
            "disk_usage_root": (
                dict(zip(("total", "used", "free"), shutil.disk_usage("/")))
                if os.name == "posix"
                else None
            ),
        },
        "qnn": {
            "environment": qnn_env,
            "tools": qnn_tools,
            "tool_versions": {
                name: run([name, "--version"]) if path else {"available": False}
                for name, path in qnn_tools.items()
            },
            "tool_hashes": qnn_hashes,
            "runtime_library_hashes": qnn_libraries(qnn_env),
            "ldconfig_matches": (
                run(["sh", "-lc", "ldconfig -p | grep -Ei 'Qnn|qai' | head -200"])
                if os.name == "posix"
                else {"available": False}
            ),
        },
        "io": {
            "video_devices": sorted(glob.glob("/dev/video*")),
            "v4l2_devices": run(["v4l2-ctl", "--list-devices"]),
            "audio_capture": run(["arecord", "-l"]),
            "audio_playback": run(["aplay", "-l"]),
        },
        "telemetry": {
            "thermal_types": glob_text("/sys/class/thermal/thermal_zone*/type"),
            "thermal_millidegrees_c": glob_text("/sys/class/thermal/thermal_zone*/temp"),
            "cpu_frequency_khz": glob_text("/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq"),
        },
        "release_gates": {
            gate["id"]: "blocked_pending_physical_evidence" for gate in manifest["mandatory_gates"]
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path")
    parser.add_argument(
        "--require-qcs8550",
        action="store_true",
        help="Return exit code 2 when a QCS8550 identity is not detected",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect(args.manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.require_qcs8550 and not report["target_detection"]["qcs8550_detected"]:
        print("QCS8550 identity not detected; physical-board gate remains blocked.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
