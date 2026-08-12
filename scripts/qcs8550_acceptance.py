#!/usr/bin/env python3
"""Evaluate QCS8550 physical evidence against the frozen release gates.

This validator does not perform benchmarks and never turns a candidate into a verified release on
its own. A passing report means that the supplied evidence is ready for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "qcs8550_deployment.json"


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def result(passed: bool, reasons: list[str]) -> dict[str, Any]:
    return {"status": "pass_candidate" if passed else "blocked", "reasons": reasons}


def evaluate(manifest: dict, preflight: dict, metrics: dict) -> dict[str, Any]:
    targets = manifest["provisional_targets_not_measurements"]
    board = preflight.get("target_detection", {})
    measurements = metrics.get("measurements", {})
    evidence = metrics.get("evidence", {})

    identity_ok = board.get("qcs8550_detected") is True
    identity = result(identity_ok, [] if identity_ok else ["QCS8550 identity was not detected"])

    hashes = evidence.get("artifact_sha256", [])
    integrity_ok = bool(hashes) and all(valid_sha256(value) for value in hashes)
    offline_ok = measurements.get("offline_boot_ok") is True
    offline = result(
        offline_ok and integrity_ok,
        [
            message
            for condition, message in (
                (offline_ok, "Offline boot did not pass"),
                (integrity_ok, "No complete artifact SHA-256 evidence"),
            )
            if not condition
        ],
    )

    io_ok = measurements.get("camera_audio_io_ok") is True
    io_gate = result(io_ok, [] if io_ok else ["Camera/audio timestamp and routing test is missing"])

    accuracy_ok = measurements.get("model_accuracy_passed") is True and valid_sha256(
        evidence.get("accuracy_report_sha256")
    )
    accuracy = result(
        accuracy_ok,
        [] if accuracy_ok else ["Held-out task accuracy report is missing or did not pass"],
    )

    latency_checks = (
        (
            measurements.get("detector_tracker_sustained_fps", -1)
            >= targets["detector_tracker_sustained_fps_min"],
            "Detector/tracker sustained FPS is below target",
        ),
        (
            measurements.get("critical_camera_to_audio_p95_ms", float("inf"))
            <= targets["critical_camera_to_audio_p95_ms_max"],
            "Critical camera-to-audio P95 exceeds target",
        ),
        (
            measurements.get("hybrid_vqa_first_text_p95_ms", float("inf"))
            <= targets["hybrid_vqa_first_text_p95_ms_max"],
            "Hybrid VQA first-text P95 exceeds target",
        ),
        (
            measurements.get("hybrid_vqa_first_audio_p95_ms", float("inf"))
            <= targets["hybrid_vqa_first_audio_p95_ms_max"],
            "Hybrid VQA first-audio P95 exceeds target",
        ),
        (measurements.get("htp_fallback_events") == 0, "HTP fallback was observed or not reported"),
    )
    latency = result(
        all(check for check, _ in latency_checks),
        [message for check, message in latency_checks if not check],
    )

    thermal_checks = (
        (measurements.get("soak_minutes", -1) >= targets["soak_minutes"], "Soak is too short"),
        (
            measurements.get("thermal_latency_regression", float("inf"))
            <= targets["thermal_latency_regression_max"],
            "Thermal latency regression exceeds target",
        ),
        (measurements.get("unbounded_memory_growth") is False, "Memory growth was observed or not reported"),
        (measurements.get("power_thermal_trace_present") is True, "Power/thermal trace is missing"),
    )
    thermal = result(
        all(check for check, _ in thermal_checks),
        [message for check, message in thermal_checks if not check],
    )

    network_ok = measurements.get("network_loss_core_actions_ok") is True
    network = result(
        network_ok,
        [] if network_ok else ["Offline detect/track/stop/help/critical-speech test is missing"],
    )

    privacy_checks = (
        (measurements.get("raw_media_in_logs") is False, "Raw media was found in logs or not checked"),
        (measurements.get("encrypted_transport") is True, "Encrypted transport is missing"),
        (measurements.get("mic_cloud_indicator") is True, "Microphone/cloud indicator is missing"),
    )
    privacy = result(
        all(check for check, _ in privacy_checks),
        [message for check, message in privacy_checks if not check],
    )

    turns = measurements.get("turns", -1)
    listening_rate = measurements.get("return_to_listening_rate", -1.0)
    turn_ok = turns >= targets["turns"] and listening_rate >= targets["return_to_listening_rate_min"]
    turn_gate = result(
        turn_ok,
        [] if turn_ok else ["100-turn return-to-listening target was not met"],
    )

    gates = {
        "identity": identity,
        "offline_boot_and_integrity": offline,
        "camera_audio_io": io_gate,
        "model_accuracy": accuracy,
        "latency_percentiles": latency,
        "memory_power_thermal": thermal,
        "network_loss_fallback": network,
        "privacy_security": privacy,
        "hundred_turn_soak": turn_gate,
    }
    all_pass = all(item["status"] == "pass_candidate" for item in gates.values())
    return {
        "schema_version": "1.0",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "ready_for_human_release_review" if all_pass else "blocked",
        "claim_boundary": "A validator pass is not physical-board verification or safety certification.",
        "gates": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.preflight.read_text(encoding="utf-8")),
        json.loads(args.metrics.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if report["decision"] != "ready_for_human_release_review":
        print("QCS8550 release evidence remains blocked.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
