#!/usr/bin/env python3
"""Benchmark isolated QNN context binaries on a physical QCS8550 without starting services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HTP_PATTERN = re.compile(r"(?:QnnHtp|\bHTP\b|Hexagon)", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("No samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def thermal_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            result[str(path)] = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return result


def qnn_command(runner: Path, backend: Path, context: Path, input_list: Path, output: Path) -> list[str]:
    return [
        str(runner),
        "--backend", str(backend),
        "--retrieve_context", str(context),
        "--input_list", str(input_list),
        "--output_dir", str(output),
        "--profiling_level", "basic",
        "--perf_profile", "burst",
    ]


def run_component(
    name: str,
    command: list[str],
    output_root: Path,
    warmups: int,
    runs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    latencies = []
    htp_evidence = False
    logs = []
    for index in range(warmups + runs):
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "QNN_LOG_LEVEL": "debug"},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        log_path = output_root / f"{name}_{index:03d}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        logs.append(str(log_path))
        htp_evidence = htp_evidence or bool(HTP_PATTERN.search(completed.stdout))
        if completed.returncode != 0:
            raise RuntimeError(f"{name} run {index} failed; see {log_path}")
        if index >= warmups:
            latencies.append(elapsed_ms)
    if not htp_evidence:
        raise RuntimeError(f"{name} produced no HTP evidence; silent CPU fallback is rejected")
    return {
        "runs": runs,
        "warmups": warmups,
        "process_wall_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "samples": [round(value, 3) for value in latencies],
        },
        "htp_evidence": True,
        "logs": logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qnn-net-run", type=Path, required=True)
    parser.add_argument("--htp-backend", type=Path, required=True)
    parser.add_argument("--detector-context", type=Path, required=True)
    parser.add_argument("--detector-input-list", type=Path, required=True)
    parser.add_argument("--depth-context", type=Path)
    parser.add_argument("--depth-input-list", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--allow-non-qcs8550", action="store_true")
    args = parser.parse_args()

    if args.runs < 1 or args.warmups < 0:
        parser.error("runs must be positive and warmups non-negative")
    identity = " ".join((platform.platform(), platform.machine()))
    try:
        identity += " " + Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    if not args.allow_non_qcs8550 and not re.search(
        r"QCS\s*8550|QCS8550|KIMQ\s*8550|QCS[_\s-]*KALAMAP", identity, re.I
    ):
        print("Physical QCS8550 identity not detected.", file=sys.stderr)
        return 2

    required = [args.qnn_net_run, args.htp_backend, args.detector_context, args.detector_input_list]
    if bool(args.depth_context) != bool(args.depth_input_list):
        parser.error("depth context and input list must be supplied together")
    required += [path for path in (args.depth_context, args.depth_input_list) if path]
    for path in required:
        if not path.is_file():
            parser.error(f"missing file: {path}")

    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": identity[:500],
        "status": "candidate",
        "claim_boundary": "QNN graph microbenchmarks are not end-to-end camera or safety validation.",
        "thermal_before": thermal_snapshot(),
        "artifact_sha256": {str(path): sha256(path) for path in required},
        "components": {},
    }
    try:
        detector_output = args.output / "detector_outputs"
        detector_output.mkdir()
        report["components"]["detector"] = run_component(
            "detector",
            qnn_command(args.qnn_net_run, args.htp_backend, args.detector_context, args.detector_input_list, detector_output),
            args.output,
            args.warmups,
            args.runs,
            args.timeout_seconds,
        )
        if args.depth_context and args.depth_input_list:
            depth_output = args.output / "depth_outputs"
            depth_output.mkdir()
            report["components"]["depth"] = run_component(
                "depth",
                qnn_command(args.qnn_net_run, args.htp_backend, args.depth_context, args.depth_input_list, depth_output),
                args.output,
                args.warmups,
                args.runs,
                args.timeout_seconds,
            )
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        report["status"] = "blocked"
        report["error"] = str(error)
    report["thermal_after"] = thermal_snapshot()
    (args.output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "candidate" else 4


if __name__ == "__main__":
    raise SystemExit(main())
