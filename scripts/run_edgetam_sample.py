#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Run the official EdgeTAM checkpoint on a video")
    parser.add_argument("video")
    parser.add_argument("--repo", default="upstream/EdgeTAM")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--point", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/edgetam/tracked.mp4")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else repo / "checkpoints/edgetam.pt"
    video_path = Path(args.video).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"EdgeTAM repository not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EdgeTAM checkpoint not found: {checkpoint}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    probe = cv2.VideoCapture(str(video_path))
    if not probe.isOpened():
        raise ValueError(f"Cannot decode input video: {video_path}")
    source_fps = float(probe.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    x, y = args.point
    if not (0 <= x < source_width and 0 <= y < source_height):
        raise ValueError(
            f"Prompt point {(x, y)} is outside video bounds {(source_width, source_height)}"
        )

    sys.path.insert(0, str(repo))
    from sam2.build_sam import build_sam2_video_predictor

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    end_to_end_started = time.perf_counter()
    load_started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        "edgetam.yaml", str(checkpoint), device=device
    )
    model_load_seconds = time.perf_counter() - load_started
    init_started = time.perf_counter()
    state = predictor.init_state(video_path=str(video_path))
    video_init_seconds = time.perf_counter() - init_started
    points = np.asarray([args.point], dtype=np.float32)
    labels = np.asarray([1], dtype=np.int32)

    segments = {}
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    propagation_started = time.perf_counter()
    with torch.inference_mode(), amp:
        predictor.add_new_points(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            points=points,
            labels=labels,
        )
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(state):
            index = [int(value) for value in object_ids].index(0)
            segments[int(frame_idx)] = (mask_logits[index] > 0).cpu().numpy().squeeze()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    propagation_seconds = time.perf_counter() - propagation_started
    if not segments:
        raise RuntimeError("EdgeTAM emitted no masks")

    encode_started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    fps, width, height = source_fps, source_width, source_height
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {destination}")
    frame_idx = 0
    areas = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = segments.get(frame_idx)
        if mask is not None:
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            mask = mask.astype(bool)
            areas.append(float(mask.mean()))
            overlay = np.zeros_like(frame)
            overlay[:, :] = (48, 210, 110)
            frame[mask] = cv2.addWeighted(frame[mask], 0.45, overlay[mask], 0.55, 0)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (35, 255, 130), 2)
        writer.write(frame)
        frame_idx += 1
    capture.release()
    writer.release()
    encode_seconds = time.perf_counter() - encode_started

    verification = cv2.VideoCapture(str(destination))
    if not verification.isOpened():
        raise RuntimeError(f"Written output cannot be decoded: {destination}")
    output_frames = int(verification.get(cv2.CAP_PROP_FRAME_COUNT))
    output_width = int(verification.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(verification.get(cv2.CAP_PROP_FRAME_HEIGHT))
    verification.release()
    if output_frames != frame_idx or (output_width, output_height) != (width, height):
        raise RuntimeError(
            f"Output validation failed: frames={output_frames}/{frame_idx}, "
            f"shape={(output_width, output_height)}/{(width, height)}"
        )

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    try:
        edge_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        edge_commit = None
    properties = torch.cuda.get_device_properties(device) if device.startswith("cuda") else None

    report = {
        "schema": "omniglass.edgetam-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "facebookresearch/EdgeTAM official checkpoint",
        "edgetam_commit": edge_commit,
        "checkpoint": os.path.relpath(checkpoint, Path.cwd()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "video": os.path.relpath(video_path, Path.cwd()),
        "prompt_point_xy": args.point,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if properties else None,
        "gpu_total_memory_bytes": properties.total_memory if properties else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "warmup_runs": 0,
        "source_frames_reported": source_frames,
        "frames": frame_idx,
        "tracked_frames": len(segments),
        "source_fps": fps,
        "model_load_seconds": model_load_seconds,
        "video_init_seconds": video_init_seconds,
        "prompt_and_propagation_seconds": propagation_seconds,
        "prompt_and_propagation_fps": len(segments) / max(propagation_seconds, 1e-9),
        "video_encode_seconds": encode_seconds,
        "end_to_end_seconds": time.perf_counter() - end_to_end_started,
        "mean_mask_area_ratio": float(np.mean(areas)) if areas else 0.0,
        "output_video": os.path.relpath(destination.resolve(), Path.cwd()),
        "output_validation": {
            "decodable": True,
            "frames": output_frames,
            "width": output_width,
            "height": output_height,
        },
    }
    report_path = destination.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
