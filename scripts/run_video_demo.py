#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from omniglass.memory import MemoryIndex
from omniglass.perception import SharedPerceptionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run OmniGlass shared perception on one video")
    parser.add_argument("video")
    parser.add_argument("--output", default="runs/cli")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--watch", default="laptop")
    parser.add_argument("--question", default="Trước mặt tôi có gì?")
    args = parser.parse_args()

    pipeline = SharedPerceptionPipeline(args.model, args.device)
    memory, video, memory_json, keyframe, timings = pipeline.process_video(
        args.video,
        Path(args.output),
        sampled_fps=args.fps,
        max_seconds=args.max_seconds,
        watch_target=args.watch,
    )
    answer, evidence = MemoryIndex(memory).answer(args.question)
    payload = {
        "answer": answer,
        "evidence": evidence,
        "annotated_video": video,
        "memory_json": memory_json,
        "last_keyframe": keyframe,
        "processed_frames": memory.processed_frames,
        "labels": Counter(item.label for item in memory.observations),
        "watch_events": [item.to_dict() for item in memory.watch_events],
        "timings_seconds": timings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
