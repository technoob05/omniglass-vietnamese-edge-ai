"""Measure local VieNeu rendering without starting the coordinator service."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from .config import load_config
from .tts import VieneuTtsBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", default="Có chai ở phía trước. Khoảng cách chưa được hiệu chuẩn.")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    backend = VieneuTtsBackend(config["tts"], Path(config["storage"]["root"]))
    results = []
    for _ in range(max(1, args.repeat)):
        started = time.monotonic_ns()
        paths = backend.render(args.text, Path(config["storage"]["root"]) / "outputs" / "tts-benchmark")
        elapsed_s = (time.monotonic_ns() - started) / 1e9
        duration_s = sum(sf.info(path).frames / sf.info(path).samplerate for path in paths)
        results.append({
            "elapsed_s": round(elapsed_s, 3),
            "audio_duration_s": round(duration_s, 3),
            "real_time_factor": round(elapsed_s / max(duration_s, 0.001), 3),
            "files": [str(path) for path in paths],
        })
    print(json.dumps({
        "ok": True,
        "runs": results,
        "health": backend.health(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
