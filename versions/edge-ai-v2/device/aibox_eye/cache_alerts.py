"""Render immutable local safety prompts after the TTS backend passes QA."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import load_config
from .tts import VieneuTtsBackend


_PROMPTS = {
    "p0_system_blind.wav": "Dừng lại. Hệ thống không nhìn thấy.",
    "p1_stop.wav": "Dừng. Vật cản phía trước.",
    "p2_warning.wav": "Cẩn thận. Có vật cản phía trước.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cached AIBOX-eye safety WAV assets")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["tts"]["backend"] != "vieneu_onnx":
        raise SystemExit("Set tts.backend to vieneu_onnx only after the local runtime is installed")
    storage = Path(config["storage"]["root"])
    target = storage / "assets" / "cached_alerts"
    staging = storage / "outputs" / "cache-build"
    target.mkdir(parents=True, exist_ok=True)
    backend = VieneuTtsBackend(config["tts"], storage)
    for filename, text in _PROMPTS.items():
        outputs = backend.render(text, staging)
        destination = target / filename
        if len(outputs) == 1:
            shutil.copy2(outputs[0], destination)
        else:
            chunks = []
            sample_rate: int | None = None
            for path in outputs:
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                if sample_rate is None:
                    sample_rate = rate
                if rate != sample_rate:
                    raise RuntimeError(f"Unexpected cache sample-rate mismatch: {rate} != {sample_rate}")
                chunks.append(audio)
            assert sample_rate is not None
            # Keep a natural, brief sentence boundary while ensuring safety
            # playback has one immutable, atomic file to pre-empt narration.
            gap = np.zeros((int(0.12 * sample_rate), chunks[0].shape[1]), dtype=np.float32)
            joined: list[np.ndarray] = []
            for index, chunk in enumerate(chunks):
                if index:
                    joined.append(gap)
                joined.append(chunk)
            sf.write(destination, np.concatenate(joined, axis=0), sample_rate)
        print(f"created {target / filename}")


if __name__ == "__main__":
    main()
