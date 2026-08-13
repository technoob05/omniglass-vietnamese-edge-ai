"""VieNeu TTS integration. Cached emergency speech is rendered during setup only."""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator


def normalize_vi_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = text.replace("~", "khoảng")
    return text


def phrase_chunks(text: str, maximum_words: int = 12) -> list[str]:
    """Bound first-audio latency without cutting within a Vietnamese word."""
    sentences = re.split(r"(?<=[.!?;:])\s+", normalize_vi_text(text))
    chunks: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        while len(words) > maximum_words:
            chunks.append(" ".join(words[:maximum_words]))
            words = words[maximum_words:]
        if words:
            chunks.append(" ".join(words))
    return chunks or [text]


class DisabledTtsBackend:
    def health(self) -> dict[str, Any]:
        return {"ok": False, "backend": "disabled", "detail": "VieNeu-TTS is not installed/configured"}

    def render(self, _text: str, _output_dir: Path) -> list[Path]:
        raise RuntimeError("TTS backend is disabled")

    def render_chunks(self, _text: str, _output_dir: Path) -> Iterator[Path]:
        raise RuntimeError("TTS backend is disabled")
        yield Path()


class VieneuTtsBackend:
    """Lazy-load VieNeu so a missing model cannot stop the safety loop."""

    def __init__(self, config: dict[str, Any], storage_root: Path) -> None:
        self.config = config
        self.storage_root = storage_root
        self._engine: Any | None = None
        self._load_error = "not loaded"

    def health(self) -> dict[str, Any]:
        return {
            "ok": self._engine is not None,
            "backend": "vieneu",
            "detail": self._load_error,
            "voice": self.config["voice"],
            "tempo": self.config["tempo"],
        }

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from vieneu import Vieneu  # type: ignore
            model_dir = self.storage_root / str(self.config.get("model_dir", "models/vieneu"))
            graph_dir = model_dir / "onnx_int8"
            codec_dir = model_dir / "codec"
            if not graph_dir.is_dir() or not codec_dir.is_dir():
                raise FileNotFoundError(
                    f"VieNeu local artifacts are missing: {graph_dir} or {codec_dir}"
                )
            # VieNeu v3 Turbo exposes an ONNX INT8 CPU backend.  Pin both
            # values instead of relying on automatic torch/GPU detection, so
            # installation of an unrelated accelerator never changes latency.
            self._engine = Vieneu(
                backbone_repo=str(model_dir),
                backend="onnx",
                precision="int8",
                threads=int(self.config.get("threads", 3)),
                onnx_dir=str(graph_dir),
                codec_dir=str(codec_dir),
            )
            self._load_error = ""
            return self._engine
        except Exception as error:
            self._load_error = str(error)
            raise RuntimeError(f"Cannot load VieNeu ONNX backend: {error}") from error

    def render(self, text: str, output_dir: Path) -> list[Path]:
        return list(self.render_chunks(text, output_dir))

    def render_chunks(self, text: str, output_dir: Path) -> Iterator[Path]:
        """Yield each playable phrase as soon as it is ready.

        The audio manager can start the first phrase while the next one is
        synthesised.  This lowers perceived dialogue latency without ever
        delaying pre-recorded P0/P1 safety audio.
        """
        engine = self._load()
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, phrase in enumerate(phrase_chunks(text)):
            stem = f"tts-{time.monotonic_ns()}-{index}"
            source = output_dir / f"{stem}-source.wav"
            output = output_dir / f"{stem}.wav"
            audio = engine.infer(normalize_vi_text(phrase), voice=self.config["voice"])
            engine.save(audio, str(source))
            self._tempo(source, output)
            yield output

    def warm(self) -> None:
        """Load local ONNX sessions before a user asks the first question."""
        self._load()

    def _tempo(self, source: Path, output: Path) -> None:
        tempo = float(self.config["tempo"])
        # atempo preserves pitch and is an explicit safe fallback when a
        # GStreamer segment-rate pipeline is unavailable.  0.5..2.0 is valid.
        command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-filter:a", f"atempo={tempo:.4f}", str(output)]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg tempo filter failed: {result.stderr.strip()}")
