"""Push-to-talk recorder and pluggable Whisper backends.

No QNN backend is enabled here by default: an additional HTP context is unsafe
until it passes the concurrency soak gate defined in the specification.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


class PushToTalkRecorder:
    def __init__(self, device: str, recordings_dir: Path) -> None:
        self.device = device
        self.recordings_dir = recordings_dir
        self._process: subprocess.Popen | None = None
        self._path: Path | None = None

    @property
    def recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> Path:
        if self.recording:
            raise RuntimeError("Push-to-talk recording already active")
        if not self.device:
            raise RuntimeError("No stable ALSA capture device configured")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.recordings_dir / f"ptt-{time.monotonic_ns()}.wav"
        command = ["arecord", "-D", self.device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", str(self._path)]
        self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return self._path

    def stop(self) -> Path:
        if not self.recording or self._process is None or self._path is None:
            raise RuntimeError("No active push-to-talk recording")
        process = self._process
        process.send_signal(signal.SIGINT)
        _stdout, stderr = process.communicate(timeout=4.0)
        path = self._path
        self._process = None
        self._path = None
        # arecord's exit code on SIGINT is not portable: alsa-utils 1.2.6 on
        # this AIBOX image returns 1 ("Aborted by signal Interrupt"), while
        # other builds return the POSIX-conventional 128+SIGINT = 130. Either
        # is a normal, intentional stop, not a failure; the file-validity
        # check below is the real correctness gate.
        if process.returncode not in (0, 1, 130):
            raise RuntimeError(f"arecord failed: {stderr.strip()}")
        if not path.is_file() or path.stat().st_size < 44:
            raise RuntimeError(f"Recorder did not create valid WAV: {stderr.strip()}")
        return path

    def abort(self) -> None:
        """Stop an active recording when an emergency safety message arrives."""
        process = self._process
        self._process = None
        self._path = None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.communicate(timeout=1.0)


class DisabledSttBackend:
    def health(self) -> dict[str, Any]:
        return {"ok": False, "backend": "disabled", "detail": "Whisper backend is not installed/configured"}

    def transcribe(self, _audio_path: Path) -> str:
        raise RuntimeError("STT backend is disabled")


class WhisperCliBackend:
    """Wraps a pinned local whisper.cpp command; no network API is involved."""

    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        self.command_template = str(config["command"])
        self.model = str(config["model"])
        self.language = str(config["language"])
        self.vad_enabled = bool(config.get("vad_enabled", False))
        self.vad_model = str(config.get("vad_model", ""))
        self.vad_threshold = float(config.get("vad_threshold", 0.50))
        self.vad_min_speech_ms = int(config.get("vad_min_speech_ms", 250))
        self.vad_min_silence_ms = int(config.get("vad_min_silence_ms", 400))
        self.initial_prompt = str(config.get("initial_prompt", ""))
        self.output_dir = output_dir

    def health(self) -> dict[str, Any]:
        command = self.command_template.split(" ", 1)[0] if self.command_template else ""
        return {
            "ok": bool(
                self.command_template
                and self.model
                and Path(self.model).is_file()
                and (not self.vad_enabled or (self.vad_model and Path(self.vad_model).is_file()))
            ),
            "backend": "whisper_cli_cpu",
            "command": command,
            "model": self.model,
            "model_name": Path(self.model).name,
            "accelerator": "cpu_neon",
            "inference_mode": "vad_bounded_final",
            "audio_context_frames": 512,
            "audio_context_seconds": 10.24,
            "vad": {
                "enabled": self.vad_enabled,
                "model": self.vad_model,
                "threshold": self.vad_threshold,
                "min_speech_ms": self.vad_min_speech_ms,
                "min_silence_ms": self.vad_min_silence_ms,
            },
        }

    def transcribe(self, audio_path: Path) -> str:
        if not self.command_template or not self.model:
            raise RuntimeError("Whisper CLI command/model not configured")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_base = self.output_dir / f"whisper-{time.monotonic_ns()}"
        command = self._command(audio_path, output_base)
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Whisper CLI failed: {result.stderr.strip()}")
        json_path = output_base.with_suffix(".json")
        if json_path.is_file():
            value = json.loads(json_path.read_text(encoding="utf-8"))
            text = value.get("transcription") or value.get("text") or ""
            if isinstance(text, list):
                text = " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in text)
            return str(text).strip()
        # Keep a known, explicit fallback for alternate Whisper command builds.
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _command(self, audio_path: Path, output_base: Path) -> list[str]:
        command_text = self.command_template.format(
            audio=str(audio_path),
            model=self.model,
            language=self.language,
            output=str(output_base),
            vad_model=self.vad_model,
            vad_threshold=self.vad_threshold,
            vad_min_speech_ms=self.vad_min_speech_ms,
            vad_min_silence_ms=self.vad_min_silence_ms,
            initial_prompt=self.initial_prompt,
        )
        return shlex.split(command_text)
