"""Priority audio playback with explicit cache validation and half-duplex state."""

from __future__ import annotations

import heapq
import itertools
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .models import AlertPriority, AudioRequest


@dataclass
class AudioStatus:
    playing: bool = False
    current_source: str = ""
    current_priority: int = 0
    last_error: str = ""
    submitted: int = 0
    completed: int = 0
    interrupted: int = 0

    def to_dict(self) -> dict:
        return {
            "playing": self.playing,
            "current_source": self.current_source,
            "current_priority": self.current_priority,
            "last_error": self.last_error,
            "submitted": self.submitted,
            "completed": self.completed,
            "interrupted": self.interrupted,
        }


class AudioManager:
    """A bounded, pre-emptive player. P0/P1 never wait behind VLM speech."""

    def __init__(
        self,
        asset_root: Path,
        playback_device: str = "",
        playback_backend: str = "auto",
    ) -> None:
        self.asset_root = asset_root
        self.playback_device = playback_device
        self.playback_backend = playback_backend
        self.status = AudioStatus()
        self._condition = threading.Condition()
        self._queue: list[tuple[int, int, AudioRequest, Path]] = []
        self._sequence = itertools.count()
        self._request_id = itertools.count(1)
        self._current_process: subprocess.Popen | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="aibox-audio", daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        with self._condition:
            return self.status.playing

    def submit_asset(
        self,
        asset_name: str,
        priority: AlertPriority,
        source: str,
        interrupt: bool = False,
        text_vi: str | None = None,
    ) -> int:
        path = self.asset_root / asset_name
        request = AudioRequest(next(self._request_id), priority, text_vi, asset_name, interrupt, source, time.monotonic_ns())
        with self._condition:
            self.status.submitted += 1
            if not path.is_file():
                self.status.last_error = f"Missing cached audio asset: {path}"
                return request.request_id
            if interrupt:
                self._preempt_locked(priority)
            heapq.heappush(self._queue, (-int(priority), next(self._sequence), request, path))
            # Keep the newest high-priority events, cap memory and stale speech.
            if len(self._queue) > 32:
                self._queue = sorted(self._queue)[:32]
                heapq.heapify(self._queue)
            self._condition.notify_all()
        return request.request_id

    def submit_file(
        self,
        path: Path,
        priority: AlertPriority,
        source: str,
        interrupt: bool = False,
        text_vi: str | None = None,
    ) -> int:
        request = AudioRequest(next(self._request_id), priority, text_vi, None, interrupt, source, time.monotonic_ns())
        with self._condition:
            self.status.submitted += 1
            if not path.is_file():
                self.status.last_error = f"Audio output not found: {path}"
                return request.request_id
            if interrupt:
                self._preempt_locked(priority)
            heapq.heappush(self._queue, (-int(priority), next(self._sequence), request, path))
            self._condition.notify_all()
        return request.request_id

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._preempt_locked(AlertPriority.EMERGENCY)
            self._condition.notify_all()
        self._worker.join(timeout=2.0)

    def _preempt_locked(self, priority: AlertPriority) -> None:
        self._queue = [item for item in self._queue if -item[0] >= int(priority)]
        heapq.heapify(self._queue)
        if self._current_process is not None and self.status.current_priority < int(priority):
            self._current_process.terminate()
            self.status.interrupted += 1

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closed)
                if self._closed:
                    return
                _negative_priority, _sequence, request, path = heapq.heappop(self._queue)
                self.status.playing = True
                self.status.current_source = request.source
                self.status.current_priority = int(request.priority)
            command = self._command(path)
            try:
                process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                with self._condition:
                    self._current_process = process
                _stdout, stderr = process.communicate()
                with self._condition:
                    if process.returncode not in (0, -15):
                        self.status.last_error = f"{command[0]} failed: {stderr.strip()}"
                    elif process.returncode == 0:
                        self.status.completed += 1
            except OSError as error:
                with self._condition:
                    self.status.last_error = f"Cannot start {command[0]}: {error}"
            finally:
                with self._condition:
                    self._current_process = None
                    self.status.playing = False
                    self.status.current_source = ""
                    self.status.current_priority = 0
                    self._condition.notify_all()

    def _command(self, path: Path) -> list[str]:
        """Build a direct player command for a service without PulseAudio."""
        use_alsa = self.playback_backend == "alsa" or self.playback_device.startswith(("hw:", "plughw:"))
        if use_alsa:
            command = ["aplay"]
            if self.playback_device:
                command.extend(["-D", self.playback_device])
            command.append(str(path))
            return command
        command = ["paplay"]
        if self.playback_device:
            command.extend(["--device", self.playback_device])
        command.append(str(path))
        return command
