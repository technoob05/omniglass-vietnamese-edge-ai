"""Local coordinator for the two-lane AIBOX-eye design.

This process is deliberately a *consumer* of the QNN perception service.  It
does not open /dev/video*, load a QNN model, or create a second FastRPC/HTP
context.  Deterministic perception/risk continues if STT, TTS, or VLM fail.
"""

from __future__ import annotations

import collections
import threading
import time
from pathlib import Path
from typing import Any

from .answers import answer, scene_facts
from .audio import AudioManager
from .intents import route
from .keyframes import accepts_for_vlm, prepare_for_vlm
from .models import AlertPriority, HazardEvent, SceneSnapshot, SystemMode
from .perception import QnnPerceptionAdapter
from .risk import RiskEngine
from .resources import ResourceGuard, ResourceSnapshot
from .scene import SceneBuilder
from .stt import DisabledSttBackend, PushToTalkRecorder, WhisperCliBackend
from .tts import DisabledTtsBackend, VieneuTtsBackend
from .vlm import LlamaVlmClient


_ALERT_ASSETS = {
    "p0_system_blind": "cached_alerts/p0_system_blind.wav",
    "p1_imminent": "cached_alerts/p1_stop.wav",
    "p2_warning": "cached_alerts/p2_warning.wav",
}


class EyeOrchestrator:
    """Thread-safe application service used by the HTTP server and risk loop."""

    def __init__(self, config: dict[str, Any], app_root: Path) -> None:
        self.config = config
        self.app_root = app_root.resolve()
        self.storage_root = Path(config["storage"]["root"])
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.recordings_dir = self.storage_root / "recordings"
        self.outputs_dir = self.storage_root / "outputs"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._scene: SceneSnapshot | None = None
        self._mode = SystemMode.BOOTING
        self._events: collections.deque[HazardEvent] = collections.deque(
            maxlen=int(config["storage"]["max_event_history"])
        )
        self._stats: dict[str, int] = {
            "polls": 0,
            "risk_events": 0,
            "ask_requests": 0,
            "vlm_requests": 0,
            "vlm_failures": 0,
            "tts_requests": 0,
            "tts_failures": 0,
            "stt_requests": 0,
            "stt_failures": 0,
        }

        perception_config = config["perception"]
        self.perception = QnnPerceptionAdapter(str(perception_config["base_url"]))
        self.scene_builder = SceneBuilder(config)
        self.risk = RiskEngine(config["risk"])
        self.resource_guard = ResourceGuard(config["resource_guard"])
        self._resource_snapshot = self.resource_guard.snapshot()
        self._latest_perception_health: dict[str, Any] = {}
        self._vlm_lock = threading.Lock()
        asset_root = _resolve_path(self.app_root, str(config["audio"]["asset_dir"]))
        self.audio = AudioManager(
            asset_root,
            str(config["audio"]["playback_device"]),
            str(config["audio"].get("playback_backend", "auto")),
        )
        self.recorder = PushToTalkRecorder(str(config["audio"]["capture_device"]), self.recordings_dir)
        self.stt = self._make_stt()
        self.tts = self._make_tts()
        self.vlm = self._make_vlm()
        self._tts_warming = False
        self._set_mode(SystemMode.SELF_TEST)

    def _make_stt(self) -> Any:
        if self.config["stt"]["backend"] == "whisper_cli_cpu":
            return WhisperCliBackend(self.config["stt"], self.outputs_dir)
        return DisabledSttBackend()

    def _make_tts(self) -> Any:
        if self.config["tts"]["backend"] == "vieneu_onnx":
            return VieneuTtsBackend(self.config["tts"], self.storage_root)
        return DisabledTtsBackend()

    def _make_vlm(self) -> LlamaVlmClient | None:
        if self.config["vlm"]["backend"] == "llama_cpp":
            return LlamaVlmClient(self.config["vlm"])
        return None

    def start(self) -> None:
        if self._poll_thread is not None:
            return
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="aibox-scene-risk", daemon=True)
        self._poll_thread.start()
        if self.config["tts"]["backend"] == "vieneu_onnx":
            self._tts_warming = True
            threading.Thread(target=self._warm_tts, name="aibox-tts-warm", daemon=True).start()

    def close(self) -> None:
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
        self.recorder.abort()
        self.audio.close()

    def _poll_loop(self) -> None:
        period_s = 1.0 / float(self.config["perception"]["poll_hz"])
        next_tick = time.monotonic()
        while not self._stop.is_set():
            update = self.perception.poll(include_depth_grid=True)
            scene = self.scene_builder.update(update.detections, update.health, update.depth_grid, update.received_ns)
            events = self.risk.evaluate(scene, update.received_ns)
            resource_snapshot = self.resource_guard.snapshot()
            with self._lock:
                self._scene = scene
                self._latest_perception_health = dict(update.health)
                self._resource_snapshot = resource_snapshot
                self._stats["polls"] += 1
                self._set_mode_locked(self._mode_for(scene, events))
            for event in events:
                self._handle_hazard(event)
            next_tick += period_s
            self._stop.wait(max(0.0, next_tick - time.monotonic()))
            # A slow local HTTP request should not make a long, compounding
            # safety-loop backlog.  Drop missed periods instead.
            if next_tick < time.monotonic() - period_s:
                next_tick = time.monotonic()

    def _mode_for(self, scene: SceneSnapshot, events: list[HazardEvent]) -> SystemMode:
        if not scene.camera_ok or not scene.depth_ok:
            return SystemMode.SYSTEM_BLIND
        if events:
            return SystemMode.ALERTING
        if self.recorder.recording:
            return SystemMode.LISTENING
        if self.audio.speaking:
            return SystemMode.SPEAKING
        return SystemMode.READY

    def _handle_hazard(self, event: HazardEvent) -> None:
        # In half-duplex mode, a high-priority alert must be audible rather
        # than competing with microphone capture or VLM/TTS narration.
        if event.interrupt_audio and self.config["audio"]["half_duplex"] and self.recorder.recording:
            self.recorder.abort()
        asset = _ALERT_ASSETS.get(event.level.value)
        if asset:
            self.audio.submit_asset(asset, event.priority, event.level.value, event.interrupt_audio, event.text_vi)
        with self._lock:
            self._events.append(event)
            self._stats["risk_events"] += 1
            self._set_mode_locked(SystemMode.ALERTING)

    def latest_scene(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._scene is None else self._scene.to_dict()

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_dict() for event in reversed(self._events)]

    def health(self) -> dict[str, Any]:
        with self._lock:
            scene = self._scene
            mode = self._mode.value
            stats = dict(self._stats)
            perception_health = dict(self._latest_perception_health)
            resource_snapshot = self._resource_snapshot
        detector_fps = _rate(perception_health, "detector_fps")
        vlm_admitted, vlm_reasons = self.resource_guard.vlm_admission(detector_fps, resource_snapshot)
        assets = {name: (self.audio.asset_root / relative).is_file() for name, relative in _ALERT_ASSETS.items()}
        return {
            "status": "ok" if scene and scene.camera_ok and scene.depth_ok else "degraded",
            "mode": mode,
            "safety": {
                "qnn_engines_owned": 0,
                "distance_calibration_enabled": bool(self.config["calibration"]["enabled"]),
                "motion_hazard_alerts_enabled": bool(self.config["calibration"]["enabled"])
                or bool(self.config["risk"]["enable_uncalibrated_motion_alerts"]),
                "dense_grid_hazards_enabled": bool(self.config["risk"]["enable_dense_grid_hazards"]),
                "cached_alert_assets_ready": all(assets.values()),
                "cached_alert_assets": assets,
            },
            "scene": None if scene is None else {
                "frame_id": scene.frame_id,
                "frame_age_ms": round(scene.frame_age_ms, 1),
                "camera_ok": scene.camera_ok,
                "depth_ok": scene.depth_ok,
            },
            "components": {
                "perception": {"ok": bool(scene and scene.camera_ok and scene.depth_ok), "detail": self.perception.last_error},
                "stt": self.stt.health(),
                "tts": self.tts.health(),
                "vlm": {
                    "ok": self.vlm is not None,
                    "backend": self.config["vlm"]["backend"],
                    "admitted": vlm_admitted,
                    "admission_reasons": vlm_reasons,
                },
                "audio": self.audio.status.to_dict(),
            },
            "tts_warming": self._tts_warming,
            "resources": {
                **resource_snapshot.to_dict(),
                "detector_fps": detector_fps,
                "vlm_admitted": vlm_admitted,
                "vlm_admission_reasons": vlm_reasons,
            },
            "stats": stats,
        }

    def prometheus_metrics(self) -> str:
        health = self.health()
        stats = health["stats"]
        scene = health["scene"] or {}
        lines = [
            "# HELP aibox_eye_poll_total Perception polling iterations.",
            "# TYPE aibox_eye_poll_total counter",
            f"aibox_eye_poll_total {stats['polls']}",
            "# HELP aibox_eye_risk_event_total Deterministic safety events.",
            "# TYPE aibox_eye_risk_event_total counter",
            f"aibox_eye_risk_event_total {stats['risk_events']}",
            f"aibox_eye_frame_age_ms {float(scene.get('frame_age_ms', -1.0))}",
            f"aibox_eye_camera_ok {1 if scene.get('camera_ok') else 0}",
            f"aibox_eye_depth_ok {1 if scene.get('depth_ok') else 0}",
        ]
        for name in ("ask_requests", "vlm_requests", "vlm_failures", "tts_requests", "tts_failures", "stt_requests", "stt_failures"):
            lines.append(f"aibox_eye_{name}_total {stats[name]}")
        return "\n".join(lines) + "\n"

    def ask(self, question_vi: str) -> dict[str, Any]:
        question_vi = question_vi.strip()
        if not question_vi or len(question_vi) > 500:
            raise ValueError("text must contain 1..500 characters")
        with self._lock:
            scene = self._scene
            self._stats["ask_requests"] += 1
            self._set_mode_locked(SystemMode.ROUTING)
        if scene is None:
            raise RuntimeError("Scene is not ready yet")
        intent = route(question_vi)
        if not intent.requires_vlm:
            response = answer(intent, scene)
            self._speak_async(response, "deterministic_answer")
            self._restore_mode()
            return {"intent": intent.name, "answer_vi": response, "source": "scene", "queued_tts": True}
        return self._ask_vlm(question_vi, intent.name, scene)

    def _ask_vlm(self, question_vi: str, intent_name: str, scene: SceneSnapshot) -> dict[str, Any]:
        if self.vlm is None:
            response = "Mô tả chi tiết theo ảnh chưa được cài hoặc benchmark trên thiết bị này."
            self._speak_async(response, "vlm_unavailable")
            self._restore_mode()
            return {"intent": intent_name, "answer_vi": response, "source": "fallback", "queued_tts": True}
        if not self._vlm_lock.acquire(blocking=False):
            response = "Mình đang xử lý một câu hỏi theo ảnh khác. Bạn hãy chờ câu trả lời đó xong."
            self._speak_async(response, "vlm_busy")
            self._restore_mode()
            return {"intent": intent_name, "answer_vi": response, "source": "fallback", "queued_tts": True}
        wait_started = time.monotonic()
        wait_limit = float(self.config["vlm"]["admission_wait_seconds"])
        poll_seconds = float(self.config["vlm"]["admission_poll_seconds"])
        try:
            # Optional VLM work waits for a safe slot instead of immediately
            # failing a user's turn. The independent perception thread keeps
            # polling QNN YOLO/depth throughout this wait.
            while True:
                with self._lock:
                    scene = self._scene or scene
                    perception_health = dict(self._latest_perception_health)
                    resource_snapshot = self._resource_snapshot
                    self._set_mode_locked(SystemMode.THINKING)
                admitted, reasons = self.resource_guard.vlm_admission(
                    _rate(perception_health, "detector_fps"), resource_snapshot
                )
                waited_seconds = time.monotonic() - wait_started
                if admitted:
                    break
                if waited_seconds >= wait_limit:
                    response = "Mô-đun mô tả ảnh vẫn đang chờ HTP hạ nhiệt; nhận biết vật cản tiếp tục hoạt động."
                    self._speak_async(response, "vlm_resource_guard")
                    return {
                        "intent": intent_name,
                        "answer_vi": response,
                        "source": "fallback",
                        "queued_tts": True,
                        "resource_guard": reasons,
                        "thermal_wait_ms": round(waited_seconds * 1000.0, 1),
                        "scene_facts": scene_facts(scene),
                    }
                self._stop.wait(min(poll_seconds, max(0.0, wait_limit - waited_seconds)))

            if not scene.camera_ok or scene.frame_age_ms > float(self.config["vlm"]["max_frame_age_ms"]):
                response = "Khung hình hiện không đủ mới để mô tả an toàn."
                return {
                    "intent": intent_name, "answer_vi": response, "source": "fallback", "queued_tts": False,
                    "thermal_wait_ms": round((time.monotonic() - wait_started) * 1000.0, 1),
                }
            jpeg = self.perception.latest_raw_jpeg()
            accepted, score = accepts_for_vlm(jpeg)
            if not accepted:
                response = "Ảnh hiện mờ hoặc phơi sáng kém, mình chưa thể mô tả đáng tin cậy."
                self._speak_async(response, "vlm_bad_frame")
                return {
                    "intent": intent_name, "answer_vi": response, "source": "fallback", "queued_tts": True,
                    "keyframe": {"sharpness": round(score.sharpness, 2), "score": round(score.score, 3)},
                }
            jpeg, frame_input = prepare_for_vlm(
                jpeg,
                max_width=int(self.config["vlm"]["max_image_width"]),
                jpeg_quality=int(self.config["vlm"]["jpeg_quality"]),
            )
            with self._lock:
                self._stats["vlm_requests"] += 1
                self._set_mode_locked(SystemMode.THINKING)
            result = self.vlm.ask(question_vi, scene, jpeg)
            response = result.answer_vi
            if result.uncertain:
                response = "Mình không chắc. " + response
            self._speak_async(response, "vlm_answer")
            return {
                "intent": intent_name, "answer_vi": response, "source": "vlm", "queued_tts": True,
                "confidence": result.confidence, "uncertain": result.uncertain,
                "evidence": result.evidence, "latency_ms": round(result.latency_ms, 1),
                "thermal_wait_ms": round((time.monotonic() - wait_started) * 1000.0 - result.latency_ms, 1),
                "vlm_input": frame_input,
                "scene_facts": scene_facts(scene),
            }
        except Exception as error:
            with self._lock:
                self._stats["vlm_failures"] += 1
            response = "Mô-đun mô tả ảnh hiện không phản hồi. Bạn có thể hỏi về vật thể đang phát hiện."
            self._speak_async(response, "vlm_error")
            return {"intent": intent_name, "answer_vi": response, "source": "fallback", "queued_tts": True, "error": str(error)}
        finally:
            self._vlm_lock.release()
            self._restore_mode()

    def _speak_async(self, text_vi: str, source: str) -> None:
        with self._lock:
            self._stats["tts_requests"] += 1
        threading.Thread(target=self._render_and_queue, args=(text_vi, source), name="aibox-tts", daemon=True).start()

    def _render_and_queue(self, text_vi: str, source: str) -> None:
        try:
            for path in self.tts.render_chunks(text_vi, self.outputs_dir / "tts"):
                self.audio.submit_file(path, AlertPriority.INFORMATION, source)
        except Exception:
            with self._lock:
                self._stats["tts_failures"] += 1
        finally:
            self._restore_mode()

    def _warm_tts(self) -> None:
        try:
            self.tts.warm()
        except Exception:
            # The health endpoint contains the concrete load error; safety and
            # scene processing intentionally remain alive.
            pass
        finally:
            self._tts_warming = False

    def start_push_to_talk(self) -> dict[str, Any]:
        if self.config["audio"]["half_duplex"] and self.audio.speaking:
            raise RuntimeError("Audio playback active; half-duplex capture is temporarily unavailable")
        path = self.recorder.start()
        self._set_mode(SystemMode.LISTENING)
        return {"recording": True, "path": str(path)}

    def stop_push_to_talk(self) -> dict[str, Any]:
        path = self.recorder.stop()
        self._set_mode(SystemMode.TRANSCRIBING)
        try:
            with self._lock:
                self._stats["stt_requests"] += 1
            text = self.stt.transcribe(path)
            if not text:
                raise RuntimeError("No speech was recognized")
            response = self.ask(text)
            return {"recording": False, "transcript_vi": text, "response": response}
        except Exception:
            with self._lock:
                self._stats["stt_failures"] += 1
            raise
        finally:
            self._restore_mode()

    def reset_demo(self) -> dict[str, Any]:
        self.recorder.abort()
        with self._lock:
            self.scene_builder = SceneBuilder(self.config)
            self.risk = RiskEngine(self.config["risk"])
            self._scene = None
            self._events.clear()
            self._set_mode_locked(SystemMode.SELF_TEST)
        return {"ok": True, "detail": "tracking, risk cooldowns, and event history reset"}

    def _restore_mode(self) -> None:
        with self._lock:
            if self._scene is not None:
                self._set_mode_locked(self._mode_for(self._scene, []))

    def _set_mode(self, mode: SystemMode) -> None:
        with self._lock:
            self._set_mode_locked(mode)

    def _set_mode_locked(self, mode: SystemMode) -> None:
        self._mode = mode


def _resolve_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _rate(health: dict[str, Any], key: str) -> float | None:
    try:
        value = float(health.get("rates", {}).get(key))
        return value if value >= 0 else None
    except (AttributeError, TypeError, ValueError):
        return None
