#!/usr/bin/env python3
"""Minimal OpenGlass-style live visual conversation demo.

The browser owns the webcam and microphone. The H100 keeps only the latest frame,
runs VLM inference on demand, and returns Vietnamese neural speech.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
import uuid
import wave

import cv2
import gradio as gr
import numpy as np
import requests

LOGGER = logging.getLogger("omniglass.simple_live")


def parse_args():
    parser = argparse.ArgumentParser(description="OpenGlass Simple Vietnamese visual conversation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7873)
    parser.add_argument("--agent-url", default="http://127.0.0.1:8780")
    return parser.parse_args()


class SimpleVisionAgent:
    GREETINGS = {"hello", "hi", "hey", "xin chào", "chào", "alo"}

    def __init__(self, agent_url: str):
        self.agent_url = agent_url.rstrip("/")
        self.frame_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.frame_id = 0
        self.captured_at = 0.0

    def _post(self, endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
        response = requests.post(f"{self.agent_url}/{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _normalize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Frame camera không hợp lệ: {frame.shape}")
        rgb = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
        height, width = rgb.shape[:2]
        if width > max_width:
            scale = max_width / width
            rgb = cv2.resize(rgb, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
        return rgb

    def capture(self, frame: np.ndarray | None):
        if frame is None:
            return "Chưa nhận được camera frame."
        try:
            normalized = self._normalize_frame(frame)
            with self.frame_lock:
                self.latest_frame = normalized.copy()
                self.frame_id += 1
                self.captured_at = time.time()
                frame_id = self.frame_id
            if frame_id == 1 or frame_id % 30 == 0:
                LOGGER.info("captured frame=%s shape=%s", frame_id, normalized.shape)
            return f"🟢 Camera live · frame #{frame_id} · sẵn sàng nhận câu hỏi"
        except Exception as exc:
            LOGGER.exception("Camera capture failed")
            return f"❌ Camera lỗi: {exc}"

    def capture_webcam(self, frame: np.ndarray | None):
        return self.capture(frame), "webcam"

    def capture_upload(self, frame: np.ndarray | None):
        return self.capture(frame), "upload"

    def _snapshot(self) -> tuple[np.ndarray, int, float]:
        with self.frame_lock:
            if self.latest_frame is None:
                raise RuntimeError("Chưa có ảnh camera. Hãy bật webcam trước.")
            return self.latest_frame.copy(), self.frame_id, self.captured_at

    @staticmethod
    def _jpeg_base64(frame: np.ndarray) -> str:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Không mã hóa được ảnh camera")
        return base64.b64encode(encoded).decode("ascii")

    @staticmethod
    def _decode_camera_data_url(value: str) -> np.ndarray:
        payload = value.split(",", 1)[1] if "," in value else value
        raw = base64.b64decode(payload, validate=True)
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Không giải mã được frame vừa chụp từ webcam")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _audio_data_url(audio: tuple[int, np.ndarray] | None) -> str:
        if audio is None:
            return ""
        rate, samples = audio
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            wav_file.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
        return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _concise_answer(text: str, max_words: int = 18) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.strip()
        shortened = " ".join(words[:max_words]).rstrip(" ,;:-")
        return shortened.rstrip(".!?") + "."

    def _speak(self, text: str) -> tuple[int, np.ndarray]:
        result = self._post("speak", {"text": text}, timeout=45)
        raw = base64.b64decode(result["audio_wav_base64"], validate=True)
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("TTS WAV phải là mono PCM16")
            rate = wav_file.getframerate()
            audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).copy()
        if audio.size == 0:
            raise ValueError("TTS trả về audio rỗng")
        return rate, audio

    def ask(
        self,
        question: str,
        active_source: str = "webcam",
        camera_frame: str | np.ndarray | None = None,
        uploaded_frame: np.ndarray | None = None,
    ):
        text = (question or "").strip()
        needs_frame = bool(text) and text.casefold().strip(" .!?,;:") not in self.GREETINGS
        try:
            frame, frame_id, captured_at = self._query_frame(
                active_source,
                camera_frame,
                uploaded_frame,
                required=needs_frame,
            )
        except Exception as exc:
            LOGGER.exception("Frame capture failed")
            return self._finish_answer(f"Tôi chưa chụp được ảnh hiện tại: {exc}", None)
        return self._answer_frame(text, frame, frame_id, captured_at)

    def ask_turn(
        self,
        question: str,
        active_source: str = "webcam",
        camera_frame: str | np.ndarray | None = None,
        uploaded_frame: np.ndarray | None = None,
    ):
        """Yield the captured frame immediately, then the completed answer.

        A separate completion signal makes the browser voice state machine recover
        even when TTS fails or the audio component does not emit a DOM event.
        """
        turn_id = uuid.uuid4().hex
        text = (question or "").strip()
        needs_frame = bool(text) and text.casefold().strip(" .!?,;:") not in self.GREETINGS
        try:
            frame, frame_id, captured_at = self._query_frame(
                active_source,
                camera_frame,
                uploaded_frame,
                required=needs_frame,
            )
        except Exception as exc:
            LOGGER.exception("Frame capture failed turn=%s", turn_id)
            answer, audio, frame = self._finish_answer(f"Tôi chưa chụp được ảnh hiện tại: {exc}", None)
            yield answer, frame, json.dumps({"phase": "done", "turn_id": turn_id, "audio_payload": ""})
            return

        yield "⏳ Đã chụp ảnh hiện tại · H100 đang phân tích…", frame, json.dumps(
            {"phase": "thinking", "turn_id": turn_id}
        )
        answer, audio, frame = self._answer_frame(text, frame, frame_id, captured_at)
        audio_duration_ms = 0
        if audio is not None:
            rate, samples = audio
            audio_duration_ms = round(len(samples) * 1000 / rate)
        yield answer, frame, json.dumps(
            {
                "phase": "done",
                "turn_id": turn_id,
                "audio_duration_ms": audio_duration_ms,
                "audio_payload": self._audio_data_url(audio),
            }
        )

    def _answer_frame(
        self,
        text: str,
        frame: np.ndarray | None,
        frame_id: int,
        captured_at: float,
    ):
        if not text:
            return self._finish_answer("Bạn hãy hỏi một câu về hình ảnh trước camera.", frame)
        if text.casefold().strip(" .!?,;:") in self.GREETINGS:
            return self._finish_answer(
                "Xin chào! Tôi đang nghe và sẵn sàng xem camera.",
                frame,
            )
        if frame is None:
            return self._finish_answer("Tôi chưa nhận được ảnh hiện tại từ camera.", None)

        try:
            age_ms = max(0, round((time.time() - captured_at) * 1000))
            prompt = (
                "Hãy trả lời câu hỏi về đúng ảnh camera hiện tại bằng tiếng Việt tự nhiên, ngắn gọn và cụ thể. "
                "Chỉ trả lời một hoặc hai câu, tối đa 45 từ. "
                "Nếu ảnh không đủ rõ thì nói không chắc chắn; không bịa vật hoặc khoảng cách. "
                f"Câu hỏi: {text}"
            )
            started = time.perf_counter()
            result = self._post(
                "analyze",
                {
                    "image_jpeg_base64": self._jpeg_base64(frame),
                    "prompt": prompt,
                    "max_new_tokens": 48,
                },
                timeout=60,
            )
            answer = self._concise_answer(str(result["answer"]).strip())
            elapsed = time.perf_counter() - started
            LOGGER.info(
                "question complete frame=%s age_ms=%s seconds=%.3f chars=%s",
                frame_id,
                age_ms,
                elapsed,
                len(answer),
            )
            return self._finish_answer(answer, frame)
        except Exception as exc:
            LOGGER.exception("Visual question failed")
            answer = f"Tôi chưa xử lý được câu hỏi: {exc}"
            return self._finish_answer(answer, frame)

    def _finish_answer(self, answer: str, frame: np.ndarray | None):
        try:
            audio = self._speak(answer)
        except Exception:
            LOGGER.exception("Vietnamese TTS failed; returning text answer")
            audio = None
        return answer, audio, frame

    def _query_frame(
        self,
        active_source: str,
        camera_frame: str | np.ndarray | None,
        uploaded_frame: np.ndarray | None,
        required: bool,
    ) -> tuple[np.ndarray | None, int, float]:
        selected = uploaded_frame if active_source == "upload" else camera_frame
        if selected is None:
            selected = camera_frame if camera_frame is not None else uploaded_frame
        if isinstance(selected, str) and not selected.strip():
            selected = None
        if selected is not None:
            if isinstance(selected, str):
                selected = self._decode_camera_data_url(selected)
            frame = self._normalize_frame(selected)
            with self.frame_lock:
                self.latest_frame = frame.copy()
                self.frame_id += 1
                self.captured_at = time.time()
                frame_id = self.frame_id
                captured_at = self.captured_at
            return frame, frame_id, captured_at
        if required:
            raise RuntimeError("Chưa chụp được frame hiện tại từ đúng phiên webcam này.")
        return None, self.frame_id, 0.0


HANDS_FREE_START = r"""() => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const status = (text) => {
    const node = document.getElementById('simple-voice-state');
    if (node) node.textContent = text;
  };
  if (!SpeechRecognition) {
    status('Trình duyệt không hỗ trợ nhận giọng nói. Hãy dùng Chrome hoặc nhập câu hỏi.');
    return [];
  }
  if (window.__openGlassSimple?.active) {
    status('Hội thoại đang bật. Bạn hãy nói một câu hỏi.');
    return [];
  }
  const state = {
    active:true, processing:false, speaking:false, recognition:null,
    restartTimer:null, watchdog:null, speechTimer:null, audioGeneration:0,
    lastTranscript:'', lastTranscriptAt:0,
  };
  const resume = () => {
    if (!state.active || state.processing || state.speaking) return;
    clearTimeout(state.restartTimer);
    state.restartTimer = setTimeout(start, 500);
  };
  const send = (transcript) => {
    const normalized = transcript.trim().toLocaleLowerCase('vi-VN');
    const now = Date.now();
    if (!normalized) return;
    if (state.processing || state.speaking) {
      status('Tôi đang xử lý câu trước. Vui lòng chờ một chút.');
      return;
    }
    if (normalized === state.lastTranscript && now - state.lastTranscriptAt < 1800) return;
    state.lastTranscript = normalized; state.lastTranscriptAt = now;
    const input = document.querySelector('#simple-question textarea, #simple-question input');
    const button = document.querySelector('#simple-send button, #simple-send');
    if (!input || !button) { status('Không tìm thấy ô câu hỏi. Hãy tải lại trang.'); return; }
    const proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(input, transcript);
    input.dispatchEvent(new Event('input', {bubbles:true}));
    state.processing = true;
    clearTimeout(state.restartTimer); clearTimeout(state.watchdog);
    try { state.recognition?.abort(); } catch (_) {}
    status(`Đã nghe: “${transcript}” · đang xem ảnh trên H100…`);
    setTimeout(() => button.click(), 100);
    state.watchdog = setTimeout(() => {
      state.processing = false;
      status('Yêu cầu quá lâu. Tôi đang nghe lại; bạn có thể hỏi lại.');
      resume();
    }, 90000);
  };
  const start = () => {
    if (!state.active || state.processing || state.speaking) return;
    clearTimeout(state.restartTimer);
    const recognition = new SpeechRecognition();
    state.recognition = recognition;
    recognition.lang = 'vi-VN'; recognition.continuous = false; recognition.interimResults = true;
    recognition.onstart = () => status('🟢 Đang nghe… hỏi tôi về hình ảnh trước camera.');
    recognition.onresult = (event) => {
      let finalText = ''; let interim = '';
      for (let i=event.resultIndex; i<event.results.length; i++) {
        const value = event.results[i][0].transcript.trim();
        if (event.results[i].isFinal) finalText += value; else interim += value;
      }
      if (interim) status(`Đang nghe: ${interim}`);
      if (!finalText) return;
      const lower = finalText.toLocaleLowerCase('vi-VN');
      if (lower.includes('dừng nghe') || lower.includes('tắt micro')) {
        state.active = false; status('Đã tắt hội thoại.'); return;
      }
      if (lower.includes('ngừng nói') || lower.includes('im lặng')) {
        if (state.audio) { state.audio.pause(); state.audio = null; }
        clearTimeout(state.speechTimer);
        state.audioGeneration += 1;
        state.speaking = false; state.processing = false;
        status('Đã ngừng nói. Tôi vẫn đang nghe.'); resume(); return;
      }
      send(finalText);
    };
    recognition.onerror = (event) => {
      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        state.active = false;
        status('Micro bị từ chối. Hãy cấp quyền micro trong Chrome rồi thử lại.');
      } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
        status(`Lỗi nhận giọng nói: ${event.error}. Đang thử lại…`);
      }
    };
    recognition.onend = resume;
    try { recognition.start(); } catch (_) { resume(); }
  };
  state.start = start; state.resume = resume;
  window.__openGlassSimple = state;
  start();
  return [];
}"""


HANDS_FREE_STOP = r"""() => {
  const state = window.__openGlassSimple;
  if (state) {
    state.active = false; state.processing = false; state.speaking = false;
    clearTimeout(state.restartTimer); clearTimeout(state.watchdog); clearTimeout(state.speechTimer);
    state.audioGeneration += 1;
    try { state.recognition?.abort(); } catch (_) {}
  }
  if (state?.audio) { state.audio.pause(); state.audio = null; }
  const node = document.getElementById('simple-voice-state');
  if (node) node.textContent = 'Đã tắt hội thoại.';
  return [];
}"""


AUDIO_READY = r"""(eventPayload) => {
  const state = window.__openGlassSimple;
  if (!state) return [];
  let event;
  try { event = JSON.parse(eventPayload || '{}'); } catch (_) { return []; }
  if (event.phase !== 'done') return [];
  const audioPayload = event.audio_payload || '';
  state.processing = false; state.speaking = true;
  clearTimeout(state.restartTimer); clearTimeout(state.watchdog);
  try { state.recognition?.abort(); } catch (_) {}
  state.audioGeneration += 1;
  const generation = state.audioGeneration;
  let finished = false;
  const finish = () => {
    if (finished || generation !== state.audioGeneration) return;
    finished = true;
    clearTimeout(state.speechTimer);
    state.speaking = false;
    const node = document.getElementById('simple-voice-state');
    if (node && state.active) node.textContent = '🟢 Tôi đang nghe câu hỏi tiếp theo…';
    state.resume?.();
  };
  if (!audioPayload) {
    state.speaking = false;
    finish();
    return [];
  }
  if (!audioPayload) { finish(); return []; }
  if (state.audio) state.audio.pause();
  const audio = document.getElementById('simple-owned-audio') || new Audio();
  audio.pause();
  audio.src = audioPayload;
  audio.currentTime = 0;
  state.audio = audio;
  audio.addEventListener('ended', finish, {once:true});
  audio.addEventListener('error', finish, {once:true});
  const declaredMs = Number(event.audio_duration_ms);
  const fallbackMs = Number.isFinite(declaredMs) && declaredMs > 0
    ? Math.min(Math.max(declaredMs + 2000, 5000), 25000)
    : 25000;
  clearTimeout(state.speechTimer);
  state.speechTimer = setTimeout(finish, fallbackMs);
  audio.addEventListener('loadedmetadata', () => {
    if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
    clearTimeout(state.speechTimer);
    state.speechTimer = setTimeout(finish, Math.min(Math.max(audio.duration + 2, 5), 25) * 1000);
  }, {once:true});
  const node = document.getElementById('simple-voice-state');
  if (node && state.active) node.textContent = '🔊 Đang trả lời bằng tiếng Việt…';
  const played = audio.play(); if (played?.catch) played.catch(finish);
  return [];
}"""


CAPTURE_CURRENT_FRAME = r"""(question, source, previousFrame, uploadedImage) => {
  if (source === 'upload' && uploadedImage) return [question, source, previousFrame, uploadedImage];
  const video = document.querySelector('#simple-camera video') || document.querySelector('video');
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return [question, source, '', uploadedImage];
  }
  const maxWidth = 1280;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
  const context = canvas.getContext('2d', {alpha:false});
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return [question, source, canvas.toDataURL('image/jpeg', 0.88), uploadedImage];
}"""


def build_demo(engine: SimpleVisionAgent):
    css = (
        ".gradio-container{max-width:1180px!important;margin:auto!important} "
        "#simple-answer{font-size:1.15rem;min-height:5rem} "
        ".note{border-left:4px solid #22c55e;padding:.7rem 1rem}"
    )
    with gr.Blocks(title="OpenGlass Simple · Vietnamese Live Vision", css=css) as demo:
        active_source = gr.State("webcam")
        gr.Markdown(
            "# 👓 OpenGlass Simple · Hội thoại thị giác tiếng Việt\n"
            "Camera và micro ở máy bạn; VLM và giọng Việt chạy trên H100. "
            "Bản đầu tiên này chỉ **xem ảnh hiện tại và trả lời**, chưa tracking/depth/memory."
        )
        gr.Markdown(
            "<div class='note'>Luồng demo: webcam local → giữ frame mới nhất → bạn hỏi bằng tiếng Việt "
            "→ Qwen-VL xem ảnh → MMS-TTS trả lời tiếng Việt.</div>"
        )
        with gr.Row():
            with gr.Tabs():
                with gr.Tab("📷 Webcam realtime") as webcam_tab:
                    camera = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        format="jpeg",
                        streaming=True,
                        label="Camera kính / webcam local",
                        height=480,
                        elem_id="simple-camera",
                        webcam_options=gr.WebcamOptions(
                            mirror=False,
                            constraints={
                                "video": {
                                    "facingMode": {"ideal": "environment"},
                                    "width": {"ideal": 1280},
                                    "height": {"ideal": 720},
                                },
                                "audio": False,
                            },
                        ),
                    )
                with gr.Tab("⬆️ Upload ảnh thử") as upload_tab:
                    upload_image = gr.Image(
                        sources=["upload"],
                        type="numpy",
                        format="jpeg",
                        label="Ảnh thử",
                        height=480,
                    )
            snapshot = gr.Image(
                label="Ảnh agent vừa phân tích",
                type="numpy",
                format="jpeg",
                height=480,
                interactive=False,
                elem_id="simple-snapshot",
            )
        camera_status = gr.Markdown(
            "Webcam preview chạy tại máy bạn; mỗi câu hỏi sẽ chụp đúng một frame hiện tại. Không cần bấm Record."
        )
        frame_payload = gr.Textbox(value="", visible=False)
        with gr.Row():
            question = gr.Textbox(
                label="Câu hỏi tiếng Việt",
                placeholder="Ví dụ: Trước mặt tôi có gì? Đọc chữ trên hộp giúp tôi.",
                scale=5,
                elem_id="simple-question",
            )
            send = gr.Button("Hỏi", variant="primary", scale=1, elem_id="simple-send")
        with gr.Row():
            start_voice = gr.Button("🎧 Bật hội thoại liên tục", variant="primary")
            stop_voice = gr.Button("■ Dừng nghe", variant="stop")
        gr.HTML(
            "<div id='simple-voice-state' role='status' aria-live='polite'>"
            "Bật camera, sau đó bấm Bật hội thoại một lần và nói tiếng Việt.</div>"
        )
        answer = gr.Markdown("Tôi sẵn sàng xem ảnh khi camera đã bật.", elem_id="simple-answer")
        gr.HTML(
            "<div class='audio-box'><strong>Câu trả lời tiếng Việt · H100</strong>"
            "<audio id='simple-owned-audio' controls preload='metadata'></audio></div>"
        )
        turn_event = gr.Textbox(value="", visible=False)

        webcam_tab.select(
            lambda: ("webcam", "Webcam local đang được chọn; frame chỉ được gửi khi bạn hỏi."),
            outputs=[active_source, camera_status],
            queue=False,
        )
        upload_tab.select(
            lambda: ("upload", "Đã chọn chế độ upload ảnh thử."),
            outputs=[active_source, camera_status],
            queue=False,
        )
        upload_image.upload(
            engine.capture_upload,
            inputs=upload_image,
            outputs=[camera_status, active_source],
            queue=False,
            show_progress="hidden",
        )
        send.click(
            engine.ask_turn,
            inputs=[question, active_source, frame_payload, upload_image],
            outputs=[answer, snapshot, turn_event],
            concurrency_limit=1,
            concurrency_id="simple_vlm",
            js=CAPTURE_CURRENT_FRAME,
        )
        question.submit(
            engine.ask_turn,
            inputs=[question, active_source, frame_payload, upload_image],
            outputs=[answer, snapshot, turn_event],
            concurrency_limit=1,
            concurrency_id="simple_vlm",
            js=CAPTURE_CURRENT_FRAME,
        )
        start_voice.click(fn=None, js=HANDS_FREE_START, queue=False)
        stop_voice.click(fn=None, js=HANDS_FREE_STOP, queue=False)
        turn_event.change(
            fn=None,
            inputs=turn_event,
            js=AUDIO_READY,
            queue=False,
        )

    return demo


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = SimpleVisionAgent(args.agent_url)
    try:
        response = requests.get(f"{engine.agent_url}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        LOGGER.info("agent health=%s", health)
    except Exception:
        # Health is a GET endpoint; startup remains possible and the UI will show request errors.
        LOGGER.info("agent health will be checked on first question")
    build_demo(engine).queue(default_concurrency_limit=2, max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
