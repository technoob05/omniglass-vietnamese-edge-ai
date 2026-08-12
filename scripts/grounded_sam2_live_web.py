#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import sys
import threading
import time
import traceback
import wave
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import requests

LOGGER = logging.getLogger("omniglass.grounded_sam2_live")


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded-SAM-2 text-prompt live webcam wrapper")
    parser.add_argument("--repo", default="upstream/Grounded-SAM-2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--prompt", default="person.")
    parser.add_argument("--detection-interval", type=int, default=20)
    parser.add_argument("--agent-url", default="http://127.0.0.1:8780")
    return parser.parse_args()


class GroundedLiveEngine:
    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        prompt: str,
        detection_interval: int,
        agent_url: str,
    ):
        sys.path.insert(0, str(repo))
        from grounded_sam2_tracking_camera_with_continuous_id import IncrementalObjectTracker

        self.lock = threading.Lock()
        self.prompt = self.normalize_prompt(prompt)
        started = time.perf_counter()
        self.tracker = IncrementalObjectTracker(
            grounding_model_id="IDEA-Research/grounding-dino-tiny",
            sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
            sam2_ckpt_path=str(checkpoint),
            device="cuda",
            prompt_text=self.prompt,
            detection_interval=detection_interval,
        )
        self.load_seconds = time.perf_counter() - started
        self.processed = 0
        self.started_at = time.perf_counter()
        self.agent_url = agent_url.rstrip("/")
        self.latest_frame: np.ndarray | None = None
        self.latest_objects: list[dict] = []
        self.last_seen: dict[str, dict] = {}
        self.tracking_enabled = True
        self.last_answer = ""
        self.source_shape: tuple[int, int] | None = None
        LOGGER.info("Grounded-SAM-2 loaded in %.2fs prompt=%s", self.load_seconds, self.prompt)

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        value = prompt.strip()
        if not value:
            raise ValueError("Text target không được để trống")
        return value if value.endswith(".") else value + "."

    def set_prompt(self, prompt: str):
        normalized = self.normalize_prompt(prompt)
        if not self.lock.acquire(timeout=5):
            return f"⏳ Tracker đang xử lý frame; thử Apply lại sau một chút.", self.prompt
        try:
            self.tracker.set_prompt(normalized)
            # Upstream resets its video state but leaves old masks/IDs alive. Those masks can
            # have a different resolution after a camera rotation or source switch.
            for name in ("last_mask_dict", "track_dict"):
                state = getattr(self.tracker, name, None)
                if state is not None:
                    setattr(self.tracker, name, type(state)())
            self.tracker.objects_count = 0
            self.prompt = normalized
            self.processed = 0
            self.started_at = time.perf_counter()
            self.tracking_enabled = True
            return f"🎯 Đã đổi target thành **{normalized}**. Grounding lại ở frame kế tiếp.", normalized
        finally:
            self.lock.release()

    @staticmethod
    def normalize_video_frame(
        frame: np.ndarray,
        target_width: int = 1280,
        target_height: int = 720,
    ) -> np.ndarray:
        """Letterbox every camera frame to one stable tracker resolution."""
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Webcam frame không hợp lệ: shape={frame.shape}")
        rgb = np.ascontiguousarray(frame[:, :, :3])
        height, width = rgb.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError(f"Webcam frame rỗng: shape={frame.shape}")
        scale = min(target_width / width, target_height / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas

    def process(self, frame: np.ndarray | None):
        if frame is None:
            return None, "Chưa nhận được webcam frame.", ""
        raw_shape = tuple(int(value) for value in frame.shape[:2])
        if raw_shape != self.source_shape:
            LOGGER.info("camera source shape changed old=%s new=%s", self.source_shape, raw_shape)
            self.source_shape = raw_shape
        frame = self.normalize_video_frame(frame)
        if not self.lock.acquire(timeout=3):
            return frame, "⏳ Grounded-SAM-2 đang bận; đã bỏ frame cũ.", ""
        started = time.perf_counter()
        try:
            self.latest_frame = frame.copy()
            output = self.tracker.add_image(frame) if self.tracking_enabled else frame
            elapsed = time.perf_counter() - started
            self.processed += 1
            stream_fps = self.processed / max(time.perf_counter() - self.started_at, 1e-6)
            labels = []
            objects = []
            for info in self.tracker.last_mask_dict.labels.values():
                name = str(getattr(info, "class_name", self.prompt.rstrip(".")))
                if name:
                    labels.append(name)
                    box = [
                        float(getattr(info, "x1", 0)),
                        float(getattr(info, "y1", 0)),
                        float(getattr(info, "x2", frame.shape[1])),
                        float(getattr(info, "y2", frame.shape[0])),
                    ]
                    item = {"label": name, "bbox_xyxy": box, "seen_at": time.time()}
                    objects.append(item)
                    self.last_seen[name.casefold()] = item
            self.latest_objects = objects
            if not self.tracking_enabled:
                narration = "Đã dừng theo dõi. Bạn vẫn có thể yêu cầu mô tả cảnh hiện tại."
                status = f"**AGENT LIVE H100** · frame #{self.processed} · tracking đang dừng"
                return frame, status, narration
            if output is None:
                narration = f"Tôi chưa tìm thấy {self.prompt.rstrip('.')} trong frame hiện tại."
                output = frame
            else:
                unique = sorted(set(labels)) or [self.prompt.rstrip(".")]
                narration = "Tôi đang theo dõi " + ", ".join(unique) + "."
            status = (
                f"**GROUNDED-SAM-2 LIVE H100** · target **{self.prompt}** · frame #{self.processed} · "
                f"{elapsed * 1000:.0f} ms callback · {stream_fps:.2f} stream FPS\n\n{narration}"
            )
            if self.processed == 1 or self.processed % 20 == 0:
                LOGGER.info(
                    "frame=%s prompt=%s elapsed_ms=%.1f objects=%s",
                    self.processed,
                    self.prompt,
                    elapsed * 1000,
                    len(labels),
                )
            return output, status, narration
        except Exception as exc:
            LOGGER.exception("Grounded-SAM-2 live frame failed")
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return frame, f"❌ **Callback lỗi:** `{detail}`", "Grounded SAM vừa gặp lỗi xử lý frame."
        finally:
            self.lock.release()

    @staticmethod
    def _jpeg_base64(frame: np.ndarray) -> str:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("Không mã hóa được webcam frame")
        return base64.b64encode(encoded).decode("ascii")

    def _snapshot(self):
        if not self.lock.acquire(timeout=3):
            raise RuntimeError("Tracker đang bận, vui lòng thử lại")
        try:
            if self.latest_frame is None:
                raise RuntimeError("Chưa có webcam frame. Hãy bật camera trước.")
            return self.latest_frame.copy(), list(self.latest_objects), self.prompt.rstrip(".")
        finally:
            self.lock.release()

    @staticmethod
    def _extract_target(command: str) -> str:
        match = re.search(
            r"(?:theo dõi|theo doi|track|watch|canh|để ý(?: giúp tôi)?|de y(?: giup toi)?|tìm|tim)\s+"
            r"(?:vật|vat|đối tượng|doi tuong|cái|cai)?\s*(.+)",
            command,
            flags=re.IGNORECASE,
        )
        return (match.group(1) if match else "").strip(" .?!,;:")

    def _post_agent(self, endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
        response = requests.post(f"{self.agent_url}/{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _speak_audio(self, text: str) -> tuple[int, np.ndarray]:
        result = self._post_agent("speak", {"text": text}, timeout=45)
        raw = base64.b64decode(result["audio_wav_base64"], validate=True)
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("TTS WAV phải là mono PCM16")
            sample_rate = wav_file.getframerate()
            audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16).copy()
        if audio.size == 0:
            raise ValueError("TTS trả về audio rỗng")
        return sample_rate, audio

    @staticmethod
    def _select_object(objects: list[dict], target: str) -> dict | None:
        wanted = target.casefold().strip(" .")
        for item in objects:
            label = str(item.get("label", "")).casefold().strip(" .")
            if wanted == label or wanted in label or label in wanted:
                return item
        return None

    def handle_command(self, command: str):
        text = (command or "").strip()
        if not text:
            return "Hãy nói hoặc nhập một lệnh.", self.prompt
        folded = text.casefold()
        try:
            if any(token in folded for token in ("dừng theo dõi", "dung theo doi", "stop tracking")):
                if not self.lock.acquire(timeout=3):
                    answer = "Tracker đang bận; chưa dừng được. Hãy thử lại."
                else:
                    try:
                        self.tracking_enabled = False
                        answer = "Đã dừng theo dõi."
                    finally:
                        self.lock.release()
            elif any(
                token in folded
                for token in (
                    "theo dõi", "theo doi", "track ", "watch ", "canh ",
                    "để ý", "de y", "tìm ", "tim ",
                )
            ):
                target = self._extract_target(text)
                if not target:
                    answer = "Bạn muốn tôi theo dõi vật gì? Ví dụ: theo dõi chai đỏ."
                else:
                    prompt_status, normalized = self.set_prompt(target)
                    if normalized != self.normalize_prompt(target):
                        answer = prompt_status.replace("**", "")
                    else:
                        answer = f"Đang theo dõi {normalized.rstrip('.')} bằng GroundingDINO và SAM2."
            elif any(token in folded for token in ("bao xa", "khoảng cách", "khoang cach", "distance")):
                frame, objects, current_target = self._snapshot()
                selected = self._select_object(objects, current_target)
                if selected is None:
                    answer = (
                        f"Tôi chưa thấy rõ {current_target} trong frame hiện tại nên chưa đo khoảng cách. "
                        "Hãy hướng camera vào vật và thử lại."
                    )
                    self.last_answer = answer
                    return answer, self.prompt
                result = self._post_agent(
                    "distance",
                    {
                        "image_jpeg_base64": self._jpeg_base64(frame),
                        "bbox_xyxy": selected["bbox_xyxy"],
                        "target_name": current_target,
                    },
                    timeout=45,
                )
                answer = result["answer"]
            elif any(token in folded for token in ("lần cuối", "lan cuoi", "last seen", "ở đâu", "o dau")):
                frame, objects, current_target = self._snapshot()
                selected = self._select_object(objects, current_target)
                if selected:
                    box = selected["bbox_xyxy"]
                    center = (box[0] + box[2]) / 2
                    width = max(float(frame.shape[1]), 1.0)
                    location = "bên trái" if center < width / 3 else "bên phải" if center > width * 2 / 3 else "ở giữa"
                    answer = f"Tôi vừa thấy {selected['label']} {location} khung hình."
                elif current_target.casefold() in self.last_seen:
                    answer = f"Tôi đã từng thấy {current_target}, nhưng hiện không còn thấy trong khung hình."
                else:
                    answer = f"Tôi chưa có quan sát nào về {current_target}."
            elif any(token in folded for token in ("mô tả", "mo ta", "thấy gì", "thay gi", "describe", "đọc chữ", "doc chu")):
                frame, _, _ = self._snapshot()
                prompt = (
                    "Mô tả ngắn gọn cảnh hiện tại cho người khiếm thị: vật/người đáng chú ý, "
                    "vị trí trái-giữa-phải và chữ dễ đọc nếu có. Không tự đoán khoảng cách mét."
                )
                result = self._post_agent(
                    "analyze",
                    {"image_jpeg_base64": self._jpeg_base64(frame), "prompt": prompt},
                )
                answer = result["answer"]
            elif any(token in folded for token in ("giúp", "help", "lệnh gì", "lenh gi")):
                answer = (
                    "Bạn có thể nói: theo dõi chai đỏ; mô tả trước mặt; vật đó cách bao xa; "
                    "lần cuối thấy nó ở đâu; hoặc dừng theo dõi."
                )
            else:
                plan = self._post_agent("plan", {"command": text}, timeout=30)
                action = plan.get("action", "chat")
                target = (plan.get("target") or "").strip()
                if action == "track" and target:
                    return self.handle_command(f"track {target}")
                if action == "describe":
                    return self.handle_command("describe scene")
                if action == "distance":
                    return self.handle_command("distance")
                if action == "memory":
                    return self.handle_command("last seen")
                if action == "stop":
                    return self.handle_command("stop tracking")
                if action == "help":
                    return self.handle_command("help")
                frame, _, _ = self._snapshot()
                result = self._post_agent(
                    "analyze",
                    {
                        "image_jpeg_base64": self._jpeg_base64(frame),
                        "prompt": plan.get("question") or text,
                    },
                )
                answer = result["answer"]
            self.last_answer = answer
            LOGGER.info("agent command=%r answer_chars=%s", text, len(answer))
            return answer, self.prompt
        except Exception as exc:
            LOGGER.exception("Agent command failed command=%r", text)
            return f"Không xử lý được lệnh: {exc}", self.prompt

    def handle_command_with_audio(self, command: str):
        answer, prompt = self.handle_command(command)
        try:
            audio = self._speak_audio(answer)
        except Exception:
            LOGGER.exception("Vietnamese neural TTS failed")
            audio = None
        return answer, prompt, audio


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    repo = Path(args.repo).resolve()
    checkpoint = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else repo / "checkpoints" / "sam2.1_hiera_large.pt"
    )
    if not (repo / "grounded_sam2_tracking_camera_with_continuous_id.py").is_file():
        raise FileNotFoundError(f"Grounded-SAM-2 repo không hợp lệ: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu SAM2.1 checkpoint: {checkpoint}")
    engine = GroundedLiveEngine(repo, checkpoint, args.prompt, args.detection_interval, args.agent_url)

    css = (
        ".gradio-container{max-width:1450px!important;margin:auto!important} #status{min-height:6rem} "
        "#agent-answer{font-size:1.1rem} .safety{border-left:4px solid #f59e0b;padding:.65rem 1rem}"
    )
    with gr.Blocks(title="WatchAnything — Grounded-SAM-2 Live", css=css) as demo:
        gr.Markdown(
            "# 👓 WatchAnything — Grounded-SAM-2 Live\n"
            "Gõ target tự do → GroundingDINO tìm box → SAM2.1 tạo mask và track qua các frame. "
            "Webcam ở máy local; model chạy trên H100."
        )
        with gr.Row():
            prompt = gr.Textbox(value=args.prompt, label="Text target", scale=4)
            apply_prompt = gr.Button("🎯 Apply target", variant="primary", scale=1)
        prompt_status = gr.Markdown(
            f"Model load **{engine.load_seconds:.1f}s**. Nhập target → Apply target → Access Webcam → Record."
        )
        with gr.Row():
            camera = gr.Image(
                sources=["webcam"], type="numpy", format="jpeg", streaming=True,
                label="Webcam local", height=460,
            )
            output = gr.Image(
                type="numpy", format="jpeg", label="Grounded mask tracking", height=460,
                interactive=False,
            )
        status = gr.Markdown("Chưa nhận frame.", elem_id="status")
        narration = gr.Textbox(label="Kính đang nói", interactive=False)
        auto_voice = gr.Checkbox(
            value=False,
            label="🔊 Giọng hệ thống cho trạng thái live (chỉ bật nếu Windows có voice Việt)",
        )
        gr.Markdown("## 🧠 Nói chuyện với OmniGlass Agent")
        gr.Markdown(
            "Ví dụ: **Theo dõi chai đỏ** · **Mô tả trước mặt tôi** · "
            "**Vật đó cách bao xa?** · **Lần cuối thấy nó ở đâu?**"
        )
        with gr.Row():
            command = gr.Textbox(
                label="Lệnh / câu hỏi",
                placeholder="Hands-free sẽ tự điền lời bạn nói vào đây…",
                scale=5,
                elem_id="agent-command",
            )
            ask = gr.Button("Gửi", variant="primary", scale=1, elem_id="agent-send")
        with gr.Row():
            handsfree = gr.Button("🎧 Bật hội thoại hands-free", variant="primary")
            stop_handsfree = gr.Button("■ Dừng nghe", variant="stop")
        gr.HTML(
            "<div id='handsfree-state' role='status' aria-live='polite'>"
            "Chưa bật micro. Sau một lần cấp quyền, kính sẽ tự nghe từng câu và tự gửi lệnh."
            "</div>"
        )
        agent_answer = gr.Markdown(
            "Agent sẵn sàng. Hãy bật webcam trước khi hỏi về cảnh.",
            elem_id="agent-answer",
        )
        neural_audio = gr.Audio(
            label="Giọng Việt neural · H100",
            type="numpy",
            autoplay=True,
            interactive=False,
            elem_id="neural-tts",
        )
        gr.Markdown(
            "<div class='safety'><b>Lưu ý an toàn:</b> khoảng cách từ camera đơn hiện là ước lượng "
            "chưa hiệu chỉnh. Prototype không thay thế gậy, chó dẫn đường hay thiết bị hỗ trợ di chuyển.</div>"
        )

        apply_prompt.click(
            engine.set_prompt,
            inputs=prompt,
            outputs=[prompt_status, prompt],
            queue=False,
        )
        prompt.submit(
            engine.set_prompt,
            inputs=prompt,
            outputs=[prompt_status, prompt],
            queue=False,
        )
        ask_event = ask.click(
            engine.handle_command_with_audio,
            inputs=command,
            outputs=[agent_answer, prompt, neural_audio],
            concurrency_limit=2,
            concurrency_id="agent_commands",
        )
        command.submit(
            engine.handle_command_with_audio,
            inputs=command,
            outputs=[agent_answer, prompt, neural_audio],
            concurrency_limit=2,
            concurrency_id="agent_commands",
        )
        handsfree.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
              const status = (text) => {
                const node = document.getElementById('handsfree-state');
                if (node) node.textContent = text;
              };
              if (!SpeechRecognition) {
                status('Trình duyệt không hỗ trợ Web Speech. Hãy dùng Chrome hoặc nhập lệnh.');
                return [];
              }
              if (window.__omniglassHandsFree?.active) {
                status('Hands-free đang bật và đang chờ câu tiếp theo.');
                return [];
              }
              const state = {
                active:true, recognition:null, speaking:false, processing:false,
                restartTimer:null, processTimer:null,
              };
              const fillAndSend = (transcript) => {
                const input = document.querySelector('#agent-command textarea, #agent-command input');
                const button = document.querySelector('#agent-send button, #agent-send');
                if (!input || !button) { status('Không tìm thấy ô lệnh; hãy tải lại trang.'); return; }
                const proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                Object.getOwnPropertyDescriptor(proto, 'value').set.call(input, transcript);
                input.dispatchEvent(new Event('input', {bubbles:true}));
                state.processing=true;
                clearTimeout(state.restartTimer);
                clearTimeout(state.processTimer);
                status(`Đã nghe: “${transcript}” · đang xử lý trên H100…`);
                setTimeout(() => button.click(), 120);
                state.processTimer=setTimeout(()=>{
                  state.processing=false;
                  status('Yêu cầu quá 45 giây. Tôi đang nghe lại; bạn có thể nói lại câu lệnh.');
                  if(state.active) state.restartTimer=setTimeout(start,700);
                },45000);
              };
              const start = () => {
                if (!state.active || state.speaking || state.processing || window.speechSynthesis?.speaking) return;
                clearTimeout(state.restartTimer);
                const recognition = new SpeechRecognition();
                state.recognition = recognition;
                recognition.lang='vi-VN'; recognition.continuous=false; recognition.interimResults=true;
                recognition.onstart=()=>status('🟢 Đang nghe liên tục… hãy nói một câu lệnh.');
                recognition.onresult=(event)=>{
                  let finalText=''; let interim='';
                  for(let i=event.resultIndex;i<event.results.length;i++) {
                    const value=event.results[i][0].transcript.trim();
                    if(event.results[i].isFinal) finalText += value; else interim += value;
                  }
                  if(interim) status(`Đang nghe: ${interim}`);
                  if(!finalText || state.speaking || window.speechSynthesis?.speaking) return;
                  const lower=finalText.toLocaleLowerCase('vi-VN');
                  if(lower.includes('dừng nghe') || lower.includes('tắt micro')) {
                    state.active=false; status('Đã tắt hands-free.'); return;
                  }
                  if(lower.includes('ngừng nói') || lower.includes('im lặng')) {
                    window.speechSynthesis?.cancel();
                    const audio=document.querySelector('#neural-tts audio'); if(audio) audio.pause();
                    state.speaking=false; status('Đã ngừng đọc. Tôi vẫn đang nghe.'); return;
                  }
                  fillAndSend(finalText);
                };
                recognition.onerror=(event)=>{
                  if(event.error==='not-allowed' || event.error==='service-not-allowed') {
                    state.active=false; status('Micro bị từ chối. Cho phép quyền micro trong Chrome rồi bật lại.');
                  } else if(event.error!=='no-speech' && event.error!=='aborted') {
                    status(`Lỗi nhận giọng nói: ${event.error}. Đang thử nối lại…`);
                  }
                };
                recognition.onend=()=>{
                  if(state.active && !state.speaking && !state.processing) {
                    clearTimeout(state.restartTimer);
                    state.restartTimer=setTimeout(start,700);
                  }
                };
                try { recognition.start(); } catch(_) { state.restartTimer=setTimeout(start,700); }
              };
              state.start=start;
              window.__omniglassHandsFree=state;
              start();
              return [];
            }""",
            queue=False,
            show_api=False,
        )
        stop_handsfree.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              const state=window.__omniglassHandsFree;
              if(state) {
                state.active=false; state.processing=false;
                clearTimeout(state.restartTimer); clearTimeout(state.processTimer);
                try{state.recognition?.abort();}catch(_){}
              }
              window.speechSynthesis?.cancel();
              const audio=document.querySelector('#neural-tts audio'); if(audio) audio.pause();
              const node=document.getElementById('handsfree-state'); if(node) node.textContent='Đã tắt hands-free.';
              return [];
            }""",
            queue=False,
            show_api=False,
        )
        camera.stream(
            engine.process,
            inputs=camera,
            outputs=[output, status, narration],
            stream_every=0.3,
            time_limit=180,
            trigger_mode="always_last",
            concurrency_limit=3,
            concurrency_id="grounded_sam2_h100",
            show_progress="hidden",
        )
        narration.change(
            fn=None,
            inputs=[narration, auto_voice],
            js="""(text, enabled) => {
              if (!enabled || !text || !('speechSynthesis' in window)) return [];
              if (window.__omniglassHandsFree?.active) return [];
              const now=Date.now(); if(window.__gs2Speech===text && now-(window.__gs2SpeechAt||0)<5000) return [];
              window.speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance(text);
              const voices=window.speechSynthesis.getVoices();
              const viVoice=voices.find(v=>(v.lang||'').toLowerCase()==='vi-vn') || voices.find(v=>(v.lang||'').toLowerCase().startsWith('vi'));
              if(viVoice) u.voice=viVoice;
              u.lang='vi-VN'; u.rate=1.05; window.speechSynthesis.speak(u);
              window.__gs2Speech=text; window.__gs2SpeechAt=now; return [];
            }""",
            queue=False,
            show_api=False,
        )
        neural_audio.change(
            fn=None,
            inputs=[neural_audio],
            js="""() => {
              const state=window.__omniglassHandsFree;
              if(state) {
                state.processing=false; state.speaking=true;
                clearTimeout(state.restartTimer); clearTimeout(state.processTimer);
                try{state.recognition?.abort();}catch(_){}
              }
              const resume=()=>{
                if(!state) return;
                state.speaking=false;
                if(state.active) {
                  clearTimeout(state.restartTimer);
                  state.restartTimer=setTimeout(()=>state.start?.(),500);
                }
              };
              setTimeout(()=>{
                const audio=document.querySelector('#neural-tts audio');
                if(!audio) { resume(); return; }
                audio.onended=resume; audio.onerror=resume;
                const node=document.getElementById('handsfree-state');
                if(node && state?.active) node.textContent='🔊 Đang phát giọng Việt neural từ H100…';
                const played=audio.play();
                if(played?.catch) played.catch(()=>resume());
              },150);
              return [];
            }""",
            queue=False,
            show_api=False,
        )
        demo.load(lambda: engine.prompt, outputs=prompt, queue=False)

    demo.queue(default_concurrency_limit=3, max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
