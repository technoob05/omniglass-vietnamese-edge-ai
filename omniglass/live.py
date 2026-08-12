from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import traceback
from collections import Counter, deque

import cv2
import gradio as gr
import numpy as np

from .perception import screen_location
from .schema import Observation

LOGGER = logging.getLogger("omniglass.live")

VI_LABELS = {
    "person": "người",
    "bottle": "chai",
    "cup": "cốc",
    "laptop": "máy tính xách tay",
    "cell phone": "điện thoại",
    "chair": "ghế",
    "book": "sách",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "tv": "màn hình",
}


class LivePerceptionEngine:
    """Single-user, stateful detector/tracker for a glasses-style live demo."""

    def __init__(self, model_name: str, device: str, history_size: int = 800):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.lock = threading.Lock()
        self.history: deque[Observation] = deque(maxlen=history_size)
        self.started_at = time.perf_counter()
        self.frame_index = 0
        self.processed = 0
        self.wall_ms_ema: float | None = None
        self.last_seen: dict[str, float] = {}
        self.current_labels: set[str] = set()

    def reset(self) -> None:
        with self.lock:
            self.history.clear()
            self.started_at = time.perf_counter()
            self.frame_index = 0
            self.processed = 0
            self.wall_ms_ema = None
            self.last_seen.clear()
            self.current_labels.clear()

    def warmup(self) -> None:
        sample = np.zeros((360, 640, 3), dtype=np.uint8)
        with self.lock:
            self.model.predict(
                sample,
                conf=0.25,
                device=self.device,
                verbose=False,
            )
            self.model.predictor = None

    def process(
        self,
        frame_rgb: np.ndarray | None,
        enabled: bool,
        watch_target: str,
        confidence: float,
    ):
        if frame_rgb is None:
            return None, "Chưa nhận được frame từ webcam.", [], ""
        if not enabled:
            return frame_rgb, "Đã dừng AI. Camera vẫn có thể đang preview trong browser.", self.timeline(), ""

        started = time.perf_counter()
        frame_rgb = np.ascontiguousarray(frame_rgb[:, :, :3])
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if not self.lock.acquire(timeout=3.0):
            return (
                frame_rgb,
                "⏳ H100 đang hoàn tất frame trước; frame cũ này đã được bỏ.",
                self.timeline(),
                "",
            )
        try:
            result = self.model.track(
                frame_bgr,
                persist=True,
                tracker="bytetrack.yaml",
                conf=float(confidence),
                device=self.device,
                verbose=False,
            )[0]
            now = time.perf_counter() - self.started_at
            boxes = result.boxes
            visible_labels: list[str] = []
            frame_observations: list[Observation] = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                track_ids = (
                    boxes.id.detach().cpu().numpy().astype(int)
                    if boxes.id is not None
                    else np.full(len(boxes), -1, dtype=int)
                )
                height, width = frame_bgr.shape[:2]
                for box, score, class_id, track_id in zip(xyxy, scores, classes, track_ids):
                    label = str(result.names[int(class_id)]).lower()
                    box_tuple = tuple(float(value) for value in box)
                    visible_labels.append(label)
                    self.last_seen[label] = now
                    observation = Observation(
                            timestamp=now,
                            frame_index=self.frame_index,
                            label=label,
                            confidence=float(score),
                            bbox_xyxy=box_tuple,
                            track_id=None if track_id < 0 else str(track_id),
                            location=screen_location(box_tuple, width, height),
                            keyframe_path=None,
                            attributes={"source": "live-yolo+bytetrack"},
                        )
                    self.history.append(observation)
                    frame_observations.append(observation)

            annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            self.current_labels = set(visible_labels)
            self.frame_index += 1
            self.processed += 1
            wall_ms = (time.perf_counter() - started) * 1000
            self.wall_ms_ema = wall_ms if self.wall_ms_ema is None else 0.2 * wall_ms + 0.8 * self.wall_ms_ema
            delivered_fps = self.processed / max(now, 1e-6)
            watch = watch_target.strip().lower()
            if watch:
                if any(watch == label or watch in label or label in watch for label in visible_labels):
                    watch_text = f"✅ đang thấy **{watch}**"
                elif watch in self.last_seen:
                    age = now - self.last_seen[watch]
                    watch_text = f"⚠️ không thấy **{watch}** trong frame hiện tại; lần cuối {age:.1f}s trước"
                else:
                    watch_text = f"… chưa từng thấy **{watch}**"
            else:
                watch_text = "chưa đặt đối tượng Watch"
            labels_text = ", ".join(sorted(set(visible_labels))) or "không có nhãn vượt threshold"
            if frame_observations:
                grouped = Counter(item.label for item in frame_observations)
                parts = []
                for label, count in grouped.most_common(5):
                    strongest = max(
                        (item for item in frame_observations if item.label == label),
                        key=lambda item: item.confidence,
                    )
                    name = VI_LABELS.get(label, label)
                    parts.append(f"{count} {name} ở {strongest.location}")
                narration = "Tôi đang thấy " + ", ".join(parts) + "."
            else:
                narration = "Hiện tại tôi chưa nhận ra đối tượng nào rõ ràng."
            inference_ms = float(result.speed.get("inference", 0.0))
            status = (
                f"**LIVE H100** · frame AI #{self.processed} · stream {delivered_fps:.1f} FPS · "
                f"model {inference_ms:.1f} ms · callback H100 {wall_ms:.1f} ms\n\n"
                f"Hiện thấy: {labels_text} · {watch_text}"
            )
            if self.processed == 1 or self.processed % 30 == 0:
                LOGGER.info(
                    "live frame processed index=%s shape=%s model_ms=%.2f callback_ms=%.2f",
                    self.processed,
                    frame_rgb.shape,
                    inference_ms,
                    wall_ms,
                )
            return annotated_rgb, status, self.timeline(), narration
        finally:
            self.lock.release()

    def timeline(self) -> list[list[object]]:
        counts = Counter(item.label for item in self.history)
        rows = []
        for label in sorted(counts):
            latest = max(
                (item for item in self.history if item.label == label),
                key=lambda item: (item.timestamp, item.confidence),
            )
            rows.append(
                [label, counts[label], f"{latest.timestamp:.1f}s", latest.location, f"{latest.confidence:.2f}"]
            )
        return rows

    def answer(self, question: str) -> str:
        text = question.strip().lower()
        with self.lock:
            if not self.history:
                return "Live memory chưa có observation nào."
            labels = sorted({item.label for item in self.history})
            if any(token in text for token in ("đang thấy", "trước mặt", "hiện tại", "what do you see")):
                current = sorted(self.current_labels)
                return (
                    "Hiện tại tôi đang thấy: " + ", ".join(current) + "."
                    if current
                    else "Hiện tại tôi chưa nhận ra đối tượng nào rõ ràng."
                )
            target = next((label for label in labels if label in text), None)
            if target:
                latest = max(
                    (item for item in self.history if item.label == target),
                    key=lambda item: (item.timestamp, item.confidence),
                )
                return (
                    f"Lần cuối thấy {target} ở {latest.location}, tại giây {latest.timestamp:.1f} "
                    f"(confidence {latest.confidence:.0%})."
                )
            return "Live memory đã thấy: " + ", ".join(labels[:10]) + "."


def parse_args():
    parser = argparse.ArgumentParser(description="OmniGlass live glasses demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--device", default=os.environ.get("OMNIGLASS_DEVICE", "cuda"))
    parser.add_argument("--model", default=os.environ.get("OMNIGLASS_DETECTOR", "yolo11n.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = LivePerceptionEngine(args.model, args.device)
    LOGGER.info("warming live model on %s", args.device)
    engine.warmup()
    LOGGER.info("live model warmup complete")

    def safe_process(frame_rgb, enabled, watch_target, confidence):
        try:
            return engine.process(frame_rgb, enabled, watch_target, confidence)
        except Exception as exc:
            LOGGER.exception("Live frame processing failed")
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            return (
                frame_rgb,
                f"❌ **Live callback lỗi:** `{detail}`. Xem `/tmp/omniglass_live.err` trên H100.",
                engine.timeline(),
                "Live AI vừa gặp lỗi xử lý frame.",
            )

    rear_camera = gr.WebcamOptions(
        mirror=False,
        constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 960},
                "height": {"ideal": 540},
            },
            "audio": False,
        },
    )
    css = """
    .gradio-container {max-width: 1450px !important; margin:auto !important;}
    #live-status {font-size:1.02rem; min-height:5rem;}
    @media(max-width:700px){.live-row{flex-direction:column !important}.live-row>*{min-width:100%!important}}
    """
    with gr.Blocks(title="OmniGlass Live Glasses", css=css) as demo:
        gr.Markdown(
            "# 👓 OmniGlass Live Glasses\n"
            "Webcam chạy trên máy bạn; inference và tracking chạy liên tục trên H100 qua SSH tunnel. "
            "Đây là visual assistance prototype, không phải thiết bị dẫn đường an toàn.\n\n"
            "**Khởi động:** Access Webcam → Record → Bật Live AI. "
            "Mode hiện tại là **YOLO class watch**; arbitrary text Grounded-SAM-2 đang được tích hợp riêng."
        )
        enabled = gr.State(False)
        with gr.Row(elem_classes=["live-row"]):
            camera = gr.Image(
                sources=["webcam"],
                type="numpy",
                format="jpeg",
                streaming=True,
                webcam_options=rear_camera,
                label="Camera kính / webcam local",
                height=460,
            )
            annotated = gr.Image(
                type="numpy",
                format="jpeg",
                streaming=True,
                label="H100 live perception",
                height=460,
                interactive=False,
            )
        with gr.Row():
            start = gr.Button("▶ Bật Live AI", variant="primary")
            stop = gr.Button("■ Dừng", variant="stop")
            reset = gr.Button("↺ Reset memory")
        with gr.Row():
            watch = gr.Textbox(
                value="person",
                label="YOLO Watch class (person, bottle, laptop…)",
                scale=2,
            )
            confidence = gr.Slider(0.15, 0.7, value=0.25, step=0.05, label="Confidence", scale=2)
            auto_voice = gr.Checkbox(value=True, label="🔊 Tự đọc khi cảnh đổi", scale=1)
        status = gr.Markdown("Nhấn **Bật Live AI**, sau đó cấp quyền webcam.", elem_id="live-status")
        with gr.Row():
            narration = gr.Textbox(
                value="",
                label="Kính đang nói",
                interactive=False,
                scale=4,
            )
            speak_now = gr.Button("🔊 Đọc cảnh ngay", scale=1)
        timeline = gr.Dataframe(
            headers=["Object", "Observations", "Last seen", "Location", "Confidence"],
            interactive=False,
            label="Continual live memory",
        )
        with gr.Row():
            question = gr.Textbox(value="Đã thấy những gì?", label="Hỏi live memory", scale=4)
            ask = gr.Button("Hỏi", scale=1)
        answer = gr.Markdown("Chưa có live memory.")

        def start_live():
            engine.reset()
            return True, "🟢 Live AI đã bật. Đưa camera quanh cảnh thật chậm."

        def stop_live():
            return False, "⏹ Live AI đã dừng; memory vẫn được giữ để hỏi."

        def reset_live():
            engine.reset()
            return [], "Đã xóa live memory và reset tracker."

        start.click(start_live, outputs=[enabled, status], queue=False)
        stop.click(stop_live, outputs=[enabled, status], queue=False)
        reset.click(reset_live, outputs=[timeline, status], queue=False)
        camera.stream(
            safe_process,
            inputs=[camera, enabled, watch, confidence],
            outputs=[annotated, status, timeline, narration],
            stream_every=0.25,
            time_limit=300,
            trigger_mode="always_last",
            concurrency_limit=3,
            concurrency_id="live_h100",
            show_progress="hidden",
        )
        ask.click(engine.answer, inputs=question, outputs=answer, queue=False)
        question.submit(engine.answer, inputs=question, outputs=answer, queue=False)
        speak_js = """(text, enabled) => {
          if (!enabled || !text || !('speechSynthesis' in window)) return [];
          const now = Date.now();
          if (window.__omniglassLastSpeech === text && now - (window.__omniglassLastSpeechAt || 0) < 4000) return [];
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'vi-VN'; utterance.rate = 1.05;
          window.speechSynthesis.speak(utterance);
          window.__omniglassLastSpeech = text; window.__omniglassLastSpeechAt = now;
          return [];
        }"""
        narration.change(
            fn=None,
            inputs=[narration, auto_voice],
            js=speak_js,
            queue=False,
            show_api=False,
        )
        speak_now.click(
            fn=None,
            inputs=[narration],
            js="""(text) => {
              if (text && 'speechSynthesis' in window) {
                window.speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(text);
                u.lang='vi-VN'; window.speechSynthesis.speak(u);
              } return [];
            }""",
            queue=False,
            show_api=False,
        )

    demo.queue(default_concurrency_limit=2, max_size=32).launch(
        server_name=args.host,
        server_port=args.port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
