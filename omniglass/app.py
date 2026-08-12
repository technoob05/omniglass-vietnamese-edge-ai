from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from pathlib import Path

import gradio as gr

from .memory import MemoryIndex, load_memory, normalize_query_label
from .perception import SharedPerceptionPipeline, normalize_video_value

LOGGER = logging.getLogger("omniglass.app")


def parse_args():
    parser = argparse.ArgumentParser(description="OmniGlass shared-perception web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--device", default=os.environ.get("OMNIGLASS_DEVICE", "cuda"))
    parser.add_argument("--model", default=os.environ.get("OMNIGLASS_DETECTOR", "yolo11n.pt"))
    parser.add_argument("--results-dir", default="runs/web")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SharedPerceptionPipeline(args.model, args.device)

    def process(camera_value, upload_value, sampled_fps, watch_target):
        video_value = upload_value or camera_value
        if not video_value:
            return None, None, None, [], "Chưa có video.", None, gr.update(visible=False)
        try:
            video_path = normalize_video_value(video_value)
            memory, output_video, memory_path, keyframe, timings = pipeline.process_video(
                video_path,
                results_dir,
                sampled_fps=float(sampled_fps),
                max_seconds=20.0,
                watch_target=watch_target,
            )
            counts = Counter(item.label for item in memory.observations)
            rows = []
            for label in sorted(counts):
                latest = max(
                    (item for item in memory.observations if item.label == label),
                    key=lambda item: (item.timestamp, item.confidence),
                )
                rows.append([
                    label,
                    counts[label],
                    f"{latest.timestamp:.1f}s",
                    latest.location,
                    f"{latest.confidence:.2f}",
                ])
            truncated = " Video dài đã được cắt ở 20 giây." if memory.metadata.get("truncated") else ""
            status = (
                f"Hoàn tất {memory.processed_frames} frame dùng chung cho See/Remember/Find/Watch "
                f"trong **{timings['total']:.1f}s**; ghi {len(memory.observations)} observations, "
                f"{len(memory.watch_events)} watch events.{truncated}"
            )
            index = MemoryIndex(memory)
            first_answer, _ = index.answer("Trước mặt tôi có gì?")
            watch_label = normalize_query_label(watch_target, index.labels)
            watch_observation = index.latest(watch_label) if watch_label else None
            evidence_keyframe = watch_observation.keyframe_path if watch_observation else keyframe
            return (
                output_video,
                evidence_keyframe,
                memory_path,
                rows,
                status + "\n\n**See:** " + first_answer,
                memory_path,
                gr.update(visible=True),
            )
        except Exception as exc:
            LOGGER.exception("OmniGlass processing failed")
            return None, None, None, [], f"Lỗi: `{type(exc).__name__}: {exc}`", None, gr.update(visible=False)

    def ask(question, memory_state):
        if not memory_state:
            return "Hãy xử lý một video trước.", None
        try:
            memory = load_memory(memory_state)
            return MemoryIndex(memory).answer(question)
        except Exception as exc:
            LOGGER.exception("Memory query failed")
            return f"Lỗi truy vấn memory: `{type(exc).__name__}: {exc}`", None

    rear_camera = gr.WebcamOptions(
        mirror=False,
        constraints={
            "video": {"facingMode": {"ideal": "environment"}, "width": {"ideal": 1280}},
            "audio": False,
        },
    )
    css = """
    .gradio-container {max-width: 1500px !important; margin: auto !important;}
    #hero {padding: 1rem 0 .3rem;}
    #status {font-size: 1.03rem;}
    .record-clock {display:flex; align-items:center; gap:.55rem; font-weight:650;}
    .record-dot {width:.65rem; height:.65rem; border-radius:50%; background:#ef4444;
      box-shadow:0 0 0 4px #ef444433;}
    @media (max-width: 640px) {
      .result-row, .answer-row {flex-direction:column !important;}
      .result-row > *, .answer-row > * {min-width:100% !important;}
    }
    """
    with gr.Blocks(title="OmniGlass — Shared Perception & Memory", css=css) as demo:
        gr.Markdown(
            "# OmniGlass — See · Remember · Find · Watch\n"
            "Một lượt perception tạo memory dùng chung cho nhiều skill. "
            "Prototype hỗ trợ nhận biết, không phải thiết bị dẫn đường an toàn.",
            elem_id="hero",
        )
        memory_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("📷 Camera sau"):
                        camera = gr.Video(
                            sources=["webcam"],
                            format=None,
                            include_audio=True,
                            webcam_options=rear_camera,
                            label="Quay 2–20 giây",
                            height=330,
                        )
                        gr.HTML(
                            '<div class="record-clock"><span class="record-dot"></span>'
                            '<span id="record-seconds">00:00</span>'
                            '<span id="record-hint">Tối đa 20 giây</span></div>'
                        )
                    with gr.Tab("⬆️ Upload video"):
                        upload = gr.Video(
                            sources=["upload"],
                            format=None,
                            include_audio=True,
                            label="Chọn video 2–20 giây",
                            height=330,
                        )
                with gr.Row():
                    sampled_fps = gr.Slider(1, 4, value=2, step=1, label="Perception FPS")
                    watch_target = gr.Textbox(value="laptop", label="Đối tượng cần canh")
                run = gr.Button("Phân tích và tạo visual memory", variant="primary")
                status = gr.Markdown("Quay 8–15 giây, lia chậm và giữ vật thể trong khung hình.", elem_id="status")
            with gr.Column(scale=1, visible=False) as result_group:
                with gr.Tabs():
                    with gr.Tab("Video perception"):
                        annotated_video = gr.Video(label="Detection + tracking", height=440)
                    with gr.Tab("Last-seen keyframe"):
                        keyframe = gr.Image(type="filepath", label="Memory evidence", height=440)
                memory_file = gr.File(label="Tải memory JSON")

        timeline = gr.Dataframe(
            headers=["Object", "Observations", "Last seen", "Location", "Confidence"],
            datatype=["str", "number", "str", "str", "str"],
            label="Shared memory index",
            interactive=False,
        )
        gr.Markdown("## Hỏi visual memory")
        with gr.Row(elem_classes=["answer-row"]):
            question = gr.Textbox(
                value="Laptop lần cuối ở đâu?",
                label="Câu hỏi",
                placeholder="Trước mặt có gì? / Chai nước đâu? / Có gì thay đổi? / Laptop có bị di chuyển không?",
                scale=4,
            )
            ask_button = gr.Button("Hỏi", variant="secondary", scale=1)
        with gr.Row():
            answer = gr.Markdown("Chưa có memory.")
            evidence = gr.Image(type="filepath", label="Evidence frame", height=320)

        start_timer_js = """() => {
          if (window.__omniglassTimer) clearInterval(window.__omniglassTimer.handle);
          const state = {startedAt: performance.now(), handle: null};
          const draw = () => {
            const seconds = Math.floor((performance.now() - state.startedAt) / 1000);
            const clock = document.getElementById('record-seconds');
            const hint = document.getElementById('record-hint');
            if (clock) clock.textContent = `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(seconds % 60).padStart(2,'0')}`;
            if (hint) hint.textContent = seconds >= 20 ? 'Đã đạt giới hạn 20 giây' : 'Đang quay • lia camera chậm';
          };
          state.handle = setInterval(draw, 200); window.__omniglassTimer = state; draw(); return [];
        }"""
        stop_timer_js = """() => {
          const state = window.__omniglassTimer;
          if (state?.handle) clearInterval(state.handle);
          const hint = document.getElementById('record-hint');
          if (hint && state?.startedAt) hint.textContent = `Đã quay ${Math.floor((performance.now()-state.startedAt)/1000)} giây`;
          window.__omniglassTimer = null; return [];
        }"""
        camera.start_recording(fn=None, js=start_timer_js, queue=False, show_api=False)
        camera.stop_recording(fn=None, js=stop_timer_js, queue=False, show_api=False)

        kickoff = run.click(
            lambda: "⏳ Đang tải và giải mã video…",
            outputs=status,
            queue=False,
        )
        kickoff.then(
            process,
            inputs=[camera, upload, sampled_fps, watch_target],
            outputs=[annotated_video, keyframe, memory_file, timeline, status, memory_state, result_group],
            queue=False,
            concurrency_limit=1,
            concurrency_id="omniglass_gpu",
        )
        ask_button.click(ask, inputs=[question, memory_state], outputs=[answer, evidence], queue=False)
        question.submit(ask, inputs=[question, memory_state], outputs=[answer, evidence], queue=False)

    demo.queue(default_concurrency_limit=2, max_size=16).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
