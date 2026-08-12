"""End-to-end browser acceptance for the dedicated /vi assistant profile."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import requests
import urllib3
from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("OPENGLASS_NATIVE_BASE_URL", "https://127.0.0.1:8006")
URL = f"{BASE_URL.rstrip('/')}/vi"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "vi_profile_e2e"
QUESTIONS = [
    "Bạn đang nhìn thấy gì?",
    "Màu nổi bật nhất trong ảnh là gì?",
    "Hãy tóm tắt cảnh trước mặt bằng một câu.",
]


def _make_microphone_fixture() -> tuple[Path, float]:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    utterances: list[bytes] = []
    sample_rate = 16_000
    total_frames = 0
    for question in QUESTIONS:
        response = requests.post(
            f"{BASE_URL.rstrip('/')}/api/tts/vi",
            json={"text": question},
            verify=False,
            timeout=30,
        )
        response.raise_for_status()
        wav_bytes = base64.b64decode(response.json()["audio_wav_base64"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav_bytes)
            wav_path = Path(handle.name)
        try:
            with wave.open(str(wav_path), "rb") as wav_in:
                assert wav_in.getnchannels() == 1
                assert wav_in.getsampwidth() == 2
                assert wav_in.getframerate() == sample_rate
                utterances.append(wav_in.readframes(wav_in.getnframes()))
                total_frames += wav_in.getnframes()
        finally:
            wav_path.unlink(missing_ok=True)

    output_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_handle.close()
    output = Path(output_handle.name)
    silence = b"\x00\x00" * sample_rate
    lead, between, trailing = 1, 8, 8
    with wave.open(str(output), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(silence * lead)
        for utterance in utterances:
            wav_out.writeframes(utterance)
            wav_out.writeframes(silence * between)
        wav_out.writeframes(silence * trailing)
    duration = lead + trailing + len(utterances) * between + total_frames / sample_rate
    return output, duration


def _json(payload: str | bytes) -> dict[str, Any] | None:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    microphone, fixture_duration = _make_microphone_fixture()
    asr_finals: list[dict[str, Any]] = []
    chat_inputs: list[dict[str, Any]] = []
    tts_statuses: list[int] = []
    console_errors: list[dict[str, Any]] = []
    page_errors: list[str] = []
    request_failures: list[str] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(EDGE),
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-audio-capture={microphone.as_posix()}",
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                ],
            )
            context = browser.new_context(
                ignore_https_errors=True,
                permissions=["camera", "microphone"],
                viewport={"width": 1280, "height": 900},
            )
            context.add_init_script(
                """
                (() => {
                  const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                  navigator.mediaDevices.getUserMedia = async constraints => {
                    const stream = await original(constraints);
                    if (!constraints?.video || !HTMLCanvasElement.prototype.captureStream) return stream;
                    const canvas = document.createElement('canvas');
                    canvas.width = 640; canvas.height = 480;
                    const ctx = canvas.getContext('2d'); let frame = 0;
                    const draw = () => {
                      frame += 1;
                      ctx.fillStyle = `hsl(${frame % 360},75%,42%)`;
                      ctx.fillRect(0,0,640,480);
                      ctx.fillStyle = '#fff'; ctx.font = 'bold 40px sans-serif';
                      ctx.fillText(`Khung hình ${frame}`, 32, 68);
                    };
                    draw(); const timer = setInterval(draw, 100);
                    const track = canvas.captureStream(10).getVideoTracks()[0];
                    track.addEventListener('ended',()=>clearInterval(timer),{once:true});
                    for (const old of stream.getVideoTracks()) {stream.removeTrack(old); old.stop();}
                    stream.addTrack(track); return stream;
                  };
                })();
                """
            )
            page = context.new_page()

            def on_websocket(websocket: Any) -> None:
                def on_sent(payload: str | bytes) -> None:
                    message = _json(payload)
                    if not message or message.get("type") != "input.append":
                        return
                    input_payload = message.get("input", {})
                    messages = input_payload.get("messages", [])
                    if not messages:
                        return
                    content = messages[-1].get("content", [])
                    image = next(
                        (item.get("data") for item in content if item.get("type") == "image"),
                        None,
                    )
                    text = next(
                        (item.get("text") for item in content if item.get("type") == "text"),
                        None,
                    )
                    chat_inputs.append(
                        {
                            "input_id": input_payload.get("input_id"),
                            "text": text,
                            "image_sha256": hashlib.sha256(base64.b64decode(image)).hexdigest()
                            if image
                            else None,
                            "history_message_count": len(messages),
                        }
                    )

                def on_received(payload: str | bytes) -> None:
                    message = _json(payload)
                    if message and message.get("type") == "asr.final":
                        asr_finals.append(
                            {
                                "final_id": message.get("final_id"),
                                "text": message.get("text"),
                                "immutable": message.get("immutable"),
                            }
                        )

                websocket.on("framesent", on_sent)
                websocket.on("framereceived", on_received)

            page.on("websocket", on_websocket)
            page.on(
                "response",
                lambda response: tts_statuses.append(response.status)
                if "/api/tts/vi/stream" in response.url
                else None,
            )
            page.on(
                "console",
                lambda message: console_errors.append(
                    {"text": message.text, "location": message.location}
                )
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("requestfailed", lambda request: request_failures.append(request.url))

            started = time.perf_counter()
            page.goto(URL, wait_until="networkidle", timeout=30_000)
            page.locator("#start").click()
            page.wait_for_function(
                "() => document.querySelector('#status').textContent.includes('Đang nghe')",
                timeout=45_000,
            )
            video_before = page.locator("#camera").evaluate(
                "el=>({time:el.currentTime,width:el.videoWidth,height:el.videoHeight})"
            )

            deadline = time.monotonic() + max(90, fixture_duration + 60)
            while time.monotonic() < deadline:
                users = page.locator("#conversation .message.user").count()
                assistants = page.locator("#conversation .message.assistant").count()
                status = page.locator("#status").inner_text()
                if users >= 3 and assistants >= 3 and "Đang nghe" in status:
                    break
                page.wait_for_timeout(500)

            page.wait_for_timeout(1000)
            video_after = page.locator("#camera").evaluate(
                "el=>({time:el.currentTime,width:el.videoWidth,height:el.videoHeight})"
            )
            user_texts = page.locator("#conversation .message.user").all_inner_texts()
            assistant_texts = page.locator("#conversation .message.assistant").all_inner_texts()
            status_before_stop = page.locator("#status").inner_text()
            screenshot = RESULT_DIR / "vietnamese_profile_e2e.png"
            page.screenshot(path=str(screenshot), full_page=True)
            page.locator("#stop").click()
            page.wait_for_function(
                "() => document.querySelector('#status').textContent.includes('Đã dừng')",
                timeout=10_000,
            )

            unique_finals = {item["final_id"] for item in asr_finals if item["final_id"]}
            image_hashes = {item["image_sha256"] for item in chat_inputs if item["image_sha256"]}
            unexpected_console_errors = [
                item
                for item in console_errors
                if not item.get("location", {}).get("url", "").endswith("/favicon.ico")
            ]
            checks = {
                "three_immutable_unique_asr_finals": len(unique_finals) == 3
                and len(asr_finals) == 3
                and all(item["immutable"] for item in asr_finals),
                "three_exactly_once_chat_inputs": len(chat_inputs) == 3
                and len({item["input_id"] for item in chat_inputs}) == 3,
                "fresh_distinct_frames": len(image_hashes) == 3,
                "three_user_and_assistant_turns": len(user_texts) == 3
                and len(assistant_texts) == 3,
                # Early-TTS may split one answer into multiple sentence requests.
                # Require every turn to produce audio and reject any failed segment;
                # exact segment ordering/deduplication is covered by e2e_vi_early_tts.py.
                "streaming_tts_covers_three_turns": len(tts_statuses) >= 3
                and all(status == 200 for status in tts_statuses),
                "returned_to_listening": "Đang nghe" in status_before_stop,
                "animated_camera_live": video_before["width"] == 640
                and video_before["height"] == 480
                and video_after["time"] > video_before["time"],
                "stop_is_final": "Đã dừng" in page.locator("#status").inner_text(),
                "no_page_errors": not page_errors,
                "no_failed_requests": not request_failures,
                "no_console_errors": not unexpected_console_errors,
            }
            report = {
                "url": URL,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "fixture_duration_seconds": round(fixture_duration, 3),
                "checks": checks,
                "passed": all(checks.values()),
                "asr_finals": asr_finals,
                "chat_inputs": chat_inputs,
                "user_texts": user_texts,
                "assistant_texts": assistant_texts,
                "tts_statuses": tts_statuses,
                "video_before": video_before,
                "video_after": video_after,
                "status_before_stop": status_before_stop,
                "console_errors": console_errors,
                "unexpected_console_errors": unexpected_console_errors,
                "page_errors": page_errors,
                "request_failures": request_failures,
                "screenshot": str(screenshot),
            }
            (RESULT_DIR / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            browser.close()
            if not report["passed"]:
                raise SystemExit(1)
    finally:
        microphone.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
