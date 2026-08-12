"""Browser soak test for the native MiniCPM-o Vietnamese full-duplex path.

This is intentionally an end-to-end browser test: a synthetic microphone plays
five Vietnamese questions, the page streams them through its real WebSocket,
and the test observes the external Vietnamese TTS/audio lifecycle in Edge.
"""

from __future__ import annotations

import base64
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
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


URL = os.environ.get("OPENGLASS_NATIVE_URL", "https://127.0.0.1:8006/omni")
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "native_vi_soak"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
QUESTIONS = [
    "Bạn đang nhìn thấy gì?",
    "Phía trước có vật gì nổi bật?",
    "Trong ảnh có người nào không?",
    "Màu sắc nổi bật nhất là gì?",
    "Hãy mô tả cảnh vật bằng một câu ngắn.",
]


def _tts_wav(text: str) -> bytes:
    response = requests.post(
        URL.removesuffix("/omni") + "/api/tts/vi",
        json={"text": text},
        verify=False,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return base64.b64decode(payload["audio_wav_base64"])


def _build_microphone_fixture() -> tuple[Path, dict[str, Any]]:
    """Build one real-time WAV with five speech/silence cycles."""
    speech: list[tuple[str, bytes, int]] = []
    for question in QUESTIONS:
        wav_bytes = _tts_wav(question)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav_bytes)
            wav_path = Path(handle.name)
        try:
            with wave.open(str(wav_path), "rb") as wav_in:
                if wav_in.getnchannels() != 1 or wav_in.getsampwidth() != 2:
                    raise RuntimeError("TTS microphone fixture must be mono PCM16")
                if wav_in.getframerate() != 16000:
                    raise RuntimeError(
                        f"TTS microphone fixture must be 16 kHz, got {wav_in.getframerate()}"
                    )
                frames = wav_in.readframes(wav_in.getnframes())
                speech.append((question, frames, wav_in.getnframes()))
        finally:
            wav_path.unlink(missing_ok=True)

    output_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_handle.close()
    output = Path(output_handle.name)
    sample_rate = 16000
    lead_silence_seconds = 1
    between_silence_seconds = 9
    trailing_silence_seconds = 12
    silence_second = b"\x00\x00" * sample_rate
    with wave.open(str(output), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(silence_second * lead_silence_seconds)
        for _, frames, _ in speech:
            wav_out.writeframes(frames)
            wav_out.writeframes(silence_second * between_silence_seconds)
        wav_out.writeframes(silence_second * trailing_silence_seconds)

    duration = lead_silence_seconds + trailing_silence_seconds + sum(
        frames_count / sample_rate + between_silence_seconds
        for _, _, frames_count in speech
    )
    return output, {
        "questions": QUESTIONS,
        "duration_seconds": round(duration, 3),
        "between_silence_seconds": between_silence_seconds,
        "sample_rate": sample_rate,
    }


def _message_summary(payload: str | bytes) -> dict[str, Any] | None:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "binary", "bytes": len(payload)}
    try:
        message = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(message, dict):
        return None
    summary: dict[str, Any] = {"type": message.get("type")}
    if "kind" in message:
        summary["kind"] = message.get("kind")
    if message.get("text"):
        summary["text"] = str(message["text"])
    if message.get("audio"):
        summary["audio_chars"] = len(message["audio"])
    if message.get("audio_data"):
        summary["audio_chars"] = len(message["audio_data"])
    if message.get("type") == "session.init":
        config = message.get("payload", {}).get("config", {})
        summary["config"] = config
        summary["use_tts"] = message.get("payload", {}).get("use_tts")
    if message.get("type") == "session.created":
        summary["session_id"] = message.get("session_id")
    return summary


def _derive_turns(received: list[dict[str, Any]]) -> tuple[list[str], int, int]:
    turns: list[str] = []
    current: list[str] = []
    native_audio_events = 0
    listen_events = 0
    for message in received:
        msg_type = message.get("type")
        kind = message.get("kind")
        if (
            (msg_type == "response.output.delta" and kind == "audio")
            or (msg_type == "response.output_audio.delta" and message.get("audio_chars"))
            or (msg_type == "result" and message.get("audio_chars"))
        ):
            native_audio_events += 1
        if msg_type == "response.output.delta" and kind == "text":
            current.append(message.get("text", ""))
        elif msg_type == "response.output_audio.delta" and message.get("text"):
            current.append(message["text"])
        is_listen = msg_type == "response.listen" or (
            msg_type == "response.output.delta" and kind == "listen"
        )
        if is_listen:
            listen_events += 1
            text = "".join(current).strip()
            if text:
                turns.append(text)
                current.clear()
    return turns, native_audio_events, listen_events


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    microphone, fixture = _build_microphone_fixture()
    sent: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = []
    websocket_urls: list[str] = []
    tts_requests: list[dict[str, Any]] = []
    tts_responses: list[dict[str, Any]] = []
    response_404s: list[dict[str, Any]] = []
    console_errors: list[dict[str, Any]] = []
    page_errors: list[str] = []
    request_failures: list[dict[str, Any]] = []

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
                viewport={"width": 1440, "height": 1000},
            )
            context.add_init_script(
                """
                (() => {
                  // Chromium's built-in fake camera may be a static 2x2 frame.
                  // Keep its fake microphone, but replace only the video track
                  // with an animated 640x480 stream for a real freshness check.
                  const nativeGetUserMedia = navigator.mediaDevices.getUserMedia
                    .bind(navigator.mediaDevices);
                  navigator.mediaDevices.getUserMedia = async constraints => {
                    const stream = await nativeGetUserMedia(constraints);
                    if (!constraints?.video || !HTMLCanvasElement.prototype.captureStream) {
                      return stream;
                    }
                    const canvas = document.createElement('canvas');
                    canvas.width = 640;
                    canvas.height = 480;
                    const ctx = canvas.getContext('2d');
                    let frame = 0;
                    const draw = () => {
                      const hue = frame++ % 360;
                      ctx.fillStyle = `hsl(${hue}, 70%, 45%)`;
                      ctx.fillRect(0, 0, canvas.width, canvas.height);
                      ctx.fillStyle = '#fff';
                      ctx.font = 'bold 42px sans-serif';
                      ctx.fillText(`OpenGlass frame ${frame}`, 36, 72);
                    };
                    draw();
                    const timer = setInterval(draw, 100);
                    const canvasTrack = canvas.captureStream(10).getVideoTracks()[0];
                    canvasTrack.addEventListener(
                      'ended', () => clearInterval(timer), {once: true}
                    );
                    for (const track of stream.getVideoTracks()) {
                      stream.removeTrack(track);
                      track.stop();
                    }
                    stream.addTrack(canvasTrack);
                    return stream;
                  };
                  window.__viAudioAudit = [];
                  window.__viStreamAudit = [];
                  window.__wsAudioAudit = [];
                  window.addEventListener('openglass:vi-tts', event => {
                    window.__viStreamAudit.push(event.detail);
                  });
                  const nativeWsSend = WebSocket.prototype.send;
                  WebSocket.prototype.send = function(data) {
                    try {
                      const message = JSON.parse(data);
                      if (message.type === 'audio_chunk' && message.audio_base64) {
                        const raw = atob(message.audio_base64);
                        let nonzero = false;
                        for (let i = 0; i < raw.length; i++) {
                          if (raw.charCodeAt(i) !== 0) { nonzero = true; break; }
                        }
                        window.__wsAudioAudit.push({
                          at: performance.now(), nonzero, bytes: raw.length,
                        });
                      }
                    } catch (_) {}
                    return nativeWsSend.call(this, data);
                  };
                  const NativeAudio = window.Audio;
                  let nextId = 1;
                  window.Audio = new Proxy(NativeAudio, {
                    construct(target, args) {
                      const audio = Reflect.construct(target, args);
                      const record = {
                        id: nextId++,
                        createdAt: performance.now(),
                        srcPrefix: String(args[0] || '').slice(0, 32),
                        srcLength: String(args[0] || '').length,
                        events: [],
                      };
                      window.__viAudioAudit.push(record);
                      for (const name of ['loadstart', 'canplay', 'play', 'playing', 'pause', 'ended', 'error']) {
                        audio.addEventListener(name, () => record.events.push({
                          name,
                          at: performance.now(),
                          currentTime: audio.currentTime,
                          duration: Number.isFinite(audio.duration) ? audio.duration : null,
                          error: audio.error ? {code: audio.error.code, message: audio.error.message} : null,
                        }));
                      }
                      return audio;
                    }
                  });
                })();
                """
            )
            page = context.new_page()

            def on_websocket(websocket: Any) -> None:
                websocket_urls.append(websocket.url)

                def on_sent(payload: str | bytes) -> None:
                    summary = _message_summary(payload)
                    if summary:
                        sent.append(summary)

                def on_received(payload: str | bytes) -> None:
                    summary = _message_summary(payload)
                    if summary:
                        received.append(summary)

                websocket.on("framesent", on_sent)
                websocket.on("framereceived", on_received)

            page.on("websocket", on_websocket)
            page.on(
                "console",
                lambda message: console_errors.append(
                    {"text": message.text, "location": message.location}
                )
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def on_request(request: Any) -> None:
                if "/api/tts/vi" in request.url:
                    tts_requests.append(
                        {
                            "url": request.url,
                            "method": request.method,
                            "post_data": request.post_data,
                            "started_at": time.time(),
                        }
                    )

            def on_response(response: Any) -> None:
                item = {
                    "url": response.url,
                    "status": response.status,
                    "resource_type": response.request.resource_type,
                }
                if response.status == 404:
                    response_404s.append(item)
                if "/api/tts/vi" in response.url:
                    tts_responses.append(item | {"received_at": time.time()})

            def on_request_failed(request: Any) -> None:
                request_failures.append(
                    {
                        "url": request.url,
                        "resource_type": request.resource_type,
                        "failure": request.failure,
                    }
                )

            page.on("request", on_request)
            page.on("response", on_response)
            page.on("requestfailed", on_request_failed)

            started = time.perf_counter()
            page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_load_state("networkidle", timeout=30_000)
            body_has_content = page.locator("body").inner_text().strip() != ""
            overlay_count = page.locator(
                "[data-nextjs-dialog], .vite-error-overlay, #webpack-dev-server-client-overlay"
            ).count()

            vi_button = page.locator('.preset-btn[data-preset-id="vietnamese_call"]')
            vi_button.wait_for(state="visible", timeout=30_000)
            vi_button.click()
            page.wait_for_function(
                """() => document.querySelector('.preset-btn[data-preset-id="vietnamese_call"]')
                  ?.classList.contains('active')""",
                timeout=15_000,
            )
            page.wait_for_function(
                """() => Array.from(document.querySelectorAll('.preset-loading-overlay'))
                  .every(el => getComputedStyle(el).display === 'none')""",
                timeout=30_000,
            )
            prompt_value = page.locator("#systemPrompt").input_value()
            selected_preset = vi_button.evaluate("el => el.classList.contains('active')")

            video_before = page.locator("#videoEl").evaluate(
                """el => ({readyState: el.readyState, currentTime: el.currentTime,
                  width: el.videoWidth, height: el.videoHeight, paused: el.paused})"""
            )
            page.wait_for_timeout(1500)
            video_after = page.locator("#videoEl").evaluate(
                """el => ({readyState: el.readyState, currentTime: el.currentTime,
                  width: el.videoWidth, height: el.videoHeight, paused: el.paused})"""
            )

            page.locator("#btnStart").click()
            page.wait_for_function(
                "() => !document.querySelector('#btnStop').disabled",
                timeout=45_000,
            )

            # Allow the real-time fixture to complete. Return early only after five
            # text/listen turns, five successful TTS responses, and five audio ends.
            deadline = time.monotonic() + max(150, fixture["duration_seconds"] + 60)
            while time.monotonic() < deadline:
                turns, _, _ = _derive_turns(received)
                audio_audit = page.evaluate("() => window.__viAudioAudit || []")
                stream_audit = page.evaluate("() => window.__viStreamAudit || []")
                ended = sum(
                    1
                    for record in audio_audit
                    if any(event.get("name") == "ended" for event in record.get("events", []))
                )
                stream_ended = sum(
                    event.get("type") == "speech-ended" for event in stream_audit
                )
                successful_tts = sum(item["status"] == 200 for item in tts_responses)
                if (
                    len(turns) >= 5
                    and successful_tts >= 5
                    and max(ended, stream_ended) >= 5
                ):
                    break
                page.wait_for_timeout(500)

            # Let the final response settle back into listening/live state.
            page.wait_for_timeout(1500)
            turns, native_audio_events, listen_events = _derive_turns(received)
            audio_audit = page.evaluate("() => window.__viAudioAudit || []")
            stream_audit = page.evaluate("() => window.__viStreamAudit || []")
            ws_audio_audit = page.evaluate("() => window.__wsAudioAudit || []")
            status_text = page.locator("#serviceStatus").inner_text().strip()
            start_text = page.locator("#btnStart").inner_text().strip()
            stop_enabled = not page.locator("#btnStop").is_disabled()
            conversation = page.locator("#conversationLog").inner_text().strip()
            session_init = next(
                (message for message in sent if message.get("type") == "session.init"),
                None,
            )
            final_receive = next(
                (
                    message
                    for message in reversed(received)
                    if message.get("type")
                    in {"response.listen", "response.output.delta", "response.output_audio.delta"}
                ),
                None,
            )
            screenshot_path = RESULT_DIR / "native_vietnamese_soak.png"
            page.screenshot(path=str(screenshot_path), full_page=True)

            ended_audio = [
                record
                for record in audio_audit
                if any(event.get("name") == "ended" for event in record.get("events", []))
            ]
            errored_audio = [
                record
                for record in audio_audit
                if any(event.get("name") == "error" for event in record.get("events", []))
            ]
            unexpected_console_errors = [
                item
                for item in console_errors
                if not (
                    item.get("location", {}).get("url", "").endswith("/favicon.ico")
                    and "404" in item.get("text", "")
                )
            ]
            played_audio = [
                record
                for record in audio_audit
                if any(event.get("name") in {"play", "playing"} for event in record.get("events", []))
            ]
            speech_intervals: list[tuple[float, float]] = []
            speech_started_at: float | None = None
            for event in stream_audit:
                if event.get("type") == "speech-started":
                    speech_started_at = float(event["at"])
                elif event.get("type") == "speech-ended" and speech_started_at is not None:
                    speech_intervals.append((speech_started_at, float(event["at"])))
                    speech_started_at = None
            ws_audio_during_speech = [
                message
                for message in ws_audio_audit
                if any(start <= float(message["at"]) <= end for start, end in speech_intervals)
            ]
            mic_suppressed_events = [
                event for event in stream_audit if event.get("type") == "mic-suppressed"
            ]
            stream_speech_started = sum(
                event.get("type") == "speech-started" for event in stream_audit
            )
            stream_speech_ended = len(speech_intervals)
            generate_audio_false = bool(
                session_init
                and session_init.get("config", {}).get("generate_audio") is False
            )
            external_tts_ok = sum(item["status"] == 200 for item in tts_responses)
            last_is_listen = bool(
                final_receive
                and (
                    final_receive.get("type") == "response.listen"
                    or (
                        final_receive.get("type") == "response.output.delta"
                        and final_receive.get("kind") == "listen"
                    )
                )
            )
            camera_fresh = bool(
                video_after["readyState"] >= 2
                and video_after["width"] > 0
                and video_after["height"] > 0
                and video_after["currentTime"] > video_before["currentTime"]
            )
            checks = {
                "page_has_meaningful_content": body_has_content,
                "no_error_overlay": overlay_count == 0,
                "vietnamese_preset_selected": selected_preset,
                "vietnamese_prompt_loaded": "tiếng Việt" in prompt_value,
                "generate_audio_false": generate_audio_false,
                "use_tts_false": bool(session_init and session_init.get("use_tts") is False),
                "zero_native_audio_events": native_audio_events == 0,
                "five_protocol_turns": len(turns) >= 5,
                "five_external_tts_200": external_tts_ok >= 5,
                "five_external_playbacks_started": max(
                    len(played_audio), stream_speech_started
                )
                >= 5,
                "five_external_playbacks_ended": max(
                    len(ended_audio), stream_speech_ended
                )
                >= 5,
                "no_owned_audio_error": len(errored_audio) == 0,
                "mic_was_suppressed_during_tts": len(mic_suppressed_events) >= 1,
                # No packet during playback is as safe as an explicit zero packet;
                # the separate suppression check proves this branch was exercised.
                "no_nonzero_mic_audio_during_tts": not any(
                    message["nonzero"] for message in ws_audio_during_speech
                ),
                "no_self_triggered_extra_turn": len(turns) == len(QUESTIONS),
                "returned_to_listening": last_is_listen,
                "ui_still_live": stop_enabled and "live" in start_text.lower(),
                "camera_preview_fresh": camera_fresh,
                "no_page_errors": not page_errors,
                "no_failed_requests": not request_failures,
                "no_unexpected_console_errors": not unexpected_console_errors,
            }
            report = {
                "url": URL,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "fixture": fixture,
                "checks": checks,
                "passed": all(checks.values()),
                "session_init": session_init,
                "websocket_urls": websocket_urls,
                "turn_count": len(turns),
                "turns": turns,
                "listen_events": listen_events,
                "native_audio_events": native_audio_events,
                "tts_request_count": len(tts_requests),
                "tts_response_statuses": [item["status"] for item in tts_responses],
                "owned_audio": {
                    "created": len(audio_audit),
                    "played": len(played_audio),
                    "ended": len(ended_audio),
                    "errored": len(errored_audio),
                    "records": audio_audit,
                },
                "stream_audio": {
                    "speech_started": stream_speech_started,
                    "speech_ended": stream_speech_ended,
                    "mic_suppressed_events": len(mic_suppressed_events),
                    "ws_audio_during_speech": ws_audio_during_speech,
                    "events": stream_audit,
                },
                "ui": {
                    "service_status": status_text,
                    "start_text": start_text,
                    "stop_enabled": stop_enabled,
                    "video_before": video_before,
                    "video_after": video_after,
                    "conversation_excerpt": conversation[-2000:],
                },
                "response_404s": response_404s,
                "console_errors": console_errors,
                "unexpected_console_errors": unexpected_console_errors,
                "page_errors": page_errors,
                "request_failures": request_failures,
                "screenshot": str(screenshot_path),
            }
            report_path = RESULT_DIR / "report.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))

            if stop_enabled:
                page.locator("#btnStop").click()
            browser.close()
            if not report["passed"]:
                raise SystemExit(1)
    finally:
        microphone.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
