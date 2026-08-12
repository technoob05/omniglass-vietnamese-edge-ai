"""Focused browser acceptance for the production Vietnamese profile.

No production service is mutated. The TTS failure case uses Playwright's
per-browser request interception; the stop/restart case uses fake media.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


URL = os.environ.get("OPENGLASS_VI_URL", "https://127.0.0.1:8006/vi")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
RESULT_DIR = Path(__file__).resolve().parents[1] / "results" / "vi_profile_resilience"


def silent_microphone(seconds: float = 180.0) -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\x00\x00" * int(16_000 * seconds))
    return path


def short_wav_base64() -> str:
    sample_rate = 16_000
    samples = [
        int(0.12 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(int(sample_rate * 0.28))
    ]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(struct.pack("<h", value) for value in samples))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def attach_diagnostics(page: Any) -> dict[str, list[Any]]:
    diagnostics: dict[str, list[Any]] = {
        "console_errors": [], "page_errors": [], "failed_requests": [], "http_errors": []
    }
    page.on(
        "console",
        lambda message: diagnostics["console_errors"].append(
            {"text": message.text, "location": message.location}
        ) if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: diagnostics["page_errors"].append(str(error)))
    page.on(
        "requestfailed",
        lambda request: diagnostics["failed_requests"].append(
            {"url": request.url, "failure": request.failure, "resource_type": request.resource_type}
        ),
    )
    page.on(
        "response",
        lambda response: diagnostics["http_errors"].append(
            {"url": response.url, "status": response.status, "resource_type": response.request.resource_type}
        ) if response.status >= 400 else None,
    )
    return diagnostics


def unexpected_console_errors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for item in items
        if not (
            item.get("location", {}).get("url", "").endswith("/favicon.ico")
            and "404" in item.get("text", "")
        )
    ]


def open_profile(context: Any) -> tuple[Any, dict[str, list[Any]]]:
    page = context.new_page()
    diagnostics = attach_diagnostics(page)
    page.goto(URL, wait_until="networkidle", timeout=30_000)
    page.wait_for_function("() => !!window.__omniglassViAssistant", timeout=15_000)
    assert page.locator("body").inner_text().strip()
    assert page.locator("[data-nextjs-dialog], .vite-error-overlay, #webpack-dev-server-client-overlay").count() == 0
    return page, diagnostics


def barge_in_acceptance(context: Any) -> dict[str, Any]:
    page, diagnostics = open_profile(context)
    page.evaluate(
        """() => {
          const assistant = window.__omniglassViAssistant;
          assistant.setState('listening', 'test-ready');
          assistant.askVlm = () => new Promise(resolve => { window.__resolveOldVlm = resolve; });
          assistant.speak = async () => { window.__oldSpeakCalled = true; };
          window.__oldTurnPromise = assistant.consumeFinal({
            type: 'asr.final', final_id: 'old-final-id', immutable: true,
            text: 'Câu hỏi cũ', routing: {vlm_eligible: true}
          }, assistant.runEpoch);
        }"""
    )
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'thinking'")
    cancellation = page.evaluate(
        """() => {
          const assistant = window.__omniglassViAssistant;
          const controller = new AbortController();
          const audit = {sourceStops: 0, audioPauses: 0, chatCloses: 0};
          assistant.ttsAbort = controller;
          assistant.ttsSources.add({stop: () => { audit.sourceStops += 1; }});
          assistant.ttsAudio = {pause: () => { audit.audioPauses += 1; }, currentTime: 9};
          assistant.chatSocket = {close: () => { audit.chatCloses += 1; }};
          assistant.state = 'speaking';
          const before = assistant.turnEpoch;
          assistant.bargeIn();
          return {
            before, after: assistant.turnEpoch, state: assistant.state,
            abortSignal: controller.signal.aborted,
            sourcesRemaining: assistant.ttsSources.size,
            audioCleared: assistant.ttsAudio === null,
            chatCleared: assistant.chatSocket === null,
            ...audit,
          };
        }"""
    )
    page.evaluate("() => window.__resolveOldVlm('CÂU TRẢ LỜI CŨ KHÔNG ĐƯỢC HIỆN')")
    page.evaluate("() => window.__oldTurnPromise")
    old_result = page.evaluate(
        """() => ({
          state: window.__omniglassViAssistant.state,
          oldSpeakCalled: !!window.__oldSpeakCalled,
          text: document.querySelector('#conversation').innerText,
        })"""
    )

    page.evaluate(
        """() => {
          const assistant = window.__omniglassViAssistant;
          assistant.askVlm = async () => 'Câu trả lời mới hợp lệ.';
          assistant.speak = async () => {};
          window.__newTurnPromise = assistant.consumeFinal({
            type: 'asr.final', final_id: 'new-final-id', immutable: true,
            text: 'Câu hỏi mới', routing: {vlm_eligible: true}
          }, assistant.runEpoch);
        }"""
    )
    page.evaluate("() => window.__newTurnPromise")
    next_turn = page.evaluate(
        """() => ({
          state: window.__omniglassViAssistant.state,
          text: document.querySelector('#conversation').innerText,
          consumed: Array.from(window.__omniglassViAssistant.consumedFinalIds),
        })"""
    )
    screenshot = RESULT_DIR / "barge_in.png"
    page.screenshot(path=str(screenshot), full_page=True)
    checks = {
        "epoch_advanced": cancellation["after"] == cancellation["before"] + 1,
        "tts_abort_signal": cancellation["abortSignal"],
        "scheduled_sources_stopped": cancellation["sourceStops"] == 1 and cancellation["sourcesRemaining"] == 0,
        "fallback_audio_stopped": cancellation["audioPauses"] == 1 and cancellation["audioCleared"],
        "chat_cancelled": cancellation["chatCloses"] == 1 and cancellation["chatCleared"],
        "returned_to_listening": cancellation["state"] == "listening",
        "stale_answer_not_rendered": "CÂU TRẢ LỜI CŨ" not in old_result["text"] and not old_result["oldSpeakCalled"],
        "next_final_consumed": "Câu hỏi mới" in next_turn["text"] and "Câu trả lời mới hợp lệ." in next_turn["text"],
        "next_turn_returns_listening": next_turn["state"] == "listening",
        "unique_final_ids": next_turn["consumed"] == ["old-final-id", "new-final-id"],
    }
    page.close()
    return {"passed": all(checks.values()), "checks": checks, "cancellation": cancellation,
            "old_result": old_result, "next_turn": next_turn, "diagnostics": diagnostics,
            "screenshot": str(screenshot)}


def stop_restart_acceptance(context: Any) -> dict[str, Any]:
    # Keep this lifecycle test deterministic and independent of Pod-B timing.
    # The same-origin ASR socket and browser capture devices are faked; the
    # deployed page code, its AudioContext pipeline and lifecycle remain real.
    # A generated MediaStream avoids a Chromium fake-audio-file limitation on
    # reacquiring a second stream after every track has been stopped.
    context.add_init_script(
        """
        (() => {
          const NativeWebSocket = window.WebSocket;
          window.__fakeAsrSockets = [];
          window.__fakeMediaStreams = [];
          navigator.mediaDevices.getUserMedia = async () => {
            const canvas = document.createElement('canvas');
            canvas.width = 640; canvas.height = 360;
            const ctx = canvas.getContext('2d');
            let tick = 0;
            const draw = () => {
              ctx.fillStyle = `hsl(${tick++ % 360} 55% 28%)`;
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              ctx.fillStyle = 'white'; ctx.font = '28px sans-serif';
              ctx.fillText(`VI lifecycle frame ${tick}`, 28, 54);
            };
            draw();
            const timer = setInterval(draw, 50);
            const videoStream = canvas.captureStream(20);
            const producer = new AudioContext();
            const oscillator = producer.createOscillator();
            const gain = producer.createGain(); gain.gain.value = 0;
            const destination = producer.createMediaStreamDestination();
            oscillator.connect(gain).connect(destination); oscillator.start();
            const stream = new MediaStream([
              ...videoStream.getVideoTracks(), ...destination.stream.getAudioTracks()
            ]);
            const nativeStop = MediaStreamTrack.prototype.stop;
            let remaining = stream.getTracks().length;
            for (const track of stream.getTracks()) {
              track.stop = function() {
                nativeStop.call(this);
                remaining -= 1;
                if (remaining === 0) {
                  clearInterval(timer); oscillator.stop(); void producer.close();
                }
              };
            }
            window.__fakeMediaStreams.push(stream);
            return stream;
          };
          class FakeAsrSocket {
            constructor(url) {
              this.url = String(url); this.readyState = 0; this.sent = [];
              window.__fakeAsrSockets.push(this);
              setTimeout(() => {
                if (this.readyState !== 0) return;
                this.readyState = 1;
                this.onopen?.({});
                this.onmessage?.({data: JSON.stringify({
                  type:'asr.ready', sample_rate:16000, channels:1,
                  encoding:'pcm_s16le', final_contract:'immutable'
                })});
              }, 20);
            }
            send(data) { this.sent.push(data); }
            close() {
              if (this.readyState >= 2) return;
              this.readyState = 3;
              setTimeout(() => this.onclose?.({code:1000, reason:'test close'}), 0);
            }
          }
          window.WebSocket = new Proxy(NativeWebSocket, {
            construct(target, args) {
              return String(args[0]).includes('/v1/asr/vi')
                ? new FakeAsrSocket(args[0])
                : Reflect.construct(target, args);
            }
          });
        })();
        """
    )
    page, diagnostics = open_profile(context)
    page.locator("#start").click()
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'listening'", timeout=45_000)
    first = page.evaluate(
        """() => { const a=window.__omniglassViAssistant; return {
          epoch:a.runEpoch, state:a.state, asr:a.asrSocket?.readyState,
          tracks:a.mediaStream ? a.mediaStream.getTracks().map(t=>t.readyState) : [],
          trackIds:a.mediaStream ? a.mediaStream.getTracks().map(t=>t.id) : [],
          asrInstances:window.__fakeAsrSockets.length,
          mediaInstances:window.__fakeMediaStreams.length,
          videoTime:document.querySelector('#camera').currentTime,
        }}"""
    )
    page.locator("#stop").click()
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'stopped'")
    page.wait_for_timeout(1200)
    stopped = page.evaluate(
        """() => { const a=window.__omniglassViAssistant; return {
          epoch:a.runEpoch, turnEpoch:a.turnEpoch, state:a.state,
          mediaNull:a.mediaStream===null, audioContextNull:a.audioContext===null,
          asrNull:a.asrSocket===null, chatNull:a.chatSocket===null,
          ttsAbortNull:a.ttsAbort===null, sources:a.ttsSources.size,
          videoSourceNull:document.querySelector('#camera').srcObject===null,
          startDisabled:document.querySelector('#start').disabled,
          stopDisabled:document.querySelector('#stop').disabled,
        }}"""
    )
    stopped_shot = RESULT_DIR / "stopped_final.png"
    page.screenshot(path=str(stopped_shot), full_page=True)

    page.locator("#start").click()
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'listening'", timeout=45_000)
    page.wait_for_timeout(600)
    restarted = page.evaluate(
        """() => { const a=window.__omniglassViAssistant; return {
          epoch:a.runEpoch, state:a.state, asr:a.asrSocket?.readyState,
          tracks:a.mediaStream ? a.mediaStream.getTracks().map(t=>t.readyState) : [],
          trackIds:a.mediaStream ? a.mediaStream.getTracks().map(t=>t.id) : [],
          asrInstances:window.__fakeAsrSockets.length,
          mediaInstances:window.__fakeMediaStreams.length,
          videoTime:document.querySelector('#camera').currentTime,
        }}"""
    )
    restarted_shot = RESULT_DIR / "restarted_live.png"
    page.screenshot(path=str(restarted_shot), full_page=True)
    page.locator("#stop").click()
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'stopped'")
    checks = {
        "first_run_listening": first["state"] == "listening" and first["asr"] == 1,
        "first_media_live": first["tracks"] and all(state == "live" for state in first["tracks"]),
        "stop_epoch_final": stopped["epoch"] > first["epoch"] and stopped["state"] == "stopped",
        "stop_released_resources": all([
            stopped["mediaNull"], stopped["audioContextNull"], stopped["asrNull"],
            stopped["chatNull"], stopped["ttsAbortNull"], stopped["videoSourceNull"],
            stopped["sources"] == 0,
        ]),
        "stop_ui_final": not stopped["startDisabled"] and stopped["stopDisabled"],
        "restart_new_epoch": restarted["epoch"] > stopped["epoch"],
        "restart_listening": restarted["state"] == "listening" and restarted["asr"] == 1,
        "restart_created_new_asr_socket": restarted["asrInstances"] == first["asrInstances"] + 1,
        "restart_created_new_media_stream": restarted["mediaInstances"] == first["mediaInstances"] + 1
        and restarted["trackIds"] != first["trackIds"],
        "restart_media_live": restarted["tracks"] and all(state == "live" for state in restarted["tracks"]),
        # The first start can reach ASR-ready before the first headless video
        # clock tick.  Fresh track ids above prove reacquisition; require the
        # restarted camera itself to advance after the explicit settle period.
        "restart_camera_advances": restarted["videoTime"] > 0,
    }
    page.close()
    return {"passed": all(checks.values()), "checks": checks, "first": first, "stopped": stopped,
            "restarted": restarted, "diagnostics": diagnostics,
            "screenshots": [str(stopped_shot), str(restarted_shot)]}


def tts_failure_acceptance(context: Any) -> dict[str, Any]:
    page, diagnostics = open_profile(context)
    calls = {"stream": 0, "mms": 0}
    wav_b64 = short_wav_base64()

    def route_tts(route: Any) -> None:
        if route.request.url.endswith("/api/tts/vi/stream"):
            calls["stream"] += 1
            body = "\n".join([
                json.dumps({"type":"meta", "schema":"omniglass.vi-tts-stream.v1", "text":"fallback", "sample_rate":48000}),
                json.dumps({"type":"error", "code":"forced_test_failure", "message":"forced before first chunk", "chunks_emitted":0}),
                "",
            ])
            route.fulfill(status=200, content_type="application/x-ndjson", body=body)
        elif route.request.url.endswith("/api/tts/vi"):
            calls["mms"] += 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps({
                    "text":"Câu trả lời dùng giọng dự phòng.", "audio_wav_base64":wav_b64,
                    "sample_rate":16000, "duration_seconds":0.28, "model":"forced-test-mms",
                }),
            )
        else:
            route.continue_()

    # Playwright's single '*' does not cross a slash; register the stream and
    # fallback URLs separately so this test never touches either production TTS.
    page.route("**/api/tts/vi/stream", route_tts)
    page.route("**/api/tts/vi", route_tts)
    started = time.perf_counter()
    page.evaluate(
        """() => {
          const assistant=window.__omniglassViAssistant;
          assistant.setState('listening','test-ready');
          assistant.askVlm=async()=> 'Câu trả lời dùng giọng dự phòng.';
          window.__fallbackPromise=assistant.consumeFinal({
            type:'asr.final',final_id:'fallback-final-id',immutable:true,
            text:'Hãy thử giọng dự phòng',routing:{vlm_eligible:true}
          },assistant.runEpoch);
        }"""
    )
    page.evaluate("() => window.__fallbackPromise")
    elapsed = time.perf_counter() - started
    result = page.evaluate(
        """() => {const a=window.__omniglassViAssistant;return {
          state:a.state,ttsAbortNull:a.ttsAbort===null,ttsAudioNull:a.ttsAudio===null,
          sources:a.ttsSources.size,text:document.querySelector('#conversation').innerText,
          consumed:Array.from(a.consumedFinalIds),
        }}"""
    )
    screenshot = RESULT_DIR / "tts_fallback_recovered.png"
    page.screenshot(path=str(screenshot), full_page=True)
    checks = {
        "vieneu_failure_exercised": calls["stream"] == 1,
        "mms_fallback_exercised": calls["mms"] == 1,
        "fallback_answer_rendered": "Câu trả lời dùng giọng dự phòng." in result["text"],
        "returned_to_listening": result["state"] == "listening",
        "tts_resources_drained": result["ttsAbortNull"] and result["ttsAudioNull"] and result["sources"] == 0,
        "final_consumed_once": result["consumed"] == ["fallback-final-id"],
    }
    page.close()
    return {"passed": all(checks.values()), "checks": checks, "calls": calls, "result": result,
            "elapsed_seconds": round(elapsed, 3), "diagnostics": diagnostics,
            "screenshot": str(screenshot)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    microphone = silent_microphone()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=str(EDGE),
                args=[
                    "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-audio-capture={microphone.as_posix()}",
                    "--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling",
                ],
            )
            context = browser.new_context(
                ignore_https_errors=True, permissions=["camera", "microphone"],
                viewport={"width":1440,"height":1000},
            )
            report = {
                "url": URL,
                "barge_in": barge_in_acceptance(context),
                "stop_restart": stop_restart_acceptance(context),
                "tts_failure_recovery": tts_failure_acceptance(context),
            }
            for scenario in report.values():
                if isinstance(scenario, dict) and "diagnostics" in scenario:
                    scenario["unexpected_console_errors"] = unexpected_console_errors(
                        scenario["diagnostics"]["console_errors"]
                    )
            report["passed"] = all(
                report[name]["passed"]
                and not report[name]["diagnostics"]["page_errors"]
                and not report[name]["diagnostics"]["failed_requests"]
                and not report[name]["unexpected_console_errors"]
                for name in ("barge_in", "stop_restart", "tts_failure_recovery")
            )
            output = RESULT_DIR / "report.json"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            browser.close()
            if not report["passed"]:
                raise SystemExit(1)
    finally:
        microphone.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
