"""Browser acceptance: a deterministic hazard speaks without invoking VLM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("OPENGLASS_NATIVE_BASE_URL", "https://127.0.0.1:8006")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def main() -> None:
    chat_inputs = 0
    tts_calls = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=str(EDGE),
            args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(ignore_https_errors=True, permissions=["camera", "microphone"])
        page = context.new_page()

        def on_ws(socket):
            nonlocal chat_inputs
            socket.on("framesent", lambda frame: set_chat(frame))

        def set_chat(frame):
            nonlocal chat_inputs
            try:
                if json.loads(frame).get("type") == "input.append":
                    chat_inputs += 1
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                pass

        page.on("websocket", on_ws)
        page.on("request", lambda request: count_tts(request))

        def count_tts(request):
            nonlocal tts_calls
            if request.url.endswith("/api/tts/vi"):
                tts_calls += 1

        page.goto(f"{BASE_URL}/vi", wait_until="networkidle")
        page.locator("#start").click()
        page.wait_for_function("() => window.__omniglassViAssistant?.audioContext?.state === 'running'", timeout=30000)
        page.evaluate("() => { clearTimeout(window.__omniglassViAssistant.perceptionTimer); window.__omniglassViAssistant.mediaStream = null; }")
        page.wait_for_timeout(2000)
        page.evaluate(
            """() => window.__omniglassViAssistant.renderSafety({
              state:'danger', primary_alert:{message_vi:'Cẩn thận! Ghế rất gần ở giữa.', rule:'center_path_very_close', confidence:.94}
            })"""
        )
        page.evaluate(
            """() => window.__omniglassViAssistant.enqueueSafetyAlert({
              should_announce:true, message_vi:'Cẩn thận! Ghế rất gần ở giữa.'
            })"""
        )
        page.wait_for_function("() => window.__omniglassViAssistant.safetySource !== null", timeout=10000)
        speed = page.evaluate("() => window.__omniglassViAssistant.safetySource?.playbackRate.value")
        page.wait_for_function("() => window.__omniglassViAssistant.safetySpeaking === false", timeout=30000)
        report = {
            "passed": page.locator("#safetyPanel").get_attribute("class").endswith("danger")
            and "Ghế rất gần" in page.locator("#safetyMessage").inner_text()
            and speed == 1.5 and tts_calls == 1 and chat_inputs == 0,
            "safety_playback_rate": speed,
            "tts_calls": tts_calls,
            "vlm_chat_inputs": chat_inputs,
            "message": page.locator("#safetyMessage").inner_text(),
        }
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
