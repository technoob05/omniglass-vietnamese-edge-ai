"""Smoke the warm vi->en switch and English VLM prompt without restarting vision."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("OPENGLASS_NATIVE_BASE_URL", "https://127.0.0.1:8006")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def main() -> None:
    sockets: list[str] = []
    input_payloads: list[dict] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=str(EDGE), args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required",
        ])
        context = browser.new_context(ignore_https_errors=True, permissions=["camera", "microphone"])
        page = context.new_page()

        def on_socket(socket):
            sockets.append(socket.url)
            def sent(payload):
                try:
                    event = json.loads(payload.decode() if isinstance(payload, bytes) else payload)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    return
                if event.get("type") == "input.append":
                    input_payloads.append(event["input"])
            socket.on("framesent", sent)

        page.on("websocket", on_socket)
        page.goto(f"{BASE_URL}/vi", wait_until="networkidle")
        page.locator("#start").click()
        page.wait_for_function("() => document.querySelector('#status').textContent.includes('Đang nghe')", timeout=45000)
        session = page.evaluate("() => window.__omniglassViAssistant.perceptionSession")
        page.locator("#language").select_option("en")
        page.wait_for_function("() => document.documentElement.lang === 'en' && document.querySelector('#status').textContent.includes('Listening')", timeout=30000)
        language_session = page.evaluate("() => window.__omniglassViAssistant.perceptionSession")
        page.locator("#workflow").select_option("find_object")
        page.wait_for_function("() => document.querySelector('#workflowBadge').textContent.includes('Find object')", timeout=10000)
        page.evaluate("() => { window.__omniglassViAssistant.worklet.port.onmessage=null; }")
        page.wait_for_timeout(3200)
        page.evaluate("() => window.__omniglassViAssistant.submitText('Where is the bottle?')")
        page.wait_for_function("() => document.querySelectorAll('#conversation .message.assistant').length >= 1", timeout=90000)
        messages = input_payloads[-1]["messages"]
        user_content = messages[-1]["content"]
        report = {
            "passed": any("language=en" in url for url in sockets)
            and language_session != session
            and page.evaluate("() => window.__omniglassViAssistant.perceptionSession") != language_session
            and "You are OpenGlass" in messages[0]["content"]
            and "[Find object mode]" in messages[0]["content"]
            and "final frame is current" in messages[0]["content"]
            and page.locator("#brandTitle").inner_text() == "OpenGlass"
            and sum(item.get("type") == "image" for item in user_content) >= 2,
            "english_asr_socket": next((url for url in sockets if "language=en" in url), None),
            "language_switch_reset_session": language_session != session,
            "workflow_switch_reset_session": page.evaluate("() => window.__omniglassViAssistant.perceptionSession") != language_session,
            "workflow": page.locator("#workflowBadge").inner_text(),
            "system_prompt": messages[0]["content"],
            "image_count": sum(item.get("type") == "image" for item in user_content),
            "answer": page.locator("#conversation .message.assistant").last.inner_text(),
        }
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
