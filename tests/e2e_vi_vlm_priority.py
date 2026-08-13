"""Acceptance: a user question interrupts rule audio and routes to multi-frame VLM."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE_URL = os.environ.get("OPENGLASS_NATIVE_BASE_URL", "https://127.0.0.1:8006")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def decode(payload):
    try:
        if isinstance(payload, bytes):
            payload = payload.decode()
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return {}


def main() -> None:
    chat_inputs: list[dict] = []
    rule_queries = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=str(EDGE),
            args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(ignore_https_errors=True, permissions=["camera", "microphone"])
        page = context.new_page()

        def on_socket(socket):
            def sent(payload):
                event = decode(payload)
                if event.get("type") != "input.append":
                    return
                messages = event.get("input", {}).get("messages", [])
                content = messages[-1].get("content", []) if messages else []
                chat_inputs.append({
                    "text": next((item.get("text") for item in content if item.get("type") == "text"), None),
                    "image_count": sum(item.get("type") == "image" for item in content),
                    "message_count": len(messages),
                    "system_prompt": messages[0].get("content", "") if messages else "",
                })
            socket.on("framesent", sent)

        def on_request(request):
            nonlocal rule_queries
            if request.url.endswith("/api/perception/vi/query"):
                rule_queries += 1

        page.on("websocket", on_socket)
        page.on("request", on_request)
        page.goto(f"{BASE_URL}/vi", wait_until="networkidle")
        page.locator("#start").click()
        page.wait_for_function("() => document.querySelector('#status').textContent.includes('Đang nghe')", timeout=45000)
        page.wait_for_timeout(4200)
        page.wait_for_function("() => window.__omniglassViAssistant.conversationOwnsAudio === false", timeout=10000)
        page.evaluate("() => { window.__omniglassViAssistant.worklet.port.onmessage=null; window.__omniglassViAssistant.playSafetyAlert('Cẩn thận, có một vật cản rất gần ở chính giữa phía trước.') }")
        page.wait_for_function("() => window.__omniglassViAssistant.safetySource !== null", timeout=10000)
        page.evaluate("""() => {
          const assistant = window.__omniglassViAssistant;
          const speechBlock = new Float32Array(Math.round(assistant.audioContext.sampleRate * 0.2));
          speechBlock.fill(0.2);
          for (let index = 0; index < 5; index += 1) assistant.onAudio(speechBlock, assistant.runEpoch);
        }""")
        page.wait_for_timeout(300)
        echo_rejected = page.evaluate("() => ({sourcePresent:window.__omniglassViAssistant.safetySource !== null, speaking:window.__omniglassViAssistant.safetySpeaking, ownsAudio:window.__omniglassViAssistant.conversationOwnsAudio})")
        page.evaluate("() => window.__omniglassViAssistant.beginPushToTalk()")
        page.wait_for_function("() => window.__omniglassViAssistant.conversationOwnsAudio === true", timeout=5000)
        acoustic_interrupt = page.evaluate("() => ({source:window.__omniglassViAssistant.safetySource, speaking:window.__omniglassViAssistant.safetySpeaking, state:window.__omniglassViAssistant.state})")
        page.evaluate("() => window.__omniglassViAssistant.endPushToTalk()")
        page.evaluate("() => window.__omniglassViAssistant.submitText('Vật cản gần nhất cách khoảng bao nhiêu mét?')")
        interrupted = page.evaluate("() => ({source:window.__omniglassViAssistant.safetySource, speaking:window.__omniglassViAssistant.safetySpeaking})")
        page.wait_for_function("() => document.querySelectorAll('#conversation .message.assistant').length >= 1", timeout=90000)
        page.wait_for_function("() => document.querySelector('#status').textContent.includes('Đang nghe')", timeout=45000)
        report = {
            "passed": len(chat_inputs) == 1 and chat_inputs[0]["image_count"] >= 2
            and rule_queries == 0 and interrupted == {"source": None, "speaking": False}
            and acoustic_interrupt == {"source": None, "speaking": False, "state": "speech"}
            and echo_rejected == {"sourcePresent": True, "speaking": True, "ownsAudio": False}
            and "Bạn là OpenGlass" in chat_inputs[0]["system_prompt"]
            and "[Chế độ hỏi đáp thị giác]" in chat_inputs[0]["system_prompt"],
            "acoustic_rule_interrupt": acoustic_interrupt,
            "speaker_echo_rejected": echo_rejected,
            "rule_audio_after_question": interrupted,
            "rule_query_calls": rule_queries,
            "vlm_inputs": chat_inputs,
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
