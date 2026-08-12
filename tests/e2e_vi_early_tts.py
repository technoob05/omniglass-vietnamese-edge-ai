"""Controlled browser acceptance for the local /vi early-sentence TTS asset.

The production page supplies the stable HTML/CSS shell, while Playwright
fulfills only vi-chat.js from the local workspace. No production file,
process, or endpoint is mutated.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


URL = os.environ.get("OPENGLASS_VI_URL", "https://127.0.0.1:8006/vi")
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ROOT = Path(__file__).resolve().parents[1]
LOCAL_JAVASCRIPT = ROOT / "native-overrides" / "vi-profile" / "vi-chat.js"
RESULT_DIR = ROOT / "results" / "vi_early_tts"


def pcm_base64(seconds: float, sample_rate: int = 48_000) -> str:
    samples = [
        int(0.035 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
        for index in range(int(sample_rate * seconds))
    ]
    return base64.b64encode(b"".join(struct.pack("<h", value) for value in samples)).decode("ascii")


def wav_base64(seconds: float = 0.08, sample_rate: int = 16_000) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


FAKE_BROWSER_RUNTIME = r"""
(() => {
  const NativeWebSocket = window.WebSocket;
  window.__chatScripts = [];
  window.__chatSockets = [];
  window.__chatDoneTimes = [];
  window.__ttsFetches = [];
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init = {}) => {
    const url = String(typeof input === 'string' ? input : input.url);
    if (url.includes('/api/tts/vi')) {
      let body = {};
      try { body = JSON.parse(init.body || '{}'); } catch (_) {}
      window.__ttsFetches.push({url, text:body.text || '', at:performance.now()});
    }
    return nativeFetch(input, init);
  };

  class FakeChatSocket {
    constructor(url) {
      this.url = String(url); this.readyState = 0; this.sent = [];
      window.__chatSockets.push(this);
      setTimeout(() => {
        if (this.readyState !== 0) return;
        this.readyState = 1; this.onopen?.({});
      }, 0);
    }
    send(raw) {
      const message = JSON.parse(raw); this.sent.push(message);
      if (message.type === 'session.init') {
        setTimeout(() => this.onmessage?.({data:JSON.stringify({type:'session.created'})}), 0);
      } else if (message.type === 'input.append') {
        const script = window.__chatScripts.shift();
        if (!script) throw new Error('Missing fake chat script');
        for (const item of script.events) {
          setTimeout(() => {
            if (item.data.type === 'response.done') window.__chatDoneTimes.push(performance.now());
            // Deliberately deliver late callbacks after close to test epoch guards.
            this.onmessage?.({data:JSON.stringify(item.data)});
          }, item.after);
        }
      }
    }
    close() {
      if (this.readyState >= 2) return;
      this.readyState = 3;
      setTimeout(() => this.onclose?.({code:1000, reason:'controlled close'}), 0);
    }
  }
  window.WebSocket = new Proxy(NativeWebSocket, {
    construct(target, args) {
      return String(args[0]).includes('/v1/realtime?mode=chat')
        ? new FakeChatSocket(args[0]) : Reflect.construct(target, args);
    }
  });
})();
"""


def attach_diagnostics(page: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {"console": [], "page": [], "requests": [], "http": []}
    page.on("console", lambda msg: result["console"].append({"type": msg.type, "text": msg.text}) if msg.type == "error" else None)
    page.on("pageerror", lambda error: result["page"].append(str(error)))
    page.on("requestfailed", lambda req: result["requests"].append({"url": req.url, "failure": req.failure}))
    page.on("response", lambda response: result["http"].append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
    return result


class TtsRouter:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.short_pcm = pcm_base64(0.12)
        self.ordered_pcm = pcm_base64(0.36)
        self.long_pcm = pcm_base64(2.0)
        self.fallback_wav = wav_base64()

    def __call__(self, route: Any) -> None:
        payload = route.request.post_data_json or {}
        text = str(payload.get("text") or "")
        kind = "stream" if route.request.url.endswith("/stream") else "mms"
        self.requests.append({"kind": kind, "text": text, "at": time.perf_counter()})
        if kind == "mms":
            route.fulfill(status=200, content_type="application/json", body=json.dumps({"audio_wav_base64": self.fallback_wav}))
            return
        if text == "Câu hai cần giọng dự phòng":
            events = [
                {"type": "meta", "sample_rate": 48_000},
                {"type": "error", "message": "controlled pre-audio failure"},
            ]
        else:
            audio = self.long_pcm if text.startswith("Câu cũ") else self.ordered_pcm if text.startswith("Phía trước") else self.short_pcm
            events = [
                {"type": "meta", "sample_rate": 48_000},
                {"type": "audio", "pcm_s16le_base64": audio},
                {"type": "done"},
            ]
        body = "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events)
        route.fulfill(status=200, content_type="application/x-ndjson", body=body)


def open_page(context: Any, early_tts: bool = True) -> tuple[Any, TtsRouter, dict[str, list[Any]]]:
    page = context.new_page()
    diagnostics = attach_diagnostics(page)
    router = TtsRouter()
    page.add_init_script(FAKE_BROWSER_RUNTIME)
    page.route(
        "**/static/vi/vi-chat.js*",
        lambda route: route.fulfill(status=200, content_type="application/javascript", body=LOCAL_JAVASCRIPT.read_text(encoding="utf-8")),
    )
    page.route("**/api/tts/vi/stream", router)
    page.route("**/api/tts/vi", router)
    separator = "&" if "?" in URL else "?"
    page.goto(f"{URL}{separator}early_tts={'1' if early_tts else '0'}", wait_until="networkidle", timeout=30_000)
    page.wait_for_function("() => !!window.__omniglassViAssistant?.beginEarlySpeech")
    page.evaluate(
        """async () => {
          const assistant = window.__omniglassViAssistant;
          assistant.runEpoch = 1; assistant.turnEpoch = 0;
          assistant.captureFrame = () => 'ZmFrZS1qcGVn';
          assistant.audioContext = new AudioContext({latencyHint:'interactive'});
          await assistant.audioContext.resume();
          assistant.setState('listening', 'controlled-ready');
        }"""
    )
    return page, router, diagnostics


def start_turn(page: Any, final_id: str, transcript: str, events: list[dict[str, Any]]) -> None:
    page.evaluate("events => { window.__chatScripts.push({events}); }", events)
    page.evaluate(
        """([finalId, transcript]) => {
          const assistant = window.__omniglassViAssistant;
          window.__activeTurn = assistant.consumeFinal({
            type:'asr.final', final_id:finalId, immutable:true, text:transcript,
            routing:{vlm_eligible:true},
          }, assistant.runEpoch);
        }""",
        [final_id, transcript],
    )


def wait_turn(page: Any, timeout: int = 15_000) -> None:
    page.wait_for_function("() => window.__omniglassViAssistant.state === 'listening'", timeout=timeout)
    page.evaluate("() => window.__activeTurn")


def ordering_and_fallback(context: Any) -> dict[str, Any]:
    page, router, diagnostics = open_page(context)
    start_turn(page, "order-1", "Mô tả phía trước", [
        {"after": 20, "data": {"type": "response.output.delta", "kind": "text", "text": "Phía trước có một chiếc ghế. "}},
        {"after": 45, "data": {"type": "response.output.delta", "kind": "text", "text": "Bên trái là cửa ra vào."}},
        {"after": 260, "data": {"type": "response.done"}},
    ])
    wait_turn(page)
    first = page.evaluate(
        """() => ({fetches:window.__ttsFetches.slice(), done:window.__chatDoneTimes[0],
          state:window.__omniglassViAssistant.state,
          sources:window.__omniglassViAssistant.ttsSources.size,
          abortNull:window.__omniglassViAssistant.ttsAbort===null,
          history:window.__omniglassViAssistant.history.slice(),
          assistantMessages:Array.from(document.querySelectorAll('.message.assistant')).map(x=>x.textContent),
        })"""
    )
    first_stream = [item for item in first["fetches"] if item["url"].endswith("/stream")]
    first_checks = {
        "first_tts_before_response_done": first_stream[0]["at"] < first["done"],
        "sentence_requests_exact_once": [item["text"] for item in first_stream] == [
            "Phía trước có một chiếc ghế.", "Bên trái là cửa ra vào."
        ],
        "fifo_waits_for_prior_playback": first_stream[1]["at"] - first_stream[0]["at"] >= 500,
        "answer_committed_once": first["assistantMessages"] == ["Phía trước có một chiếc ghế. Bên trái là cửa ra vào."],
        "history_committed_once": len(first["history"]) == 2,
        "turn_drained_and_listening": first["state"] == "listening" and first["sources"] == 0 and first["abortNull"],
    }

    fetch_count = len(first["fetches"])
    start_turn(page, "fallback-1", "Thử fallback", [
        {"after": 15, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu đầu phát bình thường. "}},
        {"after": 30, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu hai cần giọng dự phòng"}},
        {"after": 80, "data": {"type": "response.done"}},
    ])
    wait_turn(page)
    second = page.evaluate(
        """count => ({fetches:window.__ttsFetches.slice(count), state:window.__omniglassViAssistant.state,
          assistantMessages:Array.from(document.querySelectorAll('.message.assistant')).map(x=>x.textContent),
          sources:window.__omniglassViAssistant.ttsSources.size,
          abortNull:window.__omniglassViAssistant.ttsAbort===null,
          audioNull:window.__omniglassViAssistant.ttsAudio===null,
        })""",
        fetch_count,
    )
    second_stream = [item["text"] for item in second["fetches"] if item["url"].endswith("/stream")]
    second_mms = [item["text"] for item in second["fetches"] if not item["url"].endswith("/stream")]
    second_checks = {
        "trailing_fragment_flushed_once": second_stream == ["Câu đầu phát bình thường.", "Câu hai cần giọng dự phòng"],
        "mms_only_failed_segment": second_mms == ["Câu hai cần giọng dự phòng"],
        "never_replays_full_answer": "Câu đầu phát bình thường. Câu hai cần giọng dự phòng" not in second_stream + second_mms,
        "fallback_answer_once": second["assistantMessages"][-1:] == ["Câu đầu phát bình thường. Câu hai cần giọng dự phòng"],
        "fallback_drained": second["state"] == "listening" and second["sources"] == 0 and second["abortNull"] and second["audioNull"],
    }
    screenshot = RESULT_DIR / "ordering_fallback.png"
    page.screenshot(path=str(screenshot), full_page=True)
    page.evaluate("() => window.__omniglassViAssistant.stop()")
    page.close()
    return {"passed": all(first_checks.values()) and all(second_checks.values()), "ordering": first_checks,
            "fallback": second_checks, "first": first, "second": second,
            "network_requests": router.requests, "diagnostics": diagnostics, "screenshot": str(screenshot)}


def feature_flag_off(context: Any) -> dict[str, Any]:
    page, router, diagnostics = open_page(context, early_tts=False)
    start_turn(page, "flag-off", "Kiểm tra rollback", [
        {"after": 15, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu thứ nhất. "}},
        {"after": 35, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu thứ hai."}},
        {"after": 220, "data": {"type": "response.done"}},
    ])
    wait_turn(page)
    result = page.evaluate(
        """() => ({fetches:window.__ttsFetches.slice(), done:window.__chatDoneTimes[0],
          state:window.__omniglassViAssistant.state,
          answers:Array.from(document.querySelectorAll('.message.assistant')).map(x=>x.textContent)})"""
    )
    stream = [item for item in result["fetches"] if item["url"].endswith("/stream")]
    checks = {
        "single_full_answer_request": [item["text"] for item in stream] == ["Câu thứ nhất. Câu thứ hai."],
        "request_after_response_done": stream[0]["at"] >= result["done"],
        "answer_once_and_listening": result["answers"] == ["Câu thứ nhất. Câu thứ hai."] and result["state"] == "listening",
    }
    page.evaluate("() => window.__omniglassViAssistant.stop()")
    page.close()
    return {"passed": all(checks.values()), "checks": checks, "result": result,
            "network_requests": router.requests, "diagnostics": diagnostics}


def cancellation(context: Any, action: str) -> dict[str, Any]:
    page, router, diagnostics = open_page(context)
    start_turn(page, f"{action}-old", "Câu hỏi cũ", [
        {"after": 10, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu cũ thứ nhất. "}},
        {"after": 25, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu cũ thứ hai. "}},
        {"after": 900, "data": {"type": "response.done"}},
    ])
    page.wait_for_function("() => window.__ttsFetches.length === 1 && window.__omniglassViAssistant.state === 'speaking'", timeout=10_000)
    before = page.evaluate("() => ({run:window.__omniglassViAssistant.runEpoch, turn:window.__omniglassViAssistant.turnEpoch})")
    if action == "barge":
        page.evaluate("() => window.__omniglassViAssistant.bargeIn()")
    else:
        page.evaluate("() => window.__omniglassViAssistant.stop()")
    page.wait_for_timeout(1_150)
    after = page.evaluate(
        """() => { const a=window.__omniglassViAssistant; return {
          run:a.runEpoch, turn:a.turnEpoch, state:a.state, fetches:window.__ttsFetches.slice(),
          sources:a.ttsSources.size, abortNull:a.ttsAbort===null, earlyNull:a.earlySpeech===null,
          audioNull:a.ttsAudio===null, chatNull:a.chatSocket===null,
          assistantMessages:Array.from(document.querySelectorAll('.message.assistant')).map(x=>x.textContent),
          history:a.history.slice(),
        }}"""
    )
    checks = {
        "epoch_invalidated": after["turn"] == before["turn"] + 1 and (action == "barge" or after["run"] == before["run"] + 1),
        "active_and_queue_cancelled": [item["text"] for item in after["fetches"]] == ["Câu cũ thứ nhất."],
        "speech_resources_cancelled": after["sources"] == 0 and after["abortNull"] and after["earlyNull"] and after["audioNull"] and after["chatNull"],
        "late_old_response_not_committed": not after["assistantMessages"] and not after["history"],
        "final_state": after["state"] == ("listening" if action == "barge" else "stopped"),
    }
    if action == "barge":
        start_turn(page, "barge-new", "Câu hỏi mới", [
            {"after": 10, "data": {"type": "response.output.delta", "kind": "text", "text": "Câu mới hợp lệ. "}},
            {"after": 30, "data": {"type": "response.done"}},
        ])
        wait_turn(page)
        resumed = page.evaluate(
            """() => ({state:window.__omniglassViAssistant.state, fetches:window.__ttsFetches.map(x=>x.text),
              answers:Array.from(document.querySelectorAll('.message.assistant')).map(x=>x.textContent)})"""
        )
        checks["next_turn_is_heard"] = resumed["state"] == "listening" and resumed["fetches"] == ["Câu cũ thứ nhất.", "Câu mới hợp lệ."] and resumed["answers"] == ["Câu mới hợp lệ."]
    screenshot = RESULT_DIR / f"{action}_cancel.png"
    page.screenshot(path=str(screenshot), full_page=True)
    page.close()
    return {"passed": all(checks.values()), "checks": checks, "before": before, "after": after,
            "network_requests": router.requests, "diagnostics": diagnostics, "screenshot": str(screenshot)}


def clean_diagnostics(diagnostics: dict[str, list[Any]]) -> dict[str, list[Any]]:
    diagnostics = {key: list(value) for key, value in diagnostics.items()}
    diagnostics["http"] = [item for item in diagnostics["http"] if not item["url"].endswith("/favicon.ico")]
    diagnostics["console"] = [item for item in diagnostics["console"] if "favicon.ico" not in item["text"] and "404" not in item["text"]]
    return diagnostics


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, executable_path=str(EDGE),
            args=["--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling"],
        )
        context = browser.new_context(ignore_https_errors=True, viewport={"width": 1440, "height": 1000})
        report = {
            "url_shell": URL,
            "asset": str(LOCAL_JAVASCRIPT),
            "production_mutated": False,
            "ordering_and_fallback": ordering_and_fallback(context),
            "feature_flag_off": feature_flag_off(context),
            "barge_in": cancellation(context, "barge"),
            "stop": cancellation(context, "stop"),
        }
        for name in ("ordering_and_fallback", "feature_flag_off", "barge_in", "stop"):
            report[name]["diagnostics"] = clean_diagnostics(report[name]["diagnostics"])
        report["passed"] = all(
            report[name]["passed"] and not any(report[name]["diagnostics"].values())
            for name in ("ordering_and_fallback", "feature_flag_off", "barge_in", "stop")
        )
        output = RESULT_DIR / "report.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        browser.close()
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
