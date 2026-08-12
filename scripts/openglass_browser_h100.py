#!/usr/bin/env python3
"""OpenGlass-inspired browser adapter for a remote H100 visual agent.

The browser owns webcam, microphone and playback. This same-origin aiohttp
service accepts one JPEG per finalized utterance and proxies only that turn to
the persistent H100 VLM/TTS worker. It deliberately does not import or modify
the experimental upstream OpenGlass pywebview/ESP32 runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import logging
from pathlib import Path
import time
import uuid

from aiohttp import ClientSession, ClientTimeout, web


LOGGER = logging.getLogger("omniglass.browser_h100")
MAX_JPEG_BYTES = 3 * 1024 * 1024
MAX_QUESTION_CHARS = 500


def concise_answer(text: str, max_words: int = 18) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(" ,;:-.!?") + "."


def parse_args():
    parser = argparse.ArgumentParser(description="OpenGlass-inspired local browser + H100 adapter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7873)
    parser.add_argument("--agent-url", default="http://127.0.0.1:8780")
    return parser.parse_args()


async def _agent_post(app: web.Application, endpoint: str, payload: dict, timeout_seconds: float) -> dict:
    session: ClientSession = app["http_session"]
    url = f"{app['agent_url']}/{endpoint}"
    async with session.post(url, json=payload, timeout=ClientTimeout(total=timeout_seconds)) as response:
        body = await response.json(content_type=None)
        if response.status >= 400:
            raise RuntimeError(f"H100 {endpoint} HTTP {response.status}: {body}")
        return body


async def index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(request.app["index_path"])


async def health(request: web.Request) -> web.Response:
    started = time.perf_counter()
    try:
        session: ClientSession = request.app["http_session"]
        async with session.get(
            f"{request.app['agent_url']}/health", timeout=ClientTimeout(total=4)
        ) as response:
            worker = await response.json(content_type=None)
            response.raise_for_status()
        return web.json_response(
            {"ok": True, "adapter_ms": round((time.perf_counter() - started) * 1000), "worker": worker}
        )
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=503)


async def turn(request: web.Request) -> web.Response:
    if request.app["turn_lock"].locked():
        return web.json_response({"error": "Một lượt khác đang được xử lý."}, status=409)

    received_at = time.perf_counter()
    reader = await request.multipart()
    question = ""
    turn_id = ""
    jpeg = b""
    async for part in reader:
        if part.name == "question":
            question = (await part.text()).strip()
        elif part.name == "turn_id":
            turn_id = (await part.text()).strip()
        elif part.name == "frame":
            jpeg = await part.read(decode=False)

    if not question:
        return web.json_response({"error": "Câu hỏi đang trống."}, status=400)
    if len(question) > MAX_QUESTION_CHARS:
        return web.json_response({"error": "Câu hỏi quá dài."}, status=400)
    if not jpeg or len(jpeg) > MAX_JPEG_BYTES:
        return web.json_response({"error": "JPEG hiện tại bị thiếu hoặc quá lớn."}, status=400)
    if not turn_id:
        turn_id = uuid.uuid4().hex

    async with request.app["turn_lock"]:
        image_b64 = base64.b64encode(jpeg).decode("ascii")
        prompt = (
            "Trả lời đúng ảnh camera hiện tại bằng tiếng Việt tự nhiên. Chỉ một câu, tối đa 18 từ. "
            "Nếu ảnh không rõ thì nói không chắc chắn; không bịa vật hoặc khoảng cách. "
            f"Câu hỏi: {question}"
        )
        analyze_started = time.perf_counter()
        try:
            analyzed = await _agent_post(
                request.app,
                "analyze",
                {"image_jpeg_base64": image_b64, "prompt": prompt, "max_new_tokens": 48},
                timeout_seconds=20,
            )
            answer = concise_answer(str(analyzed["answer"]).strip())
        except Exception as exc:
            LOGGER.exception("turn analyze failed turn_id=%s", turn_id)
            return web.json_response({"turn_id": turn_id, "error": str(exc)}, status=502)
        analyze_ms = round((time.perf_counter() - analyze_started) * 1000)

        audio_b64 = None
        audio_seconds = None
        tts_ms = None
        warning = None
        tts_started = time.perf_counter()
        try:
            spoken = await _agent_post(request.app, "speak", {"text": answer}, timeout_seconds=10)
            audio_b64 = spoken.get("audio_wav_base64")
            audio_seconds = spoken.get("duration_seconds")
            tts_ms = round((time.perf_counter() - tts_started) * 1000)
        except Exception as exc:
            LOGGER.exception("turn TTS failed turn_id=%s", turn_id)
            warning = f"Không tạo được giọng nói: {exc}"

        total_ms = round((time.perf_counter() - received_at) * 1000)
        LOGGER.info(
            "turn complete turn_id=%s jpeg_bytes=%s analyze_ms=%s tts_ms=%s total_ms=%s chars=%s",
            turn_id,
            len(jpeg),
            analyze_ms,
            tts_ms,
            total_ms,
            len(answer),
        )
        return web.json_response(
            {
                "turn_id": turn_id,
                "answer_vi": answer,
                "audio_wav_base64": audio_b64,
                "audio_seconds": audio_seconds,
                "warning": warning,
                "timings_ms": {"analyze": analyze_ms, "tts": tts_ms, "total": total_ms},
            }
        )


async def on_startup(app: web.Application):
    app["http_session"] = ClientSession()


async def on_cleanup(app: web.Application):
    await app["http_session"].close()


def build_app(agent_url: str) -> web.Application:
    app = web.Application(client_max_size=MAX_JPEG_BYTES + 64 * 1024)
    app["agent_url"] = agent_url.rstrip("/")
    app["index_path"] = Path(__file__).resolve().parents[1] / "web" / "openglass_browser_h100.html"
    app["turn_lock"] = asyncio.Lock()
    app.router.add_get("/", index)
    app.router.add_get("/api/health", health)
    app.router.add_post("/api/turn", turn)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    web.run_app(build_app(args.agent_url), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
