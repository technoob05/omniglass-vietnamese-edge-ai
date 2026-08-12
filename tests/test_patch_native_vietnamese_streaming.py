from __future__ import annotations

import asyncio
import ast
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import httpx

from scripts.patch_native_vietnamese_tts import (
    FRONTEND_STREAM_MARKER,
    GATEWAY_HTTP_CLIENT_MARKER,
    GATEWAY_KEEPALIVE_MARKER,
    GATEWAY_STREAM_MARKER,
    patch_frontend,
    patch_frontend_streaming,
    patch_gateway,
    patch_gateway_http_client_reuse,
    patch_gateway_keepalive,
    patch_gateway_streaming,
    patch_html,
    patch_html_streaming,
)


GATEWAY_FIXTURE = '''from typing import Any, Dict

@app.get("/api/presets")
async def get_presets():
    return []
'''


FRONTEND_FIXTURE = '''let sessionRecorder = null;
let lastRecordingBlob = null;

function setupSession() {
    session = new RealtimeSession('omni', {
    });
    session.onMetrics = (data) => metricsPanel.update(data);
    session.onSpeakStart = (text) => {
        const handle = addSpeakEntry(text);
        return handle;
    };
    session.onSpeakEnd = () => {
        finishFsSpeak();
        if (sessionRecorder) sessionRecorder.finalizeSubtitle();
    };
    const preparePayload = {
        config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0 },
        use_tts: document.getElementById('ttsEnabled').checked,
    };
                media.onChunk = (chunk) => {
                    const msg = { type: 'audio_chunk', audio_base64: arrayBufferToBase64(chunk.audio.buffer) };
                    session.sendChunk(msg);
                };
}

function stopSession() {
    if (!session) return;
}

function toggleForceListen() { if (session) session.toggleForceListen(); }
'''


def test_gateway_stream_patch_is_valid_and_idempotent(tmp_path: Path) -> None:
    gateway = tmp_path / "gateway.py"
    gateway.write_text(GATEWAY_FIXTURE, encoding="utf-8")
    assert patch_gateway(gateway)
    assert patch_gateway_streaming(gateway)
    assert patch_gateway_http_client_reuse(gateway)
    assert not patch_gateway_streaming(gateway)
    assert not patch_gateway_http_client_reuse(gateway)

    source = gateway.read_text(encoding="utf-8")
    ast.parse(source)
    assert GATEWAY_STREAM_MARKER in source
    assert GATEWAY_HTTP_CLIENT_MARKER in source
    assert 'OPENGLASS_VIENEU_STREAM_URL' in source
    assert 'http://127.0.0.1:18782/stream' in source
    assert 'application/x-ndjson' in source
    assert 'response.aiter_bytes()' in source
    assert source.count('httpx.AsyncClient(') == 1
    assert 'keepalive_expiry=60.0' in source
    assert 'app.router.add_event_handler("shutdown", _close_openglass_vi_http_client)' in source
    assert source.count('@app.post("/api/tts/vi")') == 1
    assert source.count('@app.post("/api/tts/vi/stream")') == 1


def test_gateway_keepalive_patch_targets_only_public_config(tmp_path: Path) -> None:
    gateway = tmp_path / "gateway.py"
    gateway.write_text(
        '''public_config = uvicorn.Config(
            app,
            host=args.host,
            port=port,
            ws_max_size=128 * 1024 * 1024,
            **ssl_kwargs,
        )
internal_config = uvicorn.Config(internal_app, port=args.internal_port)
''',
        encoding="utf-8",
    )
    assert patch_gateway_keepalive(gateway)
    assert not patch_gateway_keepalive(gateway)
    source = gateway.read_text(encoding="utf-8")
    ast.parse(source)
    assert GATEWAY_KEEPALIVE_MARKER in source
    assert source.count("timeout_keep_alive=") == 1
    assert 'OPENGLASS_HTTP_KEEPALIVE_SECONDS", "60"' in source
    assert "internal_config = uvicorn.Config(internal_app" in source


class _TimedNdjsonStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"type":"meta","sample_rate":48000}\n'
        await asyncio.sleep(0.04)
        yield b'{"type":"audio","seq":0,"pcm_s16le_base64":"AAA="}\n'
        await asyncio.sleep(0.08)
        yield b'{"type":"done","chunks":1}\n'


def _load_patched_gateway(path: Path):
    spec = importlib.util.spec_from_file_location("patched_gateway_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_stream_is_progressive_and_pool_is_reused(tmp_path: Path) -> None:
    gateway = tmp_path / "gateway.py"
    gateway.write_text(
        '''import asyncio
import logging
import os
from typing import Any, Dict
import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
app = FastAPI()

@app.get("/api/presets")
async def get_presets():
    return []
''',
        encoding="utf-8",
    )
    assert patch_gateway(gateway)
    assert patch_gateway_streaming(gateway)
    assert patch_gateway_http_client_reuse(gateway)
    module = _load_patched_gateway(gateway)

    async def exercise() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/stream"
            return httpx.Response(
                200,
                headers={"content-type": "application/x-ndjson"},
                stream=_TimedNdjsonStream(),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        module._openglass_vi_http_client = client
        assert await module._get_openglass_vi_http_client() is client
        assert await module._get_openglass_vi_http_client() is client

        response = await module.vietnamese_tts_stream_proxy({"text": "Xin chào"})
        started = time.perf_counter()
        observed: list[tuple[str, float]] = []
        async for chunk in response.body_iterator:
            for line in chunk.decode("utf-8").splitlines():
                event = json.loads(line)
                observed.append((event["type"], time.perf_counter() - started))

        assert [kind for kind, _ in observed] == ["meta", "audio", "done"]
        timings = {kind: elapsed for kind, elapsed in observed}
        assert timings["meta"] < 0.03
        assert 0.03 <= timings["audio"] < 0.10
        assert timings["done"] >= 0.10
        assert timings["audio"] < timings["done"]

        await module._close_openglass_vi_http_client()
        assert client.is_closed
        assert module._openglass_vi_http_client is None

    asyncio.run(exercise())


def test_frontend_stream_patch_schedules_pcm_and_falls_back(tmp_path: Path) -> None:
    frontend = tmp_path / "omni-app.js"
    frontend.write_text(FRONTEND_FIXTURE, encoding="utf-8")
    assert patch_frontend(frontend)
    assert patch_frontend_streaming(frontend)
    assert not patch_frontend_streaming(frontend)

    source = frontend.read_text(encoding="utf-8")
    assert FRONTEND_STREAM_MARKER in source
    assert "fetch('/api/tts/vi/stream'" in source
    assert "response.body.getReader()" in source
    assert "context.createBuffer(1, samples.length, sampleRate)" in source
    assert "source.start(startAt)" in source
    assert "_viTtsAudit('speech-ended', { resumeDelayMs: 200 })" in source
    assert "_isVietnameseTtsSpeaking()" in source
    assert "new Float32Array(chunk.audio.length)" in source
    assert "_viTtsAudit('mic-suppressed'" in source
    assert "fetch('/api/tts/vi'" in source
    assert "if (streamError.partialAudio)" in source
    assert "_stopVietnameseTts();" in source
    checked = subprocess.run(
        ["node", "--check", str(frontend)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr


def test_html_stream_cache_bust_is_idempotent(tmp_path: Path) -> None:
    html = tmp_path / "omni.html"
    html.write_text(
        '<script type="module" src="/static/omni/omni-app.js"></script>',
        encoding="utf-8",
    )
    assert patch_html(html)
    assert patch_html_streaming(html)
    assert not patch_html(html)
    assert not patch_html_streaming(html)
    assert "omni-app.js?v=vi-tts-v3-stream" in html.read_text(encoding="utf-8")


def test_gateway_lifecycle_exports_bounded_keepalive() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/start_native_minicpmo_h100.sh",
        "scripts/reload_native_gateway_h100.sh",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert 'HTTP_KEEPALIVE_SECONDS="${OPENGLASS_HTTP_KEEPALIVE_SECONDS:-60}"' in source
        assert 'HTTP_KEEPALIVE_SECONDS < 5 || HTTP_KEEPALIVE_SECONDS > 600' in source
        assert 'export OPENGLASS_HTTP_KEEPALIVE_SECONDS="${HTTP_KEEPALIVE_SECONDS}"' in source
