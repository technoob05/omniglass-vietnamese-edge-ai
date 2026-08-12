from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from scripts.patch_native_vietnamese_profile import GATEWAY_MARKER, install_assets, patch_gateway


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "native-overrides" / "vi-profile"
GATEWAY = '''import asyncio
import os
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
app = FastAPI()
_BASE_DIR = "."

@app.get("/api/presets")
async def get_presets():
    return []
'''


def test_profile_gateway_patch_is_valid_and_idempotent(tmp_path: Path) -> None:
    gateway = tmp_path / "gateway.py"
    gateway.write_text(GATEWAY, encoding="utf-8")
    assert patch_gateway(gateway)
    assert not patch_gateway(gateway)
    source = gateway.read_text(encoding="utf-8")
    ast.parse(source)
    assert GATEWAY_MARKER in source
    assert '@app.get("/vi")' in source
    assert '@app.websocket("/v1/asr/vi")' in source
    assert 'OPENGLASS_VI_ASR_WS_URL' in source
    assert 'ws://127.0.0.1:18783/v1/asr' in source
    assert 'OPENGLASS_VI_ASR_HTTP_URL' in source
    assert 'asyncio.FIRST_COMPLETED' in source


def test_profile_assets_install_and_are_idempotent(tmp_path: Path) -> None:
    assert install_assets(tmp_path, ASSETS)
    assert not install_assets(tmp_path, ASSETS)
    assert (tmp_path / "static/vi/vi-chat.html").read_bytes() == (ASSETS / "vi-chat.html").read_bytes()
    assert (tmp_path / "static/vi/vi-chat.js").read_bytes() == (ASSETS / "vi-chat.js").read_bytes()


def test_browser_profile_contract_and_javascript_syntax() -> None:
    javascript = ASSETS / "vi-chat.js"
    source = javascript.read_text(encoding="utf-8")
    checked = subprocess.run(
        ["node", "--check", str(javascript)], check=False, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr

    # Only immutable finals are routed once.
    assert "if (!event.immutable || !finalId || this.consumedFinalIds.has(finalId)) return;" in source
    assert "this.consumedFinalIds.add(finalId);" in source
    assert "data.type === 'asr.partial'" in source

    # Frame freshness and bounded rolling text memory.
    created = source.index("type === 'session.created'")
    capture = source.index("const frame = this.captureFrame();", created)
    send = source.index("type:'input.append'", capture)
    assert created < capture < send
    assert "this.history.slice(-MAX_HISTORY_MESSAGES)" in source
    assert "MAX_HISTORY_MESSAGES = 10" in source

    # At most one retry, and never after input has been sent.
    assert "attempt < 2" in source
    assert "error.preInput" in source
    assert "if (attempt === 0 && retryable403" in source
    assert "error.preInput = !inputSent" in source

    # Epoch cancellation/barge-in plus true streaming/fallback TTS.
    assert "run !== this.runEpoch || turn !== this.turnEpoch" in source
    assert "this.turnEpoch += 1;" in source
    assert "bargeIn()" in source
    assert "resetCaptureBuffer()" in source
    assert "if (this.ttsAbort === controller) this.ttsAbort = null" in source
    assert "response.body.getReader()" in source
    assert "fetch('/api/tts/vi/stream'" in source
    assert "await this.speakMms(text, signal)" in source
    assert "echoCancellation:true" in source

    # /vi-only early speech: sentence buffering feeds a guarded FIFO chain
    # directly from text deltas, while full-answer speech remains flaggable.
    assert "const VI_EARLY_TTS_ENABLED" in source
    assert "get('early_tts') !== '0'" in source
    assert "class SentenceBuffer" in source
    assert "if (delta && onDelta) onDelta(delta);" in source
    assert "this.queueEarlySpeech(earlySpeech, delta)" in source
    assert "speech.queue = task.catch" in source
    assert "speech.run === this.runEpoch && speech.turn === this.turnEpoch" in source
    assert "this.queueEarlySpeech(speech, '', true)" in source
    assert "await this.speakText(segment" in source


def test_sentence_buffer_consumes_incremental_deltas_once(tmp_path: Path) -> None:
    source = (ASSETS / "vi-chat.js").read_text(encoding="utf-8")
    start = source.index("class SentenceBuffer")
    end = source.index("class VietnameseAssistant", start)
    harness = source[start:end] + r'''
const assert = require('node:assert/strict');
const first = new SentenceBuffer();
assert.deepEqual(first.push('Phía trước có '), []);
assert.deepEqual(first.push('một chiếc ghế.'), []);
assert.deepEqual(first.push(' Bên trái là cửa ra vào.'), ['Phía trước có một chiếc ghế.']);
assert.deepEqual(first.push('', true), ['Bên trái là cửa ra vào.']);

const quote = new SentenceBuffer();
assert.deepEqual(quote.push('Xin chào.'), []);
assert.deepEqual(quote.push('” Câu sau'), ['Xin chào.”']);
assert.deepEqual(quote.push('', true), ['Câu sau']);

const decimal = new SentenceBuffer();
assert.deepEqual(decimal.push('Khoảng cách 3.'), []);
assert.deepEqual(decimal.push('5 mét. '), ['Khoảng cách 3.5 mét.']);
assert.deepEqual(decimal.push('   ', true), []);
'''
    javascript = tmp_path / "sentence-buffer-test.js"
    javascript.write_text(harness, encoding="utf-8")
    checked = subprocess.run(
        ["node", str(javascript)], check=False, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr


def test_profile_is_separate_from_native_omni_assets() -> None:
    html = (ASSETS / "vi-chat.html").read_text(encoding="utf-8")
    source = (ASSETS / "vi-chat.js").read_text(encoding="utf-8")
    assert '/static/vi/vi-chat.js?v=4-early-tts' in html
    assert 'value="Trúc Ly" selected' in html
    assert 'value="Ngọc Trân"' in html
    assert "voice:el.voice?.value || 'Trúc Ly'" in source
    assert "OmniGlass Việt" in html
    assert "omni-app.js" not in html
