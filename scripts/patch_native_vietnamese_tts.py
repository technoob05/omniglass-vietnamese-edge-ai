#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


GATEWAY_MARKER = "# OPENGLASS_VI_TTS_PROXY_V1"
GATEWAY_STREAM_MARKER = "# OPENGLASS_VI_TTS_STREAM_PROXY_V1"
GATEWAY_HTTP_CLIENT_MARKER = "# OPENGLASS_VI_HTTP_CLIENT_V1"
GATEWAY_KEEPALIVE_MARKER = "# OPENGLASS_VI_HTTP_KEEPALIVE_V1"
FRONTEND_MARKER = "// OPENGLASS_VI_TTS_V1"
FRONTEND_STREAM_MARKER = "// OPENGLASS_VI_TTS_STREAM_V1"
FRONTEND_ECHO_GUARD_MARKER = "// OPENGLASS_VI_TTS_ECHO_GUARD_V1"
FRONTEND_AUDIO_CONFIG_MARKER = "// OPENGLASS_VI_TTS_AUDIO_CONFIG_V2"
BACKEND_DUPLEX_AUDIO_MARKER = "# OPENGLASS_VI_TTS_DUPLEX_AUDIO_V1"
BACKEND_DUPLEX_AUDIO_OUTPUT_MARKER = "# OPENGLASS_VI_TTS_DUPLEX_OUTPUT_V1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def patch_gateway(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if GATEWAY_MARKER in text:
        return False
    anchor = '@app.get("/api/presets")\nasync def get_presets():'
    block = f'''{GATEWAY_MARKER}
@app.post("/api/tts/vi")
async def vietnamese_tts_proxy(payload: Dict[str, Any] = Body(...)):
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Không có nội dung để đọc")
    if len(text) > 600:
        text = text[:600]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                os.getenv("OPENGLASS_VI_TTS_URL", "http://127.0.0.1:18781/speak"),
                json={{"text": text}},
            )
        response.raise_for_status()
        return JSONResponse(response.json())
    except Exception as exc:
        logger.exception("Vietnamese TTS proxy failed")
        raise HTTPException(status_code=503, detail=f"Vietnamese TTS unavailable: {{exc}}") from exc


{anchor}'''
    text = replace_once(text, anchor, block, "gateway preset route")
    path.write_text(text, encoding="utf-8")
    return True


def patch_gateway_streaming(path: Path) -> bool:
    """Add a byte-preserving NDJSON proxy for the VieNeu stream service."""
    text = path.read_text(encoding="utf-8")
    if GATEWAY_STREAM_MARKER in text:
        return False
    anchor = '@app.get("/api/presets")\nasync def get_presets():'
    block = f'''{GATEWAY_STREAM_MARKER}
@app.post("/api/tts/vi/stream")
async def vietnamese_tts_stream_proxy(payload: Dict[str, Any] = Body(...)):
    from fastapi.responses import StreamingResponse as FastAPIStreamingResponse

    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=422, detail="Không có nội dung để đọc")
    if len(text) > 600:
        text = text[:600]

    async def proxy_body():
        try:
            timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    os.getenv(
                        "OPENGLASS_VIENEU_STREAM_URL",
                        "http://127.0.0.1:18782/stream",
                    ),
                    json={{"text": text}},
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk
        except Exception:
            logger.exception("Vietnamese VieNeu streaming proxy failed")
            yield b'{{"type":"error","code":"upstream_unavailable","message":"VieNeu stream unavailable","chunks_emitted":0}}\\n'

    return FastAPIStreamingResponse(
        proxy_body(),
        media_type="application/x-ndjson",
        headers={{
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        }},
    )


{anchor}'''
    text = replace_once(text, anchor, block, "gateway streaming route")
    path.write_text(text, encoding="utf-8")
    return True


def patch_gateway_http_client_reuse(path: Path) -> bool:
    """Reuse one gateway-owned HTTP pool for VieNeu and MMS requests.

    This is intentionally a separate upgrade patch so it can migrate an
    already-installed V1 gateway without duplicating either FastAPI route.
    """
    text = path.read_text(encoding="utf-8")
    if GATEWAY_HTTP_CLIENT_MARKER in text:
        return False
    if GATEWAY_MARKER not in text or GATEWAY_STREAM_MARKER not in text:
        raise RuntimeError("Apply both Vietnamese TTS gateway routes first")

    client_block = f'''{GATEWAY_HTTP_CLIENT_MARKER}
_openglass_vi_http_client = None
_openglass_vi_http_client_lock = asyncio.Lock()


async def _get_openglass_vi_http_client():
    global _openglass_vi_http_client
    if _openglass_vi_http_client is None or _openglass_vi_http_client.is_closed:
        async with _openglass_vi_http_client_lock:
            if _openglass_vi_http_client is None or _openglass_vi_http_client.is_closed:
                _openglass_vi_http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                        keepalive_expiry=60.0,
                    ),
                )
    return _openglass_vi_http_client


async def _close_openglass_vi_http_client():
    global _openglass_vi_http_client
    client = _openglass_vi_http_client
    _openglass_vi_http_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


app.router.add_event_handler("shutdown", _close_openglass_vi_http_client)


'''
    text = replace_once(text, GATEWAY_STREAM_MARKER, client_block + GATEWAY_STREAM_MARKER, "shared Vietnamese HTTP client")

    old_mms = '''        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                os.getenv("OPENGLASS_VI_TTS_URL", "http://127.0.0.1:18781/speak"),
                json={"text": text},
            )
        response.raise_for_status()'''
    new_mms = '''        client = await _get_openglass_vi_http_client()
        response = await client.post(
            os.getenv("OPENGLASS_VI_TTS_URL", "http://127.0.0.1:18781/speak"),
            json={"text": text},
            timeout=30.0,
        )
        response.raise_for_status()'''
    text = replace_once(text, old_mms, new_mms, "MMS shared HTTP client")

    old_stream = '''            timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    os.getenv(
                        "OPENGLASS_VIENEU_STREAM_URL",
                        "http://127.0.0.1:18782/stream",
                    ),
                    json={"text": text},
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            yield chunk'''
    new_stream = '''            timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
            client = await _get_openglass_vi_http_client()
            async with client.stream(
                "POST",
                os.getenv(
                    "OPENGLASS_VIENEU_STREAM_URL",
                    "http://127.0.0.1:18782/stream",
                ),
                json={"text": text},
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk'''
    text = replace_once(text, old_stream, new_stream, "VieNeu shared HTTP client")
    path.write_text(text, encoding="utf-8")
    return True


def patch_gateway_keepalive(path: Path) -> bool:
    """Keep public HTTPS connections alive across normal /vi turns."""
    text = path.read_text(encoding="utf-8")
    if GATEWAY_KEEPALIVE_MARKER in text:
        return False
    anchor = "            ws_max_size=128 * 1024 * 1024,\n"
    block = f'''{anchor}            timeout_keep_alive=int(
                os.getenv("OPENGLASS_HTTP_KEEPALIVE_SECONDS", "60")
            ),  {GATEWAY_KEEPALIVE_MARKER}
'''
    text = replace_once(text, anchor, block, "public Uvicorn keep-alive")
    path.write_text(text, encoding="utf-8")
    return True


def patch_frontend(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False
    if FRONTEND_MARKER in text:
        pass
    else:
        changed = True

        globals_anchor = "let sessionRecorder = null;"
        globals_block = f'''{globals_anchor}

{FRONTEND_MARKER}
let _viTtsAudio = null;
let _viTtsAbort = null;

function _isVietnameseCall() {{
    return _omniPreset && _omniPreset.getSelectedId() === 'vietnamese_call';
}}

function _stopVietnameseTts() {{
    if (_viTtsAbort) {{ _viTtsAbort.abort(); _viTtsAbort = null; }}
    if (_viTtsAudio) {{
        try {{ _viTtsAudio.pause(); _viTtsAudio.currentTime = 0; }} catch (_) {{}}
        _viTtsAudio = null;
    }}
}}

async function _speakVietnamese(text, textEl) {{
    const spoken = (text || '').trim();
    if (!spoken) return;
    _stopVietnameseTts();
    const controller = new AbortController();
    _viTtsAbort = controller;
    try {{
        addSystemEntry('Đang tạo giọng tiếng Việt…');
        const response = await fetch('/api/tts/vi', {{
            method: 'POST',
            headers: {{ 'content-type': 'application/json' }},
            body: JSON.stringify({{ text: spoken }}),
            signal: controller.signal,
        }});
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        const result = await response.json();
        if (textEl && result.text) textEl.textContent = result.text;
        const audio = new Audio(`data:audio/wav;base64,${{result.audio_wav_base64}}`);
        _viTtsAudio = audio;
        audio.onended = () => {{ if (_viTtsAudio === audio) _viTtsAudio = null; }};
        audio.onerror = () => {{
            if (_viTtsAudio === audio) _viTtsAudio = null;
            addSystemEntry('Không phát được giọng Việt; câu trả lời chữ vẫn được giữ lại.');
        }};
        await audio.play();
    }} catch (error) {{
        if (error.name !== 'AbortError') addSystemEntry(`Giọng Việt lỗi: ${{error.message}}`);
    }} finally {{
        if (_viTtsAbort === controller) _viTtsAbort = null;
    }}
}}'''
        text = replace_once(text, globals_anchor, globals_block, "frontend globals")

        start_anchor = """session.onSpeakStart = (text) => {
        const handle = addSpeakEntry(text);"""
        start_block = """session.onSpeakStart = (text) => {
        if (_isVietnameseCall()) _stopVietnameseTts();
        const handle = addSpeakEntry(text);"""
        text = replace_once(text, start_anchor, start_block, "speak start")

        end_anchor = """session.onSpeakEnd = () => {
        finishFsSpeak();
        if (sessionRecorder) sessionRecorder.finalizeSubtitle();
    };"""
        end_block = """session.onSpeakEnd = () => {
        const finalText = session ? session.currentSpeakText : '';
        const finalTextEl = session ? session._speakHandle : null;
        finishFsSpeak();
        if (sessionRecorder) sessionRecorder.finalizeSubtitle();
        if (_isVietnameseCall()) void _speakVietnamese(finalText, finalTextEl);
    };"""
        text = replace_once(text, end_anchor, end_block, "speak end")

        payload_anchor = """use_tts: document.getElementById('ttsEnabled').checked,"""
        payload_block = """use_tts: document.getElementById('ttsEnabled').checked && !_isVietnameseCall(),"""
        text = replace_once(text, payload_anchor, payload_block, "prepare payload")

        stop_anchor = """function stopSession() {
    if (!session) return;"""
        stop_block = """function stopSession() {
    _stopVietnameseTts();
    if (!session) return;"""
        text = replace_once(text, stop_anchor, stop_block, "stop session")

        force_anchor = "function toggleForceListen() { if (session) session.toggleForceListen(); }"
        force_block = "function toggleForceListen() { _stopVietnameseTts(); if (session) session.toggleForceListen(); }"
        text = replace_once(text, force_anchor, force_block, "force listen")

    if FRONTEND_AUDIO_CONFIG_MARKER not in text:
        changed = True
        config_anchor = """config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0 },"""
        config_block = f"""config: {{
            length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0,
            generate_audio: !_isVietnameseCall(),
        }}, {FRONTEND_AUDIO_CONFIG_MARKER}"""
        text = replace_once(text, config_anchor, config_block, "duplex audio config")

        player_anchor = """    });
    session.onMetrics = (data) => metricsPanel.update(data);"""
        player_block = """    });
    const nativePlayChunk = session.audioPlayer.playChunk.bind(session.audioPlayer);
    session.audioPlayer.playChunk = (samples, recvTime) => {
        if (_isVietnameseCall()) return;
        return nativePlayChunk(samples, recvTime);
    };
    session.onMetrics = (data) => metricsPanel.update(data);"""
        text = replace_once(text, player_anchor, player_block, "native audio guard")

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def add_frontend_echo_guard(text: str) -> str:
    if FRONTEND_ECHO_GUARD_MARKER in text:
        return text
    chunk_anchor = """                media.onChunk = (chunk) => {
                    const msg = { type: 'audio_chunk', audio_base64: arrayBufferToBase64(chunk.audio.buffer) };"""
    chunk_block = f"""                media.onChunk = (chunk) => {{
                    {FRONTEND_ECHO_GUARD_MARKER}
                    const suppressMic = _isVietnameseCall() && _isVietnameseTtsSpeaking();
                    const outboundAudio = suppressMic
                        ? new Float32Array(chunk.audio.length)
                        : chunk.audio;
                    if (suppressMic) _viTtsAudit('mic-suppressed', {{ samples: outboundAudio.length }});
                    const msg = {{ type: 'audio_chunk', audio_base64: arrayBufferToBase64(outboundAudio.buffer) }};"""
    return replace_once(text, chunk_anchor, chunk_block, "Vietnamese TTS echo guard")


def patch_frontend_streaming(path: Path) -> bool:
    """Upgrade the Vietnamese frontend path to streaming PCM with MMS fallback."""
    text = path.read_text(encoding="utf-8")
    if FRONTEND_STREAM_MARKER in text:
        guarded = add_frontend_echo_guard(text)
        if guarded == text:
            return False
        path.write_text(guarded, encoding="utf-8")
        return True
    if FRONTEND_MARKER not in text:
        raise RuntimeError("Apply the base Vietnamese frontend patch first")

    start_marker = "let _viTtsAudio = null;"
    end_marker = "let lastRecordingBlob = null;"
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise RuntimeError("Could not identify the Vietnamese TTS frontend block")
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = f'''{FRONTEND_STREAM_MARKER}
let _viTtsAudio = null;
let _viTtsAbort = null;
let _viTtsContext = null;
let _viTtsNextStartTime = 0;
const _viTtsSources = new Set();
let _viTtsSpeaking = false;
let _viTtsStreamComplete = false;
let _viTtsResumeTimer = null;

function _isVietnameseCall() {{
    return _omniPreset && _omniPreset.getSelectedId() === 'vietnamese_call';
}}

function _viTtsAudit(type, detail = {{}}) {{
    window.dispatchEvent(new CustomEvent('openglass:vi-tts', {{
        detail: {{ type, at: performance.now(), ...detail }},
    }}));
}}

function _isVietnameseTtsSpeaking() {{
    return _viTtsSpeaking;
}}

function _beginVietnamesePlayback() {{
    if (_viTtsResumeTimer) {{ clearTimeout(_viTtsResumeTimer); _viTtsResumeTimer = null; }}
    if (!_viTtsSpeaking) {{
        _viTtsSpeaking = true;
        _viTtsAudit('speech-started');
    }}
}}

function _maybeFinishVietnamesePlayback() {{
    if (!_viTtsSpeaking || !_viTtsStreamComplete || _viTtsSources.size > 0 || _viTtsAudio) return;
    if (_viTtsResumeTimer) clearTimeout(_viTtsResumeTimer);
    _viTtsResumeTimer = setTimeout(() => {{
        _viTtsResumeTimer = null;
        if (_viTtsStreamComplete && _viTtsSources.size === 0 && !_viTtsAudio) {{
            _viTtsSpeaking = false;
            _viTtsAudit('speech-ended', {{ resumeDelayMs: 200 }});
        }}
    }}, 200);
}}

function _stopVietnameseTts() {{
    if (_viTtsResumeTimer) {{ clearTimeout(_viTtsResumeTimer); _viTtsResumeTimer = null; }}
    if (_viTtsAbort) {{ _viTtsAbort.abort(); _viTtsAbort = null; }}
    if (_viTtsAudio) {{
        try {{ _viTtsAudio.pause(); _viTtsAudio.currentTime = 0; }} catch (_) {{}}
        _viTtsAudio = null;
    }}
    for (const source of _viTtsSources) {{
        try {{ source.stop(); }} catch (_) {{}}
    }}
    _viTtsSources.clear();
    _viTtsNextStartTime = 0;
    _viTtsSpeaking = false;
    _viTtsStreamComplete = false;
    _viTtsAudit('stopped');
}}

async function _getVietnameseAudioContext() {{
    if (!_viTtsContext || _viTtsContext.state === 'closed') {{
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        _viTtsContext = new AudioContextClass({{ latencyHint: 'interactive' }});
    }}
    if (_viTtsContext.state === 'suspended') await _viTtsContext.resume();
    return _viTtsContext;
}}

async function _scheduleVietnamesePcm(base64Pcm, sampleRate, sequence) {{
    const binary = atob(base64Pcm);
    const samples = new Float32Array(Math.floor(binary.length / 2));
    const view = new DataView(new ArrayBuffer(binary.length));
    for (let i = 0; i < binary.length; i++) view.setUint8(i, binary.charCodeAt(i));
    for (let i = 0; i < samples.length; i++) samples[i] = view.getInt16(i * 2, true) / 32768;

    const context = await _getVietnameseAudioContext();
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(_viTtsNextStartTime, context.currentTime + 0.035);
    _viTtsNextStartTime = startAt + buffer.duration;
    _viTtsSources.add(source);
    _beginVietnamesePlayback();
    source.onended = () => {{
        _viTtsSources.delete(source);
        _viTtsAudit('chunk-ended', {{ sequence }});
        _maybeFinishVietnamesePlayback();
    }};
    source.start(startAt);
    _viTtsAudit('chunk-scheduled', {{
        sequence, samples: samples.length, sampleRate, startAt, duration: buffer.duration,
    }});
}}

async function _streamVietnameseTts(spoken, textEl, controller) {{
    _viTtsStreamComplete = false;
    const response = await fetch('/api/tts/vi/stream', {{
        method: 'POST',
        headers: {{ 'content-type': 'application/json' }},
        body: JSON.stringify({{ text: spoken }}),
        signal: controller.signal,
    }});
    if (!response.ok || !response.body) throw new Error(`VieNeu HTTP ${{response.status}}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    let sampleRate = 48000;
    let audioChunks = 0;
    let doneSeen = false;

    const consumeLine = async (line) => {{
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'meta') {{
            sampleRate = Number(event.sample_rate) || sampleRate;
            if (textEl && event.text) textEl.textContent = event.text;
            _viTtsAudit('stream-meta', {{ sampleRate }});
        }} else if (event.type === 'audio') {{
            await _scheduleVietnamesePcm(event.pcm_s16le_base64, sampleRate, event.seq);
            audioChunks += 1;
        }} else if (event.type === 'done') {{
            doneSeen = true;
            _viTtsStreamComplete = true;
            _viTtsAudit('stream-done', event);
            _maybeFinishVietnamesePlayback();
        }} else if (event.type === 'error') {{
            const error = new Error(event.message || 'VieNeu stream failed');
            error.partialAudio = audioChunks > 0;
            throw error;
        }}
    }};

    while (true) {{
        const {{ value, done }} = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), {{ stream: !done }});
        const lines = pending.split('\\n');
        pending = lines.pop() || '';
        for (const line of lines) await consumeLine(line);
        if (done) break;
    }}
    if (pending.trim()) await consumeLine(pending);
    if (audioChunks === 0 || !doneSeen) {{
        const error = new Error('VieNeu stream ended without complete audio');
        error.partialAudio = audioChunks > 0;
        throw error;
    }}
    return audioChunks;
}}

async function _speakVietnameseMms(spoken, textEl, controller) {{
    const response = await fetch('/api/tts/vi', {{
        method: 'POST',
        headers: {{ 'content-type': 'application/json' }},
        body: JSON.stringify({{ text: spoken }}),
        signal: controller.signal,
    }});
    if (!response.ok) throw new Error(`MMS HTTP ${{response.status}}`);
    const result = await response.json();
    if (textEl && result.text) textEl.textContent = result.text;
    const audio = new Audio(`data:audio/wav;base64,${{result.audio_wav_base64}}`);
    _viTtsAudio = audio;
    _viTtsStreamComplete = true;
    _beginVietnamesePlayback();
    audio.onended = () => {{
        if (_viTtsAudio === audio) _viTtsAudio = null;
        _viTtsAudit('fallback-ended');
        _maybeFinishVietnamesePlayback();
    }};
    audio.onerror = () => {{
        if (_viTtsAudio === audio) _viTtsAudio = null;
        _viTtsAudit('fallback-error');
        _maybeFinishVietnamesePlayback();
        addSystemEntry('Không phát được giọng Việt; câu trả lời chữ vẫn được giữ lại.');
    }};
    await audio.play();
    _viTtsAudit('fallback-playing');
}}

async function _speakVietnamese(text, textEl) {{
    const spoken = (text || '').trim();
    if (!spoken) return;
    _stopVietnameseTts();
    const controller = new AbortController();
    _viTtsAbort = controller;
    try {{
        addSystemEntry('Đang phát giọng tiếng Việt trực tuyến…');
        await _streamVietnameseTts(spoken, textEl, controller);
    }} catch (streamError) {{
        if (streamError.name === 'AbortError') return;
        if (streamError.partialAudio) {{
            _viTtsStreamComplete = true;
            _maybeFinishVietnamesePlayback();
            addSystemEntry(`Luồng giọng Việt bị ngắt: ${{streamError.message}}`);
            return;
        }}
        _viTtsAudit('fallback-start', {{ reason: streamError.message }});
        addSystemEntry('VieNeu chưa sẵn sàng, chuyển sang giọng dự phòng…');
        try {{
            await _speakVietnameseMms(spoken, textEl, controller);
        }} catch (fallbackError) {{
            if (fallbackError.name !== 'AbortError') {{
                addSystemEntry(`Giọng Việt lỗi: ${{fallbackError.message}}`);
            }}
        }}
    }} finally {{
        if (_viTtsAbort === controller) _viTtsAbort = null;
    }}
}}
'''
    text = text[:start] + replacement + text[end:]

    text = add_frontend_echo_guard(text)
    path.write_text(text, encoding="utf-8")
    return True


def patch_html(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    old = '<script type="module" src="/static/omni/omni-app.js"></script>'
    new = '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v2"></script>'
    streaming = '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v3-stream"></script>'
    if new in text or streaming in text:
        return False
    text = replace_once(text, old, new, "omni app cache-bust")
    path.write_text(text, encoding="utf-8")
    return True


def patch_html_streaming(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    old = '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v2"></script>'
    new = '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v3-stream"></script>'
    if new in text:
        return False
    text = replace_once(text, old, new, "streaming omni app cache-bust")
    path.write_text(text, encoding="utf-8")
    return True


def patch_backend_duplex_audio(path: Path) -> bool:
    """Apply the per-session DuplexConfig.generate_audio flag to the model.

    MiniCPM-o's DuplexView accepts generate_audio in DuplexConfig, but upstream
    never copies it to DuplexCapability.generate_audio.  Consequently the
    full-duplex path always executes CosyVoice/token2wav and emits native audio.
    This assignment happens before prepare() resets the session TTS state, so a
    following English/Chinese session can safely turn native audio back on.
    """
    text = path.read_text(encoding="utf-8")
    changed = False
    if BACKEND_DUPLEX_AUDIO_MARKER not in text:
        anchor = """        # 调用透传方法\n        prepared = self._model.duplex_prepare("""
        block = f"""        {BACKEND_DUPLEX_AUDIO_MARKER}
        duplex_capability = getattr(self._model, "duplex", None)
        if duplex_capability is not None:
            duplex_capability.generate_audio = bool(self.config.generate_audio)
        logger.info(
            "Duplex session native audio enabled=%s",
            bool(self.config.generate_audio),
        )

        # 调用透传方法
        prepared = self._model.duplex_prepare("""
        text = replace_once(text, anchor, block, "backend duplex prepare")
        changed = True

    if BACKEND_DUPLEX_AUDIO_OUTPUT_MARKER not in text:
        anchor = """        audio_data = None
        if result.get("audio_waveform") is not None:"""
        block = f"""        audio_data = None
        {BACKEND_DUPLEX_AUDIO_OUTPUT_MARKER}
        if self.config.generate_audio and result.get("audio_waveform") is not None:"""
        text = replace_once(text, anchor, block, "backend duplex audio output")
        changed = True

    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_root", type=Path)
    args = parser.parse_args()
    gateway_changed = patch_gateway(args.demo_root / "gateway.py")
    gateway_stream_changed = patch_gateway_streaming(args.demo_root / "gateway.py")
    gateway_http_client_changed = patch_gateway_http_client_reuse(
        args.demo_root / "gateway.py"
    )
    gateway_keepalive_changed = patch_gateway_keepalive(args.demo_root / "gateway.py")
    frontend_changed = patch_frontend(args.demo_root / "static/omni/omni-app.js")
    frontend_stream_changed = patch_frontend_streaming(
        args.demo_root / "static/omni/omni-app.js"
    )
    html_changed = patch_html(args.demo_root / "static/omni/omni.html")
    html_stream_changed = patch_html_streaming(args.demo_root / "static/omni/omni.html")
    backend_duplex_audio_changed = patch_backend_duplex_audio(
        args.demo_root / "core/processors/unified.py"
    )
    print(
        f"gateway_changed={gateway_changed} "
        f"gateway_stream_changed={gateway_stream_changed} "
        f"gateway_http_client_changed={gateway_http_client_changed} "
        f"gateway_keepalive_changed={gateway_keepalive_changed} "
        f"frontend_changed={frontend_changed} "
        f"frontend_stream_changed={frontend_stream_changed} "
        f"html_changed={html_changed} html_stream_changed={html_stream_changed} "
        f"backend_duplex_audio_changed={backend_duplex_audio_changed}"
    )


if __name__ == "__main__":
    main()
