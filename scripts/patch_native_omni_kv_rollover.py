#!/usr/bin/env python3
"""Add bounded automatic session rollover when MiniCPM-o fills its KV cache."""

from __future__ import annotations

import argparse
from pathlib import Path


SESSION_MARKER = "OPENGLASS_OMNI_KV_ROLLOVER_SESSION_V1"
APP_MARKER = "OPENGLASS_OMNI_KV_ROLLOVER_APP_V1"


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_realtime_session(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if SESSION_MARKER in source:
        return False
    source = _replace_once(
        source,
        """            getStopOnSlidingWindow: config.getStopOnSlidingWindow || (() => false),
            outputSampleRate: config.outputSampleRate || 24000,""",
        """            getStopOnSlidingWindow: config.getStopOnSlidingWindow || (() => false),
            // OPENGLASS_OMNI_KV_ROLLOVER_SESSION_V1
            onKvLimit: config.onKvLimit || null,
            outputSampleRate: config.outputSampleRate || 24000,""",
        label="session rollover config",
    )
    latch = """        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};"""
    latch_patched = """        this._lastKvCacheLength = 0;
        this._kvLimitTriggered = false;
        this._lastFrameMetrics = {};"""
    latch_count = source.count(latch)
    if latch_count not in {1, 2}:
        raise RuntimeError(f"session rollover latch: expected one or two anchors, found {latch_count}")
    source = source.replace(latch, latch_patched)
    source = _replace_once(
        source,
        """            if (curKv >= maxKv) {
                this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                setTimeout(() => this.stop(), 0);""",
        """            if (curKv >= maxKv && !this._kvLimitTriggered) {
                this._kvLimitTriggered = true;
                const handled = this.config.onKvLimit?.({ current: curKv, max: maxKv }) === true;
                if (!handled) {
                    this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                    setTimeout(() => this.stop(), 0);
                }""",
        label="session rollover trigger",
    )
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def patch_omni_app(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if APP_MARKER in source:
        return False
    source = _replace_once(
        source,
        """let cameraPreview = null;
let cameraPreviewPromise = null; // OPENGLASS_OMNI_MEDIA_RECOVERY_V1""",
        """let cameraPreview = null;
let cameraPreviewPromise = null; // OPENGLASS_OMNI_MEDIA_RECOVERY_V1

// OPENGLASS_OMNI_KV_ROLLOVER_APP_V1
// MiniCPM-o is trained for an 8192-token context. Keep a long-running glasses
// experience by replacing a full session instead of exceeding that limit.
let _kvRolloverActive = false;
let _kvRolloverTimer = null;
let _kvRolloverGeneration = 0;
const KV_ROLLOVER_MAX_ATTEMPTS = 3;

function _cancelKvRollover() {
    _kvRolloverGeneration += 1;
    _kvRolloverActive = false;
    if (_kvRolloverTimer) clearTimeout(_kvRolloverTimer);
    _kvRolloverTimer = null;
}

function _scheduleKvRollover(info) {
    if (_kvRolloverActive || currentMode !== 'live') return true;
    _kvRolloverActive = true;
    const generation = ++_kvRolloverGeneration;
    addSystemEntry(`↻ KV cache ${info.current.toLocaleString()}/${info.max.toLocaleString()}. Continuing in a fresh session…`);
    metricsPanel.update({ type: 'state', sessionState: 'Rolling over...' });

    const oldSession = session;
    if (oldSession) oldSession.stop();

    const attempt = async (number) => {
        if (!_kvRolloverActive || generation !== _kvRolloverGeneration) return;
        const ok = await startSession({ preserveConversation: true });
        if (generation !== _kvRolloverGeneration) return;
        if (ok) {
            _kvRolloverActive = false;
            addSystemEntry('↻ Continued automatically in a fresh session.');
            return;
        }
        if (number >= KV_ROLLOVER_MAX_ATTEMPTS) {
            _kvRolloverActive = false;
            addSystemEntry('⚠ Automatic continuation failed. Press Start to retry.');
            return;
        }
        const delay = 500 * number;
        addSystemEntry(`↻ Worker is still releasing the old session; retrying (${number + 1}/${KV_ROLLOVER_MAX_ATTEMPTS})…`);
        _kvRolloverTimer = setTimeout(() => void attempt(number + 1), delay);
    };

    // The worker marks itself idle shortly after the old WebSocket closes.
    _kvRolloverTimer = setTimeout(() => void attempt(1), 500);
    return true;
}""",
        label="app rollover state",
    )
    source = _replace_once(
        source,
        "async function startSession() {\n    if (session) return;",
        """async function startSession(options = {}) {
    if (session) return false;
    const preserveConversation = options?.preserveConversation === true;""",
        label="startSession return contract",
    )
    source = _replace_once(
        source,
        """    clearConversation();
    metricsPanel.update({ type: 'state', sessionState: 'Starting...' });""",
        """    if (!preserveConversation) clearConversation();
    metricsPanel.update({ type: 'state', sessionState: preserveConversation ? 'Continuing...' : 'Starting...' });""",
        label="preserve conversation",
    )
    source = _replace_once(
        source,
        """        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        outputSampleRate: SAMPLE_RATE_OUT,""",
        """        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        onKvLimit: _scheduleKvRollover,
        outputSampleRate: SAMPLE_RATE_OUT,""",
        label="wire rollover callback",
    )
    source = _replace_once(
        source,
        """        if (_saveShareUI && session && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
    } catch (err) {""",
        """        if (_saveShareUI && session && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
        return true;
    } catch (err) {""",
        label="start success return",
    )
    source = _replace_once(
        source,
        """        if (currentMode === 'live') startCameraPreview();
    }
}

function pauseSession()""",
        """        if (currentMode === 'live') startCameraPreview();
        return false;
    }
}

function pauseSession()""",
        label="start failure return",
    )
    source = _replace_once(
        source,
        """function stopSession() {
    _stopVietnameseTts();""",
        """function stopSession() {
    _cancelKvRollover();
    _stopVietnameseTts();""",
        label="user stop cancels rollover",
    )
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def patch_html(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    old = 'src="/static/omni/omni-app.js?v=vi-tts-v3-stream-media-v1"'
    new = 'src="/static/omni/omni-app.js?v=vi-tts-v3-stream-media-kv-v1"'
    if new in source:
        return False
    source = _replace_once(source, old, new, label="omni rollover cache bust")
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.demo_root.resolve()
    changed = {
        "session_changed": patch_realtime_session(root / "static/duplex/lib/realtime-session.js"),
        "omni_app_changed": patch_omni_app(root / "static/omni/omni-app.js"),
        "html_changed": patch_html(root / "static/omni/omni.html"),
    }
    for key, value in changed.items():
        print(f"{key}={str(value).lower()}")


if __name__ == "__main__":
    main()
