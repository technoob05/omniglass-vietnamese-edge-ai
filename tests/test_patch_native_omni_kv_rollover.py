from __future__ import annotations

from pathlib import Path

from scripts.patch_native_omni_kv_rollover import (
    APP_MARKER,
    SESSION_MARKER,
    patch_html,
    patch_omni_app,
    patch_realtime_session,
)


def test_rollover_patch_is_idempotent(tmp_path: Path) -> None:
    session = tmp_path / "realtime-session.js"
    session.write_text(
        """class R {
constructor(config) {
this.config = {
            getStopOnSlidingWindow: config.getStopOnSlidingWindow || (() => false),
            outputSampleRate: config.outputSampleRate || 24000,
};
        this._lastKvCacheLength = 0;
        this._lastFrameMetrics = {};
}
x(curKv, maxKv) {
            if (curKv >= maxKv) {
                this.onSystemLog(`⚠ KV cache (${curKv.toLocaleString()}) reached limit. Auto-stopping.`);
                setTimeout(() => this.stop(), 0);
            }
}
}
""",
        encoding="utf-8",
    )
    app = tmp_path / "omni-app.js"
    app.write_text(
        """let cameraPreview = null;
let cameraPreviewPromise = null; // OPENGLASS_OMNI_MEDIA_RECOVERY_V1
async function startSession() {
    if (session) return;
    clearConversation();
    metricsPanel.update({ type: 'state', sessionState: 'Starting...' });
const x = {
        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        outputSampleRate: SAMPLE_RATE_OUT,
};
        if (_saveShareUI && session && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
    } catch (err) {
        if (currentMode === 'live') startCameraPreview();
    }
}

function pauseSession() {}
function stopSession() {
    _stopVietnameseTts();
}
""",
        encoding="utf-8",
    )
    html = tmp_path / "omni.html"
    html.write_text(
        '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v3-stream-media-v1"></script>',
        encoding="utf-8",
    )

    assert patch_realtime_session(session)
    assert patch_omni_app(app)
    assert patch_html(html)
    assert not patch_realtime_session(session)
    assert not patch_omni_app(app)
    assert not patch_html(html)
    assert SESSION_MARKER in session.read_text(encoding="utf-8")
    source = app.read_text(encoding="utf-8")
    assert APP_MARKER in source
    assert "onKvLimit: _scheduleKvRollover" in source
    assert "KV_ROLLOVER_MAX_ATTEMPTS = 3" in source
    assert "_cancelKvRollover();" in source
    assert "media-kv-v1" in html.read_text(encoding="utf-8")
