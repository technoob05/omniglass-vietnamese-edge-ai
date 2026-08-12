from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.patch_native_omni_media_recovery import (
    AUDIO_MARKER,
    OMNI_MARKER,
    patch_audio_selector,
    patch_html,
    patch_omni_app,
)


def test_patch_is_idempotent_and_javascript_is_valid(tmp_path: Path) -> None:
    omni = tmp_path / "omni-app.js"
    omni.write_text(
        """let cameraPreview = null;
async function startSession() {
    if (currentMode === 'live') {
        if (cameraPreview && cameraPreview._previewing) { media = cameraPreview; cameraPreview = null; }
        else { media = new LiveMediaProvider(); }
    } else {
    }
                await media.start();

                // Start video recording
        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        media = null;
}
async function startCameraPreview() {
    if (session) return;
    if (cameraPreview && cameraPreview._previewing) return;
    try {
        cameraPreview = new LiveMediaProvider();
        await cameraPreview.startPreview();
        updateFullscreenBtnVisibility(true);
    } catch (err) {
        console.warn('Camera preview failed:', err.message);
        cameraPreview = null;
        updateFullscreenBtnVisibility(false);
        document.getElementById('videoPlaceholder').style.display = 'flex';
    }
}
""",
        encoding="utf-8",
    )
    selector = tmp_path / "audio-device-selector.js"
    selector.write_text(
        """async function enumerate() {
        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (_) { /* need permission to get labels */ }

        const devices = await navigator.mediaDevices.enumerateDevices();
}
""",
        encoding="utf-8",
    )
    html = tmp_path / "omni.html"
    html.write_text(
        '<script type="module" src="/static/omni/omni-app.js?v=vi-tts-v3-stream"></script>',
        encoding="utf-8",
    )

    assert patch_omni_app(omni)
    assert patch_audio_selector(selector)
    assert patch_html(html)
    assert not patch_omni_app(omni)
    assert not patch_audio_selector(selector)
    assert not patch_html(html)

    source = omni.read_text(encoding="utf-8")
    assert OMNI_MARKER in source
    assert "if (cameraPreviewPromise) return cameraPreviewPromise" in source
    assert source.index("mixerCtrl?.stopMixerMic();") < source.index("await media.start();")
    assert AUDIO_MARKER in selector.read_text(encoding="utf-8")
    assert "getTracks().forEach(track => track.stop())" in selector.read_text(encoding="utf-8")
    assert "stream-media-v1" in html.read_text(encoding="utf-8")

    result = subprocess.run(
        ["node", "--check", str(omni)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
