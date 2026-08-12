#!/usr/bin/env python3
"""Patch MiniCPM-o's Omni UI to recover cleanly from camera/mic contention.

The upstream UI can create a second LiveMediaProvider when Start is clicked while
the initial camera preview is still awaiting getUserMedia().  Chromium then
reports ``NotReadableError: Device in use`` on some Windows webcams.  The shared
audio device selector also opened a permission stream without stopping it.

This patch is deliberately small, idempotent, and limited to browser assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path


OMNI_MARKER = "OPENGLASS_OMNI_MEDIA_RECOVERY_V1"
OMNI_NO_THROW_MARKER = "OPENGLASS_OMNI_MEDIA_RECOVERY_NO_THROW_V1"
AUDIO_MARKER = "OPENGLASS_AUDIO_ENUM_RELEASE_V1"


def _replace_once(source: str, old: str, new: str, *, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def patch_omni_app(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if OMNI_MARKER in source:
        if OMNI_NO_THROW_MARKER in source:
            return False
        source = _replace_once(
            source,
            "            throw err;",
            "            // OPENGLASS_OMNI_MEDIA_RECOVERY_NO_THROW_V1\n"
            "            return;",
            label="preview failure handled",
        )
        path.write_text(source, encoding="utf-8", newline="\n")
        return True

    source = _replace_once(
        source,
        "let cameraPreview = null;",
        "let cameraPreview = null;\n"
        "let cameraPreviewPromise = null; // OPENGLASS_OMNI_MEDIA_RECOVERY_V1",
        label="preview promise declaration",
    )

    source = _replace_once(
        source,
        """    if (currentMode === 'live') {
        if (cameraPreview && cameraPreview._previewing) { media = cameraPreview; cameraPreview = null; }
        else { media = new LiveMediaProvider(); }
    } else {""",
        """    if (currentMode === 'live') {
        // Reuse the one preview acquisition already in flight. Creating a second
        // provider here races getUserMedia() and yields `Device in use` on Windows.
        if (cameraPreviewPromise) {
            try { await cameraPreviewPromise; } catch (_) { /* start below reports it */ }
        }
        if (cameraPreview) { media = cameraPreview; cameraPreview = null; }
        else { media = new LiveMediaProvider(); }
    } else {""",
        label="session preview handoff",
    )

    source = _replace_once(
        source,
        """                await media.start();

                // Start video recording""",
        """                // The mixer owns a standalone microphone while its panel is open.
                // Release it before the live provider requests the selected microphone.
                mixerCtrl?.stopMixerMic();
                await media.start();

                // Start video recording""",
        label="mixer release before media start",
    )

    source = _replace_once(
        source,
        """        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        media = null;""",
        """        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        // Always release partially acquired camera/mic tracks before retrying preview.
        if (media) { try { media.stop(); } catch (_) {} }
        media = null;""",
        label="error-path media cleanup",
    )

    old_preview = """async function startCameraPreview() {
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
}"""
    new_preview = """async function startCameraPreview() {
    if (session) return;
    if (cameraPreview && cameraPreview._previewing) return;
    if (cameraPreviewPromise) return cameraPreviewPromise;

    const candidate = cameraPreview || new LiveMediaProvider();
    cameraPreview = candidate;
    cameraPreviewPromise = (async () => {
        try {
            await candidate.startPreview();
            updateFullscreenBtnVisibility(true);
        } catch (err) {
            console.warn('Camera preview failed:', err.message);
            try { candidate.stopPreview(); } catch (_) {}
            if (cameraPreview === candidate) cameraPreview = null;
            updateFullscreenBtnVisibility(false);
            document.getElementById('videoPlaceholder').style.display = 'flex';
            // OPENGLASS_OMNI_MEDIA_RECOVERY_NO_THROW_V1
            return;
        } finally {
            cameraPreviewPromise = null;
        }
    })();
    return cameraPreviewPromise;
}"""
    source = _replace_once(source, old_preview, new_preview, label="single-flight preview")
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def patch_audio_selector(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if AUDIO_MARKER in source:
        return False
    source = _replace_once(
        source,
        """        try {
            await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (_) { /* need permission to get labels */ }

        const devices = await navigator.mediaDevices.enumerateDevices();""",
        """        // OPENGLASS_AUDIO_ENUM_RELEASE_V1
        // Permission is needed for device labels, but the temporary stream must not
        // keep the microphone locked after enumeration.
        let permissionStream = null;
        try {
            permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (_) { /* labels may remain anonymous */ }
        finally {
            permissionStream?.getTracks().forEach(track => track.stop());
        }

        const devices = await navigator.mediaDevices.enumerateDevices();""",
        label="audio enumeration release",
    )
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def patch_html(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    old = 'src="/static/omni/omni-app.js?v=vi-tts-v3-stream"'
    new = 'src="/static/omni/omni-app.js?v=vi-tts-v3-stream-media-v1"'
    if new in source:
        return False
    source = _replace_once(source, old, new, label="omni cache bust")
    path.write_text(source, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.demo_root.resolve()
    changed = {
        "omni_app_changed": patch_omni_app(root / "static/omni/omni-app.js"),
        "audio_selector_changed": patch_audio_selector(root / "static/lib/audio-device-selector.js"),
        "html_changed": patch_html(root / "static/omni/omni.html"),
    }
    for key, value in changed.items():
        print(f"{key}={str(value).lower()}")


if __name__ == "__main__":
    main()
