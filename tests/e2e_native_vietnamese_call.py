import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright


URL = os.environ.get("OPENGLASS_NATIVE_URL", "https://127.0.0.1:8006/omni")
DEFAULT_AUDIO = Path(os.environ.get("TEMP", ".")) / "openglass_vi_ref.wav"


def _padded_audio(source: Path) -> Path:
    with wave.open(str(source), "rb") as wav_in:
        params = wav_in.getparams()
        frames = wav_in.readframes(params.nframes)
    if params.nchannels != 1 or params.sampwidth != 2:
        raise RuntimeError("Vietnamese test audio must be mono PCM16")

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    output = Path(handle.name)
    silence = b"\x00\x00" * params.framerate
    with wave.open(str(output), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(params.framerate)
        wav_out.writeframes(silence + frames + silence * 2)
    return output


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source = Path(os.environ.get("OPENGLASS_VI_TEST_AUDIO", DEFAULT_AUDIO))
    if not source.exists():
        raise FileNotFoundError(source)
    padded = _padded_audio(source)
    console_errors: list[str] = []
    page_errors: list[str] = []
    vi_tts_statuses: list[int] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-audio-capture={padded.as_posix()}",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            context = browser.new_context(
                ignore_https_errors=True,
                permissions=["camera", "microphone"],
                viewport={"width": 1440, "height": 1000},
            )
            page = context.new_page()
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "response",
                lambda response: vi_tts_statuses.append(response.status)
                if "/api/tts/vi" in response.url
                else None,
            )

            page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
            vi_button = page.locator('.preset-btn[data-preset-id="vietnamese_call"]')
            vi_button.wait_for(state="visible", timeout=30_000)
            vi_button.click()
            page.wait_for_function(
                """() => document.querySelector('#systemPrompt').value
                  .includes('trả lời hoàn toàn bằng tiếng Việt')""",
                timeout=15_000,
            )
            page.wait_for_function(
                """() => Array.from(document.querySelectorAll('.preset-loading-overlay'))
                  .every(el => getComputedStyle(el).display === 'none')""",
                timeout=30_000,
            )

            started = time.perf_counter()
            page.locator("#btnStart").click()
            page.wait_for_function(
                "() => !document.querySelector('#btnStop').disabled",
                timeout=30_000,
            )
            try:
                page.wait_for_function(
                    """() => Array.from(document.querySelectorAll('.conv-entry.speak .conv-text'))
                      .some(el => el.textContent.trim().length > 0)""",
                    timeout=60_000,
                )
            except Exception:
                pass

            answers = page.locator(".conv-entry.speak .conv-text").all_inner_texts()
            for _ in range(200):
                if vi_tts_statuses:
                    break
                page.wait_for_timeout(100)
            status = page.locator("#serviceStatus").inner_text().strip()
            elapsed = time.perf_counter() - started
            page.screenshot(path="results/native_vietnamese_call.png", full_page=True)
            if not page.locator("#btnStop").is_disabled():
                page.locator("#btnStop").click()

            result = {
                "preset": "vietnamese_call",
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
                "answers": answers,
                "vi_tts_statuses": vi_tts_statuses,
                "console_errors": console_errors,
                "page_errors": page_errors,
            }
            Path("results/native_vietnamese_call_report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            browser.close()
    finally:
        padded.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
