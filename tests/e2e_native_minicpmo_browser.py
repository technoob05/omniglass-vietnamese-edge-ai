import json
import time

from playwright.sync_api import sync_playwright


URL = "https://127.0.0.1:8006/omni"
SCREENSHOT = "results/native_minicpmo_omni.png"


def main() -> None:
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
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
            lambda message: console_errors.append(
                f"{message.text} @ {message.location.get('url', '')}"
            )
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on(
            "requestfailed",
            lambda request: failed_requests.append(
                f"{request.method} {request.url}: {request.failure}"
            ),
        )

        started = time.perf_counter()
        response = page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("#btnStart", state="visible", timeout=15_000)
        page.wait_for_function(
            """() => Array.from(document.querySelectorAll('.preset-loading-overlay'))
              .every(el => getComputedStyle(el).display === 'none')""",
            timeout=30_000,
        )
        page.wait_for_timeout(500)
        load_seconds = time.perf_counter() - started

        initial = {
            "title": page.title(),
            "body_chars": len(page.locator("body").inner_text().strip()),
            "start_visible": page.locator("#btnStart").is_visible(),
            "video_visible": page.locator("#videoEl").is_visible(),
            "http_status": response.status if response else None,
        }

        page.locator("#btnStart").click()
        try:
            page.wait_for_function(
                "() => !document.querySelector('#btnStop').disabled",
                timeout=30_000,
            )
        except Exception:
            pass
        page.wait_for_timeout(2_000)
        active = {
            "service_status": page.locator("#serviceStatus").inner_text().strip(),
            "start_disabled": page.locator("#btnStart").is_disabled(),
            "stop_enabled": not page.locator("#btnStop").is_disabled(),
            "video_ready_state": page.locator("#videoEl").evaluate("el => el.readyState"),
            "video_width": page.locator("#videoEl").evaluate("el => el.videoWidth"),
            "video_height": page.locator("#videoEl").evaluate("el => el.videoHeight"),
        }
        page.screenshot(path=SCREENSHOT, full_page=True)

        if active["stop_enabled"]:
            page.locator("#btnStop").click()
            page.wait_for_timeout(1_000)

        result = {
            "url": URL,
            "load_seconds": round(load_seconds, 3),
            "initial": initial,
            "active": active,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "failed_requests": failed_requests,
            "screenshot": SCREENSHOT,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
