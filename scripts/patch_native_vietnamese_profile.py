#!/usr/bin/env python3
"""Install the isolated Vietnamese turn-based assistant into MiniCPM-o Demo."""

from __future__ import annotations

import argparse
from pathlib import Path


GATEWAY_MARKER = "# OPENGLASS_VI_ASSISTANT_PROFILE_V1"


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
@app.get("/vi")
async def vietnamese_assistant_page():
    return FileResponse(os.path.join(_BASE_DIR, "static", "vi", "vi-chat.html"))


@app.get("/api/asr/vi/health")
async def vietnamese_asr_health_proxy():
    url = os.getenv("OPENGLASS_VI_ASR_HTTP_URL", "http://127.0.0.1:18783/health")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        return JSONResponse(response.json())
    except Exception as exc:
        logger.warning("Vietnamese ASR health proxy failed: %s", exc)
        raise HTTPException(status_code=503, detail="Vietnamese ASR unavailable") from exc


@app.websocket("/v1/asr/vi")
async def vietnamese_asr_websocket_proxy(browser_ws: WebSocket):
    import websockets

    await browser_ws.accept()
    upstream_url = os.getenv("OPENGLASS_VI_ASR_WS_URL", "ws://127.0.0.1:18783/v1/asr")
    query = str(browser_ws.query_params)
    if query:
        upstream_url += ("&" if "?" in upstream_url else "?") + query
    upstream_ws = None
    tasks = []
    try:
        upstream_ws = await websockets.connect(
            upstream_url,
            open_timeout=5,
            close_timeout=2,
            max_size=16 * 1024 * 1024,
        )

        async def browser_to_asr():
            while True:
                message = await browser_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await upstream_ws.send(message["bytes"])
                elif message.get("text") is not None:
                    await upstream_ws.send(message["text"])

        async def asr_to_browser():
            async for message in upstream_ws:
                if isinstance(message, bytes):
                    await browser_ws.send_bytes(message)
                else:
                    await browser_ws.send_text(message)

        tasks = [
            asyncio.create_task(browser_to_asr()),
            asyncio.create_task(asr_to_browser()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("Vietnamese ASR WebSocket proxy failed: %s", exc)
        try:
            await browser_ws.send_json({{
                "type": "asr.error",
                "code": "upstream_unavailable",
                "detail": "Vietnamese ASR unavailable",
            }})
        except Exception:
            pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if upstream_ws is not None:
            await upstream_ws.close()
        try:
            await browser_ws.close()
        except Exception:
            pass


{anchor}'''
    path.write_text(replace_once(text, anchor, block, "gateway preset route"), encoding="utf-8")
    return True


def install_assets(demo_root: Path, assets_root: Path) -> bool:
    target = demo_root / "static" / "vi"
    target.mkdir(parents=True, exist_ok=True)
    changed = False
    for name in ("vi-chat.html", "vi-chat.js"):
        source_bytes = (assets_root / name).read_bytes()
        destination = target / name
        if not destination.exists() or destination.read_bytes() != source_bytes:
            destination.write_bytes(source_bytes)
            changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_root", type=Path)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "native-overrides" / "vi-profile",
    )
    args = parser.parse_args()
    gateway_changed = patch_gateway(args.demo_root / "gateway.py")
    assets_changed = install_assets(args.demo_root, args.assets_root)
    print(f"gateway_changed={gateway_changed} assets_changed={assets_changed}")


if __name__ == "__main__":
    main()
