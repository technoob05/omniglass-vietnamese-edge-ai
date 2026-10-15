#!/usr/bin/env python3
"""Add same-origin H100 perception proxies to MiniCPM-o-Demo gateway."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = "# OPENGLASS_VI_PERCEPTION_PROXY_V1"
LEGACY_FACE_MARKER = "# OPENGLASS_VI_FAMILIAR_FACE_PROXY_V1"
LEGACY_FACE_ROUTES = (
    ("post", "/api/perception/vi/faces/enroll", "vietnamese_perception_face_enroll"),
    ("get", "/api/perception/vi/faces", "vietnamese_perception_faces_list"),
    ("post", "/api/perception/vi/faces/delete", "vietnamese_perception_face_delete"),
    ("post", "/api/perception/vi/faces/clear", "vietnamese_perception_faces_clear"),
)


def remove_legacy_face_routes(text: str) -> str:
    cleaned = text.replace(LEGACY_FACE_MARKER + "\n", "")
    for method, route, function_name in LEGACY_FACE_ROUTES:
        pattern = re.compile(
            rf'\n*@app\.{method}\("{re.escape(route)}"\)\n'
            rf'async def {function_name}\([^\n]*\):\n'
            rf'    return await _vi_perception_proxy\([^\n]+\)\n',
        )
        cleaned = pattern.sub("\n", cleaned)
    return cleaned


def patch_gateway(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    anchor = '@app.get("/api/presets")\nasync def get_presets():'
    if text.count(anchor) != 1:
        raise RuntimeError(f"Expected one gateway anchor, found {text.count(anchor)}")
    if MARKER in text:
        cleaned = remove_legacy_face_routes(text)
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")
            return True
        return False
    block = f'''{MARKER}
async def _vi_perception_proxy(path: str, payload: dict | None = None):
    base = os.getenv("OPENGLASS_VI_PERCEPTION_URL", "http://127.0.0.1:18784")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=2.0)) as client:
            if payload is None:
                response = await client.get(base + path)
            else:
                response = await client.post(base + path, json=payload)
        response.raise_for_status()
        return JSONResponse(response.json())
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except Exception as exc:
        logger.warning("Vietnamese perception proxy failed path=%s error=%s", path, exc)
        raise HTTPException(status_code=503, detail="H100 perception unavailable") from exc


@app.get("/api/perception/vi/health")
async def vietnamese_perception_health():
    return await _vi_perception_proxy("/health")


@app.post("/api/perception/vi/frame")
async def vietnamese_perception_frame(request: Request):
    return await _vi_perception_proxy("/frame", await request.json())


@app.post("/api/perception/vi/reset")
async def vietnamese_perception_reset(request: Request):
    return await _vi_perception_proxy("/reset", await request.json())


@app.post("/api/perception/vi/query")
async def vietnamese_perception_query(request: Request):
    return await _vi_perception_proxy("/query", await request.json())


{anchor}'''
    path.write_text(text.replace(anchor, block, 1), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gateway", type=Path)
    args = parser.parse_args()
    print(f"gateway_changed={patch_gateway(args.gateway)}")


if __name__ == "__main__":
    main()
