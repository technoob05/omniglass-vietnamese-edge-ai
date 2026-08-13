"""Dependency-free local HTTP API for the AIBOX-eye service."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .orchestrator import EyeOrchestrator


_HTML = """<!doctype html><html lang=\"vi\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>AIBOX-eye control</title><style>body{font:16px system-ui;margin:2rem;max-width:60rem;background:#10151d;color:#e8edf4}pre{background:#18202b;padding:1rem;overflow:auto}button{padding:.5rem;margin-right:.5rem}input{width:30rem;max-width:100%;padding:.5rem}</style>
<h1>AIBOX-eye</h1><p>Safety lane reads the existing single-engine QNN service. VLM is on-demand only.</p><p><a href=\"http://127.0.0.1:8080/\">Perception stream</a> · <a href=\"/health\">health</a> · <a href=\"/scene\">scene</a> · <a href=\"/events\">events</a></p>
<input id=q placeholder=\"Ví dụ: phía trước có gì?\"><button onclick=\"ask()\">Hỏi</button><button onclick=\"ptt('/push-to-talk/start')\">Giữ để nói</button><button onclick=\"ptt('/push-to-talk/stop')\">Dừng nói</button><pre id=o>loading…</pre>
<script>async function get(u){let r=await fetch(u,{cache:'no-store'});return await r.json()} async function ask(){o.textContent=JSON.stringify(await get('/ask?text='+encodeURIComponent(q.value)),null,2)} async function ptt(u){let r=await fetch(u,{method:'POST'});o.textContent=await r.text()} async function refresh(){try{o.textContent=JSON.stringify(await get('/health'),null,2)}catch(e){o.textContent=e}} refresh();setInterval(refresh,2000)</script></html>""".encode("utf-8")


def handler_factory(app: EyeOrchestrator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIBOX-eye/0.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_HTML)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_HTML)
                return
            if parsed.path == "/health":
                self.send_json(app.health())
                return
            if parsed.path == "/scene":
                self.send_json({"scene": app.latest_scene()})
                return
            if parsed.path == "/events":
                self.send_json({"events": app.events()})
                return
            if parsed.path == "/audio/status":
                self.send_json(app.audio.status.to_dict())
                return
            if parsed.path == "/metrics":
                body = app.prometheus_metrics().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/ask":
                query = _query(parsed.query)
                self._ask(str(query.get("text", "")))
                return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/ask":
                    body = self._read_json_body()
                    self._ask(str(body.get("text", "")))
                    return
                if parsed.path == "/push-to-talk/start":
                    self.send_json(app.start_push_to_talk())
                    return
                if parsed.path == "/push-to-talk/stop":
                    self.send_json(app.stop_push_to_talk())
                    return
                if parsed.path == "/demo/reset":
                    self.send_json(app.reset_demo())
                    return
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error:
                self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as error:
                self.send_json({"error": f"internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _ask(self, text: str) -> None:
            try:
                self.send_json(app.ask(text))
            except ValueError as error:
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error:
                self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384:
                raise ValueError("JSON body must be 1..16384 bytes")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON body must be an object")
            return value

    return Handler


def _query(value: str) -> dict[str, str]:
    from urllib.parse import parse_qs

    return {key: values[-1] for key, values in parse_qs(value, keep_blank_values=True).items() if values}


def make_server(app: EyeOrchestrator, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_factory(app))
    server.daemon_threads = True
    return server
