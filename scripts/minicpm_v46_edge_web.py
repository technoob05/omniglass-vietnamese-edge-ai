#!/usr/bin/env python3
"""Tiny, dependency-free web demo for the MiniCPM-V 4.6 GGUF edge lane.

The service intentionally launches the CPU/HTP-capable llama-mtmd-cli for each
request.  That makes the demo slower than a resident production runtime, but it
keeps this reference portable and guarantees that no CUDA device is used.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


MAX_BODY_BYTES = 7 * 1024 * 1024
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
INFER_LOCK = threading.Lock()

PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniCPM-V 4.6 · Edge 16GB Demo</title>
<style>
:root{color-scheme:dark;--bg:#080c14;--card:#111827;--line:#263348;--cyan:#5eead4;--pink:#fb7185;--muted:#94a3b8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#17304a 0,transparent 36%),var(--bg);font:16px system-ui;color:#f8fafc}
main{max-width:1120px;margin:auto;padding:24px}.hero{display:flex;gap:16px;align-items:center;justify-content:space-between}.badge{color:var(--cyan);border:1px solid #2dd4bf55;padding:7px 11px;border-radius:999px;font-weight:700}
h1{font-size:clamp(28px,5vw,50px);line-height:1.02;margin:18px 0 8px}.sub{color:var(--muted);max-width:760px}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px;margin-top:24px}.card{background:#111827e8;border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 18px 60px #0005}
video,canvas,#snapshot{width:100%;aspect-ratio:16/9;object-fit:cover;background:#020617;border-radius:13px}canvas{display:none}#snapshot{display:none}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}button,select,input{border-radius:11px;border:1px solid #334155;background:#172033;color:#fff;padding:12px 14px;font:inherit}button{cursor:pointer;font-weight:750}button.primary{background:var(--cyan);color:#06251f;border:0}button:disabled{opacity:.45;cursor:not-allowed}input{flex:1;min-width:220px}
#answer{font-size:21px;line-height:1.55;min-height:120px;white-space:pre-wrap}.status{color:var(--cyan);margin:12px 0}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.metric{background:#0b1220;padding:12px;border-radius:12px}.metric b{display:block;font-size:20px}.metric span{color:var(--muted);font-size:12px}.warn{color:#fda4af;font-size:13px}.small{color:var(--muted);font-size:13px}@media(max-width:760px){.grid{grid-template-columns:1fr}.hero{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<div class="hero"><div><div class="badge">EDGE-FIT · CPU ONLY · NO CUDA</div><h1>MiniCPM‑V 4.6 Q4</h1><p class="sub">Web thử model MiniCPM nhỏ nhất trong stack: camera chạy tại trình duyệt, mỗi câu hỏi chụp đúng một frame rồi suy luận CPU trên máy chủ.</p></div></div>
<div class="grid"><section class="card"><video id="cam" autoplay playsinline muted></video><canvas id="canvas"></canvas><img id="snapshot" alt="Frame vừa phân tích"><div class="row"><button id="start">📷 Bật camera</button><button id="flip">↺ Đổi camera</button><label><button id="pick">⬆️ Chọn ảnh</button><input id="file" type="file" accept="image/*" hidden></label></div><p class="small">Không stream video liên tục. Ảnh chỉ được gửi khi bấm Hỏi.</p></section>
<section class="card"><div class="metrics"><div class="metric"><b>1.61 GB</b><span>Q4 + projector</span></div><div class="metric"><b>~2.0 GB</b><span>peak RSS đã đo</span></div><div class="metric"><b>0 MB</b><span>GPU dùng</span></div></div><div class="row"><select id="lang"><option value="vi">Trả lời tiếng Việt</option><option value="en">Answer in English</option></select></div><div class="row"><input id="q" value="Bạn đang thấy gì?" aria-label="Câu hỏi"><button class="primary" id="ask">Hỏi model</button></div><div class="status" id="status">Sẵn sàng.</div><div id="answer">Kết quả sẽ hiện ở đây.</div><p class="warn">Bản CPU kiểm chứng ngân sách 16 GB, chưa phải tốc độ realtime. Một lượt hiện mất khoảng 30–35 giây; QCS8550 HTP là bước benchmark tiếp theo.</p></section></div>
</main><script>
const cam=document.querySelector('#cam'),canvas=document.querySelector('#canvas'),snap=document.querySelector('#snapshot'),statusEl=document.querySelector('#status'),answer=document.querySelector('#answer');
let stream=null,facing='environment',picked=null;
async function camera(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:1280},height:{ideal:720}},audio:false});cam.srcObject=stream;await cam.play();picked=null;cam.style.display='block';snap.style.display='none';statusEl.textContent='Camera đã sẵn sàng.'}
document.querySelector('#start').onclick=()=>camera().catch(e=>statusEl.textContent='Không mở được camera: '+e.message);
document.querySelector('#flip').onclick=()=>{facing=facing==='environment'?'user':'environment';camera().catch(e=>statusEl.textContent=e.message)};
document.querySelector('#pick').onclick=()=>document.querySelector('#file').click();
document.querySelector('#file').onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{picked=r.result;snap.src=picked;snap.style.display='block';cam.style.display='none';statusEl.textContent='Ảnh đã chọn.'};r.readAsDataURL(f)};
function frame(){if(picked)return picked;if(!stream||cam.readyState<2)throw new Error('Hãy bật camera hoặc chọn một ảnh trước.');const max=768,scale=Math.min(1,max/cam.videoWidth);canvas.width=Math.round(cam.videoWidth*scale);canvas.height=Math.round(cam.videoHeight*scale);canvas.getContext('2d').drawImage(cam,0,0,canvas.width,canvas.height);const data=canvas.toDataURL('image/jpeg',.82);snap.src=data;snap.style.display='block';cam.style.display='none';setTimeout(()=>{if(!picked){cam.style.display='block';snap.style.display='none'}},1400);return data}
document.querySelector('#ask').onclick=async()=>{const btn=document.querySelector('#ask');try{const image=frame(),question=document.querySelector('#q').value.trim();if(!question)throw new Error('Bạn chưa nhập câu hỏi.');btn.disabled=true;answer.textContent='';const started=performance.now();statusEl.textContent='Model CPU đang nhìn ảnh… dự kiến 30–35 giây.';const timer=setInterval(()=>statusEl.textContent=`Đang suy luận… ${Math.round((performance.now()-started)/1000)} giây`,1000);try{const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,question,language:document.querySelector('#lang').value})});const out=await res.json();if(!res.ok)throw new Error(out.error||'Inference failed');answer.textContent=out.answer;statusEl.textContent=`Hoàn tất trong ${out.wall_seconds.toFixed(1)} giây · CPU only · ${out.model}`;}finally{clearInterval(timer)}}catch(e){statusEl.textContent='Lỗi: '+e.message}finally{btn.disabled=false}};
window.addEventListener('pagehide',()=>stream?.getTracks().forEach(t=>t.stop()));
</script></body></html>'''


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def _decode_image(data_url: str) -> tuple[bytes, str]:
    match = re.fullmatch(r"data:image/(jpeg|jpg|png);base64,([A-Za-z0-9+/=\r\n]+)", data_url)
    if not match:
        raise ValueError("Ảnh phải là JPEG hoặc PNG hợp lệ.")
    raw = base64.b64decode(match.group(2), validate=True)
    if not raw or len(raw) > 5 * 1024 * 1024:
        raise ValueError("Ảnh trống hoặc lớn hơn 5 MB.")
    return raw, ".png" if match.group(1) == "png" else ".jpg"


def _prompt(question: str, language: str) -> str:
    question = " ".join(question.strip().split())[:500]
    if language == "en":
        return f"You are a visual assistant. Answer only in concise English. Question: {question}"
    return (
        "Bạn là trợ lý thị giác. BẮT BUỘC chỉ trả lời bằng tiếng Việt tự nhiên, ngắn gọn; "
        "không dùng tiếng Anh. Nếu không chắc, hãy nói là không chắc. Câu hỏi: " + question
    )


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: argparse.Namespace):
        super().__init__(address, Handler)
        self.config = config


class Handler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"edge-web {self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            cfg = self.server.config
            self._send(HTTPStatus.OK, _json_bytes({"ok": True, "model": "MiniCPM-V-4.6-Q4_0", "cpu_only": True, "busy": INFER_LOCK.locked(), "model_exists": cfg.model.is_file(), "projector_exists": cfg.mmproj.is_file()}), "application/json; charset=utf-8")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > MAX_BODY_BYTES:
                raise ValueError("Request quá lớn hoặc trống.")
            payload = json.loads(self.rfile.read(size))
            image, suffix = _decode_image(str(payload.get("image", "")))
            prompt = _prompt(str(payload.get("question", "")), str(payload.get("language", "vi")))
            if not prompt.strip():
                raise ValueError("Câu hỏi trống.")
            if not INFER_LOCK.acquire(blocking=False):
                self._send(HTTPStatus.CONFLICT, _json_bytes({"error": "Model đang xử lý một lượt khác; thử lại sau."}), "application/json; charset=utf-8")
                return
            started = time.perf_counter()
            path = ""
            try:
                with tempfile.NamedTemporaryFile(prefix="minicpm-v46-", suffix=suffix, delete=False) as temp:
                    temp.write(image)
                    path = temp.name
                cfg = self.server.config
                cmd = [str(cfg.binary), "-m", str(cfg.model), "--mmproj", str(cfg.mmproj), "--image", path, "-p", prompt, "-c", str(cfg.context), "-n", str(cfg.tokens), "-t", str(cfg.threads), "-ngl", "0", "--no-mmproj-offload", "--no-warmup"]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=cfg.timeout,
                    check=False,
                )
                answer = ANSI_ESCAPE.sub("", proc.stdout).strip()
                if proc.returncode != 0 or not answer:
                    tail = ANSI_ESCAPE.sub("", proc.stderr)[-1200:].strip()
                    raise RuntimeError(f"Model exit {proc.returncode}: {tail}")
                result = {"answer": answer, "wall_seconds": time.perf_counter() - started, "model": "MiniCPM-V-4.6 Q4_0", "cpu_only": True}
                self._send(HTTPStatus.OK, _json_bytes(result), "application/json; charset=utf-8")
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
                INFER_LOCK.release()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, _json_bytes({"error": str(exc)}), "application/json; charset=utf-8")
        except subprocess.TimeoutExpired:
            self._send(HTTPStatus.GATEWAY_TIMEOUT, _json_bytes({"error": "Model vượt quá thời gian chờ."}), "application/json; charset=utf-8")
        except Exception as exc:  # keep a useful demo error without exposing a traceback
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _json_bytes({"error": str(exc)}), "application/json; charset=utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18846)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    for name in ("binary", "model", "mmproj"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name} not found: {path}")
    return args


def main() -> None:
    args = parse_args()
    server = DemoServer((args.host, args.port), args)
    print(f"MiniCPM-V 4.6 edge demo: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
