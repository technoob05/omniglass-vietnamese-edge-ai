#!/usr/bin/env python3
"""PC browser demo for MiniCPM-V 4.6 running on a Qualcomm box via ADB.

The browser owns the webcam. Each question pushes one resized JPEG to the box
and invokes the resident ARM64 Hexagon-capable mtmd binary. This is a demo
adapter, intentionally single-flight and turn-based.
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
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 5 * 1024 * 1024
LOCK = threading.Lock()
SERIAL = os.environ.get("QUALCOMM_SERIAL", "17513b4")
ADB = os.environ.get("ADB", r"D:\PhD_LetGoo\PhD_Farming\edge-ai\.tools\platform-tools\adb.exe")
REMOTE_ROOT = os.environ.get("QUALCOMM_REMOTE_ROOT", "/data/local/tmp/omniglass-minicpm-v46-v1")
REMOTE_INBOX = os.environ.get("QUALCOMM_REMOTE_INBOX", "/data/local/tmp/omniglass-v46-resident-v1/inbox/latest.jpg")
RESIDENT_PORT = int(os.environ.get("QUALCOMM_RESIDENT_PORT", "18191"))
RESIDENT_URL = os.environ.get("QUALCOMM_RESIDENT_URL", f"http://127.0.0.1:{RESIDENT_PORT}")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

PAGE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniCPM-V 4.6 · Qualcomm HTP</title><style>
body{font:16px system-ui;background:#080c14;color:#f8fafc;max-width:1000px;margin:auto;padding:24px}main{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:#111827;border:1px solid #334155;border-radius:16px;padding:16px}video,img{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:12px;background:#020617}button,input,select{font:inherit;padding:11px;border-radius:9px;border:1px solid #475569;background:#172033;color:white;margin:5px 0}button{cursor:pointer;font-weight:700}.primary{background:#5eead4;color:#052e2b;border:0}input{width:calc(100% - 130px)}#answer{font-size:20px;line-height:1.5;min-height:120px}.muted{color:#94a3b8}.ok{color:#5eead4}@media(max-width:760px){main{grid-template-columns:1fr}}
</style><h1>MiniCPM-V 4.6 · Qualcomm HTP</h1><p class="muted">Webcam trên máy tính → một JPEG mới nhất → ADB → MiniCPM-V chạy trên box.</p><main><section class="card"><video id="v" autoplay playsinline muted></video><canvas id="c" hidden></canvas><img id="snap" hidden><p><button id="cam">Bật camera</button><button id="flip">Đổi camera</button></p></section><section class="card"><p><select id="lang"><option value="vi">Trả lời tiếng Việt</option><option value="en">Answer in English</option></select></p><p><input id="q" value="Bạn đang thấy gì?"/><button class="primary" id="ask">Hỏi</button></p><p id="status" class="ok">Sẵn sàng.</p><div id="answer">Chưa có câu trả lời.</div><p class="muted">Single-flight · 224px · 64 visual tokens · HTP0. Một lượt khoảng vài giây.</p></section></main><script>
const v=document.querySelector('#v'),c=document.querySelector('#c'),s=document.querySelector('#snap'),st=document.querySelector('#status'),a=document.querySelector('#answer');let media=null,face='environment',busy=false;
async function start(){if(media)media.getTracks().forEach(x=>x.stop());media=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:face},width:{ideal:1280},height:{ideal:720}},audio:false});v.srcObject=media;await v.play();v.hidden=false;s.hidden=true;st.textContent='Camera đã sẵn sàng.'}
document.querySelector('#cam').onclick=()=>start().catch(e=>st.textContent='Camera: '+e.message);document.querySelector('#flip').onclick=()=>{face=face==='environment'?'user':'environment';start().catch(e=>st.textContent=e.message)};
document.querySelector('#ask').onclick=async()=>{if(busy)return;try{if(!media||v.readyState<2)await start();const scale=Math.min(1,224/Math.max(v.videoWidth,v.videoHeight));c.width=Math.max(1,Math.round(v.videoWidth*scale));c.height=Math.max(1,Math.round(v.videoHeight*scale));c.getContext('2d').drawImage(v,0,0,c.width,c.height);const image=c.toDataURL('image/jpeg',.82);s.src=image;s.hidden=false;v.hidden=true;busy=true;st.textContent='Đang gửi ảnh tới Qualcomm box…';const t=performance.now();const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,question:document.querySelector('#q').value,language:document.querySelector('#lang').value})});const o=await r.json();if(!r.ok)throw Error(o.error||'inference failed');a.textContent=o.answer;st.textContent=`HTP0 · ${(performance.now()-t)/1000|0}s · ${o.profile}`;setTimeout(()=>{v.hidden=false;s.hidden=true},1200)}catch(e){st.textContent='Lỗi: '+e.message}finally{busy=false}};addEventListener('pagehide',()=>media?.getTracks().forEach(x=>x.stop()));
</script>'''

def adb(*args: str, timeout: float = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run([ADB, "-s", SERIAL, *args], text=True, capture_output=True, timeout=timeout)

def ensure_forward() -> None:
    result = adb("forward", f"tcp:{RESIDENT_PORT}", f"tcp:{RESIDENT_PORT}", timeout=10)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ADB port forward failed")

def resident_request(path: str, payload: dict | None = None, timeout: float = 90) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{RESIDENT_URL}{path}", data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"resident MiniCPM-V service unavailable: {exc}") from exc

def run_box(image: bytes, question: str, language: str) -> dict:
    """Push only the newest frame; inference stays resident on the QCS box."""
    if not LOCK.acquire(False):
        raise RuntimeError("Qualcomm box is already processing a turn.")
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as file:
            file.write(image)
            local = file.name
        try:
            result = adb("shell", "mkdir", "-p", os.path.dirname(REMOTE_INBOX), timeout=10)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "cannot create box inbox")
            result = adb("push", local, REMOTE_INBOX, timeout=20)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "ADB push failed")
            language_rule = "Answer only in natural Vietnamese." if language == "vi" else "Answer only in concise English."
            prompt = f"You are a visual assistant. {language_rule} Question: {question[:500]}"
            ensure_forward()
            started = time.perf_counter()
            response = resident_request("/v1/vision", {"image_path": REMOTE_INBOX, "prompt": prompt})
            answer = str(response.get("answer", "")).strip()
            if not answer:
                raise RuntimeError("resident model returned no text")
            return {
                "answer": answer,
                "profile": "MiniCPM-V 4.6 resident · HTP0 vision · 64 visual tokens",
                "wall_seconds": round(time.perf_counter() - started, 2),
            }
        finally:
            Path(local).unlink(missing_ok=True)
    finally:
        LOCK.release()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        try:
            print(fmt % args, flush=True)
        except OSError:
            pass
    def send(self, code, body, typ="application/json"):
        self.send_response(code); self.send_header("Content-Type",typ+"; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path=="/": self.send(200,PAGE.encode(),"text/html")
        elif self.path=="/health":
            try:
                ensure_forward()
                status = resident_request("/health", timeout=5)
                self.send(200, json.dumps({"ok": status.get("status") == "ok", "serial": SERIAL, "profile": "resident HTP0"}).encode())
            except Exception as exc:
                self.send(503, json.dumps({"ok": False, "error": str(exc)}).encode())
        else:self.send(404,b"not found","text/plain")
    def do_POST(self):
        if self.path!="/api/ask": return self.send(404,b"not found","text/plain")
        try:
            n=int(self.headers.get("Content-Length","0"));
            if n<=0 or n>MAX_BODY: raise ValueError("request too large")
            x=json.loads(self.rfile.read(n)); m=re.fullmatch(r"data:image/jpeg;base64,([A-Za-z0-9+/=\r\n]+)",str(x.get("image","")))
            if not m: raise ValueError("JPEG không hợp lệ")
            out=run_box(base64.b64decode(m.group(1),validate=True),str(x.get("question","")),str(x.get("language","vi"))); self.send(200,json.dumps(out,ensure_ascii=False).encode())
        except Exception as e:self.send(409 if "đang xử lý" in str(e) else 500,json.dumps({"error":str(e)},ensure_ascii=False).encode())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=7876); args=ap.parse_args(); print(f"Open http://{args.host}:{args.port}"); ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()
if __name__=="__main__": main()
