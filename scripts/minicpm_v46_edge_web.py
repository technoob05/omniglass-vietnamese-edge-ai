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
import shutil
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
RUNTIME_STATS: dict[str, Any] = {}


def _rss_bytes(pid: int) -> int:
    """Read resident bytes without adding psutil to the edge reference."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    return 0


def _gpu_process_bytes(pid: int) -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    command = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    try:
        rows = subprocess.check_output(command, text=True, encoding="utf-8", errors="replace", timeout=3)
        for row in rows.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 2 and fields[0] == str(pid) and fields[1].isdigit():
                return int(fields[1]) * 1024 * 1024
    except (subprocess.SubprocessError, ValueError):
        pass
    return 0


def _cuda_buffer_bytes(stderr: str) -> int:
    values = []
    for match in re.finditer(
        r"CUDA\d* (?:model|KV|RS|compute) buffer size\s*=\s*([0-9.]+)\s*MiB", stderr
    ):
        values.append(float(match.group(1)))
    # The runtime's fit planner includes allocator/runtime overhead that is
    # absent from individual buffer lines. Prefer its projected total.
    projected = re.findall(r"projected to use\s+([0-9.]+)\s+MiB of device memory", stderr)
    if projected:
        values.append(max(float(value) for value in projected))
    return round(sum(values) * 1024 * 1024)


def _run_bounded(cmd: list[str], timeout: int, budget_bytes: int, cuda: bool) -> tuple[int, str, str, dict[str, float]]:
    """Run one inference and fail closed if host+CUDA delta crosses the gate."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout_file, tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr_file:
        proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True)
        started = time.perf_counter()
        peak_host = peak_cuda = 0
        violation = False
        while proc.poll() is None:
            peak_host = max(peak_host, _rss_bytes(proc.pid))
            if cuda:
                peak_cuda = max(peak_cuda, _gpu_process_bytes(proc.pid))
            if peak_host + peak_cuda > budget_bytes:
                violation = True
                proc.kill()
                break
            if time.perf_counter() - started > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout)
            time.sleep(0.05)
        code = proc.wait()
        peak_host = max(peak_host, _rss_bytes(proc.pid))
        stdout_file.seek(0); stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    if cuda:
        peak_cuda = max(peak_cuda, _cuda_buffer_bytes(stderr))
        if peak_cuda == 0:
            raise RuntimeError("CUDA memory is not observable; refusing to claim the 12 GiB edge gate.")
    if violation:
        raise RuntimeError(f"Memory gate exceeded: host+CUDA > {budget_bytes / 2**30:.1f} GiB.")
    return code, stdout, stderr, {
        "peak_host_gib": round(peak_host / 2**30, 3),
        "peak_cuda_delta_gib": round(peak_cuda / 2**30, 3),
        "peak_combined_gib": round((peak_host + peak_cuda) / 2**30, 3),
        "budget_gib": round(budget_bytes / 2**30, 3),
    }


def _warm_runtime(config: argparse.Namespace) -> None:
    """Run one deterministic inference before accepting requests."""
    if not config.warmup_image:
        return
    cmd = [str(config.binary), "-m", str(config.model), "--mmproj", str(config.mmproj), "--image", str(config.warmup_image), "-p", "Describe this image in one short sentence.", "-c", str(config.context), "-n", "8", "-t", str(config.threads), "-b", str(config.batch), "-ub", str(config.ubatch), "--image-min-tokens", str(config.image_tokens), "--image-max-tokens", str(config.image_tokens), "--no-warmup"]
    cmd += ["-ngl", "all", "--mmproj-offload", "--verbose", "-ctk", "q8_0", "-ctv", "q8_0", "--no-mmap"] if config.runtime == "cuda" else ["-ngl", "0", "--no-mmproj-offload"]
    started = time.perf_counter()
    code, stdout, stderr, memory = _run_bounded(cmd, config.timeout, int(config.memory_budget_gib * 2**30), config.runtime == "cuda")
    if code or not stdout.strip():
        raise RuntimeError(f"Warmup failed ({code}): {stderr[-1200:]}")
    RUNTIME_STATS.update({"warmup_seconds": round(time.perf_counter() - started, 3), "warmup_memory": memory})

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
<div class="hero"><div><div class="badge">EDGE-PROXY · CUDA · ≤12 GiB GATE</div><h1>MiniCPM‑V 4.6 Q4</h1><p class="sub">Workflow camera và hội thoại hands-free giữ như bản gốc; lõi nhìn/suy luận được thay bằng MiniCPM‑V nhỏ nhất chạy CUDA với profile giới hạn kiểu edge.</p></div></div>
<div class="grid"><section class="card"><video id="cam" autoplay playsinline muted></video><canvas id="canvas"></canvas><img id="snapshot" alt="Frame vừa phân tích"><div class="row"><button id="start">📷 Bật camera</button><button id="flip">↺ Đổi camera</button><label><button id="pick">⬆️ Chọn ảnh</button><input id="file" type="file" accept="image/*" hidden></label></div><p class="small">Không stream video liên tục. Ảnh chỉ được gửi khi bấm Hỏi.</p></section>
<section class="card"><div class="metrics"><div class="metric"><b>1.61 GB</b><span>Q4 + projector</span></div><div class="metric"><b>≤12 GiB</b><span>RAM+CUDA gate</span></div><div class="metric"><b>1</b><span>session / queue</span></div></div><div class="row"><select id="lang"><option value="vi">Trả lời tiếng Việt</option><option value="en">Answer in English</option></select></div><div class="row"><input id="q" value="Bạn đang thấy gì?" aria-label="Câu hỏi"><button class="primary" id="ask">Hỏi model</button></div><div class="row"><button id="listen">🎙 Hội thoại hands-free</button><button id="stopListen">■ Dừng nghe</button></div><div class="status" id="status">Sẵn sàng.</div><div id="answer">Kết quả sẽ hiện ở đây.</div><p class="warn">CUDA H100 chạy profile giới hạn giống edge để đo chức năng và memory. Đây không phải bằng chứng tốc độ, điện năng hay operator placement của QCS8550 HTP.</p></section></div>
</main><script>
const cam=document.querySelector('#cam'),canvas=document.querySelector('#canvas'),snap=document.querySelector('#snapshot'),statusEl=document.querySelector('#status'),answer=document.querySelector('#answer');
let stream=null,facing='environment',picked=null;
async function camera(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:facing},width:{ideal:1280},height:{ideal:720}},audio:false});cam.srcObject=stream;await cam.play();picked=null;cam.style.display='block';snap.style.display='none';statusEl.textContent='Camera đã sẵn sàng.'}
document.querySelector('#start').onclick=()=>camera().catch(e=>statusEl.textContent='Không mở được camera: '+e.message);
document.querySelector('#flip').onclick=()=>{facing=facing==='environment'?'user':'environment';camera().catch(e=>statusEl.textContent=e.message)};
document.querySelector('#pick').onclick=()=>document.querySelector('#file').click();
document.querySelector('#file').onchange=e=>{const f=e.target.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{const im=new Image();im.onload=()=>{const scale=Math.min(1,224/Math.max(im.width,im.height));canvas.width=Math.max(1,Math.round(im.width*scale));canvas.height=Math.max(1,Math.round(im.height*scale));canvas.getContext('2d').drawImage(im,0,0,canvas.width,canvas.height);picked=canvas.toDataURL('image/jpeg',.82);snap.src=picked;snap.style.display='block';cam.style.display='none';statusEl.textContent='Ảnh đã chọn · resize ≤224 px cho profile 64 token.'};im.src=r.result};r.readAsDataURL(f)};
function frame(){if(picked)return picked;if(!stream||cam.readyState<2)throw new Error('Hãy bật camera hoặc chọn một ảnh trước.');const max=224,scale=Math.min(1,max/Math.max(cam.videoWidth,cam.videoHeight));canvas.width=Math.max(1,Math.round(cam.videoWidth*scale));canvas.height=Math.max(1,Math.round(cam.videoHeight*scale));canvas.getContext('2d').drawImage(cam,0,0,canvas.width,canvas.height);const data=canvas.toDataURL('image/jpeg',.82);snap.src=data;snap.style.display='block';cam.style.display='none';setTimeout(()=>{if(!picked){cam.style.display='block';snap.style.display='none'}},1400);return data}
let recognition=null,handsFree=false,processing=false;
function speak(text){if(!handsFree||!('speechSynthesis'in window))return Promise.resolve();return new Promise(resolve=>{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang=document.querySelector('#lang').value==='vi'?'vi-VN':'en-US';u.rate=1.04;u.onend=u.onerror=()=>resolve();speechSynthesis.speak(u)})}
async function ask(){const btn=document.querySelector('#ask');if(processing)return;try{const image=frame(),question=document.querySelector('#q').value.trim();if(!question)throw new Error('Bạn chưa nhập câu hỏi.');processing=true;recognition?.abort();btn.disabled=true;answer.textContent='';const started=performance.now();statusEl.textContent='MiniCPM‑V GPU đang nhìn frame mới nhất…';const timer=setInterval(()=>statusEl.textContent=`Đang suy luận… ${((performance.now()-started)/1000).toFixed(1)} giây`,250);try{const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,question,language:document.querySelector('#lang').value})});const out=await res.json();if(!res.ok)throw new Error(out.error||'Inference failed');answer.textContent=out.answer;statusEl.textContent=`Hoàn tất ${out.wall_seconds.toFixed(2)} giây · ${out.runtime} · ${out.profile}`;await speak(out.answer)}finally{clearInterval(timer)}}catch(e){statusEl.textContent='Lỗi: '+e.message}finally{processing=false;btn.disabled=false;if(handsFree)setTimeout(()=>{try{recognition?.start()}catch(_){}},350)}}
document.querySelector('#ask').onclick=ask;
document.querySelector('#listen').onclick=async()=>{const SR=window.SpeechRecognition||window.webkitSpeechRecognition;if(!SR){statusEl.textContent='Chrome/Edge này không hỗ trợ nhận dạng giọng nói; vẫn có thể nhập text.';return}if(!stream)await camera();handsFree=true;recognition=new SR();recognition.lang=document.querySelector('#lang').value==='vi'?'vi-VN':'en-US';recognition.continuous=false;recognition.interimResults=true;recognition.onresult=e=>{let text='';for(let i=e.resultIndex;i<e.results.length;i++)text+=e.results[i][0].transcript;if(text)document.querySelector('#q').value=text;if(e.results[e.results.length-1].isFinal)ask()};recognition.onend=()=>{if(handsFree&&!processing)setTimeout(()=>{try{recognition.start()}catch(_){}},350)};recognition.onerror=e=>{if(e.error!=='aborted'&&e.error!=='no-speech')statusEl.textContent='Mic: '+e.error};recognition.start();statusEl.textContent='Đang nghe liên tục… hãy đặt câu hỏi.'};
document.querySelector('#stopListen').onclick=()=>{handsFree=false;recognition?.abort();recognition=null;speechSynthesis?.cancel();statusEl.textContent='Đã dừng nghe.'};
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
            self._send(HTTPStatus.OK, _json_bytes({"ok": True, "model": "MiniCPM-V-4.6-Q4_0", "runtime": cfg.runtime, "profile": "qcs8550-like-12g", "busy": INFER_LOCK.locked(), "model_exists": cfg.model.is_file(), "projector_exists": cfg.mmproj.is_file(), **RUNTIME_STATS}), "application/json; charset=utf-8")
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
                cmd = [str(cfg.binary), "-m", str(cfg.model), "--mmproj", str(cfg.mmproj), "--image", path, "-p", prompt, "-c", str(cfg.context), "-n", str(cfg.tokens), "-t", str(cfg.threads), "-b", str(cfg.batch), "-ub", str(cfg.ubatch), "--image-min-tokens", str(cfg.image_tokens), "--image-max-tokens", str(cfg.image_tokens), "--no-warmup"]
                if cfg.runtime == "cuda":
                    cmd += ["-ngl", "all", "--mmproj-offload", "--verbose"]
                else:
                    cmd += ["-ngl", "0", "--no-mmproj-offload"]
                if cfg.runtime == "cuda":
                    cmd += ["-ctk", "q8_0", "-ctv", "q8_0", "--no-mmap"]
                code, stdout, stderr, memory = _run_bounded(
                    cmd, cfg.timeout, int(cfg.memory_budget_gib * 2**30), cfg.runtime == "cuda"
                )
                answer = ANSI_ESCAPE.sub("", stdout).strip()
                if code != 0 or not answer:
                    tail = ANSI_ESCAPE.sub("", stderr)[-1200:].strip()
                    raise RuntimeError(f"Model exit {code}: {tail}")
                result = {"answer": answer, "wall_seconds": time.perf_counter() - started, "model": "MiniCPM-V-4.6 Q4_0", "runtime": cfg.runtime.upper(), "profile": "QCS8550-like ≤12 GiB", "memory": memory}
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
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--image-tokens", type=int, default=64)
    parser.add_argument("--runtime", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--memory-budget-gib", type=float, default=12.0)
    parser.add_argument("--warmup-image", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    for name in ("binary", "model", "mmproj"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name} not found: {path}")
    if args.warmup_image and not args.warmup_image.is_file():
        parser.error(f"--warmup-image not found: {args.warmup_image}")
    return args


def main() -> None:
    args = parse_args()
    _warm_runtime(args)
    server = DemoServer((args.host, args.port), args)
    print(f"MiniCPM-V 4.6 edge demo: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
