"""Dependency-free HTTP API and full end-to-end box demo UI."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .orchestrator import EyeOrchestrator


_HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge AI hỗ trợ thị giác</title><style>
body{font:16px system-ui;margin:0;background:#07111f;color:#e8edf4}main{max-width:1100px;margin:auto;padding:20px}h1{margin:0 0 6px}.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}.card{background:#111d2e;border:1px solid #29405d;border-radius:16px;padding:16px}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:10px}.flow{color:#8bd5ff;font-size:14px}.status{color:#73e2a7;white-space:pre-wrap} .answer{font-size:21px;line-height:1.45;min-height:100px}.muted{color:#9fb0c5}.proof{border-left:3px solid #48d597}.speaker{display:flex;align-items:center;gap:8px;margin:10px 0;color:#c7f5dd}.speaker input{width:auto}button,input{font:inherit;border-radius:10px;padding:11px;border:1px solid #45617f;background:#17283d;color:#fff}button{cursor:pointer;margin:5px 5px 0 0;font-weight:700}.talk{background:#48d597;color:#062619;border:0;width:100%;font-size:19px}.talk.active{background:#ff8e8e;color:#3a0707}input{width:calc(100% - 130px);box-sizing:border-box}pre{font-size:12px;max-height:220px;overflow:auto;background:#091421;padding:10px;border-radius:8px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><main><h1>Edge AI hỗ trợ người mù</h1><p class="flow">Camera /dev/video2 → QNN YOLO + depth / Hexagon HTP → Whisper STT → GenieX Qwen3.5 2B VL → VieNeu TTS → loa ALSA / loa máy tính</p><div class="grid"><section class="card"><h2>Camera box realtime</h2><img src="http://localhost:8080/stream.mjpg" alt="Camera QNN realtime"><p class="muted">Nguồn thật trên Qualcomm box. Detector/depth luôn giữ ưu tiên.</p></section><section class="card"><h2>Hội thoại rảnh tay</h2><p class="muted">Nhấn giữ nút, nói vào microphone của box, thả ra để chạy toàn bộ pipeline.</p><button id="talk" class="talk">Nhấn giữ để nói</button><p id="state" class="status">Đang kiểm tra hệ thống…</p><label class="speaker"><input id="pcSpeaker" type="checkbox" checked> Tự động đọc phản hồi Qwen bằng loa máy tính</label><button id="testPc">🔊 Test loa máy tính</button><p><input id="q" placeholder="Hoặc nhập câu hỏi tiếng Việt"><button id="ask">Hỏi</button><button id="speak">Phát lại trên loa box</button></p><h3>Transcript</h3><div id="transcript" class="muted">—</div><h3>Câu trả lời</h3><div id="answer" class="answer">—</div><h3>Input thật gửi vào Qwen</h3><pre id="facts" class="proof">Chưa có lượt VLM.</pre><h3>Telemetry realtime</h3><pre id="telemetry">—</pre></section></div></main><script>
const talk=document.getElementById('talk'),state=document.getElementById('state'),answer=document.getElementById('answer'),transcript=document.getElementById('transcript'),telemetry=document.getElementById('telemetry'),facts=document.getElementById('facts');let recording=false,busy=false;
async function json(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||j.detail||'request failed');return j}
async function health(){try{const h=await json('/health');const t=h.resources?.npu_temperature_c;state.textContent=`${h.mode} · camera=${h.scene?.camera_ok?'OK':'OFF'} · detector=${Number(h.resources?.detector_fps||0).toFixed(1)} FPS · NPU=${Number(t||0).toFixed(1)}°C · VLM=${h.components?.vlm?.admitted?'sẵn sàng':'đang làm mát (câu hỏi sẽ xếp hàng)'}`;telemetry.textContent=JSON.stringify({scene:h.scene,resources:h.resources,components:h.components},null,2)}catch(e){state.textContent='Lỗi health: '+e.message}}
async function start(){if(recording||busy)return;try{await json('/push-to-talk/start',{method:'POST'});recording=true;talk.classList.add('active');talk.textContent='Đang nghe… thả để xử lý';state.textContent='Đang ghi âm microphone trên box…'}catch(e){state.textContent='Không bắt đầu được: '+e.message}}
function pcSpeak(text){if(!('speechSynthesis'in window)||!text)return false;window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='vi-VN';u.rate=.95;u.pitch=1;const voices=window.speechSynthesis.getVoices();u.voice=voices.find(v=>/^vi([-_]|$)/i.test(v.lang))||voices.find(v=>/vietnam/i.test(v.name))||null;u.onstart=()=>state.textContent='Đang đọc phản hồi bằng loa máy tính…';u.onerror=e=>state.textContent='Loa máy tính lỗi: '+e.error;window.speechSynthesis.speak(u);return true}
function showTurn(j){const r=j.response||j,text=r.answer_vi||r.error||'—';answer.textContent=text;facts.textContent=JSON.stringify({source:r.source,thermal_wait_ms:r.thermal_wait_ms,first_token_ms:r.first_token_ms,vlm_latency_ms:r.latency_ms,early_tts_text:r.early_tts_text,frame:r.vlm_input,yolo_depth_scene_facts:r.scene_facts,guard:r.resource_guard},null,2);if(document.getElementById('pcSpeaker').checked)pcSpeak(text);else state.textContent=r.source==='vlm'?`VLM thật · token đầu ${Number(r.first_token_ms||0).toFixed(0)} ms · hoàn tất ${Number(r.latency_ms||0).toFixed(0)} ms · VieNeu trên box`:'Đã xử lý · TTS trên box'}
async function stop(){if(!recording)return;recording=false;talk.classList.remove('active');talk.textContent='Đang xử lý…';busy=true;try{const j=await json('/push-to-talk/stop',{method:'POST'});transcript.textContent=j.transcript_vi||'—';showTurn(j)}catch(e){state.textContent='Pipeline lỗi: '+e.message}finally{busy=false;talk.textContent='Nhấn giữ để nói'}}
async function ask(){const q=document.getElementById('q').value.trim();if(!q||busy)return;busy=true;state.textContent='Đang giữ YOLO/depth realtime và chờ khe HTP cho Qwen…';try{const j=await json('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:q})});transcript.textContent=q;showTurn(j)}catch(e){state.textContent='Pipeline lỗi: '+e.message}finally{busy=false}}
async function speak(){const shown=answer.textContent.trim(),typed=document.getElementById('q').value.trim(),text=shown&&shown!=='—'?shown:typed;if(!text)return;try{const j=await json('/tts/speak',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});state.textContent=`Đã xếp VieNeu · giọng ${j.voice} · tốc độ ${j.tempo}x · loa ALSA`}catch(e){state.textContent='TTS lỗi: '+e.message}}
talk.addEventListener('pointerdown',e=>{e.preventDefault();start()});talk.addEventListener('pointerup',e=>{e.preventDefault();stop()});talk.addEventListener('pointerleave',()=>{if(recording)stop()});document.getElementById('ask').onclick=ask;document.getElementById('speak').onclick=speak;document.getElementById('testPc').onclick=()=>pcSpeak('Loa máy tính đã hoạt động. Tôi sẵn sàng hỗ trợ bạn.');window.speechSynthesis?.getVoices();setInterval(health,2000);health();
</script></html>'''.encode("utf-8")


def handler_factory(app: EyeOrchestrator) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIBOX-eye/0.2"

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

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_HTML)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_HTML)
                return
            if parsed.path == "/health": self.send_json(app.health()); return
            if parsed.path == "/scene": self.send_json({"scene": app.latest_scene()}); return
            if parsed.path == "/events": self.send_json({"events": app.events()}); return
            if parsed.path == "/audio/status": self.send_json(app.audio.status.to_dict()); return
            if parsed.path == "/metrics":
                body = app.prometheus_metrics().encode("utf-8")
                self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if parsed.path == "/ask": self._ask(parse_qs(parsed.query).get("text", [""])[-1]); return
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/ask": self._ask(str(self._read_json_body().get("text", ""))); return
                if parsed.path == "/push-to-talk/start": self.send_json(app.start_push_to_talk()); return
                if parsed.path == "/push-to-talk/stop": self.send_json(app.stop_push_to_talk()); return
                if parsed.path == "/tts/speak": self.send_json(app.speak_text(str(self._read_json_body().get("text", "")))); return
                if parsed.path == "/demo/reset": self.send_json(app.reset_demo()); return
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as error: self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error: self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as error: self.send_json({"error": f"internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _ask(self, text: str) -> None:
            try: self.send_json(app.ask(text))
            except ValueError as error: self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error: self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384: raise ValueError("JSON body must be 1..16384 bytes")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict): raise ValueError("JSON body must be an object")
            return value

    return Handler


def make_server(app: EyeOrchestrator, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_factory(app))
    server.daemon_threads = True
    return server
