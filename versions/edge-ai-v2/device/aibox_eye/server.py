"""Dependency-free HTTP API and full end-to-end box demo UI."""

from __future__ import annotations

import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .orchestrator import EyeOrchestrator


_HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge AI hỗ trợ thị giác</title><style>
body{font:16px system-ui;margin:0;background:#07111f;color:#e8edf4}main{max-width:1100px;margin:auto;padding:20px}h1{margin:0 0 6px}.grid{display:grid;grid-template-columns:1.35fr 1fr;gap:16px}.card{background:#111d2e;border:1px solid #29405d;border-radius:16px;padding:16px}img{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:10px}.flow{color:#8bd5ff;font-size:14px}.status{color:#73e2a7;white-space:pre-wrap} .answer{font-size:21px;line-height:1.45;min-height:100px}.muted{color:#9fb0c5}.proof{border-left:3px solid #48d597}.timing{border:1px solid #3f6f91;color:#c8f1ff;white-space:pre-wrap;line-height:1.65}.speaker{display:flex;align-items:center;gap:8px;margin:10px 0;color:#c7f5dd}.speaker input{width:auto}button,input{font:inherit;border-radius:10px;padding:11px;border:1px solid #45617f;background:#17283d;color:#fff}button{cursor:pointer;margin:5px 5px 0 0;font-weight:700}.talk{background:#48d597;color:#062619;border:0;width:100%;font-size:19px}.talk.active{background:#ff8e8e;color:#3a0707}input{width:calc(100% - 130px);box-sizing:border-box}pre{font-size:12px;max-height:220px;overflow:auto;background:#091421;padding:10px;border-radius:8px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><main><h1>Edge AI hỗ trợ người mù</h1><p class="flow">Camera /dev/video2 → QNN YOLO + depth / Hexagon HTP → Whisper STT → GenieX Qwen3.5 2B VL → VieNeu TTS → loa ALSA / loa máy tính</p><div class="grid"><section class="card"><h2>Camera box realtime</h2><img src="http://localhost:8080/stream.mjpg" alt="Camera QNN realtime"><p class="muted">Nguồn thật trên Qualcomm box. Detector/depth luôn giữ ưu tiên.</p></section><section class="card"><h2>Hội thoại rảnh tay</h2><p class="muted">Nhấn giữ nút, nói vào microphone của box, thả ra để chạy toàn bộ pipeline.</p><button id="talk" class="talk">Nhấn giữ để nói</button><p id="state" class="status">Đang kiểm tra hệ thống…</p><label class="speaker"><input id="pcSpeaker" type="checkbox" checked> Tự động đọc phản hồi Qwen bằng loa máy tính</label><button id="testPc">🔊 Test loa máy tính</button><p><input id="q" placeholder="Hoặc nhập câu hỏi tiếng Việt"><button id="ask">Hỏi</button><button id="speak">Phát lại trên loa box</button></p><h3>Transcript</h3><div id="transcript" class="muted">—</div><h3>Câu trả lời</h3><div id="answer" class="answer">—</div><h3>Hiệu năng lượt hiện tại</h3><pre id="timing" class="timing">Chưa có phép đo. Hãy hỏi một câu để benchmark.</pre><h3>Input thật gửi vào Qwen</h3><pre id="facts" class="proof">Chưa có lượt VLM.</pre><h3>Telemetry realtime</h3><pre id="telemetry">—</pre></section></div></main><script>
const talk=document.getElementById('talk'),state=document.getElementById('state'),answer=document.getElementById('answer'),transcript=document.getElementById('transcript'),telemetry=document.getElementById('telemetry'),facts=document.getElementById('facts'),timing=document.getElementById('timing');let recording=false,busy=false,turnStartedAt=0,lastHealth=null;
async function json(url,opt){const r=await fetch(url,opt);const j=await r.json();if(!r.ok)throw Error(j.error||j.detail||'request failed');return j}
async function health(){try{const h=await json('/health');lastHealth=h;const t=h.resources?.npu_temperature_c;state.textContent=`${h.mode} · camera=${h.scene?.camera_ok?'OK':'OFF'} · detector=${Number(h.resources?.detector_fps||0).toFixed(1)} FPS · NPU=${Number(t||0).toFixed(1)}°C · VLM=${h.components?.vlm?.admitted?'sẵn sàng':'đang làm mát (câu hỏi sẽ xếp hàng)'}`;telemetry.textContent=JSON.stringify({scene:h.scene,resources:h.resources,components:h.components},null,2)}catch(e){state.textContent='Lỗi health: '+e.message}}
async function start(){if(recording||busy)return;try{await json('/push-to-talk/start',{method:'POST'});recording=true;talk.classList.add('active');talk.textContent='Đang nghe… thả để xử lý';state.textContent='Đang ghi âm microphone trên box…'}catch(e){state.textContent='Không bắt đầu được: '+e.message}}
let pcAudioContext=null,pcNextTime=0,pcTtsController=null;
function pcSpeakSystem(text){if(!('speechSynthesis'in window)||!text)return false;window.speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='vi-VN';u.rate=.95;const voices=window.speechSynthesis.getVoices();u.voice=voices.find(v=>/^vi([-_]|$)/i.test(v.lang))||voices.find(v=>/vietnam/i.test(v.name))||null;window.speechSynthesis.speak(u);return true}
function schedulePcm(base64Pcm,sampleRate){const raw=atob(base64Pcm),pcm=new Int16Array(raw.length/2);for(let i=0;i<pcm.length;i++)pcm[i]=(raw.charCodeAt(i*2)|(raw.charCodeAt(i*2+1)<<8))<<16>>16;const audio=pcAudioContext.createBuffer(1,pcm.length,sampleRate),out=audio.getChannelData(0);for(let i=0;i<pcm.length;i++)out[i]=pcm[i]/32768;const source=pcAudioContext.createBufferSource();source.buffer=audio;source.playbackRate.value=1.5;source.connect(pcAudioContext.destination);const now=pcAudioContext.currentTime,at=Math.max(now+.04,pcNextTime);source.start(at);pcNextTime=at+audio.duration/1.5;return {duration:audio.duration,startsIn:at-now}}
function baseMetrics(r){return [`Qwen token đầu: ${r.first_token_ms==null?'—':(r.first_token_ms/1000).toFixed(2)+' s'}`,`Qwen hoàn tất: ${r.latency_ms==null?'—':(r.latency_ms/1000).toFixed(2)+' s'}`,`Chờ nhiệt/tài nguyên: ${r.thermal_wait_ms==null?'—':Math.max(0,r.thermal_wait_ms/1000).toFixed(2)+' s'}`,`Detector realtime: ${Number(lastHealth?.resources?.detector_fps||0).toFixed(1)} FPS`,`VieNeu: ${lastHealth?.components?.tts?.voice||'—'} · 48 kHz · tốc độ 1.5x`]}
async function pcSpeak(text,r={}){if(!text)return false;const ttsStarted=performance.now(),turnStart=turnStartedAt||ttsStarted;let firstAudioAt=0,firstStartsIn=0,audioSeconds=0,chunks=0;try{pcTtsController?.abort();pcTtsController=new AbortController();pcAudioContext=pcAudioContext||new AudioContext({sampleRate:48000});await pcAudioContext.resume();pcNextTime=pcAudioContext.currentTime+.35;const response=await fetch('/tts/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal:pcTtsController.signal});if(!response.ok||!response.body)throw Error('VieNeu HTTP '+response.status);const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';state.textContent='VieNeu tiếng Việt đang truyền âm thanh về loa máy tính…';while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop()||'';for(const line of lines){if(!line.trim())continue;const event=JSON.parse(line);if(event.type==='audio'){const scheduled=schedulePcm(event.pcm_s16le_base64,event.sample_rate);if(!firstAudioAt){firstAudioAt=performance.now();firstStartsIn=scheduled.startsIn}audioSeconds+=scheduled.duration;chunks++}if(event.type==='error')throw Error(event.message)}if(done)break}const finished=performance.now(),firstTts=(firstAudioAt-ttsStarted)/1000,buttonToSound=(firstAudioAt-turnStart)/1000+firstStartsIn,playSeconds=audioSeconds/1.5,rtf=(finished-ttsStarted)/1000/audioSeconds,finishAfter=(finished-turnStart)/1000+Math.max(0,pcNextTime-pcAudioContext.currentTime);timing.textContent=[...baseMetrics(r),`TTS tới cụm âm đầu: ${firstTts.toFixed(2)} s`,`Từ bấm hỏi → bắt đầu nghe: ${buttonToSound.toFixed(2)} s`,`TTS tổng hợp xong: ${((finished-ttsStarted)/1000).toFixed(2)} s · RTF ${rtf.toFixed(2)}`,`Âm thanh: ${audioSeconds.toFixed(2)} s gốc → ${playSeconds.toFixed(2)} s ở 1.5x`,`Số cụm phát: ${chunks} · dự kiến phát xong sau ${finishAfter.toFixed(2)} s`].join('\n');return true}catch(e){if(e.name==='AbortError')return false;state.textContent='VieNeu stream lỗi, dùng giọng Việt hệ thống: '+e.message;return pcSpeakSystem(text)}}
function showTurn(j){const r=j.response||j,text=r.answer_vi||r.error||'—';answer.textContent=text;facts.textContent=JSON.stringify({source:r.source,thermal_wait_ms:r.thermal_wait_ms,first_token_ms:r.first_token_ms,vlm_latency_ms:r.latency_ms,early_tts_text:r.early_tts_text,frame:r.vlm_input,yolo_depth_scene_facts:r.scene_facts,guard:r.resource_guard},null,2);timing.textContent=[...baseMetrics(r),`API nhận phản hồi: ${turnStartedAt?((performance.now()-turnStartedAt)/1000).toFixed(2)+' s':'—'}`].join('\n');if(document.getElementById('pcSpeaker').checked)void pcSpeak(text,r);else state.textContent=r.source==='vlm'?`VLM thật · token đầu ${Number(r.first_token_ms||0).toFixed(0)} ms · hoàn tất ${Number(r.latency_ms||0).toFixed(0)} ms · VieNeu trên box`:'Đã xử lý · TTS trên box'}
async function stop(){if(!recording)return;recording=false;turnStartedAt=performance.now();talk.classList.remove('active');talk.textContent='Đang xử lý…';busy=true;try{const speak_on_box=!document.getElementById('pcSpeaker').checked;const j=await json('/push-to-talk/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({speak_on_box})});transcript.textContent=j.transcript_vi||'—';showTurn(j)}catch(e){state.textContent='Pipeline lỗi: '+e.message}finally{busy=false;talk.textContent='Nhấn giữ để nói'}}
async function ask(){const q=document.getElementById('q').value.trim();if(!q||busy)return;busy=true;turnStartedAt=performance.now();timing.textContent='Đang đo toàn bộ pipeline…';state.textContent='Đang giữ YOLO/depth realtime và chờ khe HTP cho Qwen…';try{const speak_on_box=!document.getElementById('pcSpeaker').checked;const j=await json('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:q,speak_on_box})});transcript.textContent=q;showTurn(j)}catch(e){state.textContent='Pipeline lỗi: '+e.message}finally{busy=false}}
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
                if parsed.path == "/ask":
                    body = self._read_json_body()
                    self._ask(str(body.get("text", "")), body.get("speak_on_box", True) is not False)
                    return
                if parsed.path == "/push-to-talk/start": self.send_json(app.start_push_to_talk()); return
                if parsed.path == "/push-to-talk/stop":
                    body = self._read_optional_json_body()
                    self.send_json(app.stop_push_to_talk(body.get("speak_on_box", True) is not False))
                    return
                if parsed.path == "/tts/speak": self.send_json(app.speak_text(str(self._read_json_body().get("text", "")))); return
                if parsed.path == "/tts/stream": self._tts_stream(str(self._read_json_body().get("text", ""))); return
                if parsed.path == "/demo/reset": self.send_json(app.reset_demo()); return
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except ValueError as error: self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error: self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            except Exception as error: self.send_json({"error": f"internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _ask(self, text: str, speak_on_box: bool = True) -> None:
            try: self.send_json(app.ask(text, speak_on_box=speak_on_box))
            except ValueError as error: self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error: self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)

        def _tts_stream(self, text: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            started = False
            try:
                for sequence, pcm in enumerate(app.stream_tts(text)):
                    event = {
                        "type": "audio", "seq": sequence, "sample_rate": 48000,
                        "pcm_s16le_base64": base64.b64encode(pcm).decode("ascii"),
                    }
                    self.wfile.write((json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    started = True
                if not started:
                    raise RuntimeError("VieNeu returned no streaming audio")
                self.wfile.write(b'{"type":"done"}\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                event = {"type": "error", "message": str(error)}
                try:
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                self.close_connection = True

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16_384: raise ValueError("JSON body must be 1..16384 bytes")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict): raise ValueError("JSON body must be an object")
            return value

        def _read_optional_json_body(self) -> dict[str, Any]:
            if int(self.headers.get("Content-Length", "0")) == 0:
                return {}
            return self._read_json_body()

    return Handler


def make_server(app: EyeOrchestrator, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), handler_factory(app))
    server.daemon_threads = True
    return server
