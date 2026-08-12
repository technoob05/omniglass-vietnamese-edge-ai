const TARGET_RATE = 16000;
const ASR_CHUNK_SAMPLES = 3200;
const MAX_HISTORY_MESSAGES = 10;
// Isolated /vi feature flag. Native /omni never loads this asset.
const VI_EARLY_TTS_ENABLED = new URLSearchParams(location.search).get('early_tts') !== '0';
const SYSTEM_PROMPT = 'Bạn là trợ lý thị giác cho người Việt. Trả lời bằng tiếng Việt, ngắn gọn, chính xác và chỉ dựa vào ảnh mới nhất. Nếu không chắc, hãy nói rõ.';

const el = {
  video: document.querySelector('#camera'), status: document.querySelector('#status'),
  caption: document.querySelector('#caption'), conversation: document.querySelector('#conversation'),
  partial: document.querySelector('#partial'), start: document.querySelector('#start'),
  stop: document.querySelector('#stop'), voice: document.querySelector('#voice'),
};

function wsUrl(path) {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}${path}`;
}

function bytesToBase64(bytes) {
  let binary = '';
  const stride = 0x8000;
  for (let i = 0; i < bytes.length; i += stride) {
    binary += String.fromCharCode(...bytes.subarray(i, i + stride));
  }
  return btoa(binary);
}

function pcm16Base64(samples) {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
  }
  return bytesToBase64(new Uint8Array(pcm.buffer));
}

class StreamingResampler {
  constructor(sourceRate, targetRate) {
    this.ratio = sourceRate / targetRate;
    this.pending = new Float32Array();
    this.position = 0;
  }
  process(input) {
    const data = new Float32Array(this.pending.length + input.length);
    data.set(this.pending); data.set(input, this.pending.length);
    const output = [];
    while (this.position + 1 < data.length) {
      const left = Math.floor(this.position), frac = this.position - left;
      output.push(data[left] * (1 - frac) + data[left + 1] * frac);
      this.position += this.ratio;
    }
    const consumed = Math.floor(this.position);
    this.pending = data.slice(consumed);
    this.position -= consumed;
    return Float32Array.from(output);
  }
}

class SentenceBuffer {
  constructor() { this.pending = ''; }

  push(delta, flush = false) {
    this.pending += String(delta || '');
    const ready = [];
    // Whitespace stays in the pending suffix, so every model delta is
    // consumed exactly once and no word is repeated between TTS requests.
    // Wait for whitespace after punctuation instead of treating the current
    // chunk end as final. This lets a following delta contribute a closing
    // quote and avoids prematurely splitting decimal fragments.
    const boundary = /[.!?…;]+(?:["'”’)\]]*)?(?=\s)|\n+/g;
    let match, cut = 0;
    while ((match = boundary.exec(this.pending)) !== null) {
      const end = match.index + match[0].length;
      const sentence = this.pending.slice(cut, end).trim().replace(/\s+/g, ' ');
      if (sentence) ready.push(sentence);
      cut = end;
    }
    this.pending = this.pending.slice(cut);
    if (flush) {
      const tail = this.pending.trim().replace(/\s+/g, ' ');
      if (tail) ready.push(tail);
      this.pending = '';
    }
    return ready;
  }
}

class VietnameseAssistant {
  constructor() {
    this.state = 'stopped';
    this.runEpoch = 0;
    this.turnEpoch = 0;
    this.mediaStream = null;
    this.audioContext = null;
    this.worklet = null;
    this.resampler = null;
    this.pendingPcm = new Float32Array();
    this.asrSocket = null;
    this.chatSocket = null;
    this.sequence = 0;
    this.history = [];
    this.consumedFinalIds = new Set();
    this.ttsAbort = null;
    this.ttsSources = new Set();
    this.ttsAudio = null;
    this.earlySpeech = null;
    this.bargeInMs = 0;
  }

  setState(state, caption) {
    this.state = state;
    const names = {stopped:'Đã dừng', connecting:'Đang kết nối', listening:'Đang nghe', speech:'Đang nghe bạn nói', thinking:'Đang nhìn và suy nghĩ', speaking:'Đang trả lời', error:'Có lỗi'};
    el.status.textContent = names[state] || state;
    el.status.classList.toggle('live', !['stopped', 'error'].includes(state));
    if (caption) el.caption.textContent = caption;
  }

  message(role, text) {
    const item = document.createElement('div');
    item.className = `message ${role}`;
    item.textContent = text;
    el.conversation.appendChild(item);
    el.conversation.scrollTop = el.conversation.scrollHeight;
    return item;
  }

  async start() {
    await this.stop(false);
    const run = ++this.runEpoch;
    this.setState('connecting', 'Đang mở camera và micro…');
    el.start.disabled = true; el.stop.disabled = false;
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}},
        audio: {echoCancellation:true, noiseSuppression:true, autoGainControl:true, channelCount:1},
      });
      if (run !== this.runEpoch) return;
      el.video.srcObject = this.mediaStream;
      await el.video.play();
      await Promise.all([this.connectAsr(run), this.startAudio(run)]);
    } catch (error) {
      if (run !== this.runEpoch) return;
      this.setState('error', `Không thể bắt đầu: ${error.message}`);
      this.message('system', `Lỗi khởi động: ${error.message}`);
      el.start.disabled = false; el.stop.disabled = true;
    }
  }

  async stop(increment = true) {
    if (increment) this.runEpoch += 1;
    this.turnEpoch += 1;
    this.cancelTurn();
    if (this.asrSocket?.readyState === WebSocket.OPEN) {
      try { this.asrSocket.send(JSON.stringify({type:'audio.end'})); } catch (_) {}
    }
    try { this.asrSocket?.close(); } catch (_) {}
    this.asrSocket = null;
    try { this.worklet?.disconnect(); } catch (_) {}
    this.worklet = null;
    if (this.audioContext) await this.audioContext.close().catch(() => {});
    this.audioContext = null;
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.mediaStream = null; el.video.srcObject = null;
    this.pendingPcm = new Float32Array();
    this.setState('stopped', 'Bấm Bắt đầu để hội thoại.');
    el.start.disabled = false; el.stop.disabled = true; el.partial.textContent = '';
  }

  cancelTurn() {
    if (this.earlySpeech) this.earlySpeech.cancelled = true;
    this.earlySpeech = null;
    if (this.ttsAbort) this.ttsAbort.abort();
    this.ttsAbort = null;
    for (const source of this.ttsSources) { try { source.stop(); } catch (_) {} }
    this.ttsSources.clear();
    if (this.ttsAudio) { try { this.ttsAudio.pause(); this.ttsAudio.currentTime = 0; } catch (_) {} }
    this.ttsAudio = null;
    try { this.chatSocket?.close(); } catch (_) {}
    this.chatSocket = null;
  }

  resetCaptureBuffer() {
    this.pendingPcm = new Float32Array();
    if (this.audioContext) this.resampler = new StreamingResampler(this.audioContext.sampleRate, TARGET_RATE);
  }

  bargeIn() {
    if (!['thinking', 'speaking'].includes(this.state)) return;
    this.turnEpoch += 1;
    this.cancelTurn();
    this.bargeInMs = 0;
    this.resetCaptureBuffer();
    this.setState('listening', 'Đã ngắt câu trả lời. Tôi đang nghe câu mới…');
    this.message('system', '↩ Bạn đã ngắt lời trợ lý.');
  }

  connectAsr(run) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl('/v1/asr/vi?min_silence_ms=500&speech_pad_ms=300&partial_interval_ms=1600'));
      this.asrSocket = socket;
      let ready = false;
      socket.onopen = () => {};
      socket.onerror = () => { if (!ready) reject(new Error('Không kết nối được PhoWhisper')); };
      socket.onclose = () => {
        if (!ready) reject(new Error('PhoWhisper đóng trước khi sẵn sàng'));
        if (run === this.runEpoch && this.state !== 'stopped') {
          this.setState('error', 'Kết nối nhận dạng giọng nói đã đóng.');
        }
      };
      socket.onmessage = event => {
        if (run !== this.runEpoch) return;
        let data;
        try { data = JSON.parse(event.data); }
        catch (_) { this.message('system', 'ASR gửi dữ liệu không hợp lệ.'); return; }
        if (data.type === 'asr.ready') {
          ready = true; this.setState('listening', 'Hãy hỏi bằng tiếng Việt.'); resolve();
        } else if (data.type === 'asr.speech_start' && this.state === 'listening') {
          this.setState('speech', 'Tôi đang nghe…');
        } else if (data.type === 'asr.partial' && ['listening', 'speech'].includes(this.state)) {
          el.partial.textContent = data.text || '';
        } else if (data.type === 'asr.final') {
          void this.consumeFinal(data, run);
        } else if (data.type === 'asr.error') {
          this.message('system', `ASR: ${data.detail || data.code}`);
        }
      };
    });
  }

  async startAudio(run) {
    const Context = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new Context({latencyHint:'interactive'});
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    const module = `class P extends AudioWorkletProcessor {process(i){const c=i[0]?.[0];if(c)this.port.postMessage(c.slice());return true}}registerProcessor('vi-pcm',P)`;
    const url = URL.createObjectURL(new Blob([module], {type:'text/javascript'}));
    await this.audioContext.audioWorklet.addModule(url); URL.revokeObjectURL(url);
    if (run !== this.runEpoch) return;
    this.resampler = new StreamingResampler(this.audioContext.sampleRate, TARGET_RATE);
    this.worklet = new AudioWorkletNode(this.audioContext, 'vi-pcm');
    const mute = this.audioContext.createGain(); mute.gain.value = 0;
    source.connect(this.worklet); this.worklet.connect(mute); mute.connect(this.audioContext.destination);
    this.worklet.port.onmessage = event => this.onAudio(event.data, run);
    await this.audioContext.resume();
  }

  onAudio(input, run) {
    if (run !== this.runEpoch) return;
    const rms = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0) / Math.max(1, input.length));
    if (['thinking', 'speaking'].includes(this.state)) {
      const blockMs = input.length / this.audioContext.sampleRate * 1000;
      this.bargeInMs = rms > 0.075 ? this.bargeInMs + blockMs : Math.max(0, this.bargeInMs - blockMs * 2);
      if (this.bargeInMs >= 280) this.bargeIn();
      return;
    }
    this.bargeInMs = 0;
    if (!['listening', 'speech'].includes(this.state) || this.asrSocket?.readyState !== WebSocket.OPEN) return;
    const audio = this.resampler.process(input);
    const combined = new Float32Array(this.pendingPcm.length + audio.length);
    combined.set(this.pendingPcm); combined.set(audio, this.pendingPcm.length);
    this.pendingPcm = combined;
    while (this.pendingPcm.length >= ASR_CHUNK_SAMPLES) {
      const chunk = this.pendingPcm.slice(0, ASR_CHUNK_SAMPLES);
      this.pendingPcm = this.pendingPcm.slice(ASR_CHUNK_SAMPLES);
      this.asrSocket.send(JSON.stringify({
        type:'audio.chunk', sequence:this.sequence++, timestamp_ms:performance.timeOrigin + performance.now(),
        sample_rate:TARGET_RATE, channels:1, encoding:'pcm_s16le', pcm_s16le_base64:pcm16Base64(chunk),
      }));
    }
  }

  async consumeFinal(event, run) {
    const finalId = String(event.final_id || '');
    if (!event.immutable || !finalId || this.consumedFinalIds.has(finalId)) return;
    this.consumedFinalIds.add(finalId);
    const transcript = String(event.text || '').trim();
    if (!transcript || run !== this.runEpoch || !['listening', 'speech'].includes(this.state)) return;
    const turn = ++this.turnEpoch;
    this.resetCaptureBuffer();
    el.partial.textContent = '';
    this.message('user', transcript);
    this.setState('thinking', 'Đang chụp ảnh mới sau khi model sẵn sàng…');
    const earlySpeech = VI_EARLY_TTS_ENABLED ? this.beginEarlySpeech(run, turn) : null;
    try {
      const answer = await this.askVlm(
        transcript, run, turn,
        delta => { if (earlySpeech) this.queueEarlySpeech(earlySpeech, delta); },
      );
      if (run !== this.runEpoch || turn !== this.turnEpoch) return;
      this.history.push({role:'user', content:transcript}, {role:'assistant', content:answer});
      this.history = this.history.slice(-MAX_HISTORY_MESSAGES);
      this.message('assistant', answer);
      if (earlySpeech) {
        await this.finishEarlySpeech(earlySpeech);
      } else {
        this.setState('speaking', 'Đang phát câu trả lời tiếng Việt…');
        await this.speak(answer, run, turn);
      }
      if (run === this.runEpoch && turn === this.turnEpoch) {
        this.resetCaptureBuffer();
        this.setState('listening', 'Hãy hỏi câu tiếp theo.');
      }
    } catch (error) {
      if (run !== this.runEpoch || turn !== this.turnEpoch || error.name === 'AbortError') return;
      this.cancelTurn();
      this.message('system', `Không xử lý được lượt này: ${error.message}`);
      this.setState('listening', 'Bạn có thể hỏi lại.');
    }
  }

  captureFrame() {
    if (el.video.readyState < 2 || !el.video.videoWidth) throw new Error('Camera chưa có frame');
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1280 / el.video.videoWidth);
    canvas.width = Math.round(el.video.videoWidth * scale); canvas.height = Math.round(el.video.videoHeight * scale);
    canvas.getContext('2d').drawImage(el.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
  }

  async askVlm(transcript, run, turn, onDelta = null) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try { return await this.chatAttempt(transcript, run, turn, onDelta); }
      catch (error) {
        const retryable403 = error.preInput && (error.status === 403 || error.code === 1006 || /403/.test(error.message));
        if (attempt === 0 && retryable403 && run === this.runEpoch && turn === this.turnEpoch) {
          await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 200));
          continue;
        }
        throw error;
      }
    }
    throw new Error('Chat retry exhausted');
  }

  chatAttempt(transcript, run, turn, onDelta = null) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl('/v1/realtime?mode=chat'));
      this.chatSocket = socket;
      let initSent = false, inputSent = false, settled = false, answer = '';
      const fail = (message, code = 0, status = 0) => {
        if (settled) return;
        settled = true;
        const error = new Error(message); error.code = code; error.status = status; error.preInput = !inputSent;
        try { socket.close(); } catch (_) {}
        reject(error);
      };
      const sendInit = () => {
        if (initSent || socket.readyState !== WebSocket.OPEN) return;
        initSent = true; socket.send(JSON.stringify({type:'session.init', payload:{}}));
      };
      socket.onopen = () => setTimeout(sendInit, 100);
      socket.onerror = () => { if (!settled) fail('Chat WebSocket failed before input (possible 403)', 1006, 403); };
      socket.onclose = event => {
        if (!settled) fail(`Chat closed ${event.code}: ${event.reason || 'no reason'}`, event.code, /403/.test(event.reason) ? 403 : 0);
      };
      socket.onmessage = event => {
        try {
          if (run !== this.runEpoch || turn !== this.turnEpoch) { socket.close(); return; }
          const data = JSON.parse(event.data), type = data.type, kind = data.kind;
          if (type === 'session.queue_done' || type === 'queue_done') sendInit();
          else if (type === 'session.created') {
            // Freshness invariant: capture only after a worker session exists.
            const frame = this.captureFrame();
            const messages = [
              {role:'system', content:SYSTEM_PROMPT}, ...this.history.slice(-MAX_HISTORY_MESSAGES),
              {role:'user', content:[{type:'image', data:frame}, {type:'text', text:transcript}]},
            ];
            socket.send(JSON.stringify({type:'input.append', input:{
              input_id:`${turn}-${crypto.randomUUID()}`, messages, streaming:true,
              generation:{max_new_tokens:64, do_sample:false, length_penalty:1.0},
              image:{max_slice_nums:1}, tts:{enabled:false}, use_tts_template:false,
              omni_mode:false, enable_thinking:false,
            }}));
            inputSent = true;
            this.setState('thinking', 'Đã chụp frame mới, model đang trả lời…');
          } else if (type === 'response.output.delta' && kind === 'text') {
            const delta = data.text || '';
            answer += delta;
            if (delta && onDelta) onDelta(delta);
          }
          else if (type === 'response.done') {
            if (!answer.trim() && data.text) {
              answer = data.text;
              if (onDelta) onDelta(data.text);
            }
            settled = true; socket.send(JSON.stringify({type:'session.close', reason:'turn_done'}));
            socket.close(); this.chatSocket = null;
            answer.trim() ? resolve(answer.trim()) : reject(new Error('Model không trả về chữ'));
          } else if (type === 'error') fail(data.error?.message || JSON.stringify(data.error || data));
        } catch (error) {
          fail(`Chat response error: ${error.message}`);
        }
      };
    });
  }

  beginEarlySpeech(run, turn) {
    const controller = new AbortController();
    const speech = {
      run, turn, controller, buffer:new SentenceBuffer(), queue:Promise.resolve(),
      segments:[], error:null, cancelled:false, closed:false,
    };
    this.ttsAbort = controller;
    this.earlySpeech = speech;
    return speech;
  }

  speechTurnActive(speech) {
    return !speech.cancelled && !speech.controller.signal.aborted
      && speech.run === this.runEpoch && speech.turn === this.turnEpoch;
  }

  queueEarlySpeech(speech, delta, flush = false) {
    if (!this.speechTurnActive(speech) || speech.closed) return;
    for (const segment of speech.buffer.push(delta, flush)) {
      speech.segments.push(segment);
      const task = speech.queue.then(async () => {
        if (speech.error) return;
        if (!this.speechTurnActive(speech)) throw new DOMException('Cancelled', 'AbortError');
        this.setState('speaking', 'Đang phát sớm câu trả lời tiếng Việt…');
        await this.speakText(segment, speech.controller.signal, speech.run, speech.turn);
      });
      // Attach a handler immediately: speech can fail while later VLM deltas
      // are still arriving, and must never become an unhandled rejection.
      speech.queue = task.catch(error => { if (!speech.error) speech.error = error; });
    }
  }

  async finishEarlySpeech(speech) {
    if (!this.speechTurnActive(speech)) throw new DOMException('Cancelled', 'AbortError');
    this.queueEarlySpeech(speech, '', true);
    speech.closed = true;
    await speech.queue;
    if (speech.error) throw speech.error;
    if (!speech.segments.length) throw new Error('Không có văn bản cho TTS');
    if (!this.speechTurnActive(speech)) throw new DOMException('Cancelled', 'AbortError');
    if (this.earlySpeech === speech) this.earlySpeech = null;
    if (this.ttsAbort === speech.controller) this.ttsAbort = null;
  }

  async speak(text, run, turn) {
    const controller = new AbortController();
    this.ttsAbort = controller;
    try {
      await this.speakText(text, controller.signal, run, turn);
    } finally { if (this.ttsAbort === controller) this.ttsAbort = null; }
  }

  async speakText(text, signal, run, turn) {
    if (signal.aborted || run !== this.runEpoch || turn !== this.turnEpoch) {
      throw new DOMException('Cancelled', 'AbortError');
    }
    let chunks = 0;
    try {
      const response = await fetch('/api/tts/vi/stream', {
        method:'POST', headers:{'content-type':'application/json'},
        body:JSON.stringify({text, voice:el.voice?.value || 'Trúc Ly', style:'tu_nhien'}), signal,
      });
      if (!response.ok || !response.body) throw new Error(`VieNeu HTTP ${response.status}`);
      const reader = response.body.getReader(), decoder = new TextDecoder(), ended = [];
      let pending = '', sampleRate = 48000, nextStart = 0, doneSeen = false;
      const consume = async line => {
        if (!line.trim()) return;
        const event = JSON.parse(line);
        if (event.type === 'meta') sampleRate = Number(event.sample_rate) || sampleRate;
        else if (event.type === 'audio') {
          if (run !== this.runEpoch || turn !== this.turnEpoch) throw new DOMException('Cancelled', 'AbortError');
          const raw = atob(event.pcm_s16le_base64), values = new Float32Array(raw.length / 2), view = new DataView(new ArrayBuffer(raw.length));
          for (let i=0;i<raw.length;i++) view.setUint8(i,raw.charCodeAt(i));
          for (let i=0;i<values.length;i++) values[i]=view.getInt16(i*2,true)/32768;
          const context = this.audioContext, buffer=context.createBuffer(1,values.length,sampleRate); buffer.copyToChannel(values,0);
          const source=context.createBufferSource(); source.buffer=buffer; source.connect(context.destination); this.ttsSources.add(source);
          const startAt=Math.max(nextStart,context.currentTime+.035); nextStart=startAt+buffer.duration;
          ended.push(new Promise(resolveEnded => {source.onended=()=>{this.ttsSources.delete(source);resolveEnded();};}));
          source.start(startAt); chunks += 1;
        } else if (event.type === 'done') doneSeen = true;
        else if (event.type === 'error') {const error=new Error(event.message||'VieNeu failed');error.partialAudio=chunks>0;throw error;}
      };
      while (true) {
        const {value,done}=await reader.read(); pending+=decoder.decode(value||new Uint8Array(),{stream:!done});
        const lines=pending.split('\n'); pending=lines.pop()||''; for(const line of lines) await consume(line); if(done)break;
      }
      if(pending.trim())await consume(pending);
      if(!chunks||!doneSeen){const error=new Error('VieNeu stream incomplete');error.partialAudio=chunks>0;throw error;}
      await Promise.all(ended); await new Promise(resolveDelay=>setTimeout(resolveDelay,200));
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      if (error.partialAudio) {while(this.ttsSources.size)await new Promise(r=>setTimeout(r,30));await new Promise(r=>setTimeout(r,200));return;}
      if (signal.aborted || run !== this.runEpoch || turn !== this.turnEpoch) throw new DOMException('Cancelled', 'AbortError');
      await this.speakMms(text, signal);
    }
  }

  async speakMms(text, signal) {
    const response=await fetch('/api/tts/vi',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text}),signal});
    if(!response.ok)throw new Error(`MMS HTTP ${response.status}`);
    const result=await response.json(), audio=new Audio(`data:audio/wav;base64,${result.audio_wav_base64}`); this.ttsAudio=audio;
    try {
      await new Promise((resolve,reject)=>{
        const abort=()=>{audio.pause();reject(new DOMException('Cancelled','AbortError'));}; signal.addEventListener('abort',abort,{once:true});
        audio.onended=()=>{signal.removeEventListener('abort',abort);resolve();}; audio.onerror=()=>reject(new Error('Không phát được MMS-TTS'));
        audio.play().catch(reject);
      });
      await new Promise(resolve=>setTimeout(resolve,200));
    } finally { if (this.ttsAudio === audio) this.ttsAudio=null; }
  }
}

const assistant = new VietnameseAssistant();
el.start.addEventListener('click', () => void assistant.start());
el.stop.addEventListener('click', () => void assistant.stop());
window.addEventListener('beforeunload', () => void assistant.stop());
window.__omniglassViAssistant = assistant;
