const TARGET_RATE = 16000;
const ASR_CHUNK_SAMPLES = 3200;
const MAX_HISTORY_MESSAGES = 10;
const PERCEPTION_INTERVAL_MS = 650;
const TTS_PLAYBACK_RATE = 1.5;
const BARGE_USER_RMS = 0.05;
const BARGE_HOLD_MS = 700;
// Isolated /vi feature flag. Native /omni never loads this asset.
const VI_EARLY_TTS_ENABLED = new URLSearchParams(location.search).get('early_tts') !== '0';
// Raw browser RMS cannot reliably distinguish near-field speech from speaker
// echo. Reliable interruption uses Hold-to-talk; acoustic barge-in is an
// explicit experimental opt-in via ?barge_in=1.
const VI_ACOUSTIC_BARGE_IN_ENABLED = new URLSearchParams(location.search).get('barge_in') === '1';
const SYSTEM_PROMPTS = {
  vi:'Bạn là OpenGlass, trợ lý kính thông minh cho người khiếm thị. Bạn nhìn thấy góc nhìn của người dùng và nghe câu hỏi của họ. Bạn có bốn năng lực chính: tìm đồ vật, đọc nguyên văn chữ nhìn thấy, mô tả cảnh trong ba đến năm câu, và chỉ hướng ngắn trong nhà dựa trên cửa, lối đi hoặc hành lang thực sự nhìn thấy. Bạn nhận nhiều ảnh theo thứ tự cũ đến mới; frame cuối là hiện tại. Chỉ lên tiếng khi người dùng hỏi hoặc giao nhiệm vụ. Trả lời trực tiếp, ngắn gọn, không kể quá trình suy luận. Chỉ nói điều xác nhận được từ đầu vào hiện tại; không nhìn rõ thì nói không nhìn rõ, tuyệt đối không bịa.',
  en:'You are OpenGlass, a smart-glasses visual assistant for blind and low-vision users. You see the wearer’s first-person view and hear their question. You have four primary capabilities: find objects, read visible text verbatim, describe a scene in three to five sentences, and give short indoor directions using only visible doors, paths, and corridors. You receive images from oldest to newest; the final frame is current. Speak only after the user asks a question or gives a task. Answer directly and concisely without narrating your reasoning. State only what the current input confirms; if something is unclear, say so and never invent it.'
};
const WORKFLOW_PROMPTS = {
  auto: {vi:'Tự chọn kỹ năng phù hợp cho từng câu hỏi.', en:'Select the appropriate skill for each request.'},
  find_object: {
    vi:'[Chế độ tìm đồ vật] Xác định mục tiêu người dùng muốn tìm. Nếu thấy rõ, trả lời một câu gồm tên vật, vị trí tương đối, màu hoặc đặc điểm nhận biết và vật ở gần. Nếu chỉ nghi ngờ, nói rõ là nghi ngờ; nếu không thấy, nói chưa thấy và không mô tả vật không liên quan. Không bảo người dùng cứ đi tới hoặc với tay lấy.',
    en:'[Find object mode] Identify the object the user wants. If it is clearly visible, answer in one sentence with its relative position, color or identifying feature, and nearby landmark. If uncertain, say it may be there; if absent, say it is not visible and do not describe unrelated objects. Never tell the user to simply walk toward or reach for it.'
  },
  read_text: {
    vi:'[Chế độ đọc chữ] Đọc đúng chữ, số, đơn vị và ký hiệu thực sự nhìn thấy theo thứ tự từ trên xuống dưới, trái sang phải. Không tự hoàn thành phần mờ; yêu cầu đưa camera gần hơn nếu cần.',
    en:'[Read text mode] Read only the visible words, numbers, units, and symbols in top-to-bottom, left-to-right order. Never complete blurred text; ask the user to move the camera closer when needed.'
  },
  indoor_direction: {
    vi:'[Chế độ chỉ đường trong nhà] Chỉ dựa trên lối đi, cửa và hành lang nhìn thấy để đưa hướng ngắn, rủi ro thấp và khoảng cách ước lượng. Depth là đơn mắt chưa hiệu chuẩn. Không tuyên bố đây là điều hướng được chứng nhận và yêu cầu người dùng tự xác nhận môi trường thật.',
    en:'[Indoor guidance mode] Use only visible paths, doors, and corridors to give short, low-risk directions and an estimated distance. Monocular depth is uncalibrated. Never claim certified navigation or guaranteed safety; ask the user to verify the physical environment.'
  },
  sound_watch: {
    vi:'[Chế độ theo dõi âm thanh] Chỉ xác nhận âm thanh có bằng chứng trong transcript hiện tại. Profile chat này chưa truyền raw ambient audio vào VLM, vì vậy phải nói rõ giới hạn và không được giả vờ đã nghe thấy âm thanh môi trường.',
    en:'[Sound watch mode] Confirm a sound only when the current transcript contains evidence for it. This chat profile does not send raw ambient audio to the VLM, so disclose that limitation and never pretend to hear environmental audio.'
  },
  describe_scene: {
    vi:'[Chế độ mô tả cảnh] Mô tả trong một đến ba câu các vật chính, quan hệ vị trí và trạng thái rõ ràng; ưu tiên mặt đường, cửa, chữ, người và vật cản. Không suy đoán danh tính, ý định hoặc nguy hiểm từ hình mờ.',
    en:'[Describe scene mode] In one to three sentences, describe the main objects, spatial relationships, and clearly visible states; prioritize paths, doors, text, people, and obstacles. Do not infer identity, intent, or danger from an unclear image.'
  },
  visual_qa: {vi:'[Chế độ hỏi đáp thị giác] Trả lời đúng câu hỏi dựa trên ảnh mới nhất và ngữ cảnh có cấu trúc.', en:'[Visual question answering] Answer the exact question using the newest image and structured context.'}
};
const WORKFLOW_LABELS = {
  auto:{vi:'Tự động · đủ kỹ năng',en:'Auto · all skills'}, find_object:{vi:'Tìm đồ vật',en:'Find object'},
  read_text:{vi:'Đọc chữ',en:'Read text'}, indoor_direction:{vi:'Chỉ đường trong nhà',en:'Indoor guidance'},
  sound_watch:{vi:'Theo dõi âm thanh · thử nghiệm',en:'Sound watch · experimental'},
  describe_scene:{vi:'Mô tả cảnh',en:'Describe scene'}, visual_qa:{vi:'Hỏi đáp thị giác',en:'Visual Q&A'}
};

function englishLocation(locationText) {
  const map = {'bên trái':'on the left','ở giữa':'in the center','bên phải':'on the right','phía trên':'above','phía dưới':'below'};
  return map[locationText] || locationText || 'at an unknown position';
}

function classifyWorkflow(text) {
  const value = String(text || '').toLocaleLowerCase('vi');
  if (/(đọc|chữ|văn bản|biển báo|read|text|sign|label|number|ocr)/u.test(value)) return 'read_text';
  if (/(âm thanh|tiếng gì|nghe thấy|sound|noise|hear|listen for)/u.test(value)) return 'sound_watch';
  if (/(qua đường|đi hướng|lối đi|cửa nào|hành lang|đi tới|cross|direction|which way|corridor|door|navigate)/u.test(value)) return 'indoor_direction';
  if (/(mô tả|xung quanh|cảnh|trước mặt|describe|around me|scene|surroundings|what is ahead)/u.test(value)) return 'describe_scene';
  if (/(ở đâu|nằm đâu|tìm|có thấy|where is|find|locate|look for|do you see)/u.test(value)) return 'find_object';
  return 'visual_qa';
}

function normalizeAssistiveTranscript(text) {
  return String(text || '')
    .normalize('NFC')
    .replace(/\b(?:máy|mấy|miếng|mái)\s+(?:ngon|ngón|nắm|ngắm)\s+(?:tay|cay)\b/giu, 'mấy ngón tay')
    .replace(/\bđang\s+nhớ\s+(?=mấy ngón tay\b)/giu, 'đang giơ ')
    .replace(/\b(?:công|tông|tóm)\s+tất\b/giu, 'tóm tắt')
    .replace(/\bđang\s+nhận\s+thấy\b/giu, 'đang nhìn thấy')
    .replace(/\s+/g, ' ')
    .trim();
}

const el = {
  video: document.querySelector('#camera'), status: document.querySelector('#status'),
  caption: document.querySelector('#caption'), conversation: document.querySelector('#conversation'),
  partial: document.querySelector('#partial'), start: document.querySelector('#start'),
  stop: document.querySelector('#stop'), pushToTalk:document.querySelector('#pushToTalk'), voice: document.querySelector('#voice'), language:document.querySelector('#language'), workflow:document.querySelector('#workflow'),
  boxes: document.querySelector('#boxes'), perceptionStatus: document.querySelector('#perceptionStatus'),
  memory: document.querySelector('#memory'), watchTarget: document.querySelector('#watchTarget'),
  resetMemory: document.querySelector('#resetMemory'),
  safetyPanel: document.querySelector('#safetyPanel'), safetyIcon: document.querySelector('#safetyIcon'),
  safetyMessage: document.querySelector('#safetyMessage'), safetyMeta: document.querySelector('#safetyMeta'),
  safetyAlerts: document.querySelector('#safetyAlerts'), detectorLatency: document.querySelector('#detectorLatency'),
  detectorFps: document.querySelector('#detectorFps'), depthState: document.querySelector('#depthState'),
  memoryCount: document.querySelector('#memoryCount'),
  brandTitle:document.querySelector('#brandTitle'), workflowBadge:document.querySelector('#workflowBadge'), ttsBadge:document.querySelector('#ttsBadge'),
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
    this.assistantGain = null;
    this.safetyGain = null;
    this.safetySource = null;
    this.safetyAbort = null;
    this.safetySpeaking = false;
    this.safetyQueue = Promise.resolve();
    this.conversationOwnsAudio = false;
    this.safetyEpoch = 0;
    this.pendingSafetyAlert = null;
    this.earlySpeech = null;
    this.bargeInMs = 0;
    this.safetyBargeInMs = 0;
    this.pushToTalkActive = false;
    this.micGuardUntilMs = 0;
    this.perceptionTimer = 0;
    this.perceptionBusy = false;
    this.perception = null;
    this.perceptionFailures = 0;
    this.perceptionSession = `vi-${crypto.randomUUID()}`;
    this.visualFrames = [];
    this.lastVisualFrameAt = 0;
    this.workflowMemory = {find_target:'', destination:'', sound_target:''};
    this.lastWatchTarget = '';
    this.lastWatchStatus = null;
    this.localizeUi();
    void this.probeReadiness();
  }

  async probeReadiness() {
    try {
      const [asrResponse, perceptionResponse] = await Promise.all([
        fetch('/api/asr/vi/health', {cache:'no-store'}),
        fetch('/api/perception/vi/health', {cache:'no-store'}),
      ]);
      if (!asrResponse.ok || !perceptionResponse.ok) throw new Error('health check failed');
      const [asr, perception] = await Promise.all([asrResponse.json(), perceptionResponse.json()]);
      if (!asr.ok || !perception.ok) throw new Error('service not ready');
      if (this.state !== 'stopped') return;
      el.status.textContent = this.isEnglish() ? 'Ready' : 'Sẵn sàng';
      el.status.classList.add('live');
      el.caption.textContent = this.isEnglish() ? 'OpenGlass is ready. Press Start to enable camera and microphone.' : 'OmniGlass đã sẵn sàng. Bấm Bắt đầu để cấp camera, micro và chạy pipeline.';
      el.perceptionStatus.textContent = this.isEnglish() ? `${perception.detector} loaded on ${perception.device} · depth ready · waiting for camera.` : `${perception.detector} đã nạp trên ${perception.device} · depth sẵn sàng · chờ camera.`;
    } catch (_) {
      if (this.state !== 'stopped') return;
      el.status.textContent = 'Chưa sẵn sàng';
      el.status.classList.remove('live');
      el.perceptionStatus.textContent = 'Không đọc được readiness của perception; thử tải lại trang.';
    }
  }

  setState(state, caption) {
    this.state = state;
    const names = el.language?.value === 'en'
      ? {stopped:'Stopped', connecting:'Connecting', listening:'Listening', speech:'Hearing you', thinking:'Looking and thinking', speaking:'Answering', error:'Error'}
      : {stopped:'Đã dừng', connecting:'Đang kết nối', listening:'Đang nghe', speech:'Đang nghe bạn nói', thinking:'Đang nhìn và suy nghĩ', speaking:'Đang trả lời', error:'Có lỗi'};
    el.status.textContent = names[state] || state;
    el.status.classList.toggle('live', !['stopped', 'error'].includes(state));
    if (caption) el.caption.textContent = caption;
  }

  isEnglish() { return el.language?.value === 'en'; }

  localizeUi() {
    const language = this.isEnglish() ? 'en' : 'vi';
    document.documentElement.lang = language;
    for (const item of document.querySelectorAll('[data-vi][data-en]')) item.textContent = item.dataset[language];
    for (const option of el.workflow?.options || []) option.textContent = option.dataset[language] || option.textContent;
    el.brandTitle.textContent = this.isEnglish() ? 'OpenGlass' : 'OmniGlass';
    el.voice.disabled = this.isEnglish();
    el.voice.parentElement.style.display = this.isEnglish() ? 'none' : 'flex';
    el.ttsBadge.textContent = this.isEnglish() ? 'English system TTS · 1.5×' : 'VieNeu TTS · 1.5×';
    document.querySelector('#conversationTitle').textContent = this.isEnglish() ? 'Context-aware conversation' : 'Hội thoại theo ngữ cảnh';
    for (const button of document.querySelectorAll('.sample')) {
      button.textContent = (this.isEnglish() ? button.dataset.promptEn : button.dataset.promptVi).split(/[?.]/)[0];
    }
    this.renderWorkflowBadge();
  }

  selectedWorkflow() { return el.workflow?.value || 'auto'; }

  workflowFor(transcript) {
    const selected = this.selectedWorkflow();
    return selected === 'auto' ? classifyWorkflow(transcript) : selected;
  }

  renderWorkflowBadge(active = this.selectedWorkflow()) {
    const language = this.isEnglish() ? 'en' : 'vi';
    const label = WORKFLOW_LABELS[active]?.[language] || active;
    el.workflowBadge.textContent = `${label} · VLM`;
  }

  systemPrompt(transcript) {
    const language = this.isEnglish() ? 'en' : 'vi';
    const workflow = this.workflowFor(transcript);
    const safety = language === 'en'
      ? 'For distance, report an available depth estimate as uncalibrated. For street crossing, describe visible traffic, signals, motion, and occlusion, but never declare that it is safe to cross; tell the user to verify the signal or seek nearby assistance.'
      : 'Khi hỏi khoảng cách, nêu depth nếu có và nói rõ chưa hiệu chuẩn. Khi hỏi qua đường, mô tả xe, đèn, chuyển động và vùng che khuất nhưng không bao giờ khẳng định an toàn để qua; yêu cầu người dùng xác nhận tín hiệu hoặc nhờ hỗ trợ tại chỗ.';
    const memory = JSON.stringify(this.workflowMemory);
    this.renderWorkflowBadge(workflow);
    return `${SYSTEM_PROMPTS[language]}\n\n${WORKFLOW_PROMPTS[workflow][language]}\n\n${safety}\n\nSession workflow state: ${memory}`;
  }

  rememberWorkflowGoal(transcript) {
    const workflow = this.workflowFor(transcript);
    const text = String(transcript || '').trim().replace(/[.?!]+$/u, '');
    let match;
    if (workflow === 'find_object') {
      match = text.match(/(?:tìm|tìm giúp|ở đâu|có thấy)(?: cho tôi)?\s+(.+)|(?:find|locate|look for|where is|do you see)(?: my| the| a| an)?\s+(.+)/iu);
      const target = (match?.[1] || match?.[2] || '').trim().slice(0,80);
      if (target) {
        this.workflowMemory.find_target = target;
        el.watchTarget.value = target;
      }
    } else if (workflow === 'indoor_direction') {
      this.workflowMemory.destination = text.slice(0,120);
    } else if (workflow === 'sound_watch') {
      this.workflowMemory.sound_target = text.slice(0,120);
    }
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
    this.setState('connecting', this.isEnglish() ? 'Opening camera and microphone…' : 'Đang mở camera và micro…');
    el.start.disabled = true; el.stop.disabled = false;
    el.pushToTalk.disabled = false;
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode:{ideal:'environment'}, width:{ideal:1280}, height:{ideal:720}},
        audio: {echoCancellation:true, noiseSuppression:true, autoGainControl:true, channelCount:1},
      });
      if (run !== this.runEpoch) return;
      el.video.srcObject = this.mediaStream;
      await el.video.play();
      await this.resetPerception(false);
      void this.perceptionLoop(run);
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
    this.assistantGain = null; this.safetyGain = null;
    if (this.safetyAbort) this.safetyAbort.abort();
    this.safetyAbort = null;
    if (this.safetySource) { try { this.safetySource.stop(); } catch (_) {} }
    this.safetySource = null; this.safetySpeaking = false; this.safetyQueue = Promise.resolve();
    this.safetyBargeInMs = 0;
    this.pushToTalkActive = false; el.pushToTalk.classList.remove('active'); el.pushToTalk.disabled = true;
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.mediaStream = null; el.video.srcObject = null;
    clearTimeout(this.perceptionTimer); this.perceptionTimer = 0; this.perceptionBusy = false;
    this.drawDetections([]);
    this.pendingPcm = new Float32Array();
    this.visualFrames = []; this.lastVisualFrameAt = 0;
    this.history = []; this.consumedFinalIds.clear();
    this.workflowMemory = {find_target:'', destination:'', sound_target:''};
    this.lastWatchTarget = ''; this.lastWatchStatus = null;
    this.conversationOwnsAudio = false; this.pendingSafetyAlert = null;
    this.setState('stopped', this.isEnglish() ? 'Press Start to begin.' : 'Bấm Bắt đầu để hội thoại.');
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
    this.micGuardUntilMs = 0;
    this.resetCaptureBuffer();
    this.setState('listening', this.isEnglish() ? 'Answer interrupted. I am listening…' : 'Đã ngắt câu trả lời. Tôi đang nghe câu mới…');
    this.message('system', this.isEnglish() ? '↩ You interrupted the assistant.' : '↩ Bạn đã ngắt lời trợ lý.');
  }

  beginPushToTalk() {
    if (this.pushToTalkActive || ['stopped','error','connecting'].includes(this.state)) return;
    const interruptedAnswer = ['thinking','speaking'].includes(this.state);
    const interruptedAlert = this.safetySpeaking;
    if (interruptedAnswer) { this.turnEpoch += 1; this.cancelTurn(); }
    this.pushToTalkActive = true;
    el.pushToTalk.classList.add('active');
    this.acquireConversationAudio();
    this.bargeInMs = 0; this.safetyBargeInMs = 0; this.micGuardUntilMs = 0;
    this.resetCaptureBuffer();
    if (interruptedAnswer || interruptedAlert) {
      this.message('system', this.isEnglish() ? '↩ Hold-to-talk interrupted audio output.' : '↩ Giữ-để-nói đã ngắt âm thanh đang phát.');
    }
    this.setState('speech', this.isEnglish() ? 'Hold and speak. Release when finished.' : 'Giữ nút và nói. Thả ra khi nói xong.');
  }

  endPushToTalk() {
    if (!this.pushToTalkActive) return;
    this.pushToTalkActive = false;
    el.pushToTalk.classList.remove('active');
    if (this.state === 'speech') {
      this.setState('speech', this.isEnglish() ? 'Finishing speech recognition…' : 'Đang hoàn tất nhận dạng giọng nói…');
    }
  }

  connectAsr(run) {
    return new Promise((resolve, reject) => {
      const language = el.language?.value === 'en' ? 'en' : 'vi';
      const socket = new WebSocket(wsUrl(`/v1/asr/vi?language=${language}&vad_threshold=0.45&min_silence_ms=650&speech_pad_ms=500&partial_interval_ms=1600`));
      this.asrSocket = socket;
      let ready = false;
      socket.onopen = () => {};
      socket.onerror = () => { if (!ready) reject(new Error('Không kết nối được PhoWhisper')); };
      socket.onclose = () => {
        if (socket !== this.asrSocket) return;
        if (!ready) reject(new Error('PhoWhisper đóng trước khi sẵn sàng'));
        if (run === this.runEpoch && this.state !== 'stopped') {
          this.setState('error', this.isEnglish() ? 'The speech-recognition connection closed.' : 'Kết nối nhận dạng giọng nói đã đóng.');
        }
      };
      socket.onmessage = event => {
        if (run !== this.runEpoch) return;
        let data;
        try { data = JSON.parse(event.data); }
        catch (_) { this.message('system', 'ASR gửi dữ liệu không hợp lệ.'); return; }
        if (data.type === 'asr.ready') {
          ready = true; this.setState('listening', language === 'en' ? 'Ask me about your surroundings.' : 'Hãy hỏi về xung quanh.'); resolve();
        } else if (data.type === 'asr.speech_start' && this.state === 'listening') {
          this.acquireConversationAudio();
          this.setState('speech', this.isEnglish() ? 'I am listening…' : 'Tôi đang nghe…');
        } else if (data.type === 'asr.partial' && ['listening', 'speech'].includes(this.state)) {
          el.partial.textContent = data.text || '';
        } else if (data.type === 'asr.final') {
          void this.consumeFinal(data, run);
        } else if (data.type === 'asr.no_speech') {
          this.releaseConversationAudio(run);
        } else if (data.type === 'asr.error') {
          this.message('system', `ASR: ${data.detail || data.code}`);
        }
      };
    });
  }

  async startAudio(run) {
    const Context = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new Context({latencyHint:'interactive'});
    this.assistantGain = this.audioContext.createGain(); this.assistantGain.gain.value = 1;
    this.safetyGain = this.audioContext.createGain(); this.safetyGain.gain.value = 1;
    this.assistantGain.connect(this.audioContext.destination);
    this.safetyGain.connect(this.audioContext.destination);
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
    const blockMs = input.length / this.audioContext.sampleRate * 1000;
    const rms = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0) / Math.max(1, input.length));
    if (this.safetySpeaking) {
      if (!VI_ACOUSTIC_BARGE_IN_ENABLED && !this.pushToTalkActive) { this.resetCaptureBuffer(); return; }
      this.safetyBargeInMs = rms >= BARGE_USER_RMS
        ? this.safetyBargeInMs + blockMs
        : Math.max(0, this.safetyBargeInMs - blockMs * 2);
      if (this.safetyBargeInMs < BARGE_HOLD_MS) { this.resetCaptureBuffer(); return; }
      this.interruptSafetyForUserSpeech();
    } else {
      this.safetyBargeInMs = 0;
    }
    if (performance.now() < this.micGuardUntilMs) {
      this.resetCaptureBuffer();
      return;
    }
    if (['thinking', 'speaking'].includes(this.state)) {
      if (!VI_ACOUSTIC_BARGE_IN_ENABLED) return;
      this.bargeInMs = rms >= BARGE_USER_RMS
        ? this.bargeInMs + blockMs
        : Math.max(0, this.bargeInMs - blockMs * 2);
      if (this.bargeInMs < BARGE_HOLD_MS) return;
      this.bargeIn();
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
    const rawTranscript = String(event.text || '').trim();
    const transcript = el.language?.value === 'en' ? rawTranscript : normalizeAssistiveTranscript(rawTranscript);
    if (!transcript || run !== this.runEpoch || !['listening', 'speech'].includes(this.state)) return;
    const turn = ++this.turnEpoch;
    this.acquireConversationAudio();
    this.resetCaptureBuffer();
    el.partial.textContent = '';
    if (transcript !== rawTranscript) {
      this.message('system', `ASR hiệu chỉnh theo từ vựng trợ lý: “${rawTranscript}” → “${transcript}”`);
    }
    this.rememberWorkflowGoal(transcript);
    this.message('user', transcript);
    this.setState('thinking', this.isEnglish() ? 'Capturing a fresh frame after the model is ready…' : 'Đang chụp ảnh mới sau khi model sẵn sàng…');
    const earlySpeech = VI_EARLY_TTS_ENABLED ? this.beginEarlySpeech(run, turn) : null;
    try {
      const routedAnswer = await this.routePerceptionCommand(transcript);
      const answer = routedAnswer || await this.askVlm(
          transcript, run, turn,
          delta => { if (earlySpeech) this.queueEarlySpeech(earlySpeech, delta); },
        );
      if (routedAnswer && earlySpeech) this.queueEarlySpeech(earlySpeech, routedAnswer, true);
      if (run !== this.runEpoch || turn !== this.turnEpoch) return;
      this.history.push({role:'user', content:transcript}, {role:'assistant', content:answer});
      this.history = this.history.slice(-MAX_HISTORY_MESSAGES);
      this.message('assistant', answer);
      if (earlySpeech) {
        await this.finishEarlySpeech(earlySpeech);
      } else {
        this.setState('speaking', this.isEnglish() ? 'Speaking the OpenGlass answer…' : 'Đang phát câu trả lời tiếng Việt…');
        await this.speak(answer, run, turn);
      }
      if (run === this.runEpoch && turn === this.turnEpoch) {
        this.resetCaptureBuffer();
        this.micGuardUntilMs = performance.now() + 700;
        this.setState('listening', this.isEnglish() ? 'Ask the next question.' : 'Hãy hỏi câu tiếp theo.');
        this.releaseConversationAudio(run);
      }
    } catch (error) {
      if (run !== this.runEpoch || turn !== this.turnEpoch || error.name === 'AbortError') return;
      this.cancelTurn();
      this.message('system', this.isEnglish() ? `This turn failed: ${error.message}` : `Không xử lý được lượt này: ${error.message}`);
      this.setState('listening', this.isEnglish() ? 'Please ask again.' : 'Bạn có thể hỏi lại.');
      this.releaseConversationAudio(run);
    }
  }

  async routePerceptionCommand(transcript) {
    const text = transcript.toLocaleLowerCase('vi').trim();
    if (/x[oó]a .*(visual )?(memory|bộ nhớ)|reset .*(memory|bộ nhớ)|clear (visual )?memory/u.test(text)) {
      await this.resetPerception(true);
      return el.language?.value === 'en' ? 'Visual memory for this session has been cleared.' : 'Đã xóa visual memory của phiên hiện tại.';
    }
    if (/dừng theo dõi|thôi theo dõi|stop tracking/u.test(text)) {
      el.watchTarget.value = '';
      return el.language?.value === 'en' ? 'Object tracking has stopped.' : 'Đã dừng theo dõi đối tượng.';
    }
    const watch = text.match(/(?:theo dõi liên tục|canh bằng camera)(?: cho tôi)?\s+(.+)|(?:track continuously|visually track)\s+(.+)/u);
    if (watch) {
      el.watchTarget.value = (watch[1] || watch[2]).replace(/[.?!]+$/u, '').trim().slice(0,80);
      return el.language?.value === 'en' ? `Now tracking ${el.watchTarget.value}.` : `Đã bắt đầu theo dõi ${el.watchTarget.value}.`;
    }
    // Content questions are VLM-first. YOLO/depth/memory become grounded
    // context instead of bypassing the visual reasoner.
    return null;
  }

  submitText(question) {
    const text = String(question || '').trim();
    if (!text || !['listening','speech'].includes(this.state)) return;
    void this.consumeFinal({immutable:true, final_id:`typed-${crypto.randomUUID()}`, text}, this.runEpoch);
  }

  async switchLanguage() {
    const english = this.isEnglish();
    this.history = []; this.visualFrames = []; this.lastVisualFrameAt = 0;
    this.workflowMemory = {find_target:'', destination:'', sound_target:''};
    this.localizeUi();
    el.caption.textContent = english ? 'English mode. Ask about the scene around you.' : 'Chế độ tiếng Việt. Hãy hỏi về khung cảnh xung quanh.';
    if (!this.mediaStream || this.state === 'stopped') {
      this.setState('stopped', english ? 'Press Start to begin the OpenGlass workflow.' : 'Bấm Bắt đầu để chạy workflow OpenGlass.');
      this.renderSafety(this.perception?.safety || {state:'clear'});
      return;
    }
    const old = this.asrSocket; this.asrSocket = null;
    try { old?.close(); } catch (_) {}
    this.resetCaptureBuffer(); this.acquireConversationAudio();
    this.setState('connecting', english ? 'Switching to English…' : 'Đang chuyển sang tiếng Việt…');
    await this.resetPerception(true);
    await this.connectAsr(this.runEpoch);
    this.conversationOwnsAudio = false;
    this.renderSafety(this.perception?.safety || {state:'clear'});
  }

  async switchWorkflow() {
    const language = this.isEnglish() ? 'en' : 'vi';
    this.turnEpoch += 1; this.cancelTurn();
    this.history = []; this.visualFrames = []; this.lastVisualFrameAt = 0;
    this.workflowMemory = {find_target:'', destination:'', sound_target:''};
    this.lastWatchTarget = ''; this.lastWatchStatus = null; el.watchTarget.value = '';
    this.renderWorkflowBadge();
    const label = WORKFLOW_LABELS[this.selectedWorkflow()][language];
    this.message('system', this.isEnglish() ? `OpenGlass workflow changed to ${label}. Session memory was reset.` : `Đã chuyển workflow OpenGlass sang ${label}. Bộ nhớ phiên đã được đặt lại.`);
    if (this.mediaStream && this.state !== 'stopped') {
      await this.resetPerception(true);
      this.resetCaptureBuffer(); this.micGuardUntilMs = performance.now() + 300;
      this.setState('listening', this.isEnglish() ? 'The new workflow is ready.' : 'Workflow mới đã sẵn sàng.');
      this.releaseConversationAudio(this.runEpoch);
    }
  }

  captureFrame(maxWidth = 1280, quality = 0.82) {
    if (el.video.readyState < 2 || !el.video.videoWidth) throw new Error('Camera chưa có frame');
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, maxWidth / el.video.videoWidth);
    canvas.width = Math.round(el.video.videoWidth * scale); canvas.height = Math.round(el.video.videoHeight * scale);
    canvas.getContext('2d').drawImage(el.video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', quality).split(',')[1];
  }

  async perceptionPost(path, payload) {
    const response = await fetch(path, {
      method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`perception HTTP ${response.status}`);
    return response.json();
  }

  async resetPerception(newSession = true) {
    if (newSession) this.perceptionSession = `vi-${crypto.randomUUID()}`;
    this.lastWatchTarget = ''; this.lastWatchStatus = null;
    this.perception = null; this.renderMemory([]); this.drawDetections([]);
    try {
      await this.perceptionPost('/api/perception/vi/reset', {session_id:this.perceptionSession});
      el.perceptionStatus.textContent = 'YOLO/ByteTrack sẵn sàng · depth chạy theo nhịp thấp.';
    } catch (error) {
      el.perceptionStatus.textContent = `Perception chưa sẵn sàng: ${error.message}`;
    }
  }

  async perceptionLoop(run) {
    if (run !== this.runEpoch || !this.mediaStream) return;
    if (!this.perceptionBusy && el.video.readyState >= 2) {
      this.perceptionBusy = true;
      try {
        const data = await this.perceptionPost('/api/perception/vi/frame', {
          session_id:this.perceptionSession,
          image_jpeg_base64:this.captureFrame(640, 0.72),
          confidence:0.35,
          watch_target:el.watchTarget.value || '',
        });
        if (run !== this.runEpoch) return;
        this.perception = data; this.perceptionFailures = 0;
        const now = performance.now();
        if (now - this.lastVisualFrameAt >= 1200) {
          this.visualFrames.push({captured_at_ms:Date.now(), image:this.captureFrame(768,0.76)});
          this.visualFrames = this.visualFrames.slice(-4); this.lastVisualFrameAt = now;
        }
        this.drawDetections(data.detections || []); this.renderMemory(data.memory || []);
        const metrics = data.metrics || {}, watch = data.watch || {};
        this.renderSafety(data.safety || {state:'clear'});
        this.enqueueSafetyAlert(data.safety?.primary_alert);
        this.handleWatchTransition(watch);
        el.detectorLatency.textContent = `${Number(metrics.inference_ms||0).toFixed(1)} ms`;
        el.detectorFps.textContent = `${Number(metrics.detector_fps_capacity||0).toFixed(1)} FPS`;
        el.depthState.textContent = metrics.depth_enabled ? 'ON · advisory' : 'OFF';
        el.memoryCount.textContent = String(Number(metrics.history_size||0));
        const watchText = watch.target ? ` · watch ${watch.target}: ${watch.status}` : '';
        const scene = this.isEnglish()
          ? ((data.detections || []).slice(0,8).map(item => item.label).join(', ') || 'no clear object detected')
          : data.scene_vi;
        const capacity = this.isEnglish() ? 'capacity' : 'năng lực';
        el.perceptionStatus.textContent = `frame ${data.frame_id} · ${scene} · ${Number(metrics.inference_ms||0).toFixed(1)} ms · ${capacity} ${Number(metrics.detector_fps_capacity||0).toFixed(1)} FPS · depth ${metrics.depth_enabled?'ON':'OFF'}${watchText}`;
      } catch (error) {
        this.perceptionFailures += 1;
        if (this.perceptionFailures <= 3 || this.perceptionFailures % 10 === 0) {
          el.perceptionStatus.textContent = `Perception tạm gián đoạn (${this.perceptionFailures}): ${error.message}`;
        }
      } finally {
        this.perceptionBusy = false;
      }
    }
    if (run === this.runEpoch && this.mediaStream) {
      this.perceptionTimer = setTimeout(() => void this.perceptionLoop(run), PERCEPTION_INTERVAL_MS);
    }
  }

  handleWatchTransition(watch) {
    const target = String(watch?.target || '');
    const status = watch?.status || null;
    const becameVisible = target && target === this.lastWatchTarget && status === 'visible'
      && this.lastWatchStatus && this.lastWatchStatus !== 'visible';
    this.lastWatchTarget = target; this.lastWatchStatus = status;
    if (!becameVisible || this.conversationOwnsAudio || this.state !== 'listening') return;
    this.enqueueSafetyAlert({
      should_announce:true,
      message_vi:`Đã tìm thấy ${target} trong khung hình. Hãy hỏi tôi vị trí chính xác nếu cần.`,
      message_en:`I found ${target} in view. Ask me for its exact position if needed.`,
    });
  }

  renderSafety(safety) {
    const state = ['danger','warning','caution'].includes(safety.state) ? safety.state : 'clear';
    el.safetyPanel.className = `safety-panel ${state}`;
    const alert = safety.primary_alert;
    el.safetyIcon.textContent = state === 'danger' ? '!' : state === 'warning' ? '△' : state === 'caution' ? '•' : '✓';
    el.safetyMessage.textContent = alert
      ? (el.language?.value === 'en' ? (alert.message_en || alert.message_vi) : alert.message_vi)
      : (el.language?.value === 'en' ? 'No nearby obstacle detected in the visible area.' : 'Không có vật cản gần trong vùng quan sát.');
    el.safetyMeta.textContent = alert
      ? (this.isEnglish() ? `Rule ${alert.rule || 'watch'} · ${Math.round(Number(alert.confidence||0)*100)}% · advisory depth · no VLM call` : `Rule ${alert.rule || 'watch'} · ${Math.round(Number(alert.confidence||0)*100)}% · depth chỉ tham khảo · VLM không được gọi`)
      : (this.isEnglish() ? 'The rule engine runs continuously · no VLM call.' : 'Rule engine đang chạy liên tục · VLM không được gọi.');
  }

  enqueueSafetyAlert(alert) {
    if (!alert?.should_announce || !el.safetyAlerts?.checked || !this.audioContext) return;
    if (this.conversationOwnsAudio) { this.pendingSafetyAlert = alert; return; }
    const epoch = this.safetyEpoch;
    this.safetyQueue = this.safetyQueue
      .then(() => {
        if (epoch !== this.safetyEpoch || this.conversationOwnsAudio) return;
        return this.playSafetyAlert(el.language?.value === 'en' ? (alert.message_en || alert.message_vi) : alert.message_vi);
      })
      .catch(error => { if (error.name !== 'AbortError') console.warn('Safety TTS:', error); });
  }

  acquireConversationAudio() {
    this.conversationOwnsAudio = true;
    this.safetyEpoch += 1;
    if (this.safetyAbort) this.safetyAbort.abort();
    if (this.safetySource) { try { this.safetySource.stop(); } catch (_) {} }
    this.safetyAbort = null; this.safetySource = null; this.safetySpeaking = false;
    this.safetyQueue = Promise.resolve();
    if (this.assistantGain && this.audioContext?.state !== 'closed') this.assistantGain.gain.setValueAtTime(1, this.audioContext.currentTime);
  }

  interruptSafetyForUserSpeech() {
    this.safetyBargeInMs = 0;
    this.acquireConversationAudio();
    this.micGuardUntilMs = 0;
    this.resetCaptureBuffer();
    this.setState('speech', this.isEnglish() ? 'Alert interrupted. I am listening…' : 'Đã ngắt cảnh báo. Tôi đang nghe câu hỏi…');
    this.message('system', this.isEnglish() ? '↩ You interrupted the alert to ask OpenGlass.' : '↩ Bạn đã ngắt cảnh báo để hỏi trợ lý.');
  }

  releaseConversationAudio(run) {
    setTimeout(() => {
      if (run !== this.runEpoch || !['listening','speech'].includes(this.state)) return;
      this.conversationOwnsAudio = false;
      const pending = this.pendingSafetyAlert; this.pendingSafetyAlert = null;
      if (pending && el.safetyAlerts?.checked) {
        pending.should_announce = true;
        this.enqueueSafetyAlert(pending);
      }
    }, 800);
  }

  async playSafetyAlert(text) {
    if (!this.audioContext || this.audioContext.state === 'closed' || this.conversationOwnsAudio) return;
    if (this.isEnglish()) return this.playEnglishSafetyAlert(text);
    const controller = new AbortController(); this.safetyAbort = controller; this.safetySpeaking = true;
    const context = this.audioContext;
    let source = null;
    try {
      this.assistantGain?.gain.setTargetAtTime(0.16, context.currentTime, 0.025);
      const response = await fetch('/api/tts/vi', {
        method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({text}), signal:controller.signal,
      });
      if (!response.ok) throw new Error(`Safety TTS HTTP ${response.status}`);
      const payload = await response.json(), raw = atob(payload.audio_wav_base64);
      const bytes = new Uint8Array(raw.length); for (let i=0;i<raw.length;i++) bytes[i]=raw.charCodeAt(i);
      const buffer = await context.decodeAudioData(bytes.buffer.slice(0));
      source = context.createBufferSource(); this.safetySource = source;
      source.buffer = buffer; source.playbackRate.value = TTS_PLAYBACK_RATE; source.connect(this.safetyGain || context.destination);
      await new Promise((resolve, reject) => {
        const abort = () => { try { source.stop(); } catch (_) {} reject(new DOMException('Cancelled','AbortError')); };
        controller.signal.addEventListener('abort', abort, {once:true});
        source.onended = () => { controller.signal.removeEventListener('abort', abort); resolve(); };
        source.start();
      });
    } finally {
      if (this.safetySource === source) this.safetySource = null;
      if (this.safetyAbort === controller) {
        this.safetySpeaking = false;
        this.safetyAbort = null;
      }
      if (this.assistantGain && this.audioContext?.state !== 'closed') this.assistantGain.gain.setTargetAtTime(1, this.audioContext.currentTime, 0.04);
      if (!this.conversationOwnsAudio) {
        this.resetCaptureBuffer();
        this.micGuardUntilMs = performance.now() + 700;
      }
    }
  }

  async playEnglishSafetyAlert(text) {
    if (!('speechSynthesis' in window) || this.conversationOwnsAudio) return;
    const controller = new AbortController(); this.safetyAbort = controller; this.safetySpeaking = true;
    const utterance = new SpeechSynthesisUtterance(text); utterance.lang = 'en-US'; utterance.rate = TTS_PLAYBACK_RATE;
    const voice = speechSynthesis.getVoices().find(item => item.lang?.toLowerCase().startsWith('en'));
    if (voice) utterance.voice = voice;
    try {
      await new Promise((resolve, reject) => {
        const abort = () => { speechSynthesis.cancel(); reject(new DOMException('Cancelled','AbortError')); };
        controller.signal.addEventListener('abort', abort, {once:true});
        utterance.onend = () => { controller.signal.removeEventListener('abort', abort); resolve(); };
        utterance.onerror = event => { controller.signal.removeEventListener('abort', abort); reject(new Error(event.error || 'English alert speech failed')); };
        speechSynthesis.speak(utterance);
      });
    } finally {
      if (this.safetyAbort === controller) { this.safetyAbort = null; this.safetySpeaking = false; }
      if (!this.conversationOwnsAudio) { this.resetCaptureBuffer(); this.micGuardUntilMs = performance.now() + 700; }
    }
  }

  drawDetections(detections) {
    const canvas = el.boxes, rect = el.video.getBoundingClientRect();
    if (!canvas || !rect.width || !rect.height) return;
    canvas.width = Math.round(rect.width); canvas.height = Math.round(rect.height);
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!el.video.videoWidth || !el.video.videoHeight) return;
    const scale = Math.max(canvas.width / el.video.videoWidth, canvas.height / el.video.videoHeight);
    const shownW = el.video.videoWidth * scale, shownH = el.video.videoHeight * scale;
    const ox = (canvas.width - shownW) / 2, oy = (canvas.height - shownH) / 2;
    ctx.font = 'bold 13px system-ui'; ctx.lineWidth = 2;
    for (const item of detections) {
      const [x1,y1,x2,y2] = item.bbox_norm;
      const x = ox + x1 * shownW, y = oy + y1 * shownH, w = (x2-x1)*shownW, h = (y2-y1)*shownH;
      ctx.strokeStyle = '#66f2b5'; ctx.fillStyle = 'rgba(0,0,0,.72)'; ctx.strokeRect(x,y,w,h);
      const depth = item.depth_m == null ? '' : ` ~${Number(item.depth_m).toFixed(1)}m*`;
      const label = `${this.isEnglish() ? item.label : item.label_vi} ${Math.round(item.confidence*100)}%${depth}`;
      const tw = ctx.measureText(label).width + 10; ctx.fillRect(x, Math.max(0,y-22), tw, 22);
      ctx.fillStyle = '#a8ffd5'; ctx.fillText(label, x+5, Math.max(15,y-6));
    }
  }

  renderMemory(rows) {
    if (!rows.length) { el.memory.innerHTML = `<div class="memory-item">${this.isEnglish() ? 'No observations yet.' : 'Chưa có observation.'}</div>`; return; }
    el.memory.replaceChildren(...rows.slice(0,16).map(row => {
      const item = document.createElement('div'); item.className = 'memory-item';
      const age = Number(row.last_seen_ms_ago||0)/1000;
      item.textContent = this.isEnglish()
        ? `${row.label} · ${englishLocation(row.location)} · ${age.toFixed(1)}s ago · ${Math.round(row.confidence*100)}%`
        : `${row.label_vi} · ${row.location} · ${age.toFixed(1)}s trước · ${Math.round(row.confidence*100)}%`;
      return item;
    }));
  }

  perceptionContext() {
    const data = this.perception;
    if (!data) return this.isEnglish() ? 'Realtime perception is unavailable; rely on the fresh image.' : 'Perception realtime chưa có dữ liệu; chỉ dựa vào ảnh mới.';
    const detections = (data.detections || []).slice(0,10).map(item => ({
      label:this.isEnglish() ? item.label : item.label_vi, confidence:item.confidence, location:this.isEnglish() ? englishLocation(item.location) : item.location,
      track_id:item.track_id, depth_m_uncalibrated:item.depth_m,
    }));
    const memory = (data.memory || []).slice(0,12).map(item => ({
      label:this.isEnglish() ? item.label : item.label_vi, last_seen_seconds_ago:Number(item.last_seen_ms_ago||0)/1000,
      location:this.isEnglish() ? englishLocation(item.location) : item.location, confidence:item.confidence,
    }));
    const safety = data.safety ? {state:data.safety.state, hazards:(data.safety.active_hazards||[]).slice(0,5)} : null;
    return `Structured perception context for the current scene (detector can be wrong): ${JSON.stringify({detections,memory,watch:data.watch,safety,depth_calibrated:false,multi_frame_order:'oldest_to_newest; final image is current'})}`;
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
            const priorFrames = this.visualFrames.slice(-3).map(item => ({type:'image',data:item.image}));
            const messages = [
              {role:'system', content:this.systemPrompt(transcript)},
              {role:'system', content:this.perceptionContext()},
              ...this.history.slice(-MAX_HISTORY_MESSAGES),
              {role:'user', content:[...priorFrames,{type:'image', data:frame}, {type:'text', text:transcript}]},
            ];
            socket.send(JSON.stringify({type:'input.append', input:{
              input_id:`${turn}-${crypto.randomUUID()}`, messages, streaming:true,
              generation:{max_new_tokens:64, do_sample:false, length_penalty:1.0},
              image:{max_slice_nums:1}, tts:{enabled:false}, use_tts_template:false,
              omni_mode:false, enable_thinking:false,
            }}));
            inputSent = true;
            this.setState('thinking', this.isEnglish() ? 'Fresh frame captured. OpenGlass is answering…' : 'Đã chụp frame mới, model đang trả lời…');
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
        this.setState('speaking', this.isEnglish() ? 'Streaming the OpenGlass answer…' : 'Đang phát sớm câu trả lời tiếng Việt…');
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
    if (el.language?.value === 'en') return this.speakEnglish(text, signal, run, turn);
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
          const source=context.createBufferSource(); source.buffer=buffer; source.playbackRate.value=TTS_PLAYBACK_RATE; source.connect(this.assistantGain || context.destination); this.ttsSources.add(source);
          const startAt=Math.max(nextStart,context.currentTime+.035); nextStart=startAt+buffer.duration/TTS_PLAYBACK_RATE;
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

  speakEnglish(text, signal, run, turn) {
    return new Promise((resolve, reject) => {
      if (!('speechSynthesis' in window)) { reject(new Error('English system voice is unavailable')); return; }
      const utterance = new SpeechSynthesisUtterance(text); utterance.lang='en-US'; utterance.rate=TTS_PLAYBACK_RATE;
      const voice = speechSynthesis.getVoices().find(item => item.lang?.toLowerCase().startsWith('en'));
      if (voice) utterance.voice=voice;
      const abort=()=>{speechSynthesis.cancel();reject(new DOMException('Cancelled','AbortError'));};
      signal.addEventListener('abort',abort,{once:true});
      utterance.onend=()=>{signal.removeEventListener('abort',abort);resolve();};
      utterance.onerror=event=>{signal.removeEventListener('abort',abort);reject(new Error(event.error||'English speech failed'));};
      if (signal.aborted || run!==this.runEpoch || turn!==this.turnEpoch) return abort();
      speechSynthesis.speak(utterance);
    });
  }

  async speakMms(text, signal) {
    const response=await fetch('/api/tts/vi',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text}),signal});
    if(!response.ok)throw new Error(`MMS HTTP ${response.status}`);
    const result=await response.json(), audio=new Audio(`data:audio/wav;base64,${result.audio_wav_base64}`); audio.playbackRate=TTS_PLAYBACK_RATE; this.ttsAudio=audio;
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
el.pushToTalk.addEventListener('pointerdown', event => {
  event.preventDefault();
  try { el.pushToTalk.setPointerCapture(event.pointerId); } catch (_) {}
  assistant.beginPushToTalk();
});
for (const type of ['pointerup','pointercancel','lostpointercapture']) {
  el.pushToTalk.addEventListener(type, () => assistant.endPushToTalk());
}
el.pushToTalk.addEventListener('keydown', event => {
  if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) { event.preventDefault(); assistant.beginPushToTalk(); }
});
el.pushToTalk.addEventListener('keyup', event => {
  if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); assistant.endPushToTalk(); }
});
el.resetMemory.addEventListener('click', () => void assistant.resetPerception(true));
el.language.addEventListener('change', () => void assistant.switchLanguage());
el.workflow.addEventListener('change', () => void assistant.switchWorkflow());
for (const button of document.querySelectorAll('.sample')) button.addEventListener('click', () => {
  assistant.submitText(el.language.value === 'en' ? button.dataset.promptEn : button.dataset.promptVi);
});
window.addEventListener('resize', () => assistant.drawDetections(assistant.perception?.detections || []));
window.addEventListener('beforeunload', () => void assistant.stop());
window.__omniglassViAssistant = assistant;
