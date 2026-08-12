# OmniGlass

Local-first visual assistance prototype with one shared perception pass and one continual
memory serving four skills: **See, Remember, Find, and Watch**.

This repository is an integration workspace. The runnable MVP is under `omniglass/`; research
upstreams stay isolated under `upstream/` so their code and licenses are not mixed into the app.
The public repository intentionally excludes upstream source trees, model weights, raw media,
private session traces, and infrastructure addresses; use [`UPSTREAM_LOCK.json`](UPSTREAM_LOCK.json)
and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) to fetch dependencies from their owners.

## What is working now

| Capability | Implementation | Status |
|---|---|---|
| See | YOLO11 detector over sampled video frames | Working on H100/web |
| Remember | Timestamped 2D observation JSON, inspired by DirectMe | Working |
| Find / last-seen | Vietnamese/English rule router + evidence frame | Working |
| Watch | Same-track missing/screen-region event detector | Working, 2D only |
| Promptable mask tracking | Official Meta EdgeTAM checkpoint | Working as an H100 sample |
| Native MiniCPM-o 4.5 Omni | Official browser gateway on H100 | Working; English/Chinese baseline preserved |
| Vietnamese realtime profile | PhoWhisper VAD/ASR + MiniCPM-o vision/chat + VieNeu streaming TTS | Deployed at `/vi`; 3-turn browser E2E, 10-turn protocol soak, and resilience tests passed |
| OpenGlass ESP32 sensor bridge | External research reference | Device adapter not integrated; upstream issues listed below |
| Metric 3D memory | Planned SLAM3R point-map bridge | Not yet metric-calibrated |
| Qualcomm EdgeTAM | Official AI Hub model export path | Requires AI Hub token and physical board validation |
| Qualcomm local VLM | GenieX | Device-phase; GenieX is Snapdragon-only |

The prototype is visual assistance, not a certified navigation or hazard-avoidance system.

Native runtime and Vietnamese/Edge design notes:

- [`NATIVE_OPENGLASS_BASELINE.md`](NATIVE_OPENGLASS_BASELINE.md)
- [`VIETNAMESE_REALTIME_ARCHITECTURE.md`](VIETNAMESE_REALTIME_ARCHITECTURE.md)
- [`VIETNAMESE_FINETUNING_PLAN.md`](VIETNAMESE_FINETUNING_PLAN.md)
- [`MINICPMO_VI_LORA.md`](MINICPMO_VI_LORA.md)
- [`EDGE_DEPLOYMENT.md`](EDGE_DEPLOYMENT.md)
- [`QCS8550_DEPLOYMENT.md`](QCS8550_DEPLOYMENT.md)
- [`manifests/vietnamese_h100_stack.json`](manifests/vietnamese_h100_stack.json)
- [`manifests/vietnamese_adaptation_experiments.json`](manifests/vietnamese_adaptation_experiments.json)

With the local SSH forward active, open `https://127.0.0.1:8006/vi`, accept the
self-signed certificate, allow camera/microphone, then press **Bắt đầu hội
thoại**. This profile is hands-free after the single permission/start gesture.
The upstream English/Chinese Omni page remains at `/omni`.
Sentence-level early TTS is enabled by default. Append `?early_tts=0` for an
instant full-answer-TTS rollback when comparing behavior.

## Architecture

```text
phone rear camera / uploaded video
                |
                v
     shared detector + tracker       EdgeTAM when a user prompts an object
                |                                  |
                +------------------+---------------+
                                   v
                    timestamped visual memory
              label, track, confidence, bbox, frame
                                   |
                     +-------------+-------------+
                     |             |             |
                    See           Find          Watch
                scene summary   last seen   region-change/missing
```

The MVP uses screen-relative locations (`left/center/right`, `upper/lower`). It deliberately
does not report metric distance. The next 3D adapter will take SLAM3R world/local point maps,
compute robust points inside each detected mask/bbox, and feed them into a real DirectMe adapter.
The current flat JSON is not a DirectMe scene graph. SLAM3R scale must be calibrated before
distances can be called meters.

## Quick start

Python 3.10+ and a CUDA PyTorch installation are recommended.

```bash
cd edge-ai/omniglass
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Run tests:

```bash
pytest -q
```

Run the video CLI:

```bash
python scripts/run_video_demo.py samples/indoor_demo.mp4 \
  --device cuda \
  --fps 2 \
  --max-seconds 20 \
  --watch laptop \
  --question "Laptop lần cuối ở đâu?"
```

The command produces:

- `annotated_memory.mp4`: one shared detection/tracking pass;
- `memory.json`: timestamped observations and watch events;
- keyframes used as answer evidence;
- a Vietnamese last-seen answer.

## Web demo

```bash
python -m omniglass.app \
  --host 127.0.0.1 \
  --port 7870 \
  --device cuda \
  --results-dir runs/web
```

Workflow:

1. Select the visible **Camera sau** or **Upload video** tab and provide a 2–20 second clip.
2. Enter a watch target such as `laptop`, `bottle`, or `person`.
3. Click **Phân tích và tạo visual memory**.
4. Inspect the annotated video, evidence frame, object timeline, and memory JSON.
5. Ask `Trước mặt có gì?`, `Laptop lần cuối ở đâu?`, `Có gì thay đổi?`, or
   `Laptop có đổi vùng trong khung hình không?`.

The rear-camera recorder prefers `facingMode=environment` and shows elapsed time. Stop at 20 seconds;
the OpenCV backend enforces a 20-second processing cap without requiring system FFmpeg. This supervised demo uses a temporary
public tunnel without authentication; do not upload private footage.

## Live glasses mode

This mode streams the browser webcam continuously to a persistent YOLO11n + ByteTrack process on
the H100. It drops stale work by using `always_last`; the displayed AI FPS is measured per callback.

```bash
python -m omniglass.live --host 127.0.0.1 --port 7871 --device cuda
```

When the backend is remote, forward it to the computer that owns the webcam:

```bash
ssh -N -L 7871:127.0.0.1:7871 -p PORT root@H100_HOST
```

Open `http://127.0.0.1:7871`, press **Bật Live AI**, and grant webcam permission. Running through
localhost is important because browsers allow webcam capture in a secure context. This live path
uses YOLO/ByteTrack continuously; EdgeTAM remains a prompted video tracker and Qwen VLM is invoked
on demand rather than on every frame.

### OpenGlass Simple: Vietnamese visual conversation first

Use this path before enabling tracking, depth, or continual memory. It follows OpenGlass's
sensing/compute split: the browser owns the webcam and microphone, captures exactly one current
JPEG when an utterance finishes, and sends only that turn to persistent VLM/TTS workers on H100.
The server does not reuse a process-global webcam frame from another browser session.

The reference OpenGlass clone is under `upstream/OpenGlass` at commit
`1ffe701e808cc67aa6462bc6071c61413302f17d`. Its original `glasses_panel.py` is not the entry point
for this demo: a clean local run currently stops first on the optional `pywebview` dependency, and
the upstream README also marks the panel's machine-specific MiniCPM/llama.cpp paths as not yet a
portable one-click setup.

Start the persistent Qwen-VL + Vietnamese MMS-TTS service, then the small Gradio wrapper:

```bash
python scripts/vlm_agent_service.py --host 127.0.0.1 --port 8780
python scripts/openglass_simple_live.py \
  --host 127.0.0.1 --port 7873 \
  --agent-url http://127.0.0.1:8780
```

Open `http://127.0.0.1:7873` through an SSH local forward when the service runs remotely. The UI
has two independent inputs: **Webcam realtime** and **Upload ảnh thử**. Grant webcam access; the
Gradio `Record` control is not required because each question captures the current local preview
through a browser canvas. Press **Bật hội thoại liên tục** once, then ask Vietnamese questions such
as `Trong ảnh đang có gì?` or `Đọc chữ trên hộp giúp tôi`. The exact frame sent for the turn is shown
beside the camera immediately, while Qwen-VL analyzes it and MMS-TTS generates Vietnamese speech.
The browser state machine permits one turn at a time and always returns to listening after audio,
audio failure, or timeout. Say `Dừng nghe` to end the microphone session or `Ngừng nói` to interrupt
audio.

The current H100 demo is a turn-based realtime assistant, not a continuous per-frame VLM. This is
intentional: the webcam preview stays local, network traffic is one JPEG per utterance, and the VLM
cannot build a stale-frame queue. Continuous detection/tracking should run in a separate lightweight
worker; invoke the VLM only on demand or on a debounced scene event. Keep typed tool results
(`frame_id`, capture time, target, confidence, distance method, validity) separate from the language
model response so the same coordinator can later call H100 workers or QAIRT/QNN workers on a
QCS8550/QCS6490 device without changing the interaction contract.

The versioned tool contract, QAIRT/QNN release gates, provisional latency targets and staged board
migration are documented in [`EDGE_DEPLOYMENT.md`](EDGE_DEPLOYMENT.md).

### Arbitrary text WatchAnything baseline

`upstream/Grounded-SAM-2` is pinned at `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`.
Its camera script is the closest existing baseline, but the wrapper below is required because a
remote H100 cannot open the Windows webcam with `cv2.VideoCapture(0)`.

```bash
pip install -r requirements-grounded-sam2.txt
python scripts/grounded_sam2_live_web.py \
  --repo upstream/Grounded-SAM-2 \
  --host 127.0.0.1 --port 7872 --prompt "person."
```

The verified H100 smoke test grounded and segmented `hippopotamus.` on the official 1280×720
sample. Model load took 51.58 seconds and the first grounded+segmented frame took 0.918 seconds.
This is functional proof, not a realtime FPS claim; intermediate SAM2 tracking is faster than a
new text-grounding pass, while GroundingDINO runs again every configured detection interval.

### Integrated conversational glasses agent

The `7872` wrapper now combines a hybrid command coordinator with persistent H100 workers:

- explicit commands use a deterministic low-latency route;
- ambiguous requests are planned by Qwen2.5-VL-3B as validated JSON with the fixed actions
  `track`, `describe`, `distance`, `memory`, `stop`, `help`, or `chat`;
- GroundingDINO + SAM2.1 handle arbitrary-text acquisition and mask tracking;
- Qwen2.5-VL-3B describes or reads the latest frame only when requested;
- Depth Anything V2 Metric Indoor Small estimates distance from the current tracked box;
- browser SpeechRecognition captures Vietnamese commands hands-free;
- `facebook/mms-tts-vie` synthesizes Vietnamese response WAVs on H100, so answer pronunciation no
  longer depends on a Vietnamese voice being installed in Windows.

Start the persistent planner/VLM/depth worker before the Gradio wrapper:

```bash
pip install -r requirements-agent-h100.txt
python scripts/vlm_agent_service.py --host 127.0.0.1 --port 8780

cd upstream/Grounded-SAM-2
.venv-h100/bin/python ../../scripts/grounded_sam2_live_web.py \
  --repo . --host 127.0.0.1 --port 7872 \
  --prompt "person." --agent-url http://127.0.0.1:8780
```

Open `http://127.0.0.1:7872`, grant webcam access, and try:

1. Start the webcam once.
2. Press **Bật hội thoại hands-free** once to grant Chrome microphone permission.
3. Continue entirely by voice. Say `Dừng nghe` to stop the microphone or `Ngừng nói` to cancel TTS.

Chrome requires the initial user gesture for microphone permission; a web page cannot bypass that
security rule. SpeechRecognition is fixed to `vi-VN`, final answers are checked for CJK leakage,
and neural answer audio is generated by MMS-TTS Vietnamese on the H100. Browser SpeechSynthesis is
disabled by default and remains only as an optional live-status voice for machines that already
have a Vietnamese system voice installed. The MMS-TTS Vietnamese checkpoint is CC-BY-NC 4.0, so
this configuration is for non-commercial research/education use.

- `Theo dõi chai đỏ`
- `Mô tả trước mặt tôi`
- `Vật đó cách bao xa?`
- `Lần cuối thấy nó ở đâu?`
- `Dừng theo dõi`

The distance endpoint returns `calibrated=false`. Its monocular estimate is useful for research
and camera calibration, but must not be treated as a certified navigation measurement. Until a
camera-specific calibration is validated, user-facing output should prefer coarse near/mid/far
language and preserve the safety warning.

All camera frames are letterboxed to a stable `1280x720` tracker input. Prompt changes clear old
SAM masks and track IDs, preventing the `1920 vs 1280` tensor mismatch observed when a webcam
changed resolution or orientation mid-session.

## Official EdgeTAM H100 sample

The checkpoint and code live in `upstream/EdgeTAM` and remain Apache-2.0 licensed upstream.

```bash
pip install -r requirements-edgetam.txt

python scripts/run_edgetam_sample.py \
  upstream/EdgeTAM/examples/04_coffee.mp4 \
  --repo upstream/EdgeTAM \
  --point 880 510 \
  --output runs/edgetam/coffee_cup.mp4
```

The point `(880, 510)` selects the blue cup in the official coffee example. The script loads the
official 56.1 MB checkpoint, propagates the prompt through the whole video, overlays the mask,
and writes a JSON timing report.

To run Meta's original click UI instead:

```bash
cd upstream/EdgeTAM
pip install -e ".[gradio]"
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python gradio_app.py
```

The original UI accepts uploaded videos, not a continuous browser webcam. Its predictor is a
singleton, so a public multi-user wrapper needs an inference lock and per-session state.

## Local VLM sample

The separate H100 sample uses Qwen2.5-VL-3B-Instruct to describe an EdgeTAM result. It is a real
model run, but it is not yet wired into the batch web MVP.

```bash
pip install -e ".[vlm]"
python scripts/run_vlm_frame.py results/edgetam/coffee_cup_mid.jpg \
  --prompt "Mô tả cảnh và đối tượng được tô màu xanh." \
  --output runs/vlm/coffee_cup.json
```

## Qualcomm deployment

Use Qualcomm's official AI Hub Models EdgeTAM port instead of inventing a converter from Meta's
CoreML-only export.

```bash
pip install "qai-hub-models[edgetam]" \
  "git+https://github.com/facebookresearch/EdgeTAM.git@a1209a454c9950d531498074a95ecf3a3ba02dfd"

qai-hub configure --api_token "$QAI_HUB_API_TOKEN"

python -m qai_hub_models.models.edgetam.export \
  --quantize w8a8 \
  --target-runtime qnn_context_binary \
  --device "Dragonwing RB3 Gen 2 Vision Kit"

python -m qai_hub_models.models.edgetam.export \
  --quantize w8a8 \
  --target-runtime qnn_context_binary \
  --device "QCS8550 (Proxy)"
```

The official port splits EdgeTAM into `encoder`, `memory_encoder`, and `video_decoder` QNN graphs.
Memory attention remains on CPU. QCS8550 Proxy compile/profile is not proof of full speed on a
physical HSPTEK box; camera integration, QAIRT version matching, thermal testing, and full-pipeline
latency still have to be validated on the issued hardware.

GenieX is reserved for the LLM/VLM skill router. It is Snapdragon-only and does not run on NVIDIA
H100. QCS6490 exists in GenieX device detection but is not in the currently validated platform
table; QCS8550 is also not a validated GenieX target. Treat both as device experiments until tested.

## Upstream audit notes

### OpenSQZ/OpenGlass

OpenGlass is useful as a sensor/hardware reference, not as the current application backbone.
The inspected commit is `1ffe701`.

- Paper implementation captures one on-demand JPEG and streams microphone audio from ESP32-S3;
  its 993 ms figure is query-ready to first audio at resized resolution, not continuous VLM FPS.
- Tracked firmware exposes `/ws_audio`, while the tracked host bridge hardcodes `/ws_audio_v2`.
- The panel contains maintainer-specific absolute paths and does not consume the documented local
  runtime config.
- Pinned MiniCPM/llama process commands are incompatible, and the documented `master` branch no
  longer exists.
- GitHub's license API returns 404 and the repository has no tracked LICENSE file despite README
  text. Keep it as an external research checkout; do not redistribute copied source until clarified.

### UCS-Bench / DirectMe

The inspected main commit is `1867339`. It is a strong research/schema anchor, but the clean clone
is not runnable using its documented Quick Start:

- `directme.data.frame_source`, `directme.perception.toy`, and `directme.datasets` are missing;
- `directme.eval` imports a nonexistent `Directme.directme` package;
- the full dataset is roughly 115.5 GiB;
- the default full perception path uses a very large DA3 Nested checkpoint and author-specific
  paths remain in the packaged script.

OmniGlass therefore implements its own `omniglass.observation-memory.v1` schema and Vietnamese
intent normalization without claiming compatibility or that the broken full DirectMe upstream
ran. A future explicit adapter can reuse DirectMe object fields and retrieval semantics.

### GenieX

The checkout is a device-phase reference. GenieX uses a BSD-3-Clause license plus Qualcomm terms
and model-specific licenses. It is the local VLM/LLM runtime, not the EdgeTAM executor.

## Verified results

Hardware: NVIDIA H100 80 GB pods.

| Run | Frames | Backend time | Result |
|---|---:|---:|---|
| OmniGlass indoor video, 2 perception FPS | 40 | 4.30 s | 30 observations, 7 labels, laptop missing-from-frame at 3.5 s |
| EdgeTAM official coffee example | 287 | 4.89 s propagation; 20.04 s E2E | 58.73 propagation FPS; masks emitted for 287 frames |
| Qwen2.5-VL-3B-Instruct masked frame | 1 | 2.19 s generation | Vietnamese description on H100 |
| Public mobile browser smoke test | 40 | 3.4 s backend | annotated video + JSON + timeline rendered at 390×844 |

The EdgeTAM FPS is H100 prompt propagation for this sample, not a tracking-accuracy score,
Qualcomm result, or end-to-end camera FPS. A qualitative middle-frame check shows the cup mask
still aligned. QCS8550 Proxy profiles target an Android proxy; context-binary portability to an
HSPTEK Linux BSP also depends on matching QAIRT and HTP runtime versions.

`Watch` does not claim physical object movement: a handheld camera can cause a screen-region
change. Physical movement requires ego-motion compensation or the planned SLAM3R/3D adapter.

## Repository layout

```text
omniglass/              runnable shared perception, memory, router, and web app
scripts/                CLI and official EdgeTAM sample runner
tests/                  deterministic memory/router tests
results/                generated evidence and benchmark artifacts
upstream/OpenGlass/     external research checkout
upstream/UCS-Bench/     external research checkout
upstream/EdgeTAM/       official tracker + checkpoint
upstream/GenieX/        Qualcomm local GenAI runtime reference
```

## Primary sources

- OpenGlass paper: https://aclanthology.org/2026.acl-demo.82/
- OpenGlass repository: https://github.com/OpenSQZ/OpenGlass
- UCS-Bench / DirectMe: https://github.com/cocowy1/UCS-Bench
- EdgeTAM: https://github.com/facebookresearch/EdgeTAM
- Qualcomm EdgeTAM model: https://aihub.qualcomm.com/models/edgetam
- Qualcomm AI Hub Models source: https://github.com/qualcomm/ai-hub-models
- GenieX: https://github.com/qualcomm/GenieX
