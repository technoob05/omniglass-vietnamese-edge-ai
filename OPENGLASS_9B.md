# OpenGlass 9B — main demo

Đây là entrypoint chính của repo cho demo kính nói tiếng Anh:

- Model: `MiniCPM-o 4.5` (~9B), Q4 GGUF
- Model đã có: `models/minicpm-o45/MiniCPM-o-4_5-Q4_0.gguf`
- UI native: `https://127.0.0.1:8006/omni`
- Runtime: `llama-omni-server` → `MiniCPM-o-Demo/worker.py` → `gateway.py`

## CPU smoke / run

Trên pod Linux, khai báo các đường dẫn runtime rồi chạy:

```bash
export OPENGLASS_9B_ROOT=/network-volume/icse27/edge-ai/openglass-native
export OPENGLASS_9B_MODEL="$OPENGLASS_9B_ROOT/models/MiniCPM-o-4_5-Q4_0.gguf"
export OPENGLASS_9B_SERVER="$OPENGLASS_9B_ROOT/llama.cpp-omni/build/bin/llama-omni-server"
export OPENGLASS_9B_DEMO="$OPENGLASS_9B_ROOT/MiniCPM-o-Demo"
export OPENGLASS_9B_GPU_LAYERS=0       # CPU; use 99 for H100
./scripts/run_openglass_9b_cpu.sh
```

Sau khi server lên, forward port về laptop:

```powershell
ssh -N -L 8006:127.0.0.1:8006 -p <SSH_PORT> <USER>@<H100_HOST>
```

Mở `https://127.0.0.1:8006/omni`, chấp nhận certificate, bật camera/microphone và nói tiếng Anh.

## Phân biệt các demo khác

| Path | Vai trò |
|---|---|
| `OPENGLASS_9B.md` | Demo kính chính, MiniCPM-o 4.5 ~9B |
| `upstream/OpenGlass/` | Control panel ESP32/Rokid experimental |
| `scripts/openglass_browser_h100.py` | Browser adapter tới worker H100 |
| `scripts/minicpm_v46_edge_web.py` | Demo MiniCPM-V 4.6 1.3B edge |
| `omniglass/app.py` | Perception/video demo, không phải OpenGlass 9B |
