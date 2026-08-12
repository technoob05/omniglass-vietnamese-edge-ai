# Third-party notices

This repository contains original integration code and patch tooling. It does
not vendor the referenced upstream repositories, model weights, datasets, or
recorded media. Download those artifacts from their official sources and obey
their individual terms.

| Component | Role | Upstream terms / note |
|---|---|---|
| OpenBMB MiniCPM-o 4.5 | visual-language reasoning and native EN/ZH omni baseline | Apache-2.0 model; use the official MiniCPM-o-Demo repository separately |
| VinAI PhoWhisper | Vietnamese ASR | BSD-3-Clause |
| VieNeu-TTS v3 Turbo | primary Vietnamese TTS | Apache-2.0 at the pinned upstream revision |
| facebook/mms-tts-vie | optional Vietnamese TTS fallback | CC-BY-NC-4.0; non-commercial restriction applies |
| sherpa-onnx | edge ASR runtime research | Apache-2.0 runtime; the evaluated Vietnamese 30M checkpoint is CC-BY-NC-ND-4.0 |
| Ultralytics | optional YOLO detector in the separate visual-memory MVP | review current AGPL-3.0/enterprise terms for your distribution and deployment |
| EdgeTAM / Grounded-SAM-2 / SLAM3R | optional research integrations | fetched separately; follow each upstream license |
| OpenGlass | architecture reference | the audited upstream checkout did not contain a root LICENSE; it is not redistributed here |

No model or dataset license is replaced by this repository's Apache-2.0
license. The school/research demo is not evidence of commercial-use clearance.
