# MiniCPM-o 4.5 Vietnamese LoRA lane

This lane is intentionally separate from the running OmniGlass services. It
adapts the text-producing **thinker** to Vietnamese visual/audio instructions;
it does not fine-tune MiniCPM-o's speech-generating Talker.

## What upstream currently supports

The configuration is pinned to `ms-swift` commit
`ca937fbaf8e0c3dc4ea34358889430e36475463b`. At that revision:

- `openbmb/MiniCPM-o-4_5` is registered as `minicpmo` with template
  `minicpmo4_5` and vision, video, omni, and audio tags.
- The model-specific dependencies are `transformers==4.51.3` and
  `minicpmo-utils==1.0.6`.
- `INIT_TTS` defaults to false and `INIT_AUDIO` defaults to true.
- The multimodal architecture maps the language model to `llm`, the aligner
  to `resampler`, and both `vpm` and `apm` to the multimodal tower group.
  Consequently, `--freeze_vit true` freezes both the vision and audio towers.

The model revision is pinned to
`073dbbc8c5bc0af2d789e1ce12e7c17a6be746e1`. Its BF16 safetensors total
18,743,575,332 bytes (about 17.46 GiB before runtime overhead).

## Recommended first experiment

Use the output of `scripts/prepare_vietnamese_omni_dataset.py`. Start with a
small, consented, speaker-disjoint dataset containing:

- current-camera visual questions and short natural Vietnamese answers;
- 2–8 second Vietnamese audio commands paired with the intended answer;
- all three broad accent regions and real glasses-microphone noise;
- explicit uncertainty/abstention examples;
- concise accessibility language, without invented distance or hazards.

Do not train on the held-out FLEURS/OpenViVQA evaluation rows. A successful
10-step smoke only proves that the data/model/trainer contract works; it says
nothing about quality.

Run on an **idle, dedicated** 40 GB-class H100/MIG instance:

```bash
TRAIN_GPU=0 \
WORK_ROOT=/private/runs/minicpmo-vi-lora-smoke \
bash scripts/run_minicpmo_vi_lora_smoke.sh \
  /private/omniglass-vi/train.jsonl \
  /private/omniglass-vi/validation.jsonl
```

The launcher refuses to run if the selected GPU has any compute process or
less than 36,000 MiB free. It creates a revision-pinned virtual environment
and output directory under `WORK_ROOT`; it never stops or restarts a service.
The 20 GB H100 MIG is not a target for this profile.

The 28–36 GiB peak is an engineering estimate for batch 1, 1,024 tokens,
one image slice, frozen towers, rank-8 LoRA, BF16, and gradient checkpointing.
Record the real peak on the first clean smoke before increasing image slices,
audio duration, or context length.

## Train/freeze boundary

The smoke uses `all-linear` LoRA with:

- `freeze_llm=false`: add LoRA to the `llm` component;
- `freeze_vit=true`: no LoRA in `vpm` or `apm`;
- `freeze_aligner=true`: no LoRA in `resampler`;
- `INIT_AUDIO=true`: audio input remains available through the frozen encoder;
- `INIT_TTS=false`: the Talker is not created, saving memory and avoiding an
  unsupported claim about Vietnamese speech-output training.

This is the lowest-risk first experiment. If held-out audio understanding does
not improve, the next controlled ablation is **not** full-duplex/TTS training:
unfreeze only the audio projector/aligner at a much lower learning rate and
measure regression on English/Chinese plus Vietnamese. `ms-swift` currently
groups `vpm` and `apm` under the same `freeze_vit` switch, so audio-only tower
adaptation requires an audited target regex or a small upstream patch.

## Talker/TTS limitation

This adapter can improve Vietnamese text emitted by the thinker. It does not
teach the end-to-end Talker a Vietnamese voice, timing, barge-in behavior, or
full-duplex turn policy. The current `ms-swift` path loads MiniCPM-o with
`INIT_TTS=false` for training, and no verified upstream recipe establishes
Talker/TTS LoRA compatibility. Continue to synthesize the thinker's Vietnamese
text with VieNeu in the production `/vi` path.

Also treat adapter deployment as a separate compatibility gate: `swift infer
--adapters` supports PEFT inference, but the native MiniCPM-o duplex backend
must not be assumed to load that adapter. First validate the adapter under
Transformers/ms-swift, then test a merged checkpoint in a separate staging
worker. Never overwrite the current production model directory.

## Promotion gates

Promote beyond smoke only when all of these hold:

1. No train/eval speaker, session, or media SHA-256 leakage.
2. Vietnamese text-only, image, and audio slices each improve on held-out data.
3. English and Chinese regression stays within the chosen guardrail.
4. Hallucinated object, OCR, distance, and hazard rates do not worsen.
5. A separate staging worker passes five-turn and long-session protocol soaks.
6. Peak VRAM and step time are captured on the exact H100 partition.

The machine-readable source of truth is
`manifests/minicpmo_vi_lora_smoke.json`.
