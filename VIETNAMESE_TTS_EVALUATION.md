# Vietnamese TTS evaluation for H100 and QCS8550

## Outcome

Keep **VieNeu-TTS v3 Turbo** as the Vietnamese production voice on H100. It is
the only permissively licensed candidate in this audit with an actual
streaming measurement in the deployed environment. Treat
**Kokoro-Vietnamese** as the first QCS8550 port candidate, not as a replacement
yet: its core is much smaller and its source includes an ONNX export, but no
latency or quality claim is valid until the weights can be fetched and it is
measured on the physical box.

No production process was stopped or restarted for this evaluation.

## Fixed ten-sentence measurement

The new benchmark includes visual-assistance phrases, safety language,
Vietnamese diacritics, numbers, an address, an acronym, and an English model
name. It records time to first received audio, total synthesis time, audio
duration, RTF, errors, file hashes, and a PhoWhisper ASR-back transcript.

| Engine | Backend | Success | first audio median / P95 | total median / P95 | RTF median / P95 |
|---|---|---:|---:|---:|---:|
| VieNeu v3 Turbo | deployed H100 streaming endpoint | 10/10 | 157.8 / 183.6 ms | 1.730 / 2.353 s | 0.416 / 0.489 |
| Kokoro-Vietnamese | planned CUDA, then ONNX/QNN | blocked before inference | not measured | not measured | not measured |

PhoWhisper transcribed all seven sentences without numbers/acronyms exactly
after punctuation/case normalization. The raw aggregate ASR-back CER/WER is not
reported as TTS quality because the remaining three references contain written
digits (`8`, `15`, `37`, `25`, `24`, `8550`) while the synthesized/recognized
form correctly spells those numbers out. A text-normalization-aware reference
or human listening panel is required for a fair aggregate score. ASR-back is an
intelligibility proxy and never a substitute for MOS, speaker similarity, or a
blind preference test.

The complete sanitized result is
`artifacts/vietnamese-tts-vieneu-v3-10.json`. Re-run with:

```bash
python scripts/benchmark_vietnamese_tts_candidates.py \
  --engine vieneu-http \
  --output-dir results/vieneu-v3-10
```

## Candidate audit

- **VieNeu v3 Turbo** — Apache-2.0, roughly 0.1B parameters, native chunk
  iterator, Vietnamese/English model. This remains the H100 choice. The prior
  ONNX INT8 x86 CPU result (RTF 3.27) is not realtime and says nothing about
  QCS8550 HTP performance.
- **Kokoro-Vietnamese** — Apache-2.0, 82M-family core, about 335 MB of published
  artifacts, Vietnamese `vig2p`, and an opset-18 ONNX export path. This is the
  best-sized open candidate for QCS8550. The H100 trial reached the pinned
  source and dependencies but Hugging Face's LFS CDN returned HTTP 403 for
  `kokoro_vi.pth`; therefore no fabricated timing appears in this report.
- **MeloTTS-Vietnamese** — MIT and plausibly CPU-oriented. Its published bundle
  is about 1.19 GB because it includes generator and discriminator training
  checkpoints; it needs an inference-only export before edge evaluation.
- **Vira-TTS** — Apache-2.0 and Vietnamese voice cloning, but the model card
  identifies only a custom dataset. Keep it in research until provenance is
  reviewed; its 1.33 GB artifact and autoregressive codec also make it a weaker
  first QCS8550 target.
- **Parler-TTS Vietnamese** — Apache-2.0 and useful for controlled voice-design
  research, but the Hub repository is about 25 GB with repeated training
  checkpoints. It is not an edge package.
- **Viterbox** and **MMS-TTS Vietnamese** — CC-BY-NC-4.0. They are excluded from
  the commercial/product route regardless of attractive quality or speed.
- **Chatterbox Multilingual** — MIT, but its official supported-language list
  does not include Vietnamese. It was excluded instead of being mislabeled as
  a Vietnamese baseline.

Exact revisions, artifact sizes and dispositions are in
`manifests/vietnamese_tts_candidates.json`.

## QCS8550 port gate

1. Download the pinned Kokoro weights from a network without the observed LFS
   denial and verify their checksum/revision.
2. Export opset-18 ONNX and validate waveform parity on at least the ten fixed
   sentences. Export only inference weights and one approved voicepack.
3. Run ONNX Runtime CPU ARM64 first. Record cold load, RSS, median/P95 latency,
   RTF, thermals, power, and 30-minute stability.
4. Compile fixed token buckets for QNN/HTP only after CPU parity. Keep G2P and
   sentence scheduling on CPU; investigate unsupported operators rather than
   silently falling back.
5. Acceptance: zero synthesis errors, RTF P95 below 0.8, first sentence audio
   below 500 ms, RSS within the system budget, and human Vietnamese preference
   no worse than the VieNeu H100 reference on glasses-domain prompts.
6. Until those gates pass, QCS8550 uses prerecorded safety phrases and the H100
   VieNeu service for general answers.
