# Vietnamese ASR candidate benchmark

This track adds Vietnamese recognition without removing the original English
MiniCPM-o path. The current verified production choice remains PhoWhisper-medium
until a candidate completes the same-manifest evaluation and the live endpoint
test.

## Fair comparison contract

Every candidate consumes the exact frozen 50-row FLEURS `vi_vn/test` manifest
already used for PhoWhisper-medium and PhoWhisper-large. Audio hashes, sample IDs,
order and normalization are fixed. The rows remain evaluation-only: they must not
appear in fine-tuning data, prompt selection, hotword tuning or checkpoint
selection. Batch size is one, one warmup is excluded, and CUDA synchronization
surrounds every timed inference.

This is a small clean read-speech smoke test, not a Vietnamese SOTA benchmark. It
has no regional-accent labels, all selected rows are reported male, and it does
not represent spontaneous commands, glasses microphones, noise, endpointing or
streaming.

## Reproduced H100 result

Qwen3-ASR-0.6B was reproduced on the exact same 50 rows on the H100 MIG 2g.20GB
slice. It reached 9.18% normalized WER, 4.78% CER, throughput RTF 0.054 and
inference P95 0.997 s. The model load consumed approximately 1.76 GiB of visible
GPU memory. The persistent PhoWhisper service was still healthy after the run.

On these rows, PhoWhisper-medium remains fractionally better on word accuracy
(8.97% versus 9.18% WER) and essentially tied on P95/RTF, while Qwen3-ASR-0.6B is
better on character accuracy (4.78% versus 6.06% CER). The difference is only
three word edits (130 versus 127), so this 50-row smoke does not establish a
winner or a SOTA claim. Qwen remains valuable as the official streaming and
Apache-2.0 candidate for the glasses-mic evaluation.

## Candidate order

1. **Qwen3-ASR-0.6B** is the best new experiment for this product: Apache-2.0,
   explicit Vietnamese support, offline and vLLM streaming modes, and a compact
   0.6B size. It is not yet proven on QCS8550.
2. **Whisper-large-v3-turbo** is a useful MIT-licensed control. Its four-layer
   decoder is much faster than Whisper large-v3, but it is still utterance based.
3. **Qwen3-ASR-1.7B** is the H100 quality track. Upstream multilingual aggregates
   favor it over 0.6B, but those aggregates are not interchangeable with this
   frozen Vietnamese subset.

Pinned model revisions and licenses live in
`manifests/vietnamese_asr_candidate_benchmarks.json`.

## Safe H100 execution

```bash
cd /network-volume/icse27/edge-ai/openglass-native
bash scripts/run_vi_asr_candidate_benchmark.sh qwen3-asr-0.6b
bash scripts/run_vi_asr_candidate_benchmark.sh whisper-large-v3-turbo
bash scripts/run_vi_asr_candidate_benchmark.sh qwen3-asr-1.7b
```

The launcher installs into an isolated persistent venv, checks free VRAM before
both download and model load, uses batch size one, and refuses to run below the
candidate's threshold. It never stops or restarts the production PhoWhisper or
MiniCPM-o services.

## Promotion gates

- First compare normalized WER/CER, H100 P50/P95 latency, RTF and VRAM with the
  verified PhoWhisper-medium result.
- Then run a speaker-disjoint glasses/phone-mic set with northern, central and
  southern speakers, spontaneous commands, names/numbers and realistic noise.
- Measure VAD speech-end to immutable final; offline FLEURS latency is not a live
  endpoint latency.
- Promote an edge candidate only after a real HSPTEK QCS8550 run reports runtime,
  precision, RSS, P50/P95, WER/CER, sustained power and thermal throttling.

Qwen's official repository documents Vietnamese plus offline and streaming
inference. OpenAI's official Whisper announcement documents that turbo reduces
the large-v3 decoder from 32 layers to four. Neither fact proves performance on
this product or on QCS8550.
