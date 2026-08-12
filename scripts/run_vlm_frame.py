#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Run a local Qwen2.5-VL See/Read skill")
    parser.add_argument("image")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--prompt",
        default="Describe this scene concisely and transcribe any prominent readable text.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", default="runs/vlm/result.json")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    load_seconds = time.perf_counter() - load_started

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
    generate_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    inference_seconds = time.perf_counter() - generate_started

    result = {
        "schema": "omniglass.vlm-sample.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "image": os.path.relpath(Path(args.image).resolve(), Path.cwd()),
        "prompt": args.prompt,
        "answer": answer,
        "gpu_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
