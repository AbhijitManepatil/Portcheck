"""Tiny text-generation service used as the portcheck end-to-end fixture.

Usage: E2E_MODEL=<path-or-hub-id> python serve.py "your prompt"
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gpu_stats


def main() -> int:
    if len(sys.argv) < 2:
        print('usage: serve.py "<prompt>"', file=sys.stderr)
        return 2
    model_name = os.environ.get("E2E_MODEL")
    if not model_name:
        print("Set E2E_MODEL to a model path or hub id.", file=sys.stderr)
        return 2

    if os.environ.get("USE_TRT") == "1":
        import trt_backend

        trt_backend.build_engine(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    inputs = tokenizer(sys.argv[1], return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    used = gpu_stats.memory_used()
    if used is not None:
        print(f"GPU memory in use: {used} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
