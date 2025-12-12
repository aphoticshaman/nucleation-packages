#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for Elle deployment.

Usage:
    python merge_lora.py --adapter /workspace/elle-interleaved-out --output /workspace/elle-merged
"""

import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    # IMPORTANT: Use non-AWQ base model - the LoRA was trained on the full-precision model
    # AWQ has different MLP dimensions (29696 vs 29568) which causes size mismatch errors
    parser.add_argument("--base", default="Qwen/Qwen2.5-72B-Instruct", help="Base model path or HF ID")
    parser.add_argument("--adapter", default="/workspace/elle-interleaved-out", help="LoRA adapter path")
    parser.add_argument("--output", default="/workspace/elle-merged", help="Output path for merged model")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    merged.save_pretrained(args.output, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)

    print(f"✅ Done! Merged model saved to {args.output}")

if __name__ == "__main__":
    main()
