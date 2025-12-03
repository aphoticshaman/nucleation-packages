#!/usr/bin/env python3
"""
Merge multiple LoRA adapters into a single model.
Run after both training runs complete.

Usage:
    python merge_loras.py

Expects:
    - /workspace/lattice-lora (from Pod 1 - grok training)
    - /workspace/lattice-finance-intel (from Pod 2 - finance+intel training)

Output:
    - /workspace/lattice-merged (final merged model)
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge():
    print("=" * 60)
    print("LORA MERGE - Combining trained adapters")
    print("=" * 60)

    # Load base model
    print("\n[1/4] Loading base Phi-2 model...")
    base = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    # Load and merge first adapter (grok geopolitical data)
    print("\n[2/4] Loading and merging lattice-lora (grok data)...")
    model = PeftModel.from_pretrained(base, "/workspace/lattice-lora")
    model = model.merge_and_unload()
    print("  ✓ First adapter merged")

    # Load and merge second adapter (finance + intel)
    print("\n[3/4] Loading and merging lattice-finance-intel...")
    model = PeftModel.from_pretrained(model, "/workspace/lattice-finance-intel")
    model = model.merge_and_unload()
    print("  ✓ Second adapter merged")

    # Save merged model
    print("\n[4/4] Saving merged model to /workspace/lattice-merged...")
    model.save_pretrained("/workspace/lattice-merged")
    tokenizer.save_pretrained("/workspace/lattice-merged")

    print("\n" + "=" * 60)
    print("SUCCESS! Merged model saved to /workspace/lattice-merged")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test inference with the merged model")
    print("  2. Upload to HuggingFace Hub or download locally")
    print("  3. Optionally quantize (GGUF) for deployment")

if __name__ == "__main__":
    merge()
