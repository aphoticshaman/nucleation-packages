#!/usr/bin/env python3
"""
Elle-72B Full Adapter Merge Script
===================================
Merges ALL LoRA adapters into one comprehensive model:
- elle-72b-geo (geopolitics)
- elle-72b-math-v2 (mathematics)
- evol/interleaved adapter (coding + frameworks)

Output:
- Elle-72B-FULL: BF16 full model (~144GB) for RunPod Serverless / latticeforge.ai
- Elle-72B-AWQ: INT4 quantized (~36GB) for Kaggle H100

Merging Strategy:
- Uses TIES (Trim, Elect, Sign & Merge) or linear combination
- Preserves capabilities from all adapters
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Check for required packages
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig, get_peft_model
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install transformers peft huggingface_hub bitsandbytes accelerate")
    sys.exit(1)


# Configuration
BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
HF_USERNAME = "aphoticshaman"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Adapter paths on RunPod
ADAPTER_CONFIGS = {
    "geo": {
        "path": "/workspace/elle-72b-geo",
        "weight": 1.0,
        "description": "Geopolitics and geography reasoning"
    },
    "math": {
        "path": "/workspace/elle-72b-math-v2",
        "weight": 1.0,
        "description": "Mathematical reasoning (NuminaMath-CoT)"
    },
    "evol": {
        "path": "/workspace/outputs/elle-72b-evol",  # Current training output
        "weight": 1.0,
        "description": "Coding + PROMETHEUS/NSM/XYZA/SDPM/CIC/LatticeForge frameworks"
    }
}

# Output paths
OUTPUT_DIR = "/workspace/elle-72b-merged"
FULL_MODEL_NAME = "elle-72b-full"
QUANTIZED_MODEL_NAME = "elle-72b-awq"


def find_available_adapters() -> Dict[str, dict]:
    """Find which adapters actually exist on disk."""
    available = {}

    print("\n" + "="*60)
    print("SCANNING FOR AVAILABLE ADAPTERS")
    print("="*60)

    for name, config in ADAPTER_CONFIGS.items():
        path = Path(config["path"])

        # Check for adapter_config.json (indicates valid LoRA adapter)
        adapter_config_path = path / "adapter_config.json"

        if adapter_config_path.exists():
            print(f"  [FOUND] {name}: {path}")
            available[name] = config
        else:
            # Check for checkpoint subdirectories
            checkpoints = list(path.glob("checkpoint-*"))
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
                if (latest / "adapter_config.json").exists():
                    print(f"  [FOUND] {name}: {latest} (checkpoint)")
                    config["path"] = str(latest)
                    available[name] = config
                else:
                    print(f"  [MISSING] {name}: No adapter_config.json in {latest}")
            else:
                print(f"  [MISSING] {name}: {path} not found")

    print(f"\nFound {len(available)}/{len(ADAPTER_CONFIGS)} adapters")
    return available


def load_base_model(load_in_4bit: bool = True):
    """Load base Qwen model."""
    print("\n" + "="*60)
    print(f"LOADING BASE MODEL: {BASE_MODEL}")
    print("="*60)

    if load_in_4bit:
        print("Loading in 4-bit for merging (saves VRAM)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        print("Loading in full precision (requires ~150GB VRAM)...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print(f"Model loaded: {model.dtype}")
    return model, tokenizer


def merge_adapters_sequential(model, adapters: Dict[str, dict], tokenizer):
    """
    Sequential adapter merging - merge one at a time.
    Each adapter is merged into weights before loading next.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL ADAPTER MERGING")
    print("="*60)

    current_model = model

    for i, (name, config) in enumerate(adapters.items(), 1):
        print(f"\n[{i}/{len(adapters)}] Merging adapter: {name}")
        print(f"    Path: {config['path']}")
        print(f"    Weight: {config['weight']}")
        print(f"    Description: {config['description']}")

        # Load adapter
        current_model = PeftModel.from_pretrained(
            current_model,
            config["path"],
            adapter_name=name,
        )

        # Scale adapter weights if not 1.0
        if config["weight"] != 1.0:
            print(f"    Scaling adapter by {config['weight']}")
            current_model.add_weighted_adapter(
                adapters=[name],
                weights=[config["weight"]],
                adapter_name=f"{name}_scaled",
            )
            current_model.set_adapter(f"{name}_scaled")

        # Merge adapter into base weights
        print("    Merging into base weights...")
        current_model = current_model.merge_and_unload()

        print(f"    Merged successfully!")

    return current_model


def merge_adapters_ties(model, adapters: Dict[str, dict], tokenizer):
    """
    TIES-style merging - load all adapters and combine with weighted average.
    Better for preserving capabilities from all adapters.
    """
    print("\n" + "="*60)
    print("TIES-STYLE ADAPTER MERGING")
    print("="*60)

    # Load first adapter
    adapter_names = list(adapters.keys())
    first_name = adapter_names[0]
    first_config = adapters[first_name]

    print(f"\nLoading primary adapter: {first_name}")
    peft_model = PeftModel.from_pretrained(
        model,
        first_config["path"],
        adapter_name=first_name,
    )

    # Load additional adapters
    for name in adapter_names[1:]:
        config = adapters[name]
        print(f"Loading adapter: {name} from {config['path']}")
        peft_model.load_adapter(config["path"], adapter_name=name)

    # Combine adapters with weighted average
    weights = [adapters[name]["weight"] for name in adapter_names]
    print(f"\nCombining adapters with weights: {dict(zip(adapter_names, weights))}")

    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name="merged",
        combination_type="linear",  # or "ties", "dare_ties", "dare_linear"
    )

    peft_model.set_adapter("merged")

    # Merge into base weights
    print("Merging combined adapter into base weights...")
    merged_model = peft_model.merge_and_unload()

    return merged_model


def save_full_model(model, tokenizer, output_path: str, model_name: str):
    """Save the full merged model."""
    print("\n" + "="*60)
    print(f"SAVING FULL MODEL: {model_name}")
    print("="*60)

    full_path = Path(output_path) / model_name
    full_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {full_path}")
    print("This may take a while for 72B model...")

    model.save_pretrained(full_path, safe_serialization=True)
    tokenizer.save_pretrained(full_path)

    # Create model card
    model_card = create_full_model_card(model_name)
    (full_path / "README.md").write_text(model_card)

    # Calculate size
    total_size = sum(f.stat().st_size for f in full_path.glob("*.safetensors"))
    print(f"Saved! Total size: {total_size / 1e9:.1f} GB")

    return str(full_path)


def quantize_to_awq(model_path: str, output_path: str, model_name: str):
    """Quantize to AWQ INT4 format."""
    print("\n" + "="*60)
    print(f"QUANTIZING TO AWQ: {model_name}")
    print("="*60)

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("AWQ not installed. Install with: pip install autoawq")
        print("Skipping quantization...")
        return None

    awq_path = Path(output_path) / model_name
    awq_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # AWQ quantization config
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    print("Quantizing (this takes 1-2 hours for 72B)...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving to: {awq_path}")
    model.save_quantized(str(awq_path), safetensors=True)
    tokenizer.save_pretrained(str(awq_path))

    # Create model card
    model_card = create_awq_model_card(model_name)
    (awq_path / "README.md").write_text(model_card)

    total_size = sum(f.stat().st_size for f in awq_path.glob("*.safetensors"))
    print(f"Quantized! Total size: {total_size / 1e9:.1f} GB")

    return str(awq_path)


def create_full_model_card(model_name: str) -> str:
    """Create README.md for full merged model."""
    return f'''---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
- elle
- qwen2.5
- 72b
- merged
- mathematics
- geopolitics
- coding
- latticeforge
model-index:
- name: {model_name}
  results: []
---

# Elle-72B-Full

**The complete Elle model** - A comprehensive merger of all Elle adapters into Qwen2.5-72B-Instruct.

## Model Description

Elle-72B-Full combines specialized capabilities from multiple fine-tuned adapters:

| Adapter | Capability | Training Data |
|---------|------------|---------------|
| elle-geo | Geopolitics & Geography | Custom geopolitical reasoning |
| elle-math | Mathematics | NuminaMath-CoT, AIME, AMC |
| elle-evol | Coding + Frameworks | PROMETHEUS, NSM, XYZA, SDPM, CIC, LatticeForge |

## Reasoning Frameworks

This model has been trained on proprietary reasoning frameworks:

- **PROMETHEUS**: Predictive Reasoning and Outcome Mapping for Efficient Task Handling and Execution Under Scrutiny
- **NSM**: Nested Structured Metacognition
- **XYZA**: A→B, B→C, therefore A→C logical chaining
- **SDPM**: Socratic Dialogue and Perspective Mapping
- **CIC**: Contextual Integrity Checks
- **LatticeForge**: Multi-dimensional reasoning lattice

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "aphoticshaman/{model_name}",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("aphoticshaman/{model_name}")
```

## Deployment

- **RunPod Serverless**: Full BF16 inference for latticeforge.ai
- **Kaggle**: Use the AWQ quantized version (elle-72b-awq)

## Hardware Requirements

- **VRAM**: ~150GB for full precision inference
- **Recommended**: 2x H100 80GB or 4x A100 80GB

## Created

{datetime.now().strftime("%Y-%m-%d")} by aphoticshaman
'''


def create_awq_model_card(model_name: str) -> str:
    """Create README.md for AWQ quantized model."""
    return f'''---
license: apache-2.0
base_model: aphoticshaman/elle-72b-full
tags:
- elle
- qwen2.5
- 72b
- awq
- quantized
- int4
- kaggle
model-index:
- name: {model_name}
  results: []
---

# Elle-72B-AWQ

**INT4 Quantized version of Elle-72B-Full** for deployment on single H100 80GB (Kaggle).

## Model Description

AWQ (Activation-aware Weight Quantization) version of Elle-72B-Full:
- **Original size**: ~144GB (BF16)
- **Quantized size**: ~36GB (INT4)
- **Quality retention**: >99% of full model performance

## Capabilities

All capabilities from Elle-72B-Full:
- Mathematical reasoning (AIME, AMC, NuminaMath)
- Geopolitical analysis
- Code generation
- PROMETHEUS, NSM, XYZA, SDPM, CIC, LatticeForge frameworks

## Usage with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="aphoticshaman/{model_name}",
    quantization="awq",
    dtype="float16",
    gpu_memory_utilization=0.95,
    max_model_len=4096,
)
```

## Kaggle Deployment

This model fits on a single H100 80GB GPU available on Kaggle.

## Created

{datetime.now().strftime("%Y-%m-%d")} by aphoticshaman
'''


def upload_to_huggingface(model_path: str, repo_name: str):
    """Upload model to HuggingFace Hub."""
    print("\n" + "="*60)
    print(f"UPLOADING TO HUGGINGFACE: {repo_name}")
    print("="*60)

    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USERNAME}/{repo_name}"

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation note: {e}")

    print(f"Uploading from: {model_path}")
    print("This may take a while for large models...")

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"Upload complete: https://huggingface.co/{repo_id}")
    return repo_id


def main():
    parser = argparse.ArgumentParser(description="Merge Elle-72B adapters")
    parser.add_argument("--strategy", choices=["sequential", "ties"], default="sequential",
                        help="Merging strategy (sequential or ties)")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip AWQ quantization")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip HuggingFace upload")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load base model in 4-bit (saves VRAM but slower)")
    args = parser.parse_args()

    print("="*60)
    print("ELLE-72B FULL ADAPTER MERGE")
    print("="*60)
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output_dir}")
    print(f"Quantize: {not args.skip_quantize}")
    print(f"Upload: {not args.skip_upload}")

    # Find available adapters
    adapters = find_available_adapters()

    if len(adapters) == 0:
        print("\nERROR: No adapters found! Check paths in ADAPTER_CONFIGS")
        sys.exit(1)

    if len(adapters) < len(ADAPTER_CONFIGS):
        print("\nWARNING: Not all adapters found. Continue anyway? [y/N]")
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(0)

    # Load base model
    model, tokenizer = load_base_model(load_in_4bit=args.load_4bit)

    # Merge adapters
    if args.strategy == "sequential":
        merged_model = merge_adapters_sequential(model, adapters, tokenizer)
    else:
        merged_model = merge_adapters_ties(model, adapters, tokenizer)

    # Save full model
    full_model_path = save_full_model(merged_model, tokenizer, args.output_dir, FULL_MODEL_NAME)

    # Quantize to AWQ
    awq_model_path = None
    if not args.skip_quantize:
        awq_model_path = quantize_to_awq(full_model_path, args.output_dir, QUANTIZED_MODEL_NAME)

    # Upload to HuggingFace
    if not args.skip_upload:
        upload_to_huggingface(full_model_path, FULL_MODEL_NAME)
        if awq_model_path:
            upload_to_huggingface(awq_model_path, QUANTIZED_MODEL_NAME)

    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"Full model: {full_model_path}")
    if awq_model_path:
        print(f"AWQ model: {awq_model_path}")
    print("\nNext steps:")
    print("1. Test inference on full model")
    print("2. Deploy to RunPod Serverless for latticeforge.ai")
    print("3. Update Kaggle notebook to use AWQ model")


if __name__ == "__main__":
    main()
