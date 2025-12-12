#!/usr/bin/env python3
"""
Post-Training Script for Elle-72B
Run after training completes to:
1. Merge LoRA adapter with base model
2. Quantize to AWQ INT4
3. Upload to HuggingFace
4. Prepare for Kaggle upload

Run on RunPod: python /workspace/nucleation-packages/scripts/post_training.py
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set via: export HF_TOKEN=your_token
BASE_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# Paths
ADAPTER_PATH = "/workspace/elle-interleaved-out"  # Training output
MERGED_PATH = "/workspace/outputs/elle-72b-interleaved-merged"
QUANTIZED_PATH = "/workspace/outputs/elle-72b-interleaved-awq"

# HuggingFace repos
HF_MERGED_REPO = "aphoticshaman/elle-72b-interleaved"
HF_QUANTIZED_REPO = "aphoticshaman/elle-72b-interleaved-awq"

def check_training_complete():
    """Check if training has completed."""
    adapter_file = Path(ADAPTER_PATH) / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"✗ Training not complete - no adapter found at {adapter_file}")
        print("  Wait for training to finish or check the correct output directory")
        return False
    print(f"✓ Found trained adapter at {ADAPTER_PATH}")
    return True

def merge_adapter():
    """Merge LoRA adapter with base model."""
    print("\n" + "="*60)
    print("STEP 1: Merging LoRA adapter with base model")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {MERGED_PATH}")
    os.makedirs(MERGED_PATH, exist_ok=True)
    model.save_pretrained(MERGED_PATH, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_PATH)

    print("✓ Merge complete!")
    return True

def quantize_model():
    """Quantize merged model to AWQ INT4."""
    print("\n" + "="*60)
    print("STEP 2: Quantizing to AWQ INT4")
    print("="*60)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("Installing AutoAWQ...")
        os.system("pip install autoawq autoawq-kernels -q")
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

    print(f"Loading merged model from: {MERGED_PATH}")
    model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH,
        trust_remote_code=True,
        safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)

    # Quantization config
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"
    }

    # Math-focused calibration data
    calibration_data = [
        "Solve for x: 2x + 5 = 13. First, subtract 5 from both sides...",
        "Find the sum of all positive integers less than 100 divisible by 3.",
        "Calculate the area of a triangle with vertices at (0,0), (4,0), and (2,3).",
        "Prove that for all positive integers n, n^3 - n is divisible by 6.",
        "Let f(x) = x^2 - 3x + 2. Find all values of x where f(x) = 0.",
        "In how many ways can 5 distinct books be arranged on a shelf?",
        "Find the remainder when 2^100 is divided by 7.",
        "The sum of three consecutive integers is 42. Find the integers.",
        "Calculate the determinant of the matrix [[1,2],[3,4]].",
        "Find the derivative of f(x) = x^3 * sin(x).",
    ] * 10  # Repeat for more samples

    print("Quantizing (this takes 30-60 minutes for 72B)...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)

    print(f"Saving quantized model to: {QUANTIZED_PATH}")
    os.makedirs(QUANTIZED_PATH, exist_ok=True)
    model.save_quantized(QUANTIZED_PATH)
    tokenizer.save_pretrained(QUANTIZED_PATH)

    # Check size
    total_size = sum(f.stat().st_size for f in Path(QUANTIZED_PATH).rglob("*") if f.is_file())
    print(f"✓ Quantization complete! Size: {total_size / 1e9:.1f} GB")
    return True

def create_model_card(path: str, repo_name: str, is_quantized: bool = False):
    """Create README.md model card."""

    quant_info = """
## Quantization

- **Method**: AWQ (Activation-aware Weight Quantization)
- **Precision**: INT4
- **Group Size**: 128
- **Size**: ~36GB (reduced from ~144GB)

This quantized version is optimized for inference on GPUs with limited VRAM (e.g., 80GB H100).
""" if is_quantized else ""

    card = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
  - math
  - reasoning
  - qwen2
  - aimo3
  - prometheus
  - nsm
  {"- awq" if is_quantized else "- merged"}
  {"- 4-bit" if is_quantized else ""}
library_name: {"transformers" if is_quantized else "transformers"}
pipeline_tag: text-generation
---

# Elle-72B-Interleaved{"-AWQ" if is_quantized else ""}

## Model Description

Elle-72B-Interleaved is a fine-tuned version of [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) trained with an interleaved dataset combining:

- **PROMETHEUS Protocol**: Structured problem decomposition and solution verification
- **NSM (Neuro-Symbolic Mathematics)**: Combining neural and symbolic approaches
- **XYZA Framework**: Multi-dimensional problem analysis
- **SDPM**: Structured Decision Problem Model
- **CIC Functional**: Basin clustering for answer consensus
- **LatticeForge**: Lattice-based mathematical structures
- **Code Training**: Python code generation for computational solutions

Designed for the **AI Mathematical Olympiad Progress Prize 3 (AIMO3)** competition.

## Model Details

- **Base Model**: Qwen/Qwen2.5-72B-Instruct
- **Parameters**: 72B
- **Training Method**: LoRA (r=64, α=128)
- **Training Data**: Interleaved framework examples with code
- **Precision**: {"INT4 (AWQ)" if is_quantized else "BF16"}
{quant_info}

## Training Configuration

```yaml
adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_linear: true
sequence_len: 4096
learning_rate: 2e-5
num_epochs: 2
optimizer: adamw_torch
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo_name}",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")

# Example: Mathematical problem solving
messages = [
    {{"role": "system", "content": "You are an expert mathematical problem solver. Write Python code to solve problems."}},
    {{"role": "user", "content": "Find all positive integers n < 1000 such that n^2 + n + 41 is prime."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Frameworks Integrated

| Framework | Purpose |
|-----------|---------|
| PROMETHEUS | Structured problem decomposition |
| NSM | Neuro-symbolic mathematics |
| XYZA | Multi-dimensional analysis |
| SDPM | Decision tree reasoning |
| CIC | Answer consensus clustering |
| LatticeForge | Lattice-based structures |

## License

Apache 2.0

## Citation

```bibtex
@misc{{elle-72b-interleaved,
  author = {{aphoticshaman}},
  title = {{Elle-72B-Interleaved: Mathematical Reasoning with Framework Integration}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```
"""

    readme_path = Path(path) / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)
    print(f"✓ Created model card at {readme_path}")

def upload_to_hf(local_path: str, repo_id: str):
    """Upload model to HuggingFace."""
    print(f"\nUploading {local_path} to {repo_id}...")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)

    # Create repo if doesn't exist
    try:
        create_repo(repo_id, token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Upload
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        commit_message="Upload Elle-72B-Interleaved model",
    )
    print(f"✓ Uploaded to https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Post-training processing for Elle-72B")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HF upload")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH, help="Path to trained adapter")
    args = parser.parse_args()

    global ADAPTER_PATH
    ADAPTER_PATH = args.adapter_path

    print("="*60)
    print("Elle-72B Post-Training Pipeline")
    print("="*60)
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"Merged path: {MERGED_PATH}")
    print(f"Quantized path: {QUANTIZED_PATH}")
    print(f"HF Token: {HF_TOKEN[:10]}...")

    # Check training complete
    if not check_training_complete():
        sys.exit(1)

    # Step 1: Merge
    if not args.skip_merge:
        if not Path(MERGED_PATH).exists() or not any(Path(MERGED_PATH).glob("*.safetensors")):
            merge_adapter()
        else:
            print(f"✓ Merged model already exists at {MERGED_PATH}")

    # Step 2: Quantize
    if not args.skip_quantize:
        if not Path(QUANTIZED_PATH).exists() or not any(Path(QUANTIZED_PATH).glob("*.safetensors")):
            quantize_model()
        else:
            print(f"✓ Quantized model already exists at {QUANTIZED_PATH}")

    # Create model cards
    if Path(MERGED_PATH).exists():
        create_model_card(MERGED_PATH, HF_MERGED_REPO, is_quantized=False)
    if Path(QUANTIZED_PATH).exists():
        create_model_card(QUANTIZED_PATH, HF_QUANTIZED_REPO, is_quantized=True)

    # Step 3: Upload
    if not args.skip_upload:
        if Path(MERGED_PATH).exists():
            upload_to_hf(MERGED_PATH, HF_MERGED_REPO)
        if Path(QUANTIZED_PATH).exists():
            upload_to_hf(QUANTIZED_PATH, HF_QUANTIZED_REPO)

    print("\n" + "="*60)
    print("POST-TRAINING COMPLETE!")
    print("="*60)
    print(f"Merged model: https://huggingface.co/{HF_MERGED_REPO}")
    print(f"Quantized model: https://huggingface.co/{HF_QUANTIZED_REPO}")
    print("\nNext steps for Kaggle:")
    print(f"1. Download quantized model from HF")
    print(f"2. Upload to Kaggle dataset: ryancardwell/elle-72b-interleaved-awq")
    print(f"3. Update notebook to use quantized model path")

if __name__ == "__main__":
    main()
