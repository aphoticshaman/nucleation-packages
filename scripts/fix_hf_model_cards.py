#!/usr/bin/env python3
"""
Fix HuggingFace Model Cards for Elle models
Run this on RunPod with HF token
"""

import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required. Set it with: export HF_TOKEN=your_token")
api = HfApi(token=HF_TOKEN)

# Model card templates
ELLE_72B_ULTIMATE_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
  - math
  - reasoning
  - qwen2
  - merged
  - aimo3
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: elle-72b-ultimate
    results: []
---

# Elle-72B-Ultimate

## Model Description

Elle-72B-Ultimate is a fine-tuned version of [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) optimized for mathematical reasoning and problem-solving, specifically designed for the AI Mathematical Olympiad Progress Prize 3 (AIMO3) competition.

This is a **merged full model** (LoRA adapter merged into base weights).

## Model Details

- **Base Model**: Qwen/Qwen2.5-72B-Instruct
- **Parameters**: 72B
- **Precision**: BF16
- **Format**: Safetensors (31 shards)
- **Training Method**: LoRA (r=64, α=128)

## Training Data

Fine-tuned on mathematical reasoning datasets including:
- NuminaMath-CoT
- Custom mathematical reasoning examples

## Intended Use

- Mathematical problem solving
- Olympiad-style competition problems
- Code generation for computational solutions
- Chain-of-thought reasoning

## Limitations

- **Size**: ~144GB in BF16 - requires significant VRAM
- **Quantization Recommended**: For inference on consumer hardware, use AWQ or GPTQ quantized versions

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "aphoticshaman/elle-72b-ultimate",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("aphoticshaman/elle-72b-ultimate")

messages = [
    {"role": "system", "content": "You are an expert mathematical problem solver."},
    {"role": "user", "content": "Find all positive integers n such that n^2 + 1 divides n^3 + 1."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation

```bibtex
@misc{elle-72b-ultimate,
  author = {aphoticshaman},
  title = {Elle-72B-Ultimate: Mathematical Reasoning Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/aphoticshaman/elle-72b-ultimate}
}
```
"""

ELLE_72B_GEO_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
  - math
  - geometry
  - qwen2
  - lora
  - peft
library_name: peft
pipeline_tag: text-generation
---

# Elle-72B-Geo-v1

## Model Description

Elle-72B-Geo-v1 is a LoRA adapter fine-tuned on [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) for geometry problem solving.

## Model Details

- **Base Model**: Qwen/Qwen2.5-72B-Instruct
- **Adapter Type**: LoRA
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Target Modules**: All linear layers

## Training Data

Fine-tuned on geometry-focused mathematical problems including:
- Coordinate geometry
- Triangle and circle problems
- Area and perimeter calculations

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "aphoticshaman/elle-72b-geo-v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
```

## License

Apache 2.0
"""

ELLE_72B_MATH_V2_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
  - math
  - reasoning
  - qwen2
  - lora
  - peft
  - numina
library_name: peft
pipeline_tag: text-generation
datasets:
  - AI-MO/NuminaMath-CoT
---

# Elle-72B-Math-v2

## Model Description

Elle-72B-Math-v2 is a LoRA adapter fine-tuned on [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) for mathematical reasoning using the NuminaMath-CoT dataset.

## Model Details

- **Base Model**: Qwen/Qwen2.5-72B-Instruct
- **Adapter Type**: LoRA
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Target Modules**: All linear layers
- **Training Data**: NuminaMath-CoT

## Training

Fine-tuned using:
- Chain-of-thought mathematical reasoning examples
- Step-by-step problem decomposition
- Multiple solution strategies (algebraic, numerical, symbolic)

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "aphoticshaman/elle-72b-math-v2")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
```

## License

Apache 2.0
"""

def update_model_card(repo_id: str, content: str):
    """Update the README.md for a model repo."""
    try:
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add comprehensive model card",
        )
        print(f"✓ Updated model card for {repo_id}")
    except Exception as e:
        print(f"✗ Failed to update {repo_id}: {e}")

def main():
    print("Updating HuggingFace model cards...")
    print(f"Using token: {HF_TOKEN[:10]}...")

    # Update elle-72b-ultimate
    update_model_card("aphoticshaman/elle-72b-ultimate", ELLE_72B_ULTIMATE_CARD)

    # Update elle-72b-geo-v1
    update_model_card("aphoticshaman/elle-72b-geo-v1", ELLE_72B_GEO_CARD)

    # Update elle-72b-math-v2
    update_model_card("aphoticshaman/elle-72b-math-v2", ELLE_72B_MATH_V2_CARD)

    print("\nDone!")

if __name__ == "__main__":
    main()
