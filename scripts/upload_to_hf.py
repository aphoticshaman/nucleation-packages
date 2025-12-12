#!/usr/bin/env python3
"""
Upload merged Elle model to HuggingFace with complete metadata.

Usage:
    python upload_to_hf.py --model /workspace/elle-merged --repo aphoticshaman/Elle-72B-Ultimate

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Model card template
MODEL_CARD = """---
license: apache-2.0
language:
  - en
library_name: transformers
tags:
  - geopolitical-analysis
  - risk-assessment
  - intelligence
  - fine-tuned
  - lora
  - awq
  - qwen2.5
base_model: Qwen/Qwen2.5-72B-Instruct-AWQ
datasets:
  - custom
pipeline_tag: text-generation
model-index:
  - name: Elle-72B-Ultimate
    results: []
---

# Elle-72B-Ultimate

**Elle** is a fine-tuned geopolitical intelligence model built on Qwen2.5-72B-Instruct-AWQ, specialized for:

- Real-time geopolitical risk assessment
- Multi-source intelligence synthesis
- Causal chain analysis for global events
- Regime stability detection
- Cascade risk prediction

## Model Details

| Attribute | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2.5-72B-Instruct-AWQ |
| Fine-tuning Method | LoRA (r=64, alpha=128) |
| Training Framework | Unsloth + PEFT |
| Quantization | AWQ 4-bit |
| Context Length | 32,768 tokens |
| Final Training Loss | 0.2544 |

## Training Data

Elle was trained on curated geopolitical intelligence data including:

- **GDELT Event Data**: Global event monitoring and conflict detection
- **World Bank Indicators**: Economic stability metrics
- **USGS Seismic Data**: Natural disaster risk factors
- **Curated Intel Briefings**: Expert-verified geopolitical analysis
- **Cascade Analysis**: Historical event chain patterns

Training used interleaved conversation format with system prompts, user queries, and assistant responses.

## Intended Use

Elle is designed for:

- Enterprise geopolitical risk dashboards
- Intelligence briefing generation
- Supply chain risk assessment
- Investment risk analysis
- Policy impact modeling

## Limitations

- Knowledge cutoff aligned with training data (Dec 2024)
- Requires external data feeds for real-time analysis
- Should be used as analytical support, not sole decision-maker
- May reflect biases present in training data sources

## Hardware Requirements

- **Inference**: 4x H100/H200 80GB (vLLM recommended)
- **Memory**: ~160GB VRAM for full model
- AWQ quantization enables efficient deployment

## Usage with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="aphoticshaman/Elle-72B-Ultimate",
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_model_len=32768,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=4096,
)

prompt = \"\"\"<|im_start|>system
You are Elle, an expert geopolitical intelligence analyst.
<|im_end|>
<|im_start|>user
Analyze the current risk factors affecting semiconductor supply chains.
<|im_end|>
<|im_start|>assistant
\"\"\"

outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

## Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "aphoticshaman/Elle-72B-Ultimate",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("aphoticshaman/Elle-72B-Ultimate")

messages = [
    {"role": "system", "content": "You are Elle, an expert geopolitical intelligence analyst."},
    {"role": "user", "content": "What are the key risk indicators for the South China Sea region?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Configuration

```yaml
# LoRA Configuration
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training Hyperparameters
learning_rate: 2e-5
batch_size: 2
gradient_accumulation_steps: 8
epochs: 3
warmup_ratio: 0.03
lr_scheduler: cosine
optimizer: adamw_8bit
max_seq_length: 8192
```

## Citation

```bibtex
@misc{elle-72b-ultimate,
  author = {LatticeForge},
  title = {Elle-72B-Ultimate: Fine-tuned Geopolitical Intelligence Model},
  year = {2024},
  publisher = {HuggingFace},
  url = {https://huggingface.co/aphoticshaman/Elle-72B-Ultimate}
}
```

## License

Apache 2.0 - See LICENSE file for details.

## Contact

- **Website**: [latticeforge.ai](https://latticeforge.ai)
- **Issues**: Report issues via HuggingFace discussions
"""


def main():
    parser = argparse.ArgumentParser(description="Upload merged model to HuggingFace")
    parser.add_argument("--model", default="/workspace/elle-merged", help="Path to merged model")
    parser.add_argument("--repo", default="aphoticshaman/Elle-72B-Ultimate", help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        print("Run merge_lora.py first to create the merged model")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/accessing repo: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Note: {e}")

    # Write README.md to model directory
    readme_path = model_path / "README.md"
    print(f"Writing model card to {readme_path}")
    with open(readme_path, "w") as f:
        f.write(MODEL_CARD)

    # Upload all files
    print(f"Uploading model files from {model_path} to {args.repo}...")
    print("This may take a while for a 72B model...")

    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload Elle-72B-Ultimate merged model",
    )

    print(f"\n Done! Model uploaded to: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
