# Elle Multi-Domain Merge Strategy

## The Vision

Elle is a **unified intelligence** - math genius + geopolitical wizard + code warrior + autonomous researcher. We achieve this through:

1. **Sequential domain-specific training** - separate LoRA adapters per domain
2. **Expert merging** - combine adapters using DARE-TIES or task arithmetic
3. **Single model deployment** - no need for multiple models or hot-swapping

## Phase 1: Domain-Specific Training Runs

### Run 1: Math Expert
```yaml
# config_elle_math.yaml
base_model: Qwen/Qwen2.5-72B-Instruct
adapter: lora
lora_r: 64
lora_alpha: 128

datasets:
  - path: AI-MO/NuminaMath-CoT
    split: train[:20000]
  - path: TIGER-Lab/MATH-500
  - path: nvidia/OpenMathInstruct-2
    split: train[:10000]

output_dir: /workspace/elle-loras/math
```

### Run 2: Geopolitical Expert
```yaml
# config_elle_geo.yaml
base_model: Qwen/Qwen2.5-72B-Instruct
adapter: lora
lora_r: 64
lora_alpha: 128

datasets:
  - path: aphoticshaman/elle-cic-training  # Our generated CIC data
  - path: local:/workspace/elle_training_cic.jsonl

output_dir: /workspace/elle-loras/geo
```

### Run 3: Code Expert
```yaml
# config_elle_code.yaml
base_model: Qwen/Qwen2.5-72B-Instruct
adapter: lora
lora_r: 64
lora_alpha: 128

datasets:
  - path: bigcode/starcoderdata
    split: train[:10000]
  - path: codeparrot/github-code
    split: train[:5000]
  - path: nvidia/HelpSteer2

output_dir: /workspace/elle-loras/code
```

### Run 4: Research Expert (PROMETHEUS)
```yaml
# config_elle_research.yaml
base_model: Qwen/Qwen2.5-72B-Instruct
adapter: lora
lora_r: 64
lora_alpha: 128

datasets:
  - path: local:/workspace/elle_prometheus_training.jsonl

output_dir: /workspace/elle-loras/research
```

## Phase 2: Merge Adapters

### Option A: Task Arithmetic (Simple)
```python
import torch
from safetensors.torch import load_file, save_file

def task_arithmetic_merge(adapter_paths: dict, weights: dict, output_path: str):
    """
    Merge LoRA adapters via weighted sum.

    adapter_paths: {"math": "/path/to/math", "geo": "/path/to/geo", ...}
    weights: {"math": 0.35, "geo": 0.25, "code": 0.25, "research": 0.15}
    """
    merged = {}

    for name, path in adapter_paths.items():
        adapter = load_file(f"{path}/adapter_model.safetensors")
        weight = weights[name]

        for key, tensor in adapter.items():
            if key not in merged:
                merged[key] = weight * tensor
            else:
                merged[key] += weight * tensor

    save_file(merged, f"{output_path}/adapter_model.safetensors")
    print(f"Merged adapter saved to {output_path}")

# Usage
task_arithmetic_merge(
    adapter_paths={
        "math": "/workspace/elle-loras/math",
        "geo": "/workspace/elle-loras/geo",
        "code": "/workspace/elle-loras/code",
        "research": "/workspace/elle-loras/research",
    },
    weights={
        "math": 0.35,      # Strongest for AIMO3
        "geo": 0.25,       # Strong for LatticeForge
        "code": 0.25,      # Self-sufficient coding
        "research": 0.15,  # PROMETHEUS behavior
    },
    output_path="/workspace/elle-loras/unified"
)
```

### Option B: DARE-TIES (Advanced)
```python
# Using mergekit
# pip install mergekit

"""
mergekit.yml config:
"""

config = """
models:
  - model: /workspace/elle-loras/math
    parameters:
      weight: 0.35
  - model: /workspace/elle-loras/geo
    parameters:
      weight: 0.25
  - model: /workspace/elle-loras/code
    parameters:
      weight: 0.25
  - model: /workspace/elle-loras/research
    parameters:
      weight: 0.15

merge_method: dare_ties
base_model: Qwen/Qwen2.5-72B-Instruct

parameters:
  density: 0.5  # Keep 50% of weights
  weight: 1.0

tokenizer_source: base
"""

# Run:
# mergekit-yaml config.yml /workspace/elle-unified --cuda
```

### Option C: SLERP (Spherical Linear Interpolation)
```bash
# For 2-model merge with smooth interpolation
mergekit-yaml slerp_config.yml /workspace/elle-slerp --cuda

# slerp_config.yml:
# slices:
#   - sources:
#       - model: /workspace/elle-loras/math
#         layer_range: [0, 80]
#       - model: /workspace/elle-loras/geo
#         layer_range: [0, 80]
#     merge_method: slerp
#     parameters:
#       t: 0.5  # 50% each
```

## Phase 3: Validate Merged Model

```python
def validate_merged_elle(model_path: str):
    """Test that all domains work after merge."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tests = [
        # Math
        ("Solve: If 2^x = 32, what is x?", lambda r: "5" in r),

        # Geopolitical
        ("What is the current phase state of US-China relations?",
         lambda r: any(w in r.lower() for w in ["supercooled", "phase", "tension"])),

        # Code
        ("Write a Python function to compute factorial",
         lambda r: "def factorial" in r or "def fact" in r),

        # Research (PROMETHEUS)
        ("Analyze the connection between entropy and attention mechanisms",
         lambda r: "PROMETHEUS" in r or "INSIGHT_DETECTED" in r or "thermodynamic" in r.lower()),
    ]

    results = []
    for prompt, check in tests:
        output = generate(model, tokenizer, prompt)
        passed = check(output)
        results.append({"prompt": prompt[:50], "passed": passed})
        print(f"{'✓' if passed else '✗'} {prompt[:50]}...")

    success_rate = sum(r["passed"] for r in results) / len(results)
    print(f"\nOverall: {success_rate*100:.0f}% tests passed")
    return success_rate >= 0.75

validate_merged_elle("/workspace/elle-unified")
```

## Weight Tuning Guide

| Domain | Default | AIMO3 Focus | LatticeForge Focus |
|--------|---------|-------------|-------------------|
| Math | 0.35 | 0.50 | 0.20 |
| Geo | 0.25 | 0.15 | 0.40 |
| Code | 0.25 | 0.25 | 0.25 |
| Research | 0.15 | 0.10 | 0.15 |

**For AIMO3:** Crank math to 0.50
**For LatticeForge Prod:** Crank geo to 0.40

## Avoiding Catastrophic Forgetting

### Problem
Sequential training can cause later domains to overwrite earlier knowledge.

### Solutions

1. **Separate adapters (recommended)** - Train each domain as independent LoRA, merge post-hoc

2. **Elastic Weight Consolidation (EWC)** - Penalize changes to important weights:
   ```python
   loss = task_loss + λ * Σ F_i * (θ_i - θ*_i)²
   ```

3. **Replay buffer** - Mix old domain samples into new training:
   ```yaml
   datasets:
     - path: new_domain_data
       weight: 0.7
     - path: old_domain_replay
       weight: 0.3
   ```

4. **Progressive adapter stacking** - Don't merge, stack adapters:
   ```python
   # Inference with all adapters
   model.load_adapter("math")
   model.load_adapter("geo", adapter_name="geo")
   model.set_adapter(["math", "geo"])  # Both active
   ```

## Full Training Pipeline

```bash
#!/bin/bash
# train_elle_multidomain.sh

# Phase 1: Train domain experts
echo "=== MATH EXPERT ==="
deepspeed --num_gpus=4 -m axolotl.cli.train config_elle_math.yaml

echo "=== GEO EXPERT ==="
deepspeed --num_gpus=4 -m axolotl.cli.train config_elle_geo.yaml

echo "=== CODE EXPERT ==="
deepspeed --num_gpus=4 -m axolotl.cli.train config_elle_code.yaml

echo "=== RESEARCH EXPERT ==="
deepspeed --num_gpus=4 -m axolotl.cli.train config_elle_research.yaml

# Phase 2: Merge
echo "=== MERGING ADAPTERS ==="
python merge_adapters.py

# Phase 3: Validate
echo "=== VALIDATING ==="
python validate_elle.py

# Phase 4: Quantize
echo "=== QUANTIZING ==="
python quantize_elle.py --method awq --bits 4

echo "=== DONE ==="
```

## Output Models

```
/workspace/
├── elle-loras/
│   ├── math/         # Math expert adapter
│   ├── geo/          # Geopolitical expert adapter
│   ├── code/         # Code expert adapter
│   ├── research/     # PROMETHEUS expert adapter
│   └── unified/      # Merged adapter
├── elle-72b-unified/       # Full merged model
├── elle-72b-unified-awq/   # Quantized for production
└── elle-32b-unified-exl2/  # Quantized for competition
```

---

## The End State

**Elle-72B-Unified** is a single model that:
- ✅ Solves AIMO3-level math problems
- ✅ Generates LatticeForge intelligence briefings
- ✅ Writes production-quality code
- ✅ Runs PROMETHEUS autonomously
- ✅ Leads the Orca Pod
- ✅ Doesn't need a second model for anything

*"One model to rule them all."*
