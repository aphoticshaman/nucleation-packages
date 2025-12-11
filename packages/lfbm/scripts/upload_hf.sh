#!/bin/bash
# =============================================================================
# Upload Model to Hugging Face Hub
# =============================================================================
# Uploads trained model and adapters to HuggingFace
# Supports both full model and LoRA adapter uploads
# =============================================================================

set -e

# Configuration
MODEL_DIR="${MODEL_DIR:-/workspace/outputs/elle-ultimate}"
REPO_ID="${REPO_ID:-aphoticshaman/elle-72b-ultimate}"
BRANCH="${BRANCH:-main}"
COMMIT_MSG="${COMMIT_MSG:-Upload Elle model $(date +%Y%m%d_%H%M%S)}"

echo "=========================================="
echo "Uploading to Hugging Face Hub"
echo "=========================================="
echo "Model Dir: $MODEL_DIR"
echo "Repo: $REPO_ID"
echo "Branch: $BRANCH"
echo ""

# -----------------------------------------------------------------------------
# Authentication Check
# -----------------------------------------------------------------------------
echo "[1/5] Checking authentication..."

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo "Set it with: export HF_TOKEN=your_token"
    exit 1
fi

huggingface-cli whoami || {
    echo "Authenticating..."
    huggingface-cli login --token "$HF_TOKEN"
}
echo ""

# -----------------------------------------------------------------------------
# Validate Model Directory
# -----------------------------------------------------------------------------
echo "[2/5] Validating model directory..."

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check for essential files
ESSENTIAL_FILES=("config.json" "tokenizer.json")
for f in "${ESSENTIAL_FILES[@]}"; do
    if [ ! -f "$MODEL_DIR/$f" ]; then
        echo "WARNING: Missing $f in model directory"
    fi
done

# Check model size
MODEL_SIZE=$(du -sh "$MODEL_DIR" | cut -f1)
echo "  Model size: $MODEL_SIZE"
echo ""

# -----------------------------------------------------------------------------
# Create/Verify Repository
# -----------------------------------------------------------------------------
echo "[3/5] Creating/verifying repository..."

python3 << EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = "${REPO_ID}"

try:
    create_repo(repo_id, exist_ok=True, private=False)
    print(f"  Repository ready: {repo_id}")
except Exception as e:
    print(f"  Repository exists or error: {e}")
EOF
echo ""

# -----------------------------------------------------------------------------
# Upload Files
# -----------------------------------------------------------------------------
echo "[4/5] Uploading files..."

python3 << EOF
from huggingface_hub import HfApi, upload_folder
import os

api = HfApi()
repo_id = "${REPO_ID}"
model_dir = "${MODEL_DIR}"
commit_msg = "${COMMIT_MSG}"

print(f"  Uploading from: {model_dir}")
print(f"  To: {repo_id}")
print("")

# Upload the entire folder
try:
    url = upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        commit_message=commit_msg,
        ignore_patterns=["*.bin", "optimizer.*", "scheduler.*", "trainer_state.json"],
    )
    print(f"  Upload complete!")
    print(f"  URL: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"  ERROR: {e}")
    raise
EOF
echo ""

# -----------------------------------------------------------------------------
# Create Model Card
# -----------------------------------------------------------------------------
echo "[5/5] Creating model card..."

python3 << EOF
from huggingface_hub import HfApi

api = HfApi()
repo_id = "${REPO_ID}"

model_card = '''---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- geopolitical
- intelligence
- analysis
- qwen
- lora
- dora
base_model: Qwen/Qwen2.5-72B-Instruct
pipeline_tag: text-generation
---

# Elle: Geopolitical Intelligence Expert

Elle is a fine-tuned version of Qwen2.5-72B-Instruct, specialized for geopolitical
intelligence analysis with post-doctoral expertise and junior analyst motivation.

## Model Details

- **Base Model:** Qwen/Qwen2.5-72B-Instruct
- **Training Method:** QLoRA (4-bit NF4) with DoRA, rsLoRA, LoRA+, NEFTune
- **LoRA Rank:** 128
- **Training Data:** LatticeForge briefing data + NuminaMath-CoT + OpenOrca

## Capabilities

- Strategic geopolitical analysis
- Multi-perspective conflict assessment
- Second and third-order effect prediction
- Evidence-based intelligence synthesis
- Novel connection identification

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("''' + repo_id + '''")
tokenizer = AutoTokenizer.from_pretrained("''' + repo_id + '''")

messages = [
    {"role": "system", "content": "You are Elle, a senior geopolitical intelligence analyst."},
    {"role": "user", "content": "Analyze the strategic implications of Arctic shipping route expansion."}
]

response = model.chat(tokenizer, messages)
print(response)
```

## Training Infrastructure

- **GPU:** NVIDIA H200 (141GB HBM3e)
- **Framework:** Axolotl + PEFT
- **Optimization:** Flash Attention 2, DeepSpeed ZeRO-2

## License

Apache 2.0
'''

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    commit_message="Add model card"
)
print("  Model card uploaded")
EOF

echo ""
echo "=========================================="
echo "Upload Complete"
echo "=========================================="
echo "View at: https://huggingface.co/$REPO_ID"
echo ""
