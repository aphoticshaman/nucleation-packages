#!/bin/bash
# =============================================================================
# Upload Model/Wheels to Kaggle
# =============================================================================
# Uploads trained model or wheels to Kaggle as a dataset
# For use in Kaggle notebooks and competitions
# =============================================================================

set -e

# Configuration
UPLOAD_TYPE="${UPLOAD_TYPE:-model}"  # "model" or "wheels"
MODEL_DIR="${MODEL_DIR:-/workspace/outputs/elle-ultimate}"
WHEELS_DIR="${WHEELS_DIR:-/workspace/wheels/kaggle_compatible}"
KAGGLE_USER="${KAGGLE_USERNAME:-aphoticshaman}"
DATASET_SLUG="${DATASET_SLUG:-elle-72b-ultimate}"

echo "=========================================="
echo "Uploading to Kaggle"
echo "=========================================="
echo "Type: $UPLOAD_TYPE"
echo "User: $KAGGLE_USER"
echo ""

# -----------------------------------------------------------------------------
# Authentication Check
# -----------------------------------------------------------------------------
echo "[1/5] Checking Kaggle authentication..."

if [ ! -f ~/.kaggle/kaggle.json ]; then
    if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        echo "ERROR: Kaggle credentials not configured"
        echo "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
        exit 1
    fi

    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi

kaggle --version
echo ""

# -----------------------------------------------------------------------------
# Select Upload Directory
# -----------------------------------------------------------------------------
echo "[2/5] Preparing upload..."

if [ "$UPLOAD_TYPE" = "wheels" ]; then
    UPLOAD_DIR="$WHEELS_DIR"
    DATASET_SLUG="lfbm-wheels"
    TITLE="LFBM Training Wheels (Kaggle Compatible)"
    SUBTITLE="Pre-built wheels for flash-attn, bitsandbytes, deepspeed, axolotl"
else
    UPLOAD_DIR="$MODEL_DIR"
    TITLE="Elle 72B Ultimate"
    SUBTITLE="Geopolitical intelligence expert fine-tuned from Qwen2.5-72B"
fi

if [ ! -d "$UPLOAD_DIR" ]; then
    echo "ERROR: Upload directory not found: $UPLOAD_DIR"
    exit 1
fi

echo "  Source: $UPLOAD_DIR"
echo "  Dataset: $KAGGLE_USER/$DATASET_SLUG"
echo ""

# -----------------------------------------------------------------------------
# Create Dataset Metadata
# -----------------------------------------------------------------------------
echo "[3/5] Creating dataset metadata..."

METADATA_FILE="$UPLOAD_DIR/dataset-metadata.json"

cat > "$METADATA_FILE" << EOF
{
  "title": "$TITLE",
  "id": "$KAGGLE_USER/$DATASET_SLUG",
  "subtitle": "$SUBTITLE",
  "description": "$(cat << 'DESC'
# Elle: Geopolitical Intelligence Expert

Fine-tuned from Qwen2.5-72B-Instruct using:
- QLoRA (4-bit NF4) with double quantization
- DoRA (Weight-Decomposed Low-Rank Adaptation)
- rsLoRA (Rank-Stabilized LoRA)
- LoRA+ (16x learning rate for B matrix)
- NEFTune (Noisy Embedding Fine-Tuning)

## Capabilities
- Strategic geopolitical analysis
- Macroeconomic and microeconomic analysis
- Wall Street and financial market intelligence
- Multi-perspective conflict assessment
- Evidence-based intelligence synthesis

## Usage in Kaggle Notebook

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/input/elle-72b-ultimate",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/elle-72b-ultimate")
```
DESC
)",
  "isPrivate": false,
  "licenses": [
    {"name": "apache-2.0"}
  ],
  "keywords": [
    "geopolitical",
    "intelligence",
    "economics",
    "qwen",
    "lora",
    "fine-tuning"
  ]
}
EOF

echo "  Created: $METADATA_FILE"
echo ""

# -----------------------------------------------------------------------------
# Upload to Kaggle
# -----------------------------------------------------------------------------
echo "[4/5] Uploading to Kaggle..."
echo "  This may take a while for large models..."
echo ""

# Check if dataset exists
if kaggle datasets list --user "$KAGGLE_USER" | grep -q "$DATASET_SLUG"; then
    echo "  Dataset exists, creating new version..."
    kaggle datasets version -p "$UPLOAD_DIR" -m "Update $(date +%Y%m%d_%H%M%S)" --dir-mode tar
else
    echo "  Creating new dataset..."
    kaggle datasets create -p "$UPLOAD_DIR" --dir-mode tar
fi

echo ""

# -----------------------------------------------------------------------------
# Verify Upload
# -----------------------------------------------------------------------------
echo "[5/5] Verifying upload..."

kaggle datasets status "$KAGGLE_USER/$DATASET_SLUG"

echo ""
echo "=========================================="
echo "Upload Complete"
echo "=========================================="
echo "Dataset URL: https://www.kaggle.com/datasets/$KAGGLE_USER/$DATASET_SLUG"
echo ""
echo "To use in notebook:"
echo "  /kaggle/input/$DATASET_SLUG"
echo ""
