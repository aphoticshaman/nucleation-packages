#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD ELLE: Push trained adapter to Kaggle AND HuggingFace
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./upload_elle.sh <adapter_path> <version_name> [--skip-kaggle] [--skip-hf]
#
# Examples:
#   ./upload_elle.sh /workspace/outputs/elle-math elle-72b-math-v1
#   ./upload_elle.sh /workspace/outputs/elle-geo elle-72b-geo-v1
#   ./upload_elle.sh /workspace/outputs/elle-unified elle-72b-unified-v1
#
# Prerequisites:
#   - huggingface-cli login (HF_TOKEN env var or interactive)
#   - kaggle.json in ~/.kaggle/ with API credentials
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Config
HF_USERNAME="${HF_USERNAME:-aphoticshaman}"
KAGGLE_USERNAME="${KAGGLE_USERNAME:-aphoticshaman}"

# Parse args
ADAPTER_PATH="$1"
VERSION_NAME="$2"
SKIP_KAGGLE=false
SKIP_HF=false

for arg in "$@"; do
    case $arg in
        --skip-kaggle) SKIP_KAGGLE=true ;;
        --skip-hf) SKIP_HF=true ;;
    esac
done

# Validate
if [ -z "$ADAPTER_PATH" ] || [ -z "$VERSION_NAME" ]; then
    echo -e "${RED}Usage: ./upload_elle.sh <adapter_path> <version_name>${NC}"
    echo "Example: ./upload_elle.sh /workspace/outputs/elle-math elle-72b-math-v1"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo -e "${RED}Error: Adapter path does not exist: $ADAPTER_PATH${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}       UPLOADING ELLE: $VERSION_NAME${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Adapter path: ${GREEN}$ADAPTER_PATH${NC}"
echo -e "Version name: ${GREEN}$VERSION_NAME${NC}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_HF" = false ]; then
    echo -e "${YELLOW}[1/2] Uploading to HuggingFace...${NC}"

    HF_REPO="$HF_USERNAME/$VERSION_NAME"

    # Check if logged in
    if ! huggingface-cli whoami &> /dev/null; then
        echo -e "${RED}Not logged into HuggingFace. Run: huggingface-cli login${NC}"
        exit 1
    fi

    # Create repo if doesn't exist
    echo "Creating/checking repo: $HF_REPO"
    huggingface-cli repo create "$VERSION_NAME" --type model 2>/dev/null || true

    # Create model card
    cat > "$ADAPTER_PATH/README.md" << EOF
---
license: apache-2.0
base_model: Qwen/Qwen2.5-72B-Instruct
tags:
  - elle
  - latticeforge
  - lora
  - qwen
  - fine-tuned
---

# $VERSION_NAME

Elle fine-tuned adapter for LatticeForge.

## Base Model
Qwen/Qwen2.5-72B-Instruct

## Training
- Framework: Axolotl
- Method: QLoRA (4-bit NF4)
- LoRA rank: 64
- LoRA alpha: 128

## Usage

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-72B-Instruct",
    device_map="auto",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base_model, "$HF_REPO")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
\`\`\`

## Part of Elle Series
- elle-72b-math-v1: Math expert
- elle-72b-geo-v1: Geopolitical expert
- elle-72b-code-v1: Code expert
- elle-72b-research-v1: PROMETHEUS research expert
- elle-72b-unified-v1: Merged all experts

## License
Apache 2.0
EOF

    # Upload
    echo "Uploading to $HF_REPO..."
    huggingface-cli upload "$HF_REPO" "$ADAPTER_PATH" . --revision main

    echo -e "${GREEN}✓ HuggingFace upload complete: https://huggingface.co/$HF_REPO${NC}"
    echo ""
else
    echo -e "${YELLOW}[1/2] Skipping HuggingFace upload${NC}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# KAGGLE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$SKIP_KAGGLE" = false ]; then
    echo -e "${YELLOW}[2/2] Uploading to Kaggle...${NC}"

    KAGGLE_DATASET="$KAGGLE_USERNAME/$VERSION_NAME"

    # Check kaggle credentials
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        echo -e "${RED}Kaggle credentials not found. Create ~/.kaggle/kaggle.json${NC}"
        exit 1
    fi

    # Create dataset metadata
    KAGGLE_SLUG=$(echo "$VERSION_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g')

    cat > "$ADAPTER_PATH/dataset-metadata.json" << EOF
{
    "title": "$VERSION_NAME",
    "id": "$KAGGLE_USERNAME/$KAGGLE_SLUG",
    "licenses": [{"name": "Apache 2.0"}],
    "keywords": ["elle", "latticeforge", "lora", "qwen", "fine-tuned", "aimo3"]
}
EOF

    # Create or update dataset
    echo "Uploading to Kaggle: $KAGGLE_USERNAME/$KAGGLE_SLUG"

    # Try to create new dataset, if exists update it
    if kaggle datasets status "$KAGGLE_USERNAME/$KAGGLE_SLUG" &> /dev/null; then
        echo "Dataset exists, creating new version..."
        kaggle datasets version -p "$ADAPTER_PATH" -m "Update: $VERSION_NAME" --dir-mode zip
    else
        echo "Creating new dataset..."
        kaggle datasets create -p "$ADAPTER_PATH" --dir-mode zip
    fi

    echo -e "${GREEN}✓ Kaggle upload complete: https://kaggle.com/datasets/$KAGGLE_USERNAME/$KAGGLE_SLUG${NC}"
    echo ""
else
    echo -e "${YELLOW}[2/2] Skipping Kaggle upload${NC}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}       UPLOAD COMPLETE: $VERSION_NAME${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
if [ "$SKIP_HF" = false ]; then
    echo -e "HuggingFace: ${GREEN}https://huggingface.co/$HF_USERNAME/$VERSION_NAME${NC}"
fi
if [ "$SKIP_KAGGLE" = false ]; then
    echo -e "Kaggle:      ${GREEN}https://kaggle.com/datasets/$KAGGLE_USERNAME/$KAGGLE_SLUG${NC}"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Verify uploads on both platforms"
echo "  2. Test loading the adapter"
echo "  3. Start next training run (geo/code/research)"
echo ""
