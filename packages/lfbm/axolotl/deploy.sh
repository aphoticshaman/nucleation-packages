#!/bin/bash
# =============================================================================
# LatticeForge Briefing Model - Full Deployment Script
# =============================================================================
# Prerequisites:
#   - Python 3.10+
#   - HuggingFace account with token: https://huggingface.co/settings/tokens
#   - RunPod account with API key: https://www.runpod.io/console/user/settings
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  LatticeForge Briefing Model Setup${NC}"
echo -e "${BLUE}======================================${NC}"

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}ERROR: HF_TOKEN environment variable not set${NC}"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    echo "export HF_TOKEN=hf_xxxxxxxxxxxx"
    exit 1
fi

if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}ERROR: RUNPOD_API_KEY environment variable not set${NC}"
    echo "Get your key at: https://www.runpod.io/console/user/settings"
    echo "export RUNPOD_API_KEY=xxxxxxxxxxxx"
    exit 1
fi

# Configuration
HF_USERNAME=${HF_USERNAME:-"your-username"}  # Change this!
HF_DATASET_REPO="${HF_USERNAME}/latticeforge-briefing-data"
HF_MODEL_REPO="${HF_USERNAME}/latticeforge-briefing-3b"
AXOLOTL_ENDPOINT_ID=${AXOLOTL_ENDPOINT_ID:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Install dependencies
echo -e "\n${YELLOW}Step 1: Installing Python dependencies...${NC}"
pip install huggingface_hub datasets --quiet

# Step 2: Generate training data
echo -e "\n${YELLOW}Step 2: Generating training data...${NC}"
if [ -f "training_data.jsonl" ]; then
    echo "Using existing training_data.jsonl"
else
    python prepare_data.py training_data.jsonl
fi
EXAMPLE_COUNT=$(wc -l < training_data.jsonl)
echo -e "${GREEN}Generated ${EXAMPLE_COUNT} training examples${NC}"

# Step 3: Upload to HuggingFace
echo -e "\n${YELLOW}Step 3: Uploading dataset to HuggingFace...${NC}"
python upload_to_hf.py training_data.jsonl "$HF_DATASET_REPO"
echo -e "${GREEN}Dataset uploaded to: https://huggingface.co/datasets/${HF_DATASET_REPO}${NC}"

# Step 4: Prepare RunPod request
echo -e "\n${YELLOW}Step 4: Preparing RunPod Axolotl request...${NC}"
cat runpod_request.json | \
    sed "s|\${HF_TOKEN}|$HF_TOKEN|g" | \
    sed "s|\${HF_DATASET_REPO}|$HF_DATASET_REPO|g" | \
    sed "s|\${HF_MODEL_REPO}|$HF_MODEL_REPO|g" \
    > runpod_request_filled.json
echo "Created runpod_request_filled.json"

# Step 5: Submit to RunPod (if endpoint ID is set)
if [ -n "$AXOLOTL_ENDPOINT_ID" ]; then
    echo -e "\n${YELLOW}Step 5: Submitting job to RunPod Axolotl...${NC}"

    RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${AXOLOTL_ENDPOINT_ID}/run" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        -d @runpod_request_filled.json)

    JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))")

    if [ -n "$JOB_ID" ]; then
        echo -e "${GREEN}Job submitted! ID: ${JOB_ID}${NC}"
        echo ""
        echo "Monitor training status with:"
        echo "  curl -s \"https://api.runpod.ai/v2/${AXOLOTL_ENDPOINT_ID}/status/${JOB_ID}\" \\"
        echo "    -H \"Authorization: Bearer \${RUNPOD_API_KEY}\""
        echo ""
        echo "Training will take ~1-2 hours on H200"
    else
        echo -e "${RED}Failed to submit job. Response:${NC}"
        echo "$RESPONSE"
        exit 1
    fi
else
    echo -e "\n${YELLOW}Step 5: Skipping RunPod submission (AXOLOTL_ENDPOINT_ID not set)${NC}"
    echo ""
    echo "To submit manually:"
    echo "1. Go to RunPod -> Serverless -> Axolotl Fine-Tuning"
    echo "2. Create a new endpoint"
    echo "3. Note the endpoint ID"
    echo "4. Run:"
    echo "   export AXOLOTL_ENDPOINT_ID=your-endpoint-id"
    echo "   ./deploy.sh"
    echo ""
    echo "Or submit via API:"
    echo "   curl -X POST \"https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run\" \\"
    echo "     -H \"Authorization: Bearer \${RUNPOD_API_KEY}\" \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d @runpod_request_filled.json"
fi

echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}  What's Next?${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "1. Wait for training to complete (~1-2 hours)"
echo ""
echo "2. The model will be pushed to:"
echo "   https://huggingface.co/${HF_MODEL_REPO}"
echo ""
echo "3. Deploy with vLLM on RunPod:"
echo "   - Go to RunPod -> Serverless -> vLLM"
echo "   - Model: Qwen/Qwen2.5-3B-Instruct"
echo "   - LoRA Adapter: ${HF_MODEL_REPO}"
echo ""
echo "4. Update Vercel env vars:"
echo "   LFBM_ENDPOINT=https://api.runpod.ai/v2/YOUR_VLLM_ENDPOINT"
echo "   LFBM_API_KEY=your-runpod-api-key"
echo "   PREFER_LFBM=true"
echo ""
echo -e "${GREEN}Done!${NC}"
