#!/bin/bash
# =============================================================================
# LFBM RunPod Startup Script
# =============================================================================
# Executed automatically when the pod starts
# Configures environment and validates GPU access
# =============================================================================

set -e

echo "=========================================="
echo "LFBM Training Pod Startup"
echo "=========================================="
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"

# -----------------------------------------------------------------------------
# GPU Validation
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] Validating GPU..."

if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not available. GPU driver issue."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

echo "  GPU Count: $GPU_COUNT"
echo "  GPU Type: $GPU_NAME"
echo "  GPU Memory: $GPU_MEMORY"

# Verify sufficient VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 70000 ]; then
    echo "WARNING: Less than 70GB VRAM detected. Training may OOM."
fi

# -----------------------------------------------------------------------------
# PyTorch CUDA Validation
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] Validating PyTorch CUDA..."

python3 -c "
import torch
print(f'  PyTorch Version: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
print(f'  CUDA Version: {torch.version.cuda}')
print(f'  cuDNN Version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'  Device Name: {torch.cuda.get_device_name(0)}')
    print(f'  Device Capability: {torch.cuda.get_device_capability(0)}')
"

# -----------------------------------------------------------------------------
# Flash Attention Validation
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Validating Flash Attention..."

python3 -c "
try:
    import flash_attn
    print(f'  Flash Attention Version: {flash_attn.__version__}')
    print('  Status: OK')
except ImportError as e:
    print(f'  WARNING: Flash Attention not available: {e}')
    print('  Falling back to standard attention')
"

# -----------------------------------------------------------------------------
# Hugging Face Authentication
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Configuring Hugging Face..."

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
    echo "  HF Token: Configured"
else
    echo "  WARNING: HF_TOKEN not set. Cannot download gated models."
fi

# -----------------------------------------------------------------------------
# Weights & Biases Configuration
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Configuring Weights & Biases..."

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  W&B: Configured (Project: ${WANDB_PROJECT:-lfbm-elle})"
else
    echo "  WARNING: WANDB_API_KEY not set. Logging disabled."
    export WANDB_MODE=disabled
fi

# -----------------------------------------------------------------------------
# Kaggle Configuration
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Configuring Kaggle..."

if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    echo "  Kaggle: Configured (User: $KAGGLE_USERNAME)"
else
    echo "  WARNING: Kaggle credentials not set. Upload disabled."
fi

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Setting up workspace directories..."

mkdir -p /workspace/data
mkdir -p /workspace/models
mkdir -p /workspace/outputs
mkdir -p /workspace/logs/wandb
mkdir -p /workspace/cache/huggingface
mkdir -p /workspace/scripts
mkdir -p /workspace/configs
mkdir -p /workspace/wheels

echo "  Directories created."

# -----------------------------------------------------------------------------
# Pre-download Model (Optional)
# -----------------------------------------------------------------------------
if [ "$PRELOAD_MODEL" = "true" ]; then
    echo ""
    echo "[OPTIONAL] Pre-downloading base model..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-72B-Instruct',
                  local_dir='/workspace/models/qwen-72b-base',
                  ignore_patterns=['*.bin'])  # Prefer safetensors
"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Startup Complete"
echo "=========================================="
echo ""
echo "Quick Commands:"
echo "  Train Elle:  bash /workspace/scripts/train_elle_ultimate.sh"
echo "  Monitor:     nvtop"
echo "  Logs:        tail -f /workspace/logs/training.log"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
echo ""
