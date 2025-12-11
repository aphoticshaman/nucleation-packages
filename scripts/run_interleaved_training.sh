#!/bin/bash
# Elle-72B Interleaved Training Master Script
# Run this on RunPod with 2xH100 or similar
set -e

echo "=========================================="
echo "Elle-72B Interleaved Training"
echo "Code + PROMETHEUS + NSM + XYZA + SDPM"
echo "=========================================="

# 1. Install dependencies
echo ""
echo "[1/5] Installing dependencies..."
pip install -q axolotl accelerate deepspeed bitsandbytes datasets transformers
pip install -q flash-attn --no-build-isolation

# 2. Create workspace
echo ""
echo "[2/5] Setting up workspace..."
mkdir -p /workspace/datasets
mkdir -p /workspace/configs

# 3. Copy files from nucleation-packages (assumes they're in current dir or mounted)
echo ""
echo "[3/5] Copying config files..."
# If running from nucleation-packages directory:
cp configs/config_interleaved.yaml /workspace/
cp configs/ds_config_interleaved.json /workspace/ds_config.json
cp datasets/prometheus_nsm_training.jsonl /workspace/datasets/
cp scripts/prepare_interleaved_training.py /workspace/

# 4. Prepare interleaved dataset
echo ""
echo "[4/5] Preparing interleaved dataset..."
cd /workspace
python prepare_interleaved_training.py

# 5. Start training
echo ""
echo "[5/5] Starting training..."
echo "=========================================="

accelerate launch -m axolotl.cli.train /workspace/config_interleaved.yaml

echo ""
echo "=========================================="
echo "Training complete!"
echo "Adapter saved to: /workspace/elle-interleaved-out"
echo "=========================================="
