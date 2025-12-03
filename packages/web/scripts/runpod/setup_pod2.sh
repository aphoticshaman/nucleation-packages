#!/bin/bash
# Pod 2 Setup Script - Run this after spinning up a new H100 pod
# Template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

echo "=========================================="
echo "POD 2 SETUP - Finance + Intel Training"
echo "=========================================="

# Install dependencies
echo "[1/3] Installing Python packages..."
pip install transformers peft datasets accelerate bitsandbytes -q

# Verify CUDA
echo "[2/3] Checking CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "[3/3] Setup complete!"
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo "1. Upload these files to /workspace/:"
echo "   - finance_training_data.json"
echo "   - global_intel_training.json"
echo "   - train_combined.py"
echo ""
echo "2. Start training:"
echo "   python train_combined.py"
echo ""
echo "3. After BOTH pods finish, copy lattice-lora from Pod 1 here"
echo "4. Run: python merge_loras.py"
echo "=========================================="
