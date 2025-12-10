# Kaggle Wheel Building Pipeline

## The Problem

Kaggle's default environment has conflicting versions:
- torch 2.1.x (we need 2.2+)
- transformers 4.35.x (we need 4.40+)
- bitsandbytes old version
- triton conflicts

Solution: Build wheels on RunPod, upload to Kaggle as dataset, install from wheels.

## RunPod Build Script

Run this on a fresh RunPod instance (H100 recommended for CUDA compatibility):

```bash
#!/bin/bash
# build_kaggle_wheels.sh

set -e

# Match Kaggle's Python version
PYTHON_VERSION=3.10

# Create clean env
conda create -n kaggle_build python=$PYTHON_VERSION -y
conda activate kaggle_build

# Create wheel output directory
mkdir -p /workspace/kaggle_wheels

# ============================================
# CORE DEPENDENCIES
# ============================================

# PyTorch with CUDA 12.1 (matches Kaggle H100)
pip download torch==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121 -d /workspace/kaggle_wheels

# Transformers + accelerate + peft
pip wheel transformers==4.44.0 accelerate==0.33.0 peft==0.12.0 \
    --wheel-dir /workspace/kaggle_wheels

# bitsandbytes (critical for NF4)
pip wheel bitsandbytes==0.43.1 --wheel-dir /workspace/kaggle_wheels

# Flash attention (if compatible)
pip wheel flash-attn==2.5.8 --wheel-dir /workspace/kaggle_wheels || echo "Flash attention build failed - will skip"

# vLLM (optional, for inference)
pip wheel vllm==0.4.0 --wheel-dir /workspace/kaggle_wheels || echo "vLLM build skipped"

# ============================================
# RYANSTREAM DEPS
# ============================================

pip wheel triton==2.2.0 --wheel-dir /workspace/kaggle_wheels
pip wheel einops==0.7.0 --wheel-dir /workspace/kaggle_wheels
pip wheel sentencepiece==0.2.0 --wheel-dir /workspace/kaggle_wheels
pip wheel tiktoken==0.6.0 --wheel-dir /workspace/kaggle_wheels

# Sympy for verification
pip wheel sympy==1.12 --wheel-dir /workspace/kaggle_wheels

# ============================================
# PACKAGE LIST
# ============================================

ls -la /workspace/kaggle_wheels/ > /workspace/kaggle_wheels/MANIFEST.txt
echo "Built $(ls /workspace/kaggle_wheels/*.whl | wc -l) wheels"

# ============================================
# CREATE INSTALL SCRIPT
# ============================================

cat > /workspace/kaggle_wheels/install_on_kaggle.py << 'INSTALLER'
#!/usr/bin/env python3
"""
Kaggle Wheel Installer
======================
Run this FIRST in your Kaggle notebook to set up the environment.
"""

import subprocess
import sys
import os

def run(cmd):
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARN: {result.stderr}")
    return result.returncode == 0

print("=" * 60)
print("RYANSTREAM Kaggle Installer")
print("=" * 60)

# Step 1: Uninstall conflicting packages
CONFLICTS = [
    "torch", "torchvision", "torchaudio",
    "transformers", "accelerate", "peft",
    "bitsandbytes", "triton",
    "vllm", "flash-attn"
]

print("\n[1/3] Uninstalling conflicting packages...")
for pkg in CONFLICTS:
    run(f"{sys.executable} -m pip uninstall -y {pkg}")

# Step 2: Install from wheels
WHEEL_DIR = "/kaggle/input/ryanstream-wheels"  # Kaggle dataset path

print(f"\n[2/3] Installing from {WHEEL_DIR}...")

if not os.path.exists(WHEEL_DIR):
    print(f"ERROR: Wheel directory not found: {WHEEL_DIR}")
    print("Make sure you've added 'ryanstream-wheels' dataset to your notebook")
    sys.exit(1)

# Install in order (dependencies first)
INSTALL_ORDER = [
    "torch-*.whl",
    "triton-*.whl",
    "transformers-*.whl",
    "accelerate-*.whl",
    "peft-*.whl",
    "bitsandbytes-*.whl",
    "einops-*.whl",
    "sentencepiece-*.whl",
    "tiktoken-*.whl",
    "sympy-*.whl",
]

import glob

for pattern in INSTALL_ORDER:
    wheels = glob.glob(os.path.join(WHEEL_DIR, pattern))
    for wheel in wheels:
        run(f"{sys.executable} -m pip install --no-deps {wheel}")

# Install remaining wheels
all_wheels = glob.glob(os.path.join(WHEEL_DIR, "*.whl"))
installed = set()
for pattern in INSTALL_ORDER:
    installed.update(glob.glob(os.path.join(WHEEL_DIR, pattern)))

remaining = set(all_wheels) - installed
for wheel in remaining:
    run(f"{sys.executable} -m pip install --no-deps {wheel}")

# Step 3: Verify
print("\n[3/3] Verifying installation...")

try:
    import torch
    print(f"  torch: {torch.__version__} CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"  torch: FAILED - {e}")

try:
    import transformers
    print(f"  transformers: {transformers.__version__}")
except ImportError as e:
    print(f"  transformers: FAILED - {e}")

try:
    import bitsandbytes as bnb
    print(f"  bitsandbytes: {bnb.__version__}")
except ImportError as e:
    print(f"  bitsandbytes: FAILED - {e}")

print("\n" + "=" * 60)
print("Installation complete!")
print("=" * 60)
INSTALLER

chmod +x /workspace/kaggle_wheels/install_on_kaggle.py

echo ""
echo "============================================"
echo "BUILD COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Zip the wheels: cd /workspace && zip -r kaggle_wheels.zip kaggle_wheels/"
echo "2. Upload to Kaggle as dataset: 'ryanstream-wheels'"
echo "3. In your notebook, run: exec(open('/kaggle/input/ryanstream-wheels/install_on_kaggle.py').read())"
echo ""
```

## Kaggle Notebook Usage

```python
# Cell 1: Install dependencies (run once)
exec(open('/kaggle/input/ryanstream-wheels/install_on_kaggle.py').read())

# Cell 2: Restart kernel, then import
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Cell 3: Load your model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "aphoticshaman/qwen-72b-math-nf4",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

## Troubleshooting

### "No module named 'bitsandbytes'"
The wheel needs CUDA libs. Make sure you're using a GPU notebook.

### "CUDA out of memory"
Use tensor parallelism or reduce batch size.

### "Triton compilation error"
Triton needs exact CUDA version match. Fall back to eager mode:
```python
model = model.to(dtype=torch.bfloat16)  # Skip flash attention
```
