#!/bin/bash
# =============================================================================
# CREATE COMPLETE WHEEL DATASET FOR KAGGLE (RUN ON RUNPOD)
# =============================================================================
# This script downloads vLLM + ALL dependencies for offline Kaggle use
#
# KAGGLE ENVIRONMENT (must match):
#   - Python 3.10
#   - Linux (manylinux2014 / manylinux_2_17)
#   - CUDA 12.4
#   - PyTorch 2.5.1+cu124 (pre-installed on Kaggle)
#
# USAGE:
#   1. SSH into RunPod
#   2. Save this script as: /workspace/runpod_create_wheels.sh
#   3. chmod +x /workspace/runpod_create_wheels.sh
#   4. ./runpod_create_wheels.sh
#   5. Download the zip and upload to Kaggle as: ryanaimo-vllm-wheels
# =============================================================================

set -e

# Create clean directory
rm -rf /workspace/ryanaimo_vllm_wheels
mkdir -p /workspace/ryanaimo_vllm_wheels
cd /workspace/ryanaimo_vllm_wheels

echo "=== Creating complete vLLM wheel dataset for Kaggle ==="
echo "Target: Python 3.10, manylinux, CUDA 12.x"
echo ""

# Create requirements file with ALL dependencies
cat > requirements.txt << 'EOF'
# === vLLM core ===
vllm>=0.6.0

# === vLLM dependencies ===
transformers>=4.40.0
accelerate>=0.25.0
safetensors>=0.4.0
msgspec>=0.18.0
sentencepiece>=0.2.0
tokenizers>=0.19.0
huggingface_hub>=0.23.0
pynvml
outlines
lm-format-enforcer
xformers
ray
aiohttp
uvloop
prometheus-client
fastapi

# === Notebook dependencies ===
numpy>=1.24.0
polars>=0.20.0
sympy>=1.12
scipy>=1.10.0

# === Math/execution support ===
mpmath
gmpy2

# === kaggle_evaluation deps ===
grpcio
protobuf
EOF

echo "=== Downloading wheels for Python 3.10, Linux ==="
echo ""

# Download with platform specification for Kaggle compatibility
pip download \
    --python-version 3.10 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 \
    --platform linux_x86_64 \
    --only-binary=:all: \
    -r requirements.txt \
    -d . 2>&1 | tee download.log || true

# Some packages may not have binary wheels - download as fallback
echo ""
echo "=== Downloading any remaining packages (source fallback) ==="
pip download \
    --python-version 3.10 \
    -r requirements.txt \
    -d . 2>&1 | tee -a download.log || true

# Note: NOT downloading torch - Kaggle has it pre-installed (saves ~2GB)

# Results
echo ""
echo "=== RESULTS ==="
WHL_COUNT=$(ls -1 *.whl 2>/dev/null | wc -l)
TAR_COUNT=$(ls -1 *.tar.gz 2>/dev/null | wc -l)
echo "Wheel files: $WHL_COUNT"
echo "Source tarballs: $TAR_COUNT"
echo ""
ls -lhS | head -40
echo "..."
echo ""
du -sh .

# Create manifest
ls -1 > manifest.txt
echo ""
echo "Manifest saved to manifest.txt"

# Verify key packages
echo ""
echo "=== KEY PACKAGES CHECK ==="
for pkg in vllm msgspec transformers accelerate tokenizers polars sympy; do
    if ls *${pkg}* >/dev/null 2>&1; then
        echo "  [OK] $pkg: $(ls *${pkg}* | head -1)"
    else
        echo "  [MISSING] $pkg"
    fi
done

# Create zip for upload
echo ""
echo "=== CREATING ZIP ==="
cd /workspace
rm -f ryanaimo_vllm_wheels.zip
zip -r ryanaimo_vllm_wheels.zip ryanaimo_vllm_wheels/
ls -lh ryanaimo_vllm_wheels.zip

echo ""
echo "=== DONE ==="
echo ""
echo "NEXT STEPS:"
echo "1. Download: /workspace/ryanaimo_vllm_wheels.zip"
echo "2. Upload to Kaggle as dataset: ryanaimo-vllm-wheels"
echo "3. In notebook, attach: ryanaimo-vllm-wheels"
echo ""
echo "Notebook install code:"
echo '  pip install --no-index --find-links=/kaggle/input/ryanaimo-vllm-wheels vllm transformers accelerate polars sympy'
