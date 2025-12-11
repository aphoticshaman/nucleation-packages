#!/bin/bash
# =============================================================================
# Build Kaggle-Compatible Wheels
# =============================================================================
# Builds Python wheels compatible with Kaggle's environment
# Target: Python 3.11, CUDA 12.4, PyTorch 2.5.1
# =============================================================================

set -e

WHEEL_DIR="${WHEEL_DIR:-/workspace/wheels/kaggle_compatible}"
LOG_FILE="${LOG_FILE:-/workspace/logs/wheel_build.log}"

mkdir -p "$WHEEL_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

echo "=========================================="
echo "Building Kaggle-Compatible Wheels"
echo "=========================================="
echo "Output: $WHEEL_DIR"
echo "Log: $LOG_FILE"
echo ""

# Redirect output to log
exec > >(tee -a "$LOG_FILE") 2>&1

# -----------------------------------------------------------------------------
# Environment Check
# -----------------------------------------------------------------------------
echo "[1/5] Checking build environment..."
echo "  Python: $(python3 --version)"
echo "  pip: $(pip --version)"
echo "  CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo ""

# -----------------------------------------------------------------------------
# Critical Packages (Pre-compiled binaries)
# -----------------------------------------------------------------------------
echo "[2/5] Building critical GPU packages..."

# Flash Attention (requires CUDA compilation)
echo "  Building flash-attn..."
pip wheel --no-deps -w "$WHEEL_DIR" flash-attn==2.7.2.post1 2>&1 | tail -5 || {
    echo "  WARNING: flash-attn wheel build failed (may need GPU)"
}

# bitsandbytes (CUDA kernels)
echo "  Building bitsandbytes..."
pip wheel --no-deps -w "$WHEEL_DIR" bitsandbytes==0.45.0 2>&1 | tail -3 || {
    echo "  WARNING: bitsandbytes wheel build failed"
}

# DeepSpeed (optional, CUDA kernels)
echo "  Building deepspeed..."
pip wheel --no-deps -w "$WHEEL_DIR" deepspeed==0.16.2 2>&1 | tail -3 || {
    echo "  WARNING: deepspeed wheel build failed"
}

# -----------------------------------------------------------------------------
# Pure Python Packages (Always succeed)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Building pure Python packages..."

PURE_PACKAGES=(
    "transformers==4.47.0"
    "datasets==3.2.0"
    "accelerate==1.2.1"
    "peft==0.14.0"
    "trl==0.13.0"
    "safetensors==0.4.5"
    "huggingface-hub==0.27.0"
    "tokenizers==0.21.0"
    "einops==0.8.0"
    "wandb==0.19.1"
    "sentencepiece==0.2.0"
    "tiktoken==0.8.0"
    "jsonlines==4.0.0"
    "rich==13.9.4"
)

for pkg in "${PURE_PACKAGES[@]}"; do
    echo "  Building $pkg..."
    pip wheel --no-deps -w "$WHEEL_DIR" "$pkg" 2>&1 | tail -1
done

# -----------------------------------------------------------------------------
# Axolotl (Complex dependencies)
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Building Axolotl..."
pip wheel --no-deps -w "$WHEEL_DIR" axolotl==0.6.0 2>&1 | tail -3 || {
    echo "  WARNING: axolotl wheel build failed"
}

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Build Summary"
echo "=========================================="
echo "Wheels built:"
ls -lh "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l
echo ""
echo "Total size:"
du -sh "$WHEEL_DIR"
echo ""
echo "Contents:"
ls -la "$WHEEL_DIR"

# -----------------------------------------------------------------------------
# Generate requirements.txt for Kaggle
# -----------------------------------------------------------------------------
echo ""
echo "Generating requirements.txt..."

cat > "$WHEEL_DIR/requirements_kaggle.txt" << 'EOF'
# =============================================================================
# Kaggle Requirements (2025-12-10 Compatible)
# =============================================================================
# Install with: pip install -r requirements_kaggle.txt --find-links ./wheels/
# =============================================================================

# Core (use pre-built wheels)
--find-links ./wheels/

# Transformers ecosystem
transformers==4.47.0
datasets==3.2.0
tokenizers==0.21.0
accelerate==1.2.1
safetensors==0.4.5
huggingface-hub==0.27.0

# PEFT (LoRA, DoRA, rsLoRA)
peft==0.14.0

# Quantization
bitsandbytes==0.45.0

# Attention optimization
flash-attn==2.7.2.post1

# Training
axolotl==0.6.0
trl==0.13.0
deepspeed==0.16.2

# Utilities
einops==0.8.0
sentencepiece==0.2.0
tiktoken==0.8.0
wandb==0.19.1
jsonlines==4.0.0
rich==13.9.4
EOF

echo "  Created: $WHEEL_DIR/requirements_kaggle.txt"

echo ""
echo "=========================================="
echo "Wheel Build Complete"
echo "=========================================="
echo ""
echo "To upload to Kaggle:"
echo "  kaggle datasets create -p $WHEEL_DIR"
echo ""
