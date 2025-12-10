#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# KAGGLE WHEEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
# Run on RunPod H100 to build compatible wheels for Kaggle
#
# Usage:
#   chmod +x build_wheels.sh
#   ./build_wheels.sh
#
# Output: /workspace/kaggle_wheels/ ready for Kaggle dataset upload
# ═══════════════════════════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo " KAGGLE WHEEL BUILDER - RYANSTREAM Edition"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Match Kaggle's environment
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
TORCH_VERSION="2.2.2"

WHEEL_DIR="/workspace/kaggle_wheels"
mkdir -p "$WHEEL_DIR"

echo "[1/6] Setting up build environment..."
echo "──────────────────────────────────────────────────────────────────────────────"

# Check if conda exists, otherwise use pip directly
if command -v conda &> /dev/null; then
    conda create -n kaggle_build python=$PYTHON_VERSION -y 2>/dev/null || true
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate kaggle_build
fi

pip install --upgrade pip wheel setuptools

echo ""
echo "[2/6] Downloading PyTorch wheels..."
echo "──────────────────────────────────────────────────────────────────────────────"

pip download \
    torch==${TORCH_VERSION}+cu121 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    -d "$WHEEL_DIR" \
    --python-version $PYTHON_VERSION \
    --platform manylinux2014_x86_64 \
    --only-binary=:all:

echo ""
echo "[3/6] Building transformers ecosystem wheels..."
echo "──────────────────────────────────────────────────────────────────────────────"

pip wheel \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    peft==0.12.0 \
    datasets==2.20.0 \
    tokenizers==0.19.1 \
    safetensors==0.4.3 \
    huggingface-hub==0.24.0 \
    --wheel-dir "$WHEEL_DIR"

echo ""
echo "[4/6] Building quantization wheels..."
echo "──────────────────────────────────────────────────────────────────────────────"

# bitsandbytes - critical for NF4
pip wheel bitsandbytes==0.43.1 --wheel-dir "$WHEEL_DIR"

# Try flash-attn (may fail without CUDA headers)
pip wheel flash-attn==2.5.8 --wheel-dir "$WHEEL_DIR" 2>/dev/null || \
    echo "⚠ flash-attn build skipped (needs CUDA dev headers)"

echo ""
echo "[5/6] Building RYANSTREAM dependencies..."
echo "──────────────────────────────────────────────────────────────────────────────"

pip wheel \
    triton==2.2.0 \
    einops==0.7.0 \
    sentencepiece==0.2.0 \
    tiktoken==0.6.0 \
    sympy==1.12 \
    scipy==1.13.0 \
    scikit-learn==1.4.0 \
    --wheel-dir "$WHEEL_DIR"

echo ""
echo "[6/6] Creating installer script..."
echo "──────────────────────────────────────────────────────────────────────────────"

cat > "$WHEEL_DIR/install.py" << 'EOF'
#!/usr/bin/env python3
"""
RYANSTREAM Kaggle Installer
============================
Run this FIRST in your Kaggle notebook.

Usage:
    # In Kaggle notebook cell:
    import sys
    sys.path.insert(0, '/kaggle/input/ryanstream-wheels')
    from install import install
    install()
    # Then restart kernel
"""

import subprocess
import sys
import os
import glob

def run(cmd, check=False):
    """Run shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"    ERROR: {result.stderr[:200]}")
    return result.returncode == 0

def install(wheel_dir=None):
    """Install RYANSTREAM wheels on Kaggle."""

    if wheel_dir is None:
        # Auto-detect
        candidates = [
            '/kaggle/input/ryanstream-wheels',
            '/kaggle/input/kaggle-wheels',
            os.path.dirname(os.path.abspath(__file__)),
        ]
        for c in candidates:
            if os.path.exists(c) and glob.glob(os.path.join(c, '*.whl')):
                wheel_dir = c
                break

    if not wheel_dir or not os.path.exists(wheel_dir):
        print("ERROR: Wheel directory not found!")
        print("Add 'ryanstream-wheels' dataset to your notebook.")
        return False

    print("=" * 70)
    print(" RYANSTREAM INSTALLER")
    print("=" * 70)
    print(f" Wheel dir: {wheel_dir}")
    print(f" Python: {sys.version}")
    print("=" * 70)

    # Step 1: Uninstall conflicts
    print("\n[1/4] Removing conflicting packages...")
    conflicts = [
        'torch', 'torchvision', 'torchaudio',
        'transformers', 'accelerate', 'peft',
        'bitsandbytes', 'triton', 'flash-attn',
        'tokenizers', 'safetensors', 'huggingface-hub',
    ]
    for pkg in conflicts:
        run(f"{sys.executable} -m pip uninstall -y {pkg} 2>/dev/null")

    # Step 2: Install torch first (has no deps)
    print("\n[2/4] Installing PyTorch...")
    torch_wheels = glob.glob(os.path.join(wheel_dir, 'torch-*.whl'))
    for wheel in sorted(torch_wheels):
        run(f"{sys.executable} -m pip install --no-deps '{wheel}'")

    # Step 3: Install other wheels
    print("\n[3/4] Installing dependencies...")

    # Order matters
    install_order = [
        'triton-*.whl',
        'tokenizers-*.whl',
        'safetensors-*.whl',
        'huggingface_hub-*.whl',
        'transformers-*.whl',
        'accelerate-*.whl',
        'peft-*.whl',
        'bitsandbytes-*.whl',
        'einops-*.whl',
        'sentencepiece-*.whl',
        'tiktoken-*.whl',
        'sympy-*.whl',
    ]

    installed = set()
    for pattern in install_order:
        matches = glob.glob(os.path.join(wheel_dir, pattern))
        for wheel in matches:
            if wheel not in installed:
                run(f"{sys.executable} -m pip install --no-deps '{wheel}'")
                installed.add(wheel)

    # Remaining wheels
    all_wheels = set(glob.glob(os.path.join(wheel_dir, '*.whl')))
    remaining = all_wheels - installed
    for wheel in remaining:
        if 'torch' not in os.path.basename(wheel):  # Already installed
            run(f"{sys.executable} -m pip install --no-deps '{wheel}'")

    # Step 4: Verify
    print("\n[4/4] Verifying installation...")
    print("-" * 50)

    checks = [
        ("torch", "import torch; print(f'  torch {torch.__version__} CUDA={torch.cuda.is_available()}')"),
        ("transformers", "import transformers; print(f'  transformers {transformers.__version__}')"),
        ("bitsandbytes", "import bitsandbytes; print(f'  bitsandbytes OK')"),
        ("peft", "import peft; print(f'  peft {peft.__version__}')"),
    ]

    all_ok = True
    for name, check_code in checks:
        try:
            exec(check_code)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            all_ok = False

    print("-" * 50)
    if all_ok:
        print("\n✓ Installation complete!")
        print("  IMPORTANT: Restart the kernel before importing.")
    else:
        print("\n⚠ Some packages failed. Check errors above.")

    return all_ok


if __name__ == '__main__':
    install()
EOF

# Create manifest
echo ""
echo "Creating manifest..."
ls -la "$WHEEL_DIR"/*.whl > "$WHEEL_DIR/MANIFEST.txt" 2>/dev/null || true
echo "$(ls "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l) wheels built"

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo " BUILD COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo " Output: $WHEEL_DIR"
echo ""
echo " Next steps:"
echo "   1. cd /workspace && zip -r ryanstream-wheels.zip kaggle_wheels/"
echo "   2. Upload to Kaggle: kaggle datasets create -p kaggle_wheels/"
echo "   3. In notebook: from install import install; install()"
echo ""
