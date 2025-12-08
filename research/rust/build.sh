#!/bin/bash
# Build Rust wheels for RYANAIMO
# Run this on a Linux machine with Rust installed

set -e

echo "==================================="
echo "Building RYANAIMO Rust Components"
echo "==================================="

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust not installed. Install from https://rustup.rs"
    exit 1
fi

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Build ryanaimo-clustering
echo ""
echo "Building ryanaimo-clustering..."
cd ryanaimo-clustering
maturin build --release

echo ""
echo "Build complete! Wheels are in:"
ls -la target/wheels/

echo ""
echo "To install locally:"
echo "  pip install target/wheels/ryanaimo_clustering-*.whl"

echo ""
echo "For Kaggle, copy the wheel to your dataset and install with:"
echo "  pip install /kaggle/input/your-dataset/ryanaimo_clustering-*.whl"
