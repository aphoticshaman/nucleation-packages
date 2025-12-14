#!/bin/bash
# =============================================================================
# RUNPOD FULL SETUP: Fine-tune + Quantize + Upload to Kaggle
# =============================================================================
#
# TWO-MODEL STRATEGY:
#   Model A: DeepSeek-R1 - Native <think> reasoning, algebraic proofs
#   Model B: Qwen2.5-Math-Coder - Code execution, computational brute force
#
# Each covers the other's weakness:
#   - DeepSeek: Deep reasoning, symbolic manipulation, proof construction
#   - Qwen-Coder: Python execution, enumeration, numerical verification
#
# =============================================================================

set -e

echo "=============================================="
echo "RUNPOD AIMO3 FULL SETUP"
echo "=============================================="

# =============================================================================
# STEP 1: KAGGLE CLI SETUP
# =============================================================================
echo ""
echo "=== STEP 1: KAGGLE CLI SETUP ==="
echo ""

pip install --quiet kaggle

# Create kaggle config directory
mkdir -p ~/.kaggle

echo "Enter your Kaggle credentials:"
echo "(Find at: kaggle.com -> Settings -> API -> Create New Token)"
echo ""
read -p "KAGGLE_USERNAME: " KAGGLE_USER
read -sp "KAGGLE_KEY: " KAGGLE_KEY
echo ""

# Write kaggle.json
cat > ~/.kaggle/kaggle.json << EOF
{"username": "$KAGGLE_USER", "key": "$KAGGLE_KEY"}
EOF
chmod 600 ~/.kaggle/kaggle.json

# Verify
echo "Verifying Kaggle connection..."
kaggle datasets list --mine | head -5
echo "Kaggle CLI ready!"

# =============================================================================
# STEP 2: CREATE WHEEL DATASET
# =============================================================================
echo ""
echo "=== STEP 2: CREATING WHEEL DATASET ==="
echo ""

rm -rf /workspace/ryanaimo-vllm-wheels
mkdir -p /workspace/ryanaimo-vllm-wheels
cd /workspace/ryanaimo-vllm-wheels

# Download all wheels
pip download \
    --python-version 3.10 \
    --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    vllm transformers accelerate safetensors msgspec \
    sentencepiece tokenizers huggingface_hub pynvml \
    ray aiohttp uvloop polars sympy scipy mpmath \
    outlines lm-format-enforcer xformers \
    grpcio protobuf -d . 2>&1 | tee download.log || true

# Fallback
pip download --python-version 3.10 \
    vllm transformers accelerate polars sympy msgspec -d . 2>&1 || true

echo "Downloaded $(ls *.whl 2>/dev/null | wc -l) wheels"

# Create dataset metadata
cat > dataset-metadata.json << 'EOF'
{
  "title": "ryanaimo-vllm-wheels",
  "id": "KAGGLE_USER/ryanaimo-vllm-wheels",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
sed -i "s/KAGGLE_USER/$KAGGLE_USER/g" dataset-metadata.json

# Upload to Kaggle
echo "Uploading wheels to Kaggle..."
kaggle datasets create -p /workspace/ryanaimo-vllm-wheels --dir-mode zip
echo "Wheels uploaded!"

# =============================================================================
# STEP 3: DOWNLOAD BASE MODELS
# =============================================================================
echo ""
echo "=== STEP 3: DOWNLOADING BASE MODELS ==="
echo ""

pip install --quiet huggingface_hub[cli]

# DeepSeek-R1-Distill-Qwen-32B (THE REASONING BEAST)
echo "Downloading DeepSeek-R1-Distill-Qwen-32B..."
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir /workspace/deepseek-r1-32b \
    --local-dir-use-symlinks False

# Qwen2.5-Coder-32B-Instruct (THE CODE MACHINE)
echo "Downloading Qwen2.5-Coder-32B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
    --local-dir /workspace/qwen-coder-32b \
    --local-dir-use-symlinks False

echo "Base models downloaded!"

# =============================================================================
# STEP 4: QUANTIZE MODELS (AWQ for vLLM compatibility)
# =============================================================================
echo ""
echo "=== STEP 4: QUANTIZING MODELS (AWQ) ==="
echo ""

pip install --quiet autoawq auto-gptq

# Create quantization script
cat > /workspace/quantize_awq.py << 'PYEOF'
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import sys
import os

def quantize_model(model_path, output_path, model_name):
    print(f"\n{'='*60}")
    print(f"Quantizing {model_name}")
    print(f"Input: {model_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # AWQ config optimized for math reasoning
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM"  # Best for vLLM
    }

    # Load model
    print("Loading model...")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Calibration data - math problems for better quantization
    calib_data = [
        "Let n be a positive integer. Find the remainder when 2^100 is divided by 127.",
        "Find all integers x such that x^2 + 3x + 2 = 0.",
        "Calculate the sum of the first 100 prime numbers.",
        "If f(x) = x^3 - 6x^2 + 11x - 6, find all roots of f(x) = 0.",
        "How many ways can you arrange the letters in MISSISSIPPI?",
        "Find the area of a triangle with vertices at (0,0), (4,0), and (2,3).",
        "Solve the system: x + y = 10, xy = 21",
        "What is the value of sum_{k=1}^{100} k^2?",
        "Find the number of positive divisors of 2024.",
        "Let a_n = 2a_{n-1} + 1 with a_1 = 1. Find a_10.",
    ]

    # Quantize
    print("Quantizing (this takes 30-60 min)...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    # Save
    print(f"Saving to {output_path}...")
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Done! Quantized model saved to {output_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Quantize DeepSeek (REASONING)
    if os.path.exists("/workspace/deepseek-r1-32b"):
        quantize_model(
            "/workspace/deepseek-r1-32b",
            "/workspace/deepseek-r1-32b-awq",
            "DeepSeek-R1-Distill-Qwen-32B"
        )

    # Quantize Qwen Coder (EXECUTION)
    if os.path.exists("/workspace/qwen-coder-32b"):
        quantize_model(
            "/workspace/qwen-coder-32b",
            "/workspace/qwen-coder-32b-awq",
            "Qwen2.5-Coder-32B-Instruct"
        )

    print("\n" + "="*60)
    print("ALL MODELS QUANTIZED!")
    print("="*60)
PYEOF

# Run quantization
python /workspace/quantize_awq.py

# =============================================================================
# STEP 5: UPLOAD QUANTIZED MODELS TO KAGGLE
# =============================================================================
echo ""
echo "=== STEP 5: UPLOADING MODELS TO KAGGLE ==="
echo ""

# Upload DeepSeek AWQ
if [ -d "/workspace/deepseek-r1-32b-awq" ]; then
    cd /workspace/deepseek-r1-32b-awq
    cat > dataset-metadata.json << EOF
{
  "title": "deepseek-r1-32b-awq",
  "id": "$KAGGLE_USER/deepseek-r1-32b-awq",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
    echo "Uploading DeepSeek-R1-32B-AWQ..."
    kaggle datasets create -p /workspace/deepseek-r1-32b-awq --dir-mode zip
fi

# Upload Qwen Coder AWQ
if [ -d "/workspace/qwen-coder-32b-awq" ]; then
    cd /workspace/qwen-coder-32b-awq
    cat > dataset-metadata.json << EOF
{
  "title": "qwen-coder-32b-awq",
  "id": "$KAGGLE_USER/qwen-coder-32b-awq",
  "licenses": [{"name": "apache-2.0"}]
}
EOF
    echo "Uploading Qwen-Coder-32B-AWQ..."
    kaggle datasets create -p /workspace/qwen-coder-32b-awq --dir-mode zip
fi

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Uploaded to Kaggle:"
echo "  1. ryanaimo-vllm-wheels (vLLM + all deps)"
echo "  2. deepseek-r1-32b-awq (reasoning model)"
echo "  3. qwen-coder-32b-awq (code execution model)"
echo ""
echo "In your notebook, attach:"
echo "  - ryanaimo-vllm-wheels"
echo "  - deepseek-r1-32b-awq"
echo "  - qwen-coder-32b-awq"
echo ""
