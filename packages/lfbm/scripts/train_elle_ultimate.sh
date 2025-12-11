#!/bin/bash
# =============================================================================
# ULTIMATE ELLE TRAINING SCRIPT
# =============================================================================
# One-command training for Elle: Geopolitical + Economics Intelligence Expert
# GPU: 1x H200 (141GB) or 2x A100 (80GB)
# Time: ~12-18 hours for 3 epochs
# =============================================================================

set -e

# Configuration
CONFIG_FILE="${CONFIG_FILE:-/workspace/configs/config_ultimate_elle.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/elle-ultimate}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
RESUME_FROM="${RESUME_FROM:-}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="elle-ultimate-${TIMESTAMP}"

echo "=============================================="
echo "  ELLE ULTIMATE TRAINING"
echo "=============================================="
echo "  Run: $RUN_NAME"
echo "  Config: $CONFIG_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Started: $(date)"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Pre-flight Checks
# -----------------------------------------------------------------------------
echo "[PREFLIGHT 1/8] Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 70000 ]; then
    echo "WARNING: GPU has less than 70GB VRAM. OOM likely."
    echo "Consider using 2x A100 80GB with DeepSpeed ZeRO-2"
fi
echo ""

echo "[PREFLIGHT 2/8] Checking disk space..."
DISK_FREE=$(df -BG /workspace | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$DISK_FREE" -lt 200 ]; then
    echo "WARNING: Less than 200GB free. May run out during training."
fi
echo "  Free space: ${DISK_FREE}GB"
echo ""

echo "[PREFLIGHT 3/8] Validating config file..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "  Config: OK"
echo ""

echo "[PREFLIGHT 4/8] Checking Python environment..."
python3 -c "
import torch
import transformers
import peft
import axolotl
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  Axolotl: {axolotl.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
"
echo ""

echo "[PREFLIGHT 5/8] Checking Flash Attention..."
python3 -c "
try:
    import flash_attn
    print(f'  Flash Attention: {flash_attn.__version__} - OK')
except:
    print('  Flash Attention: Not available (using standard attention)')
"
echo ""

echo "[PREFLIGHT 6/8] Checking HuggingFace authentication..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli whoami 2>/dev/null && echo "  HF: Authenticated" || {
        huggingface-cli login --token "$HF_TOKEN" 2>/dev/null
        echo "  HF: Logged in"
    }
else
    echo "  WARNING: HF_TOKEN not set. May fail to download gated models."
fi
echo ""

echo "[PREFLIGHT 7/8] Checking W&B..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "  W&B: Configured (Project: ${WANDB_PROJECT:-lfbm-elle})"
else
    echo "  WARNING: WANDB_API_KEY not set. Disabling logging."
    export WANDB_MODE=disabled
fi
echo ""

echo "[PREFLIGHT 8/8] Creating directories..."
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
echo "  Directories: OK"
echo ""

# -----------------------------------------------------------------------------
# Start Training
# -----------------------------------------------------------------------------
echo "=============================================="
echo "  STARTING TRAINING"
echo "=============================================="
echo ""

# Set up logging
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

# Export run name for W&B
export WANDB_RUN_NAME="$RUN_NAME"

# Handle resume
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    echo "Resuming from checkpoint: $RESUME_FROM"
    RESUME_ARG="--resume_from_checkpoint=$RESUME_FROM"
fi

# Run Axolotl training
echo "Running: accelerate launch -m axolotl.cli.train $CONFIG_FILE $RESUME_ARG"
echo ""

# Single GPU training
accelerate launch --num_processes=1 --mixed_precision=bf16 \
    -m axolotl.cli.train "$CONFIG_FILE" \
    --output_dir="$OUTPUT_DIR" \
    $RESUME_ARG

TRAIN_EXIT_CODE=$?

# -----------------------------------------------------------------------------
# Post-Training
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  TRAINING COMPLETE"
echo "=============================================="
echo "  Exit Code: $TRAIN_EXIT_CODE"
echo "  Finished: $(date)"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""

    # Merge LoRA weights
    echo "[POST 1/3] Merging LoRA weights..."
    python3 << EOF
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
import os

output_dir = "${OUTPUT_DIR}"
merged_dir = f"{output_dir}/merged"

print(f"  Loading adapter from: {output_dir}")
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

print("  Merging weights...")
merged_model = model.merge_and_unload()

print(f"  Saving to: {merged_dir}")
merged_model.save_pretrained(merged_dir, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
tokenizer.save_pretrained(merged_dir)

print("  Merge complete!")
EOF
    echo ""

    # Test inference
    echo "[POST 2/3] Testing inference..."
    python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

merged_dir = "${OUTPUT_DIR}/merged"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("  Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    merged_dir,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

test_prompt = """Analyze the second-order effects of Federal Reserve interest rate decisions on emerging market currencies and their implications for global supply chain stability."""

messages = [
    {"role": "system", "content": "You are Elle, a senior geopolitical and economic intelligence analyst."},
    {"role": "user", "content": test_prompt}
]

print("  Running test inference...")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n  Test Response (truncated):")
print("  " + response[-500:])
print("\n  Inference test: OK")
EOF
    echo ""

    # Summary
    echo "[POST 3/3] Training Summary"
    echo "=============================================="
    echo "  Output Directory: $OUTPUT_DIR"
    echo "  Merged Model: $OUTPUT_DIR/merged"
    echo "  Training Log: $LOG_FILE"
    echo ""
    echo "  Next Steps:"
    echo "    1. Upload to HuggingFace: bash /workspace/scripts/upload_hf.sh"
    echo "    2. Upload to Kaggle: bash /workspace/scripts/upload_kaggle.sh"
    echo "    3. Build GGUF: python -m llama_cpp.convert ..."
    echo ""
else
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check log file: $LOG_FILE"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce micro_batch_size or enable gradient_checkpointing"
    echo "  - CUDA error: Check nvidia-smi, may need to restart pod"
    echo "  - Data error: Verify dataset paths and format"
    exit $TRAIN_EXIT_CODE
fi
