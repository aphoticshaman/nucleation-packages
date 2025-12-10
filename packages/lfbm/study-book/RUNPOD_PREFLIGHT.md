# RunPod Pre-Flight Checklist for Elle Training

## Quick Start (Copy-Paste Ready)

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# ELLE TRAINING PRE-FLIGHT - RunPod H100/H200 Setup
# ═══════════════════════════════════════════════════════════════════════════════
# Run this FIRST on a fresh RunPod instance
# Estimated time: 5-10 minutes depending on network speed

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo " ELLE TRAINING PRE-FLIGHT CHECKLIST"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ============================================================
# 1. ENVIRONMENT SETUP
# ============================================================
echo "[1/7] Setting up environment..."

# Create workspace
mkdir -p /workspace/elle-training
cd /workspace/elle-training

# Set HuggingFace token (REQUIRED - get from https://huggingface.co/settings/tokens)
# REPLACE WITH YOUR TOKEN:
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Verify token is set
if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_YOUR_TOKEN_HERE" ]; then
    echo "❌ ERROR: Set your HuggingFace token!"
    echo "   Get one at: https://huggingface.co/settings/tokens"
    echo "   Then run: export HF_TOKEN='hf_xxx'"
    exit 1
fi

# Login to HuggingFace
pip install --quiet huggingface_hub
huggingface-cli login --token "$HF_TOKEN"
echo "✓ HuggingFace authenticated"

# ============================================================
# 2. CUDA VERIFICATION
# ============================================================
echo "[2/7] Verifying CUDA..."

nvidia-smi
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"

# Check for H100/H200
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "✓ Detected GPU: $GPU_NAME"

# ============================================================
# 3. INSTALL AXOLOTL + DEPENDENCIES
# ============================================================
echo "[3/7] Installing Axolotl and dependencies..."

# Core ML stack
pip install --upgrade pip wheel setuptools

# PyTorch with CUDA 12.1 (RunPod default)
pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Axolotl for fine-tuning
pip install axolotl[flash-attn,deepspeed]

# bitsandbytes for NF4 quantization
pip install bitsandbytes>=0.43.0

# Additional deps
pip install transformers>=4.40.0 accelerate>=0.33.0 peft>=0.12.0
pip install datasets sentencepiece tiktoken einops
pip install wandb  # Optional: for training visualization

echo "✓ Dependencies installed"

# ============================================================
# 4. DOWNLOAD BASE MODEL
# ============================================================
echo "[4/7] Downloading base model (this may take a while)..."

# Use aphoticshaman/qwen-72b-math-nf4 as base (already quantized)
# OR download full Qwen2.5-Math-72B-Instruct for fresh quantization

python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_id = "aphoticshaman/qwen-72b-math-nf4"  # Pre-quantized
# model_id = "Qwen/Qwen2.5-Math-72B-Instruct"  # Full model (requires more VRAM)

print(f"Downloading {model_id}...")

# Download tokenizer first (fast)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained("/workspace/elle-training/base-model")
print("✓ Tokenizer downloaded")

# Download model (slow - 30GB+)
# For NF4, we don't load the full model here - Axolotl handles it
print("✓ Model will be loaded by Axolotl during training")
print("  Base model ready at: /workspace/elle-training/base-model")
EOF

echo "✓ Base model ready"

# ============================================================
# 5. CLONE TRAINING DATA + CONFIGS
# ============================================================
echo "[5/7] Setting up training data..."

# Create training data directory
mkdir -p /workspace/elle-training/data

# Download your GDELT/briefing training data from HuggingFace
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

# Download training data (if hosted on HF)
# Uncomment and modify for your dataset:
# snapshot_download(
#     repo_id="aphoticshaman/elle-training-data",
#     repo_type="dataset",
#     local_dir="/workspace/elle-training/data"
# )

# Or create a sample training file for testing
import json

sample_data = [
    {
        "instruction": "Analyze the current geopolitical situation for Ukraine.",
        "input": "GDELT tone: -4.2, transition_risk: 0.85, basin_strength: 0.35",
        "output": '{"political":"SUPERCOOLED. UKR 85% risk. Coalition stable but fatigued.","economic":"Energy disruption ongoing. GDP -30%.","security":"Active conflict. T_c exceeded.","summary":"Critical. UKR above T_c. Monitor daily.","nsm":"Increase recon cadence. Update contingencies."}'
    },
    {
        "instruction": "Generate an intel briefing for the NATO alliance.",
        "input": "Nations: 32, avg_basin_strength: 0.72, avg_transition_risk: 0.28",
        "output": '{"political":"CRYSTALLINE. Alliance cohesion 0.72. Eastern flank reinforced.","economic":"Defense spending up 15% avg.","security":"Article 5 credibility intact.","summary":"Stable. F=0.78. No imminent concerns.","nsm":"Continue deterrence posture. Review Baltic contingencies."}'
    }
]

with open('/workspace/elle-training/data/training_sample.jsonl', 'w') as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')

print("✓ Sample training data created")
print("  Add your full dataset to: /workspace/elle-training/data/")
EOF

echo "✓ Training data directory ready"

# ============================================================
# 6. CREATE AXOLOTL CONFIG
# ============================================================
echo "[6/7] Creating Axolotl config..."

cat > /workspace/elle-training/config.yaml << 'EOF'
# ═══════════════════════════════════════════════════════════════════════════════
# ELLE FINE-TUNING CONFIG - H100/H200 Optimized
# ═══════════════════════════════════════════════════════════════════════════════

base_model: aphoticshaman/qwen-72b-math-nf4
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: true

# Data
datasets:
  - path: /workspace/elle-training/data/training_sample.jsonl
    type: alpaca

output_dir: /workspace/elle-training/output

# LoRA config (memory efficient)
adapter: lora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Training
sequence_len: 4096
micro_batch_size: 1
gradient_accumulation_steps: 8
num_epochs: 3
learning_rate: 2e-5
lr_scheduler: cosine
warmup_ratio: 0.1

# Optimization
bf16: true
tf32: true
gradient_checkpointing: true
flash_attention: true

# Advanced: DoRA (if supported)
# use_dora: true

# Logging
logging_steps: 10
save_strategy: steps
save_steps: 100
eval_steps: 100

# DeepSpeed (for multi-GPU)
deepspeed: /workspace/elle-training/ds_config.json
EOF

# DeepSpeed config for ZeRO-3
cat > /workspace/elle-training/ds_config.json << 'EOF'
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 1
}
EOF

echo "✓ Axolotl config created"

# ============================================================
# 7. PRE-FLIGHT VERIFICATION
# ============================================================
echo "[7/7] Running pre-flight verification..."

python3 << 'EOF'
import torch
import os

print("\n" + "="*70)
print(" PRE-FLIGHT CHECK RESULTS")
print("="*70)

# Check GPU
print(f"\n✓ GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f}GB")

# Check VRAM
total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
print(f"\n✓ Total VRAM: {total_vram / 1024**3:.1f}GB")

# Estimate fit
if total_vram / 1024**3 >= 80:
    print("  → 72B NF4 model: FITS ✓")
elif total_vram / 1024**3 >= 40:
    print("  → 72B NF4 model: TIGHT (enable offloading)")
else:
    print("  → 72B NF4 model: TOO SMALL (use smaller model)")

# Check HF login
from huggingface_hub import HfApi
api = HfApi()
try:
    user = api.whoami()
    print(f"\n✓ HuggingFace: Logged in as {user['name']}")
except:
    print("\n⚠ HuggingFace: Not logged in")

# Check paths
paths = [
    "/workspace/elle-training/config.yaml",
    "/workspace/elle-training/ds_config.json",
    "/workspace/elle-training/data/training_sample.jsonl"
]
print("\n✓ Files:")
for p in paths:
    status = "✓" if os.path.exists(p) else "✗"
    print(f"  {status} {p}")

print("\n" + "="*70)
print(" READY TO TRAIN!")
print("="*70)
print("\nNext steps:")
print("  1. Add your full training data to /workspace/elle-training/data/")
print("  2. Run: accelerate launch -m axolotl.cli.train /workspace/elle-training/config.yaml")
print("  3. Or for multi-GPU: deepspeed --num_gpus=4 -m axolotl.cli.train config.yaml")
print("\n")
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo " PRE-FLIGHT COMPLETE"
echo "═══════════════════════════════════════════════════════════════════════════════"
```

---

## Manual Checklist

### Before You Start

- [ ] RunPod instance with H100 80GB (or 2x H100/H200)
- [ ] HuggingFace token with write access
- [ ] Training data prepared (JSONL format)
- [ ] ~$4-18/hr budget ready

### Environment Variables

```bash
# Required
export HF_TOKEN="hf_xxx"                    # Your HuggingFace token
export WANDB_API_KEY="xxx"                  # Optional: Weights & Biases

# Optional
export CUDA_VISIBLE_DEVICES="0,1,2,3"       # Which GPUs to use
export TRANSFORMERS_CACHE="/workspace/cache" # Model cache location
```

### GPU Memory Requirements

| Model Size | Quantization | Min VRAM | Recommended |
|------------|--------------|----------|-------------|
| 72B        | NF4          | 40GB     | 80GB        |
| 72B        | Full         | 150GB+   | 2x H100     |
| 32B        | NF4          | 20GB     | 40GB        |
| 7B         | Full         | 16GB     | 24GB        |

### Training Data Format

Alpaca format (JSON Lines):
```json
{"instruction": "Task description", "input": "Context data", "output": "Expected output"}
```

Elle-specific format:
```json
{
  "instruction": "Generate intel briefing for [REGION]",
  "input": "GDELT: tone=-2.3, risk=0.45. Nations: USA, CHN, RUS. Signals: 500",
  "output": "{\"political\":\"...\",\"economic\":\"...\",\"security\":\"...\",\"summary\":\"...\",\"nsm\":\"...\"}"
}
```

---

## Training Commands

### Single GPU (H100 80GB)
```bash
cd /workspace/elle-training
accelerate launch -m axolotl.cli.train config.yaml
```

### Multi-GPU (2-8x H100/H200)
```bash
cd /workspace/elle-training
deepspeed --num_gpus=4 -m axolotl.cli.train config.yaml
```

### Resume from Checkpoint
```bash
accelerate launch -m axolotl.cli.train config.yaml --resume_from_checkpoint /workspace/elle-training/output/checkpoint-500
```

---

## Pushing to HuggingFace

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Load base + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "aphoticshaman/qwen-72b-math-nf4",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "/workspace/elle-training/output")

# Merge weights (optional - makes standalone model)
model = model.merge_and_unload()

# Push to HuggingFace
model.push_to_hub("aphoticshaman/elle-v1-geoint")
tokenizer = AutoTokenizer.from_pretrained("aphoticshaman/qwen-72b-math-nf4")
tokenizer.push_to_hub("aphoticshaman/elle-v1-geoint")
```

---

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
micro_batch_size: 1
gradient_accumulation_steps: 16  # Increase to compensate

# Enable CPU offloading in DeepSpeed
# Already in ds_config.json
```

### Slow Training
```bash
# Ensure flash attention is enabled
flash_attention: true

# Use bf16
bf16: true
tf32: true
```

### Model Won't Load
```bash
# Check disk space
df -h /workspace

# Clear cache
rm -rf ~/.cache/huggingface/hub/*

# Re-login to HuggingFace
huggingface-cli login --token $HF_TOKEN
```

---

## Cost Estimate

| Instance Type | $/hr  | Training Time (3 epochs) | Total Cost |
|---------------|-------|--------------------------|------------|
| 1x H100 80GB  | $4    | ~8-12 hours              | ~$32-48    |
| 2x H100 80GB  | $8    | ~4-6 hours               | ~$32-48    |
| 4x H100 SXM   | $14   | ~2-3 hours               | ~$28-42    |
| 8x H200 SXM   | $28   | ~1-2 hours               | ~$28-56    |

*Based on 10K training examples, 4096 sequence length*

---

## Guardian Integration Note

After training Elle, update Guardian's validation rules:
1. Test Elle output against current Guardian rules
2. Track rejection rate (should be <1%)
3. Adjust Guardian thresholds if Elle consistently produces valid format
4. Consider A/B testing Guardian rule variations

Guardian "learns" by:
- Tracking false positive rate (rejecting valid output)
- Tracking false negative rate (accepting garbage)
- Adjusting thresholds based on Elle's trained behavior
