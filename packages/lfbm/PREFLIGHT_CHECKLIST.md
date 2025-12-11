# Elle Training Preflight Checklist & Contingency Playbook

> **Mission:** Fine-tune Qwen-72B → Elle (Geopolitical + Economics Intelligence Expert)
>
> **Target:** Post-PhD senior analyst with junior analyst motivation and novel insight

---

## Phase 0: Pre-Launch Preparation

### 0.1 Account & Credentials Setup

| Task | Status | Verification | Contingency |
|------|--------|--------------|-------------|
| HuggingFace account created | ☐ | `huggingface-cli whoami` | Create at huggingface.co |
| HF write token generated | ☐ | Token starts with `hf_` | Settings → Access Tokens → Write |
| Kaggle account verified | ☐ | `kaggle competitions list` | kaggle.com/account |
| Kaggle API key downloaded | ☐ | `~/.kaggle/kaggle.json` exists | Account → API → Create New Token |
| W&B account created | ☐ | `wandb login --verify` | wandb.ai/authorize |
| RunPod account funded | ☐ | Dashboard shows balance | Add credits via stripe |
| Docker Hub account | ☐ | `docker login` succeeds | hub.docker.com signup |

**Environment Variables Required:**
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
export KAGGLE_USERNAME="aphoticshaman"
export KAGGLE_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 0.2 Training Data Preparation

| Task | Status | Verification | Contingency |
|------|--------|--------------|-------------|
| `aphoticshaman/latticeforge-briefing-data` exists | ☐ | Check HF Hub | Create & upload dataset |
| Dataset has `train` split | ☐ | Load with `datasets` | Add split in dataset card |
| Dataset format is ChatML | ☐ | Sample has `messages` key | Convert with script |
| Economics datasets accessible | ☐ | See list below | Use alternative datasets |

**Economics Dataset Verification:**
```python
from datasets import load_dataset

datasets_to_check = [
    "sujet-ai/Sujet-Finance-Instruct-177k",
    "virattt/financial-qa-10K",
    "TheFinAI/flare-finqa",
    "FinGPT/fingpt-sentiment-train",
    "winddude/econ-qa",
    "amphora/forex-news"
]

for ds_name in datasets_to_check:
    try:
        ds = load_dataset(ds_name, split="train[:10]")
        print(f"✓ {ds_name}: {len(ds)} samples loaded")
    except Exception as e:
        print(f"✗ {ds_name}: {e}")
```

**Contingency - Dataset Not Available:**
| Original Dataset | Alternative |
|------------------|-------------|
| sujet-ai/Sujet-Finance | gbharti/finance-alpaca |
| virattt/financial-qa-10K | amphora/FiQA-SA |
| TheFinAI/flare-finqa | AdaptLLM/finance-tasks |
| FinGPT/fingpt-sentiment | zeroshot/twitter-financial-news-sentiment |

---

## Phase 1: Docker Image Build

### 1.1 Local Build (Optional)

```bash
cd packages/lfbm/docker

# Build image
docker build -t aphoticshaman/lfbm-trainer:latest .

# Test locally
docker run --gpus all -it --rm aphoticshaman/lfbm-trainer:latest \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

| Checkpoint | Expected | Contingency |
|------------|----------|-------------|
| Docker build completes | < 30 min | Check network, retry with `--no-cache` |
| Image size | ~25-35 GB | Normal for ML images |
| CUDA test passes | `CUDA: True` | Install nvidia-container-toolkit |
| Flash Attention imports | No error | Build from source with `MAX_JOBS=8` |

### 1.2 Push to Docker Hub

```bash
docker login
docker push aphoticshaman/lfbm-trainer:latest
```

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Push timeout | Connection reset | Retry, check bandwidth |
| Auth failure | 401 error | Re-login with `docker login` |
| Size limit | Layer too large | Split RUN commands, use multi-stage |

---

## Phase 2: RunPod Pod Launch

### 2.1 Pod Selection

**Recommended Configurations:**

| Priority | GPU | VRAM | vCPU | RAM | Cost/hr |
|----------|-----|------|------|-----|---------|
| 1st | 1x H200 | 141 GB | 16 | 128 GB | ~$4.50 |
| 2nd | 1x H100 80GB | 80 GB | 16 | 128 GB | ~$3.50 |
| 3rd | 2x A100 80GB | 160 GB | 32 | 256 GB | ~$6.00 |
| 4th | 2x A100 40GB | 80 GB | 32 | 256 GB | ~$4.00 |

### 2.2 Launch Checklist

| Step | Action | Verification |
|------|--------|--------------|
| 1 | Select GPU type from table above | Dashboard shows selected GPU |
| 2 | Set container image: `aphoticshaman/lfbm-trainer:latest` | Image field populated |
| 3 | Set volume size: 500 GB | Storage section shows 500G |
| 4 | Add environment variables | All 4 secrets configured |
| 5 | Set Docker args: `--shm-size=64g` | Advanced options configured |
| 6 | Launch pod | Status changes to "Running" |
| 7 | SSH into pod | Terminal access works |

### 2.3 Pod Launch Issues & Resolutions

| Issue | Symptom | Resolution |
|-------|---------|------------|
| No GPUs available | "Insufficient capacity" | Try different region, wait for availability |
| Pod stuck starting | Status = "Pending" > 5 min | Cancel, try different GPU type |
| Image pull fails | "ImagePullBackOff" | Check image name, push to Docker Hub |
| OOM on start | Pod crashes immediately | Reduce shm-size, check image |
| SSH fails | Connection refused | Wait 2 min, check pod logs |
| Volume not mounted | `/workspace` empty | Check volume config, remount |

---

## Phase 3: Pre-Training Validation

### 3.1 GPU & CUDA Verification

```bash
# Run these commands after SSH into pod

# 1. GPU availability
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 545.xx.xx    Driver Version: 545.xx.xx    CUDA Version: 12.x               |
# |    GPU       Memory     Usage                                                          |
# |    H200      141 GB     ~0%                                                            |
# +-----------------------------------------------------------------------------------------+

# 2. PyTorch CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 3. Flash Attention
python -c "
import flash_attn
print(f'Flash Attention: {flash_attn.__version__}')
"

# 4. bitsandbytes
python -c "
import bitsandbytes
print(f'bitsandbytes: {bitsandbytes.__version__}')
"
```

| Check | Expected | If Fails |
|-------|----------|----------|
| nvidia-smi | Shows GPU | Restart pod, check driver |
| torch.cuda.is_available() | True | Reinstall PyTorch with CUDA |
| Flash Attention | 2.7.x | Build from source |
| bitsandbytes | 0.45.x | pip install bitsandbytes |

### 3.2 Authentication Verification

```bash
# HuggingFace
huggingface-cli whoami
# Expected: Your username

# Weights & Biases
wandb login --verify
# Expected: "wandb: Verified"

# Kaggle
kaggle competitions list
# Expected: List of competitions (no error)
```

### 3.3 Model Download Test

```bash
python -c "
from huggingface_hub import snapshot_download
import os

# Test download (small portion)
path = snapshot_download(
    'Qwen/Qwen2.5-72B-Instruct',
    allow_patterns=['config.json', 'tokenizer.json'],
    local_dir='/workspace/models/test'
)
print(f'Downloaded to: {path}')
print(f'Files: {os.listdir(path)}')
"
```

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Gated model error | 403 Forbidden | Accept model license on HF website |
| Network timeout | Connection reset | Retry, check RunPod network |
| Disk full | OSError | Increase volume, clean cache |

---

## Phase 4: Training Execution

### 4.1 Pre-Training Memory Estimation

```python
# Run this to estimate memory before training
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-72B-Instruct", trust_remote_code=True)

# Rough memory estimation
params_b = 72  # billion
bytes_per_param_4bit = 0.5  # 4-bit = 0.5 bytes
bytes_per_param_fp16 = 2    # for LoRA matrices

base_memory_gb = params_b * bytes_per_param_4bit
lora_memory_gb = (128 * 2 * 7 * 2) / 1e9 * bytes_per_param_fp16  # r=128, 7 modules, A+B
optimizer_memory_gb = lora_memory_gb * 8  # Adam states
activation_memory_gb = 15  # Rough estimate for batch=2, seq=8192

total_gb = base_memory_gb + lora_memory_gb + optimizer_memory_gb + activation_memory_gb

print(f"Estimated Memory: {total_gb:.1f} GB")
print(f"H200 (141 GB): {'OK' if total_gb < 130 else 'RISK'}")
print(f"A100 80GB: {'OK' if total_gb < 70 else 'RISK'}")
```

### 4.2 Launch Training

```bash
# Start training with logging
bash /workspace/scripts/train_elle_ultimate.sh 2>&1 | tee /workspace/logs/training_$(date +%Y%m%d_%H%M%S).log

# Or run in background with tmux
tmux new -s training
bash /workspace/scripts/train_elle_ultimate.sh
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```

### 4.3 Training Monitoring

**Terminal 1: GPU Monitoring**
```bash
watch -n 1 nvidia-smi
```

**Terminal 2: Training Logs**
```bash
tail -f /workspace/logs/training_*.log
```

**Terminal 3: W&B Dashboard**
- Navigate to: https://wandb.ai/YOUR_USER/lfbm-elle

### 4.4 Training Issues & Resolutions

| Issue | Symptom | Immediate Action | Root Cause Fix |
|-------|---------|------------------|----------------|
| OOM | CUDA out of memory | Restart, reduce batch size | Set micro_batch_size: 1 |
| NaN loss | Loss = nan | Stop training | Lower learning_rate to 1e-5 |
| Gradient explosion | Loss spikes wildly | Reduce max_grad_norm | Set max_grad_norm: 0.5 |
| Training stalls | No progress for 10+ min | Check GPU util | Reduce workers, check data |
| W&B not logging | No runs appearing | Check WANDB_API_KEY | Re-login with wandb login |
| Checkpoint save fails | OSError: No space | Delete old checkpoints | Reduce save_total_limit |
| Dataset loading fails | KeyError/IndexError | Check dataset format | Verify ChatML structure |

**OOM Recovery Procedure:**
```yaml
# Reduce memory usage progressively:

# Step 1: Reduce batch size
micro_batch_size: 1  # Was 2
gradient_accumulation_steps: 16  # Maintain effective batch

# Step 2: Reduce sequence length
sequence_len: 4096  # Was 8192

# Step 3: Disable sample packing
sample_packing: false

# Step 4: Reduce LoRA rank
lora_r: 64  # Was 128
lora_alpha: 128  # Was 256

# Step 5: Use DeepSpeed ZeRO-2
deepspeed: configs/deepspeed_zero2.json
```

---

## Phase 5: Post-Training Validation

### 5.1 Model Merge Verification

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

merged_path = "/workspace/outputs/elle-ultimate/merged"

# Check files exist
expected_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
for f in expected_files:
    path = os.path.join(merged_path, f)
    assert os.path.exists(path), f"Missing: {f}"
    print(f"✓ {f}")

# Check model loads
tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)
print(f"✓ Tokenizer loaded: {tokenizer.vocab_size} tokens")

# Quick inference test
print("Testing inference...")
```

### 5.2 Quality Validation

```python
# Test prompts covering all domains
test_prompts = [
    # Geopolitical
    "What are the strategic implications of Turkey's position on NATO expansion?",

    # Macroeconomics
    "Explain how quantitative tightening affects emerging market bond yields.",

    # Wall Street
    "Walk me through the mechanics of a credit default swap.",

    # Intersection
    "How might OPEC production cuts affect the Fed's rate decision calculus?"
]

for prompt in test_prompts:
    response = model.generate(prompt, max_tokens=200)
    print(f"\nQ: {prompt[:50]}...")
    print(f"A: {response[:200]}...")

    # Manual quality check
    input("Press Enter if response is acceptable, Ctrl+C to flag...")
```

### 5.3 Validation Failure Contingencies

| Issue | Symptom | Resolution |
|-------|---------|------------|
| Nonsense output | Gibberish text | Check merge, may need re-training |
| Repetitive output | Same phrase repeated | Lower temperature, check for training collapse |
| Wrong domain | Geopolitics Q gets math A | Review dataset mixing, retrain |
| Refuses to answer | "I cannot help with..." | Check system prompt, review safety data |
| Short answers | One word responses | Increase min_new_tokens, check training |

---

## Phase 6: Deployment

### 6.1 Upload Checklist

| Destination | Command | Verification |
|-------------|---------|--------------|
| HuggingFace | `bash /workspace/scripts/upload_hf.sh` | Check model page |
| Kaggle (Model) | `bash /workspace/scripts/upload_kaggle.sh` | Check dataset page |
| Kaggle (Wheels) | `UPLOAD_TYPE=wheels bash /workspace/scripts/upload_kaggle.sh` | Check dataset page |

### 6.2 Upload Issues

| Issue | Resolution |
|-------|------------|
| Upload timeout | Retry with smaller chunks, use `--chunk-size` |
| Rate limit | Wait 15 min, retry |
| Invalid token | Re-authenticate |
| Repo already exists | Use --overwrite or create new version |

---

## Phase 7: Cleanup & Cost Control

### 7.1 Pod Termination Checklist

| Step | Action | Verification |
|------|--------|--------------|
| 1 | Download training logs | `scp -r pod:/workspace/logs ./` |
| 2 | Verify uploads complete | Check HF/Kaggle pages |
| 3 | Stop pod | RunPod dashboard shows "Stopped" |
| 4 | Delete pod (if done) | Pod removed from list |
| 5 | Keep volume (optional) | Volume persists for next session |

### 7.2 Cost Estimation

| Phase | GPU Hours | Cost/hr | Total |
|-------|-----------|---------|-------|
| Setup & testing | 2 | $4.50 | $9.00 |
| Training (3 epochs) | 15 | $4.50 | $67.50 |
| Post-training & upload | 2 | $4.50 | $9.00 |
| **Total** | **19** | - | **~$85** |

**Cost Optimization Tips:**
- Use spot instances (30-50% cheaper, risk of interruption)
- Train during off-peak hours (often cheaper)
- Stop pod immediately after upload completes
- Delete volume if not needed (storage costs add up)

---

## Quick Reference: Critical Commands

```bash
# Start training
bash /workspace/scripts/train_elle_ultimate.sh

# Monitor GPU
nvidia-smi -l 1

# Check training progress
tail -f /workspace/logs/training_*.log

# Resume from checkpoint
RESUME_FROM=/workspace/outputs/elle-ultimate/checkpoint-1000 \
    bash /workspace/scripts/train_elle_ultimate.sh

# Emergency stop
pkill -f axolotl

# Upload to HF
bash /workspace/scripts/upload_hf.sh

# Upload to Kaggle
bash /workspace/scripts/upload_kaggle.sh
```

---

## Emergency Contacts & Resources

- **RunPod Support:** support@runpod.io
- **HuggingFace Discord:** https://discord.gg/huggingface
- **Axolotl Issues:** https://github.com/OpenAccess-AI-Collective/axolotl/issues
- **Qwen Issues:** https://github.com/QwenLM/Qwen2.5/issues

---

*Last Updated: 2025-12-11*
*Version: 1.0.0*
