# Elle Training: The Idiot's Guide

> **TL;DR**: Build docker, rent GPU, run one command, get smart model.

---

## Step 0: Get Your Keys (Do This Once)

You need 4 keys. Get them all before you start.

### HuggingFace Token
1. Go to https://huggingface.co
2. Sign up / Log in
3. Click your profile pic → Settings → Access Tokens
4. Click "New token" → Name it "elle" → Select "Write" → Create
5. Copy the token (starts with `hf_`)
6. **Save it somewhere safe**

### Weights & Biases Key
1. Go to https://wandb.ai
2. Sign up / Log in
3. Go to https://wandb.ai/authorize
4. Copy the API key
5. **Save it somewhere safe**

### Kaggle Credentials
1. Go to https://kaggle.com
2. Sign up / Log in
3. Click your profile pic → Settings
4. Scroll to "API" section → Click "Create New Token"
5. It downloads a `kaggle.json` file
6. Open it - you need `username` and `key`
7. **Save both somewhere safe**

### RunPod Account
1. Go to https://runpod.io
2. Sign up
3. Add credits (start with $50, training costs ~$85)

---

## Step 1: Build the Docker Image

On your local machine with Docker installed:

```bash
# Clone the repo (if you haven't)
git clone https://github.com/aphoticshaman/nucleation-packages.git
cd nucleation-packages/packages/lfbm/docker

# Build the image (takes 20-30 min)
docker build -t aphoticshaman/lfbm-trainer:latest .

# Login to Docker Hub
docker login
# Enter your Docker Hub username and password

# Push the image (takes 10-20 min depending on internet)
docker push aphoticshaman/lfbm-trainer:latest
```

**Don't have Docker?** Use RunPod's cloud build or skip to Step 2 and use a pre-built image.

---

## Step 2: Launch a RunPod GPU

1. Go to https://runpod.io/console/pods
2. Click "Deploy" (big purple button)
3. **Select GPU**:
   - Best: `NVIDIA H200` (141 GB) - ~$4.50/hr
   - Good: `NVIDIA H100 80GB` - ~$3.50/hr
   - OK: `2x NVIDIA A100 80GB` - ~$6/hr
4. **Container Image**: `aphoticshaman/lfbm-trainer:latest`
5. **Volume**: 500 GB (you need space for model + checkpoints)
6. **Expose Ports**: 22 (SSH), 8888 (Jupyter)
7. Click "Deploy"

Wait 2-5 minutes for pod to start.

---

## Step 3: Connect to Your Pod

### Option A: Web Terminal
1. Click "Connect" on your pod
2. Click "Start Web Terminal"
3. You're in!

### Option B: SSH (Better)
1. Click "Connect" on your pod
2. Copy the SSH command
3. Paste in your terminal:
```bash
ssh root@xxx.xxx.xxx.xxx -p 12345 -i ~/.ssh/id_rsa
```

---

## Step 4: Set Up Your Keys

Once connected, run these commands (replace with YOUR keys):

```bash
# HuggingFace
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"

# Weights & Biases
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"

# Kaggle
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"

# Make them permanent (so they survive restarts)
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
echo 'export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
echo 'export KAGGLE_USERNAME="your_kaggle_username"' >> ~/.bashrc
echo 'export KAGGLE_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
```

---

## Step 5: Prepare Your Training Data

You need to create the `aphoticshaman/latticeforge-briefing-data` dataset on HuggingFace.

### If You Have Data Ready:

```python
# Create a file called upload_data.py
from datasets import Dataset
from huggingface_hub import HfApi

# Your data should look like this:
data = [
    {
        "messages": [
            {"role": "system", "content": "You are Elle, a senior intelligence analyst..."},
            {"role": "user", "content": "Analyze China's Belt and Road Initiative."},
            {"role": "assistant", "content": "The Belt and Road Initiative represents..."}
        ]
    },
    # ... more examples
]

dataset = Dataset.from_list(data)
dataset.push_to_hub("aphoticshaman/latticeforge-briefing-data")
```

### If You Don't Have Data Yet:

Edit the config to use only public datasets:

```bash
# Edit the config
nano /workspace/configs/config_ultimate_elle.yaml

# Remove or comment out the latticeforge-briefing-data line
# The training will still work with the other datasets
```

---

## Step 6: Run Training

This is the fun part. One command:

```bash
bash /workspace/scripts/train_elle_ultimate.sh
```

That's it. Go get coffee. Or sleep. It takes **12-18 hours**.

### Watch It Train

Open a second terminal and run:
```bash
# See GPU usage
watch -n 1 nvidia-smi

# See training logs
tail -f /workspace/logs/training_*.log
```

### Check Progress Online
Go to https://wandb.ai and find your project `lfbm-elle`. You'll see:
- Loss curves (should go down)
- Learning rate schedule
- GPU memory usage
- Estimated time remaining

---

## Step 7: What If Something Goes Wrong?

### "CUDA out of memory"
Your GPU ran out of memory. Fix:
```bash
# Edit the config
nano /workspace/configs/config_ultimate_elle.yaml

# Change these lines:
micro_batch_size: 1  # was 2
gradient_accumulation_steps: 16  # was 8

# Or reduce sequence length:
sequence_len: 4096  # was 8192
```

### "Loss is NaN"
Training exploded. Fix:
```bash
# Edit the config
nano /workspace/configs/config_ultimate_elle.yaml

# Lower learning rate:
learning_rate: 1.0e-5  # was 2.0e-5
```

### Training is Too Slow
Check if GPU is actually being used:
```bash
nvidia-smi
```
- GPU Usage should be 90%+
- If it's low, you have a data loading bottleneck

### Pod Crashed / Got Interrupted
Resume from checkpoint:
```bash
RESUME_FROM=/workspace/outputs/elle-ultimate/checkpoint-1000 \
    bash /workspace/scripts/train_elle_ultimate.sh
```

---

## Step 8: Training Done! Now Upload

### Upload to HuggingFace
```bash
bash /workspace/scripts/upload_hf.sh
```

This creates: `https://huggingface.co/aphoticshaman/elle-72b-ultimate`

### Upload to Kaggle (for competitions)
```bash
bash /workspace/scripts/upload_kaggle.sh
```

This creates a Kaggle dataset you can use in notebooks.

---

## Step 9: Test Your Model

Quick test before shutting down:

```python
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the model
model_path = "/workspace/outputs/elle-ultimate/merged"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Test it
messages = [
    {"role": "system", "content": "You are Elle, a senior intelligence analyst."},
    {"role": "user", "content": "What happens to emerging markets when the Fed raises rates?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

---

## Step 10: Shut Down (IMPORTANT - Saves Money!)

1. Go to https://runpod.io/console/pods
2. Click "Stop" on your pod (this stops billing)
3. If you're 100% done, click "Delete" to remove it completely

**Keep the volume** if you might want to resume later.
**Delete the volume** if you're done (saves storage costs).

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Start training | `bash /workspace/scripts/train_elle_ultimate.sh` |
| Watch GPU | `nvidia-smi -l 1` |
| Watch logs | `tail -f /workspace/logs/training_*.log` |
| Upload to HF | `bash /workspace/scripts/upload_hf.sh` |
| Upload to Kaggle | `bash /workspace/scripts/upload_kaggle.sh` |
| Resume training | `RESUME_FROM=/path/to/checkpoint bash /workspace/scripts/train_elle_ultimate.sh` |
| Kill training | `pkill -f axolotl` |

---

## Cost Breakdown

| What | Cost |
|------|------|
| Setup & testing | ~$9 (2 hrs × $4.50) |
| Training | ~$67 (15 hrs × $4.50) |
| Upload & cleanup | ~$9 (2 hrs × $4.50) |
| **Total** | **~$85** |

**Pro tip**: Use spot instances for 30-50% off (but they can be interrupted).

---

## FAQ

**Q: How do I know if it's working?**
A: Check wandb.ai - loss should decrease over time. If loss is flat or going up, something's wrong.

**Q: Can I use a cheaper GPU?**
A: Yes, but training takes longer. A100 40GB works but needs batch size 1.

**Q: How long does it really take?**
A: On H200: ~12-18 hours. On A100 80GB: ~20-30 hours. On A100 40GB: ~35-50 hours.

**Q: What if I need to stop and resume?**
A: The script saves checkpoints every 500 steps. Use `RESUME_FROM=...` to continue.

**Q: The model sucks. What now?**
A: More/better training data, more epochs, or higher LoRA rank. Fine-tuning is iterative.

---

## You're Done!

You now have Elle, a 72B parameter model that knows:
- Geopolitics (conflicts, alliances, strategic competition)
- Macroeconomics (Fed policy, inflation, currency dynamics)
- Wall Street (trading, derivatives, market contagion)
- The intersection of all three

Use it wisely.
