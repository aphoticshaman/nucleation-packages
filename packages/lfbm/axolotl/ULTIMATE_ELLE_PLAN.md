# Ultimate Elle: 3-Hour H200 Fine-Tuning Plan

**Budget**: $22.50 (3 hrs × $7.50/hr on 2x H200)
**Goal**: Train Elle to output perfectly formatted JSON briefings, every time.

---

## The Plan

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HOUR 1: Data & First Model                                                  │
│  ├── 00:00-00:15  Upload 10K training examples to HuggingFace               │
│  ├── 00:15-00:20  Launch Qwen2.5-7B full fine-tune (5 epochs)               │
│  └── 00:20-00:55  Training runs (~35 min on 2x H200)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  HOUR 2: Scale Up to 14B                                                     │
│  ├── 01:00-01:05  Quick validation of 7B model JSON output                  │
│  ├── 01:05-01:10  Launch Qwen2.5-14B full fine-tune (3 epochs)              │
│  └── 01:10-01:55  Training runs (~45 min on 2x H200)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  HOUR 3: Polish & Deploy                                                     │
│  ├── 02:00-02:15  Evaluate both models, pick winner                         │
│  ├── 02:15-02:30  Optional: Fine-tune winner with harder examples           │
│  ├── 02:30-02:45  Deploy to vLLM serverless                                 │
│  └── 02:45-03:00  Test in production, update Vercel                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Flight Checklist (Do Before Starting H200)

### 1. Generate 10K Training Examples
```bash
cd packages/lfbm/axolotl
python prepare_data.py --count 10000 training_data_10k.jsonl
```

### 2. Upload to HuggingFace
```bash
export HF_TOKEN=your-token
python upload_to_hf.py training_data_10k.jsonl aphoticshaman/latticeforge-briefing-data
```

### 3. Verify Dataset is Public/Accessible
```bash
curl https://huggingface.co/datasets/aphoticshaman/latticeforge-briefing-data
```

---

## Hour 1: Qwen2.5-7B Full Fine-Tune

### Launch Training
```bash
# SSH into your RunPod instance or use web terminal

# Clone and setup
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e '.[flash-attn,deepspeed]'

# Download config
wget https://raw.githubusercontent.com/aphoticshaman/nucleation-packages/main/packages/lfbm/axolotl/config.yaml

# Set HuggingFace token
export HF_TOKEN=your-token

# Launch training (2x H200 with DeepSpeed)
accelerate launch --multi_gpu --num_processes 2 \
  -m axolotl.cli.train config.yaml
```

### Expected Output
- Training loss: starts ~2.5, drops to ~0.3 by epoch 5
- Time: ~35 minutes
- Output: `aphoticshaman/latticeforge-briefing-7b` on HuggingFace

---

## Hour 2: Qwen2.5-14B (The Real Elle)

### Quick Test of 7B Model
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("aphoticshaman/latticeforge-briefing-7b")
tokenizer = AutoTokenizer.from_pretrained("aphoticshaman/latticeforge-briefing-7b")

prompt = """NATIONS:
  USA: risk=15% →
  UKR: risk=85% ↑

Generate JSON briefings."""

# Test output is valid JSON
output = model.generate(...)
import json
json.loads(output)  # Should not throw
```

### Launch 14B Training
```bash
# Update config for 14B
sed -i 's/Qwen2.5-7B-Instruct/Qwen2.5-14B-Instruct/g' config.yaml
sed -i 's/micro_batch_size: 8/micro_batch_size: 4/g' config.yaml
sed -i 's/num_epochs: 5/num_epochs: 3/g' config.yaml

# Launch
accelerate launch --multi_gpu --num_processes 2 \
  -m axolotl.cli.train config.yaml
```

### Why 14B?
- 7B: Fast, good at simple tasks
- 14B: Better reasoning, more reliable JSON structure
- 32B+: Overkill for this task, slower inference

---

## Hour 3: Deploy the Winner

### Evaluate Both Models
```bash
# Test harness - run 100 prompts, count JSON parse failures
python evaluate_json_compliance.py aphoticshaman/latticeforge-briefing-7b
python evaluate_json_compliance.py aphoticshaman/latticeforge-briefing-14b
```

### Deploy to vLLM on RunPod
1. Go to RunPod → Serverless → vLLM
2. Configure:
   - Model: `aphoticshaman/latticeforge-briefing-14b` (or 7b)
   - GPU: H100 or A100 (for inference, H200 is overkill)
   - Max Workers: 3
   - Idle Timeout: 60s
3. Note endpoint ID

### Update Vercel
```bash
# In Vercel dashboard or CLI
vercel env add LFBM_ENDPOINT https://api.runpod.ai/v2/YOUR_NEW_ENDPOINT
vercel env add LFBM_MODEL aphoticshaman/latticeforge-briefing-14b
```

### Final Test
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "system", "content": "You are Elle..."},
        {"role": "user", "content": "NATIONS: USA 15%, UKR 85%..."}
      ]
    }
  }'
```

---

## Fallback: If JSON Still Broken

### Option A: Add More Diverse Training Data
```python
# Add edge cases: empty nations, extreme values, malformed inputs
# Force model to learn robust JSON output
```

### Option B: Use Constrained Decoding
vLLM supports JSON schema constraints:
```python
{
  "guided_json": {
    "type": "object",
    "properties": {
      "political": {"type": "string"},
      "economic": {"type": "string"},
      "security": {"type": "string"},
      "summary": {"type": "string"}
    },
    "required": ["political", "economic", "security", "summary"]
  }
}
```

### Option C: Guardian Catches the Rest
We already implemented Guardian - it'll fix minor issues.

---

## Expected Results

| Model | JSON Compliance | Inference Cost | Quality |
|-------|----------------|----------------|---------|
| Stock Qwen 3B | ~60% | $0.0005 | Low |
| Stock Qwen 7B | ~75% | $0.0008 | Medium |
| **Fine-tuned 7B** | ~95% | $0.0008 | High |
| **Fine-tuned 14B** | ~99% | $0.0015 | Excellent |

---

## Files Created

- `config.yaml` - Main config (7B full fine-tune, H200 optimized)
- `config_qlora.yaml` - Fallback for smaller GPUs
- `config_14b.yaml` - For the 14B model
- `deepspeed_configs/zero2.json` - Multi-GPU config
- `ULTIMATE_ELLE_PLAN.md` - This file

---

## Quick Start (TL;DR)

```bash
# 1. Generate data
python prepare_data.py --count 10000 training_data.jsonl

# 2. Upload to HuggingFace
python upload_to_hf.py training_data.jsonl aphoticshaman/latticeforge-briefing-data

# 3. On RunPod H200:
pip install axolotl[flash-attn,deepspeed]
accelerate launch --multi_gpu --num_processes 2 -m axolotl.cli.train config.yaml

# 4. Deploy to vLLM, update Vercel, profit
```

**Total Time**: 3 hours
**Total Cost**: ~$22.50
**Result**: Elle that outputs perfect JSON, every time.
