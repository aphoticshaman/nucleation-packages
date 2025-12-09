# Fine-Tune Your Own Briefing Model on RunPod

Total cost: ~$5-10 | Time: ~2 hours

## Step 1: Prepare Training Data (on your local machine)

```bash
cd packages/lfbm/axolotl

# Generate synthetic training data (3000 examples)
python prepare_data.py training_data.jsonl

# This creates training_data.jsonl (~5MB)
```

## Step 2: Create RunPod Serverless Endpoint

1. Go to RunPod → Serverless → **Axolotl Fine-Tuning**
2. Click "Deploy"
3. Configure:
   - **GPU**: Any (A100-80GB recommended, but A10 works)
   - **Max Workers**: 1
   - **Idle Timeout**: 60 seconds

## Step 3: Upload Training Data

Option A - **Via RunPod UI**:
1. In your endpoint settings, find "Network Volume"
2. Upload `training_data.jsonl` and `config.yaml`

Option B - **Via API** (recommended):
```bash
# Get your endpoint ID from RunPod dashboard
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

# Upload files (RunPod provides S3-compatible storage)
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/upload" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -F "file=@training_data.jsonl" \
  -F "file=@config.yaml"
```

## Step 4: Start Fine-Tuning

```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "config_path": "/workspace/config.yaml"
    }
  }'
```

This returns a job ID. Training takes ~1-2 hours.

## Step 5: Check Training Status

```bash
JOB_ID="your-job-id"
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## Step 6: Download Your Model

When complete, the model is at `/workspace/output/latticeforge-briefing-3b`

```bash
# Download the LoRA adapter (small, ~100MB)
curl "https://api.runpod.ai/v2/${ENDPOINT_ID}/download?path=/workspace/output/latticeforge-briefing-3b" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -o latticeforge-briefing-3b.tar.gz
```

## Step 7: Deploy with vLLM

1. Go to RunPod → Serverless → **vLLM**
2. Click "Deploy"
3. Configure:
   - **Model**: `Qwen/Qwen2.5-3B-Instruct`
   - **LoRA Adapter**: Upload your `latticeforge-briefing-3b` adapter
   - **GPU**: A10 or better
   - **Max Workers**: 1 (scale up as needed)

4. Note your endpoint URL: `https://api.runpod.ai/v2/YOUR_VLLM_ENDPOINT/`

## Step 8: Connect to LatticeForge

In Vercel environment variables:
```
LFBM_ENDPOINT=https://api.runpod.ai/v2/YOUR_VLLM_ENDPOINT
LFBM_API_KEY=your-runpod-api-key
PREFER_LFBM=true
```

## Step 9: Test It

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_VLLM_ENDPOINT/openai/v1/chat/completions" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a prose translation engine..."},
      {"role": "user", "content": "NATIONS:\n  USA: risk=15% →\n  UKR: risk=85% ↑\n\nGenerate JSON briefings."}
    ],
    "max_tokens": 1000
  }'
```

## Cost Breakdown

| Step | Time | Cost |
|------|------|------|
| Training (Axolotl) | ~2 hours | ~$5-8 |
| Inference (vLLM) | Per request | ~$0.001 |

**Monthly costs at different scales:**
- 100 briefings/month: ~$0.10
- 1,000 briefings/month: ~$1.00
- 10,000 briefings/month: ~$10.00

vs Anthropic Haiku:
- 100 briefings/month: ~$50
- 1,000 briefings/month: ~$500
- 10,000 briefings/month: ~$5,000

## Troubleshooting

**Training fails with OOM:**
- Use a larger GPU (A100-80GB)
- Or reduce `micro_batch_size` to 1 in config.yaml

**Model outputs garbage:**
- Check training completed successfully
- Verify training data format is correct
- Try increasing `num_epochs` to 5

**Slow inference:**
- Enable Flash Attention in vLLM settings
- Use A100 instead of A10 for 2-3x speedup

## Alternative: Push to HuggingFace

If you want to share your model:

1. Edit `config.yaml`:
   ```yaml
   hub_model_id: your-username/latticeforge-briefing-3b
   ```

2. Set HuggingFace token:
   ```bash
   export HF_TOKEN=your-token
   ```

3. After training, it auto-pushes to HuggingFace Hub
