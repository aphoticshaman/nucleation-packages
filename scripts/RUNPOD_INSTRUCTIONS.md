# Elle-72B-Ultimate RunPod Deployment

## Quick Start (4x H200)

### 1. Setup Environment

```bash
# Clone repo or upload scripts via RunPod GUI
cd /workspace

# Install dependencies
pip install -r requirements-deploy.txt
```

### 2. Merge LoRA Adapter

```bash
python merge_lora.py \
    --adapter /workspace/elle-interleaved-out \
    --output /workspace/elle-merged
```

**IMPORTANT**: The script uses `Qwen/Qwen2.5-72B-Instruct` (non-AWQ) as base because
the LoRA was trained on the full-precision model. AWQ has different MLP dimensions
(29696 vs 29568) which causes size mismatch errors.

This takes ~15-20 minutes on H200s (downloads ~140GB base model).

### 3. Upload to HuggingFace (Optional)

```bash
# Login first
huggingface-cli login

# Upload with full model card
python upload_to_hf.py \
    --model /workspace/elle-merged \
    --repo aphoticshaman/Elle-72B-Ultimate
```

### 4. Start vLLM Server

```bash
python deploy_vllm.py \
    --model /workspace/elle-merged \
    --tp 4 \
    --port 8000
```

Or directly with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/elle-merged \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

### 5. Test Endpoint

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/workspace/elle-merged",
        "messages": [
            {"role": "system", "content": "You are Elle, a geopolitical intelligence analyst."},
            {"role": "user", "content": "What is the current risk level in the Taiwan Strait?"}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }'
```

## RunPod Serverless

For production, configure as RunPod Serverless endpoint:

1. Use vLLM worker template
2. Set environment variables:
   - `MODEL_NAME=/workspace/elle-merged` (or HF repo after upload)
   - `TENSOR_PARALLEL_SIZE=4`
   - `MAX_MODEL_LEN=32768`
3. Configure handler to use OpenAI-compatible API

## Environment Variables for LatticeForge

After deployment, set these in Vercel:

```
LFBM_ENDPOINT=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
LFBM_API_KEY=your_runpod_api_key
```

## Files in This Directory

- `merge_lora.py` - Merge LoRA adapter with base model
- `upload_to_hf.py` - Upload to HuggingFace with model card
- `deploy_vllm.py` - Start vLLM inference server
- `requirements-deploy.txt` - Python dependencies
