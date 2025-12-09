# LFBM - LatticeForge Briefing Model

A purpose-built ~150M parameter encoder-decoder model for converting geopolitical risk metrics into intelligence prose.

## Why Not Just Use Claude/GPT?

1. **Cost**: Anthropic Haiku costs $0.25-0.75 per briefing. LFBM costs ~$0.001.
2. **Latency**: Self-hosted = no API queuing, ~100ms inference.
3. **Control**: No rate limits, no content policy issues, no knowledge cutoff problems.
4. **Privacy**: Your data never leaves your infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LFBM (~150M params)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT ENCODER (Custom)                   PROSE DECODER (Transformer)  │
│  ├─ Nation Embeddings (128d)              ├─ 12 layers, 768 hidden     │
│  ├─ Risk Value Buckets (64d)              ├─ 12 attention heads        │
│  ├─ Signal Aggregator                     ├─ 8K vocab (intel domain)   │
│  └─ Cross-attention                       └─ Constrained JSON output   │
│                                                                         │
│  Input: {nations, risks, signals}  →  Output: {briefings JSON}         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Generate Training Data

```bash
# With Supabase (extracts real interactions)
export NEXT_PUBLIC_SUPABASE_URL=your-url
export SUPABASE_SERVICE_ROLE_KEY=your-key
python data/extract_training.py

# Without Supabase (synthetic data only)
python data/extract_training.py
# Creates: training_data_synthetic.jsonl
```

### 2. Train on RunPod

```bash
# SSH into RunPod H200 instance
pip install torch transformers

# Start training (~30 min for 5K examples)
python training/train.py \
  --data training_data.jsonl \
  --epochs 10 \
  --batch_size 8 \
  --output ./checkpoints
```

### 3. Deploy Inference Server

```bash
# On RunPod (or any GPU server)
pip install torch fastapi uvicorn

python inference/server.py \
  --model ./checkpoints/lfbm_final.pt \
  --port 8000

# Test
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "nations": [{"code": "USA", "risk": 0.2}, {"code": "UKR", "risk": 0.85}],
    "signals": {"gdelt_count": 150, "avg_tone": -3.5},
    "categories": {"political": 72, "security": 85}
  }'
```

### 4. Connect from Vercel

Set in Vercel environment:
```
LFBM_ENDPOINT=https://your-runpod-endpoint:8000
LFBM_API_KEY=optional-auth-key
PREFER_LFBM=true
```

The `LFBMClient` is a drop-in replacement:
```typescript
import { getLFBMClient, shouldUseLFBM } from '@/lib/inference/LFBMClient';

if (shouldUseLFBM()) {
  const lfbm = getLFBMClient();
  const briefings = await lfbm.generateFromMetrics(nationData, signals, categories);
}
```

## Cost Comparison

| Provider | Per Briefing | 1K/month | 10K/month |
|----------|--------------|----------|-----------|
| Anthropic Haiku | $0.50 | $500 | $5,000 |
| LFBM (RunPod) | $0.001 | $1 | $10 |
| LFBM (Own H200) | $0.0001 | $0.10 | $1 |

## Training Data Format

JSONL with this structure:
```json
{
  "input_nations": [{"code": "USA", "risk": 0.2, "trend": 0.0}],
  "input_signals": {"gdelt_count": 150, "avg_tone": -3.5},
  "input_categories": {"political": 72, "security": 85},
  "output_briefings": {
    "political": "Risk indicators show stable political environment...",
    "security": "Security posture elevated in monitored regions...",
    "summary": "Global assessment: moderate risk across 15 nations."
  },
  "source": "claude",
  "quality_score": 0.9
}
```

## Files

```
packages/lfbm/
├── model/
│   └── architecture.py    # LFBM model definition
├── training/
│   └── train.py          # Training script for RunPod
├── inference/
│   └── server.py         # FastAPI server + RunPod handler
├── data/
│   └── extract_training.py  # Training data extraction
└── README.md
```

## Requirements

```
torch>=2.0
fastapi
uvicorn
pydantic
```

## License

Internal use only.
