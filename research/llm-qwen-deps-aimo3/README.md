# RYANSTREAM 1.0

Complete ML infrastructure for 72B+ math models. **14 modules, ~8,500 lines.**

## Stack Overview

```
┌─────────────────────────────────────────────────────────┐
│  Training:      Ryan-Optimizer, Pipeline, Monitor       │
│  Serialize:     RyanFormat (10x compression)            │
│  Inference:     RyanStream + Speculative (2-5x)         │
│  Sampling:      ProofSampler (constraints, backtrack)   │
│  Quantization:  RyanQuant (per-layer, outlier-aware)    │
│  Caching:       PromptCache (prefix KV reuse)           │
│  Tokenizer:     MathTokenizer (number-aware)            │
│  Distributed:   RyanParallel (tensor parallel 2-8 GPU)  │
│  Memory:        SmartCheckpoint (selective)             │
│  Kernels:       FusedAttention, FusedFFN                │
└─────────────────────────────────────────────────────────┘
```

## Usage

### ProofSampler - Constraint-Aware Decoding
```python
from ryanstream import ProofSampler, extract_answer
sampler = ProofSampler(tokenizer, mode='constrained', enable_backtrack=True)
output = sampler.sample(model, input_ids, max_new_tokens=512)
answer = extract_answer(tokenizer.decode(output[0]))
```

### RyanQuant - Math-Optimized Quantization
```python
from ryanstream import RyanQuantizer, QuantConfig
config = QuantConfig(attention_bits=8, ffn_bits=4)
model = RyanQuantizer(config).quantize(model)
```

### PromptCache - Prefix Caching
```python
from ryanstream import PromptCache, CachedGenerator
cache = PromptCache(max_size_gb=4.0)
generator = CachedGenerator(model, tokenizer, cache)
output = generator.generate(prompt)  # 50%+ latency reduction
```

### MathTokenizer - Number-Aware
```python
from ryanstream import MathTokenizer
tokenizer = MathTokenizer()
tokens = tokenizer.encode("x = 12345")  # Numbers stay intact
```

### RyanParallel - Tensor Parallel
```python
# torchrun --nproc_per_node=4 train.py
from ryanstream import setup_distributed
parallel = setup_distributed(tensor_parallel_size=4)
model = parallel.parallelize(model)
```

### SmartCheckpoint - Selective Memory
```python
from ryanstream import SmartCheckpoint
model = SmartCheckpoint(mode='auto').apply(model, sample_input)
```

### Speculative Decoding
```python
from ryanstream import SpeculativeEngine
engine = SpeculativeEngine(target_model, tokenizer, mode='self')
output = engine.generate(input_ids)  # 2-5x speedup
```

## Performance Targets

| Component | Target |
|-----------|--------|
| RyanStream | 45% fewer stalls |
| Speculative | 2-5x speedup |
| ProofSampler | 15-25% accuracy gain |
| RyanQuant | 4x smaller, same accuracy |
| PromptCache | 50%+ latency reduction |
| RyanFormat | 10x compression |
| Fused Kernels | 20% FLOP savings |

## Files

| File | Lines | Purpose |
|------|-------|---------|
| scheduler.py | 850 | RyanStream scheduler |
| speculative.py | 680 | Speculation modes |
| sampler.py | 850 | ProofSampler |
| quant.py | 650 | RyanQuant |
| cache.py | 550 | PromptCache |
| tokenizer.py | 750 | MathTokenizer |
| parallel.py | 700 | RyanParallel |
| checkpoint.py | 550 | SmartCheckpoint |
| format.py | 550 | RyanFormat |
| pipeline.py | 520 | GPU-direct loading |
| monitor.py | 650 | Alerts, cloud burst |
| kernels.py | 500 | Fused ops |
| bridge.py | 520 | Train→inference |

## Requirements
```
torch>=2.0.0
numpy>=1.21.0
triton>=2.0.0  # Optional
```

## Author
Ryan J Cardwell (Archer Phoenix) - ARC Prize 2026
