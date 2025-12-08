# LLM_INFERENCE.skill.md

## Production-Grade Large Language Model Inference: Architecture, Optimization, and Deployment

**Version**: 1.0
**Domain**: Machine Learning Systems, Distributed Computing, Performance Engineering
**Prerequisites**: Basic ML concepts, Python/C++, GPU architecture basics
**Complexity**: Advanced

---

## 1. EXECUTIVE SUMMARY

This skill file covers the complete stack for deploying LLMs in production: from tokenization through sampling strategies, memory optimization, batching, quantization, and serving infrastructure.

**Core Challenge**: LLMs are memory-bound, not compute-bound. The bottleneck is moving weights from HBM to compute units, not the actual matrix multiplications.

**Key Insight**: All optimization reduces to: reduce memory footprint, reduce memory movement, maximize hardware utilization.

---

## 2. THE INFERENCE PIPELINE

### 2.1 End-to-End Flow

```
Input Text → Tokenization → Embedding Lookup →
Transformer Blocks (N layers) →
LM Head → Logits → Sampling →
Detokenization → Output Text
```

### 2.2 Two Phases of Generation

**Prefill (Prompt Processing)**:
- Process all input tokens in parallel
- Compute-bound (matrix multiplications)
- Generates KV cache for all input positions
- One forward pass

**Decode (Token Generation)**:
- Generate one token at a time
- Memory-bound (loading weights for single token)
- Appends to KV cache each step
- Autoregressive: output becomes next input

**Implication**: Prefill and decode have different optimization strategies.

### 2.3 Computational Costs

For a transformer with:
- `d_model`: Hidden dimension
- `n_layers`: Number of layers
- `n_heads`: Attention heads
- `seq_len`: Sequence length
- `vocab_size`: Vocabulary size

**Per-token FLOPs** (approximate):
- Attention: `4 * seq_len * d_model` (scales with context)
- FFN: `16 * d_model²` (constant per token)
- LM Head: `2 * d_model * vocab_size`

**Memory**: `2 * n_layers * seq_len * d_model` bytes for KV cache (fp16)

---

## 3. TOKENIZATION

### 3.1 Tokenizer Types

**Byte-Pair Encoding (BPE)**:
- Used by GPT models
- Iteratively merges most frequent pairs
- Vocabulary learned from corpus

**WordPiece**:
- Used by BERT
- Similar to BPE but different merge criterion
- Maximizes likelihood of training data

**SentencePiece**:
- Language-agnostic
- Treats input as raw byte stream
- Unigram or BPE algorithm

**Tiktoken**:
- OpenAI's fast BPE implementation
- Regex-based pre-tokenization

### 3.2 Special Tokens

- `<|bos|>`: Beginning of sequence
- `<|eos|>`: End of sequence
- `<|pad|>`: Padding token
- `<|unk|>`: Unknown token
- Chat templates: `<|user|>`, `<|assistant|>`, etc.

### 3.3 Tokenization Gotchas

**Token Healing**: Some tokenizers split words differently based on context. "Hello" vs " Hello" may tokenize differently.

**Multilingual**: Non-English text often fragments into many tokens (poor efficiency).

**Numbers**: Often split digit-by-digit, affecting arithmetic capabilities.

---

## 4. ATTENTION MECHANISMS

### 4.1 Standard Multi-Head Attention

```python
Q = X @ W_q  # [batch, seq, d_model] @ [d_model, d_model]
K = X @ W_k
V = X @ W_v

# Split heads
Q = Q.view(batch, seq, n_heads, d_head).transpose(1, 2)
K = K.view(batch, seq, n_heads, d_head).transpose(1, 2)
V = V.view(batch, seq, n_heads, d_head).transpose(1, 2)

# Attention
scores = Q @ K.transpose(-2, -1) / sqrt(d_head)  # [batch, heads, seq, seq]
scores = scores.masked_fill(causal_mask, -inf)
attn = softmax(scores, dim=-1)
out = attn @ V  # [batch, heads, seq, d_head]

# Merge heads
out = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
out = out @ W_o
```

### 4.2 Multi-Query Attention (MQA)

All heads share same K and V projections. Reduces KV cache by `n_heads`x.

**Used by**: PaLM, Falcon

### 4.3 Grouped-Query Attention (GQA)

Compromise: Groups of heads share K/V. Reduces KV cache by `n_groups`x.

**Used by**: Llama 2 70B, Mistral

### 4.4 Flash Attention

**Problem**: Standard attention materializes `[seq, seq]` attention matrix - O(n²) memory.

**Solution**: Tiled computation that fuses operations and avoids materializing full matrix.

**Implementation**:
1. Load blocks of Q, K, V to SRAM
2. Compute attention for block
3. Write output, don't store attention matrix
4. Use online softmax normalization

**Result**: O(n) memory, 2-4x faster on GPU, exact (not approximate).

### 4.5 Paged Attention

**Problem**: KV cache is pre-allocated contiguously, wasting memory for variable-length sequences.

**Solution**: Virtual memory for KV cache - allocate in non-contiguous pages.

**Used by**: vLLM, TGI

---

## 5. KV CACHE MANAGEMENT

### 5.1 Memory Calculation

For Llama 2 7B:
- 32 layers
- 32 heads
- 128 d_head
- 4096 d_model

KV cache per token (fp16):
```
2 (K+V) * 32 layers * 4096 d_model * 2 bytes = 512 KB/token
```

For 4096 context: **2 GB just for KV cache**

### 5.2 KV Cache Compression

**Quantization**: Store KV in int8 or int4 (2-4x reduction)

**Sliding Window**: Only keep last N tokens (Mistral uses 4096 window)

**Sparse Attention**: Only attend to subset (Longformer, BigBird patterns)

**Token Dropping**: Remove unimportant tokens from cache

### 5.3 KV Cache Reuse

**Prefix Caching**: If multiple requests share prefix, compute KV once.

**Example**: System prompt is same for all requests - cache it.

**Implementation**: Hash prefix tokens, store KV in cache, lookup before compute.

---

## 6. SAMPLING STRATEGIES

### 6.1 Greedy Decoding

```python
next_token = argmax(logits)
```

**Pros**: Deterministic, fast
**Cons**: Repetitive, no diversity

### 6.2 Temperature Sampling

```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```

- `temperature < 1`: Sharper distribution (more deterministic)
- `temperature > 1`: Flatter distribution (more random)
- `temperature = 0`: Equivalent to greedy

### 6.3 Top-k Sampling

```python
top_k_logits, top_k_indices = topk(logits, k)
probs = softmax(top_k_logits)
next_token = top_k_indices[sample(probs)]
```

Only sample from top k tokens. Prevents rare token selection.

### 6.4 Top-p (Nucleus) Sampling

```python
sorted_probs, sorted_indices = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
cutoff = first_index_where(cumsum > p)
top_p_probs = sorted_probs[:cutoff]
next_token = sorted_indices[sample(normalize(top_p_probs))]
```

Dynamic k based on probability mass. Adapts to distribution shape.

### 6.5 Beam Search

Maintain k best sequences, expand each, keep top k overall.

```python
beams = [([], 0.0)]  # (tokens, log_prob)

for step in range(max_length):
    candidates = []
    for tokens, score in beams:
        logits = model(tokens)
        top_k = topk(logits, beam_width)
        for token, log_prob in top_k:
            candidates.append((tokens + [token], score + log_prob))
    beams = sorted(candidates, key=lambda x: x[1])[-beam_width:]
```

**Pros**: Finds higher probability sequences
**Cons**: Repetitive, less diverse, slower

### 6.6 Speculative Decoding

Use small "draft" model to propose N tokens, verify with large model in parallel.

```python
# Draft model proposes tokens
draft_tokens = draft_model.generate(n_tokens)

# Target model verifies in single forward pass
target_logits = target_model(draft_tokens)

# Accept prefix that matches
for i, token in enumerate(draft_tokens):
    if sample(target_logits[i]) == token:
        accept(token)
    else:
        resample_from(target_logits[i])
        break
```

**Speedup**: 2-3x for matched draft/target distributions.

### 6.7 Repetition Penalties

**Frequency Penalty**: Reduce logit based on token count in context
**Presence Penalty**: Reduce logit if token appeared at all
**Repetition Penalty**: Multiplicative penalty on repeated tokens

---

## 7. QUANTIZATION

### 7.1 Data Types

| Type | Bits | Range | Use |
|------|------|-------|-----|
| FP32 | 32 | ±3.4e38 | Training |
| FP16 | 16 | ±65504 | Inference |
| BF16 | 16 | ±3.4e38 | Training/Inference |
| INT8 | 8 | -128 to 127 | Quantized inference |
| INT4 | 4 | -8 to 7 | Aggressive quantization |
| FP8 | 8 | Various | H100 native |

### 7.2 Post-Training Quantization (PTQ)

Quantize weights after training without fine-tuning.

**Absmax Quantization**:
```python
scale = max(abs(weights)) / 127
quantized = round(weights / scale)
```

**Zero-point Quantization**:
```python
scale = (max(weights) - min(weights)) / 255
zero_point = round(-min(weights) / scale)
quantized = round(weights / scale) + zero_point
```

### 7.3 GPTQ

**Algorithm**:
1. Quantize weights column by column
2. For each column, compute optimal rounding
3. Compensate error in remaining columns
4. Uses Hessian information (second-order)

**Result**: 4-bit quantization with minimal accuracy loss.

### 7.4 AWQ (Activation-aware Weight Quantization)

**Insight**: Some weights matter more (those multiplied by large activations).

**Algorithm**:
1. Run calibration set, collect activation magnitudes
2. Scale important weights up before quantization
3. Scale activations down to compensate

**Result**: Better than GPTQ for 4-bit.

### 7.5 GGUF/GGML Quantization

llama.cpp format with various quantization levels:

- `Q4_0`: 4-bit, simple
- `Q4_K_M`: 4-bit, k-quants medium
- `Q5_K_M`: 5-bit, k-quants medium
- `Q8_0`: 8-bit

**K-quants**: Different quantization for different weight matrices based on importance.

### 7.6 Quantization-Aware Training (QAT)

Train with simulated quantization, model learns to be robust.

**Approaches**:
- QLoRA: Quantized base + LoRA adapters
- QLORA: 4-bit NormalFloat + double quantization

---

## 8. BATCHING STRATEGIES

### 8.1 Static Batching

All sequences in batch must:
- Start together
- End together (pad shorter sequences)

**Problem**: Wastes compute on padding.

### 8.2 Continuous Batching

**Insight**: Different sequences finish at different times. Immediately fill slot with new request.

**Implementation**:
1. Maintain batch of active sequences
2. When sequence finishes, remove it
3. Insert waiting request into freed slot
4. Continue generation

**Result**: Near 100% GPU utilization, much higher throughput.

### 8.3 Iteration-Level Scheduling

Each iteration (token generation), decide:
- Which sequences to process
- Whether to preempt for new requests
- How to allocate memory

**Policies**:
- FCFS: First come, first served
- Shortest job first
- Priority-based

---

## 9. MEMORY OPTIMIZATION

### 9.1 Gradient Checkpointing (Training)

Don't store all activations. Recompute during backward pass.

**Trade-off**: Memory vs compute (usually worth it).

### 9.2 Model Parallelism

**Tensor Parallelism**: Split individual layers across GPUs
**Pipeline Parallelism**: Different layers on different GPUs
**Sequence Parallelism**: Split along sequence dimension

### 9.3 Offloading

Move weights/KV cache to CPU RAM or disk, load on demand.

**Hierarchy**: GPU HBM → GPU L2 → CPU RAM → SSD

**When to use**: Model doesn't fit in GPU memory.

### 9.4 Memory Pooling

Pre-allocate memory pools, reuse for different requests.

Avoids allocation/deallocation overhead.

---

## 10. SERVING INFRASTRUCTURE

### 10.1 vLLM

**Key Features**:
- PagedAttention (non-contiguous KV cache)
- Continuous batching
- Prefix caching
- Speculative decoding

**Architecture**:
- Python frontend (FastAPI)
- C++/CUDA backend
- Custom CUDA kernels

### 10.2 TGI (Text Generation Inference)

**Key Features**:
- Rust backend (fast, safe)
- Continuous batching
- Quantization support
- Token streaming

**Used by**: HuggingFace Inference Endpoints

### 10.3 Triton Inference Server

**Key Features**:
- Multi-framework (PyTorch, TensorFlow, ONNX, TensorRT)
- Dynamic batching
- Model ensembles
- Metrics and monitoring

**Best for**: Production deployments at scale.

### 10.4 llama.cpp

**Key Features**:
- Pure C++ (no Python dependencies)
- CPU and GPU support
- Extensive quantization (GGUF)
- Cross-platform

**Best for**: Edge deployment, local inference.

### 10.5 Comparison

| Feature | vLLM | TGI | Triton | llama.cpp |
|---------|------|-----|--------|-----------|
| Speed | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Memory | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Ease | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Edge | ❌ | ❌ | ❌ | ⭐⭐⭐ |

---

## 11. PERFORMANCE METRICS

### 11.1 Throughput Metrics

**Tokens/second**: Total tokens generated per second
**Requests/second**: Completed requests per second
**Time to First Token (TTFT)**: Latency for first token

### 11.2 Latency Metrics

**Inter-token Latency (ITL)**: Time between tokens
**End-to-end Latency**: Total request time
**P50/P99 Latency**: Percentile latencies

### 11.3 Efficiency Metrics

**Model FLOPs Utilization (MFU)**: Actual FLOPs / Peak FLOPs
**Memory Bandwidth Utilization (MBU)**: Actual bandwidth / Peak bandwidth

**Typical MFU**: 30-50% for inference (memory-bound)

---

## 12. OPTIMIZATION CHECKLIST

### 12.1 Quick Wins

- [ ] Use Flash Attention
- [ ] Use continuous batching
- [ ] Enable KV cache
- [ ] Use appropriate quantization (Q4_K_M good default)
- [ ] Use BF16 if supported

### 12.2 Intermediate

- [ ] Implement prefix caching
- [ ] Tune batch size for hardware
- [ ] Use speculative decoding if draft model available
- [ ] Profile memory usage
- [ ] Optimize tokenization (batch tokenize)

### 12.3 Advanced

- [ ] Custom CUDA kernels for specific operations
- [ ] Tensor parallelism for large models
- [ ] Tune sampling parameters for use case
- [ ] Implement request prioritization
- [ ] Add monitoring and autoscaling

---

## 13. COMMON ISSUES AND SOLUTIONS

### 13.1 Out of Memory

**Symptoms**: CUDA OOM, process killed
**Solutions**:
- Reduce batch size
- Use quantization
- Enable gradient checkpointing
- Reduce context length
- Use model parallelism

### 13.2 Slow Generation

**Symptoms**: Low tokens/second
**Solutions**:
- Profile: is it prefill or decode?
- Use Flash Attention
- Increase batch size (better utilization)
- Check for CPU bottlenecks (tokenization)
- Use faster quantization format

### 13.3 Repetitive Output

**Symptoms**: Model repeats phrases
**Solutions**:
- Increase temperature
- Add repetition penalty
- Use nucleus sampling
- Check for training data issues

### 13.4 Quality Degradation

**Symptoms**: Wrong answers after quantization
**Solutions**:
- Use less aggressive quantization
- Try different quantization method (AWQ vs GPTQ)
- Calibrate on representative data
- Consider QAT

---

## 14. HARDWARE CONSIDERATIONS

### 14.1 GPU Selection

| GPU | HBM | Bandwidth | Best For |
|-----|-----|-----------|----------|
| A100 40GB | 40 GB | 1.5 TB/s | Training, large inference |
| A100 80GB | 80 GB | 2.0 TB/s | Very large models |
| H100 | 80 GB | 3.35 TB/s | Maximum performance |
| A10 | 24 GB | 600 GB/s | Cost-effective inference |
| RTX 4090 | 24 GB | 1.0 TB/s | Consumer, local |

### 14.2 Memory Bandwidth is Key

LLM inference is memory-bandwidth bound. More bandwidth = more tokens/second.

**Rule of thumb**: tokens/sec ≈ bandwidth / (2 * model_params)

For 7B model on A100 (1.5 TB/s):
```
1.5 TB/s / (2 * 7B * 2 bytes) ≈ 53 tokens/second per sequence
```

### 14.3 Multi-GPU Scaling

- **2 GPUs**: ~1.8x speedup (communication overhead)
- **4 GPUs**: ~3.2x speedup
- **8 GPUs**: ~5.5x speedup

Diminishing returns due to communication.

---

## 15. EMERGING TECHNIQUES

### 15.1 Mixture of Experts (MoE)

Only activate subset of parameters per token. More params, same FLOPs.

**Examples**: Mixtral 8x7B (47B params, 13B active)

### 15.2 State Space Models

Mamba, RWKV - linear attention alternatives. O(n) instead of O(n²).

**Trade-off**: Different quality characteristics, faster inference.

### 15.3 Sparse Attention

Only compute attention on subset of positions.

**Patterns**: Local window, stride, global tokens, learned

### 15.4 KV Cache Compression

Streaming LLM, H2O (Heavy Hitter Oracle) - dynamic cache management.

---

## 16. REFERENCES

**Papers**:
- Vaswani et al. (2017). Attention Is All You Need
- Dao et al. (2022). FlashAttention
- Frantar et al. (2022). GPTQ
- Lin et al. (2023). AWQ
- Kwon et al. (2023). PagedAttention (vLLM)
- Leviathan et al. (2022). Speculative Decoding

**Implementations**:
- vLLM: https://github.com/vllm-project/vllm
- llama.cpp: https://github.com/ggerganov/llama.cpp
- TGI: https://github.com/huggingface/text-generation-inference

---

## 17. VERSION HISTORY

- **v1.0** (2024-11): Initial comprehensive version

---

*This skill covers the full inference stack. For training, see separate EFFICIENT_TRAINING skill.*
