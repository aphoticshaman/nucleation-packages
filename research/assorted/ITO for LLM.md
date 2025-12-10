# Inference-Time Optimization for LLM Mathematical Reasoning: A Technical Guide for Competition-Grade Performance

**The paradigm has shifted**: 7B models now match OpenAI o1 on math olympiad problems through inference-time compute scaling, MCTS-guided reasoning, and process reward models. This report synthesizes cutting-edge techniques for maximizing mathematical reasoning performance under Kaggle competition constraints (H100 80GB, 9-hour runtime, offline inference).

The core insight from 2024-2025 research: **test-time compute scaling can substitute for 14× larger pretrained models** on intermediate-difficulty problems. DeepSeek-R1 achieves 79.8% on AIME 2024 with pure RL training, while rStar-Math pushes a 7B model to 90% on MATH benchmark through self-evolved MCTS reasoning. For competition environments, the winning formula combines tool-integrated reasoning (SC-TIR), process reward model verification, and compute-optimal sampling strategies.

---

## Test-time compute scales logarithmically with reasoning performance

Recent research demonstrates that allocating more inference-time computation provides diminishing but substantial returns. OpenAI o1 achieves **74% on AIME 2024 with pass@1**, jumping to **93% with best-of-1000 sampling and reranking**. DeepSeek-R1-0528 uses an average of **23K thinking tokens** per AIME problem (up from 12K in earlier versions), yielding 91.4% accuracy.

The key mechanisms for test-time scaling fall into two categories. **Sequential reasoning** (thinking longer) involves extended chain-of-thought generation, self-verification, and reflection within a single forward pass. **Parallel sampling** (thinking more) generates multiple independent solutions and aggregates via voting or verification. Stanford's s1 paper demonstrates that with only 1,000 training samples plus "budget forcing" (appending "Wait" to extend reasoning), a 32B model matches o1-preview performance.

Practical recommendations from DeepSeek-R1 include: temperature **0.6**, top_p **0.95**, max_tokens **32,768**, and forcing thinking initiation with a `<think>` prefix. Token budgets of **16K-24K** work optimally for hard olympiad problems, with diminishing returns beyond 30K tokens.

---

## Process reward models outperform outcome reward models by significant margins

The distinction between **Process Reward Models (PRMs)** and **Outcome Reward Models (ORMs)** represents one of the most impactful findings for competition settings. OpenAI's PRM800K research shows that **PRM best-of-1860 achieves 78.2% on MATH versus 72.4% for ORM and 69.6% for majority voting** at the same sample count. PRMs provide dense, step-level feedback identifying exactly where reasoning goes wrong.

Three PRM training approaches have emerged with different cost-quality tradeoffs:
- **Human annotation**: Gold standard but expensive (PRM800K approach)
- **Monte Carlo estimation**: Automated via rollout completion rates (Math-Shepherd)
- **LLM-as-judge with consensus filtering**: Combines MC estimation with language model evaluation for best quality-efficiency balance

For aggregation, **product scoring** (multiplying step probabilities) outperforms minimum-step scoring. OmegaPRM uses MCTS-based divide-and-conquer annotation to generate 1.5M process supervision labels automatically. The rStar-Math system introduces **Process Preference Models (PPMs)** trained via preference learning rather than Q-value estimation, which proves essential for effective MCTS guidance.

Best-of-N sampling shows diminishing returns after **N≈40-64**, with compute-optimal allocation achieving **4× better efficiency** than uniform sampling across all problems.

---

## Self-correction without external feedback consistently degrades performance

A landmark ICLR 2024 finding from Google DeepMind establishes that **LLMs cannot self-correct reasoning without external signals**. GPT-4 on GSM8K drops from 95.5% to 89.0% after two self-correction rounds. The analysis reveals why: when asked to verify, models change 8.8% of correct answers to incorrect while only fixing 7.6% of errors—a net negative.

However, **externally-grounded verification works reliably**. Code execution provides deterministic feedback, enabling genuine self-correction when intermediate calculations can be verified. Symbolic verification via SymPy catches numerical errors that language-only reasoning misses. Process reward models identify error locations even when models cannot.

The practical implication: **never prompt "check your work" without external verification**. Instead, use code-augmented reasoning (SC-TIR), process reward scoring, or formal verification. Multi-agent debate, despite intuitive appeal, does not outperform simple self-consistency when controlling for sample count—the benefits come from "consistency" across samples, not from "correction" through debate.

---

## Tool-integrated reasoning achieves state-of-the-art competition performance

**SC-TIR (Self-Consistency with Tool-Integrated Reasoning)** powered the winning AIMO Progress Prize solution. The approach interleaves natural language reasoning with Python code execution in format: reasoning → code → output → reasoning → code → output → final answer.

Implementation parameters from NuminaMath winning solution: **N=64 trajectories**, temperature **0.3-0.6**, max tokens **32,768**, with code execution timeouts of 30 seconds per block. Allowed imports should include sympy, numpy, math, itertools, and fractions. The final answer extraction uses a `final_answer()` function call.

**rStar-Math** represents the current small-model state-of-the-art, pushing Qwen2.5-Math-7B from 58.8% to **90.0% on MATH** through MCTS-guided reasoning with code verification. Key innovations include code-augmented chain-of-thought synthesis, process preference model training via preference learning (not Q-values), and a 4-round self-evolution recipe building policy and PPM from scratch.

For formal verification, **DeepSeek-Prover-V2** achieves 88.9% on miniF2F-test with Lean4 integration. AlphaProof's silver medal at IMO 2024 (solving the hardest problem P6) demonstrates the ceiling for formal verification approaches.

---

## Memory optimization enables larger batch sizes and higher throughput

**PagedAttention** (vLLM) reduces memory waste from 60-80% to under 4% by applying OS-style virtual memory paging to KV cache. This enables **2-4× throughput improvement** through larger effective batch sizes. Combined with **Flash Attention 3** optimized for H100 (achieving 75% GPU utilization versus 35% for FA-2), the memory efficiency gains compound.

KV-cache quantization provides additional memory savings with minimal accuracy impact:

| Format | Memory Reduction | Accuracy Impact |
|--------|-----------------|-----------------|
| FP8 | 2× | <0.1 ppl increase |
| INT8 | 2× | Near-lossless |
| INT4 | 4× | Slight degradation |

**Prefix caching** (vLLM's APC or SGLang's RadixAttention) dramatically improves throughput when problems share common prefixes—cache hit rates of 50-99% are typical for structured prompts. For competition workloads with consistent system prompts, this provides multiplicative efficiency gains.

For H100 80GB with a 70B model, the recommended configuration: **FP8 quantization** for both weights and KV cache, **chunked prefill** for balanced latency, **tensor parallelism = 1-2**, targeting **90% GPU memory utilization**. This yields approximately 100-150 concurrent sequences at 4K context length.

---

## Speculative decoding accelerates generation by 2-4× without quality loss

Speculative decoding uses a small draft model to propose multiple tokens, which the target model verifies in a single forward pass. The technique provides **lossless acceleration**—output distributions match exactly through rejection sampling.

Current state-of-the-art approaches and their speedups:

| Method | Speedup | Notes |
|--------|---------|-------|
| Standard draft model (1B→70B) | 2.8-3.6× | Llama 3.2 1B drafting for 70B |
| EAGLE-3 | 3.0-6.5× | Feature-level speculation, best performance |
| Medusa heads | 2.2-3.6× | No separate model needed |
| Self-speculative | Up to 2× | Layer skipping, zero memory overhead |

Speculative decoding helps most at **batch size = 1** (latency-sensitive scenarios). Speedup degrades significantly at higher batch sizes—from 1.3× at batch=2 to 0.7× at batch=48 with EAGLE in vLLM. For competition settings generating many samples per problem, speculative decoding may not be optimal; continuous batching and PagedAttention provide better throughput.

High temperatures reduce acceptance rates by ~29%, so speculative decoding benefits most from moderate temperatures (0.6-0.8) rather than high-temperature sampling.

---

## Multi-model architectures enable cost-quality optimization through cascading

RouterBench demonstrates that **oracle routing achieves ~96% of GPT-4 performance at ~8% of the cost**. Practical cascade routing implementations start with smaller models, escalating based on confidence thresholds.

The recommended competition architecture uses three tiers:

1. **Fast screening** (7B model): Handle easy problems with high-confidence answers (>0.9)
2. **Self-consistency** (32B-72B model): Generate 5-8 reasoning paths with weighted majority voting
3. **Full reasoning** (R1-class model): Deploy for complex problems requiring extended thinking

**Confidence-Informed Self-Consistency (CISC)** reduces required samples by 40%+ while maintaining accuracy by weighting votes based on model confidence. **Reasoning-Aware Self-Consistency (RASC)** enables dynamic early-stopping, reducing average sample count from 64 to ~20 through convergence monitoring.

For MoE models like Mixtral, **Dynamic Experts Search** varies the number of activated experts across samples, creating complementary solution sets with low Jaccard similarity—an additional diversity source beyond temperature-based sampling.

---

## Hyperparameter recommendations for competition deployment

Based on synthesis across all research findings, the following configurations optimize for AIME-level competition problems on H100 80GB:

**Sampling configuration:**
```
temperature: 0.6
top_p: 0.95
max_tokens: 32768
thinking_budget: 16000-24000 tokens
```

**SC-TIR parameters:**
```
n_trajectories: 64 (saturation point)
code_timeout: 30 seconds
max_turns: 5 iterations
allowed_imports: sympy, numpy, math, itertools, fractions
```

**Best-of-N with PRM:**
```
easy_problems (pass@1 > 60%): N=8, sequential revision
medium_problems: N=32, PRM-weighted voting
hard_problems: N=64, PRM best-of-N with answer clustering
```

**Memory optimization:**
```
kv_cache_dtype: fp8
quantization: fp8 or int4-awq
gpu_memory_utilization: 0.90
enable_chunked_prefill: true
enable_prefix_caching: true
```

---

## Model selection based on demonstrated benchmark performance

Current AIME 2024 performance by model tier:

| Model | AIME 2024 | MATH-500 | Deployment Notes |
|-------|-----------|----------|------------------|
| o3 (high compute) | ~87-93% | 96.7% | Not accessible offline |
| DeepSeek-R1-0528 | 91.4% | 97.3% | Requires API or self-hosting |
| DeepSeek-R1-Distill-32B | 72.6% | 94.3% | Fits H100 80GB with FP8 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5% | - | Memory efficient, good for cascading |
| rStar-Math (7B+PPM) | 53.3% | 90.0% | Requires MCTS infrastructure |
| Qwen2.5-Math-72B | ~60% | 85.9% | Strong base for TIR fine-tuning |

For offline competition deployment, **DeepSeek-R1-Distill-32B** with SC-TIR provides the best accuracy-feasibility tradeoff. Adding a 7B screening model for easy problems and allocating remaining compute to harder problems through adaptive N selection maximizes expected score.

---

## Conclusion

The 2024-2025 advances establish a clear hierarchy of effective techniques for competition-grade mathematical reasoning. **Test-time compute scaling, tool-integrated reasoning, and process reward verification** form the core stack, with demonstrated gains of 15-30% over baseline approaches.

Key implementation priorities for a 9-hour H100 competition:
1. Deploy SC-TIR with code execution (primary accuracy driver)
2. Use PRM-weighted best-of-N with adaptive N based on problem difficulty
3. Enable FP8 quantization and prefix caching for throughput
4. Avoid intrinsic self-correction—use only externally-grounded verification
5. Consider speculative decoding only for latency-sensitive single-generation scenarios

The field consensus points toward **MCTS + Process Rewards + RL training + Test-Time Compute** as the winning formula. Small models (7B-32B) with proper inference-time optimization now match or exceed much larger models on olympiad-level mathematics, making advanced techniques accessible within competition compute constraints.