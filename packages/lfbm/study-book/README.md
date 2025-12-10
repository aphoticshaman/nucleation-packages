# LatticeForge Fine-Tuning Study Book

**Goal**: Techniques that are 10x better than standard LoRA/QLoRA

**Budget**: $50 | **Hardware**: 1-4x H200 @ $4-$18/hr

---

## TL;DR - Which Config to Use

| Technique | GPU Hours | Cost | Expected Gain vs LoRA | Best For |
|-----------|-----------|------|----------------------|----------|
| **DoRA + NEFTune** | 6h @ 2x H200 | ~$48 | +15-25% accuracy | Most tasks |
| **GaLore** | 10h @ 1x H200 | ~$40 | Full-rank quality | Math/reasoning |
| **ReLoRA (3 iterations)** | 4h @ 2x H200 | ~$32 | Pseudo full-rank | General |
| **ORPO** | 3h @ 2x H200 | ~$24 | Better alignment | Style/format |

---

## The Techniques (Ranked by Impact)

### 1. DoRA (Weight-Decomposed Low-Rank Adaptation)
**Paper**: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)

**What it does**: Decomposes pretrained weights into **magnitude** and **direction** components, then applies LoRA only to the direction. This mimics how full fine-tuning works.

```
W = m * (W_0 + BA) / ||W_0 + BA||

Where:
- m = magnitude (learned scalar per output dimension)
- W_0 = pretrained weights
- BA = LoRA update (direction only)
```

**Why it's better**:
- LoRA changes BOTH magnitude and direction simultaneously (entangled)
- Full fine-tuning changes them independently
- DoRA matches full FT's learning pattern → +15% on commonsense reasoning

**Axolotl support**: Native in recent versions
```yaml
adapter: lora
lora_use_dora: true
```

---

### 2. NEFTune (Noisy Embeddings Fine-Tuning)
**Paper**: [NEFTune: Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)

**What it does**: Adds uniform noise to embedding vectors during training.

```
embeddings += uniform(-α/√d, α/√d)

Where:
- α = noise_alpha (typically 5-15)
- d = embedding dimension
```

**Why it's better**:
- Acts as regularization preventing overfitting
- +10-15% on AlpacaEval, MT-Bench
- Almost zero overhead
- Works with ANY other technique

**Axolotl support**: Native
```yaml
neftune_noise_alpha: 5
```

---

### 3. GaLore (Gradient Low-Rank Projection)
**Paper**: [GaLore: Memory-Efficient LLM Training](https://arxiv.org/abs/2403.03507)

**What it does**: Projects gradients to low-rank space, enabling full-rank training with LoRA-like memory.

```
Instead of storing full gradients G (d×d):
1. Project: G_low = P^T @ G @ Q  (r×r, where r << d)
2. Update: W -= lr * (P @ G_low @ Q^T)
3. Periodically re-compute P, Q via SVD
```

**Why it's better**:
- **Full-rank training** with 8-bit optimizer memory
- Better than LoRA for math/reasoning (no rank bottleneck)
- 72B model trainable on ~200GB VRAM (vs 1TB+ for AdamW)

**Axolotl support**: Via custom trainer
```yaml
optimizer: galore_adamw_8bit
galore_rank: 128
galore_update_proj_gap: 200
galore_scale: 0.25
```

---

### 4. ReLoRA (Iterative LoRA Merging)
**Paper**: [ReLoRA: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695)

**What it does**: Trains LoRA → merges into base → resets LoRA → repeat. Each iteration adds rank.

```
Iteration 1: W_1 = W_0 + B_1 @ A_1
Iteration 2: W_2 = W_1 + B_2 @ A_2  (B_2, A_2 are new)
...
After N iterations: Effective rank = N × r
```

**Why it's better**:
- Achieves full-rank capability through multiple low-rank updates
- Each iteration only needs LoRA memory
- 3 iterations of rank-64 ≈ rank-192 effective

**Implementation**: Manual script (see `relora_train.sh`)

---

### 5. LoRA+ (Asymmetric Learning Rates)
**Paper**: [LoRA+: Efficient Low Rank Adaptation](https://arxiv.org/abs/2402.12354)

**What it does**: Uses different learning rates for A and B matrices.

```
lr_B = lr_A × ratio  (ratio typically 4-16)
```

**Why it's better**:
- A matrix learns "what to look for"
- B matrix learns "how to transform"
- B needs faster learning (empirically proven)
- +2-5% accuracy, free improvement

**Axolotl support**: Via loraplus_lr_ratio
```yaml
loraplus_lr_ratio: 16
```

---

### 6. rsLoRA (Rank-Stabilized LoRA)
**Paper**: [A Rank Stabilization Scaling Factor for Fine-Tuning](https://arxiv.org/abs/2312.03732)

**What it does**: Scales LoRA by `1/√r` instead of `α/r`.

```
Standard LoRA: W = W_0 + (α/r) * BA
rsLoRA:        W = W_0 + (α/√r) * BA
```

**Why it's better**:
- Standard LoRA's scaling breaks at high ranks
- rsLoRA maintains consistent magnitude across ranks
- Enables rank-256+ without instability

**Axolotl support**: Native
```yaml
lora_use_rslora: true
```

---

### 7. ORPO (Odds Ratio Preference Optimization)
**Paper**: [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691)

**What it does**: Combines SFT and preference optimization in one step (no reward model needed).

```
L = L_SFT + β * L_OR

Where L_OR = log(odds_chosen / odds_rejected)
```

**Why it's better**:
- Single training run (vs SFT → DPO)
- No reward model needed
- Better output quality than SFT alone
- Ideal for format/style compliance

**Axolotl support**: Native
```yaml
rl: orpo
orpo_alpha: 0.1
```

---

## Combination Strategies

### Strategy A: Maximum Quality (Recommended)
**DoRA + NEFTune + LoRA+ + rsLoRA**

```yaml
adapter: lora
lora_r: 128
lora_alpha: 256
lora_use_dora: true
lora_use_rslora: true
loraplus_lr_ratio: 16
neftune_noise_alpha: 5
```

Expected: +20-30% over standard LoRA

---

### Strategy B: Full-Rank on Budget
**GaLore with 8-bit optimizer**

```yaml
# No adapter - full fine-tuning
load_in_4bit: false
optimizer: galore_adamw_8bit
galore_rank: 128
galore_update_proj_gap: 200
```

Expected: Matches full FT quality at LoRA memory cost

---

### Strategy C: Iterative Excellence
**ReLoRA × 3 iterations**

Run 3 training iterations, merging after each:
```bash
./relora_train.sh --iterations 3 --rank 64
```

Expected: rank-192 effective capacity

---

### Strategy D: Alignment Focus
**SFT with ORPO**

```yaml
rl: orpo
orpo_alpha: 0.1
# With your preference pairs dataset
```

Expected: Better instruction following, format compliance

---

## $50 Budget Optimization

### Option 1: DoRA + NEFTune on 2x H200 (~$8/hr)
- **Runtime**: ~6 hours
- **Cost**: $48
- **Result**: High-quality adapter, easy to merge/deploy

### Option 2: GaLore on 1x H200 (~$4/hr)
- **Runtime**: ~10 hours
- **Cost**: $40
- **Result**: Full-rank quality, best for math/reasoning

### Option 3: Split Budget
- 3 hours DoRA+NEFTune ($24) → base quality
- 3 hours ORPO ($24) → alignment polish
- **Total**: $48, two-stage refinement

---

## Quick Reference: Axolotl Flags

```yaml
# DoRA
lora_use_dora: true

# NEFTune
neftune_noise_alpha: 5

# rsLoRA
lora_use_rslora: true

# LoRA+
loraplus_lr_ratio: 16

# GaLore
optimizer: galore_adamw_8bit
galore_rank: 128
galore_update_proj_gap: 200
galore_scale: 0.25

# ORPO
rl: orpo
orpo_alpha: 0.1

# All together (except GaLore - incompatible with LoRA)
adapter: lora
lora_r: 128
lora_alpha: 256
lora_use_dora: true
lora_use_rslora: true
loraplus_lr_ratio: 16
neftune_noise_alpha: 5
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `config_dora_neftune.yaml` | DoRA + NEFTune + LoRA+ (recommended) |
| `config_galore.yaml` | Full-rank via GaLore |
| `config_relora.yaml` | Single ReLoRA iteration config |
| `config_orpo.yaml` | ORPO preference optimization |
| `relora_train.sh` | Multi-iteration ReLoRA script |
| `BENCHMARK.md` | Expected results vs baselines |

---

## References

1. [DoRA](https://arxiv.org/abs/2402.09353) - Weight decomposition
2. [NEFTune](https://arxiv.org/abs/2310.05914) - Noisy embeddings
3. [GaLore](https://arxiv.org/abs/2403.03507) - Gradient projection
4. [ReLoRA](https://arxiv.org/abs/2307.05695) - Iterative merging
5. [LoRA+](https://arxiv.org/abs/2402.12354) - Asymmetric LR
6. [rsLoRA](https://arxiv.org/abs/2312.03732) - Rank stabilization
7. [ORPO](https://arxiv.org/abs/2403.07691) - Preference optimization

---

*"Intelligence = argmax F[T]"*
