# ARC Prize 2025 Competitive Analysis

## Top Solutions Overview

### 1st Place: NVARC (24.0%) - $25k
**Approach:** Synthetic-data-driven ensemble + test-time-trained model + TRM components
- Paper: https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view
- Code: Kaggle notebook (Qwen3 + Unsloth + Flash LoRA)

### 2nd Place: ARChitects (16.5%) - $10k
**Approach:** 2D-aware masked-diffusion LLM + recursive self-refinement
- Paper: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/
- **Key Innovations:**
  - "Golden Gate RoPE" - 2D positional encoding for grids
  - Soft-masking recursion (continuous embedding space)
  - Token algebra: blending embeddings for hybrid inputs
  - Most-visited candidate selection during refinement
  - Per-task test-time training (128 steps, rank-32 LoRA)
- **Shape Prediction:** Separate model, 85% accuracy
- **Augmentation:** Product-of-Experts scoring across augmented candidates

### 3rd Place: MindsAI (12.6%) - $5k
**Approach:** TTFT + AIRV (Augment-Inference-Reverse-Vote)
- Paper: https://arxiv.org/abs/2506.14276
- **Key Results:** 260% boost from AIRV, 300% boost from TTFT
- **Final:** 58% on ARC private test-set

### Paper Award 1st: Tiny Recursive Models ($50k)
- ~45% on ARC-AGI-1 with small models
- Recursive architecture for iterative refinement

### Paper Award 3rd: MDL-based Neural Code Golf ($5k)
- **Directly relevant to our K(H) approach**
- No pretraining required
- MDL = Minimum Description Length

---

## Gap Analysis: What They're Missing

| Capability | ARChitects | MindsAI | NVARC | **LucidOrca** |
|------------|------------|---------|-------|---------------|
| Test-time training | ✓ | ✓ | ✓ | ✓ |
| Augmentation ensemble | ✓ | ✓ | ? | ✓ |
| **Explicit K(H) complexity filter** | ✗ | ✗ | ✗ | **✓** |
| **Explicit R(H) invariance test** | ✗ | ✗ | ✗ | **✓** |
| **Energy-based rejection** | ✗ | ✗ | ✗ | **✓** |
| **Strategic abstention** | Partial | Partial | ? | **✓** |
| 2D positional encoding | ✓ | ? | ? | TBD |
| Soft-masking refinement | ✓ | ✗ | ? | TBD |

### Key Insight
Top solutions focus on **generation quality** (better models, more augmentation, test-time training).

None explicitly verify **invariance/equivariance** post-generation.

**LucidOrca's edge:** Energy-based filtering catches brittle hacks that pass training but fail private test.

---

## Techniques to Adopt

### From ARChitects:
1. **Golden Gate RoPE** - 2D-aware positional encoding
2. **Soft-masking refinement** - Iterative improvement via continuous embeddings
3. **Product-of-Experts scoring** - Aggregate across augmentations
4. **Separate shape predictor** - Dedicated model for grid size

### From MindsAI:
1. **AIRV pipeline** - Augment → Infer → Reverse-augment → Vote
2. **Tokenizer dropout** - Regularization during test-time training

### From MDL Paper:
1. **Neural code golf** - Explicit compression objective
2. **No pretraining** - Direct optimization for simplicity

---

## LucidOrca 2026 Architecture Proposal

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATOR STAGE                          │
│  (Adopt best from ARChitects/MindsAI)                      │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 2D-RoPE LLM │ →  │ Test-Time    │ →  │ Soft-Masking  │  │
│  │ (LLaDA-8B)  │    │ Fine-Tuning  │    │ Refinement    │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                              ↓                              │
│                    100 candidates                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 CAUSAL DAMPENER (Rust)                      │
│  (Our unique contribution)                                  │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ K(H): AST   │    │ R(H): D4 +   │    │ E(H) = αK+βR  │  │
│  │ Complexity  │ +  │ Color Invari │ →  │ Filter        │  │
│  └─────────────┘    └──────────────┘    └───────────────┘  │
│                              ↓                              │
│              E(H) < threshold? Keep : Reject                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT STAGE                             │
│                                                             │
│  Survivors > 0?  →  Return argmin E(H)                     │
│  Survivors = 0?  →  ABSTAIN (strategic)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Calibration Strategy

Using ARChitects' insight that shape prediction is 85% accurate:
- Our R(H) should catch the 15% where shape is wrong
- K(H) should penalize overly complex solutions

**Threshold calibration:**
1. Run full pipeline on ARC training set
2. Compute E(H) for all correct solutions
3. Set threshold = 1.5 × median(E(correct))
4. Validate on held-out eval set

---

## Action Items

- [ ] Download NVARC paper (Google Drive PDF)
- [ ] Download MindsAI full paper from arXiv
- [ ] Clone ARChitects code from Kaggle
- [ ] Clone MindsAI TUFA code from Kaggle
- [ ] Benchmark CausalDampener on their candidate outputs
- [ ] Implement 2D-RoPE in our generator
- [ ] Test soft-masking refinement compatibility

---

## Expected Improvement

If top solutions have ~25% accuracy and we add:
- K(H) filter: Reject 30% of false positives → +3-5% accuracy
- R(H) filter: Catch invariance violations → +5-8% accuracy
- Strategic abstention: Don't guess when uncertain → Preserve score

**Conservative estimate:** 30-35% on private test
**Optimistic estimate:** 40%+ with full integration
