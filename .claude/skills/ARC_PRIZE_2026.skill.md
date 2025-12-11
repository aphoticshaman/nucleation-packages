# SKILL: ARC Prize 2026 Competition Preparation

## Purpose
Systematically prepare for ARC Prize 2026 by synthesizing winning approaches from 2025, identifying gaps we can exploit, and building our unique edge: energy-based hypothesis filtering via CausalDampener.

---

## Core Techniques from 2025 Winners

### 1. CompressARC (No Pretraining Required)
**Source:** `agi-compressarc.ipynb`, `arc-agi-without-pretraining.ipynb`

**Key Insight:** MDL (Minimum Description Length) as core objective - find the simplest program that transforms input→output.

**Implementation:**
```python
# Architecture: Multitensor systems with compression objective
import arc_compressor
import multitensor_systems
import train

# Training loop:
# 1. Initialize from random (NO pretraining)
# 2. Optimize for compression ratio
# 3. Take ~2300 steps per puzzle within 12-hour budget
```

**GPU Parallelization Strategy:**
1. Profile memory usage per task (2 steps)
2. Sort tasks by memory descending
3. Pack tasks greedily onto GPUs by memory quota
4. Run as many steps as time allows

**Critical Parameters:**
- `torch.float32` default dtype
- `cudnn.benchmark = True`
- `allow_tf32 = True`
- Memory buffer: 6GB per GPU
- Target: ~2300 steps per puzzle

### 2. Qwen-3 Transformer Approach
**Source:** `solving-arc-prize-2025-with-qwen-3-transformer.ipynb`

**Key Insight:** Few-shot prompting with 0.6B parameter model, CoT (Chain of Thought) via `/think` mode.

**Implementation:**
```python
# Prompt structure:
prompt = "Solve the following abstract reasoning challenge:\n\n"
for i, example in enumerate(task['train']):
    prompt += f"Input {i+1}:\n{example['input']}\nOutput {i+1}:\n{example['output']}\n\n"
prompt += f"Now predict:\n{test_input}\nOutput:\n"

# Parse output:
# 1. Extract [[...]] grid from response
# 2. Validate grid structure (all rows same length, all ints)
# 3. Fallback to [[0]] if parsing fails
```

**Model Settings:**
- `max_input_tokens = 3000`
- `max_new_tokens = 64`
- `do_sample = False` (greedy decoding)
- Truncate history if over token limit

### 3. Visualization-First Understanding
**Source:** `arc-agi-2025-visualization-all-1000-120-tasks.ipynb`

**Key Insight:** Visual inspection reveals pattern categories:
- Color mapping transformations
- Geometric transforms (rotation, reflection, scaling)
- Object detection and counting
- Filling/flooding operations
- Pattern completion

**ARC Color Palette:**
```python
# 0:black, 1:blue, 2:red, 3:green, 4:yellow
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
cmap = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
])
```

---

## Gap Analysis: What Winners Miss

| Capability | 2025 Winners | LucidOrca 2026 |
|------------|--------------|----------------|
| Test-time training | Yes | Yes |
| Augmentation ensemble | Some | Yes |
| **Explicit K(H) filter** | No | **Yes** |
| **Explicit R(H) invariance test** | No | **Yes** |
| **Energy-based rejection** | No | **Yes** |
| **Strategic abstention** | Partial | **Full** |
| 2D positional encoding | ARChitects only | Adopt |
| Soft-masking refinement | ARChitects only | Adopt |

---

## LucidOrca Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GENERATOR STAGE                          │
│  (Adopt best from ARChitects/MindsAI/CompressARC)          │
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
│  (Our unique edge)                                          │
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

## CausalDampener Energy Formulas

### K(H) - Description Length
```rust
// packages/causal-dampener/src/complexity.rs
K(H) = tokens × (1 + 0.1 × ln(1 + depth))
```
- `tokens`: AST node count
- `depth`: Maximum nesting depth
- Penalizes both size AND complexity

### R(H) - Invariance Penalty
```rust
// packages/causal-dampener/src/invariance.rs
R(H) = Σ |H(τ·g) - τ·H(g)| for τ ∈ D4 × ColorPerms
```
- **D4 transforms**: Identity, Rotate90/180/270, FlipH/V, FlipDiag/AntiDiag
- **Color permutations**: Swap any pair of colors (0-9)
- Low R(H) = hypothesis respects structure

### E(H) - Total Energy
```
E(H) = α·K(H) + β·R(H)
```
- **Default**: α=1.0, β=2.0 (invariance matters more)
- **Threshold**: 1.5 × median(E(correct)) on training set

---

## Techniques to Adopt from 2025

### From ARChitects (2nd Place)
1. **Golden Gate RoPE**: 2D positional encoding for grids
2. **Soft-masking refinement**: Iterative improvement in embedding space
3. **Product-of-Experts scoring**: Aggregate across augmentations
4. **Separate shape predictor**: 85% accuracy on output dimensions

### From MindsAI (3rd Place)
1. **AIRV pipeline**: Augment → Infer → Reverse-augment → Vote
2. **TTFT (Test-Time Fine-Tuning)**: Per-task adaptation
3. **Tokenizer dropout**: Regularization during TTT

### From MDL Paper
1. **Neural code golf**: Explicit compression objective
2. **No pretraining**: Direct optimization for simplicity

---

## Calibration Protocol

### Step 1: Baseline Candidates
Run generator on ARC training set (400 tasks), collect top-100 candidates per task.

### Step 2: Compute E(H) Distribution
```python
energies = []
for task in training_set:
    for candidate in candidates[task]:
        e = alpha * K(candidate) + beta * R(candidate)
        if candidate == ground_truth:
            energies.append(("correct", e))
        else:
            energies.append(("wrong", e))
```

### Step 3: Find Optimal Threshold
```python
correct_energies = [e for label, e in energies if label == "correct"]
threshold = 1.5 * np.median(correct_energies)
```

### Step 4: Validate on Eval Set
- Compute precision/recall at threshold
- Adjust α, β if needed (grid search)
- Target: 95%+ recall on correct answers, 30%+ rejection of wrong

---

## Expected Improvement

| Source | Contribution |
|--------|--------------|
| Best 2025 generator | ~25% baseline |
| K(H) filter | +3-5% (reject overcomplicated) |
| R(H) filter | +5-8% (catch invariance violations) |
| Strategic abstention | Preserve score (don't lose points) |
| **Total** | **30-40% on private test** |

---

## Implementation Checklist

### Phase 1: Build CausalDampener (Done)
- [x] Grid data structure (`grid.rs`)
- [x] D4 transforms (`transforms.rs`)
- [x] K(H) complexity computation (`complexity.rs`)
- [x] R(H) invariance testing (`invariance.rs`)
- [x] PyO3 Python bindings (`lib.rs`)

### Phase 2: Integrate Generator
- [ ] Adopt ARChitects 2D-RoPE code
- [ ] Implement AIRV pipeline
- [ ] Add TTFT capability

### Phase 3: Calibrate
- [ ] Run on full ARC training set
- [ ] Compute energy distributions
- [ ] Find optimal α, β, threshold
- [ ] Validate on eval set

### Phase 4: Competition Prep
- [ ] Package for Kaggle submission
- [ ] Optimize memory/time budget
- [ ] Add strategic abstention logic
- [ ] Run final validation

---

## Anti-Patterns to Avoid

1. **Overfitting to training set**: Use held-out eval for all calibration
2. **Ignoring abstention**: Guessing randomly loses points
3. **Complex generators without filtering**: More candidates ≠ better
4. **Single-shot inference**: Always use ensembles
5. **Ignoring symmetry**: ARC tasks often have D4 structure

---

## Resources

- ARC Prize 2025 Results: https://arcprize.org/competitions/2025/
- ARChitects Paper: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/
- MindsAI Paper: https://arxiv.org/abs/2506.14276
- CompressARC: Kaggle notebook (boristown/agi-compressarc)
- CausalDampener: `packages/causal-dampener/`

---

*Skill Version 1.0 | Target: ARC Prize 2026*
