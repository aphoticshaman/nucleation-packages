# Repository Analysis: AIMO3 Competition & Prometheus System

## Executive Summary

This repository contains a comprehensive system for solving the **AIMO3 (AI Mathematical Olympiad)** competition, targeting a $1.59M+ prize for achieving 47/50 correct answers. The codebase includes:

1. **Prometheus Solver** - Initial attempt that scored 2/50
2. **RYANAIMO Architecture** - Ground-up redesign based on CIC theory
3. **Custom Infrastructure** - RYANSTREAM inference engine with specialized components
4. **Theoretical Framework** - CIC (Compression-Integration-Causality) theory

---

## Part 1: The Prometheus System

### What is Prometheus?

**Prometheus** refers to two distinct concepts in this repository:

#### 1. P.R.O.M.E.T.H.E.U.S. Protocol (Cognitive Framework)
- **Full Name**: Protocol for Recursive Optimization, Meta-Enhanced Theoretical Heuristic Extraction, and Universal Synthesis
- **Purpose**: A system prompt/framework for generating novel knowledge by extracting "Unknown Knowns" from model latent spaces
- **Location**: `skills/PROMETHEUS.md`
- **Method**: 5-stage cognitive pipeline:
  1. Latent Space Archaeology
  2. Novel Synthesis Method
  3. Rigorous Theoretical Validation
  4. XYZA & SDPM Operationalization
  5. Output Generation

#### 2. Prometheus Solver (AIMO3 Competition System)
- **Purpose**: Mathematical problem solver for AIMO3 competition
- **Performance**: Scored **2/50** on public leaderboard
- **Location**: 
  - `prometheus_score-2.ipynb` (original submission)
  - `burner/research/prometheus/prometheus_v6_full.py` (full implementation)
  - `burner/research/prometheus/prometheus_kaggle.py` (Kaggle-optimized version)

### Prometheus Solver Architecture

The Prometheus solver uses a sophisticated multi-component system:

```
┌─────────────────────────────────────────────────────────┐
│              PROMETHEUS SOLVER ENGINE                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. UIPTEntropyWindow                                    │
│     - Rolling entropy tracker (window=32)                │
│     - Detects phase transitions:                         │
│       • Low entropy = Crystallized Logic (high conf)    │
│       • High entropy = Gas Phase (hallucination)       │
│                                                          │
│  2. DeltaKScheduler                                      │
│     - Measures: logp(preferred) - logp(sampled)        │
│     - High delta = exploring away from preference      │
│     - Low delta = sampling near peak probability        │
│                                                          │
│  3. MDLMetaScheduler                                     │
│     - Maps entropy + delta → sampling parameters        │
│     - Adjusts: temperature, top_p, top_k               │
│     - Kill decision: high entropy + high delta          │
│     - Crystal boost: low entropy + negative derivative   │
│                                                          │
│  4. Toroidal Clustering                                 │
│     - S¹ (circle) clustering for mod-1000 answers         │
│     - Handles wrap-around for modular arithmetic        │
│                                                          │
│  5. Entropic Gravity Voting                             │
│     - Score = Mass × Density^0.15 × Solomonoff_Weight   │
│     - Mass = cluster size                                │
│     - Density = 1 / (variance + epsilon)                │
│     - Solomonoff = 0.9995^code_length (Occam's razor)   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Why Prometheus Scored Only 2/50

From `PROMETHEUS_ANALYSIS.md`, the critical weaknesses:

1. **Model Choice: GPT-OSS-20B is Underpowered**
   - IMO-level problems require frontier reasoning
   - Need: Qwen-2.5-Math-72B or DeepSeek-R1-distill

2. **No True Reasoning Chain**
   - PoT (Program of Thought) jumps straight to code
   - Missing: Extended thinking BEFORE coding
   - DeepSeek-R1 style `<think>` blocks are essential

3. **Verification is Weak**
   - Only checks if code runs
   - No mathematical verification
   - No answer substitution back into constraints

4. **Single-Pass Generation**
   - Generates code, runs it, done
   - Missing: Iterative refinement
   - Missing: Multiple independent paths that must agree

5. **No Symbolic Grounding**
   - Code uses sympy but prompts don't enforce it
   - Missing: Formal verification (Lean4)
   - Missing: Algebraic manipulation verification

6. **Time Budget Misallocation**
   - 5-hour GPU limit / 50 problems = 6 min/problem
   - Hard problems need 15+ minutes
   - Current "exponential decay" is naive

---

## Part 2: The RYANAIMO Architecture

### Philosophy: Race Car, Not Turbo-Bolted Prius

RYANAIMO is a **ground-up redesign** based on CIC (Compression-Integration-Causality) theory, not a modification of Prometheus.

### Core Foundation: CIC Theory

The unified functional:
```
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Where:
  Φ(T) = Integrated Information (whole > parts)
  H(T|X) = Representation Entropy (disorder)
  C_multi(T) = Multi-scale Causal Power

Intelligence = argmax F[T]
```

### RYANAIMO Architecture Layers

```
LAYER 0: FOUNDATION (CIC Theory)
  - Every component optimizes F[T]

LAYER 1: PROBLEM UNDERSTANDING
  - Problem Classifier (NT/Comb/Alg/Geom)
  - Constraint Extractor
  - Difficulty Estimator

LAYER 2: EXTENDED REASONING (The Breakthrough)
  - <think> blocks
  - 1000+ tokens of thinking BEFORE code
  - DeepSeek-R1 style reasoning

LAYER 3: MULTI-PATH CODE SYNTHESIS
  - Path A: Direct Compute
  - Path B: SymPy Algebraic
  - Path C: MCTS Search
  - All with ProofSampler constraints

LAYER 4: EXECUTION + VERIFICATION
  - Sandboxed Python execution
  - SymPy symbolic verification
  - Numeric constraint checking

LAYER 5: CIC-AWARE ANSWER SELECTION
  - Value clustering (88% error reduction)
  - Basin refinement
  - Cluster by relative proximity

LAYER 6: CONFIDENCE CALIBRATION
  - CIC confidence = 0.5 + 0.5 × F[T]
  - Epistemic humility by construction

LAYER 7: ADAPTIVE TIME MANAGEMENT
  - Phase transition detection
  - Crystallization: dΦ/dt = λ·dH/dt
  - Early stopping when answer converges
```

### Key Innovations

1. **Extended Reasoning** - Forces 1000+ token thinking blocks before code
2. **ProofSampler** - Constraint-aware generation (brackets, equations, consistency)
3. **Value Clustering** - 88% error reduction via algorithmic similarity detection
4. **Multi-Path Verification** - 3+ independent solutions must agree
5. **Crystallization Detection** - UIPT-based early stopping

---

## Part 3: Custom Infrastructure (RYANSTREAM)

### The RYANSTREAM Stack (~8,500 lines)

Located in `llm-qwen-deps-aimo3/`:

1. **ProofSampler** (`sampler.py` - 1032 lines)
   - BracketTracker: Hard constraint on balanced brackets/LaTeX
   - EquationTracker: Variable/equation consistency
   - ProofBeamSearch: Constraint-aware beam search
   - Backtracking: Checkpoint-based recovery

2. **SpeculativeEngine** (`speculative.py` - 659 lines)
   - DraftModelSpeculator: Small model drafts, 72B verifies
   - SelfSpeculator: Early-exit drafting
   - TreeAttention: Parallel hypothesis exploration
   - Target: 2-5x speedup

3. **RyanStream Scheduler** (`scheduler.py` - 978 lines)
   - LookaheadPredictor: 10-token lookahead, EOS tracking
   - KVCacheManager: 3s timeout eviction, top-10 protection
   - AutoPrecisionManager: BF16 ↔ NF4 based on VRAM pressure
   - Target: 45% fewer stalls

4. **Ryan-Pipeline** (`pipeline.py` - 629 lines)
   - GPUBufferPool: Pre-allocated double-buffered tensors
   - GPUShuffle: GPU-side random permutation
   - MMapMathDataset: Memory-mapped dataset files
   - Target: 60% faster loading

5. **Ryan-Bridge** (`bridge.py` - 618 lines)
   - Training → Inference conversion
   - CheckpointConverter
   - DistillationPipeline: 72B → Draft model

---

## Part 4: Competition Constraints

From `AIMO3 RULES and datasets and models.txt`:

| Constraint | Value | Implication |
|------------|-------|-------------|
| GPU Runtime | 5 hours | ~6 min/problem average |
| Hardware | H100 80GB (2x) | Can fit 72B model with NF4 |
| Internet | Disabled at runtime | All models must be cached |
| Model Cutoff | Before Mar 15, 2026 | Qwen-2.5-Math-72B OK |
| Answer Range | 0-99999 integers | Use for sanity checking |
| Scoring | 1/0.5/0 per problem | Consistency matters |
| Test Set | 110 problems (50 public, 60 private) | Must score 47/50 on BOTH |

---

## Part 5: Implementation Status

### Current State

✅ **Completed:**
- Prometheus solver (scored 2/50)
- RYANAIMO architecture design
- CIC theory implementation
- RYANSTREAM infrastructure components
- ProofSampler with constraints
- Value clustering algorithm

🔄 **In Progress:**
- Integration of extended reasoning
- Multi-path verification
- Time budget optimization

❌ **Missing:**
- Qwen-72B + DeepSeek-R1 model integration
- Full RYANAIMO pipeline end-to-end
- Kaggle submission with new architecture

---

## Part 6: Key Files Reference

### Analysis Documents
- `PROMETHEUS_ANALYSIS.md` - Why Prometheus scored 2/50
- `AIMO3_WINNING_STRATEGY.md` - Path to 47/50
- `RYANAIMO_ARCHITECTURE.md` - Ground-up architecture design

### Implementation
- `prometheus_score-2.ipynb` - Original submission (2/50)
- `burner/research/prometheus/prometheus_v6_full.py` - Full Prometheus engine
- `burner/research/prometheus/prometheus_kaggle.py` - Kaggle-optimized version
- `ryanaimo/solver.py` - RYANAIMO main solver
- `llm-qwen-deps-aimo3/` - RYANSTREAM infrastructure

### Theory
- `cic_theory_validation.py` - CIC theory implementation
- `skills/PROMETHEUS.md` - P.R.O.M.E.T.H.E.U.S. Protocol
- `skills/final_nobel_synthesis.py` - Unified framework synthesis

---

## Part 7: The Path Forward

### Target: 47/50 = Overall Progress Prize ($1.59M+)

**Critical Success Factors:**

1. **Model Quality** (Most Important)
   - Current: GPT-OSS-20B → 2/50
   - Target: Qwen-72B + DeepSeek-R1 → 47/50

2. **Extended Reasoning**
   - Current: PoT jumps to code
   - Target: 1000+ token `<think>` blocks

3. **Multi-Path Verification**
   - Current: Single-pass generation
   - Target: 3+ independent paths, agreement required

4. **CIC-Aware Selection**
   - Current: Basin clustering (good start)
   - Target: Full CIC integration with value clustering

5. **Time Management**
   - Current: Naive exponential decay
   - Target: Difficulty-aware allocation with early stopping

---

## Conclusion

This repository represents a **complete research-to-production pipeline** for solving IMO-level mathematical problems:

- **Theory**: CIC framework provides mathematical foundation
- **Infrastructure**: RYANSTREAM provides custom inference engine
- **Architecture**: RYANAIMO provides ground-up solver design
- **Analysis**: Prometheus failure analysis guides improvements

The gap from 2/50 → 47/50 is **achievable** with:
1. Better models (Qwen-72B vs GPT-OSS-20B)
2. Extended reasoning (DeepSeek-R1 style thinking)
3. Formal verification (catch errors before submission)
4. Ensemble diversity (multiple independent paths)

**The infrastructure for winning AIMO3 is HERE. The missing piece was always the MODEL.**

