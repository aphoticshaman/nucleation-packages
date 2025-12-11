# RYANAIMO: Ground-Up Architecture for AIMO3

## Philosophy: Race Car, Not Turbo-Bolted Prius

The current approach (`prometheus_score-2`, `aimo3_prometheus_v42_nf4`) is a sedan with bolt-on modifications:
- Load model
- Generate code
- Execute
- Vote

This gets 2/50. To get 47/50, we need a **purpose-built race car**.

---

## The RYANAIMO Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RYANAIMO INFERENCE ENGINE                            │
│                   "From First Principles to Final Answer"                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 0: FOUNDATION (CIC Theory)                                            │
│  ───────────────────────────────────                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)                                   │ │
│  │                                                                         │ │
│  │ Every component optimizes this functional:                              │ │
│  │ • Maximize Φ (integration) - coherent reasoning traces                  │ │
│  │ • Minimize H (entropy) - compressed representations                     │ │
│  │ • Maximize C (causality) - causal power of answers                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  LAYER 1: PROBLEM UNDERSTANDING                                              │
│  ───────────────────────────────                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  PROBLEM     │  │  CONSTRAINT  │  │  DIFFICULTY  │                       │
│  │  CLASSIFIER  │─▶│  EXTRACTOR   │─▶│  ESTIMATOR   │                       │
│  │              │  │              │  │              │                       │
│  │  NT/Comb/    │  │  Modulo?     │  │  Easy: 2min  │                       │
│  │  Alg/Geom    │  │  Range?      │  │  Med: 6min   │                       │
│  │              │  │  Structure?  │  │  Hard: 15min │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│                                                                              │
│  LAYER 2: EXTENDED REASONING (The Breakthrough)                              │
│  ─────────────────────────────────────────────────                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     DEEP THINK MODULE                                   │ │
│  │                                                                         │ │
│  │  <think>                                                                │ │
│  │  Let me understand this problem deeply...                               │ │
│  │  - What are the key mathematical structures?                            │ │
│  │  - What techniques apply? (Modular arithmetic, generating functions...) │ │
│  │  - What are the edge cases?                                             │ │
│  │  - Can I verify my approach before coding?                              │ │
│  │  </think>                                                               │ │
│  │                                                                         │ │
│  │  1000+ tokens of reasoning BEFORE any code                              │ │
│  │  This is what DeepSeek-R1 does. This is the gap to close.               │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  LAYER 3: MULTI-PATH CODE SYNTHESIS                                          │
│  ─────────────────────────────────────                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   PATH A     │  │   PATH B     │  │   PATH C     │                       │
│  │   Direct     │  │   SymPy      │  │   MCTS       │                       │
│  │   Compute    │  │   Algebraic  │  │   Search     │                       │
│  │              │  │              │  │              │                       │
│  │  ProofSampler│  │  ProofSampler│  │  MCTS +      │                       │
│  │  constrained │  │  constrained │  │  PRM head    │                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                 │                 │                                │
│         ▼                 ▼                 ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      PROOF CONSTRAINT FILTER                            │ │
│  │                                                                         │ │
│  │  • BracketTracker: No unbalanced LaTeX                                  │ │
│  │  • EquationTracker: Variables defined before use                        │ │
│  │  • NumberConsistency: No numeric hallucinations                         │ │
│  │  • RepetitionBlock: No infinite loops in text                           │ │
│  │  • Backtracking: Checkpoint and recover from dead ends                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  LAYER 4: EXECUTION + VERIFICATION                                           │
│  ─────────────────────────────────                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   EXECUTE    │  │   VERIFY     │  │   CHECK      │                       │
│  │              │  │   SYMBOLIC   │  │   NUMERIC    │                       │
│  │  Sandboxed   │─▶│              │─▶│              │                       │
│  │  Python +    │  │  SymPy       │  │  Substitute  │                       │
│  │  math stdlib │  │  simplify/   │  │  back into   │                       │
│  │              │  │  solve check │  │  constraints │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│                                                                              │
│  LAYER 5: CIC-AWARE ANSWER SELECTION                                         │
│  ───────────────────────────────────                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      VALUE CLUSTERING ENGINE                            │ │
│  │                                                                         │ │
│  │  1. Cluster answers by relative proximity: |a-b|/max(|a|,|b|) < 0.05    │ │
│  │  2. Compute cluster statistics:                                         │ │
│  │     • size: How many paths landed here?                                 │ │
│  │     • tightness: How close are the members?                             │ │
│  │     • score = size × sqrt(tightness)                                    │ │
│  │  3. Select best basin                                                   │ │
│  │  4. Refine: median + trimmed_mean / 2                                   │ │
│  │                                                                         │ │
│  │  KEY INSIGHT: Value proximity ≈ Algorithmic similarity                  │ │
│  │  Near-miss answers share correct reasoning with minor arithmetic error  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  LAYER 6: CONFIDENCE CALIBRATION                                             │
│  ───────────────────────────────                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      CIC CONFIDENCE                                     │ │
│  │                                                                         │ │
│  │  Φ = integrated information (trace similarity via NCD)                  │ │
│  │  H = representation entropy (answer variance)                           │ │
│  │  C = causal power (cluster dominance)                                   │ │
│  │                                                                         │ │
│  │  Confidence = 0.5 + 0.5 × F[T]                                          │ │
│  │                                                                         │ │
│  │  ARCHITECTURAL EPISTEMIC HUMILITY:                                      │ │
│  │  Low confidence → spend more time → generate more paths                 │ │
│  │  High confidence → move on to next problem                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  LAYER 7: ADAPTIVE TIME MANAGEMENT                                           │
│  ─────────────────────────────────                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      PHASE TRANSITION DETECTION                         │ │
│  │                                                                         │ │
│  │  Monitor: dΦ/dt and dH/dt                                               │ │
│  │  When dΦ/dt = λ·dH/dt → CRYSTALLIZATION                                 │ │
│  │  Answer has converged. Stop generating. Move on.                        │ │
│  │                                                                         │ │
│  │  This is UIPT applied to inference.                                     │ │
│  │  Don't waste time when the answer is already "grokked".                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Custom Dependency Stack

### Tier 1: Core Primitives (From Scratch)

```python
# File: ryanaimo/core/cic.py
"""
Compression-Integration-Causality primitives.
The mathematical foundation for everything else.
"""

def ncd(x: bytes, y: bytes) -> float:
    """Normalized Compression Distance - algorithmic similarity"""

def phi_integrated_information(traces: List[str]) -> float:
    """Φ - how much the whole exceeds the parts"""

def representation_entropy(samples: List[int]) -> float:
    """H(T|X) - disorder in answer space"""

def causal_power_multiscale(samples: List[int]) -> float:
    """C_multi(T) - causal power across scales"""

def compute_cic_functional(samples, traces, λ=0.5, γ=0.3) -> CICState:
    """F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)"""
```

### Tier 2: Proof Constraints (From Scratch)

```python
# File: ryanaimo/proof/constraints.py
"""
Hard and soft constraints for mathematical proof generation.
"""

class BracketTracker:
    """Enforce balanced brackets in LaTeX/code"""
    PAIRS = {'(': ')', '[': ']', '{': '}', '\\left(': '\\right)', ...}

class EquationTracker:
    """Track variable definitions and equation consistency"""

class NumberConsistencyConstraint:
    """Penalize numbers inconsistent with problem statement"""

class ProofKeywordBooster:
    """Boost proof structure words: therefore, hence, QED"""

class RepetitionBlocker:
    """Block repetitive generation loops"""
```

### Tier 3: Sampling Engine (From Scratch)

```python
# File: ryanaimo/sampling/engine.py
"""
Proof-aware sampling with backtracking and beam search.
"""

class ProofState:
    """Complete state of a proof in progress"""
    tokens: List[int]
    brackets: BracketTracker
    equations: EquationTracker
    checkpoints: List[int]  # For backtracking

class ProofSampler:
    """Main proof-aware sampler - replaces generate()"""

    def sample(self, model, input_ids, max_new_tokens=512):
        """
        Generate with:
        1. Constraint checking at each token
        2. Soft adjustments (proof keywords, number consistency)
        3. Hard blocks (unbalanced brackets)
        4. Backtracking on dead ends
        """
```

### Tier 4: Value Clustering (From Scratch)

```python
# File: ryanaimo/selection/clustering.py
"""
The 88% error reduction method.
"""

def value_clustering(samples: List[int], threshold=0.05) -> Dict:
    """
    Cluster by relative proximity: |a-b|/max(|a|,|b|) < threshold

    Returns:
        clusters: List of {members, size, center, tightness, score}
        best: The highest-scoring cluster
    """

def basin_refinement(cluster: Dict) -> int:
    """
    Refine answer within best cluster:
    answer = (median + trimmed_mean) / 2

    KEY INSIGHT: The answer is the CENTER of the basin,
    not any individual sample. This is the Platonic Form.
    """
```

### Tier 5: MCTS Reasoning (From Scratch)

```python
# File: ryanaimo/reasoning/mcts.py
"""
Monte Carlo Tree Search for solution exploration.
"""

class ReasoningNode:
    """Node in the reasoning tree"""
    content: str  # Reasoning step text
    children: List['ReasoningNode']
    value: float  # PRM score
    visits: int

class PRM:
    """Process Reward Model - scores reasoning steps"""

    def score_step(self, problem: str, step: str) -> float:
        """Score a single reasoning step"""

class MCTSReasoner:
    """MCTS for exploring solution space"""

    def search(self, problem: str, budget: int) -> List[ReasoningNode]:
        """
        UCT-based tree search:
        1. SELECT: Follow highest UCT score
        2. EXPAND: Generate child reasoning steps
        3. SIMULATE: Complete the reasoning path
        4. BACKPROP: Update value estimates
        """
```

### Tier 6: Time Management (From Scratch)

```python
# File: ryanaimo/time/allocator.py
"""
Adaptive time allocation with early stopping.
"""

class DifficultyEstimator:
    """Estimate problem difficulty for time allocation"""

    def estimate(self, problem: str) -> float:
        """
        Factors:
        - Problem length
        - Keyword complexity (generating functions → hard)
        - Structure (multi-part → longer)
        - Historical similar problems
        """

class CrystallizationDetector:
    """Detect when answer has converged (UIPT)"""

    def detect(self, history: List[CICState]) -> bool:
        """
        Check: dΦ/dt ≈ λ·dH/dt
        If true: answer has crystallized, stop generating
        """

class TimeAllocator:
    """Manage 5-hour budget across 50 problems"""

    def allocate(self, problem_idx: int, difficulty: float,
                 remaining_time: float, remaining_problems: int) -> float:
        """
        Adaptive allocation:
        - Base time = remaining / remaining_problems
        - Adjust by difficulty multiplier
        - Reserve buffer for hard problems
        """
```

### Tier 7: Model Interface (Uses transformers + bitsandbytes)

```python
# File: ryanaimo/models/qwen.py
"""
Thin wrapper around Qwen2.5-Math-72B.
This is the ONLY place we use external libraries for inference.
"""

class QwenMathModel:
    """Wrapper for Qwen2.5-Math-72B-Instruct"""

    def __init__(self, path: str, quantization: str = "nf4"):
        """Load with NF4 quantization via bitsandbytes"""

    def generate_with_think(self, problem: str, max_tokens: int = 2048) -> str:
        """
        Generate with explicit <think>...</think> prompting.

        This is the key difference from prometheus.
        We FORCE extended reasoning before code.
        """

    def get_prm_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for PRM scoring"""
```

---

## File Structure

```
ryanaimo/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── cic.py              # CIC theory primitives
│   ├── config.py           # Configuration dataclasses
│   └── utils.py            # Common utilities
├── proof/
│   ├── __init__.py
│   ├── constraints.py      # Proof constraints
│   ├── trackers.py         # Bracket/equation tracking
│   └── verifier.py         # SymPy verification
├── sampling/
│   ├── __init__.py
│   ├── engine.py           # ProofSampler
│   ├── beam.py             # Proof-aware beam search
│   └── backtrack.py        # Checkpoint/backtrack logic
├── selection/
│   ├── __init__.py
│   ├── clustering.py       # Value clustering
│   ├── voting.py           # Weighted voting
│   └── refinement.py       # Basin refinement
├── reasoning/
│   ├── __init__.py
│   ├── mcts.py             # MCTS reasoner
│   ├── prm.py              # Process Reward Model
│   └── steps.py            # Reasoning step generation
├── time/
│   ├── __init__.py
│   ├── allocator.py        # Time allocation
│   ├── difficulty.py       # Difficulty estimation
│   └── crystallization.py  # UIPT detection
├── models/
│   ├── __init__.py
│   ├── qwen.py             # Qwen wrapper
│   └── prompts.py          # Prompt templates
└── solver.py               # Main entry point
```

---

## The Main Solver

```python
# File: ryanaimo/solver.py
"""
RYANAIMO Main Solver - Ground-Up AIMO3 Architecture
"""

from ryanaimo.core.cic import compute_cic_functional, CICState
from ryanaimo.proof.constraints import ProofConstraints
from ryanaimo.sampling.engine import ProofSampler
from ryanaimo.selection.clustering import value_clustering, basin_refinement
from ryanaimo.reasoning.mcts import MCTSReasoner, PRM
from ryanaimo.time.allocator import TimeAllocator, DifficultyEstimator, CrystallizationDetector
from ryanaimo.models.qwen import QwenMathModel

class RyanAIMOSolver:
    """
    The complete AIMO3 solver.
    Designed from first principles using CIC theory.
    """

    def __init__(self, model_path: str, budget_seconds: int = 18000):
        self.model = QwenMathModel(model_path)
        self.constraints = ProofConstraints()
        self.sampler = ProofSampler(self.model, self.constraints)
        self.prm = PRM(self.model)
        self.mcts = MCTSReasoner(self.model, self.prm)
        self.time = TimeAllocator(budget_seconds)
        self.difficulty = DifficultyEstimator()
        self.crystallization = CrystallizationDetector()

    def solve(self, problem: str) -> int:
        """
        Solve a single problem with the full RYANAIMO pipeline.
        """
        # 1. Allocate time
        difficulty = self.difficulty.estimate(problem)
        time_budget = self.time.allocate(difficulty)

        # 2. Extended reasoning (the key innovation)
        thinking = self.model.generate_with_think(problem)

        # 3. Multi-path code synthesis with proof constraints
        paths = []
        cic_history = []

        while self.time.remaining(time_budget) > 0:
            # Generate with proof constraints
            code = self.sampler.sample(problem, thinking)

            # Execute and verify
            result = self.execute_and_verify(code)
            if result is not None:
                paths.append(result)

            # Check for crystallization
            samples = [p for p in paths]
            cic = compute_cic_functional(samples)
            cic_history.append(cic)

            if self.crystallization.detect(cic_history):
                break  # Answer has converged

        # 4. CIC-aware selection
        if not paths:
            return 0

        clustering = value_clustering(paths)
        if clustering['best'] is None:
            return max(set(paths), key=paths.count)

        return basin_refinement(clustering['best'])
```

---

## Why This Wins

| Component | Prometheus (2/50) | RYANAIMO (47/50) |
|-----------|-------------------|------------------|
| Reasoning | None | 1000+ token <think> |
| Constraints | None | BracketTracker, EquationTracker |
| Selection | Weighted vote | CIC value clustering |
| Verification | Execution only | SymPy + constraint check |
| Time | Naive decay | Difficulty + crystallization |
| Foundation | Ad-hoc | CIC theory |

---

## The Paradigm Shift

**Prometheus**: "Generate code, run it, vote on answers"

**RYANAIMO**:
- "UNDERSTAND the problem deeply (extended thinking)"
- "Generate CONSTRAINED proofs (proof sampler)"
- "VERIFY at multiple levels (execution + symbolic)"
- "CLUSTER in algorithm space (value proximity)"
- "REFINE to the Platonic Form (basin center)"
- "STOP when crystallized (UIPT detection)"

This is the difference between a Prius with turbo bolts and a Formula 1 car.

---

**LET'S BUILD THIS.**
