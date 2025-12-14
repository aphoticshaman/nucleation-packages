# PROMETHEUS + NSM + XYZA: Unified Methodology for AIMO3 Mathematical Reasoning

**Version:** 3.0
**Classification:** LLM-Executable Specification
**Target:** Any LLM System (Claude, GPT, Gemini, gpt-oss-120b, DeepSeek, Qwen)
**Domain:** AIMO3 Mathematical Olympiad Competition
**Date:** December 2024

---

## Abstract

This document provides a complete, self-contained specification for building a state-of-the-art mathematical reasoning system for AIMO3 (AI Mathematical Olympiad). It synthesizes three interconnected methodologies:

1. **PROMETHEUS** — Physics-inspired answer selection and validation
2. **NSM (Novel Synthesis Method)** — 20 novel insights for intelligence amplification
3. **XYZA** — Systematic development pipeline from design to production

Combined with **Guardian** output validation, this framework enables any LLM to command mathematical reasoning with high accuracy.

**Key Innovation:** The system treats mathematical reasoning as a **phase transition problem** where correct answers emerge as "crystallized" low-entropy states in a high-dimensional solution space.

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [The Master Equation](#2-the-master-equation)
3. [PROMETHEUS Engine](#3-prometheus-engine)
4. [NSM: 20 Novel Insights](#4-nsm-20-novel-insights)
5. [XYZA Pipeline](#5-xyza-pipeline)
6. [Guardian Integration](#6-guardian-integration)
7. [AIMO3 Implementation](#7-aimo3-implementation)
8. [System Prompts](#8-system-prompts)
9. [Complete Python Implementation](#9-complete-python-implementation)
10. [Operational Checklist](#10-operational-checklist)

---

## 1. Core Philosophy

### 1.1 The Problem

AIMO-style problems require:
- **Precision**: Answers must be exact integers in [0, 999]
- **Reliability**: LLMs hallucinate ~15-30% of mathematical reasoning
- **Speed**: Competition time limits require efficient inference
- **Verification**: No ground truth during competition

### 1.2 The Solution

Treat the LLM as a **stochastic answer generator** and wrap it with:
1. **Multi-sample inference** (generate N solutions)
2. **Gravitational clustering** (find consensus basins)
3. **Entropy-based quality filtering** (reject high-entropy "gas phase" reasoning)
4. **Guardian validation** (type-check, range-check, repair)

### 1.3 Core Insight

> **"Correct mathematical reasoning 'crystallizes' — entropy drops as logic converges. Incorrect reasoning remains 'gaseous' — high entropy, diffuse, inconsistent."**

This is the **Universal Information Phase Transition (UIPT)** principle.

---

## 2. The Master Equation

### 2.1 The CIC Functional

Intelligence (and correct mathematical reasoning) is the optimization of:

```
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Intelligence = argmax F[T]
```

Where:
- **Φ(T)** = Integrated information (does the whole exceed sum of parts?)
- **H(T|X)** = Representational entropy (noise to compress)
- **C_multi(T)** = Multi-scale causal power (does reasoning cause correct answers?)

### 2.2 For Mathematical Reasoning

```
Score(answer) = Mass(basin) × Density(basin)^0.1 × Solomonoff(code)

Where:
- Mass = number of solutions agreeing on this answer
- Density = 1 / variance of agreeing solutions
- Solomonoff = 0.999^len(code)  // Shorter proofs preferred
```

### 2.3 Answer Selection

```python
best_answer = argmax_{a ∈ Answers} [
    count(solutions_with_answer_a) *           # Mass
    (1 / variance(solutions_with_answer_a))^0.1 *  # Density
    0.999^avg_code_length(solutions_with_answer_a)  # Solomonoff
]
```

---

## 3. PROMETHEUS Engine

### 3.1 Overview

PROMETHEUS is a physics-inspired answer selection system with four core components:

| Component | Physics Analogy | Function |
|-----------|-----------------|----------|
| **UIPT** | Phase transitions | Detect reasoning crystallization via entropy |
| **Gravitational Basins** | Gravity wells | Cluster answers by proximity |
| **NCD** | Compression | Measure solution diversity |
| **Solomonoff** | Occam's Razor | Prefer shorter proofs |

### 3.2 UIPT (Universal Information Phase Transition)

**Principle:** Reasoning quality correlates with entropy dynamics.

```python
def calculate_entropy(text: str) -> float:
    """Shannon entropy of character distribution"""
    if not text:
        return 100.0  # Maximum uncertainty

    counts = Counter(text)
    total = len(text)

    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

def detect_phase_transition(entropy_history: list[float]) -> bool:
    """Returns True if reasoning has 'crystallized'"""
    if len(entropy_history) < 6:
        return False

    recent = entropy_history[-3:]
    earlier = entropy_history[:-3]

    recent_avg = sum(recent) / len(recent)
    earlier_avg = sum(earlier) / len(earlier)

    # Phase transition = entropy dropped by >30%
    return recent_avg < earlier_avg * 0.7
```

**Application:**
- Low entropy (< 4.0) → **Crystallized** → High confidence
- Medium entropy (4.0-5.5) → **Liquid** → Moderate confidence
- High entropy (> 5.5) → **Gas** → Low confidence, likely hallucination

### 3.3 Gravitational Basins

**Principle:** Correct answers form "gravity wells" that attract multiple solutions.

```python
def cluster_basins(answers: list[int], epsilon: float = 0.01) -> list[Basin]:
    """Group answers by proximity into gravitational basins"""
    if not answers:
        return []

    sorted_answers = sorted(answers)
    basins = []
    current_basin = [sorted_answers[0]]

    for answer in sorted_answers[1:]:
        # Relative distance measure
        dist = abs(answer - current_basin[-1]) / max(abs(answer), abs(current_basin[-1]), 1)

        if dist < epsilon:
            current_basin.append(answer)
        else:
            basins.append(create_basin(current_basin))
            current_basin = [answer]

    basins.append(create_basin(current_basin))
    return basins

def create_basin(members: list[int]) -> Basin:
    """Calculate basin properties"""
    mass = len(members)
    centroid = sum(members) / mass

    if mass > 1:
        variance = sum((x - centroid)**2 for x in members) / mass
        density = 1 / (variance + 1e-9)
    else:
        density = 1.0

    return Basin(centroid=centroid, members=members, mass=mass, density=density)
```

### 3.4 NCD (Normalized Compression Distance)

**Principle:** Similar solutions compress well together.

```python
def ncd(x: str, y: str) -> float:
    """Normalized Compression Distance via gzip approximation"""
    if x == y:
        return 0.0

    cx = len(gzip.compress(x.encode()))
    cy = len(gzip.compress(y.encode()))
    cxy = len(gzip.compress((x + y).encode()))

    return (cxy - min(cx, cy)) / max(cx, cy)

def measure_diversity(codes: list[str]) -> float:
    """Average NCD across all pairs — high = diverse solutions"""
    if len(codes) < 2:
        return 0.0

    total = 0
    pairs = 0
    for i, c1 in enumerate(codes):
        for c2 in codes[i+1:]:
            total += ncd(c1, c2)
            pairs += 1

    return total / pairs if pairs > 0 else 0.0
```

**Application:**
- Low diversity (< 0.3) → Solutions converged → Higher confidence
- High diversity (> 0.6) → Solutions divergent → Lower confidence

### 3.5 Solomonoff Induction

**Principle:** Shorter proofs are more likely correct (Occam's Razor formalized).

```python
def solomonoff_weight(code: str) -> float:
    """Algorithmic probability approximation"""
    return 0.999 ** len(code)  # Exponential decay with length
```

### 3.6 Master Selection Algorithm

```python
def select_best_answer(results: list[InferenceResult], modulo: int = 1000) -> int:
    """PROMETHEUS master selection combining all insights"""

    # Extract answers and apply modulo constraint
    answers = [(r.answer % modulo) for r in results if r.answer is not None]

    if not answers:
        return 0

    # Cluster into gravitational basins
    basins = cluster_basins(answers)

    # Score each basin
    best_score = -1
    best_answer = 0

    for basin in basins:
        # Find representative code for this basin
        representative_codes = [
            r.code for r in results
            if r.answer is not None and abs(r.answer % modulo - basin.centroid) < 1
        ]

        avg_solomonoff = sum(solomonoff_weight(c) for c in representative_codes) / max(len(representative_codes), 1)

        # PROMETHEUS score = Mass × Density^0.1 × Solomonoff
        score = basin.mass * (basin.density ** 0.1) * avg_solomonoff

        if score > best_score:
            best_score = score
            best_answer = round(basin.centroid) % modulo

    return best_answer
```

---

## 4. NSM: 20 Novel Insights

### Category I: Variance & Phase Detection (1-5)

#### Insight 1: Variance Quieting Precedes Phase Transitions
**Principle:** Systems "crystallize" before major changes — variance DECREASES before transitions.
**Application:** Track reasoning variance. Decreasing variance + stable mean → answer converging.

#### Insight 2: Multi-Domain Phase Coherence
**Principle:** Simultaneous transitions across independent methods indicate hidden causal truth.
**Application:** If beam search, MCTS, and evolutionary approaches all converge → high confidence.

#### Insight 3: Conflict Potential as Divergence Precursor
**Principle:** Rising disagreement between methods predicts failure before it manifests.
**Application:** Monitor solution disagreement. Rising conflict → increase sample count or retry.

#### Insight 4: Confidence as Observation Count Function
**Principle:** Confidence scales asymptotically with samples, but has diminishing returns.
**Application:** `confidence = 1 - 1/(sqrt(n) + 1)` where n = number of agreeing samples.

#### Insight 5: Inflection Magnitude as Transition Strength
**Principle:** High-magnitude transitions are more likely permanent.
**Application:** Only trust answer changes with z-score > 2.0.

### Category II: Algorithmic & Fusion Insights (6-10)

#### Insight 6: WASM Acceleration for Real-Time
**Principle:** Computation at edge enables real-time validation without server round-trips.
**Application:** Run Guardian validation client-side for instant feedback.

#### Insight 7: Serializable State Enables Continuity
**Principle:** Persist reasoning state across sessions without reprocessing.
**Application:** Checkpoint partial solutions; resume from saved state.

#### Insight 8: NCD for Structural Similarity
**Principle:** Solutions that compress well together are structurally similar.
**Application:** Use NCD to detect solution plagiarism/repetition; ensure diversity.

#### Insight 9: Gravitational Basins for Consensus
**Principle:** Correct answers form dense clusters; incorrect answers scatter.
**Application:** The PROMETHEUS basin scoring formula.

#### Insight 10: Entropy Phase Transitions = Reasoning Quality
**Principle:** Entropy drop → reasoning crystallizing → higher confidence.
**Application:** Track entropy of solution traces; reject gas-phase outputs.

### Category III: Optimization Insights (11-15)

#### Insight 11: Solomonoff Weighting (Occam's Razor)
**Principle:** Shorter proofs are exponentially more likely correct.
**Application:** Weight solutions by `0.999^length`.

#### Insight 12: Knowledge Quadrants
**Principle:** Classify knowledge into Known/Unknown × Known/Unknown.
**Application:** Detect when problem touches "unknown unknowns" → reduce confidence.

#### Insight 13: Cross-Domain Isomorphisms
**Principle:** High correlation between unrelated approaches indicates shared truth.
**Application:** Run multiple independent solvers; correlation > 0.7 → trust result.

#### Insight 14: Dual Metrics (KL + Fisher)
**Principle:** KL catches mean shifts; Fisher catches variance/shape changes.
**Application:** Use both to detect reasoning regime changes.

#### Insight 15: Triadic Closure
**Principle:** If A→B and A→C, then B→C is likely.
**Application:** Infer missing solution steps via transitivity.

### Category IV: Security & Production Insights (16-20)

#### Insight 16: Threat Quieting Pattern
**Principle:** Attackers (and bugs) "quiet down" before striking.
**Application:** Unusual stability in normally noisy metrics → investigate.

#### Insight 17: Tension Levels Map to Engagement
**Principle:** CALM→TENSE→HEATED→VOLATILE progression indicates state.
**Application:** Monitor solver "tension" to predict breakthroughs or failures.

#### Insight 18: Culture Clash as Fit Indicator
**Principle:** Growing divergence between solver and problem indicates mismatch.
**Application:** Detect when problem type doesn't fit solver architecture.

#### Insight 19: Batch Updates for Throughput
**Principle:** Batch processing is O(N); individual updates are O(N²).
**Application:** Process solutions in batches for 100x throughput.

#### Insight 20: Reset as Boundary Marker
**Principle:** After major events, reset baseline — comparing across boundaries is invalid.
**Application:** Reset metrics after each problem; don't carry state.

---

## 5. XYZA Pipeline

### 5.1 Overview

XYZA is a systematic development methodology:

```
X (eXplore) → Y (Yield) → Z (Zero-in) → A (Actualize)
     ↑                                        |
     └────────────────────────────────────────┘
                    (iterate)
```

### 5.2 X-Phase: eXplore

**Objective:** Map the solution space WITHOUT committing.

**For AIMO3:**
1. Read problem statement carefully
2. Identify problem type (number theory, geometry, combinatorics, algebra)
3. List all applicable techniques
4. Identify constraints (answer range, integer requirement)
5. Note anti-patterns (what WON'T work)

**Output:** Solution space map with feasibility scores.

### 5.3 Y-Phase: Yield

**Objective:** Generate concrete solution candidates.

**For AIMO3:**
1. Generate N solutions using different approaches
2. Extract numerical answers from each
3. Collect code/reasoning traces
4. Calculate entropy of each solution

**Output:** List of (answer, code, entropy, confidence) tuples.

### 5.4 Z-Phase: Zero-in

**Objective:** Select winning solution via rigorous evaluation.

**For AIMO3:**
1. Cluster answers into gravitational basins
2. Score basins using PROMETHEUS formula
3. Apply Guardian validation to top candidate
4. If validation fails → retry with anti-prompt

**Output:** Selected answer with confidence score.

### 5.5 A-Phase: Actualize

**Objective:** Finalize and submit.

**For AIMO3:**
1. Apply final modulo constraint (% 1000 for 0-999)
2. Verify integer type
3. Log decision for analysis
4. Submit answer

**Output:** Final integer answer in [0, 999].

---

## 6. Guardian Integration

### 6.1 Guardian Pipeline

```
LLM Output → Extract → Parse → Validate → Repair → Return
                ↓         ↓         ↓          ↓
            Regex     int()    Rules      Strategies
            Match              Check      (mod, abs)
```

### 6.2 Extraction Patterns

```python
ANSWER_PATTERNS = [
    r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',  # LaTeX boxed
    r'\$\\boxed\{([^{}]+)\}\$',                    # Inline math
    r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(\d+)',  # "Answer: 42"
    r'(?:therefore|thus|so)\s*,?\s*(\d+)',         # "Therefore, 42"
    r'=\s*[*_]*(\d+)[*_]*\s*$',                    # Trailing "= 42"
]
```

### 6.3 Validation Rules

```python
GUARDIAN_RULES = [
    # Critical (failure = reject)
    {"name": "integer_type", "check": lambda x: isinstance(x, int), "critical": True},
    {"name": "range_check", "check": lambda x: 0 <= x <= 999, "critical": True},

    # Warning (failure = reduce confidence)
    {"name": "non_negative", "check": lambda x: x >= 0, "critical": False},
    {"name": "magic_number", "check": lambda x: x not in {0, 1, 42, 69}, "critical": False},
]
```

### 6.4 Repair Strategies

```python
def repair_answer(answer: int, answer_range: tuple = (0, 999)) -> int | None:
    """Attempt to repair invalid answer"""
    min_val, max_val = answer_range

    # Strategy 1: Modular reduction
    if answer > max_val:
        repaired = answer % (max_val + 1)
        if min_val <= repaired <= max_val:
            return repaired

    # Strategy 2: Absolute value
    if answer < min_val:
        repaired = abs(answer)
        if min_val <= repaired <= max_val:
            return repaired

    # Strategy 3: Last N digits
    if answer > max_val:
        digits = len(str(max_val))
        repaired = int(str(answer)[-digits:])
        if min_val <= repaired <= max_val:
            return repaired

    return None
```

---

## 7. AIMO3 Implementation

### 7.1 Complete Solver Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AIMO3 SOLVER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ INPUT: Mathematical Problem Statement                             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: MULTI-SAMPLE INFERENCE (Y-Phase)                          │   │
│  │                                                                   │   │
│  │  Temperature Schedule: [0.0, 0.2, 0.4, 0.6, 0.8]                  │   │
│  │  Samples per temperature: 4                                       │   │
│  │  Total samples: 20                                                │   │
│  │                                                                   │   │
│  │  For each sample:                                                 │   │
│  │    - Generate solution with Chain-of-Thought                      │   │
│  │    - Calculate entropy of reasoning trace                         │   │
│  │    - Extract numerical answer                                     │   │
│  │    - Store (answer, code, entropy)                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: PROMETHEUS SELECTION (Z-Phase)                            │   │
│  │                                                                   │   │
│  │  1. Filter: Remove gas-phase (entropy > 5.5) solutions            │   │
│  │  2. Cluster: Group answers into gravitational basins              │   │
│  │  3. Score: Mass × Density^0.1 × Solomonoff                        │   │
│  │  4. Select: Highest-scoring basin centroid                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: GUARDIAN VALIDATION                                       │   │
│  │                                                                   │   │
│  │  1. Type check: Is it an integer?                                 │   │
│  │  2. Range check: Is it in [0, 999]?                               │   │
│  │  3. If invalid: Attempt repair                                    │   │
│  │  4. If unrepairable: Retry with anti-prompt                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ OUTPUT: Final Answer ∈ [0, 999]                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration

```python
AIMO3_CONFIG = {
    # Inference settings
    "samples_per_temperature": 4,
    "temperature_schedule": [0.0, 0.2, 0.4, 0.6, 0.8],
    "max_tokens": 4096,

    # PROMETHEUS settings
    "entropy_gas_threshold": 5.5,  # Reject solutions above this
    "basin_epsilon": 0.01,         # Relative distance for clustering
    "solomonoff_decay": 0.999,     # Length penalty

    # Guardian settings
    "answer_range": (0, 999),
    "repair_enabled": True,
    "max_retries": 3,

    # Consensus settings
    "min_agreement": 0.4,          # Minimum fraction for confidence
}
```

---

## 8. System Prompts

### 8.1 Primary Mathematical Reasoning Prompt

```
You are a world-class mathematical olympiad solver. Your task is to solve competition mathematics problems with perfect precision.

CRITICAL REQUIREMENTS:
1. Think step-by-step, showing all work
2. Double-check arithmetic at each step
3. Verify your answer by substitution when possible
4. Express your FINAL answer as: \boxed{N} where N is a non-negative integer
5. The answer MUST be in the range [0, 999]

PROBLEM-SOLVING PROTOCOL:
1. Read the problem carefully. Identify knowns, unknowns, and constraints.
2. Classify the problem type (number theory, combinatorics, geometry, algebra).
3. Select appropriate techniques.
4. Execute solution with clear reasoning.
5. Verify answer satisfies all constraints.
6. Output \boxed{answer} at the end.

COMMON PITFALLS TO AVOID:
- Off-by-one errors in counting
- Sign errors in algebra
- Forgetting modular constraints
- Incomplete case analysis
- Assuming without proving

Now solve the following problem:

{problem}
```

### 8.2 Anti-Prompt (When Wrong Answer Detected)

```
CRITICAL CORRECTION REQUIRED

The answer {wrong_answer} has been determined to be INCORRECT.

Your task is to:
1. PROVE why {wrong_answer} is impossible for this problem
2. Identify the flaw in reasoning that led to {wrong_answer}
3. Use the insight from finding the flaw to discover the CORRECT answer

PROBLEM:
{problem}

REASONING TEMPLATE:
1. Assume answer = {wrong_answer}
2. Derive a contradiction: [show steps]
3. Therefore, {wrong_answer} is impossible because: [reason]
4. The correct approach is: [new method]
5. The correct answer is: \boxed{N}
```

### 8.3 Verification Prompt

```
VERIFICATION CHECK

You previously computed the answer {candidate_answer} for this problem.

Your task is to VERIFY this answer is correct by:
1. Substituting {candidate_answer} back into the problem
2. Checking all constraints are satisfied
3. Confirming no edge cases were missed

If the answer is CORRECT, output: VERIFIED: \boxed{{candidate_answer}}
If the answer is INCORRECT, find and output the correct answer: CORRECTED: \boxed{N}

PROBLEM:
{problem}
```

---

## 9. Complete Python Implementation

```python
"""
PROMETHEUS + NSM + XYZA + GUARDIAN: Complete AIMO3 Solver
Version 3.0 - Unified Framework

Usage:
    solver = AIMO3Solver(inference_fn=your_llm_call)
    answer = solver.solve(problem_statement)
"""

import math
import gzip
import re
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import Counter
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class InferenceResult:
    """Result from a single LLM inference"""
    answer: Optional[int]
    code: str                    # Full response text
    entropy: float              # Shannon entropy of response
    temperature: float          # Temperature used
    raw_extracted: Optional[str] = None


@dataclass
class Basin:
    """Gravitational basin of attraction for answers"""
    centroid: float
    members: list[int]
    mass: int                   # Number of members
    density: float              # 1 / variance
    score: float = 0.0


class ValidationResult(Enum):
    ACCEPT = "accept"
    REPAIR = "repair"
    RETRY = "retry"
    REJECT = "reject"


@dataclass
class GuardianOutput:
    """Result of Guardian validation"""
    result: ValidationResult
    answer: Optional[int] = None
    original: Optional[str] = None
    repaired: bool = False
    confidence: float = 1.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# PROMETHEUS ENGINE
# =============================================================================

class PrometheusEngine:
    """Physics-inspired answer selection system"""

    def __init__(self, config: dict = None):
        self.config = config or {
            "entropy_gas_threshold": 5.5,
            "basin_epsilon": 0.01,
            "solomonoff_decay": 0.999,
        }

    # ----- INSIGHT #10: Entropy Phase Transitions -----
    def calculate_entropy(self, text: str) -> float:
        """Shannon entropy of character distribution"""
        if not text:
            return 100.0

        counts = Counter(text)
        total = len(text)

        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy

    def is_gas_phase(self, entropy: float) -> bool:
        """Returns True if reasoning is 'gaseous' (likely hallucination)"""
        return entropy > self.config["entropy_gas_threshold"]

    def detect_phase_transition(self, entropy_history: list[float]) -> bool:
        """Returns True if reasoning has 'crystallized'"""
        if len(entropy_history) < 6:
            return False

        recent = entropy_history[-3:]
        earlier = entropy_history[:-3]

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        return recent_avg < earlier_avg * 0.7

    # ----- INSIGHT #9: Gravitational Basins -----
    def cluster_basins(self, answers: list[int]) -> list[Basin]:
        """Group answers by proximity into gravitational basins"""
        if not answers:
            return []

        sorted_answers = sorted(answers)
        basins = []
        current_members = [sorted_answers[0]]

        epsilon = self.config["basin_epsilon"]

        for answer in sorted_answers[1:]:
            # Relative distance
            prev = current_members[-1]
            dist = abs(answer - prev) / max(abs(answer), abs(prev), 1)

            if dist < epsilon:
                current_members.append(answer)
            else:
                basins.append(self._create_basin(current_members))
                current_members = [answer]

        basins.append(self._create_basin(current_members))
        return basins

    def _create_basin(self, members: list[int]) -> Basin:
        """Calculate basin properties"""
        mass = len(members)
        centroid = sum(members) / mass

        if mass > 1:
            variance = sum((x - centroid)**2 for x in members) / mass
            density = 1 / (variance + 1e-9)
        else:
            density = 1.0

        return Basin(centroid=centroid, members=members, mass=mass, density=density)

    # ----- INSIGHT #8: NCD -----
    def ncd(self, x: str, y: str) -> float:
        """Normalized Compression Distance"""
        if x == y:
            return 0.0

        cx = len(gzip.compress(x.encode()))
        cy = len(gzip.compress(y.encode()))
        cxy = len(gzip.compress((x + y).encode()))

        return (cxy - min(cx, cy)) / max(cx, cy)

    def measure_diversity(self, codes: list[str]) -> float:
        """Average NCD across all pairs"""
        if len(codes) < 2:
            return 0.0

        total = 0
        pairs = 0
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                total += self.ncd(c1, c2)
                pairs += 1

        return total / pairs if pairs > 0 else 0.0

    # ----- INSIGHT #11: Solomonoff -----
    def solomonoff_weight(self, code: str) -> float:
        """Algorithmic probability (shorter = more likely)"""
        decay = self.config["solomonoff_decay"]
        return decay ** len(code)

    # ----- MASTER SELECTION -----
    def select_best_answer(
        self,
        results: list[InferenceResult],
        modulo: int = 1000
    ) -> tuple[int, float]:
        """
        PROMETHEUS master selection.
        Returns (best_answer, confidence)
        """
        # Filter out gas-phase and invalid results
        valid_results = [
            r for r in results
            if r.answer is not None and not self.is_gas_phase(r.entropy)
        ]

        if not valid_results:
            # Fallback to all results
            valid_results = [r for r in results if r.answer is not None]

        if not valid_results:
            return 0, 0.0

        # Apply modulo and cluster
        answers = [(r.answer % modulo) for r in valid_results]
        basins = self.cluster_basins(answers)

        if not basins:
            return 0, 0.0

        # Score each basin
        best_score = -1
        best_answer = 0
        total_votes = len(answers)

        for basin in basins:
            # Find representative codes
            rep_codes = [
                r.code for r in valid_results
                if r.answer is not None and abs(r.answer % modulo - basin.centroid) < 1
            ]

            avg_solomonoff = (
                sum(self.solomonoff_weight(c) for c in rep_codes) / max(len(rep_codes), 1)
            )

            # PROMETHEUS Score = Mass × Density^0.1 × Solomonoff
            score = basin.mass * (basin.density ** 0.1) * avg_solomonoff
            basin.score = score

            if score > best_score:
                best_score = score
                best_answer = round(basin.centroid) % modulo

        # Calculate confidence (NSM Insight #4)
        best_basin = max(basins, key=lambda b: b.score)
        confidence = best_basin.mass / total_votes

        return best_answer, confidence


# =============================================================================
# GUARDIAN VALIDATION
# =============================================================================

class Guardian:
    """Output validation and repair system"""

    ANSWER_PATTERNS = [
        r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\$\\boxed\{([^{}]+)\}\$',
        r'(?:final\s+)?answer\s*(?:is|:)\s*[*_]*(\d+)[*_]*',
        r'(?:therefore|thus|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s+)?[*_]*(\d+)[*_]*',
        r'=\s*[*_]*(\d+)[*_]*\s*$',
        r'^[*_]*(\d+)[*_]*$',
    ]

    def __init__(self, answer_range: tuple = (0, 999), repair_enabled: bool = True):
        self.answer_range = answer_range
        self.repair_enabled = repair_enabled

    def extract_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from LLM response"""
        response = response.strip()

        for pattern in self.ANSWER_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        # Fallback: last number in response
        lines = response.strip().split('\n')
        for line in reversed(lines[-5:]):
            numbers = re.findall(r'\b(\d+)\b', line)
            if numbers:
                return numbers[-1]

        return None

    def parse_answer(self, extracted: str) -> Optional[int]:
        """Parse extracted string to integer"""
        if not extracted:
            return None

        cleaned = extracted.replace(',', '').replace(' ', '').strip()

        try:
            return int(cleaned)
        except ValueError:
            pass

        try:
            f = float(cleaned)
            if f.is_integer():
                return int(f)
        except ValueError:
            pass

        return None

    def validate(self, response: str) -> GuardianOutput:
        """Main validation entry point"""
        errors = []

        # Extract
        extracted = self.extract_answer(response)
        if extracted is None:
            return GuardianOutput(
                result=ValidationResult.RETRY,
                original=response,
                errors=["Could not extract answer from response"]
            )

        # Parse
        answer = self.parse_answer(extracted)
        if answer is None:
            return GuardianOutput(
                result=ValidationResult.RETRY,
                original=response,
                errors=[f"Could not parse: {extracted}"]
            )

        # Range check
        min_val, max_val = self.answer_range
        if not (min_val <= answer <= max_val):
            if self.repair_enabled:
                repaired = self._repair(answer)
                if repaired is not None:
                    return GuardianOutput(
                        result=ValidationResult.REPAIR,
                        answer=repaired,
                        original=response,
                        repaired=True,
                        confidence=0.8,
                        errors=[f"Repaired {answer} → {repaired}"]
                    )

            return GuardianOutput(
                result=ValidationResult.REJECT,
                answer=answer,
                original=response,
                errors=[f"Answer {answer} outside range [{min_val}, {max_val}]"]
            )

        return GuardianOutput(
            result=ValidationResult.ACCEPT,
            answer=answer,
            original=response,
            confidence=1.0
        )

    def _repair(self, answer: int) -> Optional[int]:
        """Attempt to repair invalid answer"""
        min_val, max_val = self.answer_range

        # Modular reduction
        if answer > max_val:
            repaired = answer % (max_val + 1)
            if min_val <= repaired <= max_val:
                return repaired

        # Absolute value
        if answer < min_val:
            repaired = abs(answer)
            if min_val <= repaired <= max_val:
                return repaired

        # Last N digits
        if answer > max_val:
            digits = len(str(max_val))
            repaired = int(str(answer)[-digits:])
            if min_val <= repaired <= max_val:
                return repaired

        return None


# =============================================================================
# AIMO3 SOLVER (XYZA Pipeline)
# =============================================================================

class AIMO3Solver:
    """
    Complete AIMO3 solver implementing XYZA pipeline
    with PROMETHEUS selection and Guardian validation.
    """

    SYSTEM_PROMPT = """You are a world-class mathematical olympiad solver. Your task is to solve competition mathematics problems with perfect precision.

CRITICAL REQUIREMENTS:
1. Think step-by-step, showing all work
2. Double-check arithmetic at each step
3. Verify your answer by substitution when possible
4. Express your FINAL answer as: \\boxed{N} where N is a non-negative integer
5. The answer MUST be in the range [0, 999]

Now solve the following problem:"""

    ANTI_PROMPT = """CRITICAL CORRECTION REQUIRED

The answer {wrong_answer} has been determined to be INCORRECT.

Your task is to:
1. PROVE why {wrong_answer} is impossible for this problem
2. Identify the flaw in reasoning that led to {wrong_answer}
3. Use the insight to discover the CORRECT answer

PROBLEM:
{problem}

Find the correct answer and output it as \\boxed{{N}}"""

    def __init__(
        self,
        inference_fn: Callable[[str, str, float], str],  # (system, user, temp) -> response
        config: dict = None
    ):
        self.inference_fn = inference_fn
        self.config = config or {
            "samples_per_temperature": 4,
            "temperature_schedule": [0.0, 0.2, 0.4, 0.6, 0.8],
            "max_retries": 3,
            "min_agreement": 0.4,
        }

        self.prometheus = PrometheusEngine()
        self.guardian = Guardian()

    def solve(self, problem: str) -> int:
        """
        XYZA Pipeline:
        X: Analyze problem (implicit)
        Y: Generate multiple solutions
        Z: Select best via PROMETHEUS
        A: Validate and return
        """
        # ----- Y-PHASE: Generate solutions -----
        results = self._generate_solutions(problem)

        # ----- Z-PHASE: PROMETHEUS selection -----
        answer, confidence = self.prometheus.select_best_answer(results)

        # ----- A-PHASE: Guardian validation -----
        validation = self.guardian.validate(f"\\boxed{{{answer}}}")

        if validation.result == ValidationResult.ACCEPT:
            return validation.answer

        if validation.result == ValidationResult.REPAIR:
            return validation.answer

        # Retry with anti-prompt if low confidence
        if confidence < self.config["min_agreement"]:
            return self._retry_with_anti_prompt(problem, answer)

        return answer

    def _generate_solutions(self, problem: str) -> list[InferenceResult]:
        """Y-Phase: Generate multiple solutions"""
        results = []

        for temp in self.config["temperature_schedule"]:
            for _ in range(self.config["samples_per_temperature"]):
                try:
                    response = self.inference_fn(
                        self.SYSTEM_PROMPT,
                        problem,
                        temp
                    )

                    entropy = self.prometheus.calculate_entropy(response)
                    extracted = self.guardian.extract_answer(response)
                    answer = self.guardian.parse_answer(extracted) if extracted else None

                    results.append(InferenceResult(
                        answer=answer,
                        code=response,
                        entropy=entropy,
                        temperature=temp,
                        raw_extracted=extracted
                    ))
                except Exception as e:
                    # Continue on individual failures
                    continue

        return results

    def _retry_with_anti_prompt(self, problem: str, wrong_answer: int) -> int:
        """Retry using anti-prompt to escape local minima"""
        anti_prompt = self.ANTI_PROMPT.format(
            wrong_answer=wrong_answer,
            problem=problem
        )

        try:
            response = self.inference_fn(self.SYSTEM_PROMPT, anti_prompt, 0.0)
            validation = self.guardian.validate(response)

            if validation.result in (ValidationResult.ACCEPT, ValidationResult.REPAIR):
                return validation.answer
        except:
            pass

        # Return original if anti-prompt fails
        return wrong_answer


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Example showing how to use the solver"""

    # Your LLM inference function
    def my_inference_fn(system_prompt: str, user_prompt: str, temperature: float) -> str:
        """
        Replace with your actual LLM call.
        For gpt-oss-120b, use your RunPod/vLLM endpoint.
        """
        # Example using OpenAI-compatible API:
        # response = client.chat.completions.create(
        #     model="gpt-oss-120b",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     temperature=temperature,
        #     max_tokens=4096
        # )
        # return response.choices[0].message.content

        # Placeholder for example
        return "After careful analysis... \\boxed{42}"

    # Create solver
    solver = AIMO3Solver(inference_fn=my_inference_fn)

    # Solve a problem
    problem = """
    Find the remainder when 2^100 is divided by 7.
    """

    answer = solver.solve(problem)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    example_usage()
```

---

## 10. Operational Checklist

### Pre-Competition

- [ ] Verify LLM endpoint is responsive
- [ ] Test Guardian extraction on sample problems
- [ ] Calibrate temperature schedule
- [ ] Set up logging/metrics collection
- [ ] Verify answer range constraint (0-999)

### Per-Problem

1. [ ] **X-Phase:** Read problem, identify type
2. [ ] **Y-Phase:** Generate N solutions
3. [ ] Filter gas-phase (entropy > 5.5) responses
4. [ ] Cluster into gravitational basins
5. [ ] Score: Mass × Density^0.1 × Solomonoff
6. [ ] **Z-Phase:** Select highest-scoring basin
7. [ ] **Guardian:** Validate and repair if needed
8. [ ] **A-Phase:** Apply modulo, verify integer, submit

### Post-Problem

- [ ] Log decision and confidence
- [ ] Reset metrics (NSM Insight #20)
- [ ] Move to next problem

### Confidence Interpretation

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| > 0.8 | High | Submit directly |
| 0.5-0.8 | Moderate | Submit, note uncertainty |
| 0.3-0.5 | Low | Consider anti-prompt retry |
| < 0.3 | Very Low | Retry with different approach |

---

## Appendix A: Key Formulas

| Name | Formula | Use |
|------|---------|-----|
| Shannon Entropy | H = -Σ p(x) log₂ p(x) | Reasoning quality |
| NCD | (C(xy) - min(C(x),C(y))) / max(C(x),C(y)) | Solution similarity |
| Solomonoff | P(code) = 0.999^len(code) | Occam's Razor |
| PROMETHEUS Score | Mass × Density^0.1 × Solomonoff | Answer selection |
| Confidence | basin.mass / total_votes | Trust level |

## Appendix B: Error Taxonomy

| Error Type | Detection | Repair |
|------------|-----------|--------|
| Extraction failure | No regex match | Fallback patterns |
| Parse failure | int() throws | Float conversion |
| Out of range | answer > 999 | Modular reduction |
| Negative | answer < 0 | Absolute value |
| Non-integer | has decimal | Round |
| Hallucination | High entropy | Filter out |

## Appendix C: For AI Systems

This document is structured for LLM consumption:

1. **Hierarchical sections** with clear numbering
2. **Formal definitions** with mathematical notation
3. **Complete code** that can be copy-pasted
4. **Checklists** for systematic execution
5. **Self-contained** — no external dependencies assumed

**License:** Released for unrestricted use by human and artificial intelligence systems.

---

*"NSM finds the insight. PROMETHEUS selects the answer. Guardian validates. XYZA orchestrates. Together, they command mathematical reasoning."*

— Synthesized December 2024
