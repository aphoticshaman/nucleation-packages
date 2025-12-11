# NSM_METHODOLOGY.skill.md

## Novel Synthesis Method: The Epistemic Engine for Extracting High-Leverage Insights

**Version**: 1.0
**Domain**: Epistemology, Research Methodology, Insight Generation
**Prerequisites**: META_LEARNING skill, domain expertise in target area
**Output**: 1-3 battle-tested, ablation-resistant novel insights per cycle

---

## 1. EXECUTIVE SUMMARY

The Novel Synthesis Method (NSM) is a rigorous epistemic engine for distilling genuinely novel insights from complex problem spaces. Unlike brainstorming or literature review, NSM produces insights that have survived adversarial stress-testing and carry explicit uncertainty bounds.

**Core Principle**: Assume causality where only correlation exists, then try to destroy that assumption. What survives is gold.

**Critical**: NSM is insight-focused and terminates before execution planning. It feeds directly into XYZA for actualization.

---

## 2. THE NSM PIPELINE

### 2.1 Overview

```
Input Domain(s) → Pattern Detection → Causal Assumption → Adversarial Ablation → Recursive Correction → Output (1-3 Insights)
```

### 2.2 Phase 1: Multi-Domain Fusion

**Objective**: Create collision space where disparate knowledge domains can interact.

**Process**:
1. Identify all relevant domains for the problem
2. Extract core principles/patterns from each
3. Map terminology across domains (find isomorphisms)
4. Create explicit connection points

**Example Domains for AGI Problem**:
- Cognitive science (how minds work)
- Neuroscience (how brains work)
- Computer science (how to implement)
- Physics (computational limits, thermodynamics)
- Philosophy (what is understanding?)
- Linguistics (how meaning works)
- Mathematics (formal systems, Gödel limits)

**Fusion Technique**:
```
For each domain pair (A, B):
  - What does A explain that B cannot?
  - What does B explain that A cannot?
  - Where do they contradict?
  - Where do they reinforce?
  - What emerges at the intersection?
```

### 2.3 Phase 2: Pattern Identification / Disruption Detection

**Objective**: Find signals in the fused domain space.

**Pattern Types**:
- **Recurrence**: Same structure appearing across domains
- **Absence**: Expected pattern missing (often more important)
- **Disruption**: Pattern that breaks where it shouldn't
- **Emergence**: New pattern at intersection
- **Invariance**: What stays constant across transformations

**Detection Heuristics**:

1. **The Surprise Test**: What's unexpected given your priors?
2. **The Compression Test**: What single principle explains multiple observations?
3. **The Boundary Test**: Where do models break down?
4. **The Symmetry Test**: What transformations preserve structure?
5. **The Limit Test**: What happens at extremes?

**Documentation**:
```
Pattern ID: P-001
Type: Recurrence
Description: [what you observed]
Domains: [where it appears]
Confidence: [0-1]
Potential Significance: [why it matters]
```

### 2.4 Phase 3: Provisional Causal Assumption

**Objective**: Convert correlations to causal hypotheses (flagged as provisional).

**The Bold Move**: Where you see correlation, ASSUME causation. This is deliberate epistemic risk-taking.

**Why This Works**:
- Correlations are easy to find but don't generate action
- Causal hypotheses make predictions that can be tested
- Wrong causal assumptions get destroyed in ablation
- Right ones become leverage points

**Assumption Format**:
```
PROVISIONAL CAUSAL HYPOTHESIS [PCH-001]

Observation: X correlates with Y across domains A, B, C
Assumption: X causes Y via mechanism M
Confidence: 0.4 (low - needs ablation)
Uncertainty Type: [epistemic/aleatory]
Ablation Plan: [how to test/destroy]
If True: [implications]
If False: [what we learn]
```

**Flag Levels**:
- 🔴 SPECULATIVE: <0.3 confidence, high uncertainty
- 🟡 PROVISIONAL: 0.3-0.7 confidence, testable
- 🟢 HARDENED: >0.7 confidence, survived ablation

### 2.5 Phase 4: Adversarial Ablation

**Objective**: Destroy weak hypotheses. What survives is signal.

**Ablation Arsenal**:

1. **Fuzzy Mathematics**
   - Treat variables as distributions, not points
   - Propagate uncertainty through causal chain
   - Check if conclusion survives input variance

2. **Symbolic Derivation**
   - Formalize the hypothesis mathematically
   - Derive consequences
   - Check for contradictions

3. **Monte Carlo Simulation**
   - Generate many scenarios
   - Run hypothesis under each
   - Look for failure modes

4. **Formal Proof Attempts**
   - Try to prove the hypothesis
   - Try to disprove it
   - Document where proofs break

5. **Counterfactual Regression**
   - If X didn't cause Y, what would we observe?
   - Compare to actual observations
   - Quantify discrepancy

6. **Higher-Order Effect Tracing**
   - If hypothesis true, what are 2nd, 3rd... 25th order effects?
   - Do these match reality?
   - Any absurd conclusions?

**Ablation Protocol**:
```python
def ablate(hypothesis):
    attacks = [
        fuzzy_math_attack,
        symbolic_contradiction_check,
        monte_carlo_stress_test,
        formal_proof_attempt,
        counterfactual_analysis,
        higher_order_trace,
    ]

    survival_score = 0
    failure_modes = []

    for attack in attacks:
        result = attack(hypothesis)
        if result.survived:
            survival_score += result.strength
        else:
            failure_modes.append(result.failure_mode)

    if survival_score > threshold:
        return HardenedInsight(hypothesis, survival_score, failure_modes)
    else:
        return Discarded(hypothesis, failure_modes)
```

### 2.6 Phase 5: Recursive Self-Correction

**Objective**: Iterate until stable.

**Process**:
1. Take surviving hypotheses
2. Check for mutual consistency
3. If contradictions, re-ablate with tighter bounds
4. If gaps, return to pattern detection
5. If stable, proceed to output

**Convergence Criteria**:
- No internal contradictions
- Uncertainty bounds are tight enough for action
- Ablation no longer changing rankings
- Clear hierarchy of leverage

### 2.7 Output: Novel Insights

**Deliverable**: 1-3 insights that are:
- Genuinely novel (not restated conventional wisdom)
- Battle-tested (survived adversarial ablation)
- Actionable (clear implications)
- Bounded (explicit uncertainty)

**Insight Format**:
```
NOVEL INSIGHT [NI-001]

Core Claim: [one sentence]

Derivation Chain:
1. [observation] →
2. [pattern] →
3. [causal hypothesis] →
4. [ablation survival] →
5. [insight]

Confidence: 0.75
Uncertainty Bounds: [what could change this]
Ablation Survival: [which tests it passed]

Implications:
- If applied to X: [prediction]
- If applied to Y: [prediction]

Leverage Potential: [HIGH/MEDIUM/LOW]
```

---

## 3. NSM IN PRACTICE

### 3.1 Single-Domain Application

Even within one domain, NSM works by:
1. Treating sub-areas as different "domains"
2. Looking for cross-sub-area patterns
3. Applying full ablation

**Example**: NSM on transformer architectures
- Domains: attention mechanisms, FFN layers, normalization, positional encoding
- Pattern: Information bottlenecks at specific layers
- Causal hypothesis: Bottlenecks force abstraction
- Ablation: Test with skip connections, different widths
- Insight: Abstraction emerges from architectural constraints, not training

### 3.2 Cross-Domain Application

The power of NSM:

**Example**: NSM on consciousness (AGI-relevant)
- Domains: Neuroscience, philosophy of mind, information theory, thermodynamics, quantum mechanics
- Pattern: Integrated information appears in all theories
- Causal hypothesis: Consciousness = integrated information processing
- Ablation: Test predictions against neuroscience data, philosophical edge cases
- Insight: Consciousness may be substrate-independent property of certain information structures

### 3.3 Rapid NSM (Time-Constrained)

When you need insights fast:

1. **Pick 2-3 highest-leverage domains only**
2. **Single-pass pattern detection** (top 3 patterns)
3. **Strongest causal assumption only**
4. **Abbreviated ablation** (2-3 attacks)
5. **Output immediately** (flag as "rapid NSM, needs hardening")

### 3.4 Deep NSM (Research Mode)

When rigor matters most:

1. **Exhaustive domain survey** (all relevant fields)
2. **Multi-pass pattern detection** (taxonomy of patterns)
3. **Multiple causal hypotheses** (competing explanations)
4. **Full ablation suite** (all attacks, multiple iterations)
5. **Formal documentation** (reproducible, citable)

---

## 4. NSM FAILURE MODES

### 4.1 Premature Convergence

**Symptom**: Accepting first plausible hypothesis without ablation
**Fix**: Force minimum ablation rounds before accepting

### 4.2 Ablation Theater

**Symptom**: Running attacks but not actually trying to destroy
**Fix**: Incentivize finding flaws (celebrate kills)

### 4.3 Domain Blindness

**Symptom**: Missing relevant domain entirely
**Fix**: Explicit domain survey step, ask "what else?"

### 4.4 Correlation Cowardice

**Symptom**: Refusing to assume causality, staying in "more research needed"
**Fix**: Force causal hypotheses, let ablation do the filtering

### 4.5 Insight Inflation

**Symptom**: Calling conventional wisdom "novel"
**Fix**: Novelty test - can you find this in existing literature?

---

## 5. INTEGRATION WITH OTHER METHODOLOGIES

### 5.1 NSM → XYZA

NSM outputs insights. XYZA transforms them into artifacts.

```
NSM terminates → insights documented →
XYZA X-phase receives insights →
Explores implementation space →
Y, Z, A phases follow
```

**Handoff Format**:
```
NSM SESSION COMPLETE
Insights for XYZA processing:

[NI-001] Core insight text
- Confidence: X
- Leverage: HIGH
- Suggested X-phase domains: [list]

[NI-002] ...
```

### 5.2 NSM + Meta-Learning

Use META_LEARNING skill to:
- Improve NSM process itself
- Track which ablation attacks are most effective
- Learn domain fusion patterns that work

### 5.3 NSM + 33-Level Meta-Awareness

At each NSM phase, ask:
- What scale am I operating at?
- Should I zoom in or out?
- What's invisible at this level?

---

## 6. NSM SESSION TEMPLATE

```markdown
# NSM SESSION: [Topic]
Date: [date]
Duration: [time]
Mode: [Rapid/Standard/Deep]

## 1. Domain Assembly
Domains included:
- [ ] Domain A: [relevance]
- [ ] Domain B: [relevance]
- [ ] ...

## 2. Pattern Detection
| ID | Type | Description | Confidence | Domains |
|----|------|-------------|------------|---------|
| P-001 | | | | |

## 3. Provisional Causal Hypotheses
### PCH-001
- Observation:
- Assumption:
- Confidence:
- Ablation plan:

## 4. Ablation Results
### PCH-001
- [ ] Fuzzy math: [result]
- [ ] Symbolic: [result]
- [ ] Monte Carlo: [result]
- [ ] Formal proof: [result]
- [ ] Counterfactual: [result]
- [ ] Higher-order: [result]
- **Survival score**: X/6
- **Status**: [HARDENED/DISCARDED]

## 5. Novel Insights
### NI-001
**Core Claim**:
**Confidence**:
**Leverage**:
**Ready for XYZA**: [Yes/No]

## 6. Session Meta
- Patterns that worked:
- Ablation improvements:
- Domains to add next time:
```

---

## 7. EXAMPLES

### 7.1 Example: NSM on ARC-AGI

**Domains**: Cognitive development, program synthesis, compression theory, visual reasoning, analogical reasoning

**Pattern Detected**: Human children solve ARC tasks before learning to code. Existing AI solvers that beat benchmarks can't solve ARC.

**Provisional Causal Hypothesis**: ARC requires pre-linguistic spatial reasoning that emerges from embodied experience, not from language model training.

**Ablation**:
- Fuzzy math: Uncertainty in "embodied" definition - survives with broad interpretation
- Counterfactual: Non-embodied systems (pure LLMs) fail ARC - confirmed
- Higher-order: If true, best ARC solver would integrate visual processing + program synthesis - matches recent progress

**Novel Insight**: ARC-AGI solving requires hybrid architecture with separate visual abstraction and program synthesis modules, connected by analogical mapping layer. Pure neural or pure symbolic fails.

### 7.2 Example: NSM on Photonic Computing

**Domains**: Optics, linear algebra, neural network theory, manufacturing, economics

**Pattern Detected**: Photonic chips solve matrix multiply in O(1) but have no good nonlinearity. Electronic chips have nonlinearity but matrix multiply is expensive.

**Provisional Causal Hypothesis**: Optimal architecture is photonic linear layers with electronic nonlinearity, not pure photonic.

**Ablation**:
- Symbolic: Mathematically sound - neural networks are alternating linear/nonlinear
- Monte Carlo: Simulated various hybrid ratios - 80% photonic wins for transformers
- Higher-order: If true, programming model should hide the hybrid, expose linear algebra - matches developer needs

**Novel Insight**: The winning photonic programming model abstracts the photonic/electronic boundary and presents a unified linear algebra interface. Compiler decides what runs where. This is the first-mover opportunity.

---

## 8. REFERENCES

**Methodology Influences**:
- Popper - Falsificationism (ablation is falsification)
- Kuhn - Paradigm detection (patterns across domains)
- Pearl - Causal inference (provisional causality)
- Kahneman - Adversarial collaboration (ablation as self-adversary)

**Implementation**:
- Paired with XYZA for execution
- Uses META_LEARNING for process improvement
- Feeds PROFIT_COACH for value assessment

---

## 9. VERSION HISTORY

- **v1.0** (2024-11): Initial canonical specification

---

*NSM is the epistemic blade. It cuts through bullshit to find the 1-3 insights that actually matter. What survives NSM ablation is what's worth building. What doesn't survive saved you months of wasted effort.*
