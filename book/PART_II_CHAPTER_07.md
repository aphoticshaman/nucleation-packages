# Chapter 7: The CIC Framework

The previous chapter established what we need from an aggregation framework: structure awareness, compression-based similarity, uncertainty quantification, multi-scale coherence, and dynamic adaptation. This chapter delivers that framework.

The Compression-Integration-Coherence (CIC) functional is:

**F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)**

This equation looks simple. It hides considerable depth.

Each term captures a fundamental aspect of inference quality. Together, they define a single objective that balances structure, uncertainty, and consistency. Maximizing F selects predictions that are algorithmically cohesive, confidently converged, and coherent across scales.

This chapter unpacks each term, explains why these particular components matter, and connects the functional to established frameworks from information theory, statistical physics, and neuroscience.

---

## 7.1 The Functional: Structure and Intuition

### The General Form

The CIC functional takes a representation T (a set of predictions, embeddings, or samples) and returns a scalar score:

**F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)**

where:

- **Φ(T)** = Information cohesion (how much predictions share algorithmic structure)
- **H(T|X)** = Representation entropy (how uncertain/dispersed predictions are)
- **C_multi(T)** = Multi-scale structural coherence (consistency across granularities)
- **λ = 0.5** = Weight on entropy penalty
- **γ = 0.3** = Weight on coherence bonus

High F means: predictions are cohesive (high Φ), certain (low H), and consistent (high C_multi).

Low F means: predictions are scattered (low Φ), uncertain (high H), or inconsistent (low C_multi).

### Why These Three Terms?

The functional's three terms aren't arbitrary. They capture three orthogonal failure modes in inference:

**Failure Mode 1: Fragmentation (Low Φ)**

Predictions that don't share structure. The model is producing different answers via different algorithms. There's no underlying consensus—just noise.

Detection: Low Φ indicates no shared algorithmic structure. Predictions compress independently, not together.

Solution: Identify subsets that do share structure. Find the coherent core.

**Failure Mode 2: Uncertainty (High H)**

Predictions that are spread out. Even if they share some structure, the spread is too wide for confident decision-making.

Detection: High H indicates high variance. The model hasn't converged on an answer.

Solution: Either gather more samples (to reduce uncertainty) or acknowledge low confidence.

**Failure Mode 3: Incoherence (Low C_multi)**

Predictions that look good at one scale but fall apart at another. The fine-grained structure doesn't match the coarse-grained structure.

Detection: Low C_multi indicates scale-dependent inconsistency. Zooming in gives different answers than zooming out.

Solution: Trust predictions that are consistent across scales. Distrust predictions that only work at one resolution.

### The Balance Equation

CIC is a balance equation. The three terms push in different directions:

- **Φ ↑** wants predictions that cluster together (shared structure)
- **H ↓** wants predictions with low variance (tight distribution)
- **C_multi ↑** wants predictions consistent across scales

But these can conflict:

- Tight clusters (low H) might achieve tightness by excluding outliers that actually contain signal
- High cohesion (high Φ) might come from systematic bias rather than correct answers
- Multi-scale coherence (high C_multi) might be satisfied by large, diffuse clusters

The weights λ and γ control the balance. At λ = 0.5 and γ = 0.3, empirical testing shows optimal tradeoffs for typical LLM inference. Different domains may require different weights.

### Connection to Optimization

From an optimization perspective, maximizing F is equivalent to finding:

**T* = argmax_T [Φ(T) - λ·H(T|X) + γ·C_multi(T)]**

This is the "best" representation—the one that best balances cohesion, certainty, and coherence.

In practice, we don't optimize over all possible representations T. We score the representations we have (the actual predictions) and select the highest-scoring cluster. But the optimization framing illuminates what we're doing: searching for the representation that maximizes the CIC objective.

---

## 7.2 Information Cohesion: The Φ Term

Information cohesion measures how much predictions share algorithmic structure.

### The Core Idea

Two predictions are algorithmically similar if they were produced by similar computations. A prediction of 19,481 and a prediction of 19,520 might be numerically close, but if one came from correct arithmetic and the other from a systematic error, they're algorithmically distant.

How do you measure algorithmic similarity without access to the generating process?

Through compression.

### Kolmogorov Complexity and Compression Distance

The theoretical foundation is Kolmogorov complexity—the length of the shortest program that generates a string. Two strings with high Kolmogorov mutual information share algorithmic structure; they can be generated by related programs.

Kolmogorov complexity is uncomputable. But practical compressors (gzip, LZ77, etc.) approximate it. The Normalized Compression Distance (NCD) operationalizes this:

**NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))**

where C(·) is compressed length.

**Interpretation:**

- NCD = 0: x and y are algorithmically identical (compressing together adds nothing new)
- NCD = 1: x and y are algorithmically independent (compressing together gains nothing)
- NCD in between: partial algorithmic similarity

### From NCD to Φ

Information cohesion averages the (inverse) compression distances across all prediction pairs:

**Φ = 1 - mean(NCD(sᵢ, sⱼ)) for all pairs i < j**

This gives:

- Φ ≈ 1: All predictions share algorithmic structure (low NCD everywhere)
- Φ ≈ 0: Predictions are algorithmically independent (high NCD everywhere)

### Extended NCD for Numeric Data

Standard NCD struggles with short numeric strings. The number "19481" compresses to almost the same size as "19520"—there's not enough redundancy to distinguish them.

Extended NCD addresses this by computing multiple representations of each number:

1. **Raw bytes**: The number as a byte string (captures magnitude)
2. **Digit string**: The decimal representation (captures digit structure)
3. **Binary string**: The binary representation (captures bit patterns)
4. **Prime residues**: Remainders mod small primes (captures number-theoretic structure)
5. **Digit histogram**: Frequency of each digit (captures distributional structure)

For each representation k, compute NCD_k. Then:

**NCD_ext(x, y) = min_k NCD_k(x, y)**

The minimum across representations gives the tightest bound on algorithmic similarity. If two numbers are similar in any representation, they're algorithmically related.

### Why Minimum?

Taking the minimum rather than average is deliberate.

If two numbers have the same digit histogram, they share some algorithmic structure—even if their binary representations differ completely. The minimum captures this: as long as *some* representation shows similarity, the numbers are related.

Average would be stricter—requiring similarity across all representations. This is too conservative. Algorithmic similarity is disjunctive: similar in any respect means similar overall.

### Computational Complexity

NCD requires O(n²) pairwise compressions, where n is the number of predictions. Each compression is O(m log m) for strings of length m.

For typical ensemble sizes (n ≤ 100) and numeric predictions (m ≤ 20 digits), this is fast—under 50ms on commodity hardware.

For larger ensembles, approximations exist:
- **Subsampling**: Compute NCD on a random subset
- **Locality-sensitive hashing**: Approximate compression distance via hash similarity
- **Cluster-first**: Identify clusters by other means, compute Φ within clusters

---

## 7.3 Representation Entropy: The H Term

Representation entropy measures uncertainty—how spread out predictions are.

### The Core Idea

A tight cluster of predictions indicates consensus. A spread-out distribution indicates uncertainty. Entropy quantifies this spread.

High H means the model is exploring—producing diverse outputs with no clear winner. Low H means the model has converged—producing consistent outputs that cluster together.

### Computing H from Samples

For a set of predictions {s₁, s₂, ..., sₙ}, we compute:

1. **Normalize**: ŝᵢ = sᵢ / |mean(s)|
2. **Variance**: Var(ŝ)
3. **Clamp**: H = min(1, Var(ŝ))

The normalization makes H scale-invariant. A spread of ±100 around 1000 has the same H as a spread of ±1 around 10.

The clamping keeps H in [0, 1]. Variance can exceed 1 for highly dispersed predictions; clamping prevents the entropy term from dominating.

### Why Variance, Not Shannon Entropy?

Classical information entropy requires a probability distribution:

**H_Shannon = -Σ p(x) log p(x)**

But we don't have a distribution—we have samples. Estimating the distribution from samples introduces additional uncertainty.

Variance-based entropy is simpler and more robust:
- Directly computable from samples
- No density estimation required
- Captures the relevant property (spread)

For Gaussian distributions, variance and Shannon entropy are monotonically related. For non-Gaussian distributions, variance remains a useful proxy.

### The λ Penalty

The full entropy term is **-λ·H(T|X)**, with λ = 0.5.

The negative sign makes this a penalty: high entropy decreases F. The λ weight controls how strongly we penalize uncertainty relative to rewarding cohesion.

At λ = 0.5:
- A spread of 1 standard deviation costs 0.5 units of F
- This balances against Φ (which ranges 0-1) appropriately
- Empirically, this gives optimal error rates on test distributions

Lower λ tolerates more uncertainty (favor exploration). Higher λ penalizes uncertainty more strongly (favor exploitation).

### Entropy Dynamics

Entropy isn't static—it evolves as you gather more predictions.

**Early Sampling**: H is typically high. The model is exploring the space of possible answers. Predictions are diverse.

**Mid Sampling**: H begins decreasing. Patterns emerge. Predictions start clustering.

**Late Sampling**: H is low. The model has converged. Predictions are consistent.

This dynamic is crucial for convergence detection (Chapter 9). The rate of entropy change—especially d²H/dt²—signals when the system is transitioning from exploration to exploitation.

---

## 7.4 Multi-Scale Structural Coherence: The C_multi Term

Multi-scale coherence measures consistency across granularities.

### The Core Idea

A prediction that looks correct at one scale should still look correct at other scales.

- At fine scale: Individual predictions should agree with their neighbors
- At medium scale: Clusters should have consistent internal structure
- At coarse scale: The overall answer should fit contextual constraints

A prediction that only works at one scale is suspicious—it might be exploiting artifacts rather than capturing truth.

### Three Scales of Coherence

We define coherence at three scales:

**Scale 1: Exact Consensus (C₁)**

How many predictions are exactly identical?

**C₁ = max_count(s) / n**

where max_count is the frequency of the most common value.

C₁ = 1 if all predictions are identical. C₁ → 0 if all predictions are unique.

This is the strictest test—perfect agreement. It's also fragile; numeric predictions rarely match exactly.

**Scale 2: Cluster Coherence (C₂)**

How tightly do predictions cluster?

**C₂ = count(pairs with relative_distance < 0.05) / total_pairs**

where relative_distance = |sᵢ - sⱼ| / max(|sᵢ|, |sⱼ|).

C₂ ≈ 1 if all predictions are within 5% of each other. C₂ ≈ 0 if predictions are scattered.

This is a softer test—near-agreement. It handles the numeric precision issues that break exact consensus.

**Scale 3: Range Constraint (C₃)**

How concentrated is the prediction range?

**C₃ = 1 / (1 + spread / center)**

where spread = max(s) - min(s) and center = median(s).

C₃ ≈ 1 if the range is narrow relative to the center. C₃ → 0 if the range is wide.

This is the loosest test—bounded disagreement. Even without tight clustering, the predictions should be in the same ballpark.

### Combining Scales

The three scales combine with weights:

**C_multi = 0.5·C₁ + 0.3·C₂ + 0.2·C₃**

The weights emphasize finer scales. Exact consensus (C₁) is most valuable; range constraint (C₃) is least valuable.

These weights were determined empirically. Different domains might weight differently:
- Categorical predictions: emphasize C₁
- Continuous predictions: emphasize C₂
- Order-of-magnitude estimates: emphasize C₃

### Why Multi-Scale?

Why not just use C₂ (cluster coherence)?

Because multi-scale analysis catches different failure modes:

**Failure Mode A: False precision**

Predictions cluster tightly at the wrong value. High C₂, but the cluster is wrong.

C₁ catches this if the tight cluster came from systematic bias producing identical wrong answers. C₃ catches this if the tight wrong cluster is actually outside the plausible range.

**Failure Mode B: Scale confusion**

The model predicts 1,948.1 when the answer is 19,481—correct digits, wrong magnitude.

C₁ and C₂ would see the wrong answer as coherent (all predictions have the same scale error). C₃ might catch it if the range constraints are informed by prior knowledge.

**Failure Mode C: Bimodal distribution**

Half the predictions cluster around A, half around B.

C₁ is low (neither A nor B dominates). C₂ is moderate (each cluster is tight internally). C₃ is low (the spread from A to B is large). Multi-scale analysis correctly identifies low coherence.

### The γ Bonus

The full coherence term is **+γ·C_multi(T)**, with γ = 0.3.

The positive sign makes this a bonus: high coherence increases F. The γ weight controls how strongly we reward coherence relative to cohesion.

At γ = 0.3:
- Perfect multi-scale coherence adds 0.3 to F
- This is less than the maximum contribution from Φ (1.0) or H (0.5)
- Coherence is important but not dominant

Lower γ cares less about scale consistency. Higher γ demands strong multi-scale agreement.

---

## 7.5 Why This Works: Theoretical Connections

The CIC functional isn't arbitrary. It recapitulates deep principles from three domains: information theory, statistical physics, and neuroscience.

### Connection 1: Information Theory and MDL

The Minimum Description Length (MDL) principle says: the best model is the one that compresses the data most.

CIC operationalizes this for inference. Φ measures how well predictions compress together—how much algorithmic structure they share. Maximizing Φ is analogous to minimizing description length for the prediction set.

More precisely:

**MDL**: Select model M* that minimizes L(M) + L(D|M)
- L(M) = description length of model
- L(D|M) = description length of data given model

**CIC**: Select prediction P* that maximizes Φ(P) - λH(P)
- Φ(P) ≈ -L(P|shared_structure) (negative description length)
- H(P) ≈ L(uncertainty|P) (description length of remaining uncertainty)

The mapping is loose but illuminating. CIC inherits MDL's theoretical guarantees about consistency and generalization.

### Connection 2: Statistical Physics and Free Energy

The Landau free energy from statistical physics is:

**F_Landau = E - TS**

where E is energy, T is temperature, and S is entropy.

Systems minimize free energy. At low temperature, E dominates and systems order. At high temperature, S dominates and systems disorder. The transition between phases occurs at a critical temperature T_c.

CIC mirrors this structure:

**F_CIC = Φ - λH + γC**

- Φ plays the role of negative energy (order increases Φ)
- H plays the role of entropy (disorder increases H)
- λ plays the role of temperature (controls the Φ-H tradeoff)
- C provides an additional ordering force (coherence across scales)

The analogy suggests that inference systems undergo phase transitions. In the "disordered phase" (high λ, high H), predictions are scattered—exploration mode. In the "ordered phase" (low λ, low H), predictions crystallize—exploitation mode. The transition occurs when conditions favor order over disorder.

Chapter 9 develops this physical analogy into a full regime classification system.

### Connection 3: Neuroscience and Free Energy

Friston's Free Energy Principle proposes that biological systems minimize variational free energy:

**F_var = D_KL(q(z|x) || p(z)) - E_q[log p(x|z)]**

which simplifies to:

**F_var = Complexity - Accuracy**

Systems should be accurate (predict observations well) but not complex (don't overfit with unnecessary structure).

CIC parallels this:

**F_CIC = Structure - Uncertainty + Coherence**

where:
- Φ (structure) ≈ accuracy—predictions that share structure predict each other well
- H (uncertainty) ≈ complexity—high variance predictions have more "degrees of freedom"
- C (coherence) ≈ a regularization term requiring consistency across scales

The correspondence is structural, not formal. CIC doesn't explicitly compute KL divergences or likelihood terms. But the balance—accuracy vs. complexity, signal vs. noise—is the same.

This connection suggests CIC might capture something deep about adaptive inference. If biological brains minimize variational free energy, and CIC has the same structure, then CIC-guided artificial systems might exhibit similar robustness.

### Connection 4: The Information Bottleneck

Tishby's Information Bottleneck optimizes:

**L_IB = I(X;T) - βI(T;Y)**

Compress X into T while preserving information about Y. The tradeoff is controlled by β.

CIC doesn't explicitly compute mutual information, but there's a connection:

- Φ measures I(T;T)—self-information within the prediction set. High Φ means predictions are informative about each other.
- H measures uncertainty in T—the "width" of the information bottleneck.
- C_multi measures whether information is preserved across scales—a multi-resolution generalization of the bottleneck.

The formal relationship is:

**Theorem (Informal):** Under sub-Gaussian noise and cluster separability assumptions, Φ lower-bounds mutual information:

**I(T;T*) ≥ α·Φ(T)**

where T* is the optimal representation and α depends on the noise level.

This theorem (proven in the formal appendix) connects CIC's practical Φ metric to information-theoretic fundamentals.

---

## 7.6 The Full Algorithm

Putting it together, the CIC scoring algorithm is:

```
ALGORITHM: CIC Scoring
INPUT: predictions {s₁, ..., sₙ}
OUTPUT: score F, confidence c

1. Compute Φ (Information Cohesion):
   a. For each pair (i, j), compute NCD_ext(sᵢ, sⱼ)
   b. Φ = 1 - mean(NCD_ext)

2. Compute H (Representation Entropy):
   a. Normalize: ŝᵢ = sᵢ / |mean(s)|
   b. H = min(1, Var(ŝ))

3. Compute C_multi (Multi-Scale Coherence):
   a. C₁ = max_count(s) / n
   b. C₂ = fraction of pairs with relative_distance < 0.05
   c. C₃ = 1 / (1 + spread / center)
   d. C_multi = 0.5·C₁ + 0.3·C₂ + 0.2·C₃

4. Compute F:
   F = Φ - 0.5·H + 0.3·C_multi

5. Compute confidence:
   c = clamp(0.5 + 0.5·F, 0.05, 0.95)

RETURN F, c
```

### Confidence Derivation

The confidence transformation:

**c = clamp(0.5 + 0.5·F, 0.05, 0.95)**

maps F ∈ [-1, 1] to confidence ∈ [0.05, 0.95].

- F = 1 (perfect) → c = 0.95 (highly confident)
- F = 0 (neutral) → c = 0.50 (uncertain)
- F = -1 (terrible) → c = 0.05 (very low confidence)

The bounds [0.05, 0.95] enforce epistemic humility:
- We never claim certainty (c < 1.0)—even perfect predictions might be wrong
- We never claim impossibility (c > 0.0)—even terrible predictions might be right

This calibration is intentional. Overconfident AI systems are dangerous. The clamping ensures CIC never produces extreme certainties that could lead to overreliance.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Φ computation | O(n²·m) | O(n²) |
| H computation | O(n) | O(1) |
| C_multi computation | O(n²) | O(1) |
| Total | O(n²·m) | O(n²) |

where n = number of predictions, m = average prediction length.

For n ≤ 100, this runs in milliseconds. For larger n, approximate algorithms reduce to O(n log n) with minor accuracy loss.

---

## 7.7 Parameter Sensitivity

The CIC functional has three key parameters: λ, γ, and the coherence weights (0.5, 0.3, 0.2).

### Finding Optimal λ and γ

The default values λ = 0.5, γ = 0.3 emerged from grid search over test distributions.

The search procedure:
1. Generate synthetic prediction sets with known ground truth
2. For each (λ, γ) pair in [0.1, 0.9] × [0.1, 0.5]:
   - Compute CIC scores
   - Measure error rate of selected predictions
3. Select (λ, γ) minimizing error rate

Results across 1000 test cases:
- Optimal λ ∈ [0.4, 0.6], with λ = 0.5 most robust
- Optimal γ ∈ [0.2, 0.4], with γ = 0.3 most robust

The optimum is fairly flat—small deviations from (0.5, 0.3) have minimal impact. This suggests the functional is robust, not finely tuned to a specific distribution.

### Domain-Specific Adjustment

Different domains might benefit from different parameters:

**High-noise domains (noisy sensors, unreliable models):**
- Increase λ (penalize uncertainty more)
- Increase γ (reward coherence more)
- Recommended: λ = 0.6, γ = 0.4

**Low-noise domains (precise instruments, reliable models):**
- Decrease λ (tolerate some uncertainty)
- Decrease γ (don't overweight coherence)
- Recommended: λ = 0.4, γ = 0.2

**High-stakes domains (medical, safety-critical):**
- Keep λ moderate (want signal over noise)
- Increase γ (demand consistency)
- Recommended: λ = 0.5, γ = 0.4

### Coherence Weight Adjustment

The default coherence weights (0.5, 0.3, 0.2) emphasize exact consensus.

For continuous predictions where exact matches are rare:
- Decrease C₁ weight: (0.2, 0.5, 0.3)

For categorical predictions where clusters are discrete:
- Increase C₁ weight: (0.7, 0.2, 0.1)

For order-of-magnitude estimates:
- Increase C₃ weight: (0.3, 0.3, 0.4)

---

## Summary: The CIC Framework

The Compression-Integration-Coherence functional provides a principled framework for aggregating predictions:

**F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)**

**Φ (Information Cohesion):** Measures shared algorithmic structure via compression distance. High Φ means predictions encode related information.

**H (Representation Entropy):** Measures uncertainty via normalized variance. High H means predictions are scattered; low H means consensus.

**C_multi (Multi-Scale Coherence):** Measures consistency across granularities. High C_multi means predictions agree at all scales.

The functional balances three imperatives:
1. Maximize shared structure (Φ ↑)
2. Minimize uncertainty (H ↓)
3. Maintain coherence (C_multi ↑)

The optimal prediction—the one that maximizes F—is the one that best satisfies all three.

Theoretical connections to:
- **MDL:** Compression-based model selection
- **Free Energy (Physics):** Phase transitions and order parameters
- **Free Energy (Neuroscience):** Accuracy-complexity tradeoffs
- **Information Bottleneck:** Compression-prediction balance

These connections aren't coincidental. CIC captures something universal about optimal inference—the same mathematical pattern that appears wherever adaptive systems aggregate information under uncertainty.

The next chapter applies this framework to the specific problem of value clustering: given multiple numeric predictions, how do you use CIC to select the best answer?
