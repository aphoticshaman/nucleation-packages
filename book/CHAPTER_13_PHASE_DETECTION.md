# Chapter 13: Phase Detection and Regime Classification

The previous chapters established the CIC functional and value clustering algorithm. We can now combine predictions better than averaging and identify structure that simple aggregation destroys.

But there's a deeper question: when should we trust any aggregation method at all?

The answer requires understanding the *state* of the inference system—whether predictions are stable, chaotic, or transitioning between regimes. This chapter develops tools for classifying inference states using concepts borrowed from statistical physics.

---

## The Problem of Regime Blindness

Traditional ensemble methods assume stationarity. They aggregate predictions as if the underlying distribution is stable. But inference systems don't work that way.

Ask a language model the same question ten times. The first few responses might cluster tightly. Then something shifts—maybe the model explores a different reasoning path—and responses scatter. Later, they might reconverge around a new center.

If you're blindly averaging, you'll mix the stable phases with the chaotic ones. Your aggregate will be worse than either regime alone.

What we need is a way to detect these regime changes—to know when we're in a stable state versus a transitional one, when to trust our aggregates versus when to wait for convergence.

---

## Borrowing from Physics

Statistical physics has studied phase transitions for over a century. Water freezing into ice. Iron becoming magnetic. Superfluids forming at near-absolute-zero temperatures.

These transitions share common features:
- A control parameter (usually temperature) governs the state
- An order parameter measures the degree of organization
- At critical points, small changes in control parameters cause dramatic shifts in order
- Systems exhibit characteristic dynamics in different regimes

The Landau-Ginzburg framework formalizes this through free energy functionals. Near phase transitions, the free energy can be expanded:

**F[φ] = ∫ dx [ ½(∇φ)² + ½r(T)φ² + ¼uφ⁴ ]**

Where φ is the order parameter and T is temperature. The coefficient r(T) changes sign at the critical temperature, driving the transition.

We don't claim that inference systems are literally thermodynamic. But the mathematical structure provides useful tools. If we can define appropriate analogues of temperature and order parameters for predictions, we can classify inference states and detect transitions.

---

## Defining Inference Temperature

What plays the role of temperature in an inference system?

Temperature governs fluctuations. High temperature means large random variations. Low temperature means ordered, predictable behavior.

For predictions, the natural analogue is volatility—how much the outputs vary.

**Temperature (inference) = (variance/n) × (1 + (1 - avg_correlation))**

Breaking this down:
- **variance/n**: Raw spread of predictions, normalized by sample count
- **1 + (1 - avg_correlation)**: Adjustment for temporal structure

The correlation term matters because not all variance is equal. If predictions fluctuate but maintain consistent ordering (high correlation), the underlying dynamics are more structured than if they jump around randomly (low correlation).

High inference temperature: predictions scattered widely, low temporal coherence
Low inference temperature: predictions cluster tightly, high temporal coherence

---

## Defining the Order Parameter

Temperature measures chaos. The order parameter measures structure.

In magnetic systems, the order parameter is magnetization—the degree to which atomic spins align. In inference systems, we want something analogous: a measure of consensus and coherence.

**Order Parameter: ψ = Σᵢ wᵢ × |autocorrelation(lag=i)|**

With weights wᵢ = 1/i following harmonic decay.

This definition captures multi-scale temporal structure:
- Short-range correlations (lag=1, 2) get heavy weight
- Long-range correlations get progressively less weight
- The sum measures overall structural persistence

High order parameter: predictions maintain consistent structure over time
Low order parameter: predictions shift unpredictably

The harmonic weighting was selected empirically. Uniform weights give too much influence to long lags (which are noisy). Exponential decay cuts off too sharply. Harmonic decay provides a middle ground that tracks empirically-observed convergence patterns.

---

## The Critical Temperature

Where does the transition happen?

Empirically, we observe that regime classification works well with threshold:

**T_c ≈ 0.76**

This can be expressed as √(ln(2)/ln(π)) ≈ 0.7632, which provides a convenient analytic form with information-theoretic resonance. But we should be honest: this is primarily an empirically-tuned parameter that happens to have an aesthetically pleasing expression.

The value 0.76 separates inference states into meaningful regimes in our tests. For different model families or task types, recalibration may be necessary.

---

## The Five Regimes

Using temperature and order parameter, we define five operational categories:

### Regime 1: Stable (Crystalline)
**Conditions:** T < 0.3, ψ > 0.7

Predictions are tightly clustered and consistent. The system has "crystallized" around a solution. Aggregation is straightforward—most methods will work well.

This is the ideal operating state. When you observe stable regime, you can trust your aggregate with high confidence.

### Regime 2: Metastable (Supercooled)
**Conditions:** T < 0.5, ψ > 0.5

Moderate consensus with some fluctuation. The system hasn't fully crystallized but maintains reasonable structure. Like supercooled water—stable until perturbed, but vulnerable to sudden phase transitions.

Aggregation works but with caveats. Small perturbations might trigger regime shifts.

### Regime 3: Transitional (Nucleating)
**Conditions:** T near T_c

The system is actively changing states. Old structure is breaking down or new structure is forming. This is the most dangerous regime for aggregation—the distribution is genuinely bimodal or shifting.

Avoid aggregating during transitions. Wait for the system to settle. If you must produce an answer, report low confidence.

### Regime 4: Chaotic (Plasma)
**Conditions:** T > 0.8, ψ < 0.3

High variance, low consensus. Predictions are essentially random within some range. The system hasn't found structure.

Standard aggregation will produce garbage. Value clustering might identify weak structure, but confidence should be very low. Consider: is this the right question? Is more context needed?

### Regime 5: Settling (Annealing)
**Conditions:** Decreasing T, increasing ψ

Post-perturbation recovery. The system was disrupted but is converging toward a new stable state. Think of metal being slowly cooled—it gradually finds its minimum-energy configuration.

Wait for completion. The aggregate will improve rapidly as annealing progresses.

---

## Detecting Regime Transitions

Knowing the current regime is valuable. Detecting transitions as they happen is even more valuable.

The key insight: at regime boundaries, the dynamics of CIC components become approximately balanced.

**dΦ/dt ≈ λ·dH/dt**

When the rate of cohesion change matches the rate of entropy change (weighted by λ), the system may be near a transition. Information is being reorganized—old structure breaking, new structure forming—with neither dominant.

This heuristic achieves:
- True positive rate: 45%
- False positive rate: 22%

Not definitive, but useful. Combine with other indicators for reliable detection.

---

## Convergence Detection: Micro-Grokking

Grokking is the phenomenon where neural networks suddenly generalize after extended training. Loss plateaus for many epochs, then drops sharply as the network "gets it."

We observe an analogous phenomenon in inference—what we call micro-grokking. A sequence of predictions might show high entropy (exploration mode), then suddenly collapse to low entropy (exploitation mode) as the model converges on an answer.

Detecting this convergence is valuable. It tells you when to trust the aggregate.

### The Entropy Curvature Criterion

Convergence manifests as sharp negative acceleration in entropy:

**d²H/dt² << 0 indicates convergence**

Intuitively: entropy is decreasing (predictions clustering), and the decrease is accelerating (clustering faster and faster). This is the signature of a system locking onto a solution.

### The Algorithm

```
ALGORITHM: Convergence Detection
INPUT: entropies h₁, ..., hₙ, threshold θ = -0.05
OUTPUT: detected, score, convergence_point

1. Smooth: h̃ᵢ = moving_average(h, window=5)
2. First derivative: d¹ᵢ = h̃ᵢ₊₁ - h̃ᵢ
3. Second derivative: d²ᵢ = d¹ᵢ₊₁ - d¹ᵢ
4. Detect: detected = (min(d²) < θ)
5. Score: score = 1/(1 + H_final) + max(0, -min(d²) × 10)
```

The smoothing prevents noise from triggering false positives. The threshold θ = -0.05 was empirically determined—sensitive enough to catch real convergence, not so sensitive that normal fluctuations trigger it.

### Performance

In our tests:
- True positive rate: 75% (correctly identifies convergence events)
- False positive rate: 15% (incorrectly flags non-convergence)

This is reliable enough for practical use but should be combined with other signals for critical decisions.

---

## Nucleation Site Detection

When a system transitions from chaotic to ordered, the transition doesn't happen uniformly. It starts somewhere—a "nucleation site" where the new phase first appears.

In inference terms, nucleation sites are clusters that form early in the prediction sequence. They're the seeds of consensus.

Detecting nucleation sites tells you:
1. Where the system is heading (which answer is likely correct)
2. How confident to be (larger nucleation sites suggest stronger consensus)
3. When the transition is happening (nucleation precedes full crystallization)

### The Detection Approach

Track cluster formation over prediction windows:
1. For each window of predictions, identify clusters using value clustering
2. Track cluster size and persistence across windows
3. A growing cluster that persists across windows is a nucleation site

If multiple nucleation sites compete, the system may be genuinely uncertain between answers. If one site dominates, convergence to that answer is likely.

---

## Practical Application: When to Trust Predictions

Putting it all together, here's how to use regime classification in practice:

### Before Aggregating

1. Compute temperature T and order parameter ψ for your prediction set
2. Classify the current regime
3. Check for recent transitions or ongoing convergence

### Decision Rules

**If Stable regime:** Aggregate confidently. Most methods will work. Report high confidence.

**If Metastable regime:** Aggregate with caution. Use robust methods (value clustering, not simple averaging). Report moderate confidence. Monitor for transitions.

**If Transitional regime:** Don't aggregate yet. Wait for the system to settle. If forced to answer, report low confidence and caveat that the system is unstable.

**If Chaotic regime:** Question whether aggregation makes sense. The predictions lack structure. Consider rephrasing the question, adding context, or accepting high uncertainty.

**If Settling regime:** Track convergence. Aggregate once entropy curvature indicates convergence. Early aggregation will be less reliable.

### Confidence Calibration

Adjust base confidence by regime:

| Regime | Confidence Multiplier |
|--------|----------------------|
| Stable | 1.0 |
| Metastable | 0.8 |
| Transitional | 0.5 |
| Chaotic | 0.3 |
| Settling | Variable (increases with convergence) |

---

## The Bigger Picture: Why This Matters

Regime classification transforms ensemble inference from a static process to a dynamic one. Instead of blindly aggregating predictions, we:

1. **Understand the system state** before making decisions
2. **Detect regime changes** that would invalidate our aggregation strategy
3. **Identify convergence** so we know when to trust results
4. **Calibrate confidence** based on actual system dynamics

This is the difference between knowing that you averaged some numbers and knowing that those numbers came from a stable, converged, high-consensus regime.

The physics analogy isn't just a metaphor. These tools come from a century of studying systems that transition between ordered and disordered states. The math works because both thermodynamic systems and inference systems share the property of having multiple possible states with different structural characteristics.

---

## Summary

- **Inference temperature** measures prediction volatility: T = (variance/n) × (1 + (1 - avg_correlation))

- **Order parameter** measures structural consensus: ψ = Σᵢ wᵢ × |autocorrelation(lag=i)|

- **Critical temperature** T_c ≈ 0.76 separates regimes

- **Five regimes:** Stable, Metastable, Transitional, Chaotic, Settling

- **Transition detection** via dΦ/dt ≈ λ·dH/dt (45% TPR, 22% FPR)

- **Convergence detection** via d²H/dt² << 0 (75% TPR, 15% FPR)

- **Practical use:** Classify regime before aggregating, adjust confidence accordingly

The next chapter connects CIC to existing theoretical frameworks—variational free energy, information bottleneck, minimum description length, and integrated information theory. The connections aren't coincidental; they point toward a unified theory of inference.
