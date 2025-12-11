# Chapter 6: The Problem with Simple Aggregation

Part I explained what transformers are (attention mechanisms), what they learn (superposed features, tessellated spaces), and how they learn it (implicit regularization, adaptive geometry, phase transitions). We now understand that neural networks are remarkably sophisticated—implicit kernel regression, Hopfield memory retrieval, gradient descent in the forward pass.

But all that sophistication doesn't prevent them from being wrong.

Ask GPT-4 to multiply 847 × 23, and it might return 19,481 (correct). Ask it again with slightly different prompting, and you might get 19,520. Ask it 100 times across different temperatures and contexts, and you'll get a distribution of answers—some close to correct, some wildly off, all delivered with equal confidence.

This is the inference problem. Given multiple predictions from the same model (or multiple models), how do you combine them into a single reliable answer?

The traditional solutions—majority voting and simple averaging—are surprisingly inadequate. Understanding why they fail reveals the structure of the problem and motivates the CIC framework we'll develop in the next chapter.

---

## 6.1 Why Majority Voting Fails

Majority voting is the simplest aggregation method: count how many times each answer appears, select the most common one.

For classification problems with clean categories (Is this email spam? Is this image a cat?), majority voting works reasonably well. If 70 out of 100 classifiers say "spam," there's a good chance the email is spam. The wisdom of crowds, and all that.

But for anything more complex, majority voting breaks down catastrophically.

### The Exact Match Problem

Consider this scenario: an LLM solves a 10-step math problem. At step 7, it makes a small arithmetic error—carrying a 1 when it should have carried a 0. The rest of the computation proceeds correctly given that error.

The final answer is wrong, but it's *structurally similar* to the correct answer. It's an off-by-one error propagated through three more operations. The answer isn't random—it's a coherent wrong answer that reveals the model almost got it right.

Now suppose we generate 100 samples:
- 35 samples get exactly 19,481 (correct)
- 28 samples get 19,520 (small arithmetic error)
- 15 samples get 19,450 (different small error)
- 12 samples get 19,475 (another variant)
- 10 samples are wildly different (18,200, 21,000, etc.)

What does majority voting return? **19,481 with 35% confidence.** That happens to be correct, but the confidence is misleadingly low. The actual signal is much stronger—65% of samples are within 0.5% of the correct answer.

But what if the distribution shifts slightly?
- 32 samples get 19,520
- 30 samples get 19,481 (correct)
- 18 samples get 19,450
- 20 samples are scattered

Now majority voting returns **19,520—the wrong answer.** Even though 68% of samples cluster around the correct value, the exact-match requirement obscures this.

### The Continuous Variable Problem

Majority voting assumes discrete categories. It counts exact matches.

For continuous variables (numbers, coordinates, measurements), exact matches are rare. Two samples of π might be 3.14159 and 3.14160—close enough to be essentially identical, but not exact matches. Majority voting treats them as completely different answers.

This makes majority voting essentially useless for numeric inference. If you generate 100 predictions of a quantity and none of them match exactly, majority voting returns a random one with 1% confidence.

### The Information Destruction Problem

The deepest failure of majority voting is information destruction.

Consider these two scenarios:

**Scenario A:** 100 samples spread uniformly across [0, 100]
- Majority vote: random selection, 1% confidence
- Actual information: complete uncertainty

**Scenario B:** 95 samples cluster between [47, 53], 5 samples are at [0, 100, 200, 300, 400]
- Majority vote: random selection from cluster, ~1% confidence
- Actual information: high confidence answer near 50

Majority voting gives the same result in both scenarios. It can't distinguish genuine uncertainty from artificial dispersion created by a few outliers. The structure of the answer distribution—the shape of the cloud of predictions—is thrown away.

This is a cardinal sin in information theory. The answer distribution *is* information. It tells you not just what the model thinks but how certain it is, what alternative hypotheses exist, and what failure modes are active. Discarding this structure is like measuring a bridge's deflection with a yes/no gauge.

---

## 6.2 Why Simple Averaging Fails

If majority voting fails because it ignores continuous structure, why not just average all the predictions?

Simple averaging has a long and venerable history. The Galton observation—that the average of 800 guesses at an ox's weight was remarkably close to the true weight—launched the field of collective intelligence. Surely averaging large language model outputs should work similarly?

No. And the reasons why reveal something important about the nature of LLM errors.

### Outliers Dominate

Simple averaging computes the arithmetic mean:

**mean = (x₁ + x₂ + ... + xₙ) / n**

The problem is that every sample contributes equally. An answer of 19,481 counts the same as an answer of 1,000,000.

Consider 100 predictions:
- 90 samples cluster around 19,481 (±50)
- 10 samples are wildly wrong: {100, 200, 50000, 80000, 100000, 150000, 200000, 300000, 400000, 500000}

The cluster mean is ~19,481. The outlier sum is 1,580,300.

Simple average = (90 × 19,481 + 1,580,300) / 100 = (1,753,290 + 1,580,300) / 100 = 33,336

The average is 33,336—almost double the correct answer—even though 90% of predictions were essentially correct. Ten bad samples have completely destroyed the signal from ninety good ones.

This isn't a contrived example. LLMs regularly produce these kinds of outliers. Ask a language model for a numerical estimate, and occasionally it will return something nonsensical—a number with too many digits, a confusion between units, a complete misparse of the question. These failure modes create heavy-tailed distributions that make averaging catastrophically unreliable.

### The Bias Toward Errors

There's an asymmetry in how errors affect averages.

When an LLM makes a correct prediction, it tends to be close to the true value. Small variations come from rounding, tokenization artifacts, and minor attention fluctuations. The correct answers cluster tightly.

When an LLM makes an error, it can be wrong by any amount. Errors aren't bounded. A wrong computation can produce a result 2x too large, 100x too large, or 10,000x too large. The wrong answers spread widely.

This asymmetry means errors contribute more variance to the average than correct predictions. Even if 95% of samples are correct, the 5% that are wrong can dominate the mean if they're wrong enough.

Mathematically: if X is correct and Y is wrong, then |Y - true| >> |X - true| typically. The average of X and Y is pulled toward Y more than toward X.

### Robust Statistics Aren't Enough

The standard response is robust statistics: use the median instead of the mean, or use trimmed means that discard outliers, or use Huber estimators that downweight extreme values.

These help. But they don't solve the fundamental problem.

**Median:** Better than mean for heavy-tailed distributions. But the median assumes unimodal data. If there are two competing answers—one correct, one consistently wrong—the median doesn't know which cluster to trust. With 40 samples near 19,481 and 50 samples near 19,520, the median is ~19,500. Still wrong.

**Trimmed Mean:** Removes some fraction of extreme values before averaging. But how do you know how much to trim? Trim too little and outliers dominate. Trim too much and you throw away valid data. The optimal trim percentage depends on the unknown error rate.

**Huber Estimator:** Applies linear loss near the center, quadratic loss in the tails. A principled tradeoff between mean and median. But it still assumes a single true value surrounded by noise. It can't handle multimodal error distributions where the model has learned two different wrong approaches.

All robust statistics share a fatal assumption: **errors are random noise around a central value.**

But LLM errors aren't random. They're *structured*. An arithmetic mistake at step 7 produces a coherent wrong answer. A sign error produces a different coherent wrong answer. A unit confusion produces yet another. The errors cluster in value space because they come from identifiable failure modes.

Robust statistics see outliers. CIC sees structure.

---

## 6.3 The Jellybean Jar Problem

The classic demonstration of collective intelligence is the jellybean jar. Put jellybeans in a jar, ask many people to guess the count, average their guesses. The average is usually remarkably close to the true count.

This works because:
1. Individual errors are roughly symmetric (underestimates balance overestimates)
2. Individual errors are roughly independent (one person's bias doesn't correlate with another's)
3. The true answer is within the distribution of guesses

None of these hold for LLM inference.

### LLM Errors Aren't Symmetric

Human guessers don't know the answer and are basically doing visual estimation. Their errors come from perceptual limitations and are roughly centered on the truth.

LLMs do computation. When they get it right, they get it very right. When they get it wrong, they fail in specific ways that tend to produce too-large or too-small answers depending on the failure mode.

A model that forgets to carry might systematically underestimate. A model that double-counts might systematically overestimate. The errors aren't symmetric around the truth—they're biased in the direction of the failure mode.

This breaks the jellybean average. If half your guessers systematically overestimate by 50% and half underestimate by 10%, the average isn't close to the truth—it's 20% too high.

### LLM Errors Aren't Independent

The most insidious problem with LLM ensemble inference is error correlation.

When you generate 100 samples from the same model with the same prompt, you're not getting 100 independent opinions. You're getting 100 samples from the same underlying distribution. If that distribution has a systematic bias, all 100 samples share that bias.

Worse: the specific way a model fails is often consistent across samples. If the model has learned an incorrect algorithm for a certain type of problem, it will apply that algorithm reliably across samples. Temperature variation changes the noise but not the underlying pattern.

This is why averaging LLM outputs gives worse results than averaging human opinions. Humans fail in diverse ways—one person overestimates, another underestimates, another gets the units wrong. LLMs fail in consistent ways—all samples from a given model share the same computational errors.

Independence is the core assumption behind the wisdom of crowds. Without it, aggregation doesn't help. If all your advisors got their information from the same source and processed it the same way, having more of them doesn't make you smarter.

### The True Answer Might Not Be In the Distribution

Perhaps the strangest failure mode: sometimes the correct answer doesn't appear at all.

If an LLM has fundamentally misunderstood a problem—parsed the question wrong, missed a crucial constraint, applied the wrong domain knowledge—then all of its samples will be wrong in the same direction. The correct answer isn't somewhere in the middle of the distribution. It's outside the distribution entirely.

No amount of averaging, voting, or robust estimation can recover an answer that isn't represented. If all 100 samples are computing the wrong thing, the aggregate is still wrong.

This is where the jellybean analogy breaks down completely. Jellybean guessers are looking at the actual jar. They might guess wrong, but their errors are bounded by the physical reality they're perceiving. LLMs are running internal computations that might have no relationship to ground truth. Their distribution of outputs can be entirely disconnected from the correct answer.

---

## 6.4 What We Actually Need

The failures of majority voting and simple averaging point toward what a proper aggregation method needs.

### Need 1: Structure Awareness

The distribution of predictions has structure. Some answers cluster together. Some stand alone. The clusters have different sizes, different tightnesses, different relationships to each other.

A good aggregation method must perceive this structure, not ignore it. It should identify clusters, characterize their properties, and use those properties to decide which cluster represents the correct answer.

This isn't just about removing outliers. It's about understanding the *topology* of the answer space—where the density is, where the gaps are, what the shape of uncertainty looks like.

### Need 2: Compression-Based Similarity

Two predictions are "similar" if they encode similar underlying algorithms.

This is deeper than numeric proximity. The predictions 19,481 and 19,520 might be 39 units apart numerically, but if they were produced by nearly-identical computation with a single-bit difference, they're algorithmically similar.

Compression distance captures this. Two strings that compress well together share algorithmic structure. The Normalized Compression Distance:

**NCD(x, y) = [C(xy) - min(C(x), C(y))] / max(C(x), C(y))**

measures how much new information y adds given x. If two predictions are algorithmically similar—produced by the same process with small variations—they'll compress well together. If they're algorithmically different—produced by completely different processes—they won't.

This is the right similarity metric for ensemble inference. Two wrong answers that came from the same computational error should be grouped together. Two answers that happen to be numerically close but came from different computations should be kept separate.

### Need 3: Uncertainty Quantification

A good aggregation method should know when it doesn't know.

If the prediction distribution is tightly clustered around a single value, confidence should be high. If it's spread across multiple competing values with no clear winner, confidence should be low.

This uncertainty quantification should be calibrated. A 90% confidence prediction should be right 90% of the time. Over-confident predictions are dangerous; under-confident predictions waste resources by calling for unnecessary verification.

Entropy provides the natural measure. High entropy = high uncertainty = many competing answers. Low entropy = low uncertainty = consensus. The rate of entropy decrease over samples tells you whether the system is converging or still exploring.

### Need 4: Coherence Across Scales

The best prediction should be coherent at multiple levels of analysis.

At the finest scale: individual predictions should cluster together.
At the medium scale: clusters should have consistent internal structure.
At the coarsest scale: the overall answer should fit with prior knowledge and constraints.

A prediction that looks good at one scale but falls apart at another is suspicious. The model might have gotten lucky, or might have exploited a pattern that doesn't generalize.

Multi-scale coherence is a robustness check. If the same answer emerges whether you look at individual samples, local clusters, or the global distribution, it's more likely to be correct than an answer that only appears at one scale.

### Need 5: Dynamic Adaptation

The aggregation method should adapt to what it observes.

Early in sampling, uncertainty is high and exploration is appropriate. Late in sampling, patterns have emerged and exploitation is appropriate. The method should know which regime it's in and adjust accordingly.

Similarly, the method should detect when conditions change. If a system that was producing consistent answers suddenly starts producing scattered ones, something has changed—the input, the model state, the problem difficulty. The method should detect this and recalibrate.

Phase transitions in physics provide the model. Systems move between ordered and disordered states as control parameters change. Inference systems do the same—from exploration to exploitation, from uncertainty to confidence, from chaos to crystallization. Detecting and navigating these transitions is crucial for robust aggregation.

---

## Summary: The Gap and the Solution

We started with a simple question: given multiple predictions, how do you combine them?

The obvious answers—majority voting and simple averaging—fail because they ignore structure. Majority voting throws away continuous information. Simple averaging lets outliers dominate. Neither perceives the clusters, shapes, and patterns that contain the real signal.

What we need is:
1. **Structure awareness** — See the topology of predictions
2. **Compression-based similarity** — Group by algorithmic relationship, not just numeric proximity
3. **Uncertainty quantification** — Know when you know and when you don't
4. **Multi-scale coherence** — Check that answers work at all levels
5. **Dynamic adaptation** — Respond to changing conditions

The CIC functional provides exactly this framework. The Φ term measures compression-based structure. The H term tracks uncertainty. The C_multi term ensures coherence. The overall functional F[T] = Φ(T) - λH(T|X) + γC_multi(T) provides a principled objective for selecting among competing predictions.

The next chapter develops CIC in full. But first, understand what it replaces: naive methods that treat predictions as independent samples from a simple distribution. Predictions from modern neural networks are not independent. They're not simple. They carry structure that encodes algorithmic relationships, failure modes, and confidence levels.

Ignoring that structure throws away most of the information. CIC preserves it.

---

## Mathematical Preview

Before diving into the full CIC framework, here's a preview of the key quantities we'll define:

**Information Cohesion (Φ):**

$$\Phi = 1 - \frac{1}{n(n-1)} \sum_{i < j} \text{NCD}(s_i, s_j)$$

High Φ means predictions share algorithmic structure. Low Φ means they're informationally independent.

**Representation Entropy (H):**

$$H = \min(1, \text{Var}(\{s_i / |\bar{s}|\}))$$

High H means high uncertainty. Low H means crystallized consensus.

**Multi-Scale Coherence (C_multi):**

$$C_{multi} = w_1 C_1 + w_2 C_2 + w_3 C_3$$

where C₁ measures exact consensus, C₂ measures cluster coherence, C₃ measures range constraint.

**The CIC Functional:**

$$F[T] = \Phi(T) - \lambda H(T|X) + \gamma C_{multi}(T)$$

where λ = 0.5 and γ = 0.3 are empirically-determined weights.

The functional F balances three imperatives:
- Maximize shared structure (Φ ↑)
- Minimize uncertainty (H ↓)
- Maintain coherence across scales (C_multi ↑)

The optimal prediction is the one that maximizes F—the one that best satisfies all three imperatives simultaneously.

This structure parallels the variational free energy from physics and neuroscience:

**F_var = -Accuracy + Complexity**

CIC recapitulates this as:

**-F_CIC = -Structure + Uncertainty - Coherence**

The formal connections—and the empirical validation—come next. But the intuition should now be clear: we need a framework that perceives and preserves structure, not one that averages it away.

The transition from Part I to Part II is complete. We understand how LLMs work; now we understand why simple aggregation doesn't work. The stage is set for CIC.
