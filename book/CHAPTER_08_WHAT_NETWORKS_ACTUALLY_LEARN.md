# Chapter 2: What Networks Actually Learn

The previous chapter explained what transformers *do*—kernel regression, graph message passing, implicit gradient descent, associative memory retrieval. But knowing the mechanism doesn't tell you what gets stored in the weights.

When you train a neural network on language, what does it learn? Not just the loss function. Not just the data distribution. What *representations* emerge? What *features* get encoded? How does 70 billion parameters organize itself to predict the next token?

This chapter explores the hidden structure of neural networks. We'll see that modern networks are far more compressed, more organized, and more surprising than their astronomical parameter counts suggest.

---

## Feature Superposition: More Features Than Dimensions

The first insight: **networks learn far more features than they have neurons to represent them.**

This seems paradoxical. A 1024-dimensional hidden layer should represent, at most, 1024 features—one per dimension. That's basic linear algebra. The rank of a matrix is at most the minimum of its dimensions.

But neural networks violate this intuition through *superposition*. They pack thousands of features into hundreds of dimensions by making features sparse and nearly orthogonal.

### The Johnson-Lindenstrauss Miracle

The mathematical foundation is the Johnson-Lindenstrauss lemma:

> You can embed n points from high-dimensional space into O(log n / ε²) dimensions while preserving pairwise distances within factor (1 ± ε).

In practical terms: 1 million features can be packed into ~1000 dimensions with only ~1% distance distortion, as long as features are sparse (not all active simultaneously).

This is why language models work despite having vocabulary sizes (50,000+) that vastly exceed their hidden dimensions (1,024–16,384). The model doesn't need a separate dimension for each token. It needs nearly-orthogonal directions for features that co-occur, and can share dimensions for features that don't.

### Superposition in Practice

Anthropic's research on superposition revealed the mechanism in detail:

1. **Features are sparse.** Most features are inactive most of the time. The feature "is_a_programming_keyword" activates on 0.01% of tokens. The feature "begins_a_question" activates on maybe 5%.

2. **Sparse features can share dimensions.** If two features are never active together, they can use the same neural dimension without interference. Like time-sharing a hotel room.

3. **Networks learn nearly-orthogonal packings.** Even when features might co-occur, networks find directions that minimize interference. The dot product between feature directions approaches zero.

4. **Superposition creates "polytope" feature geometries.** The feature directions form complex geometric structures—simplices, cross-polytopes, asymmetric configurations—that maximize packing density.

### What This Means for CIC

Superposition explains why compression metrics work for comparing neural representations:

- **High Φ (integration)** means features are well-packed—they share dimensions efficiently without interference.
- **Low Φ** means features are scattered—wasting dimensions on redundant encodings.
- **The compression distance between two representations measures how similarly they've solved the packing problem.**

When we compute NCD(repr_A, repr_B), we're measuring whether A and B found the same feature geometry. Similar geometries compress together; different geometries don't.

---

## Skip Connections: The Highway to Flat Minima

The second insight: **skip connections change what networks can learn, not just how fast they learn.**

ResNets introduced skip connections in 2015 to solve the "vanishing gradient" problem. The story usually told: gradients flow better through identity connections, enabling deeper networks.

This is true but incomplete. Skip connections do something more fundamental: they change the loss landscape geometry.

### Flat Minima and Generalization

Networks that generalize well tend to find "flat" minima—regions of parameter space where the loss doesn't change much when parameters vary. Flat minima are robust; sharp minima are brittle.

Skip connections make flat minima more accessible:

1. **Identity initialization.** A network with skip connections starts as an identity function (plus small perturbations). This initialization sits in a very flat region—doing nothing is stable.

2. **Gradual departure from identity.** Training moves the network away from identity toward useful functions. The skip connections ensure this movement happens smoothly, through flat regions.

3. **Implicit regularization.** Networks with skip connections have lower curvature in parameter space. This implicit regularization favors flat minima without explicit regularization penalties.

### The Empirical Evidence

Studies comparing networks with and without skip connections show:

- Skip-connected networks find minima with 10-100x lower Hessian eigenvalues
- These minima generalize better to held-out data
- The effect is largest in deep networks (50+ layers)

Without skip connections, very deep networks find sharp minima that overfit. With skip connections, the same depth finds flat minima that generalize.

### What This Means for CIC

The C_multi term in CIC measures multi-scale coherence. Skip connections enforce coherence by maintaining alignment between layers:

- **Each skip connection is a coherence constraint.** It says: "The output of this block should be close to its input, plus a small modification."
- **Networks with high C_multi have consistent representations across depths.** Layer 5 and layer 50 encode similar features, just refined.
- **Networks with low C_multi have incoherent depth structure.** Early layers and late layers encode unrelated features.

Skip connections are architectural implementations of the coherence principle that C_multi measures.

---

## ReLU Networks: Piecewise Linear Tessellation

The third insight: **ReLU networks partition input space into a finite number of linear regions.**

The Rectified Linear Unit—ReLU(x) = max(0, x)—is the most common activation function. It's simple, cheap to compute, and avoids vanishing gradients.

But ReLU does something surprising: it turns neural networks into piecewise linear functions.

### The Tessellation Theorem

A ReLU network partitions its input space into convex polytopes. Within each polytope, the network is perfectly linear. At boundaries between polytopes, the function has "kinks"—continuous but non-differentiable.

The number of regions grows exponentially with depth:

- 1-layer network: O(n) regions (n = neurons)
- L-layer network: O(n^L) regions

A 10-layer network with 100 neurons per layer can have up to 100^10 = 10^20 linear regions. That's more regions than atoms in a gram of matter.

### What Networks Learn is Region Placement

Training a ReLU network doesn't learn continuous functions—it learns where to place region boundaries.

The network's job is to:
1. Tessellate input space into regions
2. Assign each region a linear function (the local gradient)
3. Place boundaries so similar inputs land in similar regions

This is a discrete optimization problem disguised as continuous gradient descent.

### Implications

**Why width helps.** More neurons = more potential boundaries = finer tessellation. Wide networks can make smaller regions, capturing more detail.

**Why depth helps.** More layers = exponentially more regions. But not arbitrary regions—the composition structure constrains what tessellations are reachable.

**Why initialization matters.** Random initialization places boundaries randomly. Good initialization places boundaries where data varies most.

**Why training dynamics are chaotic.** Small parameter changes can shift boundary positions, jumping inputs between regions. This explains some of the instability in training dynamics.

### What This Means for CIC

The tessellation view explains why phase transitions occur in neural network behavior:

- **Before transition:** Boundaries are poorly placed. Many similar inputs land in different regions. High H (entropy), low Φ (integration).
- **During transition:** Boundaries reorganize. Inputs cluster into semantically coherent regions. The reorganization happens suddenly.
- **After transition:** Boundaries are well-placed. Similar inputs land together. Low H, high Φ.

Grokking—the sudden jump from memorization to generalization—is the network finding the "right" tessellation. The phase transition occurs when boundary reorganization crosses a critical threshold.

---

## The Lottery Ticket Hypothesis: Sparse Is All You Need

The fourth insight: **large networks contain small networks that would train just as well.**

The Lottery Ticket Hypothesis, proposed by Frankle and Carlin in 2018, claims:

> Randomly initialized dense networks contain subnetworks (winning tickets) that, when trained in isolation, achieve comparable performance in comparable time.

This is shocking. It means most of a large network's parameters are wasted. The winning ticket is maybe 1-10% of the original network. The rest are "losers" that don't contribute.

### Finding Winning Tickets

The original method for finding winning tickets:
1. Train the full network
2. Prune the smallest-magnitude weights (remove 20%)
3. Reset remaining weights to their *original initialization*
4. Retrain the pruned network
5. Repeat until reaching target sparsity

Networks pruned this way—to 90% sparsity—often match or exceed the original network's performance.

The key finding: **initialization matters**. If you reset pruned weights to random values, performance drops. The original random initialization contained the winning ticket's structure.

### What This Means

**Over-parameterization is search, not storage.** Large networks don't use all their parameters for representing the final function. They use extra parameters to find the right subnetwork during training.

**The winning ticket is the actual learned function.** Everything else is scaffolding—helpful for optimization but not for final prediction.

**Pruning and distillation work because winning tickets exist.** You're not destroying information when you prune; you're removing scaffolding.

### What This Means for CIC

The lottery ticket hypothesis supports the compression interpretation of intelligence:

- **The winning ticket is the compressed representation.** It's the minimal description of the learned function.
- **The full network during training is exploratory.** Like running multiple hypotheses in parallel.
- **Finding the ticket is convergence to a fixed point.** The verification operator V(P) = P when pruned weights are actually unnecessary.

CIC's Φ term measures the degree to which a representation has "found its ticket"—eliminated redundancy and concentrated information in a minimal subnetwork.

---

## Neural Scaling Laws: Why More Is Different

The fifth insight: **performance scales as a power law with compute, data, and parameters.**

The scaling laws, discovered by Kaplan et al. at OpenAI, show:

**Loss = (C/C₀)^(-α)**

where C is compute, C₀ is a constant, and α ≈ 0.05-0.07 for language models.

This means:
- 10x compute → ~15-20% loss reduction
- 100x compute → ~30-40% loss reduction
- 1000x compute → ~45-60% loss reduction

The improvement never stops (within tested range), but returns diminish as power laws do.

### The Chinchilla Insight

DeepMind's Chinchilla paper refined the scaling laws. The optimal balance between parameters N and training tokens D is:

**N ≈ D** (roughly equal investment)

Training a smaller model on more data beats training a larger model on less data. The 70B parameter Chinchilla, trained on 1.4T tokens, outperforms the 280B parameter Gopher, trained on 300B tokens.

### What Scales and What Doesn't

Scaling helps:
- Perplexity (predicting the next token)
- Factual recall
- In-context learning
- Following instructions

Scaling helps less:
- Mathematical reasoning (needs chain-of-thought)
- Multi-step planning
- Formal logic
- Causal inference

The capabilities that scale are those that benefit from more pattern storage. The capabilities that don't scale are those requiring explicit computation that current architectures don't naturally perform.

### What This Means for CIC

Scaling laws are compression laws in disguise:

- **More parameters = more compression capacity.** Larger networks can represent more complex patterns.
- **More data = more patterns to compress.** Training on more tokens provides more structure to learn.
- **Power-law scaling = logarithmic bits.** Each 10x compute adds roughly the same number of "effective bits" of world model.

The CIC functional measures how well a given representation uses its compression capacity. Optimal inference achieves the best loss for a given Φ (integration). Scaling increases the ceiling; CIC measures how close you are to the ceiling.

---

## Emergence: When Capabilities Appear Suddenly

The sixth insight: **some capabilities emerge suddenly at scale, not gradually.**

Emergent abilities are capabilities that are near-zero below some scale threshold and jump to high performance above it. Examples:

- **Arithmetic:** GPT-3 (175B) can add two 3-digit numbers. GPT-2 (1.5B) cannot.
- **Multi-step reasoning:** Only appears with chain-of-thought at 100B+ scale.
- **Code generation:** Functional code generation requires 10B+ parameters.

### Is Emergence Real or Metric Artifact?

Recent work suggests some "emergence" is measurement artifact:

- Binary metrics (correct/incorrect) show sharp transitions
- Continuous metrics (partial credit) show gradual improvement
- The underlying capability may improve smoothly; the metric transitions sharply

But not all emergence is artifact. Some capabilities—like in-context learning itself—appear to require architectural and scale thresholds that can't be explained by metric choice alone.

### The Grokking Phenomenon

Grokking is emergence during training, not during scaling:

1. Network memorizes training data (100% train accuracy, 0% test accuracy)
2. Training continues (loss already ~0)
3. Suddenly—after 10x-100x more training—test accuracy jumps
4. The network has "understood" the underlying rule

Grokking reveals that memorization and generalization are distinct phases separated by a transition. The transition occurs when the network finds the simple underlying structure after exhausting capacity for memorization.

### What This Means for CIC

Emergence and grokking are phase transitions in inference:

- **Before transition:** H is high (many possible representations), Φ is low (no integration), C_multi is low (incoherent structure)
- **During transition:** H drops rapidly, Φ spikes, C_multi increases
- **After transition:** New equilibrium with low H, high Φ, high C_multi

The CIC functional predicts when transitions will occur:

**dΦ/dH ≈ λ** is the critical condition

When the rate of integration gain equals the rate of entropy loss (scaled by λ), the system crosses the transition. Before this point, entropy dominates. After, integration dominates.

---

## Summary: The Learned Structure

Neural networks learn:

1. **Superposed features** — Thousands of features packed into hundreds of dimensions via sparse, nearly-orthogonal encodings

2. **Flat minima** — Skip connections guide training toward robust, generalizing solutions

3. **Tessellated input space** — ReLU networks partition inputs into exponentially many linear regions

4. **Sparse winning tickets** — The actual function is a small subnetwork; the rest is optimization scaffolding

5. **Scale-dependent capabilities** — Performance follows power laws, with some capabilities emerging suddenly

6. **Phase-separated dynamics** — Memorization and generalization are distinct phases separated by transitions

Understanding what networks learn—not just how they're trained—is essential for predicting when they'll succeed and when they'll fail.

The CIC framework provides a quantitative language for these phenomena:
- **Φ measures integration** — how well features are packed
- **H measures uncertainty** — which phase the system is in
- **C_multi measures coherence** — alignment across scales

In the next chapter, we'll examine the training dynamics that produce these structures—why gradient descent finds flat minima, why Adam approximates natural gradient, and why the loss landscape shapes what networks can learn.
