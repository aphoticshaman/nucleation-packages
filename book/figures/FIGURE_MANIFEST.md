# Book Figure Manifest

*Complete list of figures with file names and chapter placement*

---

## Figure Naming Convention

```
fig_{chapter}_{number}.png

Examples:
fig_01_01.png - Chapter 1, Figure 1
fig_05_03.png - Chapter 5, Figure 3
fig_A1_02.png - Appendix A, Figure 2
```

---

## Part 0: Beginner's Guide

### fig_00_01.png — Traditional vs ML Programming
**Location:** Chapter 0.1, after "The Longer Answer"
**Caption:** Fig. 0.1: Traditional programming uses explicit rules; machine learning learns patterns from data.

### fig_00_02.png — The Three Flavors of ML
**Location:** Chapter 0.2, section header
**Caption:** Fig. 0.2: Supervised, unsupervised, and reinforcement learning paradigms.

### fig_00_03.png — Learning Loop Diagram
**Location:** Chapter 0.2, "How Learning Actually Works"
**Caption:** Fig. 0.3: The machine learning feedback loop: predict, compare, adjust, repeat.

---

## Part 0.5: Neural Network Foundations

### fig_05_01.png — Biological vs Artificial Neuron
**Location:** Chapter 0.5.1, "The McCulloch-Pitts Neuron"
**Caption:** Fig. 0.5.1: Biological neuron (left) and its mathematical model (right).

### fig_05_02.png — Perceptron Diagram
**Location:** Chapter 0.5.1, "The Perceptron"
**Caption:** Fig. 0.5.2: The perceptron: weighted inputs, summation, threshold activation.

### fig_05_03.png — XOR Problem Visualization
**Location:** Chapter 0.5.1, "The Perceptron's Fatal Flaw"
**Caption:** Fig. 0.5.3: The XOR problem—no single line can separate the classes.

### fig_05_04.png — Hidden Layer Solution
**Location:** Chapter 0.5.1, "The Hidden Layer Solution"
**Caption:** Fig. 0.5.4: Hidden layers enable learning non-linear decision boundaries.

### fig_05_05.png — Backpropagation Flow
**Location:** Chapter 0.5.1, "Backpropagation"
**Caption:** Fig. 0.5.5: Error propagates backward, gradients update weights.

### fig_05_06.png — Activation Functions
**Location:** Chapter 0.5.2, "Solutions That Enabled Deep Learning"
**Caption:** Fig. 0.5.6: Common activation functions: Sigmoid, Tanh, ReLU, GELU.

### fig_05_07.png — Residual Connection
**Location:** Chapter 0.5.2, "Residual Connections"
**Caption:** Fig. 0.5.7: Residual (skip) connections allow gradients to flow directly.

### fig_05_08.png — Feature Hierarchy CNN
**Location:** Chapter 0.5.4, "Convolutional Neural Networks"
**Caption:** Fig. 0.5.8: CNN feature hierarchy: edges → textures → parts → objects.

### fig_05_09.png — RNN Unrolled
**Location:** Chapter 0.5.4, "Recurrent Neural Networks"
**Caption:** Fig. 0.5.9: RNN unrolled through time, showing recurrent connections.

### fig_05_10.png — Transformer Overview
**Location:** Chapter 0.5.4, "Transformers"
**Caption:** Fig. 0.5.10: The transformer architecture: self-attention enables parallel processing.

### fig_05_11.png — Word Embedding Space
**Location:** Chapter 0.5.6, "Word Embeddings"
**Caption:** Fig. 0.5.11: Word vectors capture semantic relationships (king - man + woman ≈ queen).

### fig_05_12.png — Encoder-Decoder Architecture
**Location:** Chapter 0.5.6, "Sequence-to-Sequence"
**Caption:** Fig. 0.5.12: Encoder-decoder architecture for sequence-to-sequence tasks.

---

## Part I: Using LLMs

### fig_01_01.png — The Parrot Metaphor
**Location:** Chapter 1, "The World's Most Impressive Parrot"
**Caption:** Fig. 1.1: The parrot doesn't understand—it predicts what comes next.

### fig_01_02.png — Parameter Scale Visualization
**Location:** Chapter 1, "The 'Large' in Large Language Model"
**Caption:** Fig. 1.2: Model size progression from GPT-1 to GPT-4.

### fig_01_03.png — Emergence Diagram
**Location:** Chapter 1, section on emergence
**Caption:** Fig. 1.3: Emergent capabilities appear at scale thresholds.

### fig_01_04.png — Training Process Overview
**Location:** Chapter 1, "How Our Parrot Learns"
**Caption:** Fig. 1.4: The training loop: input → predict → compare → adjust.

### fig_02_01.png — Prompt as Initial Conditions
**Location:** Chapter 2, "Why This Works"
**Caption:** Fig. 2.1: Your prompt sets the initial conditions; the completion is the trajectory.

### fig_02_02.png — Few-Shot vs Zero-Shot
**Location:** Chapter 2, "Use Examples (Few-Shot Prompting)"
**Caption:** Fig. 2.2: Zero-shot, one-shot, and few-shot prompting comparison.

### fig_02_03.png — Chain of Thought
**Location:** Chapter 2, "Chain-of-Thought Prompt"
**Caption:** Fig. 2.3: Chain-of-thought prompting improves reasoning accuracy.

---

## Part II: Understanding LLMs

### fig_07_01.png — Attention Mechanism
**Location:** Chapter 7, core section
**Caption:** Fig. 7.1: The attention mechanism: Query, Key, Value computation.

### fig_07_02.png — Attention Weights Heatmap
**Location:** Chapter 7, visualization section
**Caption:** Fig. 7.2: Attention weights showing which tokens attend to which.

### fig_07_03.png — Multi-Head Attention
**Location:** Chapter 7, "Multi-Head Attention"
**Caption:** Fig. 7.3: Multiple attention heads capture different relationship types.

### fig_07_04.png — Scaled Dot-Product Attention
**Location:** Chapter 7, mathematical section
**Caption:** Fig. 7.4: Scaled dot-product attention: Attention(Q,K,V) = softmax(QKᵀ/√d)V.

### fig_08_01.png — Transformer Block
**Location:** Chapter 8, architecture section
**Caption:** Fig. 8.1: A transformer block: attention, normalization, feedforward.

### fig_08_02.png — Positional Encoding
**Location:** Chapter 8, "Positional Encoding"
**Caption:** Fig. 8.2: Sinusoidal positional encoding for sequence position.

### fig_08_03.png — Full Transformer Architecture
**Location:** Chapter 8, overview section
**Caption:** Fig. 8.3: Complete transformer architecture (encoder-decoder).

### fig_09_01.png — Loss Landscape
**Location:** Chapter 9, optimization section
**Caption:** Fig. 9.1: The loss landscape—gradient descent seeks the minimum.

### fig_09_02.png — Training Curves
**Location:** Chapter 9, training dynamics
**Caption:** Fig. 9.2: Training vs validation loss over epochs.

### fig_09_03.png — Learning Rate Schedules
**Location:** Chapter 9, hyperparameters
**Caption:** Fig. 9.3: Common learning rate schedules: constant, decay, warmup.

---

## Part III: The CIC Framework

### fig_10_01.png — CIC Functional Diagram
**Location:** Chapter 10, introduction
**Caption:** Fig. 10.1: The CIC functional: F[T] = Φ(T) - λH(T|X) + γC_multi(T).

### fig_10_02.png — Information Cohesion
**Location:** Chapter 10, "Information Cohesion"
**Caption:** Fig. 10.2: Information cohesion measures mutual compression among samples.

### fig_10_03.png — Compression Distances
**Location:** Chapter 10, compression section
**Caption:** Fig. 10.3: Normalized Compression Distance (NCD) similarity matrix.

### fig_11_01.png — Multi-Scale Coherence
**Location:** Chapter 11, coherence section
**Caption:** Fig. 11.1: Multi-scale coherence: exact consensus, cluster, range.

### fig_11_02.png — Confidence Calibration
**Location:** Chapter 11, calibration
**Caption:** Fig. 11.2: Confidence calibration: predicted probability vs actual frequency.

### fig_12_01.png — Value Clustering
**Location:** Chapter 12, algorithm section
**Caption:** Fig. 12.1: Value clustering aggregation: outliers identified and downweighted.

### fig_12_02.png — Phase Detection
**Location:** Chapter 12, phases
**Caption:** Fig. 12.2: Phase detection: exploration, transition, exploitation.

---

## Part IV: Applications

### fig_16_01.png — Anomaly Detection Pipeline
**Location:** Chapter 16, architecture
**Caption:** Fig. 16.1: Real-time anomaly detection pipeline with CIC scoring.

### fig_17_01.png — Cascade Failure Diagram
**Location:** Chapter 17, cascade prediction
**Caption:** Fig. 17.1: Cascade failure propagation in networked systems.

### fig_18_01.png — Bayesian Optimization
**Location:** Chapter 18, optimization
**Caption:** Fig. 18.1: Bayesian optimization acquisition function.

### fig_19_01.png — Sensor Fusion Architecture
**Location:** Chapter 19, fusion
**Caption:** Fig. 19.1: Multi-sensor fusion with uncertainty quantification.

### fig_20_01.png — Uncertainty Representation
**Location:** Chapter 20, uncertainty
**Caption:** Fig. 20.1: Uncertainty representation: point estimate vs distribution.

### fig_21_01.png — Knowledge Quadrants
**Location:** Chapter 21, epistemic humility
**Caption:** Fig. 21.1: The knowledge quadrants: known knowns, known unknowns, etc.

### fig_21_02.png — Confidence Decay
**Location:** Chapter 21, temporal decay
**Caption:** Fig. 21.2: Confidence decay over time for different information types.

### fig_22_01.png — Wavelet Decomposition
**Location:** Chapter 22, wavelets
**Caption:** Fig. 22.1: Wavelet decomposition: signal separated by scale.

### fig_22_02.png — Multi-Resolution Analysis
**Location:** Chapter 22, multi-resolution
**Caption:** Fig. 22.2: Multi-resolution analysis revealing structure at different scales.

---

## Part V: Future & Doctrine

### fig_23_01.png — MDMP Flow
**Location:** Chapter 23, MDMP section
**Caption:** Fig. 23.1: Military Decision-Making Process adapted for AI planning.

### fig_23_02.png — Battle Rhythm
**Location:** Chapter 23, battle rhythm
**Caption:** Fig. 23.2: AI development battle rhythm: daily, weekly, monthly cycles.

### fig_24_01.png — Self-Hosted Architecture
**Location:** Chapter 24, architecture
**Caption:** Fig. 24.1: Serverless inference architecture: Next.js + WASM + Rust.

### fig_24_02.png — Model Quantization Trade-offs
**Location:** Chapter 24, quantization
**Caption:** Fig. 24.2: Quantization trade-offs: size vs quality for different bit depths.

### fig_24_03.png — Cost Comparison
**Location:** Chapter 24, cost analysis
**Caption:** Fig. 24.3: Self-hosted vs API cost comparison at various scales.

---

## Appendices

### fig_A1_01.png — Measure Theory Diagram
**Location:** Appendix A, measure theory
**Caption:** Fig. A.1: σ-algebra and probability measure visualization.

### fig_A1_02.png — Kolmogorov Complexity
**Location:** Appendix A, complexity
**Caption:** Fig. A.2: Kolmogorov complexity and incompressibility.

### fig_A1_03.png — NCD Proof Diagram
**Location:** Appendix A, NCD section
**Caption:** Fig. A.3: NCD metric properties: symmetry, triangle inequality.

### fig_C_01.png — AI Timeline
**Location:** Appendix C, main content
**Caption:** Fig. C.1: Key events in AI history, 1943-2025.

### fig_C_02.png — AI Winters
**Location:** Appendix C, winters section
**Caption:** Fig. C.2: AI funding cycles: booms and winters.

---

## Cover Assets

### cover_design_01.png — Neural Cosmos
### cover_design_02.png — Attention Mechanism
### cover_design_03.png — The Gradient
### cover_design_04.png — Emergence
### cover_design_05.png — The Equation
### cover_design_06.png — The Layers
### cover_design_07.png — Binary Roots

---

## Total Count

- Part 0: 3 figures
- Part 0.5: 12 figures
- Part I: 7 figures
- Part II: 9 figures
- Part III: 6 figures
- Part IV: 9 figures
- Part V: 6 figures
- Appendices: 5 figures
- Covers: 7 designs

**Total: 64 figures + 7 cover designs = 71 visual assets**

---

*All figures generated at 300 DPI, PNG format, RGB color mode.*
*Maximum dimensions: 2550 × 3300 pixels (8.5" × 11" at 300 DPI).*
