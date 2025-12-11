# Part 0.5: Neural Network Foundations

*From biological neurons to artificial intelligence*

---

## Chapter 0.5.1: The Brain That Inspired the Machine

### How You're Reading This Right Now

As you read this sentence, roughly 86 billion neurons in your brain are firing in coordinated patterns. Each neuron is a tiny biological cell, shaped somewhat like a tree—with branches (dendrites) that receive signals and a long trunk (axon) that sends signals to other neurons.

When a neuron receives enough incoming signals, it "fires"—sending an electrical impulse down its axon to potentially thousands of other neurons. Those neurons might then fire, or might not, depending on how much input they receive.

This is all your thoughts are: patterns of firing neurons.

Reading. Thinking. Deciding. Remembering. Creating. All of it is neurons passing signals to each other.

### The McCulloch-Pitts Neuron (1943)

In 1943, Warren McCulloch (a neuroscientist) and Walter Pitts (a logician) asked a dangerous question: Could we build a mathematical model of a neuron?

Their answer was the first artificial neuron:

```
Inputs:    x₁, x₂, x₃, ... (signals from other neurons)
Weights:   w₁, w₂, w₃, ... (how much each input matters)
Threshold: θ (how much total input is needed to fire)

Total input = w₁×x₁ + w₂×x₂ + w₃×x₃ + ...

If total input ≥ θ:
    Output = 1 (fire!)
Else:
    Output = 0 (stay quiet)
```

This is shockingly simple. And it works.

### Why Weights Matter

Think about how you decide what to eat for lunch. Multiple inputs affect your decision:

- What's in the fridge? (input x₁)
- Are you hungry? (input x₂)
- Do you have time to cook? (input x₃)
- What did you eat yesterday? (input x₄)

But these inputs don't matter equally. If you're starving (hunger = 1), that might override everything else. If you have a business lunch (time = 0), that forces the decision regardless of what's in the fridge.

Weights capture this: some inputs matter more than others.

The McCulloch-Pitts insight was that biological neurons work the same way. Some synaptic connections are stronger (higher weight), some are weaker. The brain adjusts these weights through experience—that's what learning is.

### The Perceptron (1958)

Frank Rosenblatt took the McCulloch-Pitts neuron and asked: Can we make a neuron that learns?

His Perceptron could be trained. Show it examples, tell it when it's wrong, and it adjusts its own weights to do better next time.

```
PERCEPTRON LEARNING ALGORITHM:

1. Start with random weights
2. For each training example:
   a. Calculate output
   b. If output is correct, do nothing
   c. If output is wrong:
      - If should have fired but didn't: increase weights
      - If fired but shouldn't have: decrease weights
3. Repeat until all examples are correct (or good enough)
```

This was revolutionary. A machine that programs itself!

The 1950s and 1960s were filled with wild optimism. Headlines proclaimed thinking machines were just around the corner. The New York Times wrote that the Perceptron was "the embryo of an electronic computer that [the Navy] expects will be able to walk, talk, see, write, reproduce itself and be conscious of its existence."

(Spoiler: It couldn't do any of that.)

### The Perceptron's Fatal Flaw

In 1969, Marvin Minsky and Seymour Papert published "Perceptrons," proving mathematically that a single perceptron could not learn certain simple patterns—most famously, the XOR function.

XOR ("exclusive or") outputs 1 if exactly one input is 1:

```
Input A | Input B | XOR Output
   0    |    0    |     0
   0    |    1    |     1
   1    |    0    |     1
   1    |    1    |     0
```

Try to draw a single straight line that separates the 1s from the 0s:

```
        B
        ↑
    1   ●   ○   1

    0   ○   ●   1
        └───────→ A
            0   1
```

You can't. The points aren't linearly separable. A single perceptron can only learn linearly separable patterns.

This killed neural network research for nearly two decades. The "AI Winter" had begun.

### The Hidden Layer Solution

The solution to XOR was known even in 1969, but no one knew how to train it: add more layers.

A single perceptron is like asking: "Can I draw ONE line to separate these points?"

Multiple layers ask: "Can I draw MULTIPLE lines and combine them?"

```
Layer 1: Draw two lines
    - Line 1: Separates (0,0) from (0,1) and (1,1)
    - Line 2: Separates (0,0) from (1,0) and (1,1)

Layer 2: Combine the lines
    - Fire if EITHER Line 1 OR Line 2 fired, but NOT BOTH
    - This is exactly XOR!
```

```
         ┌─────────────────────────────────────┐
Input A ─┤                                     │
         │  Hidden      Output                 │
Input B ─┤  Layer       Layer                  │
         │   ○ ─────────╲                      │
         │  ╱ ╲           ╲                    │
         │ ╱   ╲           ○──── XOR Output    │
         │╱     ╲         ╱                    │
         │       ○ ──────╱                     │
         └─────────────────────────────────────┘
```

The middle layer is called a "hidden layer" because it's not directly visible—you don't see its inputs or outputs, only the final result.

The problem was: how do you train the hidden layer? You know what output you want, but you don't know what the hidden layer should do to produce it.

### Backpropagation (1986)

The breakthrough came from asking: "How wrong is each weight?"

If the final output is wrong, some weights are more responsible than others. Backpropagation figures out each weight's share of the blame and adjusts accordingly.

```
BACKPROPAGATION (conceptually):

1. Forward pass: Compute the output
2. Calculate error: How far off was the output?
3. Backward pass: Trace the error back through the network
   - Each weight gets "blamed" proportionally
   - Weights that contributed more to the error get adjusted more
4. Update weights: Nudge each weight to reduce its contribution to error
5. Repeat
```

The mathematical trick is the chain rule from calculus. If you know how the error changes with respect to the output, and how the output changes with respect to each weight, you can compute how the error changes with respect to each weight.

This was the missing piece. Now we could train networks with hidden layers.

---

## Chapter 0.5.2: From Perceptrons to Deep Learning

### What "Deep" Means

"Deep learning" just means neural networks with many layers. How many? No strict definition, but typically:

- 1-2 layers: Shallow network
- 3-10 layers: Moderately deep
- 10-100 layers: Deep
- 100+ layers: Very deep (ResNets can have 1,000+)

Why does depth matter?

### The Hierarchy of Features

Imagine teaching a network to recognize faces. With one layer, you can detect simple patterns—lines, edges, blobs.

With two layers, the second layer can combine first-layer features: "edge + edge + edge arranged this way = eye"

With three layers: "eye + eye + nose + mouth arranged this way = face"

```
Layer 1: Edges
    ─    │    /    \    ○

Layer 2: Parts
    👁    👃    👄

Layer 3: Concepts
    😀    🙂    😐
```

Each layer builds more abstract features from simpler ones. This is called a **feature hierarchy**.

Deep networks can learn complex hierarchies automatically. You don't tell them "look for eyes"—they discover that eyes are useful features for recognizing faces.

### The Vanishing Gradient Problem

Training deep networks hit a wall. With many layers, backpropagation struggles—the gradient (the "blame" signal) gets smaller and smaller as it passes backward through layers.

```
Layer 10: Strong gradient (weights update well)
Layer 9: Decent gradient
Layer 8: Weaker gradient
...
Layer 1: Gradient ≈ 0 (weights barely change!)
```

The early layers (which detect basic features) don't learn because the error signal vanishes before reaching them.

### Solutions That Enabled Deep Learning

**ReLU Activation (2010):**

Traditional neurons used smooth "squashing" functions that compressed outputs to a range like 0-1. This caused gradients to vanish.

ReLU (Rectified Linear Unit) is dead simple:

```
ReLU(x) = max(0, x)
```

If input is negative: output 0
If input is positive: output unchanged

Gradients flow through ReLU without shrinking (for positive values). This simple change enabled training much deeper networks.

**Batch Normalization (2015):**

Normalize the outputs of each layer so they don't drift to extreme values. This stabilizes training and allows even deeper networks.

**Residual Connections (2015):**

Instead of:
```
output = layer(input)
```

Do:
```
output = input + layer(input)
```

The gradient can flow directly through the "input" path, bypassing the layer entirely. This allows training networks with hundreds of layers.

### The GPU Revolution

None of this would matter without hardware.

Neural network training is massively parallel—you're doing the same operation (multiply-add) billions of times. CPUs execute operations one at a time (mostly). GPUs execute thousands simultaneously.

```
CPU: Do this. Now this. Now this. Now this...
GPU: Do these 10,000 things AT THE SAME TIME.
```

In 2012, a GPU-trained neural network (AlexNet) won the ImageNet competition so decisively that it changed the field overnight. Error rate dropped from 26% to 16%—a leap that normally took years happened in one step.

The deep learning era had begun.

---

## Chapter 0.5.3: What Neural Networks Actually Learn

### Functions, Not Rules

A neural network is a mathematical function:

```
f(input) = output
```

Give it an image, get a classification. Give it text, get more text. Give it audio, get a transcription.

The network's architecture defines the *shape* of possible functions. Training finds *which specific function* (among trillions of possibilities) best fits your data.

### The Universal Approximation Theorem

Here's a mind-bending result: A neural network with just one hidden layer (with enough neurons) can approximate any continuous function to arbitrary precision.

This means neural networks are *universal function approximators*. If a function exists that maps your inputs to your outputs, a neural network can (in theory) learn it.

In practice, deeper networks learn faster and generalize better. But the theorem explains why neural networks are so versatile—they're not limited to specific kinds of patterns.

### What's in the Weights?

After training, knowledge is stored in the weights. But it's not stored like data in a database. There's no weight that stores "cats have pointy ears" or "the capital of France is Paris."

Instead, knowledge is distributed across many weights. Change one weight, and you slightly change everything the network knows. This is called **distributed representation**.

```
Traditional storage:
    Address 0x001: "cats have pointy ears"
    Address 0x002: "dogs have floppy ears"

Neural storage:
    Weight w₁₂₃: 0.0034  (contributes to cat knowledge, dog knowledge,
    Weight w₁₂₄: -0.891     ear knowledge, and 10,000 other concepts
    Weight w₁₂₅: 0.502      all at once)
    ...
```

This is why you can't "edit" neural network knowledge cleanly. You can't find the weight that stores "the capital of France" and change it. The knowledge is smeared across billions of weights.

### The Loss Landscape

Training a neural network is like finding the lowest point in a mountain range while blindfolded.

The "height" at any point is the loss—how wrong the network is. You want to find the lowest point (minimum loss). But you can't see the whole landscape; you can only feel the slope where you're standing.

```
Loss
  ↑
  │     ╱╲
  │    ╱  ╲   ╱╲
  │   ╱    ╲_╱  ╲____
  │  ╱               ╲
  │ ╱                 ╲_____ global minimum
  │╱
  └─────────────────────────→ weights
```

Gradient descent follows the slope downward. But it might get stuck in a local minimum—a valley that isn't the lowest point.

Modern deep learning works because the loss landscape of large networks has a surprising property: most local minima are nearly as good as the global minimum. The landscape has many valleys, but they're all roughly the same depth.

Why? Nobody fully knows. It's one of the mysteries of deep learning.

---

## Chapter 0.5.4: Types of Neural Networks

### Feedforward Networks

The simplest architecture. Information flows one direction: input → hidden layers → output. No loops, no memory.

```
Input → ○ → ○ → ○ → Output
        ○ → ○ → ○
        ○ → ○ → ○
```

Good for: Classification, regression, simple predictions.

### Convolutional Neural Networks (CNNs)

Designed for grid-like data (images, audio spectrograms). Instead of connecting every neuron to every input, they use small sliding filters that detect local patterns.

```
Image:      Filter:     Feature map:
[1 2 3]     [1 0]       [detects
[4 5 6]  ×  [0 1]    =   diagonal
[7 8 9]                  edges]
```

The same filter slides across the whole image, detecting the same pattern everywhere. This is called **weight sharing**—drastically fewer parameters than full connections.

CNNs learn hierarchies of filters:
- Early layers: Edges, colors
- Middle layers: Textures, shapes
- Late layers: Objects, faces

Good for: Images, video, any grid-structured data.

### Recurrent Neural Networks (RNNs)

Networks with loops—output feeds back as input. This gives them memory.

```
      ┌────────────┐
      │            │
Input → ○ ────────→ Output
      ↑           │
      └───────────┘
        (recurrence)
```

At each timestep, the network sees the current input AND its previous output. This lets it process sequences—words in a sentence, frames in a video, notes in a melody.

The problem: Backpropagation through time. Gradients either vanish (can't learn long-term patterns) or explode (training becomes unstable).

**LSTMs (Long Short-Term Memory):** Add special "gates" that control what to remember and what to forget. This allows learning patterns across hundreds of timesteps.

Good for: Sequences, time series, text (before transformers).

### Transformers

The architecture that powers GPT, BERT, and modern LLMs. We'll cover this in detail in Part II, but the key insight:

Instead of processing sequences step-by-step, transformers process everything at once using **attention**—a mechanism that lets each position "look at" all other positions directly.

```
Traditional sequence processing:
    "The" → process → "cat" → process → "sat" → process

Transformer processing:
    "The" "cat" "sat" → process all at once
                        ↑  ↑  ↑
                        All words can see all words
```

This allows massive parallelization and learning very long-range patterns.

Good for: Language, and increasingly everything else.

---

## Chapter 0.5.5: Training in Practice

### The Training Recipe

```
INGREDIENTS:
- Data (lots of it)
- Labels (what each data point should produce)
- Model architecture (the network structure)
- Loss function (how to measure wrongness)
- Optimizer (how to update weights)
- Compute (GPUs, TPUs, time, electricity)

INSTRUCTIONS:
1. Initialize weights randomly
2. Process a batch of data
3. Compute loss (how wrong)
4. Backpropagate (compute gradients)
5. Update weights (gradient descent)
6. Repeat steps 2-5 for hours/days/weeks
7. Evaluate on held-out test data
8. If not good enough, go back and try different ingredients
```

### Overfitting: The Mortal Enemy

A network that memorizes its training data perfectly but fails on new data has **overfit**.

```
Training error: 0.1%  ← Great!
Test error: 45%       ← Terrible!
```

The network learned the noise and quirks of the specific training examples instead of the underlying patterns.

**Solutions:**
- More data (harder to memorize)
- Regularization (penalize complexity)
- Dropout (randomly disable neurons during training)
- Early stopping (stop before memorization happens)
- Data augmentation (create variations of training data)

### Hyperparameters

Choices that affect training but aren't learned:
- Learning rate (how big are weight updates)
- Batch size (how many examples per update)
- Number of layers
- Neurons per layer
- Dropout rate
- Training time

Finding good hyperparameters is part art, part science, part luck. Researchers spend enormous effort on "hyperparameter tuning."

### The Compute Arms Race

Training modern LLMs requires staggering compute:

| Model | Training Cost | GPU-Hours |
|-------|---------------|-----------|
| GPT-2 (2019) | ~$50K | ~250 GPU-days |
| GPT-3 (2020) | ~$5M | ~3,000 GPU-days |
| GPT-4 (2023) | ~$100M+ | Unknown |

This is why AI is concentrated in a few large companies. Training frontier models requires resources that few organizations possess.

---

## Chapter 0.5.6: The Path to Language Models

### From Images to Words

Early deep learning success was in computer vision. Images are nice—they're grids of numbers (pixel values), easy to process.

Language is harder:
- Words aren't numbers
- Sentences have variable length
- Meaning depends on context
- Grammar is complicated
- Ambiguity everywhere

### Word Embeddings

The first breakthrough: represent words as vectors (lists of numbers).

```
"king"  → [0.2, 0.8, 0.1, -0.5, ...]
"queen" → [0.3, 0.7, 0.2, -0.4, ...]
"man"   → [0.1, 0.6, 0.0, -0.3, ...]
"woman" → [0.2, 0.5, 0.1, -0.2, ...]
```

The magic: These vectors capture relationships!

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

The difference between "king" and "man" captures the concept of royalty. Add that to "woman" and you get "queen."

This wasn't programmed. The network discovered it from reading lots of text.

### Sequence-to-Sequence

RNNs could process sequences, but how do you handle variable-length input AND output?

The encoder-decoder architecture (2014):

```
Encoder: Process input sequence, produce a single vector
Decoder: Take that vector, generate output sequence
```

Used for machine translation:
```
"The cat sat" → [ENCODER] → context vector → [DECODER] → "Le chat s'est assis"
```

The context vector is a "thought"—the meaning compressed into numbers.

Problem: One vector can't capture everything about a long sentence.

### Attention (2014-2015)

Instead of compressing everything into one vector, let the decoder look at all encoder states:

```
"The cat sat on the mat"
  ↓   ↓   ↓   ↓   ↓   ↓
 [e₁] [e₂] [e₃] [e₄] [e₅] [e₆]

When generating "Le":
    - Look at "The" (high attention)
    - Ignore "cat", "sat", "mat" (low attention)

When generating "chat":
    - Look at "cat" (high attention)
    - Ignore others (low attention)
```

The decoder learns *where to look* in the input for each output word. This is **attention**.

### The Transformer (2017)

The paper "Attention Is All You Need" asked: What if we use attention *everywhere*?

No RNNs. No convolutions. Just attention layers stacked on top of each other.

```
Input → [ATTENTION] → [ATTENTION] → ... → [ATTENTION] → Output
```

Key innovations:
- **Self-attention:** Each word attends to all other words in the same sequence
- **Multi-head attention:** Multiple attention patterns in parallel
- **Positional encoding:** Add position information since transformers process everything at once

This architecture proved dramatically more effective than RNNs and could be trained much faster (massive parallelism).

Every major language model since 2018 is based on transformers.

---

## Chapter 0.5.7: How LLMs Actually Work (Preview)

### The Pre-training Objective

GPT's training objective is almost absurdly simple:

```
Predict the next word.
```

That's it. Show the model a sequence of words. Ask it to predict what comes next. Tell it if it was right. Repeat trillions of times.

```
Input: "The cat sat on the"
Target: "mat"

Input: "Einstein developed the theory of"
Target: "relativity"

Input: "To be or not to"
Target: "be"
```

Why does this simple objective produce such capable models?

### The Compression Hypothesis

To predict the next word well, you need to understand:
- Grammar (what word types can come next)
- Semantics (what makes sense)
- World knowledge (what's true about cats, Einstein, Shakespeare)
- Context (what was discussed earlier)
- Pragmatics (what the writer is trying to do)

Good compression requires good understanding. A model that can predict text well must have learned something about the structure of text—and the world that text describes.

### Emergence

Larger models don't just make fewer errors. They develop qualitatively new capabilities:

| Size | Emergent Abilities |
|------|-------------------|
| 1B parameters | Basic grammar, simple Q&A |
| 10B parameters | Multi-step reasoning, basic math |
| 100B parameters | Complex reasoning, code generation |
| 1T+ parameters | Few-shot learning, novel task solving |

Nobody programmed these abilities. They emerged from scale + simple objective.

This is both exciting and concerning. Exciting because of the capabilities. Concerning because we don't fully understand why or what else might emerge.

---

## Exercises: Test Your Understanding

**Exercise 0.5.1:** A perceptron with weights w₁=2, w₂=-1 and threshold θ=1. What is the output for input (1, 1)? What about (0, 1)?

**Exercise 0.5.2:** Why can't a single perceptron learn XOR? Draw the four points and try to separate them with one line.

**Exercise 0.5.3:** If a network has 100 million parameters and you're training on 1 billion examples, approximately how many times is each parameter updated (assuming batch size of 1)?

**Exercise 0.5.4:** A model achieves 99% accuracy on training data but only 60% on test data. What's happening? What would you try to fix it?

**Exercise 0.5.5:** Why do transformers need positional encoding but RNNs don't?

---

## Key Takeaways

1. **Artificial neurons are simple:** Weighted sums + threshold. But many of them together can learn anything.

2. **Deep networks learn hierarchies:** Simple features combine into complex ones, layer by layer.

3. **Backpropagation makes learning possible:** Trace errors backward, adjust weights proportionally.

4. **Hardware matters:** GPUs enabled deep learning by parallelizing computation.

5. **Language models predict next words:** This simple objective, at scale, produces remarkable capabilities.

6. **Transformers revolutionized NLP:** Attention lets every position see every other position, enabling learning of complex dependencies.

You now have the foundations to understand what LLMs actually do. Part I will show you how to use them effectively. Part II will go deeper into the transformer architecture and attention mechanism.

The parrot has a brain now. Let's see what it can do.

---

*Part 0.5 Summary:*
- Neural networks are inspired by biological neurons but aren't exact copies
- The perceptron (1958) could learn, but couldn't handle XOR
- Hidden layers + backpropagation (1986) enabled learning complex patterns
- Deep learning works because of ReLU, batch norm, residuals, and GPUs
- Transformers (2017) use attention everywhere and power all modern LLMs
- Simple training objective (predict next word) + scale = emergent capabilities

*Concepts introduced: perceptron, hidden layer, backpropagation, gradient descent, overfitting, CNN, RNN, LSTM, transformer, attention, word embedding, emergence*
