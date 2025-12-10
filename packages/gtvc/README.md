# @nucleation/gtvc

Gauge-Theoretic Value Clustering for Signal Fusion

## Overview

This package implements the theoretical insights from the CIC (Compression-Integration-Causality) framework:

- **Gauge Symmetry**: The 5% value tolerance defines a gauge equivalence class with O(ε²) invariance
- **RG Flow**: Renormalization group flow converges to unique fixed-point answers
- **CIC Functional**: F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T) for confidence calibration
- **Phase Detection**: Crystallization detection for convergence monitoring

## Installation

```bash
npm install @nucleation/gtvc
```

## Usage

### Basic Value Clustering

```typescript
import { gaugeClustering, optimalAnswer } from '@nucleation/gtvc';

// Multiple estimates from different sources
const estimates = [42, 41.5, 43, 100, 101, 7];

// Cluster into gauge-equivalent groups
const clusters = gaugeClustering(estimates, { epsilon: 0.05 });
console.log(clusters);
// => [{ members: [41.5, 42, 43], center: 42, score: 2.45 }, ...]

// Get optimal fused answer
const answer = optimalAnswer(estimates);
console.log(answer);
// => 42.17 (refined from winning cluster)
```

### Signal Fusion with Confidence

```typescript
import { SignalFuser } from '@nucleation/gtvc';

const fuser = new SignalFuser({
  epsilon: 0.05,
  lambda: 0.5,
  gamma: 0.3,
});

// Fuse multiple signals
const result = fuser.fuse([42, 43, 41, 100, 42.5]);

console.log(result.value);       // 42.13
console.log(result.confidence);  // 0.82
console.log(result.cicState);    // { phi: 0.75, entropy: 0.3, coherence: 0.8, ... }
```

### Streaming Aggregation

```typescript
import { StreamingAggregator } from '@nucleation/gtvc';

const aggregator = new StreamingAggregator(100);

// Add values from a stream
for (const value of dataStream) {
  const state = aggregator.add(value);

  if (state) {
    console.log('Current value:', state.value);
    console.log('Phase:', aggregator.getPhase());
  }
}
```

### Multi-Source Fusion

```typescript
import { fuseMultiSource } from '@nucleation/gtvc';

const sources = new Map([
  ['model_a', [42, 43, 41]],
  ['model_b', [42.5, 42.3]],
  ['model_c', [100, 101]],  // Outlier source
]);

const weights = new Map([
  ['model_a', 1.0],
  ['model_b', 1.5],  // Trust this source more
  ['model_c', 0.5],  // Trust less
]);

const result = fuseMultiSource(sources, weights);
console.log(result.value);  // Weighted fusion
```

## Theory

### Gauge Equivalence

Two values `a` and `b` are gauge-equivalent when:

```
|a - b| / max(|a|, |b|) < ε
```

The default ε = 0.05 (5%) provides optimal balance between:
- Error correction (tolerating noise)
- Precision preservation (distinguishing different answers)

### CIC Functional

```
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
```

Where:
- **Φ** = Integrated information (compression cohesion)
- **H** = Representation entropy (disorder)
- **C_multi** = Multi-scale coherence (clustering quality)
- **λ** = 0.5 (entropy weight)
- **γ** = 0.3 (coherence weight)

### RG Flow

The cluster center is the RG fixed point—the scale-invariant representative of the gauge equivalence class. Successive coarsening converges to this unique value.

## API Reference

### Clustering

- `gaugeEquivalent(a, b, epsilon)` - Check gauge equivalence
- `gaugeClustering(values, config)` - Cluster into equivalence classes
- `rgFlow(clusters, steps)` - Apply RG flow
- `optimalAnswer(values, config)` - Get optimal fused value
- `testGaugeInvariance(values, epsilon)` - Test gauge invariance

### CIC

- `computePhi(samples)` - Compute integrated information
- `computeEntropy(values)` - Compute representation entropy
- `computeCoherence(values)` - Compute multi-scale coherence
- `computeCIC(samples, values, config)` - Full CIC computation
- `detectCrystallization(history, lambda)` - Detect convergence

### Fusion

- `SignalFuser` - Stateful signal fuser
- `StreamingAggregator` - Real-time aggregator
- `fuseSignals(values, config)` - One-shot fusion
- `fuseMultiSource(sources, weights, config)` - Multi-source fusion

## Performance

- Clustering: O(n log n) for n values
- CIC computation: O(n²) for pairwise comparisons
- Memory: O(n) for streaming with fixed window

## References

- Chapter 26: Gauge-Theoretic Foundations of Value Clustering
- Appendix G: The PROMETHEUS Methodology
- The CIC Framework papers (see Bibliography)

## License

MIT
