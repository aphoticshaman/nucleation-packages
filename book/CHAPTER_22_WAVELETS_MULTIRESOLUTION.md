# Chapter 22: Wavelets and Multi-Resolution Analysis

The previous chapters operated at single scales: one temperature, one order parameter, one confidence level. But real signals have structure at multiple scales simultaneously.

A stock price has:
- Microsecond noise (market microstructure)
- Minute-level patterns (intraday trading)
- Daily patterns (market sessions)
- Weekly patterns (economic cycles)
- Long-term trends (secular growth)

Analyzing at just one scale misses information at others. Multi-resolution analysis extracts features across scales simultaneously.

---

## The Scale Problem

### Single-Scale Limitations

**Fourier analysis** decomposes signals into sine waves:
```
f(t) = Σ aₙ exp(i×2πn×t/T)
```

This captures frequency content but loses time localization. A brief spike contributes to many frequencies without indicating when it occurred.

**Moving averages** smooth at a fixed scale:
```
smooth(t) = mean(f[t-w:t+w])
```

This captures time-local behavior but mixes scales. Short-term fluctuations contaminate long-term trend estimation.

### What We Need

Analysis that provides:
- **Frequency resolution** at low frequencies (distinguish slow patterns)
- **Time resolution** at high frequencies (localize fast events)
- **Scale separation** without losing time information

Wavelets provide exactly this.

---

## Wavelet Fundamentals

### The Mother Wavelet

A wavelet is a small, localized wave:
- Oscillates (positive and negative regions)
- Decays to zero (localized in time)
- Integrates to zero (no DC component)

**Haar wavelet:**
```
ψ(t) = +1  for 0 ≤ t < 0.5
     = -1  for 0.5 ≤ t < 1
     = 0   otherwise
```

**Daubechies wavelet:** Smoother, better frequency localization

**Morlet wavelet:** Gaussian-windowed sine wave, continuous

### Scaling and Translation

From the mother wavelet, generate a family:

```
ψₐ,ᵦ(t) = (1/√a) × ψ((t - b) / a)
```

Where:
- a = scale (stretches/compresses the wavelet)
- b = translation (shifts in time)
- 1/√a = normalization

Large a: Wide wavelets capture low frequencies
Small a: Narrow wavelets capture high frequencies

### The Wavelet Transform

Transform signal f(t) to wavelet domain:

```
W(a, b) = ∫ f(t) × ψₐ,ᵦ*(t) dt
```

W(a, b) tells you how much the signal resembles a wavelet of scale a centered at time b.

---

## Discrete Wavelet Transform

### The DWT Algorithm

For computational efficiency, use discrete scales and translations:

```
scales: a = 2ʲ for j = 0, 1, 2, ...
translations: b = k × 2ʲ for integer k
```

This gives the discrete wavelet transform (DWT).

### Multiresolution Decomposition

The DWT decomposes signals into:
- **Approximation coefficients** at each level (low-frequency content)
- **Detail coefficients** at each level (high-frequency content)

```
Signal = A₃ + D₃ + D₂ + D₁

Where:
- A₃: Lowest frequency (trend)
- D₃: Low-frequency details
- D₂: Mid-frequency details
- D₁: High-frequency details
```

### The Fast Algorithm

Mallat's algorithm computes DWT efficiently:

```
ALGORITHM: Fast Wavelet Transform

INPUT: Signal x of length N

1. Initialize: A₀ = x

2. For level j = 1 to J:
   - Low-pass filter: Aⱼ = downsample(conv(Aⱼ₋₁, h))
   - High-pass filter: Dⱼ = downsample(conv(Aⱼ₋₁, g))

OUTPUT: {Aⱼ, D₁, D₂, ..., Dⱼ}

Complexity: O(N) — linear in signal length
```

Where h and g are the low-pass and high-pass filter coefficients determined by the wavelet choice.

---

## Wavelet Denoising

### The Denoising Problem

Given noisy signal: y = f + ε

Find estimate of f that removes noise ε while preserving signal features.

### Thresholding Approach

Noise tends to produce small wavelet coefficients. Signal features produce large coefficients.

**Hard thresholding:**
```
W̃ = W  if |W| > λ
   = 0  otherwise
```

**Soft thresholding:**
```
W̃ = sign(W) × max(|W| - λ, 0)
```

Soft thresholding is usually preferred—less artifact-prone.

### Threshold Selection

**Universal threshold (Donoho-Johnstone):**
```
λ = σ × √(2 log N)
```

Where σ is noise standard deviation and N is signal length.

This provably achieves near-optimal denoising for wide signal classes.

**Adaptive thresholds:**
Different λ at each scale, adapted to local noise levels.

### The Algorithm

```
ALGORITHM: Wavelet Denoising

INPUT: Noisy signal y

1. DWT: Compute wavelet coefficients W

2. Estimate noise: σ = MAD(D₁) / 0.6745
   (MAD of finest-scale details, normalized)

3. Threshold: λ = σ × √(2 log N)

4. Shrink: W̃ = soft_threshold(W, λ)

5. Inverse DWT: f̂ = IDWT(W̃)

OUTPUT: Denoised signal f̂
```

---

## Multi-Resolution Energy Analysis

### Energy at Each Scale

Wavelet coefficients encode energy at each scale:

```
E(scale j) = Σₖ |Dⱼ[k]|²
```

The energy distribution across scales characterizes the signal.

### Scale-Energy Signatures

Different signal types have characteristic scale-energy distributions:

**White noise:** Equal energy at all scales
**Pink (1/f) noise:** Energy proportional to 1/f
**Smooth signals:** Energy concentrated at low frequencies
**Transient signals:** Energy at high frequencies during events

### Anomaly Detection via Scale-Energy

```
ALGORITHM: Multi-Resolution Anomaly Detection

1. Establish baseline:
   - Compute DWT of normal signal
   - Record energy at each scale: E_baseline(j)

2. Monitor:
   - Compute DWT of current signal
   - Compute current energy: E_current(j)

3. Detect anomalies:
   - deviation(j) = |E_current(j) - E_baseline(j)| / E_baseline(j)
   - If max deviation > threshold: flag anomaly
   - Which scale deviates indicates anomaly type
```

---

## Spectral Coherence

### Coherence Across Signals

For two signals x and y, coherence measures shared oscillation:

```
Coherence(f) = |Sxy(f)|² / (Sxx(f) × Syy(f))
```

Where S denotes power spectral density.

Coherence near 1: Signals share oscillation at frequency f
Coherence near 0: Independent at frequency f

### Wavelet Coherence

Extend coherence to time-frequency domain:

```
Wavelet Coherence(a, b) = |Wxy(a,b)|² / (|Wxx(a,b)| × |Wyy(a,b)|)
```

This reveals when and at what scale signals are coupled.

### Applications

**Market analysis:** Which assets are coupled at which frequencies?
- High-frequency coherence: Same market microstructure
- Low-frequency coherence: Same economic exposure

**System monitoring:** Which components interact at which timescales?
- Quick response coupling
- Long-term drift coupling

---

## Applications

### Financial Time Series

**Trend extraction:**
- Low-frequency approximation = underlying trend
- High-frequency details = trading noise

**Volatility estimation:**
- Energy in details indicates volatility
- Scale distribution indicates volatility structure

**Event detection:**
- Sharp coefficient spikes indicate events
- Scale of spike indicates event duration

### Signal Processing

**Audio denoising:**
- Speech has characteristic scale structure
- Background noise has different structure
- Separate by scale-selective thresholding

**Image compression:**
- 2D wavelets decompose images
- Keep significant coefficients, discard small ones
- JPEG 2000 uses wavelets

### System Monitoring

**Sensor data:**
- Separate sensor noise from signal
- Detect anomalies at appropriate scales
- Track drift at low frequencies, faults at high frequencies

---

## Implementation Notes

### Choosing the Wavelet

**Haar:**
- Simplest
- Good for piecewise constant signals
- Sharp discontinuities

**Daubechies (db4, db8):**
- Smoother
- Better frequency localization
- General purpose

**Symlets:**
- Near-symmetric
- Reduced phase distortion
- Good for symmetric signals

**Coiflets:**
- Higher vanishing moments
- Better polynomial approximation
- Good for smooth signals

### Choosing the Decomposition Depth

**Rule of thumb:**
```
max_level = floor(log₂(signal_length)) - 2
```

**Practical:**
- Choose based on timescales of interest
- Too deep: Overly smooth approximation
- Too shallow: Miss low-frequency structure

### Boundary Handling

Signals don't extend to infinity. Options:
- **Zero padding:** Introduce artifacts at boundaries
- **Symmetric extension:** Reflect signal at boundaries
- **Periodic extension:** Wrap signal around

Symmetric extension usually works best.

---

## CIC Connection

### Multi-Scale Coherence in CIC

CIC's multi-scale structural coherence (C_multi) connects to wavelets:

**C₁ (exact consensus):** Finest scale—do samples match exactly?
**C₂ (cluster coherence):** Medium scale—do samples cluster?
**C₃ (range constraint):** Coarsest scale—do samples fit in reasonable bounds?

This is wavelet decomposition in concept: measure coherence at multiple scales.

### Wavelet-Enhanced CIC

```
ALGORITHM: Wavelet-Enhanced Value Clustering

1. Encode samples as signal (sorted order or timestamp order)

2. Wavelet decompose:
   - Trend: Overall central tendency
   - Details: Sample-to-sample variation structure

3. Cluster using multi-scale features:
   - Use both original values and wavelet features
   - Clusters that share multi-scale structure are more reliable

4. Weight by scale coherence:
   - Samples coherent at multiple scales get higher weight
```

---

## Summary

Multi-resolution analysis extracts features at multiple scales:

**Wavelet fundamentals:**
- Localized oscillating functions
- Scale and translate to create family
- Transform signal to time-scale domain

**Discrete wavelet transform:**
- Efficient O(N) algorithm
- Approximation and detail coefficients
- Perfect reconstruction

**Wavelet denoising:**
- Threshold coefficients
- Donoho-Johnstone threshold selection
- Near-optimal noise removal

**Multi-resolution energy:**
- Energy at each scale characterizes signal
- Anomaly detection via scale-energy deviation
- Different signal types have different signatures

**Spectral coherence:**
- Measure coupling between signals
- Time-varying coherence via wavelets
- Identify when and at what scale systems interact

Part IV is complete. We've applied CIC principles to phase transitions, anomaly detection, cascade prediction, optimization, signal fusion, uncertainty handling, and multi-resolution analysis.

Part V addresses the future: military doctrine for AI development, human-AI cognitive fusion, and the road to 2035.
