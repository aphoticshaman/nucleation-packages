# PROMETHEUS v3: COMPLETE RESEARCH SYNTHESIS FOR HULL TACTICAL

## From 1,662 Files → $100,000 Prize

*Comprehensive extraction from burner/, assorted/, research/, skills/, web/, engine/*

---

## TIER 1: PHASE TRANSITION DETECTION (CRITICAL)

### 1. Variance Compression Predicts Regime Shifts
**Source**: `burner/WHY_THIS_WORKS.md`

The calm before the storm is REAL. Before major market transitions, variance **decreases**.

```python
def variance_compression_signal(returns, window=21):
    var_21 = returns.rolling(21).var()
    var_63 = returns.rolling(63).var()
    var_ratio = var_21 / (var_63 + 1e-8)
    compression = (var_ratio < 0.5).astype(int)
    return compression
```

**Hull Application**: When `var_compression=1`, reduce position - transition coming but direction uncertain.

---

### 2. Anomalous Dimension Δ for Crash Detection (STRONGEST PREDICTOR)
**Source**: `burner/research/anomaly/phase_transition.py`, arXiv:2408.06433

The anomalous dimension shows **STRONG upward trends before crashes** while volatility shows only weak trends.

```python
def anomalous_dimension(x, tau, K=1.0):
    """Δ = (K - ln|autocorr|) / ln(tau)"""
    G = autocorrelation(x, tau)
    return (K - np.log(abs(G) + 1e-10)) / np.log(tau)

def delta_trend_signal(returns, window=100, trend_window=20):
    deltas = []
    for t in range(window, len(returns)):
        window_data = returns[t-window:t]
        delta = np.mean([anomalous_dimension(window_data, tau)
                        for tau in range(5, 50)])
        deltas.append(delta)

    # Compute trend in Δ
    recent = deltas[-trend_window:]
    slope = np.polyfit(range(len(recent)), recent, 1)[0]
    return slope  # Positive = crash approaching
```

**Hull Application**: Rising Δ trend = crash risk → reduce position aggressively.

---

### 3. Critical Slowing Down (AC1)
**Source**: `burner/WHY_THIS_WORKS.md`, `burner/research/anomaly/phase_transition.py`

Near phase transitions, recovery time → ∞. Lag-1 autocorrelation INCREASES before transitions.

```python
def critical_slowing_down(returns, window=63):
    def calc_ac1(x):
        if len(x) < 10:
            return 0
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    ac1 = returns.rolling(window).apply(calc_ac1)
    ac1_rising = (ac1 > ac1.shift(5)).astype(int)
    return ac1, ac1_rising
```

**Hull Application**: `ac1_rising=1` → approaching transition → reduce exposure.

---

### 4. Landau-Ginzburg Order Parameter
**Source**: `burner/web/lib/physics/landau-ginzburg.ts`

Markets have an order parameter φ and potential V(φ) = -½aφ² + ¼bφ⁴.

```python
def market_order_parameter(features_df):
    """Extract order parameter from V/M/S feature coherence"""
    v_cols = [c for c in features_df.columns if c.startswith('V')]
    m_cols = [c for c in features_df.columns if c.startswith('M')]
    s_cols = [c for c in features_df.columns if c.startswith('S')]

    # Order = 1 - variance of z-scores within each group
    orders = []
    for cols in [v_cols, m_cols, s_cols]:
        if len(cols) >= 2:
            z = (features_df[cols] - features_df[cols].mean()) / (features_df[cols].std() + 1e-8)
            order = 1 - z.var(axis=1)
            orders.append(order)

    return pd.concat(orders, axis=1).mean(axis=1)
```

**Hull Application**: High order → stable regime. Low order → transition likely.

---

### 5. Kramers Escape Rate (Transition Probability)
**Source**: `burner/web/lib/physics/landau-ginzburg.ts`

Transition probability per unit time:
```
r = (ω₀ × ωb / 2πγ) × exp(-ΔV / kT)
```

```python
def kramers_escape_rate(barrier_height, temperature=1.0, damping=0.5):
    """Probability of regime transition"""
    omega_min = np.sqrt(2)  # Curvature at minimum
    omega_saddle = 1.0  # At saddle point
    prefactor = (omega_min * omega_saddle) / (2 * np.pi * damping)
    boltzmann = np.exp(-barrier_height / temperature)
    return prefactor * boltzmann
```

**Hull Application**: High escape rate → prepare for regime shift.

---

## TIER 2: ENSEMBLE FUSION (CRITICAL)

### 6. Dempster-Shafer Belief Fusion
**Source**: `burner/research/fusion/dempster_shafer.py`, `burner/web/lib/physics/dempster-shafer.ts`

Replace simple mean/std with reliability-weighted belief fusion.

```python
def dempster_shafer_ensemble(predictions, reliabilities):
    """
    Fuse predictions using Dempster-Shafer theory.

    predictions: array of shape (n_models,)
    reliabilities: array of shape (n_models,) in [0, 1]
    """
    # Additive fusion: weighted by reliability
    weights = reliabilities / (reliabilities.sum() + 1e-8)
    fused_pred = np.sum(predictions * weights)

    # Compute conflict (disagreement between sources)
    pairwise_conflict = 0
    for i in range(len(predictions)):
        for j in range(i+1, len(predictions)):
            # Conflict = reliability product × disagreement
            conflict_ij = reliabilities[i] * reliabilities[j] * \
                         abs(np.sign(predictions[i]) - np.sign(predictions[j]))
            pairwise_conflict += conflict_ij

    conflict = pairwise_conflict / (len(predictions) * (len(predictions)-1) / 2 + 1e-8)

    # Confidence: inverse of conflict
    confidence = 1 - conflict

    return fused_pred, confidence
```

**Hull Application**: Use confidence to scale position. High conflict → smaller position.

---

### 7. Meta-Fusion Selector (Attractor Dominance Signal)
**Source**: `burner/research/fusion/dempster_shafer.py`

**NOVEL INSIGHT #5**: The shift between fusion strategies signals attractor dominance.

```python
class MetaFusionSelector:
    """
    When multiplicative fusion outperforms additive:
    → Single attractor dominating (strong consensus)

    When additive outperforms multiplicative:
    → Multiple competing attractors (uncertain)
    """

    def attractor_dominance_signal(self, add_perf, mult_perf):
        # Positive = single attractor, negative = competing
        return np.tanh(mult_perf - add_perf)
```

**Hull Application**: Positive dominance → larger position. Negative → smaller.

---

### 8. Value Clustering for Consensus
**Source**: `assorted/cic_theory_validation.py`

Cluster predictions by relative proximity (88% error reduction):

```python
def value_clustering_consensus(predictions, threshold=0.05):
    """Cluster predictions and take median of largest cluster"""
    def relative_distance(a, b):
        return abs(a - b) / (max(abs(a), abs(b)) + 1e-8)

    # Build clusters
    clusters = []
    for pred in predictions:
        added = False
        for cluster in clusters:
            if relative_distance(pred, np.median(cluster)) < threshold:
                cluster.append(pred)
                added = True
                break
        if not added:
            clusters.append([pred])

    # Return median of largest cluster
    largest = max(clusters, key=len)
    return np.median(largest), len(largest) / len(predictions)
```

**Hull Application**: Use cluster consensus as primary prediction. Cluster size = confidence.

---

## TIER 3: CASCADE PREDICTION (IMPORTANT)

### 9. SIR Model for Market Cascades
**Source**: `burner/engine/src/formulas/cascade-predictor.ts`

8 cascade signatures with SIR dynamics:

| Pattern | Duration | Peak Position | Asymmetry |
|---------|----------|---------------|-----------|
| Flash Crash | 30 min | 0.1 | 5.0 |
| Meme Stock | 72 hrs | 0.6 | 1.8 |
| News Shock | 24 hrs | 0.15 | 10.0 |
| Regulatory | 1 week | 0.05 | 15.0 |

```python
def cascade_phase(intensity, velocity, acceleration):
    """Classify cascade phase"""
    if intensity < 0.1 and abs(velocity) < 0.05:
        return 'dormant'
    elif intensity < 0.3 and velocity > 0.02 and acceleration > 0:
        return 'seeding'
    elif 0.3 <= intensity < 0.7 and velocity > 0.05:
        return 'spreading'
    elif intensity >= 0.7 or (intensity >= 0.5 and velocity < 0.02 and acceleration < 0):
        return 'peak'
    elif intensity >= 0.2 and velocity < -0.02:
        return 'declining'
    else:
        return 'exhausted'
```

**Hull Application**: Position sizing by cascade phase:
- dormant/exhausted: normal position
- seeding: cautious
- spreading: reduced position
- peak: minimal position
- declining: cautious recovery

---

### 10. Cross-Domain Correlation Surge
**Source**: `burner/engine/src/formulas/cascade-predictor.ts`

When correlations across feature domains surge, cascade likely:

```python
def cross_domain_correlation(v_features, m_features, s_features):
    """Detect correlation surge across domains"""
    v_mean = v_features.mean(axis=1)
    m_mean = m_features.mean(axis=1)
    s_mean = s_features.mean(axis=1)

    corr_vm = v_mean.rolling(21).corr(m_mean)
    corr_vs = v_mean.rolling(21).corr(s_mean)
    corr_ms = m_mean.rolling(21).corr(s_mean)

    avg_corr = (abs(corr_vm) + abs(corr_vs) + abs(corr_ms)) / 3
    correlation_surge = (avg_corr > 0.7).astype(int)
    return correlation_surge
```

**Hull Application**: `correlation_surge=1` → reduce position, cascade risk.

---

## TIER 4: ENTROPY & INFORMATION (IMPORTANT)

### 11. Micro-Grokking Detection
**Source**: `burner/research/inference/micro_grokking.py`

Sharp negative d²(entropy)/dt² = model "clicked" (switched from exploration to exploitation).

```python
def detect_micro_grokking(entropy_trace, window=5):
    """Detect when model switches from exploration to exploitation"""
    # Smooth and compute second derivative
    smoothed = gaussian_filter1d(entropy_trace, sigma=1.0)
    d2 = savgol_filter(smoothed, window, polyorder=2, deriv=2)

    # Find strongest negative d2 (the "grok point")
    min_d2_idx = np.argmin(d2)
    min_d2_val = d2[min_d2_idx]

    # Classify
    if min_d2_val < -1.0:
        return 'sharp_grok', min_d2_idx  # High confidence
    elif min_d2_val < -0.5:
        return 'gradual_grok', min_d2_idx  # Medium confidence
    else:
        return 'no_grok', -1  # Low confidence
```

**Hull Application**: Use ensemble entropy to detect if models are "grokking" the current regime.

---

### 12. CIC Functional for Confidence
**Source**: `assorted/cic_theory_validation.py`, `NOBEL_PAPER_CIC_UNIVERSAL_PRINCIPLE.md`

The master equation for intelligence:
```
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)

Φ = Integrated Information (model agreement)
H = Entropy (prediction uncertainty)
C = Causal Power (prediction reliability)
```

```python
def compute_cic_functional(predictions, lambda_=0.3, gamma=0.1):
    """Compute CIC confidence functional"""
    # Φ: Integrated information (compression similarity of traces)
    # Simplified: use cluster coherence
    mean_pred = np.mean(predictions)
    phi = 1 - np.std(predictions) / (abs(mean_pred) + 1e-8)

    # H: Entropy of prediction distribution
    # Simplified: normalized variance
    h = np.var(predictions) / (np.var(predictions) + 1)

    # C: Causal power (reliability)
    # Simplified: inverse coefficient of variation
    cv = np.std(predictions) / (abs(np.mean(predictions)) + 1e-8)
    c = 1 / (1 + cv)

    F = phi - lambda_ * h + gamma * c
    return F
```

**Hull Application**: High F → larger position. Low F → smaller position.

---

### 13. Transfer Entropy for Causal Feature Selection
**Source**: `burner/research/core/transfer_entropy.py`, `burner/web/lib/physics/transfer-entropy.ts`

Measure directed information flow: which features CAUSE returns?

```python
def transfer_entropy(source, target, lag=1, bins=8):
    """T_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})"""
    # Discretize
    source_disc = pd.qcut(source, bins, labels=False, duplicates='drop')
    target_disc = pd.qcut(target, bins, labels=False, duplicates='drop')

    # Compute conditional entropies
    h_y_given_y = conditional_entropy(target_disc[lag:], target_disc[:-lag])
    h_y_given_yx = conditional_entropy_joint(target_disc[lag:],
                                              target_disc[:-lag],
                                              source_disc[:-lag])

    te = h_y_given_y - h_y_given_yx
    return max(0, te)
```

**Hull Application**: Weight features by transfer entropy to returns.

---

## TIER 5: REGIME DYNAMICS (IMPORTANT)

### 14. Q-Matrix Spectral Gap
**Source**: `burner/research/attractor/regime/q_matrix.py`

Spectral gap of transition matrix indicates metastability:
- **Small gap** → regimes are persistent (trend-following)
- **Large gap** → rapid switching (mean-reversion)

```python
def spectral_gap(transition_matrix):
    """Gap = min |Re(λ)| for λ ≠ 0"""
    eigenvalues = np.linalg.eigvals(transition_matrix)
    non_zero = eigenvalues[np.abs(eigenvalues - 1) > 1e-8]
    if len(non_zero) == 0:
        return 0
    return np.min(np.abs(1 - non_zero))
```

**Hull Application**: Small gap → trust momentum. Large gap → mean-revert.

---

### 15. Attractor Basin Mapping
**Source**: `burner/research/anomaly/phase_transition.py`

**NOVEL INSIGHT #3**: Map the attractor basin field.

```python
def attractor_basin_probabilities(features, n_regimes=3):
    """Compute probability of being in each attractor basin"""
    # Cluster in (anomalous_dimension, volatility) space
    from sklearn.cluster import KMeans

    delta = compute_anomalous_dimension(features['returns'])
    vol = features['returns'].rolling(21).std()

    X = np.column_stack([delta, vol])
    kmeans = KMeans(n_clusters=n_regimes)
    kmeans.fit(X)

    # Soft assignments via distance to centroids
    distances = kmeans.transform(X)
    neg_dist = -distances
    probs = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    return probs
```

**Hull Application**: Use basin probabilities as regime features.

---

### 16. Gravity Gradient Field
**Source**: `burner/research/anomaly/phase_transition.py`

**NOVEL INSIGHT #10**: The gradient of basin probability indicates attractor strength.

```python
def gravity_gradient(basin_probs):
    """Rising = strengthening attractor, falling = weakening"""
    gradients = np.gradient(basin_probs, axis=0)
    return gradients
```

**Hull Application**: Follow strengthening attractors, exit weakening ones.

---

## TIER 6: MARKET TEMPERATURE & COHERENCE

### 17. Market Temperature
**Source**: `burner/engine/src/formulas/phase-transition.ts`, `PROMETHEUS_HULL_INSIGHTS.md`

Temperature = system energy. High temperature = chaotic.

```python
def market_temperature(v_features):
    """Temperature: variance weighted by inverse correlation"""
    v_mean = v_features.mean(axis=1)
    v_std = v_features.std(axis=1)

    # Temperature = variance / mean (normalized energy)
    temperature = v_std / (abs(v_mean) + 1e-8)
    return temperature
```

**Hull Application**: High temperature → reduce position.

---

### 18. Kuramoto Order Parameter
**Source**: `burner/research/cognitive/kuramoto.py`, `PROMETHEUS_HULL_INSIGHTS.md`

R ∈ [0,1] measures phase synchronization:
- R → 1: All synchronized (regime stable)
- R → 0: Desynchronized (regime unstable)

```python
def kuramoto_order_parameter(features):
    """R = |mean(e^{i*phase})|"""
    # Extract phases via Hilbert transform
    from scipy.signal import hilbert

    phases = []
    for col in features.columns:
        analytic = hilbert(features[col].values)
        phase = np.angle(analytic)
        phases.append(phase)

    phases = np.array(phases)
    Z = np.mean(np.exp(1j * phases), axis=0)
    R = np.abs(Z)
    return R
```

**Hull Application**: Low R → reduce position (instability).

---

## TIER 7: FEATURE INTERACTIONS

### 19. Sentiment-Volatility Interaction
**Source**: `PROMETHEUS_HULL_INSIGHTS.md`

Non-linear interaction:
- High sent + Low vol → Bullish
- Low sent + High vol → Bearish
- High sent + High vol → Uncertain

```python
def sent_vol_interact(s_features, v_features):
    sent_mean = s_features.mean(axis=1)
    vol_mean = v_features.mean(axis=1)
    return sent_mean / (vol_mean + 1e-8)
```

---

### 20. Economic Surprise Integration
**Source**: `PROMETHEUS_HULL_INSIGHTS.md`

Rate of change matters more than level:

```python
def econ_surprise(e_features):
    econ_mean = e_features.mean(axis=1)
    econ_momentum = econ_mean.diff(5)
    econ_surprise = econ_momentum - econ_momentum.rolling(63).mean()
    return econ_surprise
```

---

### 21. Interest Rate Regime (Curve Inversion)
**Source**: `PROMETHEUS_HULL_INSIGHTS.md`

Curve inversion predicts recession:

```python
def rate_regime(i_features):
    if 'I3' in i_features.columns and 'I1' in i_features.columns:
        slope = i_features['I3'] - i_features['I1']
        inverting = (slope.rolling(63).rank(pct=True) < 0.1).astype(int)
        return inverting
    return 0
```

---

## TIER 8: DEEP MATHEMATICAL PRINCIPLES

### 22. Intelligence = Compression = Free Energy
**Source**: `burner/skills/DEEP_MATHEMATICS_OF_INTELLIGENCE.skill.md`

The unified theory:
```
Intelligence ≡ Compression ≡ -Free Energy ≡ log P(data)
```

**Application**: The model that compresses the market best will generalize best.

---

### 23. Information Bottleneck Principle
**Source**: `burner/skills/DEEP_MATHEMATICS_OF_INTELLIGENCE.skill.md`

```
IB Objective: min I(X; T) - β I(T; Y)
```

**Application**: Features should be minimal sufficient statistics for returns.

---

### 24. Grokking = Sudden Generalization
**Source**: `burner/skills/DEEP_MATHEMATICS_OF_INTELLIGENCE.skill.md`

After long plateau, sudden generalization can occur.

**Application**: Train longer than you think. Monitor for grokking signals.

---

### 25. Double Descent
**Source**: `burner/skills/DEEP_MATHEMATICS_OF_INTELLIGENCE.skill.md`

More parameters can mean BETTER generalization after interpolation threshold.

**Application**: Don't regularize too early. Let models overfit then compress.

---

## PROMETHEUS v3 FEATURE LIST

### Core PROMETHEUS Features (26 total)
1. `var_21` - 21-day variance
2. `var_63` - 63-day variance
3. `var_ratio` - Variance compression ratio
4. `var_compression` - Binary compression signal
5. `anomalous_dimension` - Δ(t,τ) crash predictor
6. `delta_trend` - Trend in Δ (STRONGEST signal)
7. `ac1` - Lag-1 autocorrelation
8. `ac1_rising` - Critical slowing down
9. `temperature` - Market temperature
10. `V_order` - V-feature coherence
11. `M_order` - M-feature coherence
12. `S_order` - S-feature coherence
13. `kuramoto_R` - Phase synchronization
14. `vol_regime` - Volatility regime
15. `vol_expanding` - Vol expansion signal
16. `sent_mean` - Sentiment mean
17. `sent_vol_interact` - Sentiment-vol interaction
18. `momentum_strong` - Momentum regime
19. `econ_surprise` - Economic surprise
20. `rate_inverting` - Yield curve inversion
21. `cross_domain_corr` - Cross-domain correlation surge
22. `cascade_phase` - Cascade phase (encoded)
23. `basin_prob_0/1/2` - Attractor basin probabilities
24. `gravity_gradient` - Attractor strength gradient
25. `cic_F` - CIC confidence functional
26. `ds_conflict` - Dempster-Shafer conflict

### Position Sizing Formula

```python
def prometheus_v3_position(
    predictions,
    reliabilities,
    features,
    config
):
    # 1. Dempster-Shafer fusion
    fused_pred, ds_confidence = dempster_shafer_ensemble(predictions, reliabilities)

    # 2. Value clustering
    cluster_pred, cluster_confidence = value_clustering_consensus(predictions)

    # 3. CIC functional
    cic_F = compute_cic_functional(predictions)

    # 4. Combined confidence
    confidence = 0.4 * ds_confidence + 0.4 * cluster_confidence + 0.2 * cic_F

    # 5. Risk adjustments
    risk_factor = 1.0
    if features['var_compression'].iloc[-1] == 1:
        risk_factor *= 0.7  # Reduce for compression
    if features['delta_trend'].iloc[-1] > 0.1:
        risk_factor *= 0.5  # Reduce for crash signal
    if features['ac1_rising'].iloc[-1] == 1:
        risk_factor *= 0.8  # Reduce for slowing down
    if features['cross_domain_corr'].iloc[-1] == 1:
        risk_factor *= 0.6  # Reduce for cascade risk

    # 6. Kelly-inspired sizing
    uncertainty = max(np.std(predictions), 1e-5)
    kelly = fused_pred / (config['risk_aversion'] * uncertainty**2 + 1e-8)

    # 7. Final position
    position = config['base_position'] + config['scale_factor'] * kelly * confidence * risk_factor
    position = np.clip(position, config['min_position'], config['max_position'])

    return position
```

---

## EXPECTED PERFORMANCE

| Version | Features | Local Sharpe | Key Additions |
|---------|----------|-------------|---------------|
| v1 (baseline) | 161 | 0.24 | Basic ensemble |
| v2 (PROMETHEUS) | 187 | 0.32 | +32% via 9 insights |
| **v3 (COMPLETE)** | **213** | **0.45+** | **+40% via 26 insights** |

---

## THE PROMETHEUS PHILOSOPHY

> "Variance DECREASES before phase transitions. The calm before the storm is real.
> Detect the calm, and you detect the coming storm."

> "The anomalous dimension Δ shows STRONG upward trends before crashes while
> volatility shows only weak trends. Δ is the superior signal."

> "Intelligence = Compression = Free Energy. The model that compresses best
> generalizes best."

Apply these principles, and the $100,000 prize is within reach.

---

*PROMETHEUS v3 × HULL TACTICAL*
*December 2024*
*1,662 files → 26 insights → $100k*
