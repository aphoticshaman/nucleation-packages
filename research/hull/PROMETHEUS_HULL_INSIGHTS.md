# PROMETHEUS × HULL TACTICAL: 20 Novel Insights

## From the Research Corpus to $100,000

*Distilled from 1,662 files across burner/, assorted/, and research/*

---

## INSIGHT #1: Variance Compression Predicts Regime Shifts

**Source**: `burner/WHY_THIS_WORKS.md`, `burner/nucleation/`

The calm before the storm is real. Before major market transitions, variance **decreases**.

```python
def detect_regime_shift(prices, window=21):
    """Negative z-score = variance compression = regime shift coming"""
    recent_var = prices[-window:].std()**2
    baseline_var = prices[-window*3:-window].std()**2
    z_score = (recent_var - baseline_var) / baseline_var
    return z_score < -2.0  # Compression detected
```

**Hull Application**: Add a `variance_compression` feature. When detected, reduce position size (transition is coming but direction is uncertain).

---

## INSIGHT #2: Phase Transition Temperature

**Source**: `burner/engine/src/formulas/phase-transition.ts`

Markets have "temperature" - a measure of system energy. High temperature = chaotic. Low temperature = crystalline/stable.

```python
def market_temperature(signals):
    """Temperature: variance weighted by inverse correlation"""
    total_var = np.mean([s.var() for s in signals])
    avg_corr = np.mean([pearsonr(signals[i], signals[j])[0]
                        for i in range(len(signals))
                        for j in range(i+1, len(signals))])
    return total_var * (1 + (1 - avg_corr))
```

**Hull Application**: Create `temperature` feature from cross-correlations of V, M, S features. High temperature → reduce position.

---

## INSIGHT #3: Kuramoto Order Parameter for Regime Coherence

**Source**: `burner/research/cognitive/kuramoto.py`

The Kuramoto order parameter R ∈ [0,1] measures phase synchronization:
- R → 1: All oscillators synchronized (regime is stable)
- R → 0: Desynchronized (regime is unstable/transitioning)

```python
def kuramoto_order_parameter(phases):
    """R = |mean(e^{i*phase})|"""
    Z = np.mean(np.exp(1j * phases))
    return np.abs(Z)
```

**Hull Application**: Extract "phases" from feature groups using Hilbert transform. Track R over time - low R predicts instability.

---

## INSIGHT #4: Anomalous Dimension for Crash Detection

**Source**: `burner/research/anomaly/phase_transition.py`, arXiv:2408.06433

The anomalous dimension Δ(t,τ) shows strong upward trends before crashes while volatility shows only weak trends.

```python
def anomalous_dimension(x, tau, K=1.0):
    """Δ = (K - ln(autocorr)) / ln(tau)"""
    G = autocorrelation(x, tau)
    return (K - np.log(abs(G))) / np.log(tau)
```

**Hull Application**: Track `delta_trend` - rising anomalous dimension signals crash risk → reduce position.

---

## INSIGHT #5: Transfer Entropy for Causal Feature Selection

**Source**: `burner/research/core/transfer_entropy.py`

Transfer entropy measures directed information flow: which features actually **cause** future returns vs just correlate.

```python
def transfer_entropy(source, target, lag=1):
    """TE = reduction in uncertainty about target from knowing source's past"""
    # High TE → source predicts target
    # Use for feature selection: keep high-TE features
```

**Hull Application**: Compute TE from each feature category (V, M, S, E, I, P) to returns. Weight features by TE.

---

## INSIGHT #6: Q-Matrix Spectral Gap for Metastability

**Source**: `burner/research/attractor/regime/q_matrix.py`

The spectral gap of the regime transition matrix indicates metastability:
- Small gap → regimes are persistent (stay in trend)
- Large gap → rapid regime switching (chop)

```python
def spectral_gap(Q):
    """Gap = min |Re(λ)| for λ ≠ 0"""
    eigenvalues = np.linalg.eigvals(Q)
    non_zero = eigenvalues[np.abs(eigenvalues) > 1e-8]
    return np.min(np.abs(np.real(non_zero)))
```

**Hull Application**: Estimate regime transition matrix from historical data. Small gap → trend-following. Large gap → mean-reversion.

---

## INSIGHT #7: Markov-Switching VAR for Regime Detection

**Source**: `burner/research/regime/msvar.py`

MS-VAR with Bayesian Group LASSO automatically selects which feature interactions matter per regime.

**Hull Application**: Train offline MS-VAR (K=3 regimes). Use smoothed regime probabilities as features:
- `p_bull`, `p_bear`, `p_neutral`

---

## INSIGHT #8: CIC Functional for Confidence Calibration

**Source**: `assorted/cic_theory_validation.py`

The CIC functional unifies confidence:
```
F[T] = Φ(T) - λ·H(T|X) + γ·C_multi(T)
```

Where:
- Φ = Integrated information (model agreement)
- H = Entropy (prediction uncertainty)
- C = Causal power (prediction reliability)

**Hull Application**: Compute F from ensemble predictions. High F → larger position. Low F → smaller position.

---

## INSIGHT #9: Value Clustering for 88% Error Reduction

**Source**: `assorted/cic_theory_validation.py`

Cluster predictions by relative proximity, not absolute distance:
```python
def relative_distance(a, b):
    return abs(a - b) / max(abs(a), abs(b))

# Cluster when relative_distance < 0.05
```

**Hull Application**: Generate multiple return predictions (different seeds). Cluster by relative proximity. Take median of largest cluster.

---

## INSIGHT #10: Phase-Coded Ensemble Voting

**Source**: `burner/research/fusion/phase_ensemble.py`

Assign complex phases to predictions based on agreement:
- Same answer → phase 0 (constructive interference)
- Opposite → phase π (destructive interference)

```python
def phase_weighted_mean(predictions, phases):
    """Complex-weighted average"""
    Z = np.sum(predictions * np.exp(1j * phases))
    return np.real(Z) / len(predictions)
```

**Hull Application**: Weight model predictions by phase coherence with ensemble. Outliers get orthogonal phases → less weight.

---

## INSIGHT #11: Critical Slowing Down

**Source**: `burner/WHY_THIS_WORKS.md`

Near phase transitions, recovery time → ∞. The system "freezes" before flipping.

**Metric**: Autocorrelation at lag 1 increases before transitions.

```python
def critical_slowing_down(x, window=21):
    """Rising AC1 = approaching transition"""
    return [autocorrelation(x[max(0,i-window):i], 1)
            for i in range(window, len(x))]
```

**Hull Application**: Track `ac1_trend`. Rising trend → reduce position exposure.

---

## INSIGHT #12: Attractor Genesis Detection

**Source**: `burner/research/regime/msvar.py` (NOVEL INSIGHT #8 in comments)

Detect when a **new** attractor (regime) is forming before it has mass.

**Signal**: Transition probabilities to a previously-rare regime suddenly increase.

**Hull Application**: Monitor regime transition matrix daily. Spike in probability to new regime → prepare for regime shift.

---

## INSIGHT #13: Volatility Regime from V-Features

**Source**: Hull data analysis, `burner/regime-shift/`

V-features (V1-V13) encode volatility information. Their mean creates a volatility regime indicator.

```python
df['vol_regime'] = df[[f'V{i}' for i in range(1,14)]].mean(axis=1)
df['vol_regime_ma21'] = df['vol_regime'].rolling(21).mean()
df['vol_expanding'] = (df['vol_regime'] > df['vol_regime_ma21']).astype(int)
```

**Hull Application**: `vol_expanding=1` → reduce position. `vol_expanding=0` → normal position.

---

## INSIGHT #14: Sentiment-Volatility Interaction

**Source**: Feature interaction analysis

The interaction between sentiment (S-features) and volatility (V-features) is non-linear:
- High sentiment + Low vol → Bullish (high position)
- Low sentiment + High vol → Bearish (low position)
- High sentiment + High vol → Uncertain (neutral position)

```python
df['sent_vol_interact'] = df['sent_mean'] / (df['vol_mean'] + 1e-8)
```

**Hull Application**: Add `sent_vol_interact` as a key feature for position sizing.

---

## INSIGHT #15: Kelly Criterion with Uncertainty Scaling

**Source**: `burner/research/fusion/phase_ensemble.py`, portfolio theory

Standard Kelly: `f* = μ / σ²`

Uncertainty-adjusted Kelly:
```python
def kelly_with_uncertainty(pred_mean, pred_std, risk_aversion=50):
    """Scale position by prediction confidence"""
    uncertainty = max(pred_std, 1e-5)
    kelly = pred_mean / (risk_aversion * uncertainty**2)
    return base_position + scale_factor * kelly
```

**Hull Application**: Already implemented! This is the core of our position sizing.

---

## INSIGHT #16: Momentum Regime Detection

**Source**: Hull data MOM features (implied), momentum literature

Momentum regimes:
- **Trending**: Momentum positive and rising → follow trend
- **Reverting**: Momentum extreme → expect mean reversion
- **Random**: Momentum near zero → reduce exposure

```python
df['momentum_regime'] = np.where(
    df['cum_ret_21'] > df['cum_ret_21'].rolling(63).std(),
    'trending',
    np.where(df['cum_ret_21'] < -df['cum_ret_21'].rolling(63).std(),
             'reverting', 'random')
)
```

**Hull Application**: Add `momentum_regime` as categorical feature or create regime-specific models.

---

## INSIGHT #17: Economic Surprise Integration

**Source**: Hull E-features (E1-E20), macro research

E-features encode economic data. The **rate of change** matters more than level:

```python
df['econ_momentum'] = df[[f'E{i}' for i in range(1,21)]].diff(5).mean(axis=1)
df['econ_surprise'] = df['econ_momentum'] - df['econ_momentum'].rolling(63).mean()
```

**Hull Application**: Positive `econ_surprise` → slightly bullish bias. Negative → bearish bias.

---

## INSIGHT #18: Interest Rate Regime

**Source**: Hull I-features (I1-I9), fixed income research

Interest rate features encode curve shape and level. Key insight: **curve inversion predicts recession**.

```python
df['rate_regime'] = df['I3'] - df['I1']  # Slope proxy
df['rate_inverting'] = (df['rate_regime'] < df['rate_regime'].rolling(63).quantile(0.1))
```

**Hull Application**: `rate_inverting=True` → defensive position (closer to min_position).

---

## INSIGHT #19: Multi-Model Disagreement as Uncertainty

**Source**: `assorted/PROMETHEUS_ANALYSIS.md`, ensemble theory

When LightGBM and XGBoost disagree strongly, uncertainty is high:

```python
def model_disagreement(lgb_pred, xgb_pred):
    """High disagreement = high uncertainty"""
    return abs(lgb_pred - xgb_pred) / (abs(lgb_pred) + abs(xgb_pred) + 1e-8)
```

**Hull Application**: Already captured in `std_pred`! But can weight by model type for more nuance.

---

## INSIGHT #20: Adaptive Position Bounds

**Source**: Portfolio optimization, Hull competition constraints

The 0-2 position range allows leverage. Use adaptively:

```python
def adaptive_bounds(volatility_regime, sentiment_regime):
    """Tighten bounds in uncertain regimes"""
    if volatility_regime == 'high' or sentiment_regime == 'negative':
        return (0.3, 1.2)  # Conservative
    elif volatility_regime == 'low' and sentiment_regime == 'positive':
        return (0.5, 1.8)  # Aggressive
    else:
        return (0.2, 1.5)  # Balanced
```

**Hull Application**: Dynamically adjust `min_position` and `max_position` based on regime.

---

## IMPLEMENTATION PRIORITY

### Tier 1 (Implement Now)
1. Variance compression (#1)
2. Market temperature (#2)
3. CIC confidence (#8)
4. Value clustering (#9)
5. Vol regime (#13)

### Tier 2 (Implement If Time)
6. Transfer entropy feature selection (#5)
7. Sentiment-vol interaction (#14)
8. Momentum regime (#16)
9. Critical slowing down (#11)
10. Adaptive bounds (#20)

### Tier 3 (Research/Backtest)
11. Kuramoto order parameter (#3)
12. Anomalous dimension (#4)
13. Q-matrix spectral gap (#6)
14. MS-VAR regime probabilities (#7)
15. Phase-coded ensemble (#10)

---

## EXPECTED IMPACT

| Insight | Expected Sharpe Improvement |
|---------|---------------------------|
| Variance compression | +0.05-0.10 |
| Market temperature | +0.03-0.07 |
| CIC confidence | +0.05-0.15 |
| Value clustering | +0.10-0.20 |
| Vol regime | +0.03-0.08 |
| **Combined** | **+0.15-0.35** |

Current baseline: 0.24 Sharpe
Target: 0.40-0.60 Sharpe

---

## THE PROMETHEUS PHILOSOPHY

> "Variance decreases before phase transitions. The calm before the storm is real.
> Detect the calm, and you detect the coming storm."

This is the unifying principle across all 20 insights:
- **Compression signals change**
- **Coherence measures stability**
- **Uncertainty scales exposure**

Apply these principles, and the $100,000 prize is within reach.

---

*PROMETHEUS × HULL TACTICAL*
*December 2024*
