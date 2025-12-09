# Chapter 20: Multi-Signal Fusion and Attention-Weighted Integration

Real decisions draw on multiple information sources. A doctor considers symptoms, lab results, patient history, and imaging. A trader weighs price action, fundamentals, sentiment, and market structure. An autonomous system fuses camera, lidar, radar, and GPS.

The challenge: these sources have different reliability, different update rates, different failure modes. How do you combine them intelligently?

This chapter develops attention-weighted fusion—combining signals based on their relevance and reliability for the specific decision at hand.

---

## The Multi-Source Problem

### Why Multiple Sources?

Single-source decisions are fragile:
- The source might fail
- The source might be manipulated
- The source might not capture relevant information

Multiple sources provide:
- **Redundancy:** If one fails, others continue
- **Validation:** Disagreement signals problems
- **Coverage:** Different sources capture different aspects

### The Naive Approach

Average all sources equally:
```
decision = mean(source_1, source_2, ..., source_n)
```

Problems with this:
- Unreliable sources corrupt the average
- Sources with different scales aren't comparable
- Context doesn't affect weighting

### The Challenge

We need weighting that accounts for:
- **Reliability:** How accurate is this source historically?
- **Relevance:** How applicable is this source to current context?
- **Currency:** How recent is this information?
- **Conflict:** What does disagreement between sources mean?

---

## Attention Mechanisms for Fusion

### Attention in Deep Learning

Transformers use attention to weight input tokens differently based on query:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

The key insight: relevance depends on context (the query).

### Attention for Signal Fusion

Adapt this for multi-source fusion:

**Signals:** V = [v₁, v₂, ..., vₙ] (source outputs)
**Context:** Q (current decision problem)
**Keys:** K = [k₁, k₂, ..., kₙ] (source characteristics)

```
weights = softmax(Q × K^T / √d)
fused = Σᵢ weightᵢ × vᵢ
```

The context determines how sources are weighted.

### Computing Keys

Each source has a key vector encoding:
- **Domain:** What kind of information does this source provide?
- **Reliability:** Historical accuracy (base rate)
- **Latency:** How old is typical information?
- **Granularity:** What resolution does this source offer?

### Computing Queries

The decision context generates a query:
- **Task type:** What are we trying to decide?
- **Time horizon:** How far ahead are we planning?
- **Risk tolerance:** How much error can we accept?
- **Domain focus:** What aspects matter most?

### The Result

High attention weight when:
- Source domain matches task needs
- Source reliability meets risk threshold
- Source latency fits time horizon
- Historical performance is strong

---

## Category Weighting

### Source Categories

Group sources by type:
- **Quantitative:** Numbers, measurements, statistics
- **Qualitative:** Assessments, opinions, narratives
- **Structural:** Relationships, dependencies, flows
- **Temporal:** Trends, patterns, cycles

Each category has characteristic strengths and weaknesses.

### Category Reliability

Some categories are more reliable for certain tasks:

| Task Type | Best Categories |
|-----------|-----------------|
| Prediction | Quantitative, Temporal |
| Diagnosis | Qualitative, Structural |
| Planning | Structural, Temporal |
| Valuation | Quantitative, Qualitative |

### Category Fusion

Fuse within category first, then across categories:

```
ALGORITHM: Hierarchical Category Fusion

1. Within each category c:
   fused_c = attention_fuse(sources in category c)

2. Compute category weights:
   category_weights = attention(task_query, category_keys)

3. Across categories:
   final = Σ_c category_weight_c × fused_c
```

This prevents low-quality sources in one category from drowning out high-quality sources in another.

---

## Reliability Estimation

### Base Rate Reliability

Each source has historical performance:
```
reliability_i = P(source_i correct | source_i commits)
```

Estimate from labeled historical data.

### Conditional Reliability

Reliability varies by context:
```
reliability_i(context) = P(source_i correct | context)
```

A source might be reliable in normal conditions but unreliable in crisis.

### Tracking Reliability Over Time

```
ALGORITHM: Adaptive Reliability Estimation

For each time step:
   1. Observe source predictions
   2. Observe ground truth (when available)
   3. Update reliability estimate:
      rel_new = α × actual_accuracy + (1-α) × rel_old
   4. Decay confidence when ground truth unavailable:
      confidence *= decay_rate
```

This gives more weight to recent performance.

---

## Conflict Detection

### Agreement and Disagreement

Sources can:
- **Agree:** All point in same direction
- **Partial disagreement:** Some differ
- **Strong conflict:** Direct contradiction

### Interpreting Conflict

Conflict isn't always bad. It might mean:
- **Genuine uncertainty:** The situation is unclear
- **Different timeframes:** Sources measure different things
- **Bias:** Some sources are systematically wrong
- **Regime change:** Old relationships are breaking down

### Conflict Metrics

**Variance-based:**
```
conflict = Var(normalized_sources)
```

**Pairwise disagreement:**
```
conflict = mean(|source_i - source_j|) for all pairs
```

**Cluster-based:**
```
conflict = 1 - (largest_cluster_size / total_sources)
```

### Conflict Response

How to adjust fusion based on conflict:

| Conflict Level | Response |
|---------------|----------|
| Low | Normal fusion, high confidence |
| Medium | Increase weight on most reliable sources |
| High | Defer decision or report uncertainty |
| Extreme | Investigate sources, possible regime change |

---

## Regime Detection from Fused Signals

### Cross-Source Regime Signals

Regime changes often appear in source relationships before they appear in individual sources:

**Correlation breakdown:** Sources that usually agree start disagreeing
**Leader shift:** Different source becomes leading indicator
**Latency change:** Information transmission speeds change

### Fusion-Based Regime Detection

```
ALGORITHM: Regime Detection from Fusion Dynamics

1. Track fusion weights over time:
   weight_history[t] = attention_weights at time t

2. Detect weight shifts:
   weight_change = |weights(t) - weights(t-1)|
   If weight_change > threshold: flag regime candidate

3. Track cross-source correlation:
   correlation_history[t] = pairwise correlations
   If correlation structure changes: flag regime candidate

4. Confirm regime change:
   If multiple flags coincide: declare regime change
```

### Adaptive Fusion

When regime changes are detected:
1. Reduce confidence in historical reliability estimates
2. Increase exploration (more equal weights)
3. Track which sources adapt fastest to new regime
4. Gradually update reliability based on new regime performance

---

## Applications

### Intelligence Fusion

Combining multiple intelligence sources:
- HUMINT (human intelligence)
- SIGINT (signals intelligence)
- OSINT (open source intelligence)
- IMINT (imagery intelligence)

Each has different reliability, latency, and coverage. Attention-weighted fusion provides:
- Context-appropriate weighting
- Conflict detection for inconsistencies
- Confidence calibration

### Sensor Fusion

Autonomous vehicles fuse:
- Cameras (good for classification, poor in bad weather)
- Lidar (precise distance, expensive, limited range)
- Radar (works in weather, lower resolution)
- GPS (absolute position, sporadic failure)

Attention-weighted fusion:
- Weather conditions affect camera/lidar weights
- Urban canyons affect GPS weight
- Speed affects required latency

### Market Signal Fusion

Combining trading signals:
- Technical indicators (price-based)
- Fundamental signals (value-based)
- Sentiment signals (behavior-based)
- Flow signals (positioning-based)

Attention-weighted fusion:
- Market regime affects optimal combination
- Volatility affects reliability estimates
- Timeframe affects relevance weights

---

## Implementation Architecture

### Signal Preprocessing

Before fusion:
```
1. Normalize: Scale each signal to comparable range
2. Timestamp: Align signals to common time grid
3. Quality tag: Mark confidence/reliability metadata
4. Missing data: Handle gaps appropriately
```

### The Fusion Layer

```
ALGORITHM: Attention-Weighted Fusion

INPUT: signals S = {(v_i, k_i, r_i, t_i)}, context query q
       where v=value, k=key, r=reliability, t=timestamp

1. Compute recency factor:
   recency_i = exp(-λ × (now - t_i))

2. Compute attention weights:
   raw_attention = softmax(q · K^T / √d)

3. Incorporate reliability and recency:
   final_weight_i = raw_attention_i × r_i × recency_i

4. Normalize:
   weight_i = final_weight_i / Σ_j final_weight_j

5. Fuse:
   fused_value = Σ_i weight_i × v_i
   fused_confidence = f(weights, individual_confidences, conflict)

OUTPUT: (fused_value, fused_confidence, weight_breakdown)
```

### The Confidence Layer

Estimate confidence in fused result:

```
confidence = base_confidence
           × agreement_factor(conflict_level)
           × coverage_factor(num_active_sources)
           × reliability_factor(weighted_avg_reliability)
```

Where:
- agreement_factor decreases with conflict
- coverage_factor increases with more sources (diminishing returns)
- reliability_factor reflects weighted source quality

---

## Failure Modes

### Source Dropout

What if sources go silent?

**Graceful degradation:**
- Redistribute weight among remaining sources
- Flag reduced coverage
- Increase uncertainty in output

**Minimum coverage threshold:**
- If too few sources remain, flag critical
- Consider deferring decisions

### Correlated Failures

Sources might fail together:
- Same underlying data provider
- Same network infrastructure
- Same fundamental assumption

**Mitigation:**
- Track source independence
- Group correlated sources
- Ensure at least one source from each independent group

### Adversarial Corruption

Sources might be manipulated:
- Deliberate misinformation
- Targeted sensor spoofing
- Data poisoning

**Detection:**
- Outlier detection on individual sources
- Consistency checking across sources
- Compare to independent ground truth when available

---

## Summary

Multi-signal fusion combines diverse information sources intelligently:

**Attention-weighted fusion:**
- Context determines source relevance
- Keys encode source characteristics
- Queries encode decision needs

**Reliability tracking:**
- Historical accuracy estimation
- Context-conditional reliability
- Adaptive updates over time

**Conflict management:**
- Detect disagreement levels
- Interpret conflict meaning
- Adjust confidence accordingly

**Regime detection:**
- Monitor weight dynamics
- Track cross-source correlations
- Adapt to regime changes

Applications span intelligence, sensor systems, and financial markets—anywhere multiple information sources must inform decisions.

The next chapter develops the epistemic humility framework: principled limits on confidence and systematic handling of uncertainty.
