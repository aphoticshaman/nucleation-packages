# Nucleation vs Alternatives

A honest comparison of early warning approaches.

---

## Quick Summary

| Approach | Detects | Timing | Setup | Cost |
|----------|---------|--------|-------|------|
| **Nucleation** | Variance compression (pre-transition) | Early | Minutes | Free (OSS) |
| **Datadog Anomaly** | Statistical outliers | During/After | Hours | $$$$ |
| **Prometheus + Alertmanager** | Threshold violations | After | Hours | Free (OSS) |
| **Custom ML Pipeline** | Whatever you train it for | Depends | Weeks | $$$ (compute) |
| **Roll Your Own** | Whatever you code | Depends | Days | $ (time) |

---

## Detailed Comparison

### vs Datadog Anomaly Detection

| Feature | Nucleation | Datadog |
|---------|------------|---------|
| **Detection method** | Variance compression | Statistical bounds (MAD, EWMA) |
| **Alert timing** | Before transition | During/after spike |
| **Setup time** | 5 minutes | 1-2 hours |
| **Configuration** | 3 presets + custom | Many knobs |
| **Pricing** | Free | ~$15/host/month + anomaly add-on |
| **Data residency** | Your infrastructure | Datadog cloud |
| **False positive rate** | Low-medium (tunable) | Medium-high |
| **Learning period** | None (immediate) | 2+ weeks recommended |

**When to use Datadog instead:**
- You're already paying for it
- You need unified observability (metrics, traces, logs)
- Your team knows it well
- You need support/SLAs

**When to use Nucleation instead:**
- You want earlier warning (before the spike)
- You don't want another SaaS vendor
- You need edge/Lambda deployment
- Budget is constrained

**Best practice: Use both**
- Nucleation for early warning
- Datadog for comprehensive observability
- They complement, not compete

---

### vs Prometheus + Alertmanager

| Feature | Nucleation | Prometheus Stack |
|---------|------------|------------------|
| **Detection method** | Variance-based early warning | Threshold + PromQL |
| **Alert timing** | Before transition | After threshold crossed |
| **What you can detect** | Phase transitions | Anything you can query |
| **Setup complexity** | npm install | Deploy Prometheus, configure scrapers, write PromQL, configure Alertmanager |
| **Flexibility** | Focused on one thing | Extremely flexible |
| **Memory footprint** | ~10MB | 100MB-10GB depending on cardinality |

**When to use Prometheus instead:**
- You need general-purpose monitoring
- You have complex alerting rules
- Your team knows PromQL
- You need long-term metric storage

**When to use Nucleation instead:**
- You specifically want phase transition detection
- You want something that works out of the box
- You're adding to an existing Prometheus stack (export metrics to it)

**Best practice: Use together**
```javascript
import { monitor, createPrometheusExporter } from 'nucleation';

const detector = await monitor('finance');
const exporter = createPrometheusExporter(detector);

// Expose via your existing /metrics endpoint
// Alert in Prometheus when nucleation_transitioning == 1
```

---

### vs Custom ML Pipeline

| Feature | Nucleation | Custom ML |
|---------|------------|-----------|
| **Detection method** | Variance analysis | Whatever you build |
| **Accuracy** | Good for phase transitions | Potentially better for specific use case |
| **Setup time** | 5 minutes | Weeks to months |
| **Maintenance** | npm update | Model retraining, drift detection, pipeline maintenance |
| **Explainability** | Simple (variance dropped) | Often black box |
| **Training data required** | None | Lots |
| **Compute cost** | Negligible | $$$-$$$$ |
| **Cold start** | Works immediately | Needs warm-up/loading |

**When to use Custom ML instead:**
- You have data scientists on staff
- You have labeled historical data
- Your pattern is unique/complex
- Accuracy is worth the investment

**When to use Nucleation instead:**
- You don't have ML expertise
- You need something now
- You want explainable alerts
- Your problem fits the variance-compression pattern

**Best practice: Start with Nucleation, upgrade to ML if needed**
- Ship Nucleation in a day
- Collect labeled data (Nucleation predictions + actual outcomes)
- Train custom model later if ROI justifies it
- Use Nucleation predictions as features in your ML model

---

### vs Roll Your Own

Here's what "roll your own" typically looks like:

```python
import numpy as np

def detect_anomaly(values, window=30):
    recent = values[-window:]
    mean = np.mean(recent)
    std = np.std(recent)
    
    latest = values[-1]
    z_score = (latest - mean) / std
    
    return abs(z_score) > 2.0
```

**Problems with this approach:**

1. **Detects spikes, not pre-transition calm** — This alerts AFTER the anomaly, not before

2. **No state management** — Doesn't track baseline variance over time

3. **No edge cases handled** — What if std=0? What about startup period?

4. **No persistence** — Can't serialize/restore for Lambda

5. **No event emission** — You'll build this eventually

6. **No testing** — Did you run ablation tests against synthetic patterns?

**What Nucleation adds:**

| Capability | DIY | Nucleation |
|------------|-----|------------|
| Pre-transition detection | ❌ | ✅ |
| Baseline tracking | Manual | Built-in |
| Edge case handling | Manual | Built-in |
| Serialization | Manual | Built-in |
| Event emitter | Manual | Built-in |
| TypeScript types | Manual | Built-in |
| Ablation tested | No | Yes |
| WASM performance | No | Yes |
| Multiple sensitivities | Manual | Built-in |

**Time to implement equivalent:**
- Basic detector: 1-2 hours
- With state management: 4-8 hours
- With persistence: +2-4 hours
- With testing: +4-8 hours
- With TypeScript: +2-4 hours
- With event system: +2-4 hours
- With tuning/ablation: +8-16 hours

**Total: 1-4 days for one domain**

Or: `npm install nucleation` (5 minutes)

**When to roll your own:**
- You have very specific requirements
- You want to learn
- You need to avoid dependencies
- Your use case doesn't fit variance-based detection

**When to use Nucleation:**
- You want to ship today
- You're solving a standard problem
- You value tested, maintained code
- You want the meta-package benefits (multiple domains, unified API)

---

## Decision Matrix

| If you need... | Use |
|----------------|-----|
| Early warning (before spike) | **Nucleation** |
| Spike detection (during/after) | Datadog, Prometheus, traditional |
| Comprehensive observability | Datadog, Prometheus |
| Custom pattern detection | Custom ML |
| Quick solution, low effort | **Nucleation** |
| Enterprise support/SLAs | Datadog |
| Edge/Lambda deployment | **Nucleation** |
| Budget-constrained | **Nucleation**, Prometheus |
| Maximum flexibility | Prometheus, Custom |

---

## The Honest Take

**Nucleation is good at:**
- Detecting phase transitions before they happen
- Working out of the box with zero training
- Deploying anywhere (edge, Lambda, Node, browser)
- Being simple to understand and debug

**Nucleation is NOT good at:**
- Replacing comprehensive monitoring
- Detecting patterns that don't follow variance compression
- Providing enterprise support
- Being a one-stop observability platform

**Use Nucleation when** the cost of being late is high, and you want an early warning layer on top of your existing monitoring.

**Don't use Nucleation when** you need general-purpose monitoring or your specific pattern doesn't follow the variance-before-transition signature.

---

## Integration Patterns

### Layer on top of existing monitoring

```
Your Data → Nucleation (early warning) → Slack/PagerDuty
     ↓
Prometheus/Datadog (comprehensive) → Existing alerts
```

### Enrich existing pipeline

```
Kafka → Nucleation → Kafka (enriched with transition predictions)
                          ↓
                     Your existing consumers
```

### Edge preprocessing

```
IoT Sensors → Cloudflare Worker (Nucleation) → Only alert when transitioning
                                                    ↓
                                              Backend (reduced traffic)
```

---

## Try It

```bash
npm install nucleation
```

```javascript
import { monitor } from 'nucleation';

const detector = await monitor('finance');

detector.on('warning', state => {
  console.log('Early warning:', state);
});
```

5 minutes to value. Free forever. MIT licensed.
