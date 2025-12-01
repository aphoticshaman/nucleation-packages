# churn-harbinger

Predict customer churn before it happens.

Behavioral variance analysis for SaaS, subscription businesses, and product teams. The core insight: **users don't suddenly churn — they disengage gradually**, and that disengagement has a measurable signature.

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) for high-performance WebAssembly execution.

## Installation

```bash
npm install churn-harbinger
```

## Quick Start

```javascript
import { ChurnDetector } from 'churn-harbinger';

const detector = new ChurnDetector();
await detector.init();

// Feed daily engagement scores
for (const day of userActivity) {
  const state = detector.update(day.engagementScore);
  
  if (state.atRisk) {
    triggerCSMAlert({
      userId: user.id,
      riskLevel: state.riskLevel,
      confidence: state.confidence
    });
  }
}
```

## One-liner

```javascript
import { assessChurnRisk } from 'churn-harbinger';

const result = await assessChurnRisk(engagementHistory);
// { atRisk: true, riskLevel: 'at-risk', confidence: 0.82 }
```

## How It Works

Traditional churn prediction looks at engagement drops. By then, the user is already gone mentally.

This library detects the **precursor signal**: the characteristic variance reduction as users settle into minimal-use patterns. Active users have variable behavior. Disengaging users become predictable — same low activity, day after day.

The "settling" pattern typically appears **days to weeks** before cancellation.

## Risk Levels

| Level | Meaning | Recommended Action |
|-------|---------|-------------------|
| `healthy` | Normal engagement variance | Continue product experience |
| `cooling` | Engagement declining | Monitor, consider nudge |
| `at-risk` | Pre-churn pattern detected | CSM outreach, intervention |
| `churning` | Active disengagement | Immediate retention action |

## Configuration

```javascript
const detector = new ChurnDetector({
  sensitivity: 'balanced',  // 'conservative', 'balanced', 'sensitive'
  windowSize: 30,           // Days for baseline (default 30)
  threshold: 2.0            // Standard deviations for alert
});
```

| Sensitivity | Use Case |
|-------------|----------|
| `conservative` | High-value accounts, minimize false alerts |
| `balanced` | General user monitoring |
| `sensitive` | Early intervention strategy |

## Engagement Metrics

Feed any engagement metric that reflects user activity:

```javascript
// Option 1: Simple session count
detector.update(dailySessions);

// Option 2: Weighted composite score
const score = (sessions * 10) + (duration * 0.5) + (actions * 0.3);
detector.update(score);

// Option 3: Feature-specific engagement
detector.update(featureUsageCount);
```

The detector cares about **variance patterns**, not absolute values.

## Cohort Analysis

Compare users against healthy cohort behavior:

```javascript
import { CohortMonitor } from 'churn-harbinger';

const cohort = new CohortMonitor();
await cohort.init();

// Track behavior distributions
// [feature_a, feature_b, feature_c, ...]
cohort.addUser('user-123', { plan: 'pro' });
cohort.updateUser('user-123', behaviorDistribution);

// Detect divergence from healthy patterns
const divergence = cohort.getDivergence('user-123', 'healthy-benchmark');
if (divergence > 0.5) {
  console.log('User diverging from healthy cohort');
}
```

## Integration Examples

### Segment

```javascript
analytics.track('Daily Engagement', { score: engagementScore });

// In your backend
segment.on('Daily Engagement', async (event) => {
  const state = detector.update(event.properties.score);
  if (state.atRisk) {
    segment.identify(event.userId, {
      churnRisk: state.riskLevel,
      churnConfidence: state.confidence
    });
  }
});
```

### Mixpanel

```javascript
// Pull engagement data
const events = await mixpanel.query(`
  SELECT DATE(time) as day, COUNT(*) as actions
  FROM events
  WHERE user_id = '${userId}'
  GROUP BY DATE(time)
  ORDER BY day
`);

const scores = events.map(e => e.actions);
const risk = await assessChurnRisk(scores);
```

### Customer.io

```javascript
if (state.atRisk) {
  customerio.track(userId, 'churn_risk_detected', {
    level: state.riskLevel,
    confidence: state.confidence,
    trend: state.trend
  });
  
  // Trigger win-back campaign
  customerio.trigger(userId, 'winback_campaign');
}
```

## API Reference

### ChurnDetector

```javascript
const detector = new ChurnDetector(config?);
await detector.init();

detector.update(engagementScore);    // Process single day
detector.updateBatch(scores);         // Process batch
detector.current();                   // Get current state
detector.reset();                     // Reset (e.g., after re-engagement)
detector.serialize();                 // Persist state
ChurnDetector.deserialize(json);      // Restore state
```

### ChurnState

```typescript
{
  riskLevel: 'healthy' | 'cooling' | 'at-risk' | 'churning',
  atRisk: boolean,       // High churn probability
  declining: boolean,    // Engagement trending down
  confidence: number,    // 0-1 confidence score
  variance: number,      // Current engagement variance
  trend: number,         // Trend indicator
  dataPoints: number     // Total observations
}
```

## Use Cases

- **CSM Prioritization**: Focus team on users showing pre-churn patterns
- **Automated Outreach**: Trigger win-back campaigns at optimal timing
- **Product Alerts**: Notify PM when feature engagement shows churn signals
- **Pricing Interventions**: Offer discounts before users decide to leave
- **Health Scoring**: Add variance-based component to customer health scores

## Performance

- ~50KB WASM bundle
- Sub-millisecond per observation
- Suitable for real-time event streams
- Runs in Node.js, browsers, edge workers

## The Math

Users approaching churn exhibit "critical slowing down" — the same phase transition dynamic that predicts:
- Financial market regime changes
- Infrastructure failures
- Social system conflicts

The variance drop before major transitions is a universal signature of systems approaching critical points.

## License

MIT

## Author

Ryan Cardwell ([@Benthic_Shadow](https://x.com/Benthic_Shadow))

## See Also

- [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) - Core detection library
- [regime-shift](https://www.npmjs.com/package/regime-shift) - Financial market version
- [threat-pulse](https://www.npmjs.com/package/threat-pulse) - Security version
- [GitHub](https://github.com/aphoticshaman/nucleation-packages)
