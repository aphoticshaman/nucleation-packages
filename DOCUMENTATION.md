# Nucleation Packages - Complete Documentation

## Overview

A suite of 10 domain-specific phase transition detectors built on a single core algorithm: **variance-based early warning systems**.

The core insight: **variance decreases before major transitions** ("calm before the storm"). This pattern appears across financial markets, security threats, user behavior, organizational dynamics, physical systems, and more.

---

## Architecture

```
nucleation-wasm (Rust/WASM core)
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                    Domain-Specific Wrappers                     │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│ regime-shift │ threat-pulse │ churn-       │ org-canary       │
│ (Finance)    │ (Security)   │ harbinger    │ (HR/M&A)         │
│              │              │ (SaaS)       │                  │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ supply-      │ sensor-shift │ crowd-phase  │ patient-drift    │
│ sentinel     │ (IoT)        │ (Social)     │ (Healthcare)     │
│ (Logistics)  │              │              │                  │
├──────────────┼──────────────┼──────────────┴──────────────────┤
│ match-pulse  │ market-      │                                 │
│ (Gaming)     │ canary       │                                 │
│              │ (General)    │                                 │
└──────────────┴──────────────┴─────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│                    Unified Harness                              │
│  • Real-time data fetching (CoinGecko, USGS, etc.)             │
│  • Cross-domain correlation                                     │
│  • World trajectory monitoring                                  │
│  • Ablation testing framework                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## Package Reference

### 1. regime-shift (Finance/Quant)
**npm:** `npm install regime-shift`

Detects market regime changes before they fully manifest.

```javascript
import { RegimeDetector } from 'regime-shift';

const detector = new RegimeDetector({ sensitivity: 'balanced' });
await detector.init();

for (const price of priceHistory) {
  const state = detector.update(price);
  if (state.isShifting) {
    console.log(`Regime shift: ${state.regime}`);
  }
}
```

**States:** `stable` → `warming` → `critical` → `shifting`

---

### 2. threat-pulse (Cybersecurity)
**npm:** `npm install threat-pulse`

Behavioral analytics for SOC teams. Detects threat escalation patterns.

```javascript
import { ThreatDetector } from 'threat-pulse';

const detector = new ThreatDetector({ sensitivity: 'aggressive' });
await detector.init();

const state = detector.update(anomalyScore);
if (state.escalating) {
  triggerSOCAlert(state.threatLevel);
}
```

**Levels:** `green` → `yellow` → `orange` → `red`

---

### 3. churn-harbinger (SaaS/Product)
**npm:** `npm install churn-harbinger`

Predicts customer churn through engagement variance analysis.

```javascript
import { ChurnDetector } from 'churn-harbinger';

const detector = new ChurnDetector({ windowSize: 30 });
await detector.init();

const state = detector.update(dailyEngagement);
if (state.atRisk) {
  triggerCSMOutreach(userId);
}
```

**Levels:** `healthy` → `cooling` → `at-risk` → `churning`

---

### 4. org-canary (HR/M&A)
**npm:** `npm install org-canary`

Culture clash detection and team health monitoring.

```javascript
import { TeamHealthMonitor, IntegrationMonitor } from 'org-canary';

// Team health
const monitor = new TeamHealthMonitor();
await monitor.init();
const state = monitor.update(weeklyHealthScore);

// M&A integration risk
const integration = new IntegrationMonitor(6);
await integration.init();
integration.registerEntity('acquirer', {}, acquirerCulture);
integration.registerEntity('target', {}, targetCulture);
const clashRisk = integration.getClashRisk('acquirer', 'target');
```

**Levels:** `thriving` → `strained` → `stressed` → `critical`

---

### 5. supply-sentinel (Supply Chain)
**npm:** `npm install supply-sentinel`

Early warning for supply chain disruptions.

```javascript
import SupplyMonitor from 'supply-sentinel';

const monitor = new SupplyMonitor({ windowSize: 30 });
await monitor.init();
const state = monitor.update(supplierReliabilityScore);
```

**Levels:** `stable` → `elevated` → `critical` → `disrupted`

---

### 6. sensor-shift (IoT/Manufacturing)
**npm:** `npm install sensor-shift`

Predictive maintenance through sensor variance analysis.

```javascript
import { SensorMonitor } from 'sensor-shift';

const monitor = new SensorMonitor({ sensitivity: 'sensitive' });
await monitor.init();
const state = monitor.update(sensorReading);
if (state.failing) {
  scheduleMaintenanceNow();
}
```

**Levels:** `normal` → `degrading` → `warning` → `failing`

---

### 7. crowd-phase (Social/Moderation)
**npm:** `npm install crowd-phase`

Community conflict detection for platforms.

```javascript
import { CrowdMonitor } from 'crowd-phase';

const monitor = new CrowdMonitor({ windowSize: 20 });
await monitor.init();
const state = monitor.update(communityTensionScore);
if (state.volatile) {
  alertModerationTeam();
}
```

**Levels:** `calm` → `tense` → `heated` → `volatile`

---

### 8. patient-drift (Healthcare)
**npm:** `npm install patient-drift`

Patient deterioration early warning.

```javascript
import { PatientMonitor } from 'patient-drift';

const monitor = new PatientMonitor({ sensitivity: 'sensitive' });
await monitor.init();
const state = monitor.update(vitalSignsScore);
if (state.critical) {
  alertClinicalTeam();
}
```

**Levels:** `stable` → `watch` → `warning` → `critical`

---

### 9. match-pulse (Gaming/Esports)
**npm:** `npm install match-pulse`

Player tilt and match quality detection.

```javascript
import { MatchMonitor } from 'match-pulse';

const monitor = new MatchMonitor();
await monitor.init();
const state = monitor.update(playerPerformanceScore);
if (state.tilted) {
  suggestBreak(playerId);
}
```

**Levels:** `focused` → `frustrated` → `tilted` → `toxic`

---

### 10. market-canary (General Analytics)
**npm:** `npm install market-canary`

General-purpose phase transition detection for any time series.

```javascript
import { TransitionDetector } from 'market-canary';

const detector = new TransitionDetector({ windowSize: 50 });
await detector.init();
const state = detector.update(timeSeriesValue);
```

**Phases:** `stable` → `approaching` → `critical` → `transitioning`

---

## Unified Harness

The `harness/` directory contains a unified system that:

1. **Fetches real data** from free public APIs:
   - CoinGecko (crypto prices)
   - Alternative.me (Fear & Greed Index)
   - USGS (earthquake data)

2. **Runs all detectors** simultaneously across domains

3. **Correlates cross-domain signals** to identify systemic risk

4. **Generates reports** with overall risk assessment

### Usage

```bash
cd harness
npm install
node src/index.js          # Single snapshot
node src/ablation-tests.js # Run validation tests
```

### Continuous Monitoring

```javascript
import { runMonitor } from './harness/src/index.js';

// Update every 60 minutes
runMonitor(60);
```

---

## Ablation Test Results

Tested all 10 detectors against 5 synthetic patterns:
- `calm-before-storm` (the target signal)
- `gradual-decline`
- `stable` (false positive test)
- `sudden-spike` (no precursor)
- `oscillating`

### Key Findings

| Detector | Best Sensitivity | Detection Quality |
|----------|------------------|-------------------|
| TeamHealthMonitor | balanced | GOOD (detects during calm) |
| CrowdMonitor | balanced | GOOD |
| PatientMonitor | balanced/sensitive | GOOD |
| MatchMonitor | balanced | GOOD (slightly late) |
| RegimeDetector | sensitive | LATE |
| ThreatDetector | all | MISSED (needs tuning) |
| ChurnDetector | sensitive | LATE |
| TransitionDetector | all | MISSED (needs tuning) |
| SensorMonitor | all | MISSED (needs tuning) |
| SupplyMonitor | sensitive | LATE |

### False Positive Rate

**Zero false positives** on stable data across all detectors at all sensitivity levels.

### Recommendations

1. **For production use:** Start with `balanced` sensitivity
2. **For early warning:** Use `sensitive` but expect more noise
3. **For high-stakes:** Use `conservative` + manual review
4. **Tune window size:** Smaller windows = faster detection, more noise

---

## Configuration Reference

All detectors accept these common options:

```javascript
{
  sensitivity: 'conservative' | 'balanced' | 'sensitive',
  windowSize: number,    // Rolling window for variance calculation
  threshold: number      // Z-score threshold for alerts
}
```

### Default Window Sizes by Domain

| Package | Default Window | Rationale |
|---------|----------------|-----------|
| regime-shift | 30 | Monthly patterns |
| threat-pulse | 50 | Baseline establishment |
| churn-harbinger | 30 | Monthly engagement |
| org-canary | 12 | Quarterly HR cycles |
| supply-sentinel | 30 | Monthly supply patterns |
| sensor-shift | 50 | Equipment baseline |
| crowd-phase | 20 | Community dynamics |
| patient-drift | 12 | Clinical observation |
| match-pulse | 15 | Gaming sessions |
| market-canary | 50 | General analysis |

---

## API Patterns

All detectors follow the same API:

```javascript
// Initialize
const detector = new Detector(config);
await detector.init();

// Stream processing
const state = detector.update(value);

// Batch processing
const state = detector.updateBatch(values);

// Current state without new data
const state = detector.current();

// Reset
detector.reset();

// Persistence
const json = detector.serialize();
const restored = await Detector.deserialize(json);
```

---

## Performance

- **Bundle size:** ~50KB WASM per detector
- **Latency:** Sub-millisecond per observation
- **Memory:** Minimal (rolling window only)
- **Runtime:** Node.js, browsers, edge workers

---

## Author

Ryan Cardwell ([@Benthic_Shadow](https://x.com/Benthic_Shadow))

15 years US Army EOD. The doctrine of addressing threats before they manifest.

---

## License

MIT
