# threat-pulse

Detect threat escalation before attacks materialize.

Behavioral variance analysis for SOC teams, SIEM integration, and proactive threat hunting. The core insight: **attacker reconnaissance creates a characteristic "quieting" before major actions** — reduced variance as they zero in on targets.

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) for high-performance WebAssembly execution.

## Installation

```bash
npm install threat-pulse
```

## Quick Start

```javascript
import { ThreatDetector } from 'threat-pulse';

const detector = new ThreatDetector();
await detector.init();

// Feed normalized anomaly scores from your SIEM
for (const event of siemEvents) {
  const state = detector.update(event.anomalyScore);
  
  if (state.escalating) {
    triggerSOCAlert({
      level: state.threatLevel,
      confidence: state.confidence,
      message: 'Threat escalation pattern detected'
    });
  }
}
```

## One-liner

```javascript
import { assessThreat } from 'threat-pulse';

const result = await assessThreat(anomalyScores);
// { escalating: true, threatLevel: 'red', confidence: 0.87 }
```

## How It Works

Traditional threat detection looks for anomaly spikes. By then, the attack is underway.

This library detects the **precursor signal**: the characteristic variance drop that precedes sophisticated attacks. Attackers conducting reconnaissance generate noise. As they identify and focus on targets, that noise decreases — the "calm before the storm."

This is the same phase transition dynamic that predicts:
- Financial market regime changes
- Infrastructure failures  
- Social system conflicts

## Threat Levels

| Level | Meaning | Recommended Action |
|-------|---------|-------------------|
| `green` | Normal activity | Continue monitoring |
| `yellow` | Elevated patterns | Increase logging, review recent events |
| `orange` | High probability of imminent threat | Alert on-call, prepare response |
| `red` | Active escalation in progress | Initiate incident response |

## Configuration

```javascript
const detector = new ThreatDetector({
  sensitivity: 'balanced',   // 'conservative', 'balanced', 'aggressive'
  windowSize: 50,            // Events for baseline calculation
  threshold: 2.0             // Standard deviations for alert
});
```

| Sensitivity | Use Case |
|-------------|----------|
| `conservative` | High-value targets, minimize false positives |
| `balanced` | General security monitoring |
| `aggressive` | Early warning, accept more false positives |

## Multi-Source Correlation

Detect coordinated attacks across multiple event sources:

```javascript
import { ThreatCorrelator } from 'threat-pulse';

const correlator = new ThreatCorrelator();
await correlator.init();

// Register event sources
correlator.registerSource('firewall');
correlator.registerSource('ids');
correlator.registerSource('endpoint');

// Update with behavioral distributions
correlator.updateSource('firewall', firewallBehavior, timestamp);
correlator.updateSource('endpoint', endpointBehavior, timestamp);

// Check for divergence (potential lateral movement)
const divergence = correlator.getCorrelation('firewall', 'endpoint');
if (divergence > 0.5) {
  console.log('ALERT: Endpoint diverging from perimeter');
}
```

## SIEM Integration

### Splunk

```javascript
// Pull anomaly scores from Splunk search
const events = await splunk.search('index=security | stats avg(risk_score) by _time');
const scores = events.map(e => e.risk_score);

const result = await assessThreat(scores);
if (result.elevated) {
  await splunk.createNotable('Threat Escalation Detected', result);
}
```

### Elastic SIEM

```javascript
// Subscribe to detection alerts
elastic.subscribe('siem-alerts', async (alert) => {
  const state = detector.update(alert.risk_score);
  
  if (state.threatLevel !== 'green') {
    await elastic.index('threat-pulse-alerts', {
      original_alert: alert,
      escalation: state
    });
  }
});
```

## API Reference

### ThreatDetector

```javascript
const detector = new ThreatDetector(config?);
await detector.init();

detector.update(anomalyScore);      // Process single event
detector.updateBatch(scores);        // Process batch
detector.current();                  // Get current state
detector.reset();                    // Reset after incident
detector.serialize();                // Persist state
ThreatDetector.deserialize(json);    // Restore state
```

### ThreatState

```typescript
{
  threatLevel: 'green' | 'yellow' | 'orange' | 'red',
  escalating: boolean,    // Active escalation detected
  elevated: boolean,      // Above baseline
  confidence: number,     // 0-1 confidence score
  variance: number,       // Current behavioral variance
  deviation: number,      // Deviation from baseline (z-score)
  eventCount: number      // Total events processed
}
```

## Use Cases

- **APT Detection**: Identify reconnaissance patterns before exploitation
- **Insider Threat**: Detect behavioral changes indicating compromise
- **Lateral Movement**: Correlate diverging behavior across network segments
- **UEBA Enhancement**: Add phase transition detection to user analytics
- **IR Prioritization**: Focus on events showing escalation patterns

## Performance

- ~50KB WASM bundle
- Sub-millisecond per event
- Suitable for real-time stream processing
- Runs in Node.js, browsers, edge workers

## The EOD Doctrine

This library is built on 15 years of US Army EOD (Explosive Ordnance Disposal) doctrine: **address threats before they manifest**. The same principles that guide bomb disposal — reading precursor signals, understanding escalation patterns, acting before detonation — apply to cyber threats.

## License

MIT

## Author

Ryan Cardwell ([@Benthic_Shadow](https://x.com/Benthic_Shadow))

Former US Army EOD. Current threat researcher.

## See Also

- [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) - Core detection library
- [regime-shift](https://www.npmjs.com/package/regime-shift) - Financial market version
- [GitHub](https://github.com/aphoticshaman/nucleation-packages)
