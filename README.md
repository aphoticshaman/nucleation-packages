# Nucleation

**Early warning systems for phase transitions. Detect the calm before the storm.**

[![npm version](https://img.shields.io/npm/v/nucleation.svg)](https://www.npmjs.com/package/nucleation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Traditional monitoring alerts **after** something breaks:

```
Normal → Normal → Normal → 🔥 FIRE! → Alert
                                       ↑
                                       Too late
```

## The Solution

Nucleation detects the **quiet period before transitions**:

```
Normal → Normal → 😶 Quiet... → Alert → (Fire happens later)
                   ↑
                   Early warning
```

This pattern—variance decreasing before major transitions—appears in financial markets, network security, user behavior, physical systems, and more. It's physics, not magic.

---

## Quick Start

```bash
npm install nucleation
```

```javascript
import { monitor } from 'nucleation';

const detector = await monitor('finance');

detector.on('warning', state => {
  console.log('⚠️ Transition approaching:', state.level);
});

detector.on('critical', state => {
  console.log('🚨 Transition imminent:', state.level);
  // Send to Slack, PagerDuty, etc.
});

// Feed your data
for (const price of priceStream) {
  detector.update(price);
}
```

[Full Quickstart Guide →](packages/nucleation/README.md)

---

## Use Cases

| Domain | Package | Detects |
|--------|---------|---------|
| **Finance** | `regime-shift` | Market regime changes before volatility spikes |
| **Security** | `threat-pulse` | Threat escalation before attacks |
| **SaaS** | `churn-harbinger` | Customer churn before they leave |
| **HR/M&A** | `org-canary` | Culture clash before it explodes |
| **Supply Chain** | `supply-sentinel` | Disruptions before they hit |
| **IoT** | `sensor-shift` | Equipment failure before breakdown |
| **Social** | `crowd-phase` | Community conflict before blowup |
| **Healthcare** | `patient-drift` | Patient deterioration before crisis |
| **Gaming** | `match-pulse` | Player tilt before toxicity |
| **General** | `market-canary` | Any time series |

---

## Installation Options

### Meta-package (recommended)

```bash
npm install nucleation
```

Includes all detectors with unified API.

### Individual packages

```bash
npm install regime-shift    # Finance
npm install threat-pulse    # Security
npm install churn-harbinger # SaaS
# etc.
```

### Core only

```bash
npm install nucleation-wasm
```

Just the WASM algorithm, build your own wrapper.

---

## API

### Zero-config

```javascript
import { monitor } from 'nucleation';

const detector = await monitor('finance');
detector.on('warning', handleWarning);
detector.update(value);
```

### Full control

```javascript
import { RegimeDetector } from 'nucleation';

const detector = new RegimeDetector({
  sensitivity: 'sensitive',  // 'conservative' | 'balanced' | 'sensitive'
  windowSize: 20,
  threshold: 1.5,
});

await detector.init();
const state = detector.update(value);
```

### Webhook server

```javascript
import { createWebhookProcessor } from 'nucleation';

createWebhookProcessor({
  domain: 'finance',
  port: 8080,
  onAlert: state => sendSlack(state),
}).start();

// POST { "value": 100.5 } to :8080
```

### Prometheus export

```javascript
import { monitor, createPrometheusExporter } from 'nucleation';

const detector = await monitor('finance');
const exporter = createPrometheusExporter(detector);

app.get('/metrics', (req, res) => res.send(exporter.metrics()));
```

---

## Output Format

All detectors return normalized state:

```javascript
{
  level: 'warning',          // Human-readable
  levelNumeric: 2,           // 0=green, 1=yellow, 2=orange, 3=red
  transitioning: true,       // Is transition occurring?
  confidence: 0.87,          // Detection confidence (0-1)
  variance: 0.00234,         // Current variance
  timestamp: 1701408127901,
  raw: { ... }               // Original detector output
}
```

---

## Why This Works

Variance decreases before major transitions. The system gets quieter before it explodes.

This is documented in:
- Financial econometrics (pre-crash volatility compression)
- Complex systems theory (critical slowing down)
- Ecology (ecosystem tipping points)
- Network security (pre-attack synchronization)

[Read the full explanation →](docs/WHY_THIS_WORKS.md)

---

## Comparison

| vs | Nucleation wins when... | Alternative wins when... |
|----|------------------------|-------------------------|
| **Datadog** | Early warning, cost, edge deployment | Unified observability, enterprise support |
| **Prometheus** | Out-of-box detection, simplicity | General monitoring, PromQL flexibility |
| **Custom ML** | Speed to deploy, no training data | Unique patterns, high accuracy needs |
| **DIY** | Time saved, tested edge cases | Specific requirements, learning |

[Full comparison →](docs/COMPARISON.md)

---

## Architecture

```
nucleation-wasm           ← Core algorithm (Rust/WASM)
├── regime-shift          ← Finance
├── threat-pulse          ← Security
├── churn-harbinger       ← SaaS
├── org-canary            ← HR/M&A
├── supply-sentinel       ← Supply Chain
├── sensor-shift          ← IoT
├── crowd-phase           ← Social
├── patient-drift         ← Healthcare
├── match-pulse           ← Gaming
├── market-canary         ← General
└── nucleation            ← Meta-package (unified API)
```

---

## Performance

- **Latency:** <1ms per observation
- **Memory:** ~10MB including WASM
- **Bundle:** ~50KB per detector
- **Cold start:** <100ms WASM initialization
- **Runtime:** Node.js 18+, browsers, Cloudflare Workers, Deno

---

## Docs

- [5-Minute Quickstart](packages/nucleation/README.md)
- [Why This Works (The Math)](docs/WHY_THIS_WORKS.md)
- [Comparison vs Alternatives](docs/COMPARISON.md)
- [Full API Reference](DOCUMENTATION.md)

---

## Contributing

Issues and PRs welcome. This is early-stage—feedback shapes direction.

---

## Author

[@Benthic_Shadow](https://x.com/Benthic_Shadow)

The doctrine of addressing threats before they manifest.

[GitHub](https://github.com/aphoticshaman)
---

## License

MIT
