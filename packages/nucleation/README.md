# Nucleation: 5-Minute Quickstart

Detect phase transitions before they happen. The calm before the storm.

## Install

```bash
npm install nucleation
```

## 1. Basic Usage (30 seconds)

```javascript
import { monitor } from 'nucleation';

// Create a detector
const detector = await monitor('finance');

// Feed it data
detector.update(100.5);
detector.update(101.2);
detector.update(99.8);

// Check state
const state = detector.current();
console.log(state.level);        // 'stable'
console.log(state.transitioning); // false
```

## 2. With Alerts (1 minute)

```javascript
import { monitor } from 'nucleation';

const detector = await monitor('finance');

// Get notified on state changes
detector.on('warning', state => {
  console.log('⚠️ Warning:', state.level);
});

detector.on('critical', state => {
  console.log('🚨 Critical:', state.level);
  // Send to Slack, PagerDuty, etc.
});

// Process your data stream
for (const price of priceStream) {
  detector.update(price);
}
```

## 3. Real-Time Crypto Monitor (2 minutes)

```javascript
import { monitor } from 'nucleation';

const detector = await monitor('finance', { sensitivity: 'balanced' });

detector.on('warning', state => {
  console.log(`⚠️ BTC regime change approaching (confidence: ${(state.confidence * 100).toFixed(0)}%)`);
});

detector.on('critical', state => {
  console.log(`🚨 BTC REGIME SHIFT: ${state.level}`);
});

// Fetch and monitor Bitcoin
async function run() {
  const res = await fetch('https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30');
  const data = await res.json();
  
  // Convert to returns (what we actually detect on)
  const prices = data.prices.map(p => p[1]);
  for (let i = 1; i < prices.length; i++) {
    const logReturn = Math.log(prices[i] / prices[i-1]);
    const state = detector.update(logReturn);
    
    if (state.transitioning) {
      console.log(`Day ${i}: ${state.level} (variance: ${state.variance?.toFixed(6)})`);
    }
  }
  
  console.log('\nFinal state:', detector.current());
}

run();
```

## 4. Webhook Server (2 minutes)

Receive data via HTTP, get alerts automatically:

```javascript
import { createWebhookProcessor } from 'nucleation';

const processor = createWebhookProcessor({
  domain: 'finance',
  port: 8080,
  sensitivity: 'balanced',
  
  // Extract value from incoming JSON
  extract: data => data.price,
  
  // Called on warning or critical
  onAlert: async (state) => {
    await fetch('https://hooks.slack.com/services/YOUR/WEBHOOK/URL', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: `🚨 Nucleation Alert: ${state.level} (confidence: ${(state.confidence * 100).toFixed(0)}%)`
      })
    });
  }
});

processor.start();
// Now POST to http://localhost:8080 with { "price": 100.5 }
```

## 5. Multiple Domains (1 minute)

```javascript
import { createMonitors } from 'nucleation';

const monitors = await createMonitors({
  finance: { sensitivity: 'balanced' },
  security: { sensitivity: 'sensitive' },
  social: { sensitivity: 'conservative' }
});

// Each has the same API
monitors.finance.on('warning', handleFinanceWarning);
monitors.security.on('critical', handleSecurityCritical);
monitors.social.on('transition', handleSocialTransition);

// Update independently
monitors.finance.update(priceData);
monitors.security.update(threatScore);
monitors.social.update(sentimentValue);
```

## 6. Prometheus Metrics (1 minute)

```javascript
import { monitor, createPrometheusExporter } from 'nucleation';
import { createServer } from 'http';

const detector = await monitor('finance');
const exporter = createPrometheusExporter(detector, {
  prefix: 'trading_',
  labels: { asset: 'btc', env: 'prod' }
});

// Your existing metrics endpoint
createServer((req, res) => {
  if (req.url === '/metrics') {
    res.setHeader('Content-Type', 'text/plain');
    res.end(exporter.metrics());
  }
}).listen(9090);

// Output:
// trading_level{asset="btc",env="prod"} 0
// trading_variance{asset="btc",env="prod"} 0.00234
// trading_confidence{asset="btc",env="prod"} 0.87
// trading_transitioning{asset="btc",env="prod"} 0
```

## Available Domains

| Domain | Use Case | Install Standalone |
|--------|----------|-------------------|
| `finance` | Market regime detection | `npm i regime-shift` |
| `security` | Threat escalation | `npm i threat-pulse` |
| `saas` / `churn` | Customer churn prediction | `npm i churn-harbinger` |
| `hr` / `org` | Team health, M&A risk | `npm i org-canary` |
| `supply` | Supply chain disruption | `npm i supply-sentinel` |
| `iot` / `sensor` | Predictive maintenance | `npm i sensor-shift` |
| `social` / `community` | Community conflict | `npm i crowd-phase` |
| `health` / `patient` | Patient deterioration | `npm i patient-drift` |
| `gaming` / `esports` | Player tilt detection | `npm i match-pulse` |
| `general` | Any time series | `npm i market-canary` |

## Configuration

```javascript
const detector = await monitor('finance', {
  sensitivity: 'balanced',  // 'conservative' | 'balanced' | 'sensitive'
  windowSize: 30,           // Rolling window for variance calculation
  threshold: 2.0,           // Z-score threshold for alerts
});
```

## State Shape

Every detector returns the same normalized state:

```javascript
{
  level: 'stable',           // Human-readable level
  levelNumeric: 0,           // 0=green, 1=yellow, 2=orange, 3=red
  transitioning: false,      // Is a phase shift occurring?
  confidence: 0.87,          // Detection confidence (0-1)
  variance: 0.00234,         // Current variance value
  timestamp: 1701408127901,  // Unix timestamp
  raw: { ... }               // Original detector output
}
```

## The Algorithm (30 seconds)

Variance decreases before major transitions. 

```
Normal:     ~~~~~~~~~~~  (high variance)
Pre-storm:  ___________  (low variance) ← DETECT HERE
Storm:      ∿∿∿∿∿∿∿∿∿∿∿  (explosion)
```

This pattern appears in:
- Financial markets before crashes
- Network traffic before attacks  
- User behavior before churn
- Sensor readings before equipment failure
- Social dynamics before conflict

Nucleation detects the calm before the storm.

## Next Steps

- [Full Documentation](https://github.com/aphoticshaman/nucleation-packages)
- [Why This Works (Blog Post)](https://github.com/aphoticshaman/nucleation-packages/blob/main/docs/WHY_THIS_WORKS.md)
- [Comparison vs Alternatives](https://github.com/aphoticshaman/nucleation-packages/blob/main/docs/COMPARISON.md)

## License

MIT
