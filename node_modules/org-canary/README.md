# org-canary

Detect organizational dysfunction before it surfaces.

Culture clash prediction, M&A integration risk, and team health monitoring. The core insight: **organizational conflict builds through measurable tension patterns** — teams approaching dysfunction show characteristic variance changes before problems become visible.

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm).

## Installation

```bash
npm install org-canary
```

## Quick Start

```javascript
import { TeamHealthMonitor } from 'org-canary';

const monitor = new TeamHealthMonitor();
await monitor.init();

for (const week of teamMetrics) {
  const state = monitor.update(week.healthScore);
  
  if (state.stressed) {
    alertHRBP({
      team: team.id,
      level: state.healthLevel,
      confidence: state.confidence
    });
  }
}
```

## Use Cases

- **Team Health**: Monitor engagement surveys, collaboration metrics, sentiment
- **M&A Integration**: Predict culture clash between merging organizations
- **Reorg Planning**: Identify teams at risk during organizational change
- **Turnover Prediction**: Detect team-level disengagement before attrition spikes

## Health Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `thriving` | Healthy dynamics | Continue current approach |
| `strained` | Some tension | Monitor, check in with leads |
| `stressed` | Dysfunction risk | HRBP intervention |
| `critical` | Active dysfunction | Immediate action required |

## M&A Integration Risk

```javascript
import { IntegrationMonitor } from 'org-canary';

const monitor = new IntegrationMonitor(6); // 6 culture dimensions
await monitor.init();

// Register organizations with culture profiles
// [innovation, hierarchy, collaboration, risk-tolerance, pace, formality]
monitor.registerEntity('acquirer', { name: 'BigCorp' });
monitor.registerEntity('target', { name: 'Startup' });

monitor.updateEntity('acquirer', new Float64Array([0.1, 0.3, 0.2, 0.1, 0.2, 0.1]));
monitor.updateEntity('target', new Float64Array([0.3, 0.1, 0.3, 0.2, 0.05, 0.05]));

const risk = monitor.getClashRisk('acquirer', 'target');
// Higher values = more culture clash risk
```

## API

### TeamHealthMonitor

```javascript
const monitor = new TeamHealthMonitor({
  sensitivity: 'balanced',  // 'conservative', 'balanced', 'sensitive'
  windowSize: 12,           // Weeks for baseline
  threshold: 2.0
});

await monitor.init();
monitor.update(healthScore);
monitor.updateBatch(scores);
monitor.current();
monitor.reset();
```

### IntegrationMonitor

```javascript
const monitor = new IntegrationMonitor(dimensions);
await monitor.init();

monitor.registerEntity(id, metadata, cultureProfile);
monitor.updateEntity(id, cultureMetrics, timestamp);
monitor.getClashRisk(entityA, entityB);
monitor.checkAllPairs();
```

## License

MIT

## Author

Ryan Cardwell ([@Benthic_Shadow](https://x.com/Benthic_Shadow))
