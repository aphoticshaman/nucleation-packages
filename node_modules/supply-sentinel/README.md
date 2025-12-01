# supply-sentinel

Predict supply chain disruptions before they cascade.

Supplier risk monitoring, logistics early warning, and resilience scoring using variance analysis. The core insight: **supply chains show warning signs before failure** — lead time variance drops as suppliers max out capacity, then disruption hits.

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm).

## Installation

```bash
npm install supply-sentinel
```

## Quick Start

```javascript
import { DisruptionDetector } from 'supply-sentinel';

const detector = new DisruptionDetector();
await detector.init();

for (const shipment of recentShipments) {
  const state = detector.update(shipment.leadTime);
  
  if (state.elevated) {
    alertProcurement({
      supplier: shipment.supplierId,
      riskLevel: state.riskLevel
    });
  }
}
```

## Risk Levels

| Level | Meaning | Action |
|-------|---------|--------|
| `stable` | Normal variance | Continue monitoring |
| `elevated` | Variance pattern changing | Increase safety stock |
| `high` | Disruption likely | Activate backup suppliers |
| `critical` | Disruption in progress | Emergency procurement |

## Multi-Supplier Networks

```javascript
import { SupplierNetwork } from 'supply-sentinel';

const network = new SupplierNetwork();
await network.init();

network.registerSupplier('primary', { tier: 1 });
network.registerSupplier('backup', { tier: 2 });

// Check for correlated risk
const cascadeRisk = network.getCascadeRisk('primary', 'backup');
```

## Use Cases

- **Lead time monitoring**: Detect supplier stress before delays
- **Quality variance**: Predict quality issues from inspection data
- **Demand spikes**: Identify unusual order pattern changes
- **Multi-tier visibility**: Monitor cascade risk across supplier network

## License

MIT
