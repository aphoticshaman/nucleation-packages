# sensor-shift

Predict equipment failures before they happen.

Sensor anomaly detection for predictive maintenance. The core insight: **sensor readings stabilize before failure** — vibration variance drops as bearings seize, temperature variance drops as cooling fails.

## Quick Start

```javascript
import { FailureDetector } from 'sensor-shift';

const detector = new FailureDetector();
await detector.init();

sensorStream.on('reading', (value) => {
  const state = detector.update(value);
  if (state.degraded) {
    scheduleMaintenanceInspection(assetId);
  }
});
```

## Health Levels

| Level | Meaning |
|-------|---------|
| `normal` | Operating within spec |
| `degrading` | Variance pattern changing |
| `warning` | Failure likely imminent |
| `failing` | Active failure mode |

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm). MIT License.
