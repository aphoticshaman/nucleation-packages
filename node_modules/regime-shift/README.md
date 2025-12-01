# regime-shift

Detect market regime changes before they happen.

Uses variance-based phase transition detection to identify regime shifts in price data, returns, or any financial time series. The core insight: **variance typically decreases before major market transitions** (the "calm before the storm").

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) for high-performance WebAssembly execution.

## Installation

```bash
npm install regime-shift
```

## Quick Start

```javascript
import { RegimeDetector } from 'regime-shift';

const detector = new RegimeDetector();
await detector.init();

// Feed price data
for (const price of closingPrices) {
  const state = detector.update(price);
  
  if (state.isShifting) {
    console.log('Regime shift detected!');
    console.log(`Confidence: ${state.confidence}`);
  }
}
```

## One-liner

```javascript
import { detectRegimeShift } from 'regime-shift';

const result = await detectRegimeShift(closingPrices);
// { shifting: true, warning: false, regime: 'shifting', confidence: 0.87 }
```

## How It Works

Traditional regime detection looks for volatility spikes. By then, it's too late.

This library detects the **precursor signal**: the characteristic variance drop that precedes most major market transitions. It's the mathematical signature of the "calm before the storm."

The detector tracks:
- **Rolling variance** of your input series
- **Variance inflection** (second derivative) to catch the transition point
- **Confidence scoring** based on signal strength

## Regimes

| Regime | Meaning |
|--------|---------|
| `stable` | Normal market conditions, no shift imminent |
| `warming` | Variance pattern changing, potential shift ahead |
| `critical` | High probability of imminent regime change |
| `shifting` | Regime change in progress |

## Configuration

```javascript
const detector = new RegimeDetector({
  sensitivity: 'balanced',  // 'conservative', 'balanced', or 'sensitive'
  windowSize: 50,           // Rolling window for variance calculation
  threshold: 2.0            // Z-score threshold for detection
});
```

| Sensitivity | Use Case |
|-------------|----------|
| `conservative` | Fewer false positives, may miss subtle shifts |
| `balanced` | Good general-purpose setting |
| `sensitive` | Catches early signals, more false positives |

## API

### `RegimeDetector`

```javascript
const detector = new RegimeDetector(config?);
await detector.init();

// Stream processing
const state = detector.update(price);

// Batch processing
const state = detector.updateBatch(prices);

// Current state without new data
const state = detector.current();

// Reset
detector.reset();

// Persistence
const json = detector.serialize();
const restored = await RegimeDetector.deserialize(json);
```

### `RegimeState`

```typescript
{
  regime: 'stable' | 'warming' | 'critical' | 'shifting',
  isShifting: boolean,    // True if regime change detected
  isWarning: boolean,     // True if shift approaching
  confidence: number,     // 0-1 confidence score
  variance: number,       // Current rolling variance
  inflection: number,     // Variance inflection (z-score)
  observations: number    // Total data points processed
}
```

## Use Cases

- **Risk management**: Adjust position sizing when regime shift is imminent
- **Strategy switching**: Transition between trend-following and mean-reversion
- **Volatility trading**: Time VIX entries based on regime state
- **Portfolio rebalancing**: Trigger rebalance on regime change
- **Alerting**: Notify traders of changing market conditions

## Performance

- ~50KB WASM bundle
- Sub-millisecond per observation
- Runs in Node.js, browsers, and edge workers

## Example: Trading Strategy

```javascript
import { RegimeDetector } from 'regime-shift';

const detector = new RegimeDetector({ sensitivity: 'balanced' });
await detector.init();

function onNewPrice(price) {
  const state = detector.update(price);
  
  if (state.regime === 'stable') {
    // Normal operations
    executeTrendStrategy();
  } else if (state.isWarning) {
    // Reduce exposure
    reducePositionSize(0.5);
  } else if (state.isShifting) {
    // Hedge or exit
    closePositions();
    waitForNewRegime();
  }
}
```

## Validation

The underlying algorithm achieves F1 = 0.77 on phase transition detection across multiple domains, with particularly strong performance on "calm before the storm" dynamics.

## License

MIT

## Author

Ryan Cardwell ([@Benthic_Shadow](https://x.com/Benthic_Shadow))

## See Also

- [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm) - Core detection library
- [GitHub](https://github.com/aphoticshaman/regime-shift)
