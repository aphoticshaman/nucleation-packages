# match-pulse

Detect player tilt and match quality issues.

Esports analytics for player behavior monitoring. The core insight: **players show tilt patterns before tilting** — performance variance drops as frustration builds, then collapses.

## Quick Start

```javascript
import { TiltDetector } from 'match-pulse';

const detector = new TiltDetector();
await detector.init();

gameEvents.on('action', (action) => {
  const state = detector.update(action.performanceScore);
  if (state.frustrated) {
    suggestBreak(playerId);
  }
});
```

## Player States

| State | Meaning |
|-------|---------|
| `focused` | Optimal performance variance |
| `frustrated` | Performance tightening |
| `tilting` | Tilt imminent |
| `tilted` | Full tilt mode |

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm). MIT License.
