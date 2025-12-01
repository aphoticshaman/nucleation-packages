# crowd-phase

Detect community conflict before it erupts.

Social dynamics monitoring for trust & safety teams. The core insight: **communities show warning signs before conflict** — sentiment variance drops as factions polarize.

## Quick Start

```javascript
import { ConflictDetector } from 'crowd-phase';

const detector = new ConflictDetector();
await detector.init();

commentStream.on('comment', (comment) => {
  const state = detector.update(comment.sentimentScore);
  if (state.elevated) {
    escalateToModerators(threadId);
  }
});
```

## Community States

| State | Meaning |
|-------|---------|
| `healthy` | Normal discourse variance |
| `tense` | Polarization emerging |
| `volatile` | Conflict likely |
| `conflict` | Active conflict mode |

Built on [nucleation-wasm](https://www.npmjs.com/package/nucleation-wasm). MIT License.
