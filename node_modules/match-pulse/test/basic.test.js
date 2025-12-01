import { test } from 'node:test';
import assert from 'node:assert';
import { MatchMonitor, assess, TiltLevel } from '../src/index.js';

test('MatchMonitor initializes', async () => {
  const d = new MatchMonitor();
  await d.init();
  assert.strictEqual(d.current().level, TiltLevel.FOCUSED);
});

test('MatchMonitor processes data', async () => {
  const d = new MatchMonitor();
  await d.init();
  for (let i = 0; i < 50; i++) d.update(Math.random() * 100);
  assert.strictEqual(d.current().dataPoints, 50);
});

test('assess batch works', async () => {
  const values = Array.from({ length: 30 }, () => Math.random() * 100);
  const result = await assess(values);
  assert.ok('tilted' in result);
  assert.ok('level' in result);
});
