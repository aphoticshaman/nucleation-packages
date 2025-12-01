import { test } from 'node:test';
import assert from 'node:assert';
import { CrowdMonitor, assess, TensionLevel } from '../src/index.js';

test('CrowdMonitor initializes', async () => {
  const d = new CrowdMonitor();
  await d.init();
  assert.strictEqual(d.current().level, TensionLevel.CALM);
});

test('CrowdMonitor processes data', async () => {
  const d = new CrowdMonitor();
  await d.init();
  for (let i = 0; i < 50; i++) d.update(Math.random() * 100);
  assert.strictEqual(d.current().dataPoints, 50);
});

test('assess batch works', async () => {
  const values = Array.from({ length: 30 }, () => Math.random() * 100);
  const result = await assess(values);
  assert.ok('volatile' in result);
  assert.ok('level' in result);
});
