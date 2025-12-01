import { test } from 'node:test';
import assert from 'node:assert';
import { SupplyMonitor, assess, RiskLevel } from '../src/index.js';

test('SupplyMonitor initializes', async () => {
  const d = new SupplyMonitor();
  await d.init();
  assert.strictEqual(d.current().level, RiskLevel.STABLE);
});

test('SupplyMonitor processes data', async () => {
  const d = new SupplyMonitor();
  await d.init();
  for (let i = 0; i < 50; i++) d.update(Math.random() * 100);
  assert.strictEqual(d.current().dataPoints, 50);
});

test('assess batch works', async () => {
  const values = Array.from({ length: 30 }, () => Math.random() * 100);
  const result = await assess(values);
  assert.ok('atRisk' in result);
  assert.ok('level' in result);
});
