import { test } from 'node:test';
import assert from 'node:assert';
import { SensorMonitor, assess, HealthLevel } from '../src/index.js';

test('SensorMonitor initializes', async () => {
  const d = new SensorMonitor();
  await d.init();
  assert.strictEqual(d.current().level, HealthLevel.NORMAL);
});

test('SensorMonitor processes data', async () => {
  const d = new SensorMonitor();
  await d.init();
  for (let i = 0; i < 50; i++) d.update(Math.random() * 100);
  assert.strictEqual(d.current().dataPoints, 50);
});

test('assess batch works', async () => {
  const values = Array.from({ length: 30 }, () => Math.random() * 100);
  const result = await assess(values);
  assert.ok('failing' in result);
  assert.ok('level' in result);
});
