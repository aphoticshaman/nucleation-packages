import { test } from 'node:test';
import assert from 'node:assert';
import { RegimeDetector, detectRegimeShift, Regime } from '../src/index.js';

test('RegimeDetector initializes', async () => {
  const detector = new RegimeDetector();
  await detector.init();
  const state = detector.current();
  assert.strictEqual(state.regime, Regime.STABLE);
  assert.strictEqual(state.observations, 0);
});

test('RegimeDetector processes data', async () => {
  const detector = new RegimeDetector();
  await detector.init();

  // Feed some data
  for (let i = 0; i < 100; i++) {
    detector.update(Math.random() * 100);
  }

  const state = detector.current();
  assert.strictEqual(state.observations, 100);
  assert.ok(state.variance >= 0);
});

test('detectRegimeShift batch function works', async () => {
  const prices = Array.from({ length: 50 }, () => Math.random() * 100);
  const result = await detectRegimeShift(prices);

  assert.ok('shifting' in result);
  assert.ok('regime' in result);
  assert.ok('confidence' in result);
});
