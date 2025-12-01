import { test } from 'node:test';
import assert from 'node:assert';
import { TransitionDetector, detectTransition, PhaseLevel } from '../src/index.js';

test('TransitionDetector initializes', async () => {
  const d = new TransitionDetector();
  await d.init();
  assert.strictEqual(d.current().phase, PhaseLevel.STABLE);
});

test('TransitionDetector processes data', async () => {
  const d = new TransitionDetector();
  await d.init();
  for (let i = 0; i < 50; i++) d.update(Math.random() * 100);
  assert.strictEqual(d.current().dataPoints, 50);
});

test('detectTransition batch works', async () => {
  const values = Array.from({ length: 30 }, () => Math.random() * 100);
  const result = await detectTransition(values);
  assert.ok('transitioning' in result);
  assert.ok('phase' in result);
});
