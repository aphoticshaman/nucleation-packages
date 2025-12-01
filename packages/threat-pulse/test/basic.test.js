import { test } from 'node:test';
import assert from 'node:assert';
import { ThreatDetector, ThreatCorrelator, assessThreat, ThreatLevel } from '../src/index.js';

test('ThreatDetector initializes', async () => {
  const detector = new ThreatDetector();
  await detector.init();
  const state = detector.current();
  assert.strictEqual(state.threatLevel, ThreatLevel.GREEN);
  assert.strictEqual(state.eventCount, 0);
});

test('ThreatDetector processes events', async () => {
  const detector = new ThreatDetector();
  await detector.init();

  // Feed anomaly scores
  for (let i = 0; i < 100; i++) {
    detector.update(Math.random());
  }

  const state = detector.current();
  assert.strictEqual(state.eventCount, 100);
  assert.ok(state.variance >= 0);
});

test('assessThreat batch function works', async () => {
  const scores = Array.from({ length: 50 }, () => Math.random());
  const result = await assessThreat(scores);

  assert.ok('escalating' in result);
  assert.ok('threatLevel' in result);
  assert.ok('confidence' in result);
});

test('ThreatCorrelator tracks sources', async () => {
  const correlator = new ThreatCorrelator(5);
  await correlator.init();

  correlator.registerSource('source1');
  correlator.registerSource('source2');

  const sources = correlator.getSources();
  assert.ok(sources.includes('source1'));
  assert.ok(sources.includes('source2'));
});
