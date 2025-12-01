import { test } from 'node:test';
import assert from 'node:assert';
import { ChurnDetector, CohortMonitor, assessChurnRisk, RiskLevel } from '../src/index.js';

test('ChurnDetector initializes', async () => {
  const detector = new ChurnDetector();
  await detector.init();
  const state = detector.current();
  assert.strictEqual(state.riskLevel, RiskLevel.HEALTHY);
  assert.strictEqual(state.dataPoints, 0);
});

test('ChurnDetector processes engagement data', async () => {
  const detector = new ChurnDetector();
  await detector.init();

  // Feed engagement scores
  for (let i = 0; i < 60; i++) {
    detector.update(50 + Math.random() * 50);
  }

  const state = detector.current();
  assert.strictEqual(state.dataPoints, 60);
  assert.ok(state.variance >= 0);
});

test('assessChurnRisk batch function works', async () => {
  const scores = Array.from({ length: 30 }, () => 50 + Math.random() * 50);
  const result = await assessChurnRisk(scores);

  assert.ok('atRisk' in result);
  assert.ok('riskLevel' in result);
  assert.ok('confidence' in result);
});

test('CohortMonitor tracks users', async () => {
  const cohort = new CohortMonitor(5);
  await cohort.init();

  cohort.addUser('user-1', { plan: 'pro' });
  cohort.addUser('user-2', { plan: 'enterprise' });

  const users = cohort.getUsers();
  assert.ok(users.includes('user-1'));
  assert.ok(users.includes('user-2'));

  const metadata = cohort.getUserMetadata('user-1');
  assert.strictEqual(metadata.plan, 'pro');
});

test('ChurnDetector serialization works', async () => {
  const detector = new ChurnDetector();
  await detector.init();

  for (let i = 0; i < 30; i++) {
    detector.update(Math.random() * 100);
  }

  const json = detector.serialize();
  assert.ok(json.length > 0);

  const restored = await ChurnDetector.deserialize(json);
  const state = restored.current();
  assert.strictEqual(state.dataPoints, 30);
});
