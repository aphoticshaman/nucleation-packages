import { test } from 'node:test';
import assert from 'node:assert';
import {
  TeamHealthMonitor,
  IntegrationMonitor,
  assessTeamHealth,
  HealthLevel,
} from '../src/index.js';

test('TeamHealthMonitor initializes', async () => {
  const monitor = new TeamHealthMonitor();
  await monitor.init();
  const state = monitor.current();
  assert.strictEqual(state.healthLevel, HealthLevel.THRIVING);
});

test('TeamHealthMonitor processes data', async () => {
  const monitor = new TeamHealthMonitor();
  await monitor.init();

  for (let i = 0; i < 24; i++) {
    monitor.update(70 + Math.random() * 20);
  }

  assert.strictEqual(monitor.current().dataPoints, 24);
});

test('IntegrationMonitor tracks entities', async () => {
  const monitor = new IntegrationMonitor(5);
  await monitor.init();

  monitor.registerEntity('org-a', { name: 'OrgA' });
  monitor.registerEntity('org-b', { name: 'OrgB' });

  assert.ok(monitor.getEntities().includes('org-a'));
  assert.strictEqual(monitor.getEntityMetadata('org-a').name, 'OrgA');
});

test('assessTeamHealth works', async () => {
  const scores = Array.from({ length: 20 }, () => 70 + Math.random() * 20);
  const result = await assessTeamHealth(scores);

  assert.ok('stressed' in result);
  assert.ok('healthLevel' in result);
});
