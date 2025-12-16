#!/usr/bin/env npx ts-node
/**
 * LatticeForge Smoke Test Script
 *
 * Run with: npx ts-node scripts/smoke-test.ts
 * Or: npm run smoke-test (after adding to package.json)
 *
 * Tests critical paths for demo readiness:
 * 1. Public pages load
 * 2. API endpoints respond
 * 3. Database connectivity
 * 4. Cache functioning
 */

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

interface TestResult {
  name: string;
  passed: boolean;
  latency: number;
  error?: string;
  details?: unknown;
}

const results: TestResult[] = [];

async function test(name: string, fn: () => Promise<void>): Promise<void> {
  const start = Date.now();
  try {
    await fn();
    results.push({ name, passed: true, latency: Date.now() - start });
    console.log(`✅ ${name} (${Date.now() - start}ms)`);
  } catch (error) {
    const latency = Date.now() - start;
    const errorMsg = error instanceof Error ? error.message : String(error);
    results.push({ name, passed: false, latency, error: errorMsg });
    console.log(`❌ ${name} (${latency}ms): ${errorMsg}`);
  }
}

async function fetchJson(url: string, options?: RequestInit): Promise<unknown> {
  const response = await fetch(url, options);
  if (!response.ok && response.status !== 503) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchPage(path: string): Promise<void> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const html = await response.text();
  if (!html.includes('</html>')) {
    throw new Error('Invalid HTML response');
  }
}

async function runTests(): Promise<void> {
  console.log(`\n🧪 LatticeForge Smoke Tests\n${'='.repeat(50)}\n`);
  console.log(`Base URL: ${BASE_URL}\n`);

  // Public Pages
  console.log('📄 Public Pages\n');
  await test('Landing page loads', () => fetchPage('/'));
  await test('Pricing page loads', () => fetchPage('/pricing'));
  await test('Login page loads', () => fetchPage('/login'));
  await test('Signup page loads', () => fetchPage('/signup'));

  // API Endpoints
  console.log('\n🔌 API Endpoints\n');

  await test('Intel briefing API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/intel-briefing`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset: 'global' }),
    });
    if (!data || typeof data !== 'object') throw new Error('Invalid response');
  });

  await test('US Brief API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/us-brief`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    if (!data || typeof data !== 'object') throw new Error('Invalid response');
  });

  await test('Signals API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/query/signals?source=freshness`, {
      headers: { 'x-user-tier': 'enterprise_tier' },
    }) as { freshness?: unknown };
    if (!data.freshness) throw new Error('No freshness data');
  });

  await test('Alerts API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/query/alerts`, {
      headers: { 'x-user-tier': 'enterprise_tier' },
    }) as { alerts?: unknown[] };
    if (!Array.isArray(data.alerts)) throw new Error('Invalid alerts response');
  });

  await test('Cascades API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/query/cascades`, {
      headers: { 'x-user-tier': 'enterprise_tier' },
    }) as { topCascades?: unknown };
    if (!data) throw new Error('Invalid cascades response');
  });

  await test('Doctrine API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/doctrine`, {
      headers: { 'x-user-tier': 'enterprise_tier' },
    }) as { doctrines?: unknown[] };
    if (!Array.isArray(data.doctrines)) throw new Error('Invalid doctrine response');
  });

  await test('Export API responds', async () => {
    const data = await fetchJson(`${BASE_URL}/api/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-user-tier': 'enterprise_tier',
      },
      body: JSON.stringify({ data_type: 'signals', format: 'json' }),
    }) as { data?: unknown[] };
    if (!Array.isArray(data.data)) throw new Error('Invalid export response');
  });

  // Tier Enforcement
  console.log('\n🔒 Tier Enforcement\n');

  await test('Free tier blocked from signals API', async () => {
    const response = await fetch(`${BASE_URL}/api/query/signals`, {
      headers: { 'x-user-tier': 'free' },
    });
    if (response.status !== 403) throw new Error(`Expected 403, got ${response.status}`);
  });

  await test('Free tier blocked from doctrine API', async () => {
    const response = await fetch(`${BASE_URL}/api/doctrine`, {
      headers: { 'x-user-tier': 'free' },
    });
    if (response.status !== 403) throw new Error(`Expected 403, got ${response.status}`);
  });

  // App Pages (require auth but should load shell)
  console.log('\n📱 App Pages\n');
  await test('Consumer app page loads', () => fetchPage('/app'));
  await test('Dashboard page loads', () => fetchPage('/dashboard'));
  await test('Admin health page loads', () => fetchPage('/admin/health'));

  // Summary
  console.log(`\n${'='.repeat(50)}`);
  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  const avgLatency = Math.round(results.reduce((sum, r) => sum + r.latency, 0) / results.length);

  console.log(`\n📊 Results: ${passed}/${results.length} passed (${failed} failed)`);
  console.log(`⏱️  Average latency: ${avgLatency}ms`);

  if (failed > 0) {
    console.log('\n❌ Failed tests:');
    results.filter(r => !r.passed).forEach(r => {
      console.log(`   - ${r.name}: ${r.error}`);
    });
    process.exit(1);
  } else {
    console.log('\n✅ All smoke tests passed!');
    process.exit(0);
  }
}

runTests().catch(console.error);
