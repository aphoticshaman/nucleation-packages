import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Only test TypeScript files in packages with TypeScript
    include: ['packages/core/**/*.test.ts', 'packages/social-pulse/**/*.test.ts'],
    // Exclude legacy JS packages - they use Node.js test runner
    exclude: [
      'node_modules/**',
      'dist/**',
      'packages/regime-shift/**',
      'packages/threat-pulse/**',
      'packages/churn-harbinger/**',
      'packages/org-canary/**',
      'packages/supply-sentinel/**',
      'packages/sensor-shift/**',
      'packages/crowd-phase/**',
      'packages/patient-drift/**',
      'packages/match-pulse/**',
      'packages/market-canary/**',
      'packages/nucleation/**',
      'harness/**',
    ],
    // Stable settings for enterprise
    testTimeout: 30000,
    hookTimeout: 30000,
    globals: true,
    // Allow passing when no TS tests exist yet (JS tests use Node.js runner)
    passWithNoTests: true,
  },
});
