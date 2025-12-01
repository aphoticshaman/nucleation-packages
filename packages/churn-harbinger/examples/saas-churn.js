/**
 * Example: SaaS Churn Prediction
 *
 * Simulates user engagement data with typical churn patterns.
 * Demonstrates early detection of disengagement before cancellation.
 *
 * Run with: node examples/saas-churn.js
 */

import { ChurnDetector, CohortMonitor, assessChurnRisk } from '../src/index.js';

/**
 * Simulated user engagement data showing churn pattern:
 * 1. Healthy engagement (high, variable activity)
 * 2. Cooling period (declining, more regular)
 * 3. Disengagement (low, flat activity)
 * 4. Pre-churn (minimal, near-zero variance)
 */
function generateUserEngagement(type = 'churning') {
  const days = [];

  if (type === 'healthy') {
    // Healthy user: consistent high engagement with natural variance
    for (let i = 0; i < 60; i++) {
      days.push({
        day: i + 1,
        sessions: 3 + Math.floor(Math.random() * 5),
        duration: 15 + Math.random() * 30,
        actions: 20 + Math.floor(Math.random() * 40),
      });
    }
  } else if (type === 'churning') {
    // Phase 1: Healthy engagement (days 1-25)
    for (let i = 0; i < 25; i++) {
      days.push({
        day: i + 1,
        sessions: 3 + Math.floor(Math.random() * 4),
        duration: 15 + Math.random() * 25,
        actions: 25 + Math.floor(Math.random() * 35),
      });
    }

    // Phase 2: Cooling - declining but still variable (days 26-40)
    for (let i = 0; i < 15; i++) {
      const decay = 1 - i / 30;
      days.push({
        day: 26 + i,
        sessions: Math.max(1, Math.floor((2 + Math.random() * 3) * decay)),
        duration: Math.max(5, (10 + Math.random() * 15) * decay),
        actions: Math.max(5, Math.floor((15 + Math.random() * 20) * decay)),
      });
    }

    // Phase 3: Disengaged - low, flat activity (days 41-50)
    for (let i = 0; i < 10; i++) {
      days.push({
        day: 41 + i,
        sessions: 1,
        duration: 3 + Math.random() * 2,
        actions: 3 + Math.floor(Math.random() * 3),
      });
    }

    // Phase 4: Pre-churn - minimal, near-zero variance (days 51-60)
    for (let i = 0; i < 10; i++) {
      days.push({
        day: 51 + i,
        sessions: 1,
        duration: 2 + Math.random() * 0.5,
        actions: 2,
      });
    }
  }

  return days;
}

/**
 * Calculate engagement score from multiple metrics
 */
function engagementScore(day) {
  // Weighted combination of metrics
  return day.sessions * 10 + day.duration * 0.5 + day.actions * 0.3;
}

async function main() {
  console.log('=== SaaS Churn Prediction Demo ===\n');

  // Generate data for two users
  const churningUser = generateUserEngagement('churning');
  const healthyUser = generateUserEngagement('healthy');

  console.log(`Analyzing ${churningUser.length} days of user engagement\n`);

  // Method 1: Individual user monitoring
  console.log('--- User A (Churning Pattern) ---\n');

  const detectorA = new ChurnDetector({
    sensitivity: 'sensitive',
    windowSize: 14, // 2-week baseline
  });
  await detectorA.init();

  let lastRisk = null;
  let firstWarningDay = null;

  for (const day of churningUser) {
    const score = engagementScore(day);
    const state = detectorA.update(score);

    if (state.riskLevel !== lastRisk) {
      console.log(
        `Day ${day.day}: Risk level ${lastRisk || 'init'} → ${state.riskLevel.toUpperCase()}`
      );
      console.log(
        `        Engagement: ${score.toFixed(1)}, Variance: ${state.variance.toFixed(2)}`
      );

      if (state.atRisk && !firstWarningDay) {
        firstWarningDay = day.day;
        console.log(`        ⚠️  FIRST AT-RISK SIGNAL`);
      }
      console.log();

      lastRisk = state.riskLevel;
    }
  }

  if (firstWarningDay) {
    const daysBeforeChurn = 60 - firstWarningDay;
    console.log(`Early warning: ${daysBeforeChurn} days before end of observation period\n`);
  }

  // Method 2: Healthy user comparison
  console.log('--- User B (Healthy Pattern) ---\n');

  const detectorB = new ChurnDetector({ sensitivity: 'sensitive', windowSize: 14 });
  await detectorB.init();

  const scoresB = healthyUser.map(engagementScore);
  const stateB = detectorB.updateBatch(scoresB);

  console.log(`Final risk level: ${stateB.riskLevel.toUpperCase()}`);
  console.log(`At risk: ${stateB.atRisk}`);
  console.log(`Confidence: ${(stateB.confidence * 100).toFixed(1)}%\n`);

  // Method 3: Quick batch assessment
  console.log('--- Quick Risk Assessment ---\n');

  const scoresA = churningUser.map(engagementScore);

  const resultA = await assessChurnRisk(scoresA, { sensitivity: 'balanced' });
  const resultB = await assessChurnRisk(scoresB, { sensitivity: 'balanced' });

  console.log('User A (churning):', resultA);
  console.log('User B (healthy):', resultB);

  // Method 4: Cohort comparison
  console.log('\n--- Cohort Analysis ---\n');

  const cohort = new CohortMonitor(5);
  await cohort.init();

  // Add users with metadata
  cohort.addUser('user-a', { plan: 'pro', signupDate: '2024-01-15' });
  cohort.addUser('user-b', { plan: 'pro', signupDate: '2024-01-20' });
  cohort.addUser('user-c', { plan: 'enterprise', signupDate: '2024-02-01' });

  // Simulate behavior distributions
  // [feature_use, collaboration, exports, integrations, support]
  const healthyBehavior = new Float64Array([0.3, 0.25, 0.2, 0.15, 0.1]);
  const churningBehavior = new Float64Array([0.7, 0.05, 0.05, 0.1, 0.1]); // Over-reliant on one feature

  cohort.updateUser('user-a', churningBehavior);
  cohort.updateUser('user-b', healthyBehavior);
  cohort.updateUser('user-c', healthyBehavior);

  // Check divergence
  const abDivergence = cohort.getDivergence('user-a', 'user-b');
  const bcDivergence = cohort.getDivergence('user-b', 'user-c');

  console.log('Behavioral divergence (higher = more different from cohort):');
  console.log(`  User A vs B: ${abDivergence?.toFixed(4) || 'N/A'}`);
  console.log(`  User B vs C: ${bcDivergence?.toFixed(4) || 'N/A'}`);

  if (abDivergence > bcDivergence * 2) {
    console.log('\n⚠️  User A showing divergent behavior from healthy cohort');
    console.log('   Recommendation: Trigger CSM outreach');
  }

  console.log('\n--- Summary ---');
  console.log('The detector identified:');
  console.log('1. Healthy phase with natural engagement variance');
  console.log('2. Cooling phase as engagement declined');
  console.log('3. Pre-churn "settling" pattern (low variance)');
  console.log('4. At-risk signal before complete disengagement');
}

main().catch(console.error);
