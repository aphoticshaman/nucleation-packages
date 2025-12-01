/**
 * Example: Team Health & M&A Integration Monitoring
 *
 * Run with: node examples/team-health.js
 */

import { TeamHealthMonitor, IntegrationMonitor, assessTeamHealth } from '../src/index.js';

// Simulated weekly team health scores
function generateTeamData(pattern = 'declining') {
  const weeks = [];

  if (pattern === 'healthy') {
    for (let i = 0; i < 24; i++) {
      weeks.push({ week: i + 1, score: 75 + Math.random() * 20 });
    }
  } else if (pattern === 'declining') {
    // Healthy period
    for (let i = 0; i < 12; i++) {
      weeks.push({ week: i + 1, score: 75 + Math.random() * 15 });
    }
    // Tension building - variance drops
    for (let i = 0; i < 6; i++) {
      weeks.push({ week: 13 + i, score: 60 + Math.random() * 5 });
    }
    // Dysfunction
    for (let i = 0; i < 6; i++) {
      weeks.push({ week: 19 + i, score: 40 + Math.random() * 20 });
    }
  }

  return weeks;
}

async function main() {
  console.log('=== Organizational Health Monitoring Demo ===\n');

  // Team health monitoring
  console.log('--- Team Health Over Time ---\n');

  const teamData = generateTeamData('declining');
  const monitor = new TeamHealthMonitor({ sensitivity: 'sensitive', windowSize: 8 });
  await monitor.init();

  let lastLevel = null;

  for (const { week, score } of teamData) {
    const state = monitor.update(score);

    if (state.healthLevel !== lastLevel) {
      console.log(`Week ${week}: ${lastLevel || 'start'} → ${state.healthLevel.toUpperCase()}`);
      console.log(`        Score: ${score.toFixed(1)}, Variance: ${state.variance.toFixed(2)}`);
      lastLevel = state.healthLevel;
    }
  }

  // M&A integration risk
  console.log('\n--- M&A Integration Risk ---\n');

  const integration = new IntegrationMonitor(6);
  await integration.init();

  // Culture dimensions: [innovation, hierarchy, collaboration, risk-tolerance, pace, formality]
  integration.registerEntity('acquirer', { name: 'TechCorp', employees: 5000 });
  integration.registerEntity('target', { name: 'StartupCo', employees: 200 });

  // Very different cultures
  const acquirerCulture = new Float64Array([0.15, 0.3, 0.15, 0.1, 0.15, 0.15]); // Hierarchical, formal
  const targetCulture = new Float64Array([0.35, 0.05, 0.25, 0.2, 0.1, 0.05]); // Innovative, flat

  integration.updateEntity('acquirer', acquirerCulture);
  integration.updateEntity('target', targetCulture);

  const clashRisk = integration.getClashRisk('acquirer', 'target');

  console.log(`Culture clash risk: ${clashRisk?.toFixed(4) || 'N/A'}`);

  if (clashRisk > 0.1) {
    console.log('⚠️  HIGH INTEGRATION RISK');
    console.log('   Key differences: Innovation focus, hierarchy expectations');
    console.log('   Recommendation: Extended integration timeline, culture bridge programs');
  }

  // Quick assessment
  console.log('\n--- Quick Assessment ---\n');
  const healthyTeam = generateTeamData('healthy').map((w) => w.score);
  const decliningTeam = generateTeamData('declining').map((w) => w.score);

  console.log('Healthy team:', await assessTeamHealth(healthyTeam));
  console.log('Declining team:', await assessTeamHealth(decliningTeam));
}

main().catch(console.error);
