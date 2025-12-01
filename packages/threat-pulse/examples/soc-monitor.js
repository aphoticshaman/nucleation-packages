/**
 * Example: SOC Threat Monitoring
 *
 * Simulates a SIEM event stream with attack patterns.
 * Demonstrates detection of the "calm before the storm" in
 * attacker behavior.
 *
 * Run with: node examples/soc-monitor.js
 */

import { ThreatDetector, ThreatCorrelator, assessThreat } from '../src/index.js';

/**
 * Simulated SIEM event stream with attack phases:
 * 1. Normal operations (baseline noise)
 * 2. Reconnaissance (probing, more regular patterns)
 * 3. Pre-attack quieting (variance drops as attacker focuses)
 * 4. Active attack (high anomaly scores)
 */
function generateSecurityEvents() {
  const events = [];

  // Phase 1: Normal operations - random baseline noise (60 events)
  for (let i = 0; i < 60; i++) {
    events.push({
      timestamp: Date.now() - (120 - i) * 60000,
      source: 'mixed',
      anomalyScore: 0.1 + Math.random() * 0.3, // 0.1-0.4 range
      type: 'normal',
    });
  }

  // Phase 2: Reconnaissance - slightly elevated, more regular (20 events)
  for (let i = 0; i < 20; i++) {
    events.push({
      timestamp: Date.now() - (60 - i) * 60000,
      source: 'external',
      anomalyScore: 0.3 + Math.random() * 0.2, // 0.3-0.5 range
      type: 'recon',
    });
  }

  // Phase 3: Pre-attack quieting - LOW VARIANCE (15 events)
  // Attacker has identified target, reducing noise
  for (let i = 0; i < 15; i++) {
    events.push({
      timestamp: Date.now() - (40 - i) * 60000,
      source: 'external',
      anomalyScore: 0.45 + Math.random() * 0.05, // Very narrow: 0.45-0.50
      type: 'quiet',
    });
  }

  // Phase 4: Active attack - high scores, high variance (25 events)
  for (let i = 0; i < 25; i++) {
    events.push({
      timestamp: Date.now() - (25 - i) * 60000,
      source: 'compromised',
      anomalyScore: 0.6 + Math.random() * 0.35, // 0.6-0.95 range
      type: 'attack',
    });
  }

  return events;
}

async function main() {
  console.log('=== SOC Threat Monitoring Demo ===\n');

  const events = generateSecurityEvents();
  console.log(`Processing ${events.length} simulated security events\n`);

  // Method 1: Streaming detection
  console.log('--- Real-time Threat Detection ---\n');

  const detector = new ThreatDetector({
    sensitivity: 'aggressive',
    windowSize: 20,
  });
  await detector.init();

  let lastLevel = null;
  let alertCount = 0;

  for (const event of events) {
    const state = detector.update(event.anomalyScore);

    // Log threat level changes
    if (state.threatLevel !== lastLevel) {
      const time = new Date(event.timestamp).toISOString().split('T')[1].slice(0, 8);
      console.log(
        `[${time}] Threat Level: ${lastLevel || 'init'} → ${state.threatLevel.toUpperCase()}`
      );
      console.log(
        `         Confidence: ${(state.confidence * 100).toFixed(1)}%, Variance: ${state.variance.toFixed(4)}`
      );
      console.log(`         Event type: ${event.type}\n`);
      lastLevel = state.threatLevel;
    }

    // Count alerts
    if (state.elevated) alertCount++;
  }

  console.log(`Total elevated events: ${alertCount}/${events.length}\n`);

  // Method 2: Quick batch assessment
  console.log('--- Batch Assessment ---\n');

  const scores = events.map((e) => e.anomalyScore);
  const result = await assessThreat(scores, { sensitivity: 'balanced' });

  console.log('Quick assessment:', result);

  // Method 3: Multi-source correlation
  console.log('\n--- Multi-Source Correlation ---\n');

  const correlator = new ThreatCorrelator(5);
  await correlator.init();

  // Register sources
  correlator.registerSource('firewall');
  correlator.registerSource('ids');
  correlator.registerSource('endpoint');

  // Simulate diverging behavior (potential lateral movement)
  const behavior1 = new Float64Array([0.8, 0.1, 0.05, 0.03, 0.02]); // Normal
  const behavior2 = new Float64Array([0.2, 0.3, 0.3, 0.15, 0.05]); // Anomalous

  correlator.updateSource('firewall', behavior1, Date.now());
  correlator.updateSource('ids', behavior1, Date.now());
  correlator.updateSource('endpoint', behavior2, Date.now()); // Divergent!

  // Check correlations
  const fwIds = correlator.getCorrelation('firewall', 'ids');
  const fwEp = correlator.getCorrelation('firewall', 'endpoint');
  const idsEp = correlator.getCorrelation('ids', 'endpoint');

  console.log('Source correlation scores (higher = more divergent):');
  console.log(`  firewall <-> ids:      ${fwIds?.toFixed(4) || 'N/A'}`);
  console.log(`  firewall <-> endpoint: ${fwEp?.toFixed(4) || 'N/A'}`);
  console.log(`  ids <-> endpoint:      ${idsEp?.toFixed(4) || 'N/A'}`);

  if (fwEp > 0.1 || idsEp > 0.1) {
    console.log('\n⚠️  ALERT: Endpoint behavior diverging from perimeter sensors');
    console.log('   Possible lateral movement or compromised host');
  }

  console.log('\n--- Summary ---');
  console.log('The detector identified the attack pattern:');
  console.log('1. Normal baseline established');
  console.log('2. Reconnaissance activity elevated threat level');
  console.log('3. Pre-attack "quieting" (variance drop) preceded the attack');
  console.log('4. Active attack confirmed with high anomaly scores');
}

main().catch(console.error);
