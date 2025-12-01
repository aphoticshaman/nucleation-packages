/**
 * Example: Supply Chain Disruption Detection
 *
 * Demonstrates early warning for supplier disruptions using
 * lead time variance analysis.
 */

import { DisruptionDetector, SupplierNetwork, assessDisruptionRisk } from '../src/index.js';

// Simulated lead times showing disruption pattern
function generateLeadTimes(type = 'disruption') {
  const data = [];

  if (type === 'stable') {
    for (let i = 0; i < 40; i++) {
      data.push({ day: i + 1, leadTime: 5 + Math.random() * 3 }); // 5-8 days
    }
  } else {
    // Normal operations
    for (let i = 0; i < 20; i++) {
      data.push({ day: i + 1, leadTime: 5 + Math.random() * 3 });
    }
    // Stress - variance dropping as supplier maxes out
    for (let i = 0; i < 10; i++) {
      data.push({ day: 21 + i, leadTime: 7.5 + Math.random() * 0.5 });
    }
    // Disruption - lead times spike
    for (let i = 0; i < 10; i++) {
      data.push({ day: 31 + i, leadTime: 12 + Math.random() * 8 });
    }
  }

  return data;
}

async function main() {
  console.log('=== Supply Chain Disruption Detection ===\n');

  const disruptionData = generateLeadTimes('disruption');

  const detector = new DisruptionDetector({ sensitivity: 'sensitive', windowSize: 10 });
  await detector.init();

  let lastLevel = null;

  for (const { day, leadTime } of disruptionData) {
    const state = detector.update(leadTime);

    if (state.riskLevel !== lastLevel) {
      console.log(`Day ${day}: ${lastLevel || 'init'} → ${state.riskLevel.toUpperCase()}`);
      console.log(
        `        Lead time: ${leadTime.toFixed(1)} days, Variance: ${state.variance.toFixed(3)}`
      );

      if (state.elevated) console.log(`        ⚠️  ELEVATED RISK`);
      console.log();

      lastLevel = state.riskLevel;
    }
  }

  // Multi-supplier network
  console.log('--- Supplier Network Risk ---\n');

  const network = new SupplierNetwork(5);
  await network.init();

  // Risk dimensions: [capacity, quality, geo, financial, concentration]
  const reliableSupplier = new Float64Array([0.1, 0.1, 0.2, 0.1, 0.5]);
  const riskySupplier = new Float64Array([0.4, 0.2, 0.15, 0.15, 0.1]);

  network.registerSupplier('supplier-A', { region: 'domestic' });
  network.registerSupplier('supplier-B', { region: 'overseas' });

  network.updateSupplier('supplier-A', reliableSupplier);
  network.updateSupplier('supplier-B', riskySupplier);

  const cascadeRisk = network.getCascadeRisk('supplier-A', 'supplier-B');
  console.log(`Cascade risk between suppliers: ${cascadeRisk?.toFixed(4) || 'N/A'}`);

  // Quick assessment
  const leadTimes = disruptionData.map((d) => d.leadTime);
  const result = await assessDisruptionRisk(leadTimes);
  console.log('\nQuick assessment:', result);
}

main().catch(console.error);
