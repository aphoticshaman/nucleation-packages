/**
 * Integration test for LogicalAgent with realistic data
 *
 * Run with: npx tsx packages/web/lib/signals/__tests__/LogicalAgent.integration.test.ts
 */

import {
  runInference,
  generateLogicalBriefing,
  type Inference,
  type InferenceResult,
  type LogicalBriefing,
} from '../LogicalAgent';
import type { ProcessedSignal, BaselineStats } from '../SignalProcessor';

// Simulate realistic nation risk data (like actual GDELT/nation state)
const testNationRisks: Record<string, number> = {
  UKR: 0.50,  // Ukraine - elevated due to conflict
  RUS: 0.45,  // Russia - moderately elevated
  ISR: 0.48,  // Israel - elevated due to regional tensions
  IRN: 0.42,  // Iran - elevated
  CHN: 0.35,  // China - moderate
  USA: 0.20,  // United States - baseline
  DEU: 0.18,  // Germany - baseline
  GBR: 0.15,  // UK - baseline
  JPN: 0.12,  // Japan - stable
  SAU: 0.38,  // Saudi Arabia - moderate
};

// Simulate processed signals
const testSignals: ProcessedSignal[] = [
  {
    id: 'test-1',
    text: 'Ukrainian forces advance in eastern regions amid ongoing conflict',
    source: 'gdelt',
    timestamp: new Date(),
    features: {
      sentimentScore: -0.4,
      urgencyScore: 0.7,
      primaryTopic: 'conflict',
      detectedEntities: ['UKR', 'RUS'],
      namedEntities: ['Ukraine', 'Russia'],
      isAnomalous: false,
      anomalyScore: 0,
      isBreaking: true,
    },
    processingTime: 50,
  },
  {
    id: 'test-2',
    text: 'Middle East tensions rise as diplomatic efforts stall',
    source: 'gdelt',
    timestamp: new Date(),
    features: {
      sentimentScore: -0.3,
      urgencyScore: 0.6,
      primaryTopic: 'diplomacy',
      detectedEntities: ['ISR', 'IRN'],
      namedEntities: ['Israel', 'Iran'],
      isAnomalous: false,
      anomalyScore: 0,
      isBreaking: false,
    },
    processingTime: 45,
  },
  {
    id: 'test-3',
    text: 'Energy markets react to supply concerns from major producers',
    source: 'gdelt',
    timestamp: new Date(),
    features: {
      sentimentScore: -0.2,
      urgencyScore: 0.4,
      primaryTopic: 'economic',
      detectedEntities: ['SAU', 'RUS'],
      namedEntities: ['Saudi Arabia', 'Russia'],
      isAnomalous: false,
      anomalyScore: 0,
      isBreaking: false,
    },
    processingTime: 40,
  },
];

const testBaseline: BaselineStats = {
  sentimentMean: 0,
  sentimentStd: 0.5,
  urgencyMean: 0.2,
  urgencyStd: 0.3,
  topicDistribution: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
  signalCount: 100,
  lastUpdated: new Date(),
};

// Test runner
function runTests(): void {
  console.log('='.repeat(60));
  console.log('LogicalAgent Integration Tests');
  console.log('='.repeat(60));
  console.log('');

  let passed = 0;
  let failed = 0;

  // Test 1: runInference generates inferences
  console.log('Test 1: runInference generates inferences with realistic data');
  try {
    const result: InferenceResult = runInference(testSignals, testNationRisks, testBaseline);

    console.log(`  Facts used: ${result.factsUsed}`);
    console.log(`  Rules evaluated: ${result.rulesEvaluated}`);
    console.log(`  Inferences generated: ${result.inferences.length}`);
    console.log(`  Processing time: ${result.processingTimeMs}ms`);

    if (result.inferences.length > 0) {
      console.log('  ✅ PASSED - Generated inferences');
      passed++;
    } else {
      console.log('  ❌ FAILED - No inferences generated');
      failed++;
    }

    // Log inference details
    console.log('\n  Inferences:');
    for (const inf of result.inferences.slice(0, 5)) {
      console.log(`    [${inf.severity}] ${inf.type}: ${inf.conclusion}`);
    }
    console.log('');
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Test 2: Alliance cascade rule fires
  console.log('Test 2: Alliance cascade rule fires for UKR (50% risk > 0.4 threshold)');
  try {
    const result = runInference(testSignals, testNationRisks, testBaseline);
    const cascadeWarnings = result.inferences.filter(i => i.type === 'cascade_warning');

    if (cascadeWarnings.length > 0) {
      console.log(`  ✅ PASSED - ${cascadeWarnings.length} cascade warnings generated`);
      passed++;
    } else {
      console.log('  ❌ FAILED - No cascade warnings');
      failed++;
    }
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Test 3: Conflict escalation rule fires
  console.log('Test 3: Conflict escalation rule fires for UKR-RUS pair');
  try {
    const result = runInference(testSignals, testNationRisks, testBaseline);
    const conflictWarnings = result.inferences.filter(i =>
      i.type === 'cascade_warning' && i.subject.includes('UKR')
    );

    if (conflictWarnings.length > 0) {
      console.log(`  ✅ PASSED - ${conflictWarnings.length} conflict warnings for UKR`);
      passed++;
    } else {
      console.log('  ❌ FAILED - No conflict warnings for UKR');
      failed++;
    }
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Test 4: generateLogicalBriefing produces valid output
  console.log('Test 4: generateLogicalBriefing produces valid output');
  try {
    const result = runInference(testSignals, testNationRisks, testBaseline);
    const briefing: LogicalBriefing = generateLogicalBriefing(result, testNationRisks);

    console.log(`  Summary length: ${briefing.summary.length}`);
    console.log(`  Key findings: ${briefing.keyFindings.length}`);
    console.log(`  Risk alerts: ${briefing.riskAlerts.length}`);
    console.log(`  Cascade warnings: ${briefing.cascadeWarnings.length}`);
    console.log(`  Action items: ${briefing.actionItems.length}`);

    if (briefing.summary.length > 0 && briefing.keyFindings.length > 0) {
      console.log('  ✅ PASSED - Valid briefing generated');
      passed++;
    } else {
      console.log('  ❌ FAILED - Empty briefing');
      failed++;
    }

    // Log briefing details
    console.log('\n  Summary:');
    console.log(`    ${briefing.summary.substring(0, 200)}...`);
    console.log('\n  Key Findings:');
    for (const finding of briefing.keyFindings.slice(0, 3)) {
      console.log(`    - ${finding}`);
    }
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Test 5: Entity concentration rule
  console.log('\nTest 5: Entity concentration rule fires for frequently mentioned entities');
  try {
    const result = runInference(testSignals, testNationRisks, testBaseline);
    const correlations = result.inferences.filter(i => i.type === 'correlation');

    if (correlations.length > 0) {
      console.log(`  ✅ PASSED - ${correlations.length} correlations detected`);
      passed++;
    } else {
      console.log('  ❌ FAILED - No entity correlations');
      failed++;
    }
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Test 6: Urgency spike rule
  console.log('Test 6: Urgency spike rule fires for breaking signals');
  try {
    const result = runInference(testSignals, testNationRisks, testBaseline);
    const actionNeeded = result.inferences.filter(i => i.type === 'action_needed');

    if (actionNeeded.length > 0) {
      console.log(`  ✅ PASSED - ${actionNeeded.length} action items generated`);
      passed++;
    } else {
      console.log('  ❌ FAILED - No urgency spikes detected');
      failed++;
    }
  } catch (e) {
    console.log(`  ❌ FAILED - Error: ${e}`);
    failed++;
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('='.repeat(60));

  if (failed > 0) {
    process.exit(1);
  }
}

// Run tests
runTests();
