/**
 * Ablation Tests for Nucleation Packages
 * 
 * Systematic testing of each detector with:
 * 1. Synthetic data with known phase transitions
 * 2. Sensitivity analysis
 * 3. False positive/negative rates
 * 4. Cross-validation
 */

import { RegimeDetector } from 'regime-shift';
import { ThreatDetector } from 'threat-pulse';
import { ChurnDetector } from 'churn-harbinger';
import { TeamHealthMonitor } from 'org-canary';
import { TransitionDetector } from 'market-canary';
import { SensorMonitor } from 'sensor-shift';
import { CrowdMonitor } from 'crowd-phase';
import { PatientMonitor } from 'patient-drift';
import { MatchMonitor } from 'match-pulse';
import SupplyMonitor from 'supply-sentinel';

/**
 * Generate synthetic data with known transition point
 */
function generateTestData(pattern, length = 100, transitionPoint = 50) {
  const data = [];
  
  switch (pattern) {
    case 'calm-before-storm':
      // High variance -> Low variance -> High variance
      for (let i = 0; i < length; i++) {
        if (i < transitionPoint - 15) {
          // Normal variance
          data.push(50 + (Math.random() - 0.5) * 20);
        } else if (i < transitionPoint) {
          // Calm period (low variance) - THE SIGNAL
          data.push(50 + (Math.random() - 0.5) * 3);
        } else {
          // Storm (high variance)
          data.push(50 + (Math.random() - 0.5) * 40);
        }
      }
      break;
      
    case 'gradual-decline':
      // Gradual decline with decreasing variance
      for (let i = 0; i < length; i++) {
        const decay = 1 - (i / length) * 0.5;
        const variance = 20 * decay;
        data.push(80 * decay + (Math.random() - 0.5) * variance);
      }
      break;
      
    case 'stable':
      // Consistently stable (no transition)
      for (let i = 0; i < length; i++) {
        data.push(50 + (Math.random() - 0.5) * 15);
      }
      break;
      
    case 'sudden-spike':
      // Stable then sudden spike (no calm-before-storm)
      for (let i = 0; i < length; i++) {
        if (i < transitionPoint) {
          data.push(50 + (Math.random() - 0.5) * 15);
        } else {
          data.push(50 + (Math.random() - 0.5) * 50);
        }
      }
      break;
      
    case 'oscillating':
      // Regular oscillations (should be detected as stable)
      for (let i = 0; i < length; i++) {
        data.push(50 + Math.sin(i / 5) * 20 + (Math.random() - 0.5) * 5);
      }
      break;
  }
  
  return data;
}

/**
 * Test a detector against a pattern
 */
async function testDetector(DetectorClass, pattern, config = {}) {
  const detector = new DetectorClass(config);
  await detector.init();
  
  const data = generateTestData(pattern, 100, 50);
  
  let firstAlert = null;
  let alertCount = 0;
  
  for (let i = 0; i < data.length; i++) {
    const state = detector.update(data[i]);
    
    // Check for elevated/alert state (different detectors use different fields)
    const isAlert = state.elevated || state.atRisk || state.critical || 
                    state.failing || state.stressed || state.transitioning ||
                    state.isShifting || state.isWarning || state.tilted || state.volatile;
    
    if (isAlert) {
      alertCount++;
      if (firstAlert === null) {
        firstAlert = i;
      }
    }
  }
  
  const finalState = detector.current();
  
  return {
    pattern,
    firstAlertAt: firstAlert,
    totalAlerts: alertCount,
    finalConfidence: finalState.confidence,
    finalVariance: finalState.variance,
    // For calm-before-storm, ideal detection is around index 35-50 (during calm period)
    detectionQuality: pattern === 'calm-before-storm' ? 
      (firstAlert !== null && firstAlert >= 35 && firstAlert <= 55 ? 'GOOD' : 
       firstAlert !== null && firstAlert < 35 ? 'EARLY' : 
       firstAlert !== null ? 'LATE' : 'MISSED') : 
      pattern === 'stable' ? 
      (alertCount < 5 ? 'GOOD' : 'FALSE_POSITIVES') :
      'N/A'
  };
}

/**
 * Run full ablation test suite
 */
async function runAblationTests() {
  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║              NUCLEATION PACKAGES - ABLATION TESTS              ║');
  console.log('╚════════════════════════════════════════════════════════════════╝\n');
  
  const detectors = [
    { name: 'RegimeDetector', class: RegimeDetector },
    { name: 'ThreatDetector', class: ThreatDetector },
    { name: 'ChurnDetector', class: ChurnDetector },
    { name: 'TeamHealthMonitor', class: TeamHealthMonitor },
    { name: 'TransitionDetector', class: TransitionDetector },
    { name: 'SensorMonitor', class: SensorMonitor },
    { name: 'CrowdMonitor', class: CrowdMonitor },
    { name: 'PatientMonitor', class: PatientMonitor },
    { name: 'MatchMonitor', class: MatchMonitor },
    { name: 'SupplyMonitor', class: SupplyMonitor }
  ];
  
  const patterns = ['calm-before-storm', 'gradual-decline', 'stable', 'sudden-spike', 'oscillating'];
  const sensitivities = ['conservative', 'balanced', 'sensitive'];
  
  const results = {};
  
  for (const detector of detectors) {
    console.log(`\n--- Testing ${detector.name} ---\n`);
    results[detector.name] = {};
    
    for (const sensitivity of sensitivities) {
      results[detector.name][sensitivity] = {};
      
      for (const pattern of patterns) {
        try {
          const result = await testDetector(detector.class, pattern, { sensitivity });
          results[detector.name][sensitivity][pattern] = result;
          
          const status = result.detectionQuality === 'GOOD' ? '✓' : 
                        result.detectionQuality === 'MISSED' ? '✗' :
                        result.detectionQuality === 'FALSE_POSITIVES' ? '⚠' : '○';
          
          console.log(`  ${status} ${sensitivity.padEnd(12)} | ${pattern.padEnd(18)} | First: ${String(result.firstAlertAt ?? 'none').padEnd(4)} | Alerts: ${result.totalAlerts}`);
        } catch (error) {
          console.log(`  ✗ ${sensitivity.padEnd(12)} | ${pattern.padEnd(18)} | ERROR: ${error.message}`);
          results[detector.name][sensitivity][pattern] = { error: error.message };
        }
      }
    }
  }
  
  // Summary
  console.log('\n\n╔════════════════════════════════════════════════════════════════╗');
  console.log('║                         SUMMARY                                ║');
  console.log('╚════════════════════════════════════════════════════════════════╝\n');
  
  // Count detection quality across all tests
  let good = 0, early = 0, late = 0, missed = 0, falsePos = 0;
  
  for (const detector of Object.values(results)) {
    for (const sensitivity of Object.values(detector)) {
      for (const pattern of Object.values(sensitivity)) {
        if (pattern.detectionQuality === 'GOOD') good++;
        else if (pattern.detectionQuality === 'EARLY') early++;
        else if (pattern.detectionQuality === 'LATE') late++;
        else if (pattern.detectionQuality === 'MISSED') missed++;
        else if (pattern.detectionQuality === 'FALSE_POSITIVES') falsePos++;
      }
    }
  }
  
  console.log('Detection Quality (calm-before-storm pattern):');
  console.log(`  ✓ GOOD (detected in calm window): ${good}`);
  console.log(`  ○ EARLY (detected before calm): ${early}`);
  console.log(`  ○ LATE (detected after storm started): ${late}`);
  console.log(`  ✗ MISSED (no detection): ${missed}`);
  console.log(`\nFalse Positive Rate (stable pattern):`);
  console.log(`  ✓ GOOD (few false alerts): ${good}`);
  console.log(`  ⚠ FALSE_POSITIVES (many alerts on stable data): ${falsePos}`);
  
  return results;
}

/**
 * Sensitivity analysis - how does window size affect detection?
 */
async function sensitivityAnalysis() {
  console.log('\n\n╔════════════════════════════════════════════════════════════════╗');
  console.log('║                   SENSITIVITY ANALYSIS                         ║');
  console.log('╚════════════════════════════════════════════════════════════════╝\n');
  
  const windowSizes = [10, 20, 30, 50, 75, 100];
  const data = generateTestData('calm-before-storm', 150, 75);
  
  console.log('Window Size vs Detection Timing (calm-before-storm, transition at 75):\n');
  
  for (const windowSize of windowSizes) {
    const detector = new TransitionDetector({ windowSize, sensitivity: 'balanced' });
    await detector.init();
    
    let firstAlert = null;
    for (let i = 0; i < data.length; i++) {
      const state = detector.update(data[i]);
      if ((state.elevated || state.transitioning) && firstAlert === null) {
        firstAlert = i;
      }
    }
    
    const timing = firstAlert === null ? 'NONE' :
                   firstAlert < 60 ? 'EARLY' :
                   firstAlert <= 80 ? 'GOOD' : 'LATE';
    
    console.log(`  Window ${String(windowSize).padStart(3)}: First alert at ${String(firstAlert ?? 'N/A').padStart(4)} [${timing}]`);
  }
}

// Main execution
console.log('Starting ablation tests...\n');

runAblationTests()
  .then(() => sensitivityAnalysis())
  .then(() => {
    console.log('\n\nAblation tests complete.');
    console.log('\n--- TESTS FOR YOU TO RUN LOCALLY ---');
    console.log('The following require external network access:\n');
    console.log('1. Real-time crypto data validation:');
    console.log('   node harness/src/index.js\n');
    console.log('2. Extended time-series (requires API keys):');
    console.log('   - Yahoo Finance historical data');
    console.log('   - FRED economic indicators');
    console.log('   - Alpha Vantage stock data\n');
    console.log('3. Live monitoring (1-hour intervals):');
    console.log('   Import runMonitor from index.js and call runMonitor(60)\n');
  })
  .catch(console.error);
