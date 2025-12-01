/**
 * Example: Detecting regime shifts in stock price data
 *
 * Run with: node examples/stock-regime.js
 */

import { RegimeDetector, detectRegimeShift } from '../src/index.js';

// Simulated price data with a regime shift
// First 50: stable trending market
// Next 20: calm before the storm (variance drops significantly)
// Last 30: high volatility regime
function generatePriceData() {
  const prices = [];
  let price = 100;

  // Phase 1: Normal trending market (moderate volatility)
  for (let i = 0; i < 50; i++) {
    price += (Math.random() - 0.45) * 3; // Normal volatility
    prices.push(price);
  }

  // Phase 2: Calm before the storm (variance drops dramatically)
  for (let i = 0; i < 25; i++) {
    price += (Math.random() - 0.5) * 0.3; // Very low volatility - the signal
    prices.push(price);
  }

  // Phase 3: High volatility regime (the storm)
  for (let i = 0; i < 35; i++) {
    price += (Math.random() - 0.55) * 8; // High volatility crash
    prices.push(price);
  }

  return prices;
}

async function main() {
  console.log('=== Regime Shift Detection Example ===\n');

  const prices = generatePriceData();

  // Method 1: Stream processing
  console.log('--- Streaming Detection ---');
  const detector = new RegimeDetector({ sensitivity: 'balanced' });
  await detector.init();

  let lastRegime = null;

  prices.forEach((price, i) => {
    const state = detector.update(price);

    // Report regime changes
    if (state.regime !== lastRegime) {
      console.log(
        `[${i}] Regime change: ${lastRegime || 'initial'} -> ${state.regime} (confidence: ${state.confidence.toFixed(2)})`
      );
      lastRegime = state.regime;
    }

    // Report warnings
    if (state.isWarning && !state.isShifting) {
      console.log(
        `[${i}] Warning: Potential regime shift approaching (inflection: ${state.inflection.toFixed(2)})`
      );
    }
  });

  console.log(`\nFinal state:`, detector.current());

  // Method 2: Batch processing
  console.log('\n--- Batch Detection ---');
  const result = await detectRegimeShift(prices);
  console.log('Quick analysis result:', result);

  // Method 3: Different sensitivities
  console.log('\n--- Sensitivity Comparison ---');
  for (const sensitivity of ['conservative', 'balanced', 'sensitive']) {
    const result = await detectRegimeShift(prices, { sensitivity });
    console.log(
      `${sensitivity}: regime=${result.regime}, shifting=${result.shifting}, confidence=${result.confidence.toFixed(2)}`
    );
  }
}

main().catch(console.error);
