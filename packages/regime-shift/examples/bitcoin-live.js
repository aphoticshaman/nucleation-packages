/**
 * Example: Detecting regime shifts in REAL Bitcoin price data
 * Uses CoinGecko free API - no auth required
 * Falls back to embedded sample data if API unavailable
 *
 * Run with: node examples/bitcoin-live.js
 */

import { RegimeDetector } from '../src/index.js';

const COINGECKO_API = 'https://api.coingecko.com/api/v3';

// Sample BTC-like data showing clear regime shift pattern
// Phase 1: Normal volatility (~2-3% daily moves)
// Phase 2: Calm before storm (~0.1% daily moves)
// Phase 3: High volatility breakout (~5-10% daily moves)
const SAMPLE_BTC_PRICES = [
  // Phase 1: Normal market (days 1-50)
  67234, 68891, 67102, 69543, 66892, 68123, 66234, 69102, 67543, 70765, 68234, 66654, 69123, 67543,
  70987, 68234, 66102, 69234, 67543, 71102, 69234, 67102, 70543, 68234, 66102, 69543, 67234, 70102,
  68543, 66876, 69234, 67102, 70234, 68543, 66876, 69234, 67102, 70543, 68876, 66234, 68102, 70234,
  67543, 69876, 67234, 70102, 68543, 66876, 69234, 71102,
  // Phase 2: Calm before storm - LOW VOLATILITY (days 51-70)
  70234, 70312, 70287, 70298, 70276, 70301, 70289, 70295, 70282, 70291, 70288, 70294, 70279, 70286,
  70293, 70281, 70290, 70284, 70292, 70287,
  // Phase 3: The storm - HIGH VOLATILITY breakout (days 71-90)
  72234, 68102, 75543, 71876, 78234, 73102, 80543, 76876, 83234, 78102, 86543, 81234, 89102, 84543,
  92876, 87234, 95102, 89543, 98876, 93234,
];

/**
 * Fetch real price data from CoinGecko
 * @param {string} coin - Coin ID (bitcoin, ethereum, etc.)
 * @param {number} days - Number of days of history
 */
async function fetchPrices(coin = 'bitcoin', days = 90) {
  try {
    const url = `${COINGECKO_API}/coins/${coin}/market_chart?vs_currency=usd&days=${days}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();

    // Extract prices: [[timestamp, price], ...]
    return data.prices.map(([timestamp, price]) => ({
      timestamp: new Date(timestamp),
      price,
    }));
  } catch (e) {
    console.log(`⚠️  API unavailable (${e.message}), using sample data\n`);
    // Return sample data with fake timestamps
    const now = Date.now();
    const dayMs = 24 * 60 * 60 * 1000;
    return SAMPLE_BTC_PRICES.map((price, i) => ({
      timestamp: new Date(now - (SAMPLE_BTC_PRICES.length - i) * dayMs),
      price,
    }));
  }
}

/**
 * Calculate log returns from prices
 */
function toReturns(prices) {
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push({
      timestamp: prices[i].timestamp,
      value: Math.log(prices[i].price / prices[i - 1].price),
    });
  }
  return returns;
}

async function main() {
  console.log('=== Bitcoin Regime Detection (Live Data) ===\n');

  // Fetch real data
  console.log('Fetching 90 days of Bitcoin price data from CoinGecko...');
  const priceData = await fetchPrices('bitcoin', 90);
  console.log(`Got ${priceData.length} data points\n`);

  // Convert to returns (better for regime detection)
  const returns = toReturns(priceData);

  // Initialize detector with sensitive settings for demo
  const detector = new RegimeDetector({
    sensitivity: 'sensitive',
    windowSize: 20, // Shorter window to catch changes faster
    threshold: 1.5, // Lower threshold for demo
  });
  await detector.init();

  // Track regime changes
  let lastRegime = null;
  const regimeChanges = [];

  console.log('--- Scanning for regime shifts ---\n');

  for (const { timestamp, value } of returns) {
    const state = detector.update(value);

    if (state.regime !== lastRegime) {
      const change = {
        date: timestamp.toISOString().split('T')[0],
        from: lastRegime || 'start',
        to: state.regime,
        confidence: state.confidence.toFixed(3),
        variance: state.variance.toFixed(6),
      };
      regimeChanges.push(change);

      if (lastRegime !== null) {
        console.log(
          `${change.date}: ${change.from} → ${change.to} (confidence: ${change.confidence})`
        );
      }

      lastRegime = state.regime;
    }
  }

  // Summary
  console.log('\n--- Summary ---');
  console.log(`Total regime changes: ${regimeChanges.length - 1}`); // Exclude initial
  console.log(`Current regime: ${lastRegime}`);

  const final = detector.current();
  console.log(`Current confidence: ${final.confidence.toFixed(3)}`);
  console.log(`Current variance: ${final.variance.toFixed(6)}`);
  console.log(`Inflection magnitude: ${final.inflection.toFixed(3)}`);

  // Warning status
  if (final.isWarning) {
    console.log('\n⚠️  WARNING: Potential regime shift approaching');
  }
  if (final.isShifting) {
    console.log('\n🔴 ALERT: Regime shift in progress');
  }

  // Compare using sample data variations (simulates multi-asset)
  console.log('\n--- Multi-Asset Comparison (Sample Data) ---');
  const assets = [
    { name: 'bitcoin', volatilityMult: 1.0 },
    { name: 'ethereum', volatilityMult: 1.3 },
    { name: 'solana', volatilityMult: 1.8 },
  ];

  for (const asset of assets) {
    const data = SAMPLE_BTC_PRICES.slice(-30).map((p, i) => {
      // Add asset-specific volatility
      const noise = (Math.random() - 0.5) * p * 0.02 * asset.volatilityMult;
      return p * (asset.name === 'ethereum' ? 0.04 : asset.name === 'solana' ? 0.002 : 1) + noise;
    });
    const ret = toReturns(data.map((price, i) => ({ timestamp: new Date(), price })));

    const det = new RegimeDetector({ sensitivity: 'balanced' });
    await det.init();

    const state = det.updateBatch(ret.map((r) => r.value));
    console.log(
      `${asset.name.padEnd(10)}: ${state.regime.padEnd(10)} (confidence: ${state.confidence.toFixed(2)}, var: ${state.variance.toFixed(6)})`
    );
  }
}

main().catch(console.error);
