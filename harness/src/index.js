/**
 * World Trajectory Monitor
 *
 * Unified system integrating all nucleation packages to monitor
 * global phase transitions across multiple domains simultaneously.
 *
 * Domains:
 * - Financial markets (regime-shift, market-canary)
 * - Cyber threat landscape (threat-pulse)
 * - Social dynamics (crowd-phase)
 * - Geophysical systems (sensor-shift)
 * - Sentiment/fear indices (market-canary)
 */

import { RegimeDetector } from 'regime-shift';
import { ThreatDetector, ThreatCorrelator } from 'threat-pulse';
import { TransitionDetector } from 'market-canary';
import { CrowdMonitor } from 'crowd-phase';
import { SensorMonitor } from 'sensor-shift';

import { fetchAllData, pricesToReturns } from './data-fetcher.js';

/**
 * World state summary
 */
class WorldState {
  constructor() {
    this.timestamp = new Date().toISOString();
    this.domains = {};
    this.alerts = [];
    this.overallRisk = 0;
  }

  addDomain(name, state) {
    this.domains[name] = state;
    if (state.elevated || state.atRisk || state.critical || state.transitioning) {
      this.alerts.push({ domain: name, ...state });
    }
  }

  calculateOverallRisk() {
    const domainStates = Object.values(this.domains);
    if (domainStates.length === 0) return 0;

    let riskScore = 0;
    for (const state of domainStates) {
      if (state.transitioning || state.critical || state.atRisk) riskScore += 3;
      else if (state.elevated || state.declining) riskScore += 1;
    }

    this.overallRisk = riskScore / (domainStates.length * 3); // Normalize to 0-1
    return this.overallRisk;
  }

  toJSON() {
    return {
      timestamp: this.timestamp,
      overallRisk: this.overallRisk,
      alertCount: this.alerts.length,
      domains: this.domains,
      alerts: this.alerts,
    };
  }
}

/**
 * Main monitor class
 */
export class WorldTrajectoryMonitor {
  #detectors = {};
  #initialized = false;
  #history = [];

  constructor(config = {}) {
    this.config = {
      sensitivity: config.sensitivity || 'balanced',
      historyLimit: config.historyLimit || 100,
      ...config,
    };
  }

  async init() {
    console.log('Initializing World Trajectory Monitor...\n');

    // Financial markets
    this.#detectors.btcRegime = new RegimeDetector({
      sensitivity: this.config.sensitivity,
      windowSize: 30,
    });
    await this.#detectors.btcRegime.init();

    this.#detectors.ethRegime = new RegimeDetector({
      sensitivity: this.config.sensitivity,
      windowSize: 30,
    });
    await this.#detectors.ethRegime.init();

    // Sentiment/fear tracking
    this.#detectors.sentiment = new TransitionDetector({
      sensitivity: this.config.sensitivity,
      windowSize: 14,
    });
    await this.#detectors.sentiment.init();

    // Geophysical (earthquake activity as instability proxy)
    this.#detectors.geophysical = new SensorMonitor({
      sensitivity: this.config.sensitivity,
      windowSize: 7,
    });
    await this.#detectors.geophysical.init();

    // Cross-domain correlator
    this.#detectors.correlator = new ThreatCorrelator(5);
    await this.#detectors.correlator.init();
    this.#detectors.correlator.registerSource('financial');
    this.#detectors.correlator.registerSource('sentiment');
    this.#detectors.correlator.registerSource('geophysical');

    this.#initialized = true;
    console.log('All detectors initialized.\n');
  }

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('WorldTrajectoryMonitor not initialized. Call init() first.');
    }
  }

  /**
   * Process real data and return world state
   */
  async update() {
    this.#ensureInit();

    const state = new WorldState();

    // Fetch real data
    const data = await fetchAllData();

    if (data.crypto.error) {
      console.log('Warning: Crypto data fetch failed:', data.crypto.error);
    } else {
      // Process Bitcoin
      if (data.crypto.bitcoin?.prices) {
        const btcReturns = pricesToReturns(data.crypto.bitcoin.prices);
        const btcState = this.#detectors.btcRegime.updateBatch(btcReturns.map((r) => r.value));
        state.addDomain('bitcoin', {
          regime: btcState.regime,
          transitioning: btcState.isShifting,
          elevated: btcState.isWarning,
          confidence: btcState.confidence,
          variance: btcState.variance,
          currentPrice: data.crypto.bitcoin.prices.slice(-1)[0]?.price,
        });
      }

      // Process Ethereum
      if (data.crypto.ethereum?.prices) {
        const ethReturns = pricesToReturns(data.crypto.ethereum.prices);
        const ethState = this.#detectors.ethRegime.updateBatch(ethReturns.map((r) => r.value));
        state.addDomain('ethereum', {
          regime: ethState.regime,
          transitioning: ethState.isShifting,
          elevated: ethState.isWarning,
          confidence: ethState.confidence,
          variance: ethState.variance,
          currentPrice: data.crypto.ethereum.prices.slice(-1)[0]?.price,
        });
      }
    }

    // Process sentiment
    if (data.sentiment?.history) {
      const sentimentValues = data.sentiment.history.map((h) => h.value);
      const sentState = this.#detectors.sentiment.updateBatch(sentimentValues);
      state.addDomain('sentiment', {
        phase: sentState.phase,
        transitioning: sentState.transitioning,
        elevated: sentState.elevated,
        confidence: sentState.confidence,
        currentValue: data.sentiment.current,
        classification: data.sentiment.classification,
      });
    }

    // Process geophysical
    if (data.geophysical?.byDay) {
      const dailyMags = Object.values(data.geophysical.byDay).map((d) => d.maxMag);
      if (dailyMags.length > 0) {
        const geoState = this.#detectors.geophysical.updateBatch(dailyMags);
        state.addDomain('geophysical', {
          level: geoState.level,
          failing: geoState.failing,
          elevated: geoState.elevated,
          confidence: geoState.confidence,
          weeklyQuakes: data.geophysical.total,
          significantQuakes: data.geophysical.significant,
        });
      }
    }

    // Cross-domain correlation
    const now = Date.now();
    if (state.domains.bitcoin) {
      this.#detectors.correlator.updateSource(
        'financial',
        new Float64Array([
          state.domains.bitcoin.variance || 0,
          state.domains.bitcoin.confidence || 0,
          state.domains.bitcoin.transitioning ? 1 : 0,
          0,
          0,
        ]),
        now
      );
    }
    if (state.domains.sentiment) {
      this.#detectors.correlator.updateSource(
        'sentiment',
        new Float64Array([
          (state.domains.sentiment.currentValue || 50) / 100,
          state.domains.sentiment.confidence || 0,
          state.domains.sentiment.transitioning ? 1 : 0,
          0,
          0,
        ]),
        now
      );
    }
    if (state.domains.geophysical) {
      this.#detectors.correlator.updateSource(
        'geophysical',
        new Float64Array([
          (state.domains.geophysical.weeklyQuakes || 0) / 1000,
          state.domains.geophysical.confidence || 0,
          state.domains.geophysical.failing ? 1 : 0,
          0,
          0,
        ]),
        now
      );
    }

    // Check cross-domain correlations
    const finSentCorr = this.#detectors.correlator.getCorrelation('financial', 'sentiment');
    const finGeoCorr = this.#detectors.correlator.getCorrelation('financial', 'geophysical');

    state.addDomain('correlations', {
      financialSentiment: finSentCorr,
      financialGeophysical: finGeoCorr,
      elevated: finSentCorr > 0.3 || finGeoCorr > 0.3,
    });

    // Calculate overall risk
    state.calculateOverallRisk();

    // Store in history
    this.#history.push(state.toJSON());
    if (this.#history.length > this.config.historyLimit) {
      this.#history.shift();
    }

    return state;
  }

  /**
   * Get historical states
   */
  getHistory() {
    return this.#history;
  }

  /**
   * Generate summary report
   */
  generateReport(state) {
    const lines = [
      '╔════════════════════════════════════════════════════════════════╗',
      '║           WORLD TRAJECTORY MONITOR - STATUS REPORT            ║',
      '╠════════════════════════════════════════════════════════════════╣',
      `║  Timestamp: ${state.timestamp.padEnd(48)}║`,
      `║  Overall Risk: ${(state.overallRisk * 100).toFixed(1).padStart(5)}%${' '.repeat(44)}║`,
      `║  Active Alerts: ${String(state.alerts.length).padEnd(46)}║`,
      '╠════════════════════════════════════════════════════════════════╣',
      '║  DOMAIN STATUS                                                 ║',
      '╟────────────────────────────────────────────────────────────────╢',
    ];

    for (const [domain, data] of Object.entries(state.domains)) {
      const status =
        data.transitioning || data.failing || data.atRisk ? '🔴' : data.elevated ? '🟡' : '🟢';
      const conf = data.confidence ? `${(data.confidence * 100).toFixed(0)}%` : 'N/A';
      lines.push(
        `║  ${status} ${domain.padEnd(15)} Conf: ${conf.padEnd(6)} ${this.#formatDomainDetail(domain, data).padEnd(30)}║`
      );
    }

    lines.push('╠════════════════════════════════════════════════════════════════╣');

    if (state.alerts.length > 0) {
      lines.push('║  ⚠️  ALERTS                                                    ║');
      lines.push('╟────────────────────────────────────────────────────────────────╢');
      for (const alert of state.alerts) {
        lines.push(`║    • ${alert.domain}: Phase transition detected${' '.repeat(24)}║`);
      }
    } else {
      lines.push('║  ✓ No active alerts                                            ║');
    }

    lines.push('╚════════════════════════════════════════════════════════════════╝');

    return lines.join('\n');
  }

  #formatDomainDetail(domain, data) {
    switch (domain) {
      case 'bitcoin':
      case 'ethereum':
        return `$${data.currentPrice?.toFixed(0) || 'N/A'} | ${data.regime || 'unknown'}`;
      case 'sentiment':
        return `${data.currentValue || 'N/A'} (${data.classification || 'unknown'})`;
      case 'geophysical':
        return `${data.weeklyQuakes || 0} quakes (${data.significantQuakes || 0} sig)`;
      case 'correlations':
        return `F-S: ${data.financialSentiment?.toFixed(2) || 'N/A'}`;
      default:
        return '';
    }
  }
}

/**
 * Run continuous monitoring
 */
export async function runMonitor(intervalMinutes = 60) {
  const monitor = new WorldTrajectoryMonitor({ sensitivity: 'balanced' });
  await monitor.init();

  const tick = async () => {
    try {
      const state = await monitor.update();
      console.clear();
      console.log(monitor.generateReport(state.toJSON()));
      console.log('\nNext update in', intervalMinutes, 'minutes...');
    } catch (error) {
      console.error('Monitor error:', error.message);
    }
  };

  // Initial run
  await tick();

  // Schedule updates
  setInterval(tick, intervalMinutes * 60 * 1000);
}

// CLI execution
if (import.meta.url === `file://${process.argv[1]}`) {
  const monitor = new WorldTrajectoryMonitor({ sensitivity: 'balanced' });

  monitor
    .init()
    .then(() => monitor.update())
    .then((state) => {
      console.log(monitor.generateReport(state.toJSON()));
      console.log('\n--- Raw State ---');
      console.log(JSON.stringify(state.toJSON(), null, 2));
    })
    .catch(console.error);
}

export default WorldTrajectoryMonitor;
