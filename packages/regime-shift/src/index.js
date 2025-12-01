/**
 * regime-shift
 *
 * Detect market regime changes before they happen using variance-based
 * phase transition detection. Built on nucleation-wasm.
 *
 * @example
 * ```js
 * import { RegimeDetector } from 'regime-shift';
 *
 * const detector = new RegimeDetector();
 *
 * for (const price of priceHistory) {
 *   const regime = detector.update(price);
 *   if (regime.isShifting) {
 *     console.log(`Regime shift detected! Confidence: ${regime.confidence}`);
 *   }
 * }
 * ```
 */

import { createRequire } from 'module';
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// Dynamic import for nucleation-wasm
let nucleationModule = null;
let initialized = false;

/**
 * Initialize the WASM module. Called automatically on first use.
 * @returns {Promise<void>}
 */
export async function initialize() {
  if (!initialized) {
    // Import the module
    nucleationModule = await import('nucleation-wasm');

    // In Node.js, we need to provide the WASM bytes directly
    if (typeof window === 'undefined') {
      const require = createRequire(import.meta.url);
      const wasmPath = require.resolve('nucleation-wasm/nucleation_bg.wasm');
      const wasmBytes = await readFile(wasmPath);
      await nucleationModule.default(wasmBytes);
    } else {
      await nucleationModule.default();
    }

    initialized = true;
  }
}

// Lazy getters for exports
function getExport(name) {
  if (!nucleationModule) {
    throw new Error('Module not initialized. Call initialize() first.');
  }
  return nucleationModule[name];
}

const NucleationDetector = {
  get new() {
    return getExport('NucleationDetector');
  },
};
const DetectorConfig = {
  get new() {
    return getExport('DetectorConfig');
  },
};
const Phase = {
  get Stable() {
    return getExport('Phase').Stable;
  },
  get Approaching() {
    return getExport('Phase').Approaching;
  },
  get Critical() {
    return getExport('Phase').Critical;
  },
  get Transitioning() {
    return getExport('Phase').Transitioning;
  },
};

/**
 * Market regime types
 * @readonly
 * @enum {string}
 */
export const Regime = {
  /** Low volatility, stable trend */
  STABLE: 'stable',
  /** Volatility increasing, potential shift ahead */
  WARMING: 'warming',
  /** High probability of imminent regime change */
  CRITICAL: 'critical',
  /** Regime change in progress */
  SHIFTING: 'shifting',
};

/**
 * Map internal phase to market regime
 * @param {number} phase
 * @returns {string}
 */
function phaseToRegime(phase) {
  const Phase = nucleationModule.Phase;
  switch (phase) {
    case Phase.Stable:
      return Regime.STABLE;
    case Phase.Approaching:
      return Regime.WARMING;
    case Phase.Critical:
      return Regime.CRITICAL;
    case Phase.Transitioning:
      return Regime.SHIFTING;
    default:
      return Regime.STABLE;
  }
}

/**
 * Configuration options for RegimeDetector
 * @typedef {Object} RegimeConfig
 * @property {'conservative'|'balanced'|'sensitive'} [sensitivity='balanced'] - Detection sensitivity
 * @property {number} [windowSize=50] - Rolling window size for variance calculation
 * @property {number} [threshold=2.0] - Z-score threshold for regime detection
 */

/**
 * Result from regime detection update
 * @typedef {Object} RegimeState
 * @property {string} regime - Current regime (stable, warming, critical, shifting)
 * @property {boolean} isShifting - True if regime change detected
 * @property {boolean} isWarning - True if approaching regime change
 * @property {number} confidence - Confidence in current assessment (0-1)
 * @property {number} variance - Current rolling variance
 * @property {number} inflection - Variance inflection magnitude (z-score)
 * @property {number} observations - Total observations processed
 */

/**
 * Market regime change detector.
 *
 * Uses variance inflection detection to identify regime changes
 * before they fully manifest. The core insight: variance typically
 * *decreases* before major market transitions (the "calm before the storm").
 *
 * @example
 * ```js
 * // Basic usage with price data
 * const detector = new RegimeDetector();
 * await detector.init();
 *
 * prices.forEach(price => {
 *   const state = detector.update(price);
 *   console.log(`Regime: ${state.regime}, Confidence: ${state.confidence}`);
 * });
 * ```
 *
 * @example
 * ```js
 * // Using returns instead of prices
 * const detector = new RegimeDetector({ sensitivity: 'sensitive' });
 * await detector.init();
 *
 * returns.forEach(ret => {
 *   const state = detector.update(ret);
 *   if (state.isWarning) {
 *     console.log('Potential regime shift approaching');
 *   }
 * });
 * ```
 */
export class RegimeDetector {
  #detector = null;
  #config = null;
  #initialized = false;

  /**
   * Create a new regime detector.
   * @param {RegimeConfig} [config={}] - Configuration options
   */
  constructor(config = {}) {
    this.#config = {
      sensitivity: config.sensitivity || 'balanced',
      windowSize: config.windowSize,
      threshold: config.threshold,
    };
  }

  /**
   * Initialize the detector. Must be called before use.
   * @returns {Promise<void>}
   */
  async init() {
    await initialize();

    const { DetectorConfig, NucleationDetector } = nucleationModule;

    let detectorConfig;

    switch (this.#config.sensitivity) {
      case 'conservative':
        detectorConfig = DetectorConfig.conservative();
        break;
      case 'sensitive':
        detectorConfig = DetectorConfig.sensitive();
        break;
      default:
        detectorConfig = new DetectorConfig();
    }

    if (this.#config.windowSize) {
      detectorConfig.window_size = this.#config.windowSize;
    }
    if (this.#config.threshold) {
      detectorConfig.threshold = this.#config.threshold;
    }

    this.#detector = new NucleationDetector(detectorConfig);
    this.#initialized = true;
  }

  /**
   * Ensure detector is initialized
   * @private
   */
  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('RegimeDetector not initialized. Call init() first.');
    }
  }

  /**
   * Process a single price/return observation.
   * @param {number} value - Price or return value
   * @returns {RegimeState} Current regime state
   */
  update(value) {
    this.#ensureInit();

    const phase = this.#detector.update(value);
    const regime = phaseToRegime(phase);

    return {
      regime,
      isShifting: phase === Phase.Transitioning,
      isWarning: phase === Phase.Approaching || phase === Phase.Critical,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      inflection: this.#detector.inflectionMagnitude(),
      observations: this.#detector.count(),
    };
  }

  /**
   * Process multiple observations at once.
   * @param {number[]|Float64Array} values - Array of price/return values
   * @returns {RegimeState} Final regime state after all observations
   */
  updateBatch(values) {
    this.#ensureInit();

    const arr = values instanceof Float64Array ? values : new Float64Array(values);
    const phase = this.#detector.update_batch(arr);
    const regime = phaseToRegime(phase);

    return {
      regime,
      isShifting: phase === Phase.Transitioning,
      isWarning: phase === Phase.Approaching || phase === Phase.Critical,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      inflection: this.#detector.inflectionMagnitude(),
      observations: this.#detector.count(),
    };
  }

  /**
   * Get the current regime without adding new data.
   * @returns {RegimeState} Current regime state
   */
  current() {
    this.#ensureInit();

    const phase = this.#detector.currentPhase();
    const regime = phaseToRegime(phase);

    return {
      regime,
      isShifting: phase === Phase.Transitioning,
      isWarning: phase === Phase.Approaching || phase === Phase.Critical,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      inflection: this.#detector.inflectionMagnitude(),
      observations: this.#detector.count(),
    };
  }

  /**
   * Reset the detector state.
   */
  reset() {
    this.#ensureInit();
    this.#detector.reset();
  }

  /**
   * Serialize detector state for persistence.
   * @returns {string} JSON string of detector state
   */
  serialize() {
    this.#ensureInit();
    return this.#detector.serialize();
  }

  /**
   * Create a detector from serialized state.
   * @param {string} json - Serialized detector state
   * @returns {Promise<RegimeDetector>} Restored detector
   */
  static async deserialize(json) {
    await initialize();
    const { NucleationDetector } = nucleationModule;
    const detector = new RegimeDetector();
    detector.#detector = NucleationDetector.deserialize(json);
    detector.#initialized = true;
    return detector;
  }
}

/**
 * Quick check if a price series shows regime shift signals.
 * Convenience function for one-off analysis.
 *
 * @param {number[]} prices - Price series to analyze
 * @param {RegimeConfig} [config={}] - Detection configuration
 * @returns {Promise<{shifting: boolean, regime: string, confidence: number}>}
 *
 * @example
 * ```js
 * import { detectRegimeShift } from 'regime-shift';
 *
 * const result = await detectRegimeShift(closingPrices);
 * if (result.shifting) {
 *   console.log('Regime shift detected!');
 * }
 * ```
 */
export async function detectRegimeShift(prices, config = {}) {
  const detector = new RegimeDetector(config);
  await detector.init();

  const state = detector.updateBatch(prices);

  return {
    shifting: state.isShifting,
    warning: state.isWarning,
    regime: state.regime,
    confidence: state.confidence,
  };
}

export default RegimeDetector;
