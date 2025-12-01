/**
 * churn-harbinger
 *
 * Predict customer churn before it happens using behavioral variance
 * analysis. Built for SaaS, subscription businesses, and product teams.
 *
 * The core insight: users don't just stop using your product — they
 * disengage gradually. That disengagement has a signature: reduced
 * variance in behavior as they settle into minimal-use patterns before
 * churning.
 *
 * @example
 * ```js
 * import { ChurnDetector } from 'churn-harbinger';
 *
 * const detector = new ChurnDetector();
 * await detector.init();
 *
 * // Feed daily engagement scores
 * for (const day of userActivity) {
 *   const state = detector.update(day.engagementScore);
 *   if (state.atRisk) {
 *     console.log(`User at churn risk: ${state.riskLevel}`);
 *   }
 * }
 * ```
 */

import { createRequire } from 'module';
import { readFile } from 'fs/promises';

let nucleationModule = null;
let initialized = false;

/**
 * Initialize the WASM module.
 * @returns {Promise<void>}
 */
export async function initialize() {
  if (!initialized) {
    nucleationModule = await import('nucleation-wasm');

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

/**
 * Churn risk levels
 * @readonly
 * @enum {string}
 */
export const RiskLevel = {
  /** Healthy engagement, low churn risk */
  HEALTHY: 'healthy',
  /** Engagement declining, monitor closely */
  COOLING: 'cooling',
  /** High churn probability, intervene now */
  AT_RISK: 'at-risk',
  /** Active disengagement pattern */
  CHURNING: 'churning',
};

/**
 * Map internal phase to risk level
 */
function phaseToRiskLevel(phase) {
  const Phase = nucleationModule.Phase;
  switch (phase) {
    case Phase.Stable:
      return RiskLevel.HEALTHY;
    case Phase.Approaching:
      return RiskLevel.COOLING;
    case Phase.Critical:
      return RiskLevel.AT_RISK;
    case Phase.Transitioning:
      return RiskLevel.CHURNING;
    default:
      return RiskLevel.HEALTHY;
  }
}

/**
 * Configuration for ChurnDetector
 * @typedef {Object} ChurnConfig
 * @property {'conservative'|'balanced'|'sensitive'} [sensitivity='balanced'] - Detection sensitivity
 * @property {number} [windowSize=30] - Days/events for baseline (default 30 for monthly patterns)
 * @property {number} [threshold=2.0] - Standard deviations for risk flag
 */

/**
 * Current churn risk assessment
 * @typedef {Object} ChurnState
 * @property {string} riskLevel - Current risk level (healthy/cooling/at-risk/churning)
 * @property {boolean} atRisk - True if high churn probability
 * @property {boolean} declining - True if engagement trending down
 * @property {number} confidence - Confidence in assessment (0-1)
 * @property {number} variance - Current engagement variance
 * @property {number} trend - Engagement trend indicator
 * @property {number} dataPoints - Total observations processed
 */

/**
 * Customer churn detector for SaaS and subscription products.
 *
 * Monitors engagement variance to identify the characteristic
 * "settling" pattern that precedes churn. Users don't suddenly
 * leave — they gradually disengage, and that disengagement has
 * a measurable signature.
 *
 * @example
 * ```js
 * // Track user engagement over time
 * const detector = new ChurnDetector({ sensitivity: 'sensitive' });
 * await detector.init();
 *
 * userDailyActivity.forEach(day => {
 *   const state = detector.update(day.sessions * day.avgDuration);
 *   if (state.riskLevel === 'at-risk') {
 *     triggerCSMAlert(userId, state);
 *   }
 * });
 * ```
 */
export class ChurnDetector {
  #detector = null;
  #config = null;
  #initialized = false;

  /**
   * Create a new churn detector.
   * @param {ChurnConfig} [config={}]
   */
  constructor(config = {}) {
    this.#config = {
      sensitivity: config.sensitivity || 'balanced',
      windowSize: config.windowSize || 30,
      threshold: config.threshold,
    };
  }

  /**
   * Initialize the detector.
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

    detectorConfig.window_size = this.#config.windowSize;

    if (this.#config.threshold) {
      detectorConfig.threshold = this.#config.threshold;
    }

    this.#detector = new NucleationDetector(detectorConfig);
    this.#initialized = true;
  }

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('ChurnDetector not initialized. Call init() first.');
    }
  }

  /**
   * Process a single engagement observation.
   * @param {number} engagementScore - Engagement metric (sessions, time, actions, etc.)
   * @returns {ChurnState}
   */
  update(engagementScore) {
    this.#ensureInit();

    const phase = this.#detector.update(engagementScore);
    const riskLevel = phaseToRiskLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      riskLevel,
      atRisk: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  /**
   * Process batch of engagement data.
   * @param {number[]|Float64Array} scores
   * @returns {ChurnState}
   */
  updateBatch(scores) {
    this.#ensureInit();

    const arr = scores instanceof Float64Array ? scores : new Float64Array(scores);
    const phase = this.#detector.update_batch(arr);
    const riskLevel = phaseToRiskLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      riskLevel,
      atRisk: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  /**
   * Get current state without new data.
   * @returns {ChurnState}
   */
  current() {
    this.#ensureInit();

    const phase = this.#detector.currentPhase();
    const riskLevel = phaseToRiskLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      riskLevel,
      atRisk: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  /**
   * Reset detector (e.g., after user re-engages).
   */
  reset() {
    this.#ensureInit();
    this.#detector.reset();
  }

  /**
   * Serialize state.
   * @returns {string}
   */
  serialize() {
    this.#ensureInit();
    return this.#detector.serialize();
  }

  /**
   * Restore from serialized state.
   * @param {string} json
   * @returns {Promise<ChurnDetector>}
   */
  static async deserialize(json) {
    await initialize();
    const { NucleationDetector } = nucleationModule;
    const detector = new ChurnDetector();
    detector.#detector = NucleationDetector.deserialize(json);
    detector.#initialized = true;
    return detector;
  }
}

/**
 * Cohort-level churn monitoring.
 * Tracks multiple users and identifies cohort-wide disengagement patterns.
 */
export class CohortMonitor {
  #shepherd = null;
  #initialized = false;
  #users = new Map(); // userId -> metadata

  /**
   * Create a cohort monitor.
   * @param {number} [behaviorCategories=10] - Behavior category count
   */
  constructor(behaviorCategories = 10) {
    this.categories = behaviorCategories;
  }

  async init() {
    await initialize();
    const { Shepherd } = nucleationModule;
    this.#shepherd = new Shepherd(this.categories);
    this.#initialized = true;
  }

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('CohortMonitor not initialized. Call init() first.');
    }
  }

  /**
   * Add a user to the cohort.
   * @param {string} userId
   * @param {Object} [metadata={}] - User metadata (plan, signup date, etc.)
   * @param {Float64Array} [initialBehavior] - Initial behavior distribution
   */
  addUser(userId, metadata = {}, initialBehavior = null) {
    this.#ensureInit();
    this.#shepherd.registerActor(userId, initialBehavior);
    this.#users.set(userId, metadata);
  }

  /**
   * Update user behavior.
   * @param {string} userId
   * @param {Float64Array} behavior - Behavior distribution across categories
   * @param {number} [timestamp=Date.now()]
   * @returns {Array} Any churn alerts triggered
   */
  updateUser(userId, behavior, timestamp = Date.now()) {
    this.#ensureInit();
    if (!this.#users.has(userId)) {
      this.addUser(userId);
    }
    return this.#shepherd.updateActor(userId, behavior, timestamp);
  }

  /**
   * Get behavioral divergence between two users.
   * High divergence from healthy users = potential churn signal.
   * @param {string} userA
   * @param {string} userB
   * @returns {number|undefined}
   */
  getDivergence(userA, userB) {
    this.#ensureInit();
    return this.#shepherd.conflictPotential(userA, userB);
  }

  /**
   * Check entire cohort for churn signals.
   * @param {number} [timestamp=Date.now()]
   * @returns {Array}
   */
  checkCohort(timestamp = Date.now()) {
    this.#ensureInit();
    return this.#shepherd.checkAllDyads(timestamp);
  }

  /**
   * Get all tracked users.
   * @returns {string[]}
   */
  getUsers() {
    return Array.from(this.#users.keys());
  }

  /**
   * Get user metadata.
   * @param {string} userId
   * @returns {Object|undefined}
   */
  getUserMetadata(userId) {
    return this.#users.get(userId);
  }
}

/**
 * Quick churn risk assessment for a user's engagement history.
 * @param {number[]} engagementHistory - Array of engagement scores
 * @param {ChurnConfig} [config={}]
 * @returns {Promise<{atRisk: boolean, riskLevel: string, confidence: number}>}
 */
export async function assessChurnRisk(engagementHistory, config = {}) {
  const detector = new ChurnDetector(config);
  await detector.init();

  const state = detector.updateBatch(engagementHistory);

  return {
    atRisk: state.atRisk,
    declining: state.declining,
    riskLevel: state.riskLevel,
    confidence: state.confidence,
  };
}

export default ChurnDetector;
