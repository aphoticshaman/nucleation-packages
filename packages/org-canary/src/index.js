/**
 * org-canary
 *
 * Detect organizational dysfunction before it surfaces. Culture clash
 * prediction, M&A integration risk, and team health monitoring.
 *
 * The core insight: organizational conflict doesn't explode suddenly —
 * it builds through measurable tension patterns. Teams approaching
 * dysfunction show characteristic variance changes in communication,
 * collaboration, and sentiment metrics.
 *
 * @example
 * ```js
 * import { TeamHealthMonitor } from 'org-canary';
 *
 * const monitor = new TeamHealthMonitor();
 * await monitor.init();
 *
 * // Feed weekly team metrics
 * for (const week of teamMetrics) {
 *   const state = monitor.update(week.healthScore);
 *   if (state.stressed) {
 *     console.log(`Team stress detected: ${state.healthLevel}`);
 *   }
 * }
 * ```
 */

import { createRequire } from 'module';
import { readFile } from 'fs/promises';

let nucleationModule = null;
let initialized = false;

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
 * Organizational health levels
 * @readonly
 * @enum {string}
 */
export const HealthLevel = {
  /** Healthy team dynamics */
  THRIVING: 'thriving',
  /** Some tension, worth monitoring */
  STRAINED: 'strained',
  /** Significant dysfunction risk */
  STRESSED: 'stressed',
  /** Active dysfunction/conflict */
  CRITICAL: 'critical',
};

function phaseToHealthLevel(phase) {
  const Phase = nucleationModule.Phase;
  switch (phase) {
    case Phase.Stable:
      return HealthLevel.THRIVING;
    case Phase.Approaching:
      return HealthLevel.STRAINED;
    case Phase.Critical:
      return HealthLevel.STRESSED;
    case Phase.Transitioning:
      return HealthLevel.CRITICAL;
    default:
      return HealthLevel.THRIVING;
  }
}

/**
 * @typedef {Object} OrgConfig
 * @property {'conservative'|'balanced'|'sensitive'} [sensitivity='balanced']
 * @property {number} [windowSize=12] - Weeks for baseline (default 12 = quarterly)
 * @property {number} [threshold=2.0]
 */

/**
 * @typedef {Object} OrgHealthState
 * @property {string} healthLevel
 * @property {boolean} stressed - True if significant dysfunction risk
 * @property {boolean} declining - True if health trending down
 * @property {number} confidence
 * @property {number} variance
 * @property {number} trend
 * @property {number} dataPoints
 */

/**
 * Team health monitor for organizational dynamics.
 */
export class TeamHealthMonitor {
  #detector = null;
  #config = null;
  #initialized = false;

  constructor(config = {}) {
    this.#config = {
      sensitivity: config.sensitivity || 'balanced',
      windowSize: config.windowSize || 12,
      threshold: config.threshold,
    };
  }

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
      throw new Error('TeamHealthMonitor not initialized. Call init() first.');
    }
  }

  /**
   * Process a single health observation.
   * @param {number} healthScore - Team health metric
   * @returns {OrgHealthState}
   */
  update(healthScore) {
    this.#ensureInit();

    const phase = this.#detector.update(healthScore);
    const healthLevel = phaseToHealthLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      healthLevel,
      stressed: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  updateBatch(scores) {
    this.#ensureInit();
    const arr = scores instanceof Float64Array ? scores : new Float64Array(scores);
    const phase = this.#detector.update_batch(arr);
    const healthLevel = phaseToHealthLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      healthLevel,
      stressed: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  current() {
    this.#ensureInit();
    const phase = this.#detector.currentPhase();
    const healthLevel = phaseToHealthLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      healthLevel,
      stressed: phase === Phase.Critical || phase === Phase.Transitioning,
      declining: phase !== Phase.Stable,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      trend: this.#detector.inflectionMagnitude(),
      dataPoints: this.#detector.count(),
    };
  }

  reset() {
    this.#ensureInit();
    this.#detector.reset();
  }

  serialize() {
    this.#ensureInit();
    return this.#detector.serialize();
  }

  static async deserialize(json) {
    await initialize();
    const { NucleationDetector } = nucleationModule;
    const monitor = new TeamHealthMonitor();
    monitor.#detector = NucleationDetector.deserialize(json);
    monitor.#initialized = true;
    return monitor;
  }
}

/**
 * M&A integration risk monitor.
 * Tracks culture clash between merging organizations.
 */
export class IntegrationMonitor {
  #shepherd = null;
  #initialized = false;
  #entities = new Map();

  constructor(cultureDimensions = 8) {
    this.dimensions = cultureDimensions;
  }

  async init() {
    await initialize();
    const { Shepherd } = nucleationModule;
    this.#shepherd = new Shepherd(this.dimensions);
    this.#initialized = true;
  }

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('IntegrationMonitor not initialized. Call init() first.');
    }
  }

  /**
   * Register an organization/team/department.
   * @param {string} entityId
   * @param {Object} [metadata={}]
   * @param {Float64Array} [cultureProfile]
   */
  registerEntity(entityId, metadata = {}, cultureProfile = null) {
    this.#ensureInit();
    this.#shepherd.registerActor(entityId, cultureProfile);
    this.#entities.set(entityId, metadata);
  }

  /**
   * Update entity culture metrics.
   * @param {string} entityId
   * @param {Float64Array} cultureMetrics - Distribution across culture dimensions
   * @param {number} [timestamp=Date.now()]
   * @returns {Array} Integration alerts
   */
  updateEntity(entityId, cultureMetrics, timestamp = Date.now()) {
    this.#ensureInit();
    if (!this.#entities.has(entityId)) {
      this.registerEntity(entityId);
    }
    return this.#shepherd.updateActor(entityId, cultureMetrics, timestamp);
  }

  /**
   * Get culture clash risk between two entities.
   * @param {string} entityA
   * @param {string} entityB
   * @returns {number|undefined} Clash score (higher = more risk)
   */
  getClashRisk(entityA, entityB) {
    this.#ensureInit();
    return this.#shepherd.conflictPotential(entityA, entityB);
  }

  /**
   * Check all entity pairs for integration risk.
   * @param {number} [timestamp=Date.now()]
   * @returns {Array}
   */
  checkAllPairs(timestamp = Date.now()) {
    this.#ensureInit();
    return this.#shepherd.checkAllDyads(timestamp);
  }

  getEntities() {
    return Array.from(this.#entities.keys());
  }

  getEntityMetadata(entityId) {
    return this.#entities.get(entityId);
  }
}

/**
 * Quick team health assessment.
 * @param {number[]} healthScores
 * @param {OrgConfig} [config={}]
 */
export async function assessTeamHealth(healthScores, config = {}) {
  const monitor = new TeamHealthMonitor(config);
  await monitor.init();
  const state = monitor.updateBatch(healthScores);

  return {
    stressed: state.stressed,
    declining: state.declining,
    healthLevel: state.healthLevel,
    confidence: state.confidence,
  };
}

export default TeamHealthMonitor;
