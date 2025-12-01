/**
 * threat-pulse
 *
 * Detect threat escalation before attacks materialize using behavioral
 * variance analysis. Built for SOC teams, SIEM integration, and proactive
 * threat hunting.
 *
 * The core insight: attacker reconnaissance creates a characteristic
 * "quieting" pattern before major actions - reduced variance in probing
 * behavior as they zero in on targets.
 *
 * @example
 * ```js
 * import { ThreatDetector } from 'threat-pulse';
 *
 * const detector = new ThreatDetector();
 * await detector.init();
 *
 * // Feed normalized event scores (0-1)
 * for (const event of securityEvents) {
 *   const state = detector.update(event.anomalyScore);
 *   if (state.escalating) {
 *     console.log(`ALERT: Threat escalation detected (${state.threatLevel})`);
 *   }
 * }
 * ```
 */

import { createRequire } from 'module';
import { readFile } from 'fs/promises';

let nucleationModule = null;
let initialized = false;

/**
 * Initialize the WASM module. Called automatically on first use.
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
 * Threat levels matching SOC severity conventions
 * @readonly
 * @enum {string}
 */
export const ThreatLevel = {
  /** Normal activity, no indicators */
  GREEN: 'green',
  /** Elevated activity, worth monitoring */
  YELLOW: 'yellow',
  /** High probability of imminent threat */
  ORANGE: 'orange',
  /** Active threat escalation in progress */
  RED: 'red',
};

/**
 * Map internal phase to threat level
 * @param {number} phase
 * @returns {string}
 */
function phaseToThreatLevel(phase) {
  const Phase = nucleationModule.Phase;
  switch (phase) {
    case Phase.Stable:
      return ThreatLevel.GREEN;
    case Phase.Approaching:
      return ThreatLevel.YELLOW;
    case Phase.Critical:
      return ThreatLevel.ORANGE;
    case Phase.Transitioning:
      return ThreatLevel.RED;
    default:
      return ThreatLevel.GREEN;
  }
}

/**
 * Configuration for ThreatDetector
 * @typedef {Object} ThreatConfig
 * @property {'conservative'|'balanced'|'aggressive'} [sensitivity='balanced'] - Detection sensitivity
 * @property {number} [windowSize=50] - Events to consider for baseline
 * @property {number} [threshold=2.0] - Standard deviations for alert
 */

/**
 * Current threat assessment
 * @typedef {Object} ThreatState
 * @property {string} threatLevel - Current threat level (green/yellow/orange/red)
 * @property {boolean} escalating - True if active escalation detected
 * @property {boolean} elevated - True if above normal baseline
 * @property {number} confidence - Confidence in assessment (0-1)
 * @property {number} variance - Current behavioral variance
 * @property {number} deviation - Deviation from baseline (z-score)
 * @property {number} eventCount - Total events processed
 */

/**
 * Threat escalation detector for security operations.
 *
 * Monitors behavioral variance in security event streams to identify
 * the characteristic "quieting" that precedes major attacks. Based on
 * the same phase transition dynamics that predict other complex system
 * failures.
 *
 * @example
 * ```js
 * // Monitor authentication anomaly scores
 * const detector = new ThreatDetector({ sensitivity: 'aggressive' });
 * await detector.init();
 *
 * authEvents.forEach(event => {
 *   const state = detector.update(event.riskScore);
 *   if (state.threatLevel === 'orange' || state.threatLevel === 'red') {
 *     triggerSOCAlert(state);
 *   }
 * });
 * ```
 */
export class ThreatDetector {
  #detector = null;
  #config = null;
  #initialized = false;

  /**
   * Create a new threat detector.
   * @param {ThreatConfig} [config={}] - Configuration options
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
      case 'aggressive':
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

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('ThreatDetector not initialized. Call init() first.');
    }
  }

  /**
   * Process a single security event.
   * @param {number} anomalyScore - Normalized anomaly/risk score (0-1 recommended)
   * @returns {ThreatState} Current threat assessment
   */
  update(anomalyScore) {
    this.#ensureInit();

    const phase = this.#detector.update(anomalyScore);
    const threatLevel = phaseToThreatLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      threatLevel,
      escalating: phase === Phase.Transitioning,
      elevated:
        phase === Phase.Approaching || phase === Phase.Critical || phase === Phase.Transitioning,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      deviation: this.#detector.inflectionMagnitude(),
      eventCount: this.#detector.count(),
    };
  }

  /**
   * Process a batch of security events.
   * @param {number[]|Float64Array} scores - Array of anomaly scores
   * @returns {ThreatState} Final threat assessment
   */
  updateBatch(scores) {
    this.#ensureInit();

    const arr = scores instanceof Float64Array ? scores : new Float64Array(scores);
    const phase = this.#detector.update_batch(arr);
    const threatLevel = phaseToThreatLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      threatLevel,
      escalating: phase === Phase.Transitioning,
      elevated:
        phase === Phase.Approaching || phase === Phase.Critical || phase === Phase.Transitioning,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      deviation: this.#detector.inflectionMagnitude(),
      eventCount: this.#detector.count(),
    };
  }

  /**
   * Get current threat state without new data.
   * @returns {ThreatState}
   */
  current() {
    this.#ensureInit();

    const phase = this.#detector.currentPhase();
    const threatLevel = phaseToThreatLevel(phase);
    const Phase = nucleationModule.Phase;

    return {
      threatLevel,
      escalating: phase === Phase.Transitioning,
      elevated:
        phase === Phase.Approaching || phase === Phase.Critical || phase === Phase.Transitioning,
      confidence: this.#detector.confidence(),
      variance: this.#detector.currentVariance(),
      deviation: this.#detector.inflectionMagnitude(),
      eventCount: this.#detector.count(),
    };
  }

  /**
   * Reset detector state (e.g., after incident resolution).
   */
  reset() {
    this.#ensureInit();
    this.#detector.reset();
  }

  /**
   * Serialize state for persistence.
   * @returns {string}
   */
  serialize() {
    this.#ensureInit();
    return this.#detector.serialize();
  }

  /**
   * Restore from serialized state.
   * @param {string} json
   * @returns {Promise<ThreatDetector>}
   */
  static async deserialize(json) {
    await initialize();
    const { NucleationDetector } = nucleationModule;
    const detector = new ThreatDetector();
    detector.#detector = NucleationDetector.deserialize(json);
    detector.#initialized = true;
    return detector;
  }
}

/**
 * Multi-source threat correlation detector.
 * Monitors multiple event streams and detects when sources
 * begin converging (potential coordinated attack).
 */
export class ThreatCorrelator {
  #shepherd = null;
  #initialized = false;
  #sources = new Set();

  /**
   * Create a correlator for the given number of behavior categories.
   * @param {number} [categories=10] - Behavior category count
   */
  constructor(categories = 10) {
    this.categories = categories;
  }

  /**
   * Initialize the correlator.
   */
  async init() {
    await initialize();
    const { Shepherd } = nucleationModule;
    this.#shepherd = new Shepherd(this.categories);
    this.#initialized = true;
  }

  #ensureInit() {
    if (!this.#initialized) {
      throw new Error('ThreatCorrelator not initialized. Call init() first.');
    }
  }

  /**
   * Register a new event source (host, user, service, etc.).
   * @param {string} sourceId - Unique identifier for the source
   * @param {Float64Array} [initialProfile] - Optional initial behavior distribution
   */
  registerSource(sourceId, initialProfile = null) {
    this.#ensureInit();
    this.#shepherd.registerActor(sourceId, initialProfile);
    this.#sources.add(sourceId);
  }

  /**
   * Update a source with new behavioral observation.
   * @param {string} sourceId - Source identifier
   * @param {Float64Array} observation - Behavior distribution
   * @param {number} [timestamp=Date.now()] - Event timestamp
   * @returns {Array} Any correlation alerts triggered
   */
  updateSource(sourceId, observation, timestamp = Date.now()) {
    this.#ensureInit();
    if (!this.#sources.has(sourceId)) {
      this.registerSource(sourceId);
    }
    return this.#shepherd.updateActor(sourceId, observation, timestamp);
  }

  /**
   * Get threat correlation between two sources.
   * Higher values = more divergent behavior = potential threat.
   * @param {string} sourceA
   * @param {string} sourceB
   * @returns {number|undefined} Correlation score
   */
  getCorrelation(sourceA, sourceB) {
    this.#ensureInit();
    return this.#shepherd.conflictPotential(sourceA, sourceB);
  }

  /**
   * Check all source pairs for correlation alerts.
   * @param {number} [timestamp=Date.now()]
   * @returns {Array} All triggered alerts
   */
  checkAllCorrelations(timestamp = Date.now()) {
    this.#ensureInit();
    return this.#shepherd.checkAllDyads(timestamp);
  }

  /**
   * List all registered sources.
   * @returns {string[]}
   */
  getSources() {
    return Array.from(this.#sources);
  }
}

/**
 * Quick threat assessment for a single event stream.
 * @param {number[]} anomalyScores - Array of anomaly scores
 * @param {ThreatConfig} [config={}]
 * @returns {Promise<{escalating: boolean, threatLevel: string, confidence: number}>}
 */
export async function assessThreat(anomalyScores, config = {}) {
  const detector = new ThreatDetector(config);
  await detector.init();

  const state = detector.updateBatch(anomalyScores);

  return {
    escalating: state.escalating,
    elevated: state.elevated,
    threatLevel: state.threatLevel,
    confidence: state.confidence,
  };
}

export default ThreatDetector;
