/**
 * Base detector class that all domain-specific detectors extend.
 *
 * This class provides the common functionality for all nucleation detectors,
 * eliminating code duplication across packages.
 */

import { initialize, getModule, type NucleationDetectorInstance } from './wasm-loader.js';
import {
  type DetectorConfig,
  type DetectorState,
  DEFAULT_CONFIG,
  PHASE_TO_NUMERIC,
} from './types.js';
import { validateNumber, validateConfig, NucleationError } from './validation.js';

/**
 * Level mapping configuration for domain-specific detectors
 */
export interface LevelMapping<TLevel extends string> {
  stable: TLevel;
  approaching: TLevel;
  critical: TLevel;
  transitioning: TLevel;
}

/**
 * Abstract base class for all nucleation detectors.
 *
 * @typeParam TLevel - The domain-specific level type (e.g., 'healthy' | 'at-risk')
 * @typeParam TState - The domain-specific state type extending DetectorState
 *
 * @example
 * ```typescript
 * class ChurnDetector extends BaseDetector<ChurnLevel, ChurnState> {
 *   protected readonly levelMapping = {
 *     stable: 'healthy',
 *     approaching: 'cooling',
 *     critical: 'at-risk',
 *     transitioning: 'churning',
 *   };
 *
 *   protected createState(baseState: DetectorState<ChurnLevel>): ChurnState {
 *     return {
 *       ...baseState,
 *       atRisk: baseState.levelNumeric >= 2,
 *     };
 *   }
 * }
 * ```
 */
export abstract class BaseDetector<
  TLevel extends string = string,
  TState extends DetectorState<TLevel> = DetectorState<TLevel>,
> {
  /** The underlying WASM detector instance */
  protected detector: NucleationDetectorInstance | null = null;

  /** Configuration for this detector */
  protected readonly config: Required<DetectorConfig>;

  /** Whether the detector has been initialized */
  private initialized = false;

  /**
   * Level mapping from internal phases to domain-specific levels.
   * Must be implemented by subclasses.
   */
  protected abstract readonly levelMapping: LevelMapping<TLevel>;

  /**
   * Create a new detector instance.
   *
   * @param config - Configuration options
   */
  constructor(config: DetectorConfig = {}) {
    validateConfig(config);
    this.config = {
      sensitivity: config.sensitivity ?? DEFAULT_CONFIG.sensitivity,
      windowSize: config.windowSize ?? this.getDefaultWindowSize(),
      threshold: config.threshold ?? DEFAULT_CONFIG.threshold,
    };
  }

  /**
   * Get the default window size for this detector type.
   * Can be overridden by subclasses for domain-specific defaults.
   */
  protected getDefaultWindowSize(): number {
    return DEFAULT_CONFIG.windowSize;
  }

  /**
   * Initialize the detector. Must be called before use.
   *
   * @returns Promise that resolves when initialization is complete
   * @throws NucleationError if initialization fails
   */
  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    await initialize();
    const module = getModule();
    const { DetectorConfig, NucleationDetector } = module;

    let detectorConfig: InstanceType<typeof DetectorConfig>;

    switch (this.config.sensitivity) {
      case 'conservative':
        detectorConfig = DetectorConfig.conservative();
        break;
      case 'sensitive':
        detectorConfig = DetectorConfig.sensitive();
        break;
      default:
        detectorConfig = new DetectorConfig();
    }

    detectorConfig.window_size = this.config.windowSize;
    detectorConfig.threshold = this.config.threshold;

    this.detector = new NucleationDetector(detectorConfig);
    this.initialized = true;
  }

  /**
   * Ensure the detector is initialized before use.
   *
   * @throws NucleationError if not initialized
   */
  protected ensureInit(): void {
    if (!this.initialized || !this.detector) {
      throw new NucleationError(
        `${this.constructor.name} not initialized. Call init() first.`,
        'NOT_INITIALIZED'
      );
    }
  }

  /**
   * Process a single observation.
   *
   * @param value - The value to process
   * @returns The current detector state
   */
  update(value: number): TState {
    this.ensureInit();
    validateNumber(value);

    const phase = this.detector!.update(value);
    return this.buildState(phase);
  }

  /**
   * Process multiple observations at once.
   *
   * @param values - Array of values to process
   * @returns The final detector state after all observations
   */
  updateBatch(values: number[] | Float64Array): TState {
    this.ensureInit();

    const arr = values instanceof Float64Array ? values : new Float64Array(values);

    // Validate all values
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        throw new NucleationError(`values[${i}] must be a finite number`, 'INVALID_VALUE');
      }
    }

    const phase = this.detector!.update_batch(arr);
    return this.buildState(phase);
  }

  /**
   * Get the current state without adding new data.
   *
   * @returns The current detector state
   */
  current(): TState {
    this.ensureInit();
    const phase = this.detector!.currentPhase();
    return this.buildState(phase);
  }

  /**
   * Reset the detector state.
   */
  reset(): void {
    this.ensureInit();
    this.detector!.reset();
  }

  /**
   * Serialize detector state for persistence.
   *
   * @returns JSON string of detector state
   */
  serialize(): string {
    this.ensureInit();
    return this.detector!.serialize();
  }

  /**
   * Build the state object from a phase value.
   */
  private buildState(phaseValue: number): TState {
    const module = getModule();
    const { Phase } = module;

    // Map numeric phase to string
    let phaseName: keyof typeof PHASE_TO_NUMERIC;
    switch (phaseValue) {
      case Phase.Stable:
        phaseName = 'Stable';
        break;
      case Phase.Approaching:
        phaseName = 'Approaching';
        break;
      case Phase.Critical:
        phaseName = 'Critical';
        break;
      case Phase.Transitioning:
        phaseName = 'Transitioning';
        break;
      default:
        phaseName = 'Stable';
    }

    const levelNumeric = PHASE_TO_NUMERIC[phaseName];
    const level = this.mapPhaseToLevel(phaseName);

    const baseState: DetectorState<TLevel> = {
      level,
      levelNumeric,
      transitioning: levelNumeric >= 3,
      elevated: levelNumeric >= 1,
      confidence: this.detector!.confidence(),
      variance: this.detector!.currentVariance(),
      inflection: this.detector!.inflectionMagnitude(),
      observations: this.detector!.count(),
    };

    return this.createState(baseState);
  }

  /**
   * Map a phase name to a domain-specific level.
   */
  private mapPhaseToLevel(phase: keyof typeof PHASE_TO_NUMERIC): TLevel {
    switch (phase) {
      case 'Stable':
        return this.levelMapping.stable;
      case 'Approaching':
        return this.levelMapping.approaching;
      case 'Critical':
        return this.levelMapping.critical;
      case 'Transitioning':
        return this.levelMapping.transitioning;
      default:
        return this.levelMapping.stable;
    }
  }

  /**
   * Create the domain-specific state object.
   * Override in subclasses to add domain-specific properties.
   *
   * @param baseState - The base detector state
   * @returns The domain-specific state
   */
  protected createState(baseState: DetectorState<TLevel>): TState {
    return baseState as TState;
  }
}
