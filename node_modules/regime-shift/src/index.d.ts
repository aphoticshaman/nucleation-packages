/**
 * Market regime types
 */
export declare const Regime: {
  readonly STABLE: 'stable';
  readonly WARMING: 'warming';
  readonly CRITICAL: 'critical';
  readonly SHIFTING: 'shifting';
};

/**
 * Configuration options for RegimeDetector
 */
export interface RegimeConfig {
  /** Detection sensitivity: 'conservative', 'balanced', or 'sensitive' */
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  /** Rolling window size for variance calculation */
  windowSize?: number;
  /** Z-score threshold for regime detection */
  threshold?: number;
}

/**
 * Result from regime detection update
 */
export interface RegimeState {
  /** Current regime (stable, warming, critical, shifting) */
  regime: 'stable' | 'warming' | 'critical' | 'shifting';
  /** True if regime change detected */
  isShifting: boolean;
  /** True if approaching regime change */
  isWarning: boolean;
  /** Confidence in current assessment (0-1) */
  confidence: number;
  /** Current rolling variance */
  variance: number;
  /** Variance inflection magnitude (z-score) */
  inflection: number;
  /** Total observations processed */
  observations: number;
}

/**
 * Initialize the WASM module. Called automatically on first use.
 */
export declare function initialize(): Promise<void>;

/**
 * Market regime change detector.
 */
export declare class RegimeDetector {
  /**
   * Create a new regime detector.
   * @param config - Configuration options
   */
  constructor(config?: RegimeConfig);

  /**
   * Initialize the detector. Must be called before use.
   */
  init(): Promise<void>;

  /**
   * Process a single price/return observation.
   * @param value - Price or return value
   */
  update(value: number): RegimeState;

  /**
   * Process multiple observations at once.
   * @param values - Array of price/return values
   */
  updateBatch(values: number[] | Float64Array): RegimeState;

  /**
   * Get the current regime without adding new data.
   */
  current(): RegimeState;

  /**
   * Reset the detector state.
   */
  reset(): void;

  /**
   * Serialize detector state for persistence.
   */
  serialize(): string;

  /**
   * Create a detector from serialized state.
   * @param json - Serialized detector state
   */
  static deserialize(json: string): Promise<RegimeDetector>;
}

/**
 * Quick check if a price series shows regime shift signals.
 */
export declare function detectRegimeShift(
  prices: number[],
  config?: RegimeConfig
): Promise<{
  shifting: boolean;
  warning: boolean;
  regime: string;
  confidence: number;
}>;

export default RegimeDetector;
