/**
 * Core types for nucleation detectors
 */

/**
 * Detection sensitivity levels
 */
export type Sensitivity = 'conservative' | 'balanced' | 'sensitive';

/**
 * Phase states from the WASM module
 */
export type Phase = 'Stable' | 'Approaching' | 'Critical' | 'Transitioning';

/**
 * Phase constants for comparison
 */
export const PHASE = {
  STABLE: 'Stable',
  APPROACHING: 'Approaching',
  CRITICAL: 'Critical',
  TRANSITIONING: 'Transitioning',
} as const;

/**
 * Configuration for detectors
 */
export interface DetectorConfig {
  /** Detection sensitivity */
  sensitivity?: Sensitivity;
  /** Rolling window size for variance calculation */
  windowSize?: number;
  /** Z-score threshold for detection */
  threshold?: number;
}

/**
 * Base state returned by all detectors
 */
export interface DetectorState<TLevel extends string = string> {
  /** Domain-specific level (e.g., 'stable', 'warning', 'critical') */
  level: TLevel;
  /** Numeric level for comparison (0-3) */
  levelNumeric: number;
  /** Whether a phase transition is detected */
  transitioning: boolean;
  /** Whether state is elevated from baseline */
  elevated: boolean;
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
 * Default configuration values
 */
export const DEFAULT_CONFIG: Required<DetectorConfig> = {
  sensitivity: 'balanced',
  windowSize: 50,
  threshold: 2.0,
};

/**
 * Level mapping for converting phases to numeric values
 */
export const PHASE_TO_NUMERIC: Record<Phase, number> = {
  Stable: 0,
  Approaching: 1,
  Critical: 2,
  Transitioning: 3,
};
