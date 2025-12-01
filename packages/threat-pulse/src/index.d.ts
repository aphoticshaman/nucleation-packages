/**
 * Threat levels matching SOC severity conventions
 */
export declare const ThreatLevel: {
  readonly GREEN: 'green';
  readonly YELLOW: 'yellow';
  readonly ORANGE: 'orange';
  readonly RED: 'red';
};

/**
 * Configuration for ThreatDetector
 */
export interface ThreatConfig {
  /** Detection sensitivity: 'conservative', 'balanced', or 'aggressive' */
  sensitivity?: 'conservative' | 'balanced' | 'aggressive';
  /** Events to consider for baseline */
  windowSize?: number;
  /** Standard deviations for alert */
  threshold?: number;
}

/**
 * Current threat assessment
 */
export interface ThreatState {
  /** Current threat level (green/yellow/orange/red) */
  threatLevel: 'green' | 'yellow' | 'orange' | 'red';
  /** True if active escalation detected */
  escalating: boolean;
  /** True if above normal baseline */
  elevated: boolean;
  /** Confidence in assessment (0-1) */
  confidence: number;
  /** Current behavioral variance */
  variance: number;
  /** Deviation from baseline (z-score) */
  deviation: number;
  /** Total events processed */
  eventCount: number;
}

/**
 * Initialize the WASM module.
 */
export declare function initialize(): Promise<void>;

/**
 * Threat escalation detector for security operations.
 */
export declare class ThreatDetector {
  constructor(config?: ThreatConfig);
  init(): Promise<void>;
  update(anomalyScore: number): ThreatState;
  updateBatch(scores: number[] | Float64Array): ThreatState;
  current(): ThreatState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<ThreatDetector>;
}

/**
 * Multi-source threat correlation detector.
 */
export declare class ThreatCorrelator {
  constructor(categories?: number);
  init(): Promise<void>;
  registerSource(sourceId: string, initialProfile?: Float64Array | null): void;
  updateSource(sourceId: string, observation: Float64Array, timestamp?: number): any[];
  getCorrelation(sourceA: string, sourceB: string): number | undefined;
  checkAllCorrelations(timestamp?: number): any[];
  getSources(): string[];
}

/**
 * Quick threat assessment for a single event stream.
 */
export declare function assessThreat(
  anomalyScores: number[],
  config?: ThreatConfig
): Promise<{
  escalating: boolean;
  elevated: boolean;
  threatLevel: string;
  confidence: number;
}>;

export default ThreatDetector;
