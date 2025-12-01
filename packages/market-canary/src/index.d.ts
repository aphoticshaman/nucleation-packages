export declare const PhaseLevel: {
  readonly STABLE: 'stable';
  readonly APPROACHING: 'approaching';
  readonly CRITICAL: 'critical';
  readonly TRANSITIONING: 'transitioning';
};

export interface TransitionDetectorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface TransitionState {
  phase: 'stable' | 'approaching' | 'critical' | 'transitioning';
  transitioning: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  inflection: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class TransitionDetector {
  constructor(config?: TransitionDetectorConfig);
  init(): Promise<void>;
  update(value: number): TransitionState;
  updateBatch(values: number[] | Float64Array): TransitionState;
  current(): TransitionState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<TransitionDetector>;
}

export declare function detectTransition(
  values: number[],
  config?: TransitionDetectorConfig
): Promise<{
  transitioning: boolean;
  elevated: boolean;
  phase: string;
  confidence: number;
}>;

export default TransitionDetector;
