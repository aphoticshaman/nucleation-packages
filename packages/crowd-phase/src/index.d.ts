export declare const TensionLevel: {
  readonly CALM: 'calm';
  readonly TENSE: 'tense';
  readonly HEATED: 'heated';
  readonly VOLATILE: 'volatile';
};

export interface CrowdMonitorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface CrowdState {
  level: 'calm' | 'tense' | 'heated' | 'volatile';
  volatile: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class CrowdMonitor {
  constructor(config?: CrowdMonitorConfig);
  init(): Promise<void>;
  update(value: number): CrowdState;
  updateBatch(values: number[] | Float64Array): CrowdState;
  current(): CrowdState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<CrowdMonitor>;
}

export declare function assess(
  values: number[],
  config?: CrowdMonitorConfig
): Promise<{
  volatile: boolean;
  elevated: boolean;
  level: string;
  confidence: number;
}>;

export default CrowdMonitor;
