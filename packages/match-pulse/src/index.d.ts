export declare const TiltLevel: {
  readonly FOCUSED: 'focused';
  readonly FRUSTRATED: 'frustrated';
  readonly TILTED: 'tilted';
  readonly TOXIC: 'toxic';
};

export interface MatchMonitorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface MatchState {
  level: 'focused' | 'frustrated' | 'tilted' | 'toxic';
  tilted: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class MatchMonitor {
  constructor(config?: MatchMonitorConfig);
  init(): Promise<void>;
  update(value: number): MatchState;
  updateBatch(values: number[] | Float64Array): MatchState;
  current(): MatchState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<MatchMonitor>;
}

export declare function assess(
  values: number[],
  config?: MatchMonitorConfig
): Promise<{
  tilted: boolean;
  elevated: boolean;
  level: string;
  confidence: number;
}>;

export default MatchMonitor;
