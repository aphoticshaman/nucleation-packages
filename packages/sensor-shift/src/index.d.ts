export declare const HealthLevel: {
  readonly NORMAL: 'normal';
  readonly DEGRADING: 'degrading';
  readonly WARNING: 'warning';
  readonly FAILING: 'failing';
};

export interface SensorMonitorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface SensorState {
  level: 'normal' | 'degrading' | 'warning' | 'failing';
  failing: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class SensorMonitor {
  constructor(config?: SensorMonitorConfig);
  init(): Promise<void>;
  update(value: number): SensorState;
  updateBatch(values: number[] | Float64Array): SensorState;
  current(): SensorState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<SensorMonitor>;
}

export declare function assess(
  values: number[],
  config?: SensorMonitorConfig
): Promise<{
  failing: boolean;
  elevated: boolean;
  level: string;
  confidence: number;
}>;

export default SensorMonitor;
