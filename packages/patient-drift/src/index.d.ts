export declare const AlertLevel: {
  readonly STABLE: 'stable';
  readonly WATCH: 'watch';
  readonly WARNING: 'warning';
  readonly CRITICAL: 'critical';
};

export interface PatientMonitorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface PatientState {
  level: 'stable' | 'watch' | 'warning' | 'critical';
  critical: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class PatientMonitor {
  constructor(config?: PatientMonitorConfig);
  init(): Promise<void>;
  update(value: number): PatientState;
  updateBatch(values: number[] | Float64Array): PatientState;
  current(): PatientState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<PatientMonitor>;
}

export declare function assess(
  values: number[],
  config?: PatientMonitorConfig
): Promise<{
  critical: boolean;
  elevated: boolean;
  level: string;
  confidence: number;
}>;

export default PatientMonitor;
