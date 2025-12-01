export declare const RiskLevel: {
  readonly STABLE: 'stable';
  readonly ELEVATED: 'elevated';
  readonly CRITICAL: 'critical';
  readonly DISRUPTED: 'disrupted';
};

export interface SupplyMonitorConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface SupplyState {
  level: 'stable' | 'elevated' | 'critical' | 'disrupted';
  atRisk: boolean;
  elevated: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class SupplyMonitor {
  constructor(config?: SupplyMonitorConfig);
  init(): Promise<void>;
  update(value: number): SupplyState;
  updateBatch(values: number[] | Float64Array): SupplyState;
  current(): SupplyState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<SupplyMonitor>;
}

export declare function assess(
  values: number[],
  config?: SupplyMonitorConfig
): Promise<{
  atRisk: boolean;
  elevated: boolean;
  level: string;
  confidence: number;
}>;

export default SupplyMonitor;
