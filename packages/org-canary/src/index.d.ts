export declare const HealthLevel: {
  readonly THRIVING: 'thriving';
  readonly STRAINED: 'strained';
  readonly STRESSED: 'stressed';
  readonly CRITICAL: 'critical';
};

export interface OrgConfig {
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  windowSize?: number;
  threshold?: number;
}

export interface OrgHealthState {
  healthLevel: 'thriving' | 'strained' | 'stressed' | 'critical';
  stressed: boolean;
  declining: boolean;
  confidence: number;
  variance: number;
  trend: number;
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

export declare class TeamHealthMonitor {
  constructor(config?: OrgConfig);
  init(): Promise<void>;
  update(healthScore: number): OrgHealthState;
  updateBatch(scores: number[] | Float64Array): OrgHealthState;
  current(): OrgHealthState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<TeamHealthMonitor>;
}

export declare class IntegrationMonitor {
  constructor(cultureDimensions?: number);
  init(): Promise<void>;
  registerEntity(entityId: string, metadata?: object, cultureProfile?: Float64Array | null): void;
  updateEntity(entityId: string, cultureMetrics: Float64Array, timestamp?: number): any[];
  getClashRisk(entityA: string, entityB: string): number | undefined;
  checkAllPairs(timestamp?: number): any[];
  getEntities(): string[];
  getEntityMetadata(entityId: string): object | undefined;
}

export declare function assessTeamHealth(
  healthScores: number[],
  config?: OrgConfig
): Promise<{
  stressed: boolean;
  declining: boolean;
  healthLevel: string;
  confidence: number;
}>;

export default TeamHealthMonitor;
