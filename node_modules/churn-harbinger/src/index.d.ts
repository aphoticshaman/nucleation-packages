/**
 * Churn risk levels
 */
export declare const RiskLevel: {
  readonly HEALTHY: 'healthy';
  readonly COOLING: 'cooling';
  readonly AT_RISK: 'at-risk';
  readonly CHURNING: 'churning';
};

/**
 * Configuration for ChurnDetector
 */
export interface ChurnConfig {
  /** Detection sensitivity */
  sensitivity?: 'conservative' | 'balanced' | 'sensitive';
  /** Days/events for baseline (default 30) */
  windowSize?: number;
  /** Standard deviations for risk flag */
  threshold?: number;
}

/**
 * Current churn risk assessment
 */
export interface ChurnState {
  /** Current risk level */
  riskLevel: 'healthy' | 'cooling' | 'at-risk' | 'churning';
  /** True if high churn probability */
  atRisk: boolean;
  /** True if engagement trending down */
  declining: boolean;
  /** Confidence in assessment (0-1) */
  confidence: number;
  /** Current engagement variance */
  variance: number;
  /** Engagement trend indicator */
  trend: number;
  /** Total observations processed */
  dataPoints: number;
}

export declare function initialize(): Promise<void>;

/**
 * Customer churn detector for SaaS and subscription products.
 */
export declare class ChurnDetector {
  constructor(config?: ChurnConfig);
  init(): Promise<void>;
  update(engagementScore: number): ChurnState;
  updateBatch(scores: number[] | Float64Array): ChurnState;
  current(): ChurnState;
  reset(): void;
  serialize(): string;
  static deserialize(json: string): Promise<ChurnDetector>;
}

/**
 * Cohort-level churn monitoring.
 */
export declare class CohortMonitor {
  constructor(behaviorCategories?: number);
  init(): Promise<void>;
  addUser(userId: string, metadata?: object, initialBehavior?: Float64Array | null): void;
  updateUser(userId: string, behavior: Float64Array, timestamp?: number): any[];
  getDivergence(userA: string, userB: string): number | undefined;
  checkCohort(timestamp?: number): any[];
  getUsers(): string[];
  getUserMetadata(userId: string): object | undefined;
}

/**
 * Quick churn risk assessment.
 */
export declare function assessChurnRisk(
  engagementHistory: number[],
  config?: ChurnConfig
): Promise<{
  atRisk: boolean;
  declining: boolean;
  riskLevel: string;
  confidence: number;
}>;

export default ChurnDetector;
