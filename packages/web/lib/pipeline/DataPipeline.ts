/**
 * DUAL DATA PIPELINE SYSTEM
 *
 * Two distinct data flows for users:
 * 1. RAW PIPELINE - Untouched data directly from APIs
 * 2. PROCESSED PIPELINE - Our meta-analyzed, quantified output
 *
 * Users can choose which pipeline feeds their dashboard widgets,
 * enabling both "show me the raw numbers" and "show me your analysis"
 * modes simultaneously.
 */

import type { FinancialSource } from '../signals/financialSources';

// ============================================
// PIPELINE TYPES
// ============================================

export type PipelineType = 'raw' | 'processed';

export interface DataPoint {
  id: string;
  timestamp: Date;
  source: string;
  sourceId: string;
  pipeline: PipelineType;
  dataType: string;
  value: number | string | Record<string, unknown>;
  metadata: DataPointMetadata;
}

export interface DataPointMetadata {
  // Source tracking
  sourceUrl?: string;
  sourceName: string;
  sourceReliability: 'official' | 'reputable' | 'aggregated' | 'unofficial';
  fetchTimestamp: Date;

  // Freshness
  dataTimestamp: Date; // When the source says data is from
  ageSeconds: number;
  isStale: boolean;

  // For processed pipeline only
  processingDetails?: {
    model?: string;
    confidence?: number;
    methodology?: string;
    inputSources: string[];
    processingTimestamp: Date;
  };
}

// ============================================
// SOURCE TRACKING
// ============================================

export interface SourceContribution {
  sourceId: string;
  sourceName: string;
  dataPoints: number;
  lastFetch: Date;
  freshness: 'fresh' | 'recent' | 'stale' | 'error';
  reliability: 'official' | 'reputable' | 'aggregated' | 'unofficial';
}

export interface PipelineSummary {
  pipeline: PipelineType;
  topic: string;
  timeRange: {
    start: Date;
    end: Date;
    hoursSpan: number;
  };
  sources: SourceContribution[];
  totalDataPoints: number;
  uniqueSources: number;

  // User-facing summary
  summaryText: string; // "6 sources over past 72 hours"
}

// ============================================
// DASHBOARD DATA CONFIG
// ============================================

export interface DashboardDataConfig {
  id: string;
  name: string;
  description: string;
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;

  // Focus area
  focus: DashboardFocus;

  // Data sources enabled
  sources: DataSourceConfig[];

  // Refresh settings
  refresh: RefreshConfig;

  // Pipeline preferences per widget
  widgetPipelines: Record<string, PipelineType>;
}

export interface DashboardFocus {
  type: 'global' | 'regional' | 'country' | 'sector' | 'topic' | 'custom';

  // For regional/country focus
  regions?: string[];
  countries?: string[];

  // For sector focus
  sectors?: string[];
  industries?: string[];

  // For topic focus
  topics?: string[]; // e.g., ['quantum_computing', 'semiconductors']
  keywords?: string[];

  // For market focus
  symbols?: string[];
  assetClasses?: string[];
}

export interface DataSourceConfig {
  sourceId: string;
  enabled: boolean;
  priority: number;
  // Override default refresh rate
  refreshOverride?: number; // seconds

  // Data filtering
  dataTypes?: string[];
  includePatterns?: string[];
  excludePatterns?: string[];
}

export interface RefreshConfig {
  // Global refresh settings
  autoRefresh: boolean;
  intervalSeconds: number;

  // Per-source overrides
  sourceOverrides: Record<string, number>;

  // Staleness alerts
  alertOnStale: boolean;
  stalenessThresholdSeconds: number;
}

// ============================================
// PRE-BUILT FOCUS TEMPLATES
// ============================================

export const FOCUS_TEMPLATES: Record<string, DashboardFocus> = {
  // Market Focuses
  'us-equities': {
    type: 'sector',
    sectors: ['technology', 'healthcare', 'financials', 'industrials', 'consumer'],
    assetClasses: ['stocks', 'etfs'],
  },
  'us-fixed-income': {
    type: 'sector',
    sectors: ['bonds', 'treasury', 'corporate_debt'],
    assetClasses: ['bonds'],
  },
  'global-forex': {
    type: 'global',
    assetClasses: ['forex'],
    topics: ['currency', 'central_banks', 'monetary_policy'],
  },
  'crypto-markets': {
    type: 'global',
    assetClasses: ['crypto'],
    topics: ['defi', 'regulations', 'adoption'],
  },
  'commodities': {
    type: 'global',
    assetClasses: ['commodities'],
    topics: ['energy', 'metals', 'agriculture'],
  },

  // Sector Deep-Dives
  'us-tech-deep': {
    type: 'topic',
    topics: ['technology', 'ai', 'cloud', 'semiconductors', 'cybersecurity'],
    keywords: ['NVIDIA', 'Microsoft', 'Apple', 'Google', 'Amazon', 'Meta'],
    sectors: ['technology'],
  },
  'us-quantum': {
    type: 'topic',
    topics: ['quantum_computing', 'quantum_technology'],
    keywords: [
      'IonQ', 'Rigetti', 'D-Wave', 'IBM Quantum', 'Google Quantum',
      'qubit', 'quantum supremacy', 'quantum advantage',
    ],
    industries: ['quantum_computing', 'defense_tech', 'semiconductors'],
  },
  'us-defense': {
    type: 'topic',
    topics: ['defense', 'military', 'aerospace'],
    keywords: [
      'Lockheed Martin', 'Raytheon', 'Northrop Grumman', 'General Dynamics',
      'BAE Systems', 'L3Harris', 'Pentagon', 'DoD',
    ],
    sectors: ['industrials', 'aerospace_defense'],
  },
  'us-energy-transition': {
    type: 'topic',
    topics: ['renewable_energy', 'clean_tech', 'ev', 'batteries'],
    keywords: [
      'Tesla', 'solar', 'wind', 'lithium', 'hydrogen',
      'grid', 'storage', 'nuclear',
    ],
    sectors: ['energy', 'utilities'],
  },

  // Regional Focuses
  'china-watch': {
    type: 'country',
    countries: ['CHN'],
    topics: ['china_economy', 'us_china_relations', 'taiwan', 'supply_chain'],
    keywords: [
      'Xi Jinping', 'CCP', 'Belt and Road', 'yuan', 'Alibaba', 'Tencent',
    ],
  },
  'europe-macro': {
    type: 'regional',
    regions: ['europe'],
    topics: ['ecb', 'eurozone', 'eu_policy', 'energy_crisis'],
    countries: ['DEU', 'FRA', 'GBR', 'ITA', 'ESP'],
  },
  'emerging-markets': {
    type: 'regional',
    regions: ['emerging'],
    countries: ['BRA', 'IND', 'IDN', 'MEX', 'TUR', 'ZAF', 'SAU'],
    topics: ['emerging_markets', 'frontier_markets', 'development'],
  },

  // Geopolitical Focuses
  'conflict-risk': {
    type: 'topic',
    topics: ['conflict', 'war', 'military', 'sanctions'],
    keywords: [
      'Ukraine', 'Russia', 'Taiwan', 'Israel', 'Gaza', 'Iran',
      'North Korea', 'sanctions', 'NATO',
    ],
  },
  'supply-chain': {
    type: 'topic',
    topics: ['supply_chain', 'logistics', 'shipping', 'trade'],
    keywords: [
      'semiconductor shortage', 'port congestion', 'freight rates',
      'container', 'Suez', 'Panama Canal',
    ],
  },
};

// ============================================
// PIPELINE REGISTRY
// ============================================

export interface PipelineRegistry {
  sources: Map<string, SourceStatus>;
  lastUpdate: Date;
  overallHealth: 'healthy' | 'degraded' | 'critical';
}

export interface SourceStatus {
  sourceId: string;
  sourceName: string;
  isEnabled: boolean;
  lastFetch: Date | null;
  lastSuccess: Date | null;
  lastError: string | null;
  errorCount: number;
  successRate: number; // last 24h
  avgLatencyMs: number;
  dataPointsLast24h: number;
  freshness: 'fresh' | 'stale' | 'error' | 'unknown';
}

/**
 * Create initial pipeline registry
 */
export function createPipelineRegistry(
  sources: FinancialSource[]
): PipelineRegistry {
  const registry: PipelineRegistry = {
    sources: new Map(),
    lastUpdate: new Date(),
    overallHealth: 'healthy',
  };

  for (const source of sources) {
    registry.sources.set(source.id, {
      sourceId: source.id,
      sourceName: source.name,
      isEnabled: source.enabled,
      lastFetch: null,
      lastSuccess: null,
      lastError: null,
      errorCount: 0,
      successRate: 100,
      avgLatencyMs: 0,
      dataPointsLast24h: 0,
      freshness: 'unknown',
    });
  }

  return registry;
}

/**
 * Generate human-readable source summary
 */
export function generateSourceSummary(
  contributions: SourceContribution[],
  hoursSpan: number
): string {
  const uniqueSources = contributions.length;
  const totalPoints = contributions.reduce((sum, c) => sum + c.dataPoints, 0);

  const freshCount = contributions.filter(c => c.freshness === 'fresh').length;
  const officialCount = contributions.filter(
    c => c.reliability === 'official'
  ).length;

  let timeStr: string;
  if (hoursSpan < 1) {
    timeStr = `${Math.round(hoursSpan * 60)} minutes`;
  } else if (hoursSpan < 24) {
    timeStr = `${Math.round(hoursSpan)} hours`;
  } else {
    timeStr = `${Math.round(hoursSpan / 24)} days`;
  }

  let reliability = '';
  if (officialCount === uniqueSources) {
    reliability = ' (all official)';
  } else if (officialCount > 0) {
    reliability = ` (${officialCount} official)`;
  }

  let freshness = '';
  if (freshCount === uniqueSources) {
    freshness = ', all fresh';
  } else if (freshCount < uniqueSources / 2) {
    freshness = ', some stale';
  }

  return `${uniqueSources} sources over past ${timeStr}${reliability}${freshness} | ${totalPoints.toLocaleString()} data points`;
}

// ============================================
// DATA QUALITY INDICATORS
// ============================================

export interface DataQualityIndicator {
  overallScore: number; // 0-100
  factors: {
    freshness: number;
    coverage: number;
    reliability: number;
    consistency: number;
  };
  warnings: string[];
  recommendations: string[];
}

/**
 * Calculate data quality score for a dashboard
 */
export function calculateDataQuality(
  summary: PipelineSummary
): DataQualityIndicator {
  const factors = {
    freshness: 0,
    coverage: 0,
    reliability: 0,
    consistency: 0,
  };

  const warnings: string[] = [];
  const recommendations: string[] = [];

  // Freshness score
  const freshSources = summary.sources.filter(s => s.freshness === 'fresh');
  factors.freshness = (freshSources.length / summary.sources.length) * 100;

  if (factors.freshness < 50) {
    warnings.push('More than half of data sources are stale');
    recommendations.push('Check API connections and refresh intervals');
  }

  // Coverage score (based on unique sources)
  factors.coverage = Math.min(100, summary.uniqueSources * 20); // 5+ sources = 100%

  if (summary.uniqueSources < 3) {
    warnings.push('Limited source diversity');
    recommendations.push('Enable additional data sources for cross-validation');
  }

  // Reliability score
  const officialSources = summary.sources.filter(
    s => s.reliability === 'official' || s.reliability === 'reputable'
  );
  factors.reliability = (officialSources.length / summary.sources.length) * 100;

  if (factors.reliability < 50) {
    warnings.push('High reliance on unofficial sources');
    recommendations.push('Prioritize official and reputable data sources');
  }

  // Consistency score (based on data point distribution)
  const avgPoints = summary.totalDataPoints / summary.uniqueSources;
  const variance =
    summary.sources.reduce(
      (sum, s) => sum + Math.pow(s.dataPoints - avgPoints, 2),
      0
    ) / summary.sources.length;
  const cv = Math.sqrt(variance) / avgPoints; // coefficient of variation
  factors.consistency = Math.max(0, 100 - cv * 50);

  // Overall score (weighted average)
  const overallScore =
    factors.freshness * 0.3 +
    factors.coverage * 0.2 +
    factors.reliability * 0.3 +
    factors.consistency * 0.2;

  return {
    overallScore: Math.round(overallScore),
    factors,
    warnings,
    recommendations,
  };
}

// Types are exported inline at their declarations above
