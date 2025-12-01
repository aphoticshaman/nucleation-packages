/**
 * Engine Types
 * Core type definitions for the detection engine
 */

/**
 * Data provenance tracking for compliance and audit
 */
export interface DataProvenance {
  sourceId: string;
  sourceTier: 'official' | 'news' | 'financial' | 'social' | 'research';
  fetchedAt: string;
  attribution: string;
  license: string;
  transformations: string[];
  hash: string;
}

/**
 * Trace entry for audit trail
 */
export interface TraceEntry {
  id: string;
  timestamp: string;
  type: 'fetch' | 'filter' | 'normalize' | 'fuse' | 'detect' | 'batch' | 'transform';
  input: unknown;
  output: unknown;
  duration_ms: number;
  source?: string;
  metadata?: Record<string, unknown>;
  provenance?: DataProvenance;
}

/**
 * Complete data trace for export
 */
export interface DataTrace {
  sessionId: string;
  startTime: string;
  endTime: string;
  entries: TraceEntry[];
  summary: {
    totalEntries: number;
    byType: Record<string, number>;
    totalDuration_ms: number;
    sourcesUsed: string[];
    filtersApplied: string[];
  };
}

/**
 * External API data source configuration
 */
export interface ExternalAPIConfig {
  name: string;
  endpoint: string;
  headers?: Record<string, string>;
  rateLimit?: number;
  timeout?: number;
  transform?: (data: unknown) => number[];
  tier: DataProvenance['sourceTier'];
  attribution: string;
  license: string;
}

/**
 * Fusion result from multi-signal detection
 */
export interface FusionResult {
  phase: number;
  confidence: number;
  signals: Map<string, number[]>;
  fusedSignal: number[];
  timestamp: string;
  provenance: DataProvenance[];
}

/**
 * WASM bridge status
 */
export interface WasmStatus {
  available: boolean;
  version?: string;
  lastCheck: string;
  error?: string;
}

/**
 * Engine configuration
 */
export interface EngineConfig {
  enableTracing?: boolean;
  enableWasm?: boolean;
  defaultPollInterval?: string;
  respectRateLimits?: boolean;
  maxConcurrentRequests?: number;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  wasmCalls: number;
  jsCalls: number;
  avgWasmTime: number;
  avgJsTime: number;
  cacheHits: number;
  cacheMisses: number;
}
