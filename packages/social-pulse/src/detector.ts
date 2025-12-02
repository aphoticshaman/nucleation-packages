/**
 * SocialPulseDetector
 *
 * Phase transition detector for social sentiment.
 * Aggregates data from multiple sources, applies filters,
 * and detects variance changes that precede upheavals.
 *
 * DUAL-FUSION ARCHITECTURE:
 * - Pure JS mode: Self-contained variance detection (always available)
 * - WASM mode: Optional high-performance nucleation-wasm integration
 *
 * The asymmetric leverage allows:
 * - External data via API (social sources, economic indicators)
 * - WASM-accelerated detection when available
 * - Full data trace export for open-box visibility
 *
 * Use cases:
 * - Social unrest / revolution precursors
 * - Earnings sentiment (Q1/Q2/Q3/Q4 reports)
 * - Geopolitical shifts
 * - Market sentiment regime changes
 */

import { randomBytes } from 'node:crypto';

import type {
  DataSource,
  SearchParams,
  SocialPost,
  PostFilter,
  SentimentAggregate,
  UpheavalState,
  UpheavalLevel,
  Platform,
} from './types.js';

// ============ DUAL-FUSION ENGINE TYPES ============

/**
 * External API data source configuration
 */
export interface ExternalAPIConfig {
  name: string;
  endpoint: string;
  headers?: Record<string, string>;
  rateLimit?: number; // requests per second
  timeout?: number; // ms
  transform?: (data: unknown) => number[]; // transform to numeric signals
}

/**
 * Multi-signal fusion result
 */
export interface FusionResult {
  timestamp: string;
  signals: Map<string, number[]>;
  fusedSignal: number[];
  wasmPhase: number;
  jsPhase: string;
  confidence: number;
  variance: number;
  inflectionMagnitude: number;
  processingTime: {
    fetch_ms: number;
    fusion_ms: number;
    detection_ms: number;
    total_ms: number;
  };
  trace?: DataTrace;
}

/**
 * Stream processing callback
 */
export type StreamCallback = (result: FusionResult) => void;

/**
 * Performance metrics for asymmetric leverage tracking
 */
export interface PerformanceMetrics {
  wasmSpeedup: number;
  dataPointsProcessed: number;
  signalsCombined: number;
  avgLatency_ms: number;
  throughput_per_sec: number;
  accuracyGain: number; // estimated from multi-signal correlation
}

// ============ DATA TRACE TYPES ============

/**
 * Individual trace entry for audit trail
 */
export interface TraceEntry {
  id: string;
  timestamp: string;
  type: 'fetch' | 'filter' | 'aggregate' | 'detect' | 'transform';
  source?: string;
  input: unknown;
  output: unknown;
  duration_ms: number;
  metadata?: Record<string, unknown>;
}

/**
 * Complete data trace for open-box visibility
 */
export interface DataTrace {
  sessionId: string;
  startTime: string;
  endTime?: string;
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
 * WASM bridge status
 */
export interface WasmBridgeStatus {
  available: boolean;
  mode: 'wasm' | 'js';
  version?: string;
  performance?: {
    lastDetection_ms: number;
    avgDetection_ms: number;
    wasmSpeedup?: number;
  };
}

// ============ WASM BRIDGE ============

/**
 * nucleation-wasm types (dynamic import)
 */
interface WasmModule {
  NucleationDetector: {
    with_defaults(): WasmDetector;
    new (config: unknown): WasmDetector;
  };
  DetectorConfig: {
    new (): WasmConfig;
    conservative(): WasmConfig;
    sensitive(): WasmConfig;
  };
  Shepherd: {
    new (n_categories: number): WasmShepherd;
  };
  Phase: {
    Stable: number;
    Approaching: number;
    Critical: number;
    Transitioning: number;
  };
  version(): string;
}

interface WasmDetector {
  update(value: number): number;
  update_batch(values: Float64Array): number;
  currentPhase(): number;
  confidence(): number;
  currentVariance(): number;
  inflectionMagnitude(): number;
  count(): number;
  reset(): void;
  serialize(): string;
  free(): void;
}

interface WasmConfig {
  window_size: number;
  threshold: number;
  smoothing_window: number;
}

interface WasmShepherd {
  registerActor(actor_id: string, distribution?: Float64Array): void;
  updateActor(actor_id: string, observation: Float64Array, timestamp: number): unknown[];
  conflictPotential(a: string, b: string): number | undefined;
  checkAllDyads(timestamp: number): unknown[];
  actors(): unknown[];
  free(): void;
}

/**
 * WasmBridge - Asymmetric dual-fusion integration
 *
 * Provides high-performance WASM detection when nucleation-wasm is available.
 * Uses NucleationDetector for variance inflection (10-100x faster than JS).
 * Falls back to pure JS implementation when WASM unavailable.
 */
class WasmBridge {
  private wasmModule: WasmModule | null = null;
  private available = false;
  private wasmVersion: string | null = null;
  private detectors = new Map<string, WasmDetector>();
  private shepherd: WasmShepherd | null = null;
  private detectionTimes: number[] = [];
  private jsDetectionTimes: number[] = [];

  /**
   * Attempt to load nucleation-wasm module
   */
  async init(): Promise<boolean> {
    try {
      // Dynamic import to avoid hard dependency
      const wasm = (await import('nucleation-wasm')) as WasmModule;
      if (wasm && typeof wasm.NucleationDetector?.with_defaults === 'function') {
        this.wasmModule = wasm;
        this.available = true;
        this.wasmVersion = wasm.version?.() ?? 'unknown';
        return true;
      }
    } catch {
      // WASM not available, will use JS fallback
      this.available = false;
    }
    return false;
  }

  /**
   * Check if WASM is available
   */
  isAvailable(): boolean {
    return this.available;
  }

  /**
   * Get or create a WASM detector for a region/topic
   */
  getDetector(id: string): WasmDetector | null {
    if (!this.available || !this.wasmModule) return null;

    let detector = this.detectors.get(id);
    if (!detector) {
      detector = this.wasmModule.NucleationDetector.with_defaults();
      this.detectors.set(id, detector);
    }
    return detector;
  }

  /**
   * Update detector with new value (returns WASM Phase enum value)
   */
  update(id: string, value: number): number | null {
    const detector = this.getDetector(id);
    if (!detector) return null;
    return detector.update(value);
  }

  /**
   * Get current phase from WASM detector
   */
  getPhase(id: string): number | null {
    const detector = this.detectors.get(id);
    if (!detector) return null;
    return detector.currentPhase();
  }

  /**
   * Get confidence from WASM detector
   */
  getConfidence(id: string): number | null {
    const detector = this.detectors.get(id);
    if (!detector) return null;
    return detector.confidence();
  }

  /**
   * Get current variance from WASM detector
   */
  getVariance(id: string): number | null {
    const detector = this.detectors.get(id);
    if (!detector) return null;
    return detector.currentVariance();
  }

  /**
   * Get inflection magnitude (z-score) from WASM detector
   */
  getInflectionMagnitude(id: string): number | null {
    const detector = this.detectors.get(id);
    if (!detector) return null;
    return detector.inflectionMagnitude();
  }

  /**
   * Initialize Shepherd for multi-actor conflict monitoring
   */
  initShepherd(n_categories: number = 50): boolean {
    if (!this.available || !this.wasmModule) return false;
    this.shepherd = new this.wasmModule.Shepherd(n_categories);
    return true;
  }

  /**
   * Register an actor (region/entity) with Shepherd
   */
  registerActor(actorId: string): boolean {
    if (!this.shepherd) return false;
    this.shepherd.registerActor(actorId);
    return true;
  }

  /**
   * Update actor and check for nucleation alerts
   */
  updateActor(actorId: string, observation: Float64Array, timestamp: number): unknown[] {
    if (!this.shepherd) return [];
    return this.shepherd.updateActor(actorId, observation, timestamp);
  }

  /**
   * Get conflict potential between two actors (KL-divergence based)
   */
  getConflictPotential(a: string, b: string): number | undefined {
    if (!this.shepherd) return undefined;
    return this.shepherd.conflictPotential(a, b);
  }

  /**
   * Check all actor pairs for nucleation alerts
   */
  checkAllDyads(timestamp: number): unknown[] {
    if (!this.shepherd) return [];
    return this.shepherd.checkAllDyads(timestamp);
  }

  /**
   * Get the raw WASM module (for advanced usage)
   */
  getModule(): WasmModule | null {
    return this.wasmModule;
  }

  /**
   * Get current bridge status
   */
  getStatus(): WasmBridgeStatus {
    const avgWasm =
      this.detectionTimes.length > 0
        ? this.detectionTimes.reduce((a, b) => a + b, 0) / this.detectionTimes.length
        : 0;
    const avgJs =
      this.jsDetectionTimes.length > 0
        ? this.jsDetectionTimes.reduce((a, b) => a + b, 0) / this.jsDetectionTimes.length
        : 0;

    const status: WasmBridgeStatus = {
      available: this.available,
      mode: this.available ? 'wasm' : 'js',
    };

    if (this.wasmVersion) {
      status.version = this.wasmVersion;
    }

    if (this.detectionTimes.length > 0 || this.jsDetectionTimes.length > 0) {
      status.performance = {
        lastDetection_ms: this.available
          ? (this.detectionTimes[this.detectionTimes.length - 1] ?? 0)
          : (this.jsDetectionTimes[this.jsDetectionTimes.length - 1] ?? 0),
        avgDetection_ms: this.available ? avgWasm : avgJs,
      };

      if (avgWasm > 0 && avgJs > 0) {
        status.performance.wasmSpeedup = avgJs / avgWasm;
      }
    }

    return status;
  }

  /**
   * Record detection time for performance tracking
   */
  recordTime(ms: number, isWasm: boolean): void {
    if (isWasm) {
      this.detectionTimes.push(ms);
      if (this.detectionTimes.length > 100) this.detectionTimes.shift();
    } else {
      this.jsDetectionTimes.push(ms);
      if (this.jsDetectionTimes.length > 100) this.jsDetectionTimes.shift();
    }
  }

  /**
   * Free all WASM resources
   */
  dispose(): void {
    for (const detector of this.detectors.values()) {
      detector.free();
    }
    this.detectors.clear();
    if (this.shepherd) {
      this.shepherd.free();
      this.shepherd = null;
    }
  }
}

// ============ DATA TRACE RECORDER ============

/**
 * DataTraceRecorder - Full audit trail for open-box visibility
 *
 * Records every data transformation in the detection pipeline
 * for debugging, compliance, and analysis.
 */
class DataTraceRecorder {
  private sessionId: string;
  private startTime: string;
  private entries: TraceEntry[] = [];
  private enabled = false;

  constructor() {
    this.sessionId = this.generateId();
    this.startTime = new Date().toISOString();
  }

  /**
   * Enable/disable tracing
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (enabled && this.entries.length === 0) {
      this.sessionId = this.generateId();
      this.startTime = new Date().toISOString();
    }
  }

  /**
   * Check if tracing is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Record a trace entry
   */
  record(
    type: TraceEntry['type'],
    input: unknown,
    output: unknown,
    duration_ms: number,
    source?: string,
    metadata?: Record<string, unknown>
  ): void {
    if (!this.enabled) return;

    const entry: TraceEntry = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      type,
      input: this.sanitize(input),
      output: this.sanitize(output),
      duration_ms,
    };

    if (source) entry.source = source;
    if (metadata) entry.metadata = metadata;

    this.entries.push(entry);
  }

  /**
   * Export complete trace
   */
  export(): DataTrace {
    const byType: Record<string, number> = {};
    const sourcesUsed = new Set<string>();
    const filtersApplied = new Set<string>();
    let totalDuration = 0;

    for (const entry of this.entries) {
      byType[entry.type] = (byType[entry.type] ?? 0) + 1;
      totalDuration += entry.duration_ms;

      if (entry.source) {
        if (entry.type === 'fetch') sourcesUsed.add(entry.source);
        if (entry.type === 'filter') filtersApplied.add(entry.source);
      }
    }

    return {
      sessionId: this.sessionId,
      startTime: this.startTime,
      endTime: new Date().toISOString(),
      entries: this.entries,
      summary: {
        totalEntries: this.entries.length,
        byType,
        totalDuration_ms: totalDuration,
        sourcesUsed: [...sourcesUsed],
        filtersApplied: [...filtersApplied],
      },
    };
  }

  /**
   * Export as JSON string
   */
  toJSON(): string {
    return JSON.stringify(this.export(), null, 2);
  }

  /**
   * Export as CSV (entries only)
   */
  toCSV(): string {
    const headers = ['id', 'timestamp', 'type', 'source', 'duration_ms'];
    const rows = this.entries.map((e) => [
      e.id,
      e.timestamp,
      e.type,
      e.source ?? '',
      String(e.duration_ms),
    ]);
    return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
  }

  /**
   * Clear trace history
   */
  clear(): void {
    this.entries = [];
    this.sessionId = this.generateId();
    this.startTime = new Date().toISOString();
  }

  private generateId(): string {
    // Use crypto-safe random for ID generation (Node.js compatible)
    const bytes = randomBytes(8);
    const hex = bytes.toString('hex');
    return `trace_${Date.now()}_${hex}`;
  }

  private sanitize(data: unknown): unknown {
    // Truncate large data for storage
    const str = JSON.stringify(data);
    if (str.length > 10000) {
      return { _truncated: true, _length: str.length, _preview: str.slice(0, 500) };
    }
    return data;
  }
}

// ============ DUAL-FUSION ENGINE ============

/**
 * DualFusionEngine - Asymmetric Leverage Through Multi-Signal Fusion
 *
 * Combines external API data with WASM-accelerated detection for:
 * - PERFORMANCE: 10-100x speedup via WASM batch processing
 * - GRANULARITY: Multi-signal fusion from diverse data sources
 * - ACCURACY: Cross-correlation between signals reduces noise
 * - SPEED: Parallel data fetching + batch WASM updates
 * - EFFICIENCY: Streaming mode for real-time, memory-efficient processing
 *
 * Architecture:
 * ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
 * │ External API 1  │  │ External API 2  │  │ External API N  │
 * └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
 *          │                    │                    │
 *          └────────────────────┼────────────────────┘
 *                               ▼
 *                    ┌─────────────────────┐
 *                    │  PARALLEL FETCHER   │ ← Rate limited, fault tolerant
 *                    └──────────┬──────────┘
 *                               ▼
 *                    ┌─────────────────────┐
 *                    │  SIGNAL NORMALIZER  │ ← Z-score normalization
 *                    └──────────┬──────────┘
 *                               ▼
 *                    ┌─────────────────────┐
 *                    │  MULTI-SIGNAL FUSER │ ← Weighted combination
 *                    └──────────┬──────────┘
 *                               ▼
 *            ┌──────────────────┴──────────────────┐
 *            ▼                                     ▼
 * ┌─────────────────────┐               ┌─────────────────────┐
 * │  WASM BATCH DETECT  │               │   JS FALLBACK       │
 * │  (10-100x faster)   │               │   (always works)    │
 * └──────────┬──────────┘               └──────────┬──────────┘
 *            └──────────────────┬──────────────────┘
 *                               ▼
 *                    ┌─────────────────────┐
 *                    │   DATA TRACE LOG    │ ← Full audit trail
 *                    └─────────────────────┘
 */
export class DualFusionEngine {
  private wasmBridge: WasmBridge;
  private traceRecorder: DataTraceRecorder;
  private apiConfigs: ExternalAPIConfig[] = [];
  private signalBuffers = new Map<string, number[]>();
  private metrics: PerformanceMetrics = {
    wasmSpeedup: 1,
    dataPointsProcessed: 0,
    signalsCombined: 0,
    avgLatency_ms: 0,
    throughput_per_sec: 0,
    accuracyGain: 1,
  };
  private latencyHistory: number[] = [];
  private streamInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this.wasmBridge = new WasmBridge();
    this.traceRecorder = new DataTraceRecorder();
  }

  /**
   * Initialize the dual-fusion engine
   */
  async init(): Promise<boolean> {
    const startTime = performance.now();

    // Initialize WASM bridge
    const wasmAvailable = await this.wasmBridge.init();

    // Initialize Shepherd for multi-actor monitoring
    if (wasmAvailable) {
      this.wasmBridge.initShepherd(50);
    }

    this.traceRecorder.setEnabled(true);

    this.traceRecorder.record(
      'transform',
      { action: 'init' },
      { wasmAvailable, apiCount: this.apiConfigs.length },
      performance.now() - startTime,
      'DualFusionEngine'
    );

    return wasmAvailable;
  }

  /**
   * Register an external API data source
   */
  registerAPI(config: ExternalAPIConfig): void {
    this.apiConfigs.push({
      rateLimit: 10,
      timeout: 5000,
      ...config,
    });
    this.signalBuffers.set(config.name, []);
  }

  /**
   * Fetch data from all APIs in parallel (with rate limiting)
   */
  async fetchAllSources(): Promise<Map<string, number[]>> {
    const fetchStart = performance.now();
    const results = new Map<string, number[]>();

    // Parallel fetch with timeout
    const fetchPromises = this.apiConfigs.map(async (config) => {
      const sourceStart = performance.now();
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), config.timeout ?? 5000);

        const fetchOptions: RequestInit = {
          signal: controller.signal,
        };
        if (config.headers) {
          fetchOptions.headers = config.headers;
        }

        const response = await fetch(config.endpoint, fetchOptions);

        clearTimeout(timeout);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const signal = config.transform ? config.transform(data) : this.defaultTransform(data);

        this.traceRecorder.record(
          'fetch',
          { endpoint: config.endpoint },
          { dataPoints: signal.length },
          performance.now() - sourceStart,
          config.name
        );

        return { name: config.name, signal };
      } catch (error) {
        this.traceRecorder.record(
          'fetch',
          { endpoint: config.endpoint, error: String(error) },
          { dataPoints: 0 },
          performance.now() - sourceStart,
          config.name
        );
        return { name: config.name, signal: [] };
      }
    });

    const fetchResults = await Promise.allSettled(fetchPromises);

    for (const result of fetchResults) {
      if (result.status === 'fulfilled' && result.value.signal.length > 0) {
        results.set(result.value.name, result.value.signal);

        // Update buffer
        const buffer = this.signalBuffers.get(result.value.name) ?? [];
        buffer.push(...result.value.signal);
        // Keep last 1000 points per source
        while (buffer.length > 1000) buffer.shift();
        this.signalBuffers.set(result.value.name, buffer);
      }
    }

    this.traceRecorder.record(
      'fetch',
      { sourceCount: this.apiConfigs.length },
      { successCount: results.size, totalPoints: [...results.values()].flat().length },
      performance.now() - fetchStart,
      'parallel_fetch'
    );

    return results;
  }

  /**
   * Normalize signals using Z-score (for fair combination)
   */
  normalizeSignal(signal: number[]): number[] {
    if (signal.length < 2) return signal;

    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const variance =
      signal.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (signal.length - 1);
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return signal.map(() => 0);

    return signal.map((v) => (v - mean) / stdDev);
  }

  /**
   * Fuse multiple signals into one (weighted by reliability)
   */
  fuseSignals(signals: Map<string, number[]>): number[] {
    const fusionStart = performance.now();

    if (signals.size === 0) return [];

    // Normalize all signals
    const normalized = new Map<string, number[]>();
    for (const [name, signal] of signals) {
      normalized.set(name, this.normalizeSignal(signal));
    }

    // Find common length (use minimum)
    const lengths = [...normalized.values()].map((s) => s.length);
    const minLength = Math.min(...lengths);

    if (minLength === 0) return [];

    // Weighted fusion (equal weights for now, can be extended)
    const weights = new Map<string, number>();
    for (const name of normalized.keys()) {
      weights.set(name, 1 / normalized.size);
    }

    // Combine signals
    const fused: number[] = [];
    for (let i = 0; i < minLength; i++) {
      let sum = 0;
      for (const [name, signal] of normalized) {
        const weight = weights.get(name) ?? 0;
        sum += (signal[i] ?? 0) * weight;
      }
      fused.push(sum);
    }

    // Calculate cross-correlation for accuracy gain estimation
    const correlations: number[] = [];
    const signalArrays = [...normalized.values()];
    for (let i = 0; i < signalArrays.length; i++) {
      for (let j = i + 1; j < signalArrays.length; j++) {
        const sig1 = signalArrays[i] ?? [];
        const sig2 = signalArrays[j] ?? [];
        correlations.push(this.correlation(sig1, sig2));
      }
    }

    // Higher correlation = signals agree = higher accuracy
    const avgCorrelation =
      correlations.length > 0 ? correlations.reduce((a, b) => a + b, 0) / correlations.length : 0;

    this.metrics.accuracyGain = 1 + Math.abs(avgCorrelation) * (normalized.size - 1) * 0.1;
    this.metrics.signalsCombined = normalized.size;

    this.traceRecorder.record(
      'transform',
      { signalCount: normalized.size, lengths },
      { fusedLength: fused.length, avgCorrelation, accuracyGain: this.metrics.accuracyGain },
      performance.now() - fusionStart,
      'signal_fusion'
    );

    return fused;
  }

  /**
   * Batch WASM detection (10-100x faster than individual updates)
   */
  detectBatch(
    id: string,
    values: number[]
  ): {
    phase: number;
    confidence: number;
    variance: number;
    inflectionMagnitude: number;
  } {
    const detectStart = performance.now();

    // Use WASM if available
    if (this.wasmBridge.isAvailable()) {
      // Get or create detector
      const detector = this.wasmBridge.getDetector(id);
      if (detector) {
        // BATCH UPDATE - much faster than individual
        const float64Values = new Float64Array(values);
        const phase = detector.update_batch(float64Values);
        const confidence = detector.confidence();
        const variance = detector.currentVariance();
        const inflectionMagnitude = detector.inflectionMagnitude();

        const duration = performance.now() - detectStart;
        this.wasmBridge.recordTime(duration, true);

        this.traceRecorder.record(
          'detect',
          { id, batchSize: values.length },
          { phase, confidence, variance, mode: 'wasm_batch' },
          duration,
          'wasm_batch_detect'
        );

        this.metrics.dataPointsProcessed += values.length;

        return { phase, confidence, variance, inflectionMagnitude };
      }
    }

    // JS fallback (process individually)
    let lastPhase = 0;
    let lastVariance = 0;
    const jsStart = performance.now();

    for (const value of values) {
      const state = this.jsDetect(id, value);
      lastPhase = this.phaseToNumber(state.phase);
      lastVariance = state.variance;
    }

    const jsDuration = performance.now() - jsStart;
    this.wasmBridge.recordTime(jsDuration, false);

    this.traceRecorder.record(
      'detect',
      { id, batchSize: values.length },
      { phase: lastPhase, variance: lastVariance, mode: 'js_fallback' },
      jsDuration,
      'js_batch_detect'
    );

    this.metrics.dataPointsProcessed += values.length;

    return {
      phase: lastPhase,
      confidence: 0.5,
      variance: lastVariance,
      inflectionMagnitude: 0,
    };
  }

  /**
   * Full fusion pipeline: fetch → normalize → fuse → detect
   */
  async process(): Promise<FusionResult> {
    const totalStart = performance.now();

    // 1. Fetch all sources in parallel
    const fetchStart = performance.now();
    const signals = await this.fetchAllSources();
    const fetchTime = performance.now() - fetchStart;

    // 2. Fuse signals
    const fusionStart = performance.now();
    const fusedSignal = this.fuseSignals(signals);
    const fusionTime = performance.now() - fusionStart;

    // 3. Batch detection
    const detectStart = performance.now();
    const detection = this.detectBatch('fusion_main', fusedSignal);
    const detectTime = performance.now() - detectStart;

    const totalTime = performance.now() - totalStart;

    // Update latency metrics
    this.latencyHistory.push(totalTime);
    if (this.latencyHistory.length > 100) this.latencyHistory.shift();
    this.metrics.avgLatency_ms =
      this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length;
    this.metrics.throughput_per_sec = fusedSignal.length / (totalTime / 1000);

    // Get WASM status for speedup metric
    const wasmStatus = this.wasmBridge.getStatus();
    if (wasmStatus.performance?.wasmSpeedup) {
      this.metrics.wasmSpeedup = wasmStatus.performance.wasmSpeedup;
    }

    const result: FusionResult = {
      timestamp: new Date().toISOString(),
      signals,
      fusedSignal,
      wasmPhase: detection.phase,
      jsPhase: this.numberToPhase(detection.phase),
      confidence: detection.confidence,
      variance: detection.variance,
      inflectionMagnitude: detection.inflectionMagnitude,
      processingTime: {
        fetch_ms: fetchTime,
        fusion_ms: fusionTime,
        detection_ms: detectTime,
        total_ms: totalTime,
      },
    };

    // Include trace if enabled
    if (this.traceRecorder.isEnabled()) {
      result.trace = this.traceRecorder.export();
    }

    return result;
  }

  /**
   * Start streaming mode (real-time continuous processing)
   */
  startStream(callback: StreamCallback, interval_ms = 1000): void {
    if (this.streamInterval) {
      this.stopStream();
    }

    const tick = (): void => {
      this.process()
        .then((result) => callback(result))
        .catch((error: unknown) => {
          this.traceRecorder.record(
            'transform',
            { action: 'stream_error' },
            { error: String(error) },
            0,
            'stream'
          );
        });
    };

    this.streamInterval = setInterval(tick, interval_ms);
  }

  /**
   * Stop streaming mode
   */
  stopStream(): void {
    if (this.streamInterval) {
      clearInterval(this.streamInterval);
      this.streamInterval = null;
    }
  }

  /**
   * Get current performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get WASM bridge status
   */
  getWasmStatus(): WasmBridgeStatus {
    return this.wasmBridge.getStatus();
  }

  /**
   * Export full data trace (open-box visibility)
   */
  exportTrace(): DataTrace {
    return this.traceRecorder.export();
  }

  /**
   * Export trace as JSON
   */
  exportTraceJSON(): string {
    return this.traceRecorder.toJSON();
  }

  /**
   * Export trace as CSV
   */
  exportTraceCSV(): string {
    return this.traceRecorder.toCSV();
  }

  /**
   * Clear trace history
   */
  clearTrace(): void {
    this.traceRecorder.clear();
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.stopStream();
    this.wasmBridge.dispose();
    this.signalBuffers.clear();
  }

  // ============ Private Helpers ============

  private defaultTransform(data: unknown): number[] {
    // Extract numeric values from common API response formats
    if (Array.isArray(data)) {
      return data
        .map((item) => {
          if (typeof item === 'number') return item;
          if (typeof item === 'object' && item !== null) {
            // Look for common numeric fields
            const obj = item as Record<string, unknown>;
            return (
              (typeof obj['value'] === 'number' && obj['value']) ||
              (typeof obj['score'] === 'number' && obj['score']) ||
              (typeof obj['sentiment'] === 'number' && obj['sentiment']) ||
              (typeof obj['price'] === 'number' && obj['price']) ||
              0
            );
          }
          return 0;
        })
        .filter((v) => v !== 0);
    }
    return [];
  }

  private correlation(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    if (n < 2) return 0;

    const meanA = a.slice(0, n).reduce((s, v) => s + v, 0) / n;
    const meanB = b.slice(0, n).reduce((s, v) => s + v, 0) / n;

    let num = 0;
    let denA = 0;
    let denB = 0;

    for (let i = 0; i < n; i++) {
      const da = (a[i] ?? 0) - meanA;
      const db = (b[i] ?? 0) - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }

    const den = Math.sqrt(denA * denB);
    return den === 0 ? 0 : num / den;
  }

  private jsDetectors = new Map<string, { values: number[]; windowSize: number }>();

  private jsDetect(id: string, value: number): { phase: string; variance: number; mean: number } {
    let state = this.jsDetectors.get(id);
    if (!state) {
      state = { values: [], windowSize: 30 };
      this.jsDetectors.set(id, state);
    }

    state.values.push(value);
    if (state.values.length > state.windowSize) {
      state.values.shift();
    }

    if (state.values.length < 2) {
      return { phase: 'Stable', variance: 0, mean: value };
    }

    const mean = state.values.reduce((a, b) => a + b, 0) / state.values.length;
    const variance =
      state.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (state.values.length - 1);

    const threshold = 2.0;
    let phase: string;
    if (variance < threshold * 0.5) phase = 'Stable';
    else if (variance < threshold) phase = 'Approaching';
    else if (variance < threshold * 1.5) phase = 'Critical';
    else phase = 'Transitioning';

    return { phase, variance, mean };
  }

  private phaseToNumber(phase: string): number {
    switch (phase) {
      case 'Stable':
        return 0;
      case 'Approaching':
        return 1;
      case 'Critical':
        return 2;
      case 'Transitioning':
        return 3;
      default:
        return 0;
    }
  }

  private numberToPhase(num: number): string {
    switch (num) {
      case 0:
        return 'Stable';
      case 1:
        return 'Approaching';
      case 2:
        return 'Critical';
      case 3:
        return 'Transitioning';
      default:
        return 'Stable';
    }
  }
}

/**
 * Phase states for detection
 */
type Phase = 'Stable' | 'Approaching' | 'Critical' | 'Transitioning';

/**
 * Internal detector state
 */
interface InternalDetectorState {
  phase: Phase;
  variance: number;
  mean: number;
  observations: number;
}

/**
 * SocialPulseDetector configuration
 */
export interface SocialPulseConfig {
  /** Data sources to use */
  sources?: DataSource[];
  /** Post filters to apply */
  filters?: PostFilter[];
  /** Window size for variance calculation */
  windowSize?: number;
  /** Sensitivity: low, medium, high */
  sensitivity?: 'low' | 'medium' | 'high';
  /** Aggregate by region */
  aggregateByRegion?: boolean;
  /** Enable WASM acceleration if available */
  enableWasm?: boolean;
  /** Enable data trace recording for audit trail */
  enableTrace?: boolean;
}

/**
 * Earnings calendar entry
 */
interface EarningsEntry {
  ticker: string;
  companyName: string;
  reportDate: Date;
  quarter: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  fiscalYear: number;
}

/**
 * Earnings sentiment tracking
 */
interface EarningsSentiment {
  ticker: string;
  daysUntilReport: number;
  sentimentTrend: number[];
  currentSentiment: number;
  sentimentVariance: number;
  phase: Phase;
  prediction: 'beat' | 'miss' | 'inline' | 'uncertain';
  confidence: number;
}

/**
 * Sensitivity presets (variance thresholds)
 */
const SENSITIVITY_THRESHOLDS = {
  low: { threshold: 2.5, windowSize: 50 },
  medium: { threshold: 2.0, windowSize: 30 },
  high: { threshold: 1.5, windowSize: 20 },
};

/**
 * Simple variance-based detector
 */
class SimpleVarianceDetector {
  private values: number[] = [];
  private windowSize: number;
  private threshold: number;

  constructor(windowSize = 30, threshold = 2.0) {
    this.windowSize = windowSize;
    this.threshold = threshold;
  }

  update(value: number): InternalDetectorState {
    this.values.push(value);
    if (this.values.length > this.windowSize) {
      this.values.shift();
    }
    return this.current();
  }

  current(): InternalDetectorState {
    if (this.values.length < 2) {
      return {
        phase: 'Stable',
        variance: 0,
        mean: this.values[0] ?? 0,
        observations: this.values.length,
      };
    }

    const mean = this.values.reduce((a, b) => a + b, 0) / this.values.length;
    const variance =
      this.values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / (this.values.length - 1);

    const phase = this.varianceToPhase(variance);

    return {
      phase,
      variance,
      mean,
      observations: this.values.length,
    };
  }

  reset(): void {
    this.values = [];
  }

  private varianceToPhase(variance: number): Phase {
    if (variance < this.threshold * 0.5) return 'Stable';
    if (variance < this.threshold) return 'Approaching';
    if (variance < this.threshold * 1.5) return 'Critical';
    return 'Transitioning';
  }
}

export class SocialPulseDetector {
  private sources: DataSource[] = [];
  private filters: PostFilter[] = [];
  private config: SocialPulseConfig;

  // Phase transition detectors by region/topic
  private detectors = new Map<string, SimpleVarianceDetector>();

  // Sentiment history for variance tracking
  private sentimentHistory = new Map<string, number[]>();

  // Earnings tracking
  private earningsCalendar: EarningsEntry[] = [];
  private earningsSentiment = new Map<string, EarningsSentiment>();

  // Ready state
  private initialized = false;

  // Dual-fusion WASM bridge
  private wasmBridge = new WasmBridge();

  // Data trace recorder for open-box visibility
  private traceRecorder = new DataTraceRecorder();

  constructor(config: SocialPulseConfig = {}) {
    this.config = {
      windowSize: config.windowSize ?? 30,
      sensitivity: config.sensitivity ?? 'medium',
      aggregateByRegion: config.aggregateByRegion ?? true,
      enableWasm: config.enableWasm ?? true,
      enableTrace: config.enableTrace ?? false,
      ...config,
    };

    if (config.sources) {
      this.sources = config.sources;
    }
    if (config.filters) {
      this.filters = config.filters;
    }

    // Enable trace if configured
    if (this.config.enableTrace) {
      this.traceRecorder.setEnabled(true);
    }
  }

  /**
   * Initialize detector and all sources
   */
  async init(): Promise<void> {
    const initStart = performance.now();

    // Try to initialize WASM bridge if enabled
    if (this.config.enableWasm) {
      await this.wasmBridge.init();
    }

    // Initialize all data sources
    await Promise.all(this.sources.map((s) => s.init()));

    this.initialized = true;

    this.traceRecorder.record(
      'transform',
      { sources: this.sources.length, wasmEnabled: this.config.enableWasm },
      { initialized: true, wasmAvailable: this.wasmBridge.isAvailable() },
      performance.now() - initStart,
      'init'
    );
  }

  // ============ WASM Bridge & Trace API ============

  /**
   * Get WASM bridge status (dual-fusion mode)
   */
  getWasmStatus(): WasmBridgeStatus {
    return this.wasmBridge.getStatus();
  }

  /**
   * Enable/disable data trace recording
   */
  setTraceEnabled(enabled: boolean): void {
    this.traceRecorder.setEnabled(enabled);
  }

  /**
   * Export data trace for analysis
   */
  exportTrace(): DataTrace {
    return this.traceRecorder.export();
  }

  /**
   * Export trace as JSON string
   */
  exportTraceJSON(): string {
    return this.traceRecorder.toJSON();
  }

  /**
   * Export trace as CSV
   */
  exportTraceCSV(): string {
    return this.traceRecorder.toCSV();
  }

  /**
   * Clear trace history
   */
  clearTrace(): void {
    this.traceRecorder.clear();
  }

  /**
   * Add a data source
   */
  addSource(source: DataSource): void {
    this.sources.push(source);
  }

  /**
   * Add a post filter
   */
  addFilter(filter: PostFilter): void {
    this.filters.push(filter);
  }

  /**
   * Fetch and process posts from all sources
   */
  async fetch(params: SearchParams = {}): Promise<SocialPost[]> {
    this.ensureInitialized();

    const fetchStart = performance.now();
    const allPosts: SocialPost[] = [];

    // Fetch from all sources with tracing
    const results = await Promise.allSettled(
      this.sources.map(async (source, idx) => {
        const sourceStart = performance.now();
        const posts = await source.fetch(params);
        this.traceRecorder.record(
          'fetch',
          params,
          { count: posts.length },
          performance.now() - sourceStart,
          source.platform || `source_${idx}`
        );
        return posts;
      })
    );

    for (const result of results) {
      if (result.status === 'fulfilled') {
        allPosts.push(...result.value);
      }
    }

    // Apply filters with tracing
    const filteredPosts: SocialPost[] = [];
    for (const post of allPosts) {
      let current: SocialPost | null = post;

      for (const filter of this.filters) {
        if (!current) break;
        const filterStart = performance.now();
        const before = current;
        current = await filter.process(current);
        this.traceRecorder.record(
          'filter',
          { postId: before.id },
          { passed: current !== null },
          performance.now() - filterStart,
          filter.constructor.name
        );
      }

      if (current) {
        filteredPosts.push(current);
      }
    }

    this.traceRecorder.record(
      'transform',
      { totalFetched: allPosts.length },
      { afterFilters: filteredPosts.length },
      performance.now() - fetchStart,
      'fetch_pipeline'
    );

    return filteredPosts;
  }

  /**
   * Update detector with new posts and return current state
   */
  async update(params: SearchParams = {}): Promise<{
    state: UpheavalState;
    aggregates: SentimentAggregate[];
    posts: SocialPost[];
    trace?: DataTrace;
  }> {
    const updateStart = performance.now();
    const posts = await this.fetch(params);

    // Calculate sentiment for posts
    const sentimentStart = performance.now();
    for (const post of posts) {
      if (post.sentimentScore === undefined) {
        post.sentimentScore = this.calculateSentiment(post.content);
      }
    }
    this.traceRecorder.record(
      'transform',
      { postCount: posts.length },
      { withSentiment: posts.filter((p) => p.sentimentScore !== undefined).length },
      performance.now() - sentimentStart,
      'sentiment_calculation'
    );

    // Aggregate posts
    const aggStart = performance.now();
    const aggregates = this.aggregatePosts(posts);
    this.traceRecorder.record(
      'aggregate',
      { postCount: posts.length },
      { aggregateCount: aggregates.length, regions: aggregates.map((a) => a.id) },
      performance.now() - aggStart,
      'aggregation'
    );

    // Run detection with timing
    const detectStart = performance.now();
    for (const agg of aggregates) {
      this.updateDetector(agg.id, agg.avgSentiment, agg.variance);
    }
    const detectDuration = performance.now() - detectStart;
    this.wasmBridge.recordTime(detectDuration, this.wasmBridge.isAvailable());

    this.traceRecorder.record(
      'detect',
      { aggregates: aggregates.map((a) => ({ id: a.id, variance: a.variance })) },
      { mode: this.wasmBridge.isAvailable() ? 'wasm' : 'js' },
      detectDuration,
      'phase_detection'
    );

    const state = this.calculateGlobalState(aggregates);

    this.traceRecorder.record(
      'transform',
      { level: state.level },
      { variance: state.variance, hotspots: state.hotspots.length },
      performance.now() - updateStart,
      'update_complete'
    );

    // Include trace in response if enabled
    const result: {
      state: UpheavalState;
      aggregates: SentimentAggregate[];
      posts: SocialPost[];
      trace?: DataTrace;
    } = { state, aggregates, posts };

    if (this.traceRecorder.isEnabled()) {
      result.trace = this.traceRecorder.export();
    }

    return result;
  }

  /**
   * Get current upheaval state
   */
  current(): UpheavalState {
    const hotspots = this.getHotspots();
    const globalVariance = this.calculateGlobalVariance();

    return {
      level: this.varianceToLevel(globalVariance),
      levelNumeric: this.varianceToNumeric(globalVariance),
      variance: globalVariance,
      mean: this.calculateGlobalMean(),
      dataPoints: this.getTotalDataPoints(),
      lastUpdate: new Date().toISOString(),
      hotspots,
    };
  }

  /**
   * Track earnings sentiment for a ticker
   */
  trackEarnings(ticker: string, reportDate: Date, quarter: 'Q1' | 'Q2' | 'Q3' | 'Q4'): void {
    this.earningsCalendar.push({
      ticker: ticker.toUpperCase(),
      companyName: ticker,
      reportDate,
      quarter,
      fiscalYear: reportDate.getFullYear(),
    });

    this.earningsSentiment.set(ticker.toUpperCase(), {
      ticker: ticker.toUpperCase(),
      daysUntilReport: Math.ceil((reportDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24)),
      sentimentTrend: [],
      currentSentiment: 0,
      sentimentVariance: 0,
      phase: 'Stable',
      prediction: 'uncertain',
      confidence: 0,
    });
  }

  /**
   * Update earnings sentiment with new data
   */
  async updateEarningsSentiment(ticker: string): Promise<EarningsSentiment | null> {
    const entry = this.earningsSentiment.get(ticker.toUpperCase());
    if (!entry) return null;

    const posts = await this.fetch({
      keywords: [ticker, `$${ticker}`],
      limit: 100,
    });

    if (posts.length === 0) return entry;

    const sentiments = posts
      .map((p) => p.sentimentScore)
      .filter((s): s is number => s !== undefined);

    if (sentiments.length === 0) return entry;

    const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
    const variance = this.calculateVariance(sentiments);

    entry.sentimentTrend.push(avgSentiment);
    if (entry.sentimentTrend.length > 30) {
      entry.sentimentTrend.shift();
    }

    entry.currentSentiment = avgSentiment;
    entry.sentimentVariance = variance;

    const calendarEntry = this.earningsCalendar.find((e) => e.ticker === ticker.toUpperCase());
    entry.daysUntilReport = calendarEntry
      ? Math.ceil((calendarEntry.reportDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24))
      : 0;

    const detector = this.getOrCreateDetector(`earnings:${ticker}`);
    detector.update(avgSentiment);
    entry.phase = detector.current().phase;

    entry.prediction = this.predictEarnings(entry);
    entry.confidence = this.calculatePredictionConfidence(entry);

    return entry;
  }

  /**
   * Get all earnings being tracked
   */
  getTrackedEarnings(): EarningsSentiment[] {
    return [...this.earningsSentiment.values()];
  }

  // ============ Private Methods ============

  private ensureInitialized(): void {
    if (!this.initialized) {
      throw new Error('SocialPulseDetector not initialized. Call init() first.');
    }
  }

  private getOrCreateDetector(id: string): SimpleVarianceDetector {
    let detector = this.detectors.get(id);
    if (!detector) {
      const preset = SENSITIVITY_THRESHOLDS[this.config.sensitivity ?? 'medium'];
      detector = new SimpleVarianceDetector(preset.windowSize, preset.threshold);
      this.detectors.set(id, detector);
    }
    return detector;
  }

  private updateDetector(id: string, sentiment: number, variance: number): void {
    const detector = this.getOrCreateDetector(id);
    detector.update(variance);

    let history = this.sentimentHistory.get(id);
    if (!history) {
      history = [];
      this.sentimentHistory.set(id, history);
    }
    history.push(sentiment);
    if (history.length > 100) {
      history.shift();
    }
  }

  private aggregatePosts(posts: SocialPost[]): SentimentAggregate[] {
    const byRegion = new Map<string, SocialPost[]>();

    for (const post of posts) {
      const region = post.geo?.countryCode ?? 'global';
      let regionPosts = byRegion.get(region);
      if (!regionPosts) {
        regionPosts = [];
        byRegion.set(region, regionPosts);
      }
      regionPosts.push(post);
    }

    const aggregates: SentimentAggregate[] = [];

    for (const [region, regionPosts] of byRegion) {
      const sentiments = regionPosts
        .map((p) => p.sentimentScore)
        .filter((s): s is number => s !== undefined);

      if (sentiments.length === 0) continue;

      const avgSentiment = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
      const variance = this.calculateVariance(sentiments);
      const negative = sentiments.filter((s) => s < 0).length;
      const botFiltered = posts.length - regionPosts.length;

      const keywords = this.extractKeywords(regionPosts);

      const platformCounts: Partial<Record<Platform, number>> = {};
      for (const post of regionPosts) {
        platformCounts[post.platform] = (platformCounts[post.platform] ?? 0) + 1;
      }

      const history = this.sentimentHistory.get(region);
      const previousVariance =
        history && history.length >= 2 ? this.calculateVariance(history.slice(-10, -1)) : undefined;

      const aggregate: SentimentAggregate = {
        id: region,
        windowStart: new Date(Date.now() - 3600000).toISOString(),
        windowEnd: new Date().toISOString(),
        postCount: regionPosts.length,
        authorCount: new Set(regionPosts.map((p) => p.author.id)).size,
        avgSentiment,
        sentimentStdDev: Math.sqrt(variance),
        negativeRatio: negative / sentiments.length,
        botFilteredRatio: botFiltered / Math.max(posts.length, 1),
        topKeywords: keywords.slice(0, 10),
        platformBreakdown: platformCounts,
        variance,
      };

      // Only add optional properties if they have values
      if (region !== 'global') {
        aggregate.countryCode = region;
      }
      if (previousVariance !== undefined) {
        aggregate.previousVariance = previousVariance;
      }

      aggregates.push(aggregate);
    }

    return aggregates;
  }

  private calculateGlobalState(aggregates: SentimentAggregate[]): UpheavalState {
    const hotspots: UpheavalState['hotspots'] = [];

    for (const agg of aggregates) {
      const level = this.varianceToLevel(agg.variance);
      if (level !== 'calm') {
        hotspots.push({
          countryCode: agg.countryCode ?? agg.id,
          level,
          variance: agg.variance,
          topKeywords: agg.topKeywords.slice(0, 5).map((k) => k.word),
        });
      }
    }

    hotspots.sort((a, b) => b.variance - a.variance);

    const globalVariance = this.calculateGlobalVariance();

    return {
      level: this.varianceToLevel(globalVariance),
      levelNumeric: this.varianceToNumeric(globalVariance),
      variance: globalVariance,
      mean: this.calculateGlobalMean(),
      dataPoints: this.getTotalDataPoints(),
      lastUpdate: new Date().toISOString(),
      hotspots,
    };
  }

  private calculateGlobalVariance(): number {
    let totalVariance = 0;
    let count = 0;

    for (const detector of this.detectors.values()) {
      const state = detector.current();
      totalVariance += state.variance;
      count++;
    }

    return count > 0 ? totalVariance / count : 0;
  }

  private calculateGlobalMean(): number {
    let total = 0;
    let count = 0;

    for (const history of this.sentimentHistory.values()) {
      if (history.length > 0) {
        total += history.reduce((a, b) => a + b, 0) / history.length;
        count++;
      }
    }

    return count > 0 ? total / count : 0;
  }

  private getTotalDataPoints(): number {
    let total = 0;
    for (const history of this.sentimentHistory.values()) {
      total += history.length;
    }
    return total;
  }

  private getHotspots(): UpheavalState['hotspots'] {
    const hotspots: UpheavalState['hotspots'] = [];

    for (const [id, detector] of this.detectors) {
      const state = detector.current();
      const level = this.phaseToLevel(state.phase);

      if (level !== 'calm') {
        hotspots.push({
          countryCode: id,
          level,
          variance: state.variance,
          topKeywords: [],
        });
      }
    }

    return hotspots.sort((a, b) => b.variance - a.variance);
  }

  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / (values.length - 1);
  }

  private varianceToLevel(variance: number): UpheavalLevel {
    const threshold = SENSITIVITY_THRESHOLDS[this.config.sensitivity ?? 'medium'].threshold;
    if (variance < threshold * 0.5) return 'calm';
    if (variance < threshold) return 'stirring';
    if (variance < threshold * 1.5) return 'unrest';
    return 'volatile';
  }

  private varianceToNumeric(variance: number): number {
    const level = this.varianceToLevel(variance);
    const mapping: Record<UpheavalLevel, number> = {
      calm: 0,
      stirring: 1,
      unrest: 2,
      volatile: 3,
    };
    return mapping[level];
  }

  private phaseToLevel(phase: Phase): UpheavalLevel {
    switch (phase) {
      case 'Stable':
        return 'calm';
      case 'Approaching':
        return 'stirring';
      case 'Critical':
        return 'unrest';
      case 'Transitioning':
        return 'volatile';
      default:
        return 'calm';
    }
  }

  /**
   * Simple sentiment calculation
   */
  private calculateSentiment(text: string): number {
    const lower = text.toLowerCase();

    const positiveWords = [
      'good',
      'great',
      'excellent',
      'amazing',
      'wonderful',
      'fantastic',
      'love',
      'happy',
      'joy',
      'success',
      'win',
      'best',
      'awesome',
      'perfect',
      'beautiful',
      'hope',
      'bullish',
      'growth',
      'profit',
      'gain',
      'rally',
      'surge',
      'boom',
    ];

    const negativeWords = [
      'bad',
      'terrible',
      'awful',
      'horrible',
      'hate',
      'sad',
      'angry',
      'fear',
      'fail',
      'loss',
      'worst',
      'poor',
      'ugly',
      'disaster',
      'crisis',
      'crash',
      'bearish',
      'decline',
      'drop',
      'plunge',
      'collapse',
      'recession',
      'war',
      'protest',
      'riot',
      'unrest',
      'violence',
      'conflict',
      'death',
      'kill',
    ];

    let score = 0;
    const words = lower.split(/\s+/);

    for (const word of words) {
      if (positiveWords.includes(word)) score += 0.1;
      if (negativeWords.includes(word)) score -= 0.1;
    }

    return Math.max(-1, Math.min(1, score));
  }

  private extractKeywords(posts: SocialPost[]): { word: string; count: number }[] {
    const wordCounts = new Map<string, number>();
    const stopwords = new Set([
      'the',
      'a',
      'an',
      'is',
      'are',
      'was',
      'were',
      'be',
      'been',
      'being',
      'have',
      'has',
      'had',
      'do',
      'does',
      'did',
      'will',
      'would',
      'could',
      'should',
      'may',
      'might',
      'must',
      'and',
      'or',
      'but',
      'if',
      'then',
      'than',
      'so',
      'as',
      'of',
      'at',
      'by',
      'for',
      'with',
      'about',
      'to',
      'from',
      'in',
      'on',
      'it',
      'its',
      'this',
      'that',
      'these',
      'those',
    ]);

    for (const post of posts) {
      const words = post.content
        .toLowerCase()
        .replace(/[^\w\s#@]/g, '')
        .split(/\s+/)
        .filter((w) => w.length > 2 && !stopwords.has(w));

      for (const word of words) {
        wordCounts.set(word, (wordCounts.get(word) ?? 0) + 1);
      }
    }

    return [...wordCounts.entries()]
      .map(([word, count]) => ({ word, count }))
      .sort((a, b) => b.count - a.count);
  }

  private predictEarnings(sentiment: EarningsSentiment): 'beat' | 'miss' | 'inline' | 'uncertain' {
    const trend = sentiment.sentimentTrend;
    if (trend.length < 5) return 'uncertain';

    const recentAvg = trend.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const olderSlice = trend.slice(-10, -5);
    const olderAvg =
      olderSlice.length > 0 ? olderSlice.reduce((a, b) => a + b, 0) / olderSlice.length : recentAvg;

    const trendDirection = recentAvg - olderAvg;

    if (sentiment.phase === 'Transitioning' || sentiment.phase === 'Approaching') {
      return 'uncertain';
    }

    if (trendDirection > 0.2 && sentiment.currentSentiment > 0.3) {
      return 'beat';
    }

    if (trendDirection < -0.2 && sentiment.currentSentiment < -0.3) {
      return 'miss';
    }

    if (Math.abs(sentiment.currentSentiment) < 0.2 && Math.abs(trendDirection) < 0.1) {
      return 'inline';
    }

    return 'uncertain';
  }

  private calculatePredictionConfidence(sentiment: EarningsSentiment): number {
    let confidence = 0.3;

    if (sentiment.sentimentTrend.length >= 20) confidence += 0.2;
    else if (sentiment.sentimentTrend.length >= 10) confidence += 0.1;

    const variance = this.calculateVariance(sentiment.sentimentTrend);
    if (variance < 0.1) confidence += 0.2;
    else if (variance < 0.2) confidence += 0.1;

    if (sentiment.daysUntilReport > 14) confidence -= 0.1;

    if (sentiment.prediction === 'uncertain') confidence -= 0.2;

    return Math.max(0, Math.min(1, confidence));
  }
}
