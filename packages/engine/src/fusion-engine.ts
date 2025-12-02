/**
 * FusionEngine
 *
 * Multi-signal fusion for phase transition detection.
 * Combines data from multiple sources with WASM acceleration.
 */

import type {
  EngineConfig,
  ExternalAPIConfig,
  FusionResult,
  PerformanceMetrics,
  DataTrace,
  DataProvenance,
} from './types.js';
import { WasmBridge } from './wasm-bridge.js';
import { DataTraceRecorder } from './trace-recorder.js';

export class FusionEngine {
  private wasmBridge: WasmBridge;
  private traceRecorder: DataTraceRecorder;
  private config: Required<EngineConfig>;
  private apiSources: Map<string, ExternalAPIConfig> = new Map();
  private signalCache: Map<string, { data: number[]; timestamp: number }> = new Map();
  private metrics: PerformanceMetrics = {
    wasmCalls: 0,
    jsCalls: 0,
    avgWasmTime: 0,
    avgJsTime: 0,
    cacheHits: 0,
    cacheMisses: 0,
  };
  private streamingInterval: ReturnType<typeof setInterval> | null = null;

  constructor(config: EngineConfig = {}) {
    this.config = {
      enableTracing: config.enableTracing ?? false,
      enableWasm: config.enableWasm ?? true,
      defaultPollInterval: config.defaultPollInterval ?? '1h',
      respectRateLimits: config.respectRateLimits ?? true,
      maxConcurrentRequests: config.maxConcurrentRequests ?? 5,
    };

    this.wasmBridge = new WasmBridge();
    this.traceRecorder = new DataTraceRecorder();

    if (this.config.enableTracing) {
      this.traceRecorder.setEnabled(true);
    }
  }

  /**
   * Initialize the engine
   */
  async initialize(): Promise<void> {
    if (this.config.enableWasm) {
      await this.wasmBridge.initialize();
    }
  }

  /**
   * Register an external API source
   */
  registerSource(config: ExternalAPIConfig): void {
    this.apiSources.set(config.name, config);
  }

  /**
   * Unregister an API source
   */
  unregisterSource(name: string): boolean {
    return this.apiSources.delete(name);
  }

  /**
   * Get registered source names
   */
  getRegisteredSources(): string[] {
    return [...this.apiSources.keys()];
  }

  /**
   * Fetch data from a registered source
   */
  async fetchSource(name: string): Promise<number[]> {
    const config = this.apiSources.get(name);
    if (!config) {
      throw new Error(`Source not registered: ${name}`);
    }

    const startTime = performance.now();

    // Check cache
    const cached = this.signalCache.get(name);
    const cacheAge = cached ? Date.now() - cached.timestamp : Infinity;
    const cacheMaxAge = this.parseDuration(this.config.defaultPollInterval);

    if (cached && cacheAge < cacheMaxAge) {
      this.metrics.cacheHits++;
      return cached.data;
    }

    this.metrics.cacheMisses++;

    try {
      const fetchOptions: RequestInit = {
        method: 'GET',
        signal: AbortSignal.timeout(config.timeout ?? 30000),
      };

      if (config.headers) {
        fetchOptions.headers = config.headers;
      }

      const response = await fetch(config.endpoint, fetchOptions);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const rawData = await response.json();
      const data = config.transform ? config.transform(rawData) : (rawData as number[]);

      // Cache the result
      this.signalCache.set(name, { data, timestamp: Date.now() });

      const duration = performance.now() - startTime;

      // Record trace with provenance
      const provenance: DataProvenance = {
        sourceId: name,
        sourceTier: config.tier,
        fetchedAt: new Date().toISOString(),
        attribution: config.attribution,
        license: config.license,
        transformations: config.transform ? ['custom-transform'] : [],
        hash: this.hashData(data),
      };

      this.traceRecorder.record(
        'fetch',
        { endpoint: config.endpoint },
        { length: data.length },
        duration,
        name,
        undefined,
        provenance
      );

      return data;
    } catch (error) {
      const duration = performance.now() - startTime;
      this.traceRecorder.record(
        'fetch',
        { endpoint: config.endpoint },
        { error: error instanceof Error ? error.message : 'Unknown error' },
        duration,
        name
      );
      throw error;
    }
  }

  /**
   * Normalize signals using Z-score
   */
  normalizeSignals(signals: number[]): number[] {
    if (signals.length === 0) return [];
    if (signals.length === 1) return [0];

    const mean = signals.reduce((a, b) => a + b, 0) / signals.length;
    const variance =
      signals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / signals.length;
    const stdDev = Math.sqrt(variance);

    if (stdDev === 0) return signals.map(() => 0);

    const startTime = performance.now();
    const normalized = signals.map((val) => (val - mean) / stdDev);
    const duration = performance.now() - startTime;

    this.traceRecorder.record(
      'normalize',
      { length: signals.length, mean, stdDev },
      { length: normalized.length },
      duration
    );

    return normalized;
  }

  /**
   * Fuse multiple signals into one
   */
  fuseSignals(signalMap: Map<string, number[]>): number[] {
    if (signalMap.size === 0) return [];

    const startTime = performance.now();

    // Normalize each signal
    const normalized: number[][] = [];
    for (const [, signal] of signalMap) {
      normalized.push(this.normalizeSignals(signal));
    }

    // Find minimum length
    const minLength = Math.min(...normalized.map((s) => s.length));
    if (minLength === 0) return [];

    // Average across signals
    const fused: number[] = [];
    for (let i = 0; i < minLength; i++) {
      const sum = normalized.reduce((acc, signal) => acc + (signal[i] ?? 0), 0);
      fused.push(sum / normalized.length);
    }

    const duration = performance.now() - startTime;
    this.traceRecorder.record(
      'fuse',
      { signalCount: signalMap.size, lengths: [...signalMap.values()].map((s) => s.length) },
      { length: fused.length },
      duration
    );

    return fused;
  }

  /**
   * Detect phase from fused signal
   */
  detect(data: number[]): { phase: number; usedWasm: boolean } {
    const startTime = performance.now();
    const result = this.wasmBridge.detectPhase(data);
    const duration = performance.now() - startTime;

    if (result.usedWasm) {
      this.metrics.wasmCalls++;
      this.metrics.avgWasmTime =
        (this.metrics.avgWasmTime * (this.metrics.wasmCalls - 1) + duration) /
        this.metrics.wasmCalls;
    } else {
      this.metrics.jsCalls++;
      this.metrics.avgJsTime =
        (this.metrics.avgJsTime * (this.metrics.jsCalls - 1) + duration) / this.metrics.jsCalls;
    }

    this.traceRecorder.record(
      'detect',
      { dataLength: data.length },
      { phase: result.phase, usedWasm: result.usedWasm },
      duration
    );

    return result;
  }

  /**
   * Batch detect phases
   */
  detectBatch(data: number[], windowSize: number): { phases: number[]; usedWasm: boolean } {
    const startTime = performance.now();
    const result = this.wasmBridge.detectPhaseBatch(data, windowSize);
    const duration = performance.now() - startTime;

    this.traceRecorder.record(
      'batch',
      { dataLength: data.length, windowSize },
      { phasesCount: result.phases.length, usedWasm: result.usedWasm },
      duration
    );

    return result;
  }

  /**
   * Full fusion detection pipeline
   */
  async detectFused(sourceNames?: string[]): Promise<FusionResult> {
    const sources = sourceNames ?? this.getRegisteredSources();
    const signals = new Map<string, number[]>();
    const provenanceList: DataProvenance[] = [];

    // Fetch all sources in parallel (respecting concurrency limit)
    const chunks = this.chunkArray(sources, this.config.maxConcurrentRequests);

    for (const chunk of chunks) {
      const results = await Promise.allSettled(
        chunk.map(async (name) => {
          const data = await this.fetchSource(name);
          return { name, data };
        })
      );

      for (const result of results) {
        if (result.status === 'fulfilled') {
          signals.set(result.value.name, result.value.data);
          const config = this.apiSources.get(result.value.name);
          if (config) {
            provenanceList.push({
              sourceId: result.value.name,
              sourceTier: config.tier,
              fetchedAt: new Date().toISOString(),
              attribution: config.attribution,
              license: config.license,
              transformations: [],
              hash: this.hashData(result.value.data),
            });
          }
        }
      }
    }

    // Fuse signals
    const fusedSignal = this.fuseSignals(signals);

    // Detect phase
    const { phase } = this.detect(fusedSignal);

    return {
      phase,
      confidence: this.calculateConfidence(signals),
      signals,
      fusedSignal,
      timestamp: new Date().toISOString(),
      provenance: provenanceList,
    };
  }

  /**
   * Start streaming detection mode
   */
  startStreaming(callback: (result: FusionResult) => void, intervalMs = 60000): void {
    if (this.streamingInterval) {
      this.stopStreaming();
    }

    const runDetection = (): void => {
      this.detectFused()
        .then(callback)
        .catch((err) => console.error('Streaming detection error:', err));
    };

    // Run immediately, then on interval
    runDetection();
    this.streamingInterval = setInterval(runDetection, intervalMs);
  }

  /**
   * Stop streaming detection mode
   */
  stopStreaming(): void {
    if (this.streamingInterval) {
      clearInterval(this.streamingInterval);
      this.streamingInterval = null;
    }
  }

  /**
   * Get WASM status
   */
  getWasmStatus() {
    return this.wasmBridge.getStatus();
  }

  /**
   * Get performance metrics
   */
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Export trace data
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
   * Clear signal cache
   */
  clearCache(): void {
    this.signalCache.clear();
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.stopStreaming();
    this.clearCache();
    this.clearTrace();
    this.apiSources.clear();
  }

  // Private helpers

  private parseDuration(duration: string): number {
    const match = duration.match(/^(\d+)(ms|s|m|h|d)$/);
    if (!match) return 3600000; // Default 1 hour

    const value = parseInt(match[1], 10);
    const unit = match[2];

    const multipliers: Record<string, number> = {
      ms: 1,
      s: 1000,
      m: 60000,
      h: 3600000,
      d: 86400000,
    };

    return value * (multipliers[unit] ?? 3600000);
  }

  private calculateConfidence(signals: Map<string, number[]>): number {
    if (signals.size === 0) return 0;
    if (signals.size === 1) return 0.5;

    // More sources = higher confidence (up to a point)
    const sourceConfidence = Math.min(signals.size / 5, 1);

    // Longer signals = higher confidence
    const avgLength = [...signals.values()].reduce((sum, s) => sum + s.length, 0) / signals.size;
    const lengthConfidence = Math.min(avgLength / 100, 1);

    return (sourceConfidence + lengthConfidence) / 2;
  }

  private hashData(data: number[]): string {
    // Simple hash for data integrity
    let hash = 0;
    const str = JSON.stringify(data);
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return hash.toString(16);
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}
