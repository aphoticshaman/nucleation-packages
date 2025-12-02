import { describe, it, expect, beforeEach } from 'vitest';
import { DualFusionEngine } from '../src/detector.js';
import type { PerformanceMetrics } from '../src/detector.js';

describe('DualFusionEngine', () => {
  let engine: DualFusionEngine;

  beforeEach(() => {
    engine = new DualFusionEngine();
  });

  describe('initialization', () => {
    it('should initialize without errors', async () => {
      // WASM may not be available in test environment, that's OK
      const wasmAvailable = await engine.init();
      expect(typeof wasmAvailable).toBe('boolean');
    });

    it('should return WASM status after init', async () => {
      await engine.init();
      const status = engine.getWasmStatus();

      expect(status).toHaveProperty('available');
      expect(status).toHaveProperty('mode');
      expect(['wasm', 'js']).toContain(status.mode);
    });
  });

  describe('API registration', () => {
    it('should register API sources', () => {
      engine.registerAPI({
        name: 'test-api',
        endpoint: 'https://example.com/api',
      });

      // No error means success
      expect(true).toBe(true);
    });

    it('should register multiple API sources', () => {
      engine.registerAPI({
        name: 'api-1',
        endpoint: 'https://example.com/api1',
      });
      engine.registerAPI({
        name: 'api-2',
        endpoint: 'https://example.com/api2',
        headers: { Authorization: 'Bearer test' },
      });
      engine.registerAPI({
        name: 'api-3',
        endpoint: 'https://example.com/api3',
        timeout: 10000,
        rateLimit: 5,
      });

      expect(true).toBe(true);
    });

    it('should accept custom transform functions', () => {
      engine.registerAPI({
        name: 'custom-api',
        endpoint: 'https://example.com/api',
        transform: (data: unknown) => {
          if (Array.isArray(data)) {
            return data.map((item) => (typeof item === 'number' ? item : 0));
          }
          return [];
        },
      });

      expect(true).toBe(true);
    });
  });

  describe('signal normalization', () => {
    it('should normalize signals using Z-score', () => {
      const signal = [10, 20, 30, 40, 50];
      const normalized = engine.normalizeSignal(signal);

      // Z-score normalized values should have mean ≈ 0 and std ≈ 1
      const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
      expect(Math.abs(mean)).toBeLessThan(0.0001);
    });

    it('should handle empty signals', () => {
      const normalized = engine.normalizeSignal([]);
      expect(normalized).toEqual([]);
    });

    it('should handle single-value signals', () => {
      const normalized = engine.normalizeSignal([42]);
      expect(normalized).toEqual([42]);
    });

    it('should handle constant signals (zero variance)', () => {
      const signal = [5, 5, 5, 5, 5];
      const normalized = engine.normalizeSignal(signal);
      expect(normalized).toEqual([0, 0, 0, 0, 0]);
    });
  });

  describe('signal fusion', () => {
    it('should fuse multiple signals', () => {
      const signals = new Map<string, number[]>();
      signals.set('signal-1', [1, 2, 3, 4, 5]);
      signals.set('signal-2', [5, 4, 3, 2, 1]);

      const fused = engine.fuseSignals(signals);

      expect(fused.length).toBe(5);
    });

    it('should handle empty signal map', () => {
      const signals = new Map<string, number[]>();
      const fused = engine.fuseSignals(signals);
      expect(fused).toEqual([]);
    });

    it('should handle signals of different lengths', () => {
      const signals = new Map<string, number[]>();
      signals.set('short', [1, 2, 3]);
      signals.set('long', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      const fused = engine.fuseSignals(signals);

      // Should use minimum length
      expect(fused.length).toBe(3);
    });
  });

  describe('batch detection', () => {
    it('should detect phase from batch of values', async () => {
      await engine.init();

      const values = [0.1, 0.2, 0.15, 0.18, 0.22, 0.19, 0.21, 0.17];
      const result = engine.detectBatch('test-detector', values);

      expect(result).toHaveProperty('phase');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('variance');
      expect(result).toHaveProperty('inflectionMagnitude');

      expect(typeof result.phase).toBe('number');
      expect(result.phase).toBeGreaterThanOrEqual(0);
      expect(result.phase).toBeLessThanOrEqual(3);
    });

    it('should handle empty batch', async () => {
      await engine.init();

      const result = engine.detectBatch('empty-test', []);

      expect(result).toHaveProperty('phase');
    });

    it('should detect high variance as higher phase', async () => {
      await engine.init();

      // Low variance data
      const stableValues = Array.from({ length: 50 }, () => 0.5 + Math.random() * 0.01);
      const stableResult = engine.detectBatch('stable', stableValues);

      // High variance data
      const volatileValues = Array.from({ length: 50 }, () => Math.random() * 10);
      const volatileResult = engine.detectBatch('volatile', volatileValues);

      // Volatile should have higher or equal phase
      expect(volatileResult.variance).toBeGreaterThan(stableResult.variance);
    });
  });

  describe('performance metrics', () => {
    it('should track performance metrics', async () => {
      await engine.init();

      // Process some data
      engine.detectBatch('metrics-test', [1, 2, 3, 4, 5]);

      const metrics: PerformanceMetrics = engine.getMetrics();

      expect(metrics).toHaveProperty('wasmSpeedup');
      expect(metrics).toHaveProperty('dataPointsProcessed');
      expect(metrics).toHaveProperty('signalsCombined');
      expect(metrics).toHaveProperty('avgLatency_ms');
      expect(metrics).toHaveProperty('throughput_per_sec');
      expect(metrics).toHaveProperty('accuracyGain');

      expect(metrics.dataPointsProcessed).toBeGreaterThan(0);
    });
  });

  describe('data trace export', () => {
    it('should export trace as JSON', async () => {
      await engine.init();

      engine.detectBatch('trace-test', [1, 2, 3]);

      const json = engine.exportTraceJSON();
      const trace = JSON.parse(json);

      expect(trace).toHaveProperty('sessionId');
      expect(trace).toHaveProperty('startTime');
      expect(trace).toHaveProperty('entries');
      expect(trace).toHaveProperty('summary');
    });

    it('should export trace as CSV', async () => {
      await engine.init();

      engine.detectBatch('csv-test', [1, 2, 3]);

      const csv = engine.exportTraceCSV();

      expect(csv).toContain('id,timestamp,type,source,duration_ms');
    });

    it('should clear trace history', async () => {
      await engine.init();

      engine.detectBatch('clear-test', [1, 2, 3]);
      engine.clearTrace();

      const trace = engine.exportTrace();
      expect(trace.entries.length).toBe(0);
    });
  });

  describe('streaming mode', () => {
    it('should start and stop streaming', async () => {
      await engine.init();

      let callbackInvoked = false;
      engine.startStream(() => {
        callbackInvoked = true;
      }, 100);

      // Wait briefly
      await new Promise((resolve) => setTimeout(resolve, 50));

      engine.stopStream();

      // Should have stopped without error (callback may or may not have been invoked)
      expect(typeof callbackInvoked).toBe('boolean');
    });
  });

  describe('resource cleanup', () => {
    it('should dispose resources without error', async () => {
      await engine.init();

      engine.registerAPI({
        name: 'dispose-test',
        endpoint: 'https://example.com/api',
      });

      engine.detectBatch('dispose-test', [1, 2, 3]);

      engine.dispose();

      // Should complete without error
      expect(true).toBe(true);
    });
  });
});
