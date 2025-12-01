/**
 * WasmBridge
 *
 * Bridge to nucleation-wasm for accelerated detection.
 * Falls back to pure JS when WASM unavailable.
 */

import type { WasmStatus } from './types.js';

// Type for nucleation-wasm module
interface NucleationWasm {
  detect_phase: (data: Float64Array) => number;
  detect_phase_batch: (data: Float64Array, windowSize: number) => Float64Array;
  get_version: () => string;
}

export class WasmBridge {
  private wasmModule: NucleationWasm | null = null;
  private status: WasmStatus = {
    available: false,
    lastCheck: new Date().toISOString(),
  };

  /**
   * Initialize WASM module
   */
  async initialize(): Promise<WasmStatus> {
    try {
      // Dynamic import of nucleation-wasm
      const wasm = (await import('nucleation-wasm')) as NucleationWasm;

      if (wasm && typeof wasm.detect_phase === 'function') {
        this.wasmModule = wasm;
        this.status = {
          available: true,
          version: typeof wasm.get_version === 'function' ? wasm.get_version() : 'unknown',
          lastCheck: new Date().toISOString(),
        };
      } else {
        throw new Error('Invalid WASM module structure');
      }
    } catch (error) {
      this.status = {
        available: false,
        lastCheck: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }

    return this.status;
  }

  /**
   * Get current WASM status
   */
  getStatus(): WasmStatus {
    return { ...this.status };
  }

  /**
   * Check if WASM is available
   */
  isAvailable(): boolean {
    return this.status.available && this.wasmModule !== null;
  }

  /**
   * Detect phase using WASM (with JS fallback)
   */
  detectPhase(data: number[]): { phase: number; usedWasm: boolean } {
    if (this.wasmModule) {
      try {
        const arr = new Float64Array(data);
        const phase = this.wasmModule.detect_phase(arr);
        return { phase, usedWasm: true };
      } catch {
        // Fall through to JS implementation
      }
    }

    // Pure JS fallback
    const phase = this.detectPhaseJS(data);
    return { phase, usedWasm: false };
  }

  /**
   * Batch detect phases using WASM (with JS fallback)
   */
  detectPhaseBatch(
    data: number[],
    windowSize: number
  ): { phases: number[]; usedWasm: boolean } {
    if (this.wasmModule) {
      try {
        const arr = new Float64Array(data);
        const result = this.wasmModule.detect_phase_batch(arr, windowSize);
        return { phases: Array.from(result), usedWasm: true };
      } catch {
        // Fall through to JS implementation
      }
    }

    // Pure JS fallback
    const phases = this.detectPhaseBatchJS(data, windowSize);
    return { phases, usedWasm: false };
  }

  /**
   * Pure JS phase detection (fallback)
   */
  private detectPhaseJS(data: number[]): number {
    if (data.length < 2) return 0;

    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const variance =
      data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;

    // Simple variance-based phase detection
    // Higher variance = higher phase (more instability)
    const normalizedVariance = Math.min(variance / (mean * mean + 1), 1);
    return normalizedVariance;
  }

  /**
   * Pure JS batch phase detection (fallback)
   */
  private detectPhaseBatchJS(data: number[], windowSize: number): number[] {
    const phases: number[] = [];

    for (let i = 0; i <= data.length - windowSize; i++) {
      const window = data.slice(i, i + windowSize);
      phases.push(this.detectPhaseJS(window));
    }

    return phases;
  }

  /**
   * Get WASM version
   */
  getVersion(): string | undefined {
    return this.status.version;
  }
}
