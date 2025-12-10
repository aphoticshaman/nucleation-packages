/**
 * WASM module loader for LatticeForge Core
 *
 * Provides high-performance Rust implementations:
 * - CIC Framework (Compression-Integration-Coherence)
 * - GTVC Clustering (Gauge-Theoretic Value Clustering)
 * - Q-matrix operations for regime transitions
 * - Persistence/TDA computations
 * - Geospatial system with nation dynamics
 */

// Dynamically imported WASM module
let wasmModule: typeof import('@/public/wasm/latticeforge_core') | null = null;
let wasmLoading: Promise<typeof wasmModule> | null = null;

export interface WasmCore {
  available: boolean;
  module?: typeof wasmModule;
}

export interface WasmSwarm {
  free: () => void;
  step: () => void;
  run: (n_steps: number) => unknown;
  get_positions: () => Float64Array;
  get_metrics: () => unknown;
  get_n_particles: () => number;
  get_time: () => number;
  set_attractor: (x: number, y: number) => void;
}

export interface WasmGeospatialSystem {
  free: () => void;
  add_nation: (code: string, name: string, lat: number, lon: number, regime: number) => void;
  set_esteem: (source: string, target: string, esteem: number) => void;
  get_esteem: (source: string, target: string) => number;
  step: () => void;
  run: (n_steps: number) => void;
  get_time: () => number;
  get_nation_count: () => number;
  to_geojson_basin: () => unknown;
  to_geojson_risk: () => unknown;
  get_nation: (code: string) => unknown;
  get_all_nations: () => unknown;
}

export interface WasmPersistence {
  [key: string]: unknown;
}

/**
 * Load the WASM module (lazy, singleton)
 */
export async function loadWasm(): Promise<WasmCore> {
  // Already loaded
  if (wasmModule) {
    return { available: true, module: wasmModule };
  }

  // Loading in progress
  if (wasmLoading) {
    try {
      wasmModule = await wasmLoading;
      return { available: !!wasmModule, module: wasmModule ?? undefined };
    } catch {
      return { available: false };
    }
  }

  // Start loading
  wasmLoading = (async () => {
    try {
      // Dynamic import for tree-shaking
      const mod = await import('@/public/wasm/latticeforge_core');

      // Initialize WASM (calls the init() function defined in wasm.rs)
      if (typeof mod.default === 'function') {
        await mod.default();
      }

      console.log('[WASM] LatticeForge Core loaded successfully');
      return mod;
    } catch (error) {
      console.warn('[WASM] Failed to load LatticeForge Core:', error);
      return null;
    }
  })();

  wasmModule = await wasmLoading;
  return { available: !!wasmModule, module: wasmModule ?? undefined };
}

/**
 * Check if WASM is loaded
 */
export function isWasmLoaded(): boolean {
  return wasmModule !== null;
}

/**
 * Get the WASM module (must be loaded first)
 */
export function getWasm() {
  return wasmModule;
}

// ============================================================
// Convenience wrappers for common operations
// ============================================================

/**
 * Compute CIC functional (WASM-accelerated)
 */
export async function computeCIC(
  samples: string[],
  values: number[],
  lambda = 0.5,
  gamma = 0.3
): Promise<{ phi: number; entropy: number; coherence: number; functional: number; confidence: number } | null> {
  const wasm = await loadWasm();
  if (!wasm.available || !wasm.module) return null;

  try {
    const result = wasm.module.wasm_compute_cic(
      JSON.stringify(samples),
      new Float64Array(values),
      lambda,
      gamma
    );
    return result as { phi: number; entropy: number; coherence: number; functional: number; confidence: number };
  } catch (error) {
    console.error('[WASM] CIC computation failed:', error);
    return null;
  }
}

/**
 * Perform GTVC clustering (WASM-accelerated)
 */
export async function gaugeClustering(
  values: number[],
  epsilon = 0.05
): Promise<Array<{ center: number; size: number; confidence: number }> | null> {
  const wasm = await loadWasm();
  if (!wasm.available || !wasm.module) return null;

  try {
    const result = wasm.module.wasm_gauge_clustering(
      new Float64Array(values),
      epsilon
    );
    return result as Array<{ center: number; size: number; confidence: number }>;
  } catch (error) {
    console.error('[WASM] Gauge clustering failed:', error);
    return null;
  }
}

/**
 * Fuse signals using GTVC + CIC (WASM-accelerated)
 */
export async function fuseSignals(
  values: number[],
  epsilon = 0.05,
  lambda = 0.5,
  gamma = 0.3
): Promise<{ value: number; confidence: number; n_clusters: number; phase: string } | null> {
  const wasm = await loadWasm();
  if (!wasm.available || !wasm.module) return null;

  try {
    const result = wasm.module.wasm_fuse_signals(
      new Float64Array(values),
      epsilon,
      lambda,
      gamma
    );
    return result as { value: number; confidence: number; n_clusters: number; phase: string };
  } catch (error) {
    console.error('[WASM] Signal fusion failed:', error);
    return null;
  }
}

/**
 * Batch fuse multiple signal sets (WASM-accelerated)
 * Ideal for processing many GDELT signals at once
 */
export async function batchFuseSignals(
  valueSets: number[][],
  epsilon = 0.05,
  lambda = 0.5,
  gamma = 0.3
): Promise<Array<{ value: number; confidence: number; n_clusters: number; phase: string }> | null> {
  const wasm = await loadWasm();
  if (!wasm.available || !wasm.module) return null;

  try {
    const result = wasm.module.wasm_batch_fuse(
      JSON.stringify(valueSets),
      epsilon,
      lambda,
      gamma
    );
    return result as Array<{ value: number; confidence: number; n_clusters: number; phase: string }>;
  } catch (error) {
    console.error('[WASM] Batch fusion failed:', error);
    return null;
  }
}

/**
 * Create a new geospatial system (WASM-accelerated)
 */
export async function createGeospatialSystem(): Promise<WasmGeospatialSystem | null> {
  const wasm = await loadWasm();
  if (!wasm.available || !wasm.module) return null;

  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const WasmGeospatialSystem = (wasm.module as any).WasmGeospatialSystem;
    return WasmGeospatialSystem.with_defaults() as WasmGeospatialSystem;
  } catch (error) {
    console.error('[WASM] Failed to create geospatial system:', error);
    return null;
  }
}
