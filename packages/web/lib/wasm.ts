// WASM module loader
// After running `npm run wasm:copy`, the WASM module will be in ./wasm/

let wasmModule: typeof import('./wasm/latticeforge_core') | null = null;
let wasmLoading: Promise<typeof import('./wasm/latticeforge_core')> | null = null;

export interface WasmCore {
  // Particle swarm
  WasmSwarm: new (n: number, dims: number) => WasmSwarm;

  // Persistence
  wasm_compute_persistence: (points: Float64Array, n: number, dims: number, max_edge: number) => WasmPersistence;

  // Q-matrix
  wasm_build_q_matrix: (k: number, kappaUp: number, kappaDown: number) => Float64Array;
  wasm_stationary_distribution: (qMatrix: Float64Array, k: number) => Float64Array;

  // Geospatial
  WasmGeospatialSystem: new (nDims: number) => WasmGeospatialSystem;
}

export interface WasmSwarm {
  step: () => void;
  run: (n: number) => void;
  get_positions: () => Float64Array;
  get_mean_position: () => Float64Array;
  get_mean_field_strength: () => number;
}

export interface WasmPersistence {
  h0_pairs: Float64Array;
  h1_pairs: Float64Array;
  total_persistence: number;
  persistent_entropy: number;
}

export interface WasmGeospatialSystem {
  add_nation: (code: string, name: string, lat: number, lon: number, position: Float64Array, regime: number) => void;
  set_esteem: (source: string, target: string, esteem: number) => void;
  step: () => void;
  run: (n: number) => void;
  to_geojson: (layer: string) => string;
  get_comparison: (code1: string, code2: string) => string;
}

export async function loadWasm(): Promise<WasmCore> {
  if (wasmModule) {
    return wasmModule as unknown as WasmCore;
  }

  if (wasmLoading) {
    return wasmLoading as unknown as Promise<WasmCore>;
  }

  wasmLoading = (async () => {
    try {
      // Dynamic import of WASM module
      const wasm = await import('./wasm/latticeforge_core');
      await wasm.default();
      wasmModule = wasm;
      return wasm;
    } catch (error) {
      console.error('Failed to load WASM:', error);
      throw error;
    }
  })();

  return wasmLoading as unknown as Promise<WasmCore>;
}

// Helper to check if WASM is loaded
export function isWasmLoaded(): boolean {
  return wasmModule !== null;
}
