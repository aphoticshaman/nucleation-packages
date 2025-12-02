// WASM module loader - loads from /public/wasm at runtime
// This avoids webpack trying to parse the WASM at build time

let wasmModule: WasmCore | null = null;
let wasmLoading: Promise<WasmCore> | null = null;

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
    return wasmModule;
  }

  if (wasmLoading) {
    return wasmLoading;
  }

  wasmLoading = (async () => {
    try {
      // Load WASM from public folder at runtime
      const wasmUrl = '/wasm/latticeforge_core_bg.wasm';

      // Fetch and compile the WASM module
      const wasmResponse = await fetch(wasmUrl);
      const wasmBytes = await wasmResponse.arrayBuffer();

      // For wasm-bindgen generated modules, we need to load the JS glue
      // and initialize it with the WASM bytes
      // @ts-expect-error - Runtime import from public folder, no types available
      const initModule = await import(/* webpackIgnore: true */ '/wasm/latticeforge_core.js');
      const init = initModule.default;

      await init(wasmBytes);

      // Re-import to get the exports after init
      // @ts-expect-error - Runtime import from public folder, no types available
      const wasm = await import(/* webpackIgnore: true */ '/wasm/latticeforge_core.js');

      wasmModule = wasm as unknown as WasmCore;
      return wasmModule;
    } catch (error) {
      console.error('Failed to load WASM:', error);
      // Return a mock for graceful degradation
      throw error;
    }
  })();

  return wasmLoading;
}

// Helper to check if WASM is loaded
export function isWasmLoaded(): boolean {
  return wasmModule !== null;
}
