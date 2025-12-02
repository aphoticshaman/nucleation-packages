// WASM module loader - loads from /public/wasm at runtime
// This avoids webpack trying to parse the WASM at build time

let wasmModule: WasmCore | null = null;
let wasmLoading: Promise<WasmCore> | null = null;

export interface WasmCore {
  // Particle swarm - actual constructor signature from wasm-bindgen
  WasmSwarm: new (
    n_particles: number,
    dt: number,
    diffusion: number,
    interaction_strength: number,
    attractor_x: number,
    attractor_y: number,
    attractor_strength: number,
    seed: bigint
  ) => WasmSwarm;

  // Persistence
  wasm_compute_persistence: (
    points_flat: Float64Array,
    n_points: number,
    max_edge: number
  ) => WasmPersistence;

  // Q-matrix
  wasm_build_q_matrix: (rates_flat: Float64Array, n: number) => Float64Array;
  wasm_analyze_q: (q_flat: Float64Array, n: number) => unknown;

  // Distance matrix
  wasm_distance_matrix: (points_flat: Float64Array, n_points: number) => Float64Array;

  // Geodesic integration
  wasm_integrate_geodesic_fisher: (
    x0: Float64Array,
    v0: Float64Array,
    dt: number,
    n_steps: number
  ) => unknown;

  // Markov chain simulation
  wasm_simulate_markov_chain: (
    q_flat: Float64Array,
    n: number,
    r0: number,
    total_time: number,
    dt: number,
    seed: bigint
  ) => unknown;

  // Persistent entropy
  wasm_persistent_entropy: (births: Float64Array, deaths: Float64Array) => number;

  // Init function
  init: () => void;
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

export interface WasmPersistence {
  // Return type from wasm_compute_persistence
  [key: string]: unknown;
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
