/* tslint:disable */
/* eslint-disable */

export class WasmSwarm {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get current metrics
   */
  get_metrics(): any;
  /**
   * Get current positions as flat array
   */
  get_positions(): Float64Array;
  /**
   * Set attractor position
   */
  set_attractor(x: number, y: number): void;
  /**
   * Get number of particles
   */
  get_n_particles(): number;
  /**
   * Create new swarm
   */
  constructor(
    n_particles: number,
    dt: number,
    diffusion: number,
    interaction_strength: number,
    attractor_x: number,
    attractor_y: number,
    attractor_strength: number,
    seed: bigint
  );
  /**
   * Run multiple steps
   */
  run(n_steps: number): any;
  /**
   * Step simulation
   */
  step(): void;
  /**
   * Get time
   */
  get_time(): number;
}

/**
 * Geospatial attractor system for nation-level dynamics
 */
export class WasmGeospatialSystem {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create new geospatial system
   */
  constructor(
    n_dims: number,
    interaction_decay: number,
    min_influence: number,
    dt: number,
    diffusion: number
  );
  /**
   * Create with default configuration
   */
  static with_defaults(): WasmGeospatialSystem;
  /**
   * Add a nation to the system
   */
  add_nation(code: string, name: string, lat: number, lon: number, regime: number): void;
  /**
   * Add a nation with initial position in attractor space
   */
  add_nation_with_position(
    code: string,
    name: string,
    lat: number,
    lon: number,
    position: Float64Array,
    regime: number
  ): void;
  /**
   * Set esteem relationship between nations
   */
  set_esteem(source: string, target: string, esteem: number): void;
  /**
   * Get esteem from source to target
   */
  get_esteem(source: string, target: string): number;
  /**
   * Run one simulation step
   */
  step(): void;
  /**
   * Run multiple simulation steps
   */
  run(n_steps: number): void;
  /**
   * Get current simulation time
   */
  get_time(): number;
  /**
   * Get number of nations
   */
  get_nation_count(): number;
  /**
   * Get number of influence edges
   */
  get_edge_count(): number;
  /**
   * Export to GeoJSON for basin strength visualization
   */
  to_geojson_basin(): any;
  /**
   * Export to GeoJSON for transition risk visualization
   */
  to_geojson_risk(): any;
  /**
   * Export to GeoJSON for influence flow visualization
   */
  to_geojson_influence(): any;
  /**
   * Export to GeoJSON for regime cluster visualization
   */
  to_geojson_regime(): any;
  /**
   * Compare two nations
   */
  compare_nations(code1: string, code2: string): any;
  /**
   * Get nation data as JSON
   */
  get_nation(code: string): any;
  /**
   * Get all nations as JSON array
   */
  get_all_nations(): any;
  /**
   * Get all edges as JSON array
   */
  get_all_edges(): any;
  /**
   * Serialize entire system state
   */
  serialize(): string;
  /**
   * Deserialize system state
   */
  static deserialize(json: string): WasmGeospatialSystem;
}

/**
 * Compute haversine distance between two points
 */
export function wasm_haversine_distance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number;

/**
 * Initialize WASM module
 */
export function init(): void;

/**
 * Analyze Q-matrix (WASM)
 */
export function wasm_analyze_q(q_flat: Float64Array, n: number): any;

/**
 * Build Q-matrix from rates (WASM)
 */
export function wasm_build_q_matrix(rates_flat: Float64Array, n: number): Float64Array;

/**
 * Compute persistence diagram (WASM)
 */
export function wasm_compute_persistence(
  points_flat: Float64Array,
  n_points: number,
  max_edge: number
): any;

/**
 * Compute distance matrix (WASM)
 */
export function wasm_distance_matrix(points_flat: Float64Array, n_points: number): Float64Array;

/**
 * Integrate geodesic on Fisher metric (WASM)
 */
export function wasm_integrate_geodesic_fisher(
  x0: Float64Array,
  v0: Float64Array,
  dt: number,
  n_steps: number
): any;

/**
 * Compute persistent entropy (WASM)
 */
export function wasm_persistent_entropy(births: Float64Array, deaths: Float64Array): number;

/**
 * Simulate Markov chain (WASM)
 */
export function wasm_simulate_markov_chain(
  q_flat: Float64Array,
  n: number,
  r0: number,
  total_time: number,
  dt: number,
  seed: bigint
): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmswarm_free: (a: number, b: number) => void;
  readonly wasm_analyze_q: (a: number, b: number, c: number) => [number, number, number];
  readonly wasm_build_q_matrix: (
    a: number,
    b: number,
    c: number
  ) => [number, number, number, number];
  readonly wasm_compute_persistence: (
    a: number,
    b: number,
    c: number,
    d: number
  ) => [number, number, number];
  readonly wasm_distance_matrix: (a: number, b: number, c: number) => [number, number];
  readonly wasm_integrate_geodesic_fisher: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number
  ) => [number, number, number];
  readonly wasm_persistent_entropy: (a: number, b: number, c: number, d: number) => number;
  readonly wasm_simulate_markov_chain: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: bigint
  ) => [number, number, number];
  readonly wasmswarm_get_metrics: (a: number) => any;
  readonly wasmswarm_get_n_particles: (a: number) => number;
  readonly wasmswarm_get_positions: (a: number) => [number, number];
  readonly wasmswarm_get_time: (a: number) => number;
  readonly wasmswarm_new: (
    a: number,
    b: number,
    c: number,
    d: number,
    e: number,
    f: number,
    g: number,
    h: bigint
  ) => number;
  readonly wasmswarm_run: (a: number, b: number) => any;
  readonly wasmswarm_set_attractor: (a: number, b: number, c: number) => void;
  readonly wasmswarm_step: (a: number) => void;
  readonly init: () => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init(
  module_or_path?:
    | { module_or_path: InitInput | Promise<InitInput> }
    | InitInput
    | Promise<InitInput>
): Promise<InitOutput>;
