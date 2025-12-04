// WASM module loader - STUB (nucleation-wasm removed)
// This file provides type stubs for any code that imports from here

export interface WasmCore {
  available: boolean;
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
  [key: string]: unknown;
}

// Stub - always returns null since WASM was removed
export async function loadWasm(): Promise<WasmCore> {
  return { available: false };
}

export function isWasmLoaded(): boolean {
  return false;
}
