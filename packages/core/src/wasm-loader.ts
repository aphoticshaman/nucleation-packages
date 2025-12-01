/**
 * WASM module loader with proper initialization for Node.js and browser environments.
 *
 * This module handles the complexity of loading WASM in different environments:
 * - Node.js: Reads WASM file from disk
 * - Browser: Uses fetch to load WASM
 * - Edge workers: Uses bundled WASM
 */

import { createRequire } from 'node:module';
import { readFile } from 'node:fs/promises';

// Type definitions for the WASM module
interface NucleationWasmModule {
  default: (wasmBytes?: Uint8Array) => Promise<void>;
  NucleationDetector: NucleationDetectorClass;
  DetectorConfig: DetectorConfigClass;
  Shepherd: ShepherdClass;
  Phase: PhaseEnum;
}

interface NucleationDetectorClass {
  new (config: InstanceType<DetectorConfigClass>): NucleationDetectorInstance;
  deserialize(json: string): NucleationDetectorInstance;
}

interface NucleationDetectorInstance {
  update(value: number): number;
  update_batch(values: Float64Array): number;
  currentPhase(): number;
  confidence(): number;
  currentVariance(): number;
  inflectionMagnitude(): number;
  count(): number;
  reset(): void;
  serialize(): string;
}

interface DetectorConfigClass {
  new (): DetectorConfigInstance;
  conservative(): DetectorConfigInstance;
  sensitive(): DetectorConfigInstance;
}

interface DetectorConfigInstance {
  window_size: number;
  threshold: number;
}

interface ShepherdClass {
  new (categories: number): ShepherdInstance;
}

interface ShepherdInstance {
  registerActor(id: string, initialProfile: Float64Array | null): void;
  updateActor(id: string, observation: Float64Array, timestamp: number): unknown[];
  conflictPotential(idA: string, idB: string): number | undefined;
  checkAllDyads(timestamp: number): unknown[];
}

interface PhaseEnum {
  Stable: number;
  Approaching: number;
  Critical: number;
  Transitioning: number;
}

// Module state
let wasmModule: NucleationWasmModule | null = null;
let initPromise: Promise<void> | null = null;
let initialized = false;

/**
 * Initialize the WASM module. Safe to call multiple times.
 *
 * @returns Promise that resolves when initialization is complete
 * @throws Error if WASM module cannot be loaded
 */
export async function initialize(): Promise<void> {
  if (initialized) {
    return;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = doInitialize();
  return initPromise;
}

async function doInitialize(): Promise<void> {
  try {
    // Dynamic import of the WASM module
    const imported = await import('nucleation-wasm');
    wasmModule = imported as unknown as NucleationWasmModule;

    // Node.js environment - load WASM bytes directly
    const g = globalThis as { window?: unknown; Deno?: unknown };
    if (typeof g.window === 'undefined' && typeof g.Deno === 'undefined') {
      const require = createRequire(import.meta.url);
      const wasmPath = require.resolve('nucleation-wasm/nucleation_bg.wasm');
      const wasmBytes = await readFile(wasmPath);
      await wasmModule.default(new Uint8Array(wasmBytes));
    } else {
      // Browser or Deno - let the module handle it
      await wasmModule.default();
    }

    initialized = true;
  } catch (error) {
    initPromise = null;
    throw new Error(
      `Failed to initialize nucleation WASM module: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Check if the WASM module is initialized
 */
export function isInitialized(): boolean {
  return initialized;
}

/**
 * Get the WASM module. Throws if not initialized.
 *
 * @returns The initialized WASM module
 * @throws Error if module is not initialized
 */
export function getModule(): NucleationWasmModule {
  if (!initialized || !wasmModule) {
    throw new Error('Nucleation WASM module not initialized. Call initialize() first.');
  }
  return wasmModule;
}

// Export types for use in other modules
export type {
  NucleationWasmModule,
  NucleationDetectorInstance,
  DetectorConfigInstance,
  ShepherdInstance,
  PhaseEnum,
};
