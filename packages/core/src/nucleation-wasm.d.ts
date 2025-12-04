/**
 * Type stub for nucleation-wasm module.
 * The actual implementation is internalized - this just satisfies TypeScript.
 * Dynamic imports will fail gracefully and fall back to JS implementation.
 */
declare module 'nucleation-wasm' {
  export interface DetectorConfig {
    window_size: number;
    threshold: number;
    smoothing_window?: number;
  }

  export interface NucleationDetector {
    update(value: number): number;
    update_batch(values: Float64Array): number;
    currentPhase(): number;
    confidence(): number;
    currentVariance(): number;
    inflectionMagnitude(): number;
    count(): number;
    reset(): void;
    serialize(): string;
    free(): void;
  }

  export interface NucleationDetectorStatic {
    new (config: DetectorConfig): NucleationDetector;
    with_defaults(): NucleationDetector;
    deserialize(json: string): NucleationDetector;
  }

  export interface DetectorConfigStatic {
    new (): DetectorConfig;
    conservative(): DetectorConfig;
    sensitive(): DetectorConfig;
  }

  export interface Shepherd {
    registerActor(id: string, distribution?: Float64Array): void;
    updateActor(id: string, observation: Float64Array, timestamp: number): unknown[];
    conflictPotential(a: string, b: string): number | undefined;
    checkAllDyads(timestamp: number): unknown[];
    actors(): unknown[];
    free(): void;
  }

  export interface ShepherdStatic {
    new (n_categories: number): Shepherd;
  }

  export const NucleationDetector: NucleationDetectorStatic;
  export const DetectorConfig: DetectorConfigStatic;
  export const Shepherd: ShepherdStatic;

  export const Phase: {
    Stable: number;
    Approaching: number;
    Critical: number;
    Transitioning: number;
  };

  export function version(): string;

  export default function init(wasmBytes?: Uint8Array): Promise<void>;
}
