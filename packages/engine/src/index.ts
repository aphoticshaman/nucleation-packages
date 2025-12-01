/**
 * @nucleation/engine
 *
 * Detection engine with WASM acceleration and data provenance tracking.
 */

export { FusionEngine } from './fusion-engine.js';
export { WasmBridge } from './wasm-bridge.js';
export { DataTraceRecorder } from './trace-recorder.js';

export type {
  DataProvenance,
  TraceEntry,
  DataTrace,
  ExternalAPIConfig,
  FusionResult,
  WasmStatus,
  EngineConfig,
  PerformanceMetrics,
} from './types.js';
