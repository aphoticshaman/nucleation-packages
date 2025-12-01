/**
 * @nucleation/core
 *
 * Shared infrastructure for nucleation phase transition detectors.
 * Provides WASM initialization, base classes, and common utilities.
 *
 * @packageDocumentation
 */

export { initialize, isInitialized, getModule } from './wasm-loader.js';
export { BaseDetector } from './base-detector.js';
export { BaseCorrelator } from './base-correlator.js';
export {
  type DetectorConfig,
  type DetectorState,
  type Phase,
  type Sensitivity,
  PHASE,
  DEFAULT_CONFIG,
} from './types.js';
export { validateNumber, validateConfig, NucleationError } from './validation.js';
export {
  createSecureWebhookServer,
  type WebhookAuth,
  type RateLimitConfig,
  type WebhookServerConfig,
} from './webhook-server.js';

// Re-export types from base detector for convenience
export type { LevelMapping } from './base-detector.js';
