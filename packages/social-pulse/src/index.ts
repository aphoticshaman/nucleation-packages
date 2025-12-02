/**
 * social-pulse
 *
 * Social media sentiment aggregator for detecting upheaval precursors.
 *
 * Features:
 * - Multi-platform data collection (Bluesky, Telegram, X, GDELT, etc.)
 * - Bot detection and filtering
 * - Language and geolocation inference
 * - Fact-checking via cross-referencer
 * - Phase transition detection (using nucleation-wasm)
 * - Earnings sentiment tracking
 * - Economic indicators (employment, housing, education, freedom indices)
 * - Institutional flow tracking
 *
 * @packageDocumentation
 */

// Main detector
export { SocialPulseDetector, type SocialPulseConfig } from './detector.js';

// Dual-Fusion Engine (asymmetric leverage)
export { DualFusionEngine } from './detector.js';

export type {
  ExternalAPIConfig,
  FusionResult,
  StreamCallback,
  PerformanceMetrics,
  TraceEntry,
  DataTrace,
  WasmBridgeStatus,
} from './detector.js';

// Types
export type {
  SocialPost,
  Platform,
  AuthorInfo,
  EngagementMetrics,
  GeoInfo,
  SourceConfig,
  SearchParams,
  SentimentAggregate,
  UpheavalState,
  UpheavalLevel,
  DataSource,
  PostFilter,
} from './types.js';

// Sources
export {
  BlueskySource,
  TelegramSource,
  GdeltSource,
  ArxivSource,
  InstitutionalFlowSource,
  EconomicIndicatorsSource,
} from './sources/index.js';

export type {
  IndicatorType,
  IndicatorObservation,
  FreedomDataPoint,
  TrajectoryAnalysis,
} from './sources/index.js';

// Filters
export { BotFilter, LanguageDetector, GeolocationFilter } from './filters/index.js';

export type {
  BotFilterConfig,
  BotSignal,
  LanguageDetectorConfig,
  GeolocationConfig,
} from './filters/index.js';

// Validators
export { CrossReferencer } from './validators/index.js';

export type {
  CrossReferencerConfig,
  VerificationResult,
  FactCheckResult,
  RedFlag,
  RedFlagType,
} from './validators/index.js';
