/**
 * Social Pulse Data Sources
 *
 * Available sources for sentiment aggregation.
 */

export { BlueskySource } from './bluesky.js';
export { TelegramSource } from './telegram.js';
export { GdeltSource } from './gdelt.js';
export { ArxivSource } from './arxiv.js';
export { InstitutionalFlowSource } from './institutional-flow.js';
export { EconomicIndicatorsSource } from './economic-indicators.js';

// Re-export types
export type {
  IndicatorType,
  IndicatorObservation,
  FreedomDataPoint,
  TrajectoryAnalysis,
} from './economic-indicators.js';
