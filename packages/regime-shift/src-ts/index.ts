/**
 * regime-shift
 *
 * Detect market regime changes before they happen using variance-based
 * phase transition detection. Built on @nucleation/core.
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import { RegimeDetector } from 'regime-shift';
 *
 * const detector = new RegimeDetector();
 * await detector.init();
 *
 * for (const price of priceHistory) {
 *   const state = detector.update(price);
 *   if (state.isShifting) {
 *     console.log(`Regime shift detected! Confidence: ${state.confidence}`);
 *   }
 * }
 * ```
 */

import {
  BaseDetector,
  type DetectorConfig,
  type DetectorState,
  type LevelMapping,
  initialize,
  getModule,
} from '@nucleation/core';

/**
 * Market regime levels
 */
export type RegimeLevel = 'stable' | 'warming' | 'critical' | 'shifting';

/**
 * Regime state with market-specific properties
 */
export interface RegimeState extends DetectorState<RegimeLevel> {
  /** Alias for level */
  regime: RegimeLevel;
  /** True if regime change detected */
  isShifting: boolean;
  /** True if approaching regime change */
  isWarning: boolean;
}

/**
 * Market regime levels as constants
 */
export const Regime = {
  STABLE: 'stable',
  WARMING: 'warming',
  CRITICAL: 'critical',
  SHIFTING: 'shifting',
} as const;

/**
 * Market regime change detector.
 *
 * Uses variance inflection detection to identify regime changes
 * before they fully manifest. The core insight: variance typically
 * *decreases* before major market transitions (the "calm before the storm").
 *
 * @example
 * ```typescript
 * // Basic usage with price data
 * const detector = new RegimeDetector();
 * await detector.init();
 *
 * prices.forEach(price => {
 *   const state = detector.update(price);
 *   console.log(`Regime: ${state.regime}, Confidence: ${state.confidence}`);
 * });
 * ```
 *
 * @example
 * ```typescript
 * // Using returns instead of prices
 * const detector = new RegimeDetector({ sensitivity: 'sensitive' });
 * await detector.init();
 *
 * returns.forEach(ret => {
 *   const state = detector.update(ret);
 *   if (state.isWarning) {
 *     console.log('Potential regime shift approaching');
 *   }
 * });
 * ```
 */
export class RegimeDetector extends BaseDetector<RegimeLevel, RegimeState> {
  protected readonly levelMapping: LevelMapping<RegimeLevel> = {
    stable: 'stable',
    approaching: 'warming',
    critical: 'critical',
    transitioning: 'shifting',
  };

  /**
   * Finance-specific default: 30-day window for monthly patterns
   */
  protected override getDefaultWindowSize(): number {
    return 30;
  }

  /**
   * Create domain-specific state with market terminology
   */
  protected override createState(baseState: DetectorState<RegimeLevel>): RegimeState {
    return {
      ...baseState,
      regime: baseState.level,
      isShifting: baseState.levelNumeric >= 3,
      isWarning: baseState.levelNumeric >= 2,
    };
  }

  /**
   * Create a detector from serialized state.
   *
   * @param json - Serialized detector state
   * @returns Restored detector
   */
  static async deserialize(json: string): Promise<RegimeDetector> {
    await initialize();
    const module = getModule();
    const { NucleationDetector } = module;

    const detector = new RegimeDetector();
    // Access the protected property using type assertion
    (detector as unknown as { detector: unknown }).detector = NucleationDetector.deserialize(json);
    (detector as unknown as { initialized: boolean }).initialized = true;

    return detector;
  }
}

/**
 * Quick check if a price series shows regime shift signals.
 * Convenience function for one-off analysis.
 *
 * @param prices - Price series to analyze
 * @param config - Detection configuration
 * @returns Analysis result
 *
 * @example
 * ```typescript
 * import { detectRegimeShift } from 'regime-shift';
 *
 * const result = await detectRegimeShift(closingPrices);
 * if (result.shifting) {
 *   console.log('Regime shift detected!');
 * }
 * ```
 */
export async function detectRegimeShift(
  prices: number[],
  config: DetectorConfig = {}
): Promise<{
  shifting: boolean;
  warning: boolean;
  regime: RegimeLevel;
  confidence: number;
}> {
  const detector = new RegimeDetector(config);
  await detector.init();

  const state = detector.updateBatch(prices);

  return {
    shifting: state.isShifting,
    warning: state.isWarning,
    regime: state.regime,
    confidence: state.confidence,
  };
}

// Re-export core utilities for convenience
export { initialize } from '@nucleation/core';
export type { DetectorConfig } from '@nucleation/core';

// Default export
export default RegimeDetector;
