/**
 * ECAP - Entangled Co-Adaptive Protocol
 *
 * A real-time bi-directional adaptation system for human-AI co-evolution.
 *
 * @example
 * ```typescript
 * import { ECAPEngine, createCognitiveStateFromTelemetry, createAIStateFromModel } from '@/lib/ecap';
 *
 * const engine = new ECAPEngine({ targetCorrelation: 0.8 });
 *
 * const human = createCognitiveStateFromTelemetry({
 *   keystrokeLatencies: [120, 150, 100, 180],
 *   errorCount: 2,
 *   sessionDuration: 300,
 * });
 *
 * const ai = createAIStateFromModel({ version: 'v1.0.0' });
 *
 * const result = engine.processInteraction(human, ai);
 * console.log(`Correlation: ${result.correlation.toFixed(3)}`);
 * console.log(`Adaptation rate: ${result.adaptationRate.toFixed(3)}`);
 * ```
 */

export * from './types';
export {
  ECAPEngine,
  createCognitiveStateFromTelemetry,
  createAIStateFromModel,
} from './engine';
