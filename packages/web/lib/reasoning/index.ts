/**
 * LATTICE REASONING MODULE
 *
 * The "sleeping brain" that learns from every interaction.
 *
 * Components:
 * - Orchestrator: Routes queries to reasoning engines
 * - Learner: Captures data for future training
 * - Security: Protects API and prevents abuse
 *
 * Usage:
 * ```typescript
 * import { getReasoningOrchestrator, getLearningCollector, getSecurityGuardian } from '@/lib/reasoning';
 *
 * const orchestrator = getReasoningOrchestrator();
 * const learner = getLearningCollector();
 * const security = getSecurityGuardian();
 * ```
 */

export {
  ReasoningOrchestrator,
  getReasoningOrchestrator,
  type ReasoningQuery,
  type ReasoningResult,
  type ReasoningStep,
  type CausalFactor,
  type HistoricalCase,
  type UncertaintyBounds,
} from './orchestrator';

export {
  LearningCollector,
  TrainingDataExporter,
  getLearningCollector,
  getTrainingDataExporter,
  type LearningEventType,
  type LearningEvent,
  type AnonymizedData,
  type EventMetadata,
} from './learner';

export {
  SecurityGuardian,
  APIKeyGuardian,
  getSecurityGuardian,
  type SecurityResult,
  type RateLimitResult,
} from './security';
