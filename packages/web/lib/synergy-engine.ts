/**
 * LatticeForge Synergy Engine (v1.0)
 * Implementation of Riedl-Weidmann (2025) Human-AI Synergy Metrics
 *
 * Core Insight: Collaborative Ability (κ) is orthogonal to Solo Ability (θ).
 * This engine measures κ in real-time and triggers ToM interventions when drift is detected.
 *
 * References:
 * - Riedl, C. & Weidmann, B. (2025). "Quantifying Human-AI Synergy." Under review.
 * - Weidmann, B. & Deming, D.J. (2021). Econometrica, 89(6), 2637-2657. DOI: 10.3982/ECTA18461
 */

// =============================================================================
// TYPES
// =============================================================================

export enum InteractionComplexity {
  LOW = 0.1, // Fact retrieval, simple lookups
  MID = 0.5, // Synthesis, summarization, comparison
  HIGH = 0.9, // Strategic reasoning, code generation, multi-step analysis
}

export interface UserSynergyState {
  userId: string;
  sessionId: string;
  rollingKappa: number[];
  currentToMState: number;
  interactionCount: number;
  lastUpdated: Date;
}

export interface SynergyResult {
  kappaScore: number; // Collaborative ability score (-1 to +1)
  tomState: number; // Current Theory of Mind state (rolling average)
  interventionTrigger: boolean; // True if ToM Primer should activate
  synergyLift: number; // Raw lift over solo baseline
  interventionReason?: string; // Why intervention triggered
}

export interface ToMProxies {
  clarificationRequests: number; // "What do you mean by..."
  perspectiveShifts: number; // "From the model's view..."
  constraintStatements: number; // "Assuming X, the model should..."
  rawDemands: number; // "Just give me the answer"
}

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
  // Baseline stats (calibrate from your user population over time)
  POPULATION_MEAN_SOLO: 0.5,
  POPULATION_MEAN_KAPPA: 0.0,

  // Drift detection thresholds
  DRIFT_THRESHOLD: -1.5, // Z-score drop triggering intervention
  MIN_INTERACTIONS_FOR_DRIFT: 3, // Need at least N interactions to detect drift

  // Smoothing factor for exponential moving average
  EMA_ALPHA: 0.3,

  // Complexity-based expected boost (Insight 6: harder tasks = higher potential lift)
  BASE_EXPECTED_BOOST: 0.15,
  COMPLEXITY_BOOST_FACTOR: 0.3,
};

// =============================================================================
// SYNERGY ENGINE
// =============================================================================

export class SynergyEngine {
  private activeSessions: Map<string, UserSynergyState> = new Map();

  /**
   * Calculate Kappa (Collaborative Ability) orthogonal to Theta (Solo Ability).
   *
   * Key insights from Riedl-Weidmann (2025):
   * - Insight 4: Solo ability (θ) has near-zero correlation with AI output quality
   * - Insight 6: Harder tasks allow for greater synergy expression (ρs = 0.67)
   */
  private calculateKappa(
    soloScore: number,
    jointScore: number,
    complexity: InteractionComplexity
  ): number {
    // Expected baseline boost for an "average" collaborator
    // Deeper tasks offer higher marginal utility for collaboration
    const expectedBoost =
      CONFIG.BASE_EXPECTED_BOOST + CONFIG.COMPLEXITY_BOOST_FACTOR * complexity;

    // The actual lift achieved
    const actualLift = jointScore - soloScore;

    // Synergy Residual (Kappa): Did they extract more than expected?
    const rawKappa = actualLift - expectedBoost;

    // Normalize by complexity (harder to get lift on easy tasks)
    // Dampening factor prevents score explosion on edge cases
    const normalizedKappa = rawKappa / (0.5 + complexity);

    return normalizedKappa;
  }

  /**
   * Detect ToM drift from prompt text patterns.
   * Insight 29: Prompt histories contain diagnostic information about user ToM state.
   */
  detectToMProxies(promptText: string): ToMProxies {
    const lower = promptText.toLowerCase();

    return {
      // High-ToM indicators
      clarificationRequests:
        (lower.match(/what do you mean|clarify|explain what/g) || []).length +
        (lower.match(/i'm assuming|the model (doesn't|does not) know/g) || [])
          .length,

      perspectiveShifts:
        (lower.match(/from (your|the model's|ai's) (perspective|view)/g) || [])
          .length +
        (lower.match(/you might (think|assume|interpret)/g) || []).length,

      constraintStatements:
        (lower.match(/assuming|given that|if we assume|constrain/g) || [])
          .length +
        (lower.match(/do not (assume|hallucinate|make up)/g) || []).length,

      // Low-ToM indicators
      rawDemands:
        (lower.match(/just (give|tell|show) me/g) || []).length +
        (lower.match(/^(what is|who is|when did)/g) || []).length,
    };
  }

  /**
   * Calculate ToM score from proxies.
   * High score = user is actively modeling the AI's perspective.
   */
  calculateToMFromProxies(proxies: ToMProxies): number {
    const positiveSignals =
      proxies.clarificationRequests * 0.3 +
      proxies.perspectiveShifts * 0.4 +
      proxies.constraintStatements * 0.3;

    const negativeSignals = proxies.rawDemands * 0.5;

    // Normalize to -1 to +1 range
    const rawScore = positiveSignals - negativeSignals;
    return Math.max(-1, Math.min(1, rawScore));
  }

  /**
   * Main entry point: Track an interaction and return synergy metrics.
   *
   * @param userId - Unique user identifier
   * @param promptComplexity - Complexity level of the query
   * @param outcomeQualityScore - Quality score of the output (0.0 - 1.0)
   * @param promptText - Optional: raw prompt text for ToM proxy detection
   */
  trackInteraction(
    userId: string,
    promptComplexity: InteractionComplexity,
    outcomeQualityScore: number,
    promptText?: string
  ): SynergyResult {
    // 1. Initialize session if needed
    if (!this.activeSessions.has(userId)) {
      this.activeSessions.set(userId, {
        userId,
        sessionId: `session_${Date.now()}`,
        rollingKappa: [],
        currentToMState: 0.0,
        interactionCount: 0,
        lastUpdated: new Date(),
      });
    }

    const state = this.activeSessions.get(userId)!;

    // 2. Estimate "Solo Score" (Theta)
    // Insight 4: This matters less than we thought, but we baseline it
    const estimatedSolo = CONFIG.POPULATION_MEAN_SOLO;

    // 3. Calculate instantaneous Kappa
    const kappa = this.calculateKappa(
      estimatedSolo,
      outcomeQualityScore,
      promptComplexity
    );

    // 4. Update rolling state (Exponential Moving Average)
    let newToMState: number;
    if (state.rollingKappa.length === 0) {
      newToMState = kappa;
    } else {
      newToMState =
        CONFIG.EMA_ALPHA * kappa +
        (1 - CONFIG.EMA_ALPHA) * state.currentToMState;
    }

    state.rollingKappa.push(kappa);
    state.currentToMState = newToMState;
    state.interactionCount++;
    state.lastUpdated = new Date();

    // 5. Detect drift (Insight 3: Within-user deviation matters)
    let interventionTrigger = false;
    let interventionReason: string | undefined;

    if (state.rollingKappa.length >= CONFIG.MIN_INTERACTIONS_FOR_DRIFT) {
      const sessionMean =
        state.rollingKappa.reduce((a, b) => a + b, 0) / state.rollingKappa.length;
      const sessionStd = Math.sqrt(
        state.rollingKappa
          .map((k) => Math.pow(k - sessionMean, 2))
          .reduce((a, b) => a + b, 0) / state.rollingKappa.length
      );

      const drift = (state.currentToMState - sessionMean) / (sessionStd + 1e-6);

      if (drift < CONFIG.DRIFT_THRESHOLD) {
        interventionTrigger = true;
        interventionReason = `Collaboration drift detected (z=${drift.toFixed(2)}). Your recent interactions show reduced model alignment.`;
      }
    }

    // 6. Check complexity-based intervention (Insight 6)
    if (
      promptComplexity === InteractionComplexity.HIGH &&
      state.interactionCount <= 2
    ) {
      // First high-complexity query in session - always primer
      interventionTrigger = true;
      interventionReason =
        'High-complexity query detected. ToM Primer recommended for optimal output.';
    }

    // 7. Check ToM proxies from prompt text if available
    if (promptText) {
      const proxies = this.detectToMProxies(promptText);
      const tomScore = this.calculateToMFromProxies(proxies);

      if (tomScore < -0.3 && promptComplexity >= InteractionComplexity.MID) {
        interventionTrigger = true;
        interventionReason =
          'Low perspective-taking detected in prompt. Consider adding context about model limitations.';
      }
    }

    return {
      kappaScore: Math.round(kappa * 1000) / 1000,
      tomState: Math.round(newToMState * 1000) / 1000,
      interventionTrigger,
      synergyLift: Math.round((outcomeQualityScore - estimatedSolo) * 1000) / 1000,
      interventionReason,
    };
  }

  /**
   * Get current state for a user (for dashboard display).
   */
  getUserState(userId: string): UserSynergyState | null {
    return this.activeSessions.get(userId) || null;
  }

  /**
   * Reset a user's session (e.g., on logout or session timeout).
   */
  resetSession(userId: string): void {
    this.activeSessions.delete(userId);
  }

  /**
   * Get aggregate stats for analytics.
   */
  getAggregateStats(): {
    activeSessions: number;
    avgKappa: number;
    avgToMState: number;
  } {
    const sessions = Array.from(this.activeSessions.values());
    if (sessions.length === 0) {
      return { activeSessions: 0, avgKappa: 0, avgToMState: 0 };
    }

    const avgKappa =
      sessions.reduce(
        (sum, s) =>
          sum +
          (s.rollingKappa.length > 0
            ? s.rollingKappa[s.rollingKappa.length - 1]
            : 0),
        0
      ) / sessions.length;

    const avgToMState =
      sessions.reduce((sum, s) => sum + s.currentToMState, 0) / sessions.length;

    return {
      activeSessions: sessions.length,
      avgKappa: Math.round(avgKappa * 1000) / 1000,
      avgToMState: Math.round(avgToMState * 1000) / 1000,
    };
  }
}

// =============================================================================
// SINGLETON INSTANCE
// =============================================================================

export const synergyEngine = new SynergyEngine();

// =============================================================================
// UTILITY: Classify prompt complexity
// =============================================================================

export function classifyComplexity(prompt: string): InteractionComplexity {
  const lower = prompt.toLowerCase();
  const wordCount = prompt.split(/\s+/).length;

  // High complexity indicators
  const highIndicators = [
    /strateg/i,
    /analyz/i,
    /compar.*and.*contrast/i,
    /implication/i,
    /synthesiz/i,
    /evaluat/i,
    /predict/i,
    /assess.*risk/i,
    /code|implement|build/i,
    /multi.*step/i,
  ];

  // Low complexity indicators
  const lowIndicators = [
    /^what is/i,
    /^who is/i,
    /^when did/i,
    /^where is/i,
    /define/i,
    /^list/i,
  ];

  const highScore = highIndicators.filter((r) => r.test(lower)).length;
  const lowScore = lowIndicators.filter((r) => r.test(lower)).length;

  // Long prompts with context tend to be more complex
  const lengthBonus = wordCount > 100 ? 0.3 : wordCount > 50 ? 0.15 : 0;

  const complexityScore = highScore * 0.2 - lowScore * 0.3 + lengthBonus;

  if (complexityScore > 0.3) return InteractionComplexity.HIGH;
  if (complexityScore < -0.1) return InteractionComplexity.LOW;
  return InteractionComplexity.MID;
}
