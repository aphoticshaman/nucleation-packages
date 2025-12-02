/**
 * Cascade Predictor
 *
 * Proprietary algorithm for predicting viral/cascade events
 * across social and market networks.
 *
 * Models information cascades using epidemiological SIR dynamics
 * adapted for attention/sentiment propagation.
 *
 * © 2025 Crystalline Labs LLC - Trade Secret
 */

export interface CascadeSignature {
  /** Unique fingerprint of cascade type */
  id: string;
  /** Human-readable name */
  name: string;
  /** Domain: social, market, news, regulatory */
  domain: 'social' | 'market' | 'news' | 'regulatory' | 'cross-domain';
  /** Typical duration in hours */
  duration: number;
  /** Peak intensity reached */
  peakIntensity: number;
  /** Shape parameters for matching */
  shape: {
    riseRate: number;
    peakPosition: number; // 0-1, where in lifecycle peak occurs
    decayRate: number;
    asymmetry: number; // >1 = fast rise slow decay, <1 = slow rise fast decay
  };
}

export interface CascadePrediction {
  probability: number;
  estimatedPeak: Date;
  estimatedIntensity: number;
  matchedSignature: CascadeSignature | null;
  triggerSignals: string[];
  affectedDomains: string[];
  confidence: number;
}

export interface CascadeState {
  phase: 'dormant' | 'seeding' | 'spreading' | 'peak' | 'declining' | 'exhausted';
  currentIntensity: number;
  velocity: number; // Rate of change
  acceleration: number; // Second derivative
  susceptiblePool: number; // Remaining "population" that can be affected
  recoveredPool: number; // Those who've already reacted
}

/**
 * Cascade Predictor
 *
 * SECRET SAUCE:
 * - Library of historical cascade signatures for pattern matching
 * - SIR epidemiological model adapted for information spread
 * - Multi-domain contagion tracking
 * - Early warning indicators derived from network topology proxies
 */
export class CascadePredictor {
  // Proprietary cascade signatures (derived from historical analysis)
  private static readonly SIGNATURES: CascadeSignature[] = [
    {
      id: 'flash-crash',
      name: 'Flash Crash Pattern',
      domain: 'market',
      duration: 0.5, // 30 minutes
      peakIntensity: 0.95,
      shape: { riseRate: 15, peakPosition: 0.1, decayRate: 3, asymmetry: 5 },
    },
    {
      id: 'meme-stock',
      name: 'Meme Stock Rally',
      domain: 'cross-domain',
      duration: 72, // 3 days
      peakIntensity: 0.85,
      shape: { riseRate: 2.5, peakPosition: 0.6, decayRate: 1.5, asymmetry: 1.8 },
    },
    {
      id: 'news-shock',
      name: 'Breaking News Shock',
      domain: 'news',
      duration: 24,
      peakIntensity: 0.75,
      shape: { riseRate: 8, peakPosition: 0.15, decayRate: 0.8, asymmetry: 10 },
    },
    {
      id: 'regulatory-bomb',
      name: 'Regulatory Announcement',
      domain: 'regulatory',
      duration: 168, // 1 week
      peakIntensity: 0.7,
      shape: { riseRate: 5, peakPosition: 0.05, decayRate: 0.3, asymmetry: 15 },
    },
    {
      id: 'viral-social',
      name: 'Viral Social Post',
      domain: 'social',
      duration: 48,
      peakIntensity: 0.6,
      shape: { riseRate: 3, peakPosition: 0.3, decayRate: 1.2, asymmetry: 2.5 },
    },
    {
      id: 'coordinated-pump',
      name: 'Coordinated Pump',
      domain: 'cross-domain',
      duration: 6,
      peakIntensity: 0.9,
      shape: { riseRate: 12, peakPosition: 0.7, decayRate: 8, asymmetry: 0.6 },
    },
    {
      id: 'slow-burn',
      name: 'Slow Burn Trend',
      domain: 'social',
      duration: 336, // 2 weeks
      peakIntensity: 0.5,
      shape: { riseRate: 0.5, peakPosition: 0.8, decayRate: 0.3, asymmetry: 0.6 },
    },
    {
      id: 'earnings-surprise',
      name: 'Earnings Surprise',
      domain: 'market',
      duration: 48,
      peakIntensity: 0.65,
      shape: { riseRate: 10, peakPosition: 0.08, decayRate: 0.6, asymmetry: 12 },
    },
  ];

  // SIR model parameters (proprietary calibration)
  private static readonly SIR_PARAMS = {
    baseTransmission: 0.3, // Beta: base infection rate
    recoveryRate: 0.1, // Gamma: recovery rate
    networkDensity: 0.7, // How connected the population is
    attentionDecay: 0.05, // How fast attention fades
  };

  // Early warning thresholds
  private static readonly WARNING_THRESHOLDS = {
    velocitySpike: 2.5, // Std devs above mean
    accelerationSpike: 3.0,
    correlationSurge: 0.7, // Cross-domain correlation
    volumeAnomaly: 2.0,
  };

  private history: CascadeState[] = [];
  private domainSignals: Map<string, number[]> = new Map();

  /**
   * Analyze current cascade state from multi-domain signals
   */
  analyzeState(signals: Map<string, number[]>): CascadeState {
    this.domainSignals = signals;

    // Aggregate signal to single intensity metric
    const intensity = this.calculateIntensity(signals);
    const velocity = this.calculateVelocity(signals);
    const acceleration = this.calculateAcceleration(signals);

    // Estimate SIR pools
    const { susceptible, recovered } = this.estimateSirPools(intensity);

    // Determine phase
    const phase = this.classifyPhase(intensity, velocity, acceleration);

    const state: CascadeState = {
      phase,
      currentIntensity: intensity,
      velocity,
      acceleration,
      susceptiblePool: susceptible,
      recoveredPool: recovered,
    };

    this.history.push(state);
    if (this.history.length > 200) {
      this.history.shift();
    }

    return state;
  }

  /**
   * Predict upcoming cascades
   */
  predict(signals: Map<string, number[]>): CascadePrediction {
    const state = this.analyzeState(signals);
    const warnings = this.checkEarlyWarnings(signals);
    const matchedSignature = this.matchSignature(signals);

    // Calculate cascade probability
    const probability = this.calculateCascadeProbability(state, warnings);

    // Estimate peak timing if cascade likely
    const estimatedPeak = this.estimatePeakTiming(state, matchedSignature);

    // Estimate intensity
    const estimatedIntensity = this.estimateIntensity(state, matchedSignature);

    // Identify triggers
    const triggerSignals = this.identifyTriggers(signals);

    // Identify affected domains
    const affectedDomains = this.predictAffectedDomains(signals, matchedSignature);

    // Calculate confidence
    const confidence = this.calculateConfidence(state, matchedSignature, warnings);

    return {
      probability,
      estimatedPeak,
      estimatedIntensity,
      matchedSignature,
      triggerSignals,
      affectedDomains,
      confidence,
    };
  }

  /**
   * Run SIR simulation forward
   */
  simulateSir(initialInfected: number, steps: number, transmissionMod = 1): number[] {
    const { baseTransmission, recoveryRate, networkDensity } = CascadePredictor.SIR_PARAMS;

    let S = 1 - initialInfected; // Susceptible
    let I = initialInfected; // Infected (active spreaders)
    let R = 0; // Recovered (already reacted)

    const trajectory: number[] = [I];
    const beta = baseTransmission * transmissionMod * networkDensity;
    const gamma = recoveryRate;

    for (let t = 0; t < steps; t++) {
      const newInfected = beta * S * I;
      const newRecovered = gamma * I;

      S -= newInfected;
      I += newInfected - newRecovered;
      R += newRecovered;

      // Clamp values
      S = Math.max(0, Math.min(1, S));
      I = Math.max(0, Math.min(1, I));
      R = Math.max(0, Math.min(1, R));

      trajectory.push(I);
    }

    return trajectory;
  }

  /**
   * Calculate aggregate intensity from signals
   */
  private calculateIntensity(signals: Map<string, number[]>): number {
    let totalIntensity = 0;
    let count = 0;

    for (const [, values] of signals) {
      if (values.length > 0) {
        // Normalize to 0-1 based on recent range
        const recent = values.slice(-20);
        const max = Math.max(...recent);
        const min = Math.min(...recent);
        const range = max - min || 1;
        const current = values[values.length - 1];
        const normalized = (current - min) / range;
        totalIntensity += normalized;
        count++;
      }
    }

    return count > 0 ? totalIntensity / count : 0;
  }

  /**
   * Calculate velocity (first derivative) of intensity
   */
  private calculateVelocity(_signals: Map<string, number[]>): number {
    if (this.history.length < 2) return 0;

    const recent = this.history.slice(-5);
    if (recent.length < 2) return 0;

    // Simple linear regression slope
    let sumX = 0,
      sumY = 0,
      sumXY = 0,
      sumX2 = 0;
    for (let i = 0; i < recent.length; i++) {
      sumX += i;
      sumY += recent[i].currentIntensity;
      sumXY += i * recent[i].currentIntensity;
      sumX2 += i * i;
    }
    const n = recent.length;
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    return isNaN(slope) ? 0 : slope;
  }

  /**
   * Calculate acceleration (second derivative)
   */
  private calculateAcceleration(_signals: Map<string, number[]>): number {
    if (this.history.length < 3) return 0;

    const recent = this.history.slice(-5);
    if (recent.length < 3) return 0;

    // Finite difference approximation
    const v1 = recent[recent.length - 2].velocity;
    const v2 = recent[recent.length - 1].velocity;

    return v2 - v1;
  }

  /**
   * Estimate SIR model pools from current intensity
   */
  private estimateSirPools(intensity: number): {
    susceptible: number;
    recovered: number;
  } {
    // Rough estimation based on cascade lifecycle
    // Early: high S, low R
    // Peak: moderate S and R
    // Late: low S, high R

    const { attentionDecay } = CascadePredictor.SIR_PARAMS;

    // Estimate from recent intensity trend
    let recovered = 0;
    for (const state of this.history.slice(-20)) {
      recovered += state.currentIntensity * attentionDecay;
    }
    recovered = Math.min(0.95, recovered);

    const susceptible = Math.max(0.05, 1 - intensity - recovered);

    return { susceptible, recovered };
  }

  /**
   * Classify cascade phase
   */
  private classifyPhase(
    intensity: number,
    velocity: number,
    acceleration: number
  ): CascadeState['phase'] {
    // Dormant: low everything
    if (intensity < 0.1 && Math.abs(velocity) < 0.05) {
      return 'dormant';
    }

    // Seeding: low intensity, positive velocity, positive acceleration
    if (intensity < 0.3 && velocity > 0.02 && acceleration > 0) {
      return 'seeding';
    }

    // Spreading: moderate intensity, high velocity
    if (intensity >= 0.3 && intensity < 0.7 && velocity > 0.05) {
      return 'spreading';
    }

    // Peak: high intensity, velocity near zero or turning negative
    if (intensity >= 0.7 || (intensity >= 0.5 && velocity < 0.02 && acceleration < 0)) {
      return 'peak';
    }

    // Declining: moderate intensity, negative velocity
    if (intensity >= 0.2 && velocity < -0.02) {
      return 'declining';
    }

    // Exhausted: low intensity after having been higher
    if (
      intensity < 0.2 &&
      this.history.length > 10 &&
      Math.max(...this.history.slice(-10).map((h) => h.currentIntensity)) > 0.5
    ) {
      return 'exhausted';
    }

    return 'dormant';
  }

  /**
   * Check early warning indicators
   */
  private checkEarlyWarnings(signals: Map<string, number[]>): Map<string, boolean> {
    const warnings = new Map<string, boolean>();

    // Velocity spike
    if (this.history.length > 20) {
      const velocities = this.history.slice(-20).map((h) => h.velocity);
      const mean = velocities.reduce((a, b) => a + b, 0) / velocities.length;
      const std = Math.sqrt(
        velocities.reduce((sum, v) => sum + (v - mean) ** 2, 0) / velocities.length
      );
      const current = this.history[this.history.length - 1]?.velocity ?? 0;

      warnings.set(
        'velocitySpike',
        current > mean + CascadePredictor.WARNING_THRESHOLDS.velocitySpike * std
      );
    }

    // Cross-domain correlation surge
    const domains = Array.from(signals.keys());
    if (domains.length >= 2) {
      let totalCorr = 0;
      let pairs = 0;
      for (let i = 0; i < domains.length; i++) {
        for (let j = i + 1; j < domains.length; j++) {
          const corr = this.correlation(signals.get(domains[i])!, signals.get(domains[j])!);
          totalCorr += Math.abs(corr);
          pairs++;
        }
      }
      const avgCorr = pairs > 0 ? totalCorr / pairs : 0;
      warnings.set(
        'correlationSurge',
        avgCorr > CascadePredictor.WARNING_THRESHOLDS.correlationSurge
      );
    }

    return warnings;
  }

  /**
   * Match current pattern to known signatures
   */
  private matchSignature(_signals: Map<string, number[]>): CascadeSignature | null {
    if (this.history.length < 10) return null;

    const recent = this.history.slice(-20);
    const intensities = recent.map((h) => h.currentIntensity);

    // Calculate shape metrics
    const maxIdx = intensities.indexOf(Math.max(...intensities));
    const peakPosition = maxIdx / intensities.length;

    // Rise rate (slope to peak)
    const riseSlice = intensities.slice(0, maxIdx + 1);
    const riseRate =
      riseSlice.length > 1
        ? (riseSlice[riseSlice.length - 1] - riseSlice[0]) / riseSlice.length
        : 0;

    // Decay rate (slope after peak)
    const decaySlice = intensities.slice(maxIdx);
    const decayRate =
      decaySlice.length > 1
        ? Math.abs(decaySlice[decaySlice.length - 1] - decaySlice[0]) / decaySlice.length
        : 0;

    // Match against signatures
    let bestMatch: CascadeSignature | null = null;
    let bestScore = 0;

    for (const sig of CascadePredictor.SIGNATURES) {
      const peakDiff = Math.abs(peakPosition - sig.shape.peakPosition);
      const riseDiff = Math.abs(riseRate * 10 - sig.shape.riseRate) / sig.shape.riseRate;
      const decayDiff = Math.abs(decayRate * 10 - sig.shape.decayRate) / sig.shape.decayRate;

      // Score is inverse of total difference
      const score = 1 / (1 + peakDiff + riseDiff + decayDiff);

      if (score > bestScore && score > 0.3) {
        bestScore = score;
        bestMatch = sig;
      }
    }

    return bestMatch;
  }

  /**
   * Calculate cascade probability
   */
  private calculateCascadeProbability(state: CascadeState, warnings: Map<string, boolean>): number {
    let probability = 0;

    // Base probability from state
    switch (state.phase) {
      case 'dormant':
        probability = 0.05;
        break;
      case 'seeding':
        probability = 0.35;
        break;
      case 'spreading':
        probability = 0.75;
        break;
      case 'peak':
        probability = 0.95;
        break;
      case 'declining':
        probability = 0.3;
        break;
      case 'exhausted':
        probability = 0.05;
        break;
    }

    // Boost for warnings
    if (warnings.get('velocitySpike')) probability += 0.15;
    if (warnings.get('correlationSurge')) probability += 0.2;

    // Boost for high susceptible pool
    if (state.susceptiblePool > 0.7) probability += 0.1;

    return Math.min(1, probability);
  }

  /**
   * Estimate peak timing
   */
  private estimatePeakTiming(state: CascadeState, signature: CascadeSignature | null): Date {
    const now = new Date();

    if (state.phase === 'peak') {
      return now;
    }

    // Use signature duration if matched
    if (signature) {
      const hoursToSeak = signature.duration * signature.shape.peakPosition;
      return new Date(now.getTime() + hoursToSeak * 60 * 60 * 1000);
    }

    // Default estimate based on velocity
    if (state.velocity > 0) {
      const hoursToSeak = Math.max(1, (0.8 - state.currentIntensity) / state.velocity);
      return new Date(now.getTime() + hoursToSeak * 60 * 60 * 1000);
    }

    // Already past peak
    return now;
  }

  /**
   * Estimate cascade intensity
   */
  private estimateIntensity(state: CascadeState, signature: CascadeSignature | null): number {
    if (signature) {
      return signature.peakIntensity;
    }

    // Estimate from current trajectory
    if (state.velocity > 0) {
      // Project forward
      return Math.min(1, state.currentIntensity + state.velocity * 10);
    }

    return state.currentIntensity;
  }

  /**
   * Identify trigger signals
   */
  private identifyTriggers(signals: Map<string, number[]>): string[] {
    const triggers: string[] = [];

    for (const [name, values] of signals) {
      if (values.length < 5) continue;

      const recent = values.slice(-5);
      const earlier = values.slice(-10, -5);

      if (earlier.length === 0) continue;

      const recentMean = recent.reduce((a, b) => a + b, 0) / recent.length;
      const earlierMean = earlier.reduce((a, b) => a + b, 0) / earlier.length;

      // Significant increase = trigger
      if (recentMean > earlierMean * 1.5) {
        triggers.push(name);
      }
    }

    return triggers;
  }

  /**
   * Predict affected domains
   */
  private predictAffectedDomains(
    signals: Map<string, number[]>,
    signature: CascadeSignature | null
  ): string[] {
    if (signature?.domain === 'cross-domain') {
      return ['market', 'social', 'news'];
    }

    if (signature?.domain) {
      return [signature.domain];
    }

    // Return domains with highest recent activity
    const activity: Array<{ domain: string; activity: number }> = [];

    for (const [name, values] of signals) {
      if (values.length < 2) continue;
      const change = Math.abs(values[values.length - 1] - values[values.length - 2]);
      activity.push({ domain: name, activity: change });
    }

    return activity
      .sort((a, b) => b.activity - a.activity)
      .slice(0, 3)
      .map((a) => a.domain);
  }

  /**
   * Calculate prediction confidence
   */
  private calculateConfidence(
    state: CascadeState,
    signature: CascadeSignature | null,
    warnings: Map<string, boolean>
  ): number {
    let confidence = 0.5;

    // Signature match increases confidence
    if (signature) confidence += 0.2;

    // Clear phase increases confidence
    if (state.phase !== 'dormant') confidence += 0.1;

    // Consistent warnings increase confidence
    const warningCount = Array.from(warnings.values()).filter(Boolean).length;
    confidence += warningCount * 0.1;

    // History length affects confidence
    if (this.history.length > 50) confidence += 0.1;

    return Math.min(1, confidence);
  }

  private correlation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 2) return 0;

    const xSlice = x.slice(-n);
    const ySlice = y.slice(-n);

    const meanX = xSlice.reduce((a, b) => a + b, 0) / n;
    const meanY = ySlice.reduce((a, b) => a + b, 0) / n;

    let num = 0,
      denX = 0,
      denY = 0;
    for (let i = 0; i < n; i++) {
      const dx = xSlice[i] - meanX;
      const dy = ySlice[i] - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }

    const den = Math.sqrt(denX * denY);
    return den === 0 ? 0 : num / den;
  }
}
