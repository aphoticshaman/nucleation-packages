/**
 * Markov-Switching Regime Detection Engine
 *
 * THEORETICAL BASIS:
 * Hamilton (1989) regime-switching model where observations depend on
 * an unobserved discrete Markov chain.
 *
 * S_t ∈ {1, 2, ..., K} follows Markov chain with transition matrix P
 * Y_t | S_t = k ~ N(μ_k, σ_k²)
 *
 * Key quantities:
 * - Filtered probability: P(S_t = k | Y_1:t)
 * - Smoothed probability: P(S_t = k | Y_1:T)
 * - Transition probability: P(S_{t+1} = j | S_t = i)
 *
 * APPLICATION: Detect regime changes in geopolitical stability
 * - Regime 1: Stable (low volatility, predictable)
 * - Regime 2: Volatile (medium volatility, uncertain)
 * - Regime 3: Crisis (high volatility, breakdown)
 */

export type RegimeId = 0 | 1 | 2;
export const REGIME_NAMES: Record<RegimeId, string> = {
  0: 'STABLE',
  1: 'VOLATILE',
  2: 'CRISIS',
};

export interface RegimeParameters {
  mean: number;
  variance: number;
  color: string;
}

export interface MarkovSwitchingConfig {
  numRegimes: number;
  regimeParams: RegimeParameters[];
  transitionMatrix: number[][];  // P[i][j] = P(S_{t+1}=j | S_t=i)
  initialProbs: number[];        // π_0
}

export interface FilteredState {
  timestamp: number;
  observation: number;
  filteredProbs: number[];      // P(S_t = k | Y_1:t)
  predictedProbs: number[];     // P(S_t = k | Y_1:t-1)
  likelihood: number;           // P(Y_t | Y_1:t-1)
  mostLikelyRegime: RegimeId;
}

export interface SmoothedState extends FilteredState {
  smoothedProbs: number[];      // P(S_t = k | Y_1:T)
}

export interface RegimeChangeEvent {
  timestamp: number;
  fromRegime: RegimeId;
  toRegime: RegimeId;
  transitionProb: number;
  surprise: number;  // -log P(transition)
}

// Default 3-regime model
const DEFAULT_CONFIG: MarkovSwitchingConfig = {
  numRegimes: 3,
  regimeParams: [
    { mean: 0.02, variance: 0.01, color: '#3b82f6' },    // Stable: low mean, low variance
    { mean: 0.0, variance: 0.04, color: '#f59e0b' },     // Volatile: zero mean, medium variance
    { mean: -0.05, variance: 0.09, color: '#ef4444' },   // Crisis: negative mean, high variance
  ],
  transitionMatrix: [
    [0.95, 0.04, 0.01],  // From Stable
    [0.10, 0.80, 0.10],  // From Volatile
    [0.05, 0.15, 0.80],  // From Crisis
  ],
  initialProbs: [0.7, 0.2, 0.1],
};

/**
 * Gaussian PDF for observation likelihood
 */
function gaussianPDF(x: number, mean: number, variance: number): number {
  const sigma = Math.sqrt(variance);
  const z = (x - mean) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

/**
 * Hamilton Filter: Compute filtered probabilities
 *
 * P(S_t = k | Y_1:t) ∝ P(Y_t | S_t = k) × P(S_t = k | Y_1:t-1)
 */
export function hamiltonFilter(
  observations: number[],
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): FilteredState[] {
  const { numRegimes, regimeParams, transitionMatrix, initialProbs } = config;
  const T = observations.length;
  const filtered: FilteredState[] = [];

  let priorProbs = [...initialProbs];

  for (let t = 0; t < T; t++) {
    const y = observations[t];

    // Prediction step: P(S_t = k | Y_1:t-1)
    const predictedProbs: number[] = new Array(numRegimes).fill(0);
    for (let k = 0; k < numRegimes; k++) {
      for (let j = 0; j < numRegimes; j++) {
        predictedProbs[k] += priorProbs[j] * transitionMatrix[j][k];
      }
    }

    // Likelihood for each regime
    const likelihoods: number[] = regimeParams.map(params =>
      gaussianPDF(y, params.mean, params.variance)
    );

    // Filtering step: P(S_t = k | Y_1:t)
    const joint = predictedProbs.map((p, k) => p * likelihoods[k]);
    const marginalLikelihood = joint.reduce((sum, j) => sum + j, 0);

    const filteredProbs = joint.map(j => j / marginalLikelihood);

    // Most likely regime
    let maxProb = 0;
    let mostLikelyRegime: RegimeId = 0;
    filteredProbs.forEach((p, k) => {
      if (p > maxProb) {
        maxProb = p;
        mostLikelyRegime = k as RegimeId;
      }
    });

    filtered.push({
      timestamp: t,
      observation: y,
      filteredProbs,
      predictedProbs,
      likelihood: marginalLikelihood,
      mostLikelyRegime,
    });

    priorProbs = filteredProbs;
  }

  return filtered;
}

/**
 * Kim Smoother: Compute smoothed probabilities
 *
 * P(S_t = k | Y_1:T) using backward recursion
 */
export function kimSmoother(
  filtered: FilteredState[],
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): SmoothedState[] {
  const { numRegimes, transitionMatrix } = config;
  const T = filtered.length;

  // Initialize with filtered at T
  const smoothed: SmoothedState[] = filtered.map(f => ({
    ...f,
    smoothedProbs: [...f.filteredProbs],
  }));

  // Backward pass
  for (let t = T - 2; t >= 0; t--) {
    const smoothedProbs: number[] = new Array(numRegimes).fill(0);

    for (let k = 0; k < numRegimes; k++) {
      let sum = 0;
      for (let j = 0; j < numRegimes; j++) {
        // P(S_{t+1} = j | Y_1:T) × P(S_{t+1} = j | S_t = k) / P(S_{t+1} = j | Y_1:t)
        const smoothedNext = smoothed[t + 1].smoothedProbs[j];
        const transition = transitionMatrix[k][j];
        const predictedNext = filtered[t + 1].predictedProbs[j];

        if (predictedNext > 1e-10) {
          sum += smoothedNext * transition / predictedNext;
        }
      }
      smoothedProbs[k] = filtered[t].filteredProbs[k] * sum;
    }

    // Normalize
    const total = smoothedProbs.reduce((s, p) => s + p, 0);
    smoothed[t].smoothedProbs = smoothedProbs.map(p => p / total);
  }

  return smoothed;
}

/**
 * Detect regime changes from smoothed probabilities
 */
export function detectRegimeChanges(
  smoothed: SmoothedState[],
  threshold: number = 0.5
): RegimeChangeEvent[] {
  const changes: RegimeChangeEvent[] = [];

  for (let t = 1; t < smoothed.length; t++) {
    const prevRegime = smoothed[t - 1].mostLikelyRegime;
    const currRegime = smoothed[t].mostLikelyRegime;

    if (prevRegime !== currRegime) {
      const transitionProb = smoothed[t].smoothedProbs[currRegime];

      if (transitionProb > threshold) {
        changes.push({
          timestamp: t,
          fromRegime: prevRegime,
          toRegime: currRegime,
          transitionProb,
          surprise: -Math.log(transitionProb),
        });
      }
    }
  }

  return changes;
}

/**
 * Compute expected regime duration
 *
 * E[Duration in regime k] = 1 / (1 - P_kk)
 */
export function expectedRegimeDuration(
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): number[] {
  return config.transitionMatrix.map((row, k) =>
    1 / (1 - row[k])
  );
}

/**
 * Compute stationary distribution of regimes
 *
 * π such that π = π × P
 */
export function stationaryDistribution(
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): number[] {
  const { numRegimes, transitionMatrix } = config;

  // Power iteration
  let pi = new Array(numRegimes).fill(1 / numRegimes);

  for (let iter = 0; iter < 1000; iter++) {
    const newPi = new Array(numRegimes).fill(0);

    for (let j = 0; j < numRegimes; j++) {
      for (let i = 0; i < numRegimes; i++) {
        newPi[j] += pi[i] * transitionMatrix[i][j];
      }
    }

    // Check convergence
    const diff = pi.reduce((sum, p, i) => sum + Math.abs(p - newPi[i]), 0);
    pi = newPi;

    if (diff < 1e-10) break;
  }

  return pi;
}

/**
 * Forecast future regime probabilities
 */
export function forecastRegimes(
  currentProbs: number[],
  horizons: number[],
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): Map<number, number[]> {
  const { transitionMatrix } = config;
  const forecasts = new Map<number, number[]>();

  for (const h of horizons) {
    // Compute P^h (transition matrix to power h)
    let probs = [...currentProbs];

    for (let step = 0; step < h; step++) {
      const newProbs = new Array(config.numRegimes).fill(0);
      for (let j = 0; j < config.numRegimes; j++) {
        for (let i = 0; i < config.numRegimes; i++) {
          newProbs[j] += probs[i] * transitionMatrix[i][j];
        }
      }
      probs = newProbs;
    }

    forecasts.set(h, probs);
  }

  return forecasts;
}

/**
 * Compute transition intensity (hazard rate)
 *
 * λ_ij = -log(1 - P_ij) for i ≠ j
 */
export function transitionIntensity(
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): number[][] {
  return config.transitionMatrix.map((row, i) =>
    row.map((p, j) =>
      i === j ? 0 : -Math.log(1 - Math.min(0.99, p))
    )
  );
}

/**
 * Generate regime data for visualization
 */
export interface RegimeVisualizationData {
  timestamp: string;
  stable: number;
  volatile: number;
  crisis: number;
  mostLikely: string;
  transitionRisk: number;
}

export function generateRegimeVisualization(
  smoothed: SmoothedState[],
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): RegimeVisualizationData[] {
  return smoothed.map((state, t) => {
    // Transition risk: probability of leaving current regime
    const currentRegime = state.mostLikelyRegime;
    const stayProb = config.transitionMatrix[currentRegime][currentRegime];
    const transitionRisk = 1 - stayProb;

    return {
      timestamp: new Date(Date.now() - (smoothed.length - t) * 3600000).toISOString(),
      stable: state.smoothedProbs[0],
      volatile: state.smoothedProbs[1],
      crisis: state.smoothedProbs[2],
      mostLikely: REGIME_NAMES[currentRegime],
      transitionRisk,
    };
  });
}

/**
 * Estimate transition matrix from observed regime sequence
 */
export function estimateTransitionMatrix(
  regimeSequence: RegimeId[],
  numRegimes: number = 3
): number[][] {
  const counts: number[][] = Array.from({ length: numRegimes }, () =>
    new Array(numRegimes).fill(0)
  );

  for (let t = 1; t < regimeSequence.length; t++) {
    const from = regimeSequence[t - 1];
    const to = regimeSequence[t];
    counts[from][to]++;
  }

  // Normalize rows
  return counts.map(row => {
    const total = row.reduce((s, c) => s + c, 0);
    return total > 0 ? row.map(c => c / total) : row.map(() => 1 / numRegimes);
  });
}

/**
 * Full regime analysis pipeline
 */
export function analyzeRegimes(
  observations: number[],
  config: MarkovSwitchingConfig = DEFAULT_CONFIG
): {
  filtered: FilteredState[];
  smoothed: SmoothedState[];
  changes: RegimeChangeEvent[];
  durations: number[];
  stationary: number[];
  visualization: RegimeVisualizationData[];
  currentRegime: RegimeId;
  transitionRisk: number;
} {
  const filtered = hamiltonFilter(observations, config);
  const smoothed = kimSmoother(filtered, config);
  const changes = detectRegimeChanges(smoothed);
  const durations = expectedRegimeDuration(config);
  const stationary = stationaryDistribution(config);
  const visualization = generateRegimeVisualization(smoothed, config);

  const currentState = smoothed[smoothed.length - 1];
  const currentRegime = currentState.mostLikelyRegime;
  const transitionRisk = 1 - config.transitionMatrix[currentRegime][currentRegime];

  return {
    filtered,
    smoothed,
    changes,
    durations,
    stationary,
    visualization,
    currentRegime,
    transitionRisk,
  };
}
