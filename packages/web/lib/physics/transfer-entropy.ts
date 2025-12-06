/**
 * Transfer Entropy Engine
 *
 * THEORETICAL BASIS:
 * Transfer Entropy T_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})
 *
 * Measures directed information flow from X to Y beyond what Y's past predicts.
 * Used for causal edge weighting in the CausalGraph topology.
 *
 * Reference: Schreiber, T. (2000). "Measuring Information Transfer"
 */

export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
}

export interface TransferEntropyResult {
  sourceId: string;
  targetId: string;
  transferEntropy: number;  // T_{X→Y} in bits
  normalizedTE: number;     // 0-1 scale
  significance: number;     // p-value via surrogate testing
  lag: number;              // Optimal lag in time steps
  direction: 'forward' | 'backward' | 'bidirectional';
}

export interface TEConfig {
  embeddingDim: number;     // k: history length for target
  sourceLag: number;        // l: lag for source
  numBins: number;          // Discretization bins
  numSurrogates: number;    // For significance testing
  alpha: number;            // Significance threshold
}

const DEFAULT_CONFIG: TEConfig = {
  embeddingDim: 3,
  sourceLag: 1,
  numBins: 8,
  numSurrogates: 100,
  alpha: 0.05,
};

/**
 * Discretize continuous time series into bins
 * Uses equiprobable binning for maximum entropy
 */
function discretize(series: number[], numBins: number): number[] {
  const sorted = [...series].sort((a, b) => a - b);
  const binEdges: number[] = [];

  for (let i = 1; i < numBins; i++) {
    const idx = Math.floor((i / numBins) * sorted.length);
    binEdges.push(sorted[idx]);
  }

  return series.map(value => {
    let bin = 0;
    for (const edge of binEdges) {
      if (value >= edge) bin++;
    }
    return bin;
  });
}

/**
 * Compute Shannon entropy H(X) from discretized series
 */
function shannonEntropy(symbols: number[]): number {
  const counts = new Map<number, number>();
  for (const s of symbols) {
    counts.set(s, (counts.get(s) || 0) + 1);
  }

  let entropy = 0;
  const n = symbols.length;
  for (const count of counts.values()) {
    const p = count / n;
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

/**
 * Create joint symbol from multiple series at different lags
 */
function createJointSymbols(
  target: number[],
  source: number[],
  k: number,
  l: number
): { targetNext: number[]; targetPast: number[][]; sourcePast: number[][] } {
  const n = Math.min(target.length, source.length);
  const startIdx = Math.max(k, l);

  const targetNext: number[] = [];
  const targetPast: number[][] = [];
  const sourcePast: number[][] = [];

  for (let t = startIdx; t < n; t++) {
    targetNext.push(target[t]);

    const tPast: number[] = [];
    for (let i = 1; i <= k; i++) {
      tPast.push(target[t - i]);
    }
    targetPast.push(tPast);

    const sPast: number[] = [];
    for (let i = 1; i <= l; i++) {
      sPast.push(source[t - i]);
    }
    sourcePast.push(sPast);
  }

  return { targetNext, targetPast, sourcePast };
}

/**
 * Encode joint state as unique integer for counting
 */
function encodeState(symbols: number[], base: number): number {
  let code = 0;
  let multiplier = 1;
  for (const s of symbols) {
    code += s * multiplier;
    multiplier *= base;
  }
  return code;
}

/**
 * Compute conditional entropy H(Y|X) using counting
 */
function conditionalEntropy(
  Y: number[],
  X: number[][],
  numBins: number
): number {
  // Joint distribution P(Y, X)
  const jointCounts = new Map<string, number>();
  const xCounts = new Map<string, number>();

  for (let i = 0; i < Y.length; i++) {
    const xKey = X[i].join(',');
    const jointKey = `${Y[i]}|${xKey}`;

    jointCounts.set(jointKey, (jointCounts.get(jointKey) || 0) + 1);
    xCounts.set(xKey, (xCounts.get(xKey) || 0) + 1);
  }

  // H(Y|X) = -sum P(y,x) log P(y|x)
  let condEntropy = 0;
  const n = Y.length;

  for (const [jointKey, jointCount] of jointCounts) {
    const [, xKey] = jointKey.split('|');
    const xCount = xCounts.get(xKey) || 1;

    const pJoint = jointCount / n;
    const pConditional = jointCount / xCount;

    if (pConditional > 0) {
      condEntropy -= pJoint * Math.log2(pConditional);
    }
  }

  return condEntropy;
}

/**
 * Compute Transfer Entropy from source X to target Y
 *
 * T_{X→Y} = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-l})
 */
export function computeTransferEntropy(
  source: TimeSeriesPoint[],
  target: TimeSeriesPoint[],
  config: Partial<TEConfig> = {}
): number {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Extract values and discretize
  const sourceValues = discretize(source.map(p => p.value), cfg.numBins);
  const targetValues = discretize(target.map(p => p.value), cfg.numBins);

  // Create joint symbols
  const { targetNext, targetPast, sourcePast } = createJointSymbols(
    targetValues,
    sourceValues,
    cfg.embeddingDim,
    cfg.sourceLag
  );

  if (targetNext.length < 10) {
    return 0; // Not enough data
  }

  // H(Y_t | Y_past)
  const H_Y_given_Ypast = conditionalEntropy(targetNext, targetPast, cfg.numBins);

  // H(Y_t | Y_past, X_past) - combine past vectors
  const combinedPast = targetPast.map((tp, i) => [...tp, ...sourcePast[i]]);
  const H_Y_given_YXpast = conditionalEntropy(targetNext, combinedPast, cfg.numBins);

  // Transfer Entropy
  const TE = H_Y_given_Ypast - H_Y_given_YXpast;

  return Math.max(0, TE); // TE should be non-negative
}

/**
 * Compute significance via surrogate testing (shuffle surrogates)
 */
export function computeSignificance(
  source: TimeSeriesPoint[],
  target: TimeSeriesPoint[],
  observedTE: number,
  config: Partial<TEConfig> = {}
): number {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  let countGreater = 0;

  for (let i = 0; i < cfg.numSurrogates; i++) {
    // Shuffle source to destroy temporal structure
    const shuffledSource = [...source].sort(() => Math.random() - 0.5);
    const surrogateTE = computeTransferEntropy(shuffledSource, target, cfg);

    if (surrogateTE >= observedTE) {
      countGreater++;
    }
  }

  return countGreater / cfg.numSurrogates;
}

/**
 * Find optimal lag by maximizing TE
 */
export function findOptimalLag(
  source: TimeSeriesPoint[],
  target: TimeSeriesPoint[],
  maxLag: number = 10,
  config: Partial<TEConfig> = {}
): { lag: number; te: number } {
  let bestLag = 1;
  let bestTE = 0;

  for (let lag = 1; lag <= maxLag; lag++) {
    const te = computeTransferEntropy(source, target, { ...config, sourceLag: lag });
    if (te > bestTE) {
      bestTE = te;
      bestLag = lag;
    }
  }

  return { lag: bestLag, te: bestTE };
}

/**
 * Full Transfer Entropy analysis between two time series
 */
export async function analyzeTransferEntropy(
  sourceId: string,
  targetId: string,
  sourceSeries: TimeSeriesPoint[],
  targetSeries: TimeSeriesPoint[],
  config: Partial<TEConfig> = {}
): Promise<TransferEntropyResult> {
  const cfg = { ...DEFAULT_CONFIG, ...config };

  // Forward direction: source → target
  const { lag: forwardLag, te: forwardTE } = findOptimalLag(sourceSeries, targetSeries, 10, cfg);

  // Backward direction: target → source
  const { lag: backwardLag, te: backwardTE } = findOptimalLag(targetSeries, sourceSeries, 10, cfg);

  // Significance testing
  const forwardPValue = computeSignificance(sourceSeries, targetSeries, forwardTE, cfg);
  const backwardPValue = computeSignificance(targetSeries, sourceSeries, backwardTE, cfg);

  // Determine dominant direction
  let direction: 'forward' | 'backward' | 'bidirectional';
  let primaryTE: number;
  let primaryLag: number;
  let significance: number;

  const forwardSignificant = forwardPValue < cfg.alpha;
  const backwardSignificant = backwardPValue < cfg.alpha;

  if (forwardSignificant && backwardSignificant) {
    direction = 'bidirectional';
    primaryTE = Math.max(forwardTE, backwardTE);
    primaryLag = forwardTE > backwardTE ? forwardLag : backwardLag;
    significance = Math.min(forwardPValue, backwardPValue);
  } else if (forwardSignificant) {
    direction = 'forward';
    primaryTE = forwardTE;
    primaryLag = forwardLag;
    significance = forwardPValue;
  } else if (backwardSignificant) {
    direction = 'backward';
    primaryTE = backwardTE;
    primaryLag = backwardLag;
    significance = backwardPValue;
  } else {
    direction = 'forward'; // Default
    primaryTE = forwardTE;
    primaryLag = forwardLag;
    significance = forwardPValue;
  }

  // Normalize TE to 0-1 range (empirical max ~2 bits for typical data)
  const normalizedTE = Math.min(1, primaryTE / 2);

  return {
    sourceId,
    targetId,
    transferEntropy: primaryTE,
    normalizedTE,
    significance,
    lag: primaryLag,
    direction,
  };
}

/**
 * Batch computation for causal graph edge weights
 */
export async function computeEdgeWeights(
  nodes: Array<{ id: string; timeSeries: TimeSeriesPoint[] }>,
  config: Partial<TEConfig> = {}
): Promise<TransferEntropyResult[]> {
  const results: TransferEntropyResult[] = [];

  for (let i = 0; i < nodes.length; i++) {
    for (let j = 0; j < nodes.length; j++) {
      if (i === j) continue;

      const result = await analyzeTransferEntropy(
        nodes[i].id,
        nodes[j].id,
        nodes[i].timeSeries,
        nodes[j].timeSeries,
        config
      );

      // Only include significant edges
      if (result.significance < (config.alpha || DEFAULT_CONFIG.alpha)) {
        results.push(result);
      }
    }
  }

  return results;
}
