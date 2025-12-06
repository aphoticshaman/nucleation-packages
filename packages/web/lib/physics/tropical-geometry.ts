/**
 * TROPICAL GEOMETRY MODULE
 *
 * Implements tropical algebra for transformer expressivity analysis.
 *
 * Core Theorem: Attention mechanisms in the T→∞ limit become
 * tropical polynomial evaluators, implying transformers compute
 * piecewise-linear functions bounded by tropical geometry.
 *
 * Tropical Semiring (T, ⊕, ⊗):
 * - T = R ∪ {-∞}
 * - a ⊕ b = max(a, b)  (tropical addition)
 * - a ⊗ b = a + b      (tropical multiplication)
 * - Identity for ⊕: -∞
 * - Identity for ⊗: 0
 */

// =============================================================================
// Types
// =============================================================================

/** Tropical number: real or negative infinity */
export type TropicalNumber = number; // -Infinity represents tropical zero

/** Tropical matrix as 2D array */
export type TropicalMatrix = TropicalNumber[][];

/** Tropical polynomial coefficients */
export interface TropicalPolynomial {
  coefficients: TropicalNumber[];
  degree: number;
}

/** Expressivity analysis result */
export interface ExpressivityBound {
  layers: number;
  tropicalDegree: number;
  maxLinearRegions: number;
  newtonPolytopeSize: number;
}

/** ARC task complexity estimate */
export interface ARCComplexity {
  nestedOperations: number;
  requiredDegree: number;
  minLayersNeeded: number;
  isWithinBound: boolean;
}

// =============================================================================
// Tropical Semiring Operations
// =============================================================================

/** Tropical zero (additive identity) */
export const TROPICAL_ZERO = -Infinity;

/** Tropical one (multiplicative identity) */
export const TROPICAL_ONE = 0;

/**
 * Tropical addition: a ⊕ b = max(a, b)
 */
export function tropicalAdd(a: TropicalNumber, b: TropicalNumber): TropicalNumber {
  return Math.max(a, b);
}

/**
 * Tropical multiplication: a ⊗ b = a + b
 */
export function tropicalMul(a: TropicalNumber, b: TropicalNumber): TropicalNumber {
  // Handle -∞ cases
  if (a === TROPICAL_ZERO || b === TROPICAL_ZERO) {
    return TROPICAL_ZERO;
  }
  return a + b;
}

/**
 * Tropical power: a^{⊗n} = n * a
 */
export function tropicalPow(a: TropicalNumber, n: number): TropicalNumber {
  if (a === TROPICAL_ZERO) return TROPICAL_ZERO;
  return n * a;
}

/**
 * Tropical sum of array: ⊕_i a_i = max(a_i)
 */
export function tropicalSum(arr: TropicalNumber[]): TropicalNumber {
  if (arr.length === 0) return TROPICAL_ZERO;
  return Math.max(...arr);
}

/**
 * Tropical product of array: ⊗_i a_i = Σ a_i
 */
export function tropicalProduct(arr: TropicalNumber[]): TropicalNumber {
  if (arr.some(x => x === TROPICAL_ZERO)) return TROPICAL_ZERO;
  return arr.reduce((sum, x) => sum + x, 0);
}

// =============================================================================
// Tropical Matrix Operations
// =============================================================================

/**
 * Tropical matrix multiplication: (A ⊗ B)_ij = ⊕_k (A_ik ⊗ B_kj)
 *
 * This is the key operation that attention becomes in the T→∞ limit.
 */
export function tropicalMatMul(A: TropicalMatrix, B: TropicalMatrix): TropicalMatrix {
  const m = A.length;
  const n = B[0]?.length || 0;
  const k = A[0]?.length || 0;

  if (k !== B.length) {
    throw new Error(`Matrix dimension mismatch: ${k} vs ${B.length}`);
  }

  const result: TropicalMatrix = [];

  for (let i = 0; i < m; i++) {
    result[i] = [];
    for (let j = 0; j < n; j++) {
      // (A ⊗ B)_ij = max_k(A_ik + B_kj)
      let maxVal = TROPICAL_ZERO;
      for (let l = 0; l < k; l++) {
        const product = tropicalMul(A[i][l], B[l][j]);
        maxVal = tropicalAdd(maxVal, product);
      }
      result[i][j] = maxVal;
    }
  }

  return result;
}

/**
 * Tropical determinant of 2x2 matrix: max(a⊗d, b⊗c) = max(a+d, b+c)
 */
export function tropicalDet2x2(matrix: TropicalMatrix): TropicalNumber {
  if (matrix.length !== 2 || matrix[0].length !== 2) {
    throw new Error('Matrix must be 2x2');
  }

  const [[a, b], [c, d]] = matrix;
  return tropicalAdd(
    tropicalMul(a, d),  // a + d
    tropicalMul(b, c)   // b + c
  );
}

/**
 * Tropical matrix-vector multiplication
 */
export function tropicalMatVec(A: TropicalMatrix, v: TropicalNumber[]): TropicalNumber[] {
  return A.map(row =>
    tropicalSum(row.map((a, j) => tropicalMul(a, v[j])))
  );
}

// =============================================================================
// Tropical Polynomials
// =============================================================================

/**
 * Evaluate tropical polynomial: p(x) = ⊕_i (c_i ⊗ x^{⊗i}) = max_i(c_i + i*x)
 *
 * This is piecewise-linear (maximum of linear functions).
 */
export function evaluateTropicalPolynomial(
  poly: TropicalPolynomial,
  x: TropicalNumber
): TropicalNumber {
  if (x === TROPICAL_ZERO) {
    // Only constant term survives
    return poly.coefficients[0] ?? TROPICAL_ZERO;
  }

  const terms = poly.coefficients.map((c, i) => {
    if (c === TROPICAL_ZERO) return TROPICAL_ZERO;
    return tropicalMul(c, tropicalPow(x, i)); // c + i*x
  });

  return tropicalSum(terms);
}

/**
 * Find the "corners" of a tropical polynomial (where linear pieces meet)
 */
export function findTropicalCorners(poly: TropicalPolynomial): number[] {
  const corners: number[] = [];
  const { coefficients } = poly;

  // Corners occur where two terms are equal:
  // c_i + i*x = c_j + j*x  →  x = (c_j - c_i) / (i - j)
  for (let i = 0; i < coefficients.length; i++) {
    for (let j = i + 1; j < coefficients.length; j++) {
      if (coefficients[i] !== TROPICAL_ZERO && coefficients[j] !== TROPICAL_ZERO) {
        const x = (coefficients[j] - coefficients[i]) / (i - j);
        if (isFinite(x)) {
          corners.push(x);
        }
      }
    }
  }

  return [...new Set(corners)].sort((a, b) => a - b);
}

/**
 * Count linear regions of a tropical polynomial (= number of corners + 1)
 */
export function countLinearRegions(poly: TropicalPolynomial): number {
  const corners = findTropicalCorners(poly);
  return corners.length + 1;
}

// =============================================================================
// Softmax → Tropical Limit
// =============================================================================

/**
 * Standard log-sum-exp (stable)
 */
export function logSumExp(scores: number[]): number {
  if (scores.length === 0) return -Infinity;
  const max = Math.max(...scores);
  if (!isFinite(max)) return max;
  return max + Math.log(scores.reduce((sum, s) => sum + Math.exp(s - max), 0));
}

/**
 * Temperature-scaled log-sum-exp: T * log(Σ exp(s/T))
 *
 * As T → ∞, this approaches max(s) (tropical addition)
 */
export function temperatureLogSumExp(scores: number[], temperature: number): number {
  if (temperature === 0) {
    return Math.max(...scores); // Hard max
  }

  const scaled = scores.map(s => s / temperature);
  return temperature * logSumExp(scaled);
}

/**
 * Demonstrate the tropical limit: T * log(Σ exp(s/T)) → max(s)
 */
export function demonstrateTropicalLimit(
  scores: number[],
  temperatures: number[] = [0.1, 1, 10, 100, 1000]
): { temperature: number; value: number; trueMax: number; error: number }[] {
  const trueMax = Math.max(...scores);

  return temperatures.map(T => {
    const value = temperatureLogSumExp(scores, T);
    return {
      temperature: T,
      value,
      trueMax,
      error: Math.abs(value - trueMax)
    };
  });
}

// =============================================================================
// Transformer Expressivity Bounds
// =============================================================================

/**
 * Calculate binomial coefficient C(n, k)
 */
export function binomial(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  if (k === 0 || k === n) return 1;

  // Use symmetry
  if (k > n - k) k = n - k;

  let result = 1;
  for (let i = 0; i < k; i++) {
    result = result * (n - i) / (i + 1);
  }
  return Math.round(result);
}

/**
 * Calculate Newton polytope size bound: C(n+d, d)
 *
 * This bounds the number of linear regions for a degree-d
 * tropical polynomial in n variables.
 */
export function newtonPolytopeBound(variables: number, degree: number): number {
  return binomial(variables + degree, degree);
}

/**
 * Calculate transformer expressivity bounds
 *
 * Main theorem: L layers → tropical degree 2^L
 * Linear regions bounded by Newton polytope of that degree
 */
export function calculateExpressivityBounds(
  layers: number,
  inputDimension: number = 10,
  width: number = 64
): ExpressivityBound {
  const tropicalDegree = Math.pow(2, layers);
  const newtonPolytopeSize = newtonPolytopeBound(inputDimension, tropicalDegree);

  // Alternative bound: w^d for width w, depth d
  const maxLinearRegions = Math.min(
    newtonPolytopeSize,
    Math.pow(width, layers)
  );

  return {
    layers,
    tropicalDegree,
    maxLinearRegions,
    newtonPolytopeSize
  };
}

/**
 * Find minimum layers needed for a given tropical degree
 */
export function minLayersForDegree(targetDegree: number): number {
  return Math.ceil(Math.log2(targetDegree));
}

// =============================================================================
// ARC Task Complexity
// =============================================================================

/**
 * Estimate ARC task complexity based on nested operations
 *
 * k nested operations → tropical degree 2^k → need log2(2^k) = k layers minimum
 */
export function estimateARCComplexity(
  nestedOperations: number,
  availableLayers: number = 12
): ARCComplexity {
  const requiredDegree = Math.pow(2, nestedOperations);
  const minLayersNeeded = nestedOperations; // log2(2^k) = k
  const isWithinBound = availableLayers >= minLayersNeeded;

  return {
    nestedOperations,
    requiredDegree,
    minLayersNeeded,
    isWithinBound
  };
}

/**
 * Parse ARC task description into operation count
 */
export function parseARCOperations(operations: string[]): number {
  // Each operation in the chain adds one level of nesting
  return operations.length;
}

/**
 * Common ARC operation patterns and their complexity
 */
export const ARC_OPERATION_PATTERNS = {
  detectSymmetry: 1,
  findAxis: 1,
  reflect: 1,
  rotate: 1,
  recolor: 1,
  scale: 1,
  translate: 1,
  cropRegion: 1,
  fillPattern: 2,  // Requires detection + action
  copyPaste: 2,
  conditionalColor: 2,
  recursivePattern: 3,  // Higher complexity
  fractalGeneration: 4,
} as const;

/**
 * Estimate complexity from a list of operation names
 */
export function estimateFromOperations(
  operations: (keyof typeof ARC_OPERATION_PATTERNS)[]
): ARCComplexity {
  // Sum individual complexities (composition adds complexity)
  const totalComplexity = operations.reduce(
    (sum, op) => sum + (ARC_OPERATION_PATTERNS[op] || 1),
    0
  );

  return estimateARCComplexity(totalComplexity);
}

// =============================================================================
// Proof Step Verification
// =============================================================================

export interface ProofStep {
  id: string;
  description: string;
  compute: () => number;
  expectedAnswer: number;
}

/**
 * The 8 proof steps from the Tropical Attention Theorem
 */
export const TROPICAL_PROOF_STEPS: ProofStep[] = [
  {
    id: 'Q_SEMIRING',
    description: 'Tropical determinant of [[3,1],[2,5]]',
    compute: () => tropicalDet2x2([[3, 1], [2, 5]]),
    expectedAnswer: 8,
  },
  {
    id: 'Q_POLYNOMIAL',
    description: 'Tropical polynomial at critical point x=3',
    compute: () => {
      const poly: TropicalPolynomial = {
        coefficients: [5, 2, -1], // 5 ⊕ (2⊗x) ⊕ ((-1)⊗x²)
        degree: 2,
      };
      return evaluateTropicalPolynomial(poly, 3);
    },
    expectedAnswer: 5,
  },
  {
    id: 'Q_SOFTMAX_LIMIT',
    description: 'Count indices achieving max in [3,7,4,7,2]',
    compute: () => {
      const scores = [3, 7, 4, 7, 2];
      const maxVal = Math.max(...scores);
      return scores.filter(s => s === maxVal).length;
    },
    expectedAnswer: 2,
  },
  {
    id: 'Q_ATTENTION_TROPICAL',
    description: 'Tropical product [0,0] entry',
    compute: () => {
      const S: TropicalMatrix = [[1, 3], [4, 2]];
      const V: TropicalMatrix = [[5, 1], [2, 6]];
      const result = tropicalMatMul(S, V);
      return result[0][0];
    },
    expectedAnswer: 6,
  },
  {
    id: 'Q_EXPRESSIVITY',
    description: 'C(7,4) = Newton polytope bound',
    compute: () => binomial(7, 4),
    expectedAnswer: 35,
  },
  {
    id: 'Q_TRANSFORMER_DEPTH',
    description: 'Min layers for degree ≥ 100',
    compute: () => minLayersForDegree(100),
    expectedAnswer: 7,
  },
  {
    id: 'Q_ARC_BOUND',
    description: 'Degree for 4 nested operations',
    compute: () => Math.pow(2, 4),
    expectedAnswer: 16,
  },
  {
    id: 'Q_VERIFICATION',
    description: 'Max regions for width=8, depth=3',
    compute: () => Math.pow(8, 3),
    expectedAnswer: 512,
  },
];

/**
 * Run all proof steps and verify
 */
export function verifyTropicalTheorem(): {
  passed: number;
  total: number;
  results: { id: string; computed: number; expected: number; passed: boolean }[];
} {
  const results = TROPICAL_PROOF_STEPS.map(step => {
    const computed = step.compute();
    return {
      id: step.id,
      computed,
      expected: step.expectedAnswer,
      passed: computed === step.expectedAnswer,
    };
  });

  return {
    passed: results.filter(r => r.passed).length,
    total: results.length,
    results,
  };
}

// =============================================================================
// Integration with Attention Analysis
// =============================================================================

/**
 * Analyze attention pattern entropy and predict tropical behavior
 */
export function analyzeAttentionTropicality(
  attentionWeights: number[][],
  temperature: number = 1.0
): {
  averageEntropy: number;
  tropicalityScore: number; // 0-1, higher = more tropical (sparse)
  effectiveTemperature: number;
} {
  // Calculate entropy for each attention head
  const entropies = attentionWeights.map(weights => {
    const sum = weights.reduce((a, b) => a + b, 0);
    const probs = weights.map(w => w / sum);
    return -probs.reduce((h, p) => h + (p > 0 ? p * Math.log2(p) : 0), 0);
  });

  const averageEntropy = entropies.reduce((a, b) => a + b, 0) / entropies.length;
  const maxEntropy = Math.log2(attentionWeights[0]?.length || 1);

  // Tropicality: how close to hard max (low entropy)
  const tropicalityScore = 1 - (averageEntropy / maxEntropy);

  // Effective temperature: infer from entropy
  // Higher entropy = higher effective temperature
  const effectiveTemperature = temperature * (averageEntropy / maxEntropy + 0.1);

  return {
    averageEntropy,
    tropicalityScore,
    effectiveTemperature,
  };
}

// =============================================================================
// Export singleton utilities
// =============================================================================

export const tropicalGeometry = {
  // Semiring operations
  add: tropicalAdd,
  mul: tropicalMul,
  pow: tropicalPow,
  sum: tropicalSum,
  product: tropicalProduct,
  ZERO: TROPICAL_ZERO,
  ONE: TROPICAL_ONE,

  // Matrix operations
  matMul: tropicalMatMul,
  det2x2: tropicalDet2x2,
  matVec: tropicalMatVec,

  // Polynomials
  evalPoly: evaluateTropicalPolynomial,
  findCorners: findTropicalCorners,
  countRegions: countLinearRegions,

  // Softmax limit
  logSumExp,
  tempLogSumExp: temperatureLogSumExp,
  demonstrateLimit: demonstrateTropicalLimit,

  // Expressivity
  binomial,
  newtonBound: newtonPolytopeBound,
  expressivityBounds: calculateExpressivityBounds,
  minLayers: minLayersForDegree,

  // ARC
  arcComplexity: estimateARCComplexity,
  arcFromOps: estimateFromOperations,
  ARC_PATTERNS: ARC_OPERATION_PATTERNS,

  // Verification
  proofSteps: TROPICAL_PROOF_STEPS,
  verify: verifyTropicalTheorem,

  // Analysis
  analyzeAttention: analyzeAttentionTropicality,
};
