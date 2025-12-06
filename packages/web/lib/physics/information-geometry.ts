/**
 * Information Geometry Engine
 *
 * THEORETICAL BASIS:
 * Information geometry treats probability distributions as points on a
 * Riemannian manifold with the Fisher Information Matrix as the metric tensor.
 *
 * Fisher Information: I(θ)_ij = E[(∂logP/∂θ_i)(∂logP/∂θ_j)]
 *
 * Natural Gradient: ∇̃f = I(θ)⁻¹ ∇f
 *
 * This provides:
 * 1. Intrinsic distance between probability distributions (geodesics)
 * 2. Optimal parameter update directions (natural gradient descent)
 * 3. Sensitivity analysis (which parameters matter most)
 *
 * APPLICATION: Measuring "surprise" in geopolitical regime shifts
 * - KL divergence: How different is new state from expected?
 * - Fisher curvature: How sensitive is system to parameter changes?
 */

export interface DistributionParams {
  mean: number;
  variance: number;
  // For more complex distributions, extend with additional params
  skewness?: number;
  kurtosis?: number;
}

export interface FisherMetric {
  dimension: number;
  components: number[][]; // Metric tensor g_ij
  determinant: number;
  eigenvalues: number[];
  condition: number;      // Condition number (numerical stability)
}

export interface GeodesicPath {
  points: DistributionParams[];
  arcLength: number;
  curvature: number[];
}

/**
 * Compute Fisher Information Matrix for Gaussian distribution
 *
 * For N(μ, σ²):
 * I(μ, σ) = [[1/σ², 0], [0, 2/σ²]]
 */
export function fisherGaussian(params: DistributionParams): FisherMetric {
  const { variance } = params;
  const sigma2 = variance;

  const I11 = 1 / sigma2;          // d²logL/dμ²
  const I22 = 2 / sigma2;          // d²logL/dσ²
  const I12 = 0;                    // Cross term (zero for Gaussian)

  const components = [[I11, I12], [I12, I22]];
  const determinant = I11 * I22 - I12 * I12;
  const eigenvalues = [I11, I22].sort((a, b) => b - a);
  const condition = eigenvalues[0] / eigenvalues[1];

  return {
    dimension: 2,
    components,
    determinant,
    eigenvalues,
    condition,
  };
}

/**
 * Compute Fisher Information for multinomial distribution
 *
 * For categorical with probabilities p_1, ..., p_k:
 * I(p)_ij = δ_ij/p_i + 1/p_k (constraint: sum p_i = 1)
 */
export function fisherMultinomial(probabilities: number[]): FisherMetric {
  const k = probabilities.length;
  const components: number[][] = [];

  for (let i = 0; i < k - 1; i++) {
    const row: number[] = [];
    for (let j = 0; j < k - 1; j++) {
      if (i === j) {
        row.push(1 / probabilities[i] + 1 / probabilities[k - 1]);
      } else {
        row.push(1 / probabilities[k - 1]);
      }
    }
    components.push(row);
  }

  // Compute determinant (simplified for small matrices)
  const det = computeDeterminant(components);
  const eigenvalues = computeEigenvalues(components);
  const condition = eigenvalues.length > 0
    ? eigenvalues[0] / eigenvalues[eigenvalues.length - 1]
    : 1;

  return {
    dimension: k - 1,
    components,
    determinant: det,
    eigenvalues,
    condition,
  };
}

/**
 * Kullback-Leibler Divergence: D_KL(P || Q)
 *
 * D_KL(P || Q) = ∫ P(x) log(P(x)/Q(x)) dx
 *
 * For Gaussians:
 * D_KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
 */
export function klDivergenceGaussian(
  P: DistributionParams,
  Q: DistributionParams
): number {
  const { mean: mu1, variance: var1 } = P;
  const { mean: mu2, variance: var2 } = Q;

  const sigma1 = Math.sqrt(var1);
  const sigma2 = Math.sqrt(var2);

  return Math.log(sigma2 / sigma1) +
    (var1 + Math.pow(mu1 - mu2, 2)) / (2 * var2) - 0.5;
}

/**
 * Symmetric KL Divergence (Jensen-Shannon-like)
 * D_sym(P, Q) = 0.5 * D_KL(P||Q) + 0.5 * D_KL(Q||P)
 */
export function symmetricKL(P: DistributionParams, Q: DistributionParams): number {
  return 0.5 * klDivergenceGaussian(P, Q) + 0.5 * klDivergenceGaussian(Q, P);
}

/**
 * Fisher-Rao geodesic distance between Gaussian distributions
 *
 * For the Gaussian manifold, this is:
 * d(P, Q)² = (μ₁-μ₂)²/(σ₁σ₂) + 2log²(σ₁/σ₂)
 */
export function fisherRaoDistance(P: DistributionParams, Q: DistributionParams): number {
  const { mean: mu1, variance: var1 } = P;
  const { mean: mu2, variance: var2 } = Q;

  const sigma1 = Math.sqrt(var1);
  const sigma2 = Math.sqrt(var2);

  const meanTerm = Math.pow(mu1 - mu2, 2) / (sigma1 * sigma2);
  const varTerm = 2 * Math.pow(Math.log(sigma1 / sigma2), 2);

  return Math.sqrt(meanTerm + varTerm);
}

/**
 * Compute geodesic path between two distributions
 *
 * For Gaussians, the geodesic is:
 * μ(t) = (1-t)μ₁ + tμ₂
 * σ(t) = σ₁^(1-t) * σ₂^t
 */
export function computeGeodesic(
  P: DistributionParams,
  Q: DistributionParams,
  numPoints: number = 20
): GeodesicPath {
  const points: DistributionParams[] = [];
  const curvature: number[] = [];

  const { mean: mu1, variance: var1 } = P;
  const { mean: mu2, variance: var2 } = Q;
  const sigma1 = Math.sqrt(var1);
  const sigma2 = Math.sqrt(var2);

  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);

    const mean = (1 - t) * mu1 + t * mu2;
    const sigma = Math.pow(sigma1, 1 - t) * Math.pow(sigma2, t);

    points.push({ mean, variance: sigma * sigma });

    // Curvature along geodesic (simplified: rate of change of tangent)
    if (i > 0 && i < numPoints - 1) {
      const prev = points[i - 1];
      const next = {
        mean: (1 - (i + 1) / (numPoints - 1)) * mu1 + ((i + 1) / (numPoints - 1)) * mu2,
        variance: Math.pow(
          Math.pow(sigma1, 1 - (i + 1) / (numPoints - 1)) *
          Math.pow(sigma2, (i + 1) / (numPoints - 1)), 2
        )
      };

      const dmu = (next.mean - prev.mean) / 2;
      const dsigma = (Math.sqrt(next.variance) - Math.sqrt(prev.variance)) / 2;
      curvature.push(Math.sqrt(dmu * dmu + dsigma * dsigma));
    }
  }

  return {
    points,
    arcLength: fisherRaoDistance(P, Q),
    curvature,
  };
}

/**
 * Natural gradient: I(θ)⁻¹ ∇f
 *
 * The natural gradient accounts for the curvature of the parameter manifold,
 * providing more efficient optimization.
 */
export function naturalGradient(
  euclideanGradient: number[],
  fisherMetric: FisherMetric
): number[] {
  // Invert Fisher matrix (2x2 case)
  const [[a, b], [c, d]] = fisherMetric.components;
  const det = fisherMetric.determinant;

  if (Math.abs(det) < 1e-10) {
    return euclideanGradient; // Fallback if singular
  }

  const invFisher = [
    [d / det, -b / det],
    [-c / det, a / det],
  ];

  // Matrix-vector multiplication
  return euclideanGradient.map((_, i) =>
    invFisher[i].reduce((sum, val, j) => sum + val * euclideanGradient[j], 0)
  );
}

/**
 * Compute information gain (reduction in entropy)
 *
 * IG = H(prior) - H(posterior)
 */
export function informationGain(
  prior: DistributionParams,
  posterior: DistributionParams
): number {
  // Entropy of Gaussian: H = 0.5 * log(2πeσ²)
  const H_prior = 0.5 * Math.log(2 * Math.PI * Math.E * prior.variance);
  const H_posterior = 0.5 * Math.log(2 * Math.PI * Math.E * posterior.variance);

  return H_prior - H_posterior;
}

/**
 * Surprise metric: How unexpected is observation x given model?
 *
 * S(x) = -log P(x) (self-information)
 */
export function surpriseGaussian(
  observation: number,
  model: DistributionParams
): number {
  const { mean, variance } = model;
  const sigma = Math.sqrt(variance);

  // Gaussian PDF
  const z = (observation - mean) / sigma;
  const logP = -0.5 * Math.log(2 * Math.PI * variance) - 0.5 * z * z;

  return -logP; // Surprise is negative log probability
}

/**
 * Detect anomalies using Fisher Information and KL divergence
 *
 * Flags states that are "far" from expected distribution
 */
export interface AnomalyResult {
  isAnomaly: boolean;
  surprise: number;
  klDivergence: number;
  fisherDistance: number;
  threshold: number;
}

export function detectAnomaly(
  observed: DistributionParams,
  expected: DistributionParams,
  threshold: number = 3.0  // Standard deviations
): AnomalyResult {
  const kl = klDivergenceGaussian(observed, expected);
  const fisherDist = fisherRaoDistance(observed, expected);
  const surprise = surpriseGaussian(observed.mean, expected);

  // Combine metrics for anomaly score
  const anomalyScore = Math.sqrt(kl) + 0.5 * fisherDist;
  const isAnomaly = anomalyScore > threshold;

  return {
    isAnomaly,
    surprise,
    klDivergence: kl,
    fisherDistance: fisherDist,
    threshold,
  };
}

// Helper functions for matrix operations

function computeDeterminant(matrix: number[][]): number {
  const n = matrix.length;
  if (n === 1) return matrix[0][0];
  if (n === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

  // LU decomposition for larger matrices (simplified)
  let det = 1;
  const lu = matrix.map(row => [...row]);

  for (let i = 0; i < n; i++) {
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(lu[k][i]) > Math.abs(lu[maxRow][i])) {
        maxRow = k;
      }
    }

    if (maxRow !== i) {
      [lu[i], lu[maxRow]] = [lu[maxRow], lu[i]];
      det *= -1;
    }

    if (Math.abs(lu[i][i]) < 1e-10) return 0;

    det *= lu[i][i];

    for (let k = i + 1; k < n; k++) {
      const factor = lu[k][i] / lu[i][i];
      for (let j = i; j < n; j++) {
        lu[k][j] -= factor * lu[i][j];
      }
    }
  }

  return det;
}

function computeEigenvalues(matrix: number[][]): number[] {
  const n = matrix.length;

  if (n === 2) {
    // Quadratic formula for 2x2
    const [[a, b], [c, d]] = matrix;
    const trace = a + d;
    const det = a * d - b * c;
    const discriminant = trace * trace - 4 * det;

    if (discriminant < 0) {
      return [trace / 2]; // Complex eigenvalues, return real part
    }

    const sqrtDisc = Math.sqrt(discriminant);
    return [(trace + sqrtDisc) / 2, (trace - sqrtDisc) / 2].sort((a, b) => b - a);
  }

  // For larger matrices, use power iteration (simplified)
  return [matrix[0][0]]; // Placeholder
}

/**
 * Map geopolitical metrics to distribution parameters
 *
 * Converts risk scores, confidence intervals, and uncertainty
 * into a proper probability distribution for geometric analysis.
 */
export function metricsToDistribution(
  riskScore: number,          // 0-100
  confidenceInterval: number, // Width of CI
  historicalVolatility: number // Standard deviation of past values
): DistributionParams {
  return {
    mean: riskScore,
    variance: Math.pow(confidenceInterval / 4, 2) + Math.pow(historicalVolatility, 2),
  };
}

/**
 * Compute parameter sensitivity using Fisher Information
 *
 * Higher Fisher Information = parameter is more "identifiable"
 * Lower Fisher Information = parameter changes have little effect
 */
export function parameterSensitivity(
  params: DistributionParams
): { meanSensitivity: number; varianceSensitivity: number } {
  const fisher = fisherGaussian(params);

  return {
    meanSensitivity: fisher.components[0][0],
    varianceSensitivity: fisher.components[1][1],
  };
}
