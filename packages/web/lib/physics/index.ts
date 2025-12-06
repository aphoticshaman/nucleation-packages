/**
 * Physics Engine Index
 *
 * PROMETHEUS-powered computational physics for geopolitical intelligence.
 * Force-fused from thermodynamics, information theory, and dynamical systems.
 */

// Transfer Entropy - Directed information flow measurement
export {
  computeTransferEntropy,
  computeSignificance,
  findOptimalLag,
  analyzeTransferEntropy,
  computeEdgeWeights,
  type TimeSeriesPoint,
  type TransferEntropyResult,
  type TEConfig,
} from './transfer-entropy';

// Landau-Ginzburg - Phase transition dynamics
export {
  potential,
  potentialDerivative,
  findEquilibria,
  determineBasin,
  computeStability,
  langevinStep,
  simulateTrajectory,
  detectTransitions,
  kramersEscapeRate,
  stressToLandauCoefficient,
  generatePotentialCurve,
  createPhaseState,
  type PhaseState,
  type LandauGinzburgConfig,
  type TransitionEvent,
} from './landau-ginzburg';

// Information Geometry - Probability manifold analysis
export {
  fisherGaussian,
  fisherMultinomial,
  klDivergenceGaussian,
  symmetricKL,
  fisherRaoDistance,
  computeGeodesic,
  naturalGradient,
  informationGain,
  surpriseGaussian,
  detectAnomaly,
  metricsToDistribution,
  parameterSensitivity,
  type DistributionParams,
  type FisherMetric,
  type GeodesicPath,
  type AnomalyResult,
} from './information-geometry';

// Dempster-Shafer - Evidence fusion under uncertainty
export {
  createBeliefFunction,
  computeBelief,
  computePlausibility,
  computeBeliefIntervals,
  dempsterCombine,
  fuseMultipleSources,
  pignisticTransform,
  beliefEntropy,
  assessmentToBeliefFunction,
  fuseIntelligenceAssessments,
  pairwiseConflict,
  identifyConflictingSources,
  REGIME_FRAME,
  type MassAssignment,
  type BeliefFunction,
  type FusionResult,
  type BeliefInterval,
  type SourceAssessment,
} from './dempster-shafer';

// Markov-Switching - Regime detection and forecasting
export {
  hamiltonFilter,
  kimSmoother,
  detectRegimeChanges,
  expectedRegimeDuration,
  stationaryDistribution,
  forecastRegimes,
  transitionIntensity,
  generateRegimeVisualization,
  estimateTransitionMatrix,
  analyzeRegimes,
  REGIME_NAMES,
  type RegimeId,
  type RegimeParameters,
  type MarkovSwitchingConfig,
  type FilteredState,
  type SmoothedState,
  type RegimeChangeEvent,
  type RegimeVisualizationData,
} from './markov-switching';

// Tropical Geometry - Attention mechanisms and expressivity bounds
export {
  // Semiring operations
  tropicalAdd,
  tropicalMul,
  tropicalPow,
  tropicalSum,
  tropicalProduct,
  TROPICAL_ZERO,
  TROPICAL_ONE,

  // Matrix operations
  tropicalMatMul,
  tropicalDet2x2,
  tropicalMatVec,

  // Polynomials
  evaluateTropicalPolynomial,
  findTropicalCorners,
  countLinearRegions,

  // Softmax → Tropical limit
  logSumExp,
  temperatureLogSumExp,
  demonstrateTropicalLimit,

  // Transformer expressivity bounds
  binomial,
  newtonPolytopeBound,
  calculateExpressivityBounds,
  minLayersForDegree,

  // ARC task complexity
  estimateARCComplexity,
  estimateFromOperations,
  parseARCOperations,
  ARC_OPERATION_PATTERNS,

  // Proof verification
  TROPICAL_PROOF_STEPS,
  verifyTropicalTheorem,

  // Analysis
  analyzeAttentionTropicality,
  tropicalGeometry,

  type TropicalNumber,
  type TropicalMatrix,
  type TropicalPolynomial,
  type ExpressivityBound,
  type ARCComplexity,
  type ProofStep,
} from './tropical-geometry';
