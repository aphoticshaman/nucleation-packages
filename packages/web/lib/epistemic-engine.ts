/**
 * LATTICEFORGE EPISTEMIC HUMILITY ENGINE
 *
 * Core principle: Know what you don't know, question what you think you know.
 *
 * Framework:
 * - KNOWN KNOWNS: Verified facts with high confidence
 * - KNOWN UNKNOWNS: Identified gaps in knowledge
 * - UNKNOWN UNKNOWNS: Blind spots we haven't identified
 * - UNKNOWN KNOWNS: Implicit knowledge not yet formalized (back-of-mind insights)
 *
 * NSM→XYZA Pipeline:
 * Novel Synthesis Mode transforms raw signals through:
 * X: eXtract (identify key variables)
 * Y: Yoke (connect to historical correlates)
 * Z: Zero-in (focus on causal mechanisms)
 * A: Ablate (test by removal)
 */

// ============================================================================
// EPISTEMIC KNOWLEDGE QUADRANTS
// ============================================================================

export type KnowledgeQuadrant =
  | 'known_known'      // Facts we know we know (high confidence)
  | 'known_unknown'    // Questions we know we need to answer
  | 'unknown_unknown'  // Blind spots (detected via anomaly/correlation gaps)
  | 'unknown_known';   // Implicit knowledge waiting to be formalized

export interface EpistemicClaim {
  id: string;
  claim: string;
  quadrant: KnowledgeQuadrant;
  confidence: number; // 0-1
  uncertainty: number; // epistemic uncertainty 0-1
  sources: string[]; // citation IDs
  historicalCorrelates: HistoricalCorrelate[];
  hypotheses: Hypothesis[];
  derivatives: Derivative[];
  ablationResults?: AblationResult[];
  simulationId?: string;
  lastUpdated: string;
}

export interface HistoricalCorrelate {
  eventId: string;
  eventName: string;
  period: string; // e.g., "1618-1648 Thirty Years War"
  yearsAgo: number;
  correlationStrength: number; // 0-1
  correlationType: 'causal' | 'structural' | 'cyclical' | 'analogical';
  keyVariables: string[];
  divergences: string[]; // where historical parallel breaks down
}

export interface Hypothesis {
  id: string;
  statement: string;
  predictedOutcome: string;
  probabilityIfTrue: number; // P(evidence|hypothesis)
  priorProbability: number; // P(hypothesis) before evidence
  posteriorProbability: number; // P(hypothesis|evidence) after update
  testableImplications: string[];
  falsificationCriteria: string[];
  status: 'proposed' | 'testing' | 'supported' | 'refuted' | 'indeterminate';
}

export interface Derivative {
  id: string;
  variable: string;
  firstDerivative: number; // rate of change
  secondDerivative: number; // acceleration/deceleration
  trendDirection: 'accelerating' | 'decelerating' | 'stable' | 'inflection';
  criticalThreshold?: number;
  timeToThreshold?: number; // estimated time units
}

export interface AblationResult {
  id: string;
  removedFactor: string;
  impactOnPrediction: number; // how much prediction changes
  essentiality: 'critical' | 'important' | 'minor' | 'negligible';
  interactionEffects: { factor: string; effect: number }[];
}

// ============================================================================
// FUZZY MATHEMATICS FOR UNCERTAINTY
// ============================================================================

export interface FuzzyNumber {
  low: number;
  peak: number;
  high: number;
  confidence: number;
}

export function fuzzyAdd(a: FuzzyNumber, b: FuzzyNumber): FuzzyNumber {
  return {
    low: a.low + b.low,
    peak: a.peak + b.peak,
    high: a.high + b.high,
    confidence: Math.min(a.confidence, b.confidence),
  };
}

export function fuzzyMultiply(a: FuzzyNumber, scalar: number): FuzzyNumber {
  return {
    low: a.low * scalar,
    peak: a.peak * scalar,
    high: a.high * scalar,
    confidence: a.confidence,
  };
}

export function fuzzyIntersect(a: FuzzyNumber, b: FuzzyNumber): FuzzyNumber {
  return {
    low: Math.max(a.low, b.low),
    peak: (a.peak + b.peak) / 2,
    high: Math.min(a.high, b.high),
    confidence: (a.confidence + b.confidence) / 2,
  };
}

export function defuzzify(fuzzy: FuzzyNumber): number {
  // Centroid defuzzification
  return (fuzzy.low + 2 * fuzzy.peak + fuzzy.high) / 4;
}

export function fuzzyRisk(
  probability: FuzzyNumber,
  impact: FuzzyNumber
): FuzzyNumber {
  return {
    low: probability.low * impact.low,
    peak: probability.peak * impact.peak,
    high: probability.high * impact.high,
    confidence: Math.min(probability.confidence, impact.confidence),
  };
}

// ============================================================================
// NSM→XYZA PIPELINE
// ============================================================================

export interface NSMSignal {
  id: string;
  rawContent: string;
  domain: string;
  timestamp: string;
  entities: string[];
  initialRiskScore: number;
}

export interface XYZAOutput {
  signal: NSMSignal;

  // X: eXtract - key variables identified
  extractedVariables: {
    name: string;
    type: 'actor' | 'action' | 'location' | 'resource' | 'temporal' | 'structural';
    value: string | number;
    uncertainty: FuzzyNumber;
  }[];

  // Y: Yoke - historical correlates found
  yokedHistoricalEvents: HistoricalCorrelate[];

  // Z: Zero-in - causal mechanisms identified
  causalMechanisms: {
    mechanism: string;
    confidence: number;
    supportingEvidence: string[];
    counterfactual: string; // what would change if mechanism didn't exist
  }[];

  // A: Ablate - sensitivity analysis
  ablationResults: AblationResult[];

  // Final synthesis
  synthesizedRiskScore: FuzzyNumber;
  knowledgeGaps: KnowledgeGap[];
  recommendedActions: string[];
}

export interface KnowledgeGap {
  id: string;
  description: string;
  quadrant: KnowledgeQuadrant;
  priority: 'critical' | 'high' | 'medium' | 'low';
  proposedResolution: string;
  estimatedEffortHours: number;
}

// ============================================================================
// EPISTEMIC HUMILITY PROOFS
// ============================================================================

/**
 * Epistemic Humility Proof System
 *
 * AXIOM 1: All predictions have uncertainty
 * AXIOM 2: Historical correlates provide bounds, not certainties
 * AXIOM 3: Unknown unknowns exist in all complex systems
 * AXIOM 4: Confidence should decrease with prediction horizon
 *
 * THEOREM 1: No prediction of complex human systems can exceed 0.95 confidence
 * PROOF: Given Axiom 3 (unknown unknowns exist), there is always ε > 0.05
 *        probability of unmeasured factors affecting outcomes.
 *
 * THEOREM 2: Historical correlation ≠ Future causation
 * PROOF: Given structural changes (technology, norms, institutions),
 *        historical correlates provide analogical bounds only.
 *
 * THEOREM 3: Model uncertainty compounds with system complexity
 * PROOF: For n interdependent variables with individual uncertainty σ,
 *        system uncertainty ≥ σ√n (assuming independence, higher if correlated).
 */

export interface EpistemicProof {
  id: string;
  name: string;
  statement: string;
  axioms: string[];
  derivationSteps: string[];
  confidenceBound: number;
  applicableDomains: string[];
}

export const EPISTEMIC_PROOFS: EpistemicProof[] = [
  {
    id: 'proof-1',
    name: 'Maximum Confidence Bound',
    statement: 'No prediction of complex human systems can exceed 0.95 confidence',
    axioms: [
      'All predictions have uncertainty',
      'Unknown unknowns exist in all complex systems',
    ],
    derivationSteps: [
      'Let P be a prediction about complex human system S',
      'By Axiom 3, there exist factors F_unknown affecting S that are not in our model',
      'Let ε = P(F_unknown materially affects outcome)',
      'ε ≥ 0.05 for any sufficiently complex system (empirical bound from Black Swan events)',
      'Therefore, max(confidence(P)) = 1 - ε ≤ 0.95',
    ],
    confidenceBound: 0.95,
    applicableDomains: ['geopolitical', 'economic', 'military', 'social'],
  },
  {
    id: 'proof-2',
    name: 'Temporal Confidence Decay',
    statement: 'Prediction confidence decays exponentially with time horizon',
    axioms: [
      'All predictions have uncertainty',
      'Confidence should decrease with prediction horizon',
    ],
    derivationSteps: [
      'Let C(t) be confidence at time t',
      'Let λ be the decay constant (domain-dependent)',
      'C(t) = C(0) * e^(-λt)',
      'For geopolitical events, empirical λ ≈ 0.1/month',
      'At t=12 months, C(12) ≈ C(0) * 0.30',
    ],
    confidenceBound: 0.30,
    applicableDomains: ['geopolitical', 'economic'],
  },
  {
    id: 'proof-3',
    name: 'Cascade Uncertainty Amplification',
    statement: 'Cascade predictions have compounding uncertainty',
    axioms: [
      'Model uncertainty compounds with system complexity',
    ],
    derivationSteps: [
      'Let cascade C have n sequential stages',
      'Let σ_i be uncertainty at stage i',
      'Total uncertainty σ_total = √(Σσ_i²) for independent stages',
      'For correlated stages: σ_total = √(Σσ_i² + 2ΣΣρ_ij*σ_i*σ_j)',
      'As n increases, σ_total dominates the prediction',
    ],
    confidenceBound: 0.50,
    applicableDomains: ['cascade', 'systemic'],
  },
];

// ============================================================================
// HISTORIAN AGENT INTERFACE
// ============================================================================

export interface HistorianQuery {
  modernEvent: string;
  domain: string;
  keyActors: string[];
  keyVariables: string[];
  excludeAfterYear: number; // e.g., 1524 for "500 years ago"
  maxResults: number;
}

export interface HistorianResponse {
  query: HistorianQuery;
  correlates: HistoricalCorrelate[];
  patterns: {
    pattern: string;
    instances: string[];
    frequency: string;
    lastOccurrence: string;
  }[];
  warnings: string[]; // divergences from historical patterns
  synthesisPrompt: string; // for feeding back to modern agents
}

// Curated historical events database (500+ years ago)
export const HISTORICAL_EVENTS_DATABASE = [
  // Ancient & Classical
  {
    id: 'peloponnesian-war',
    name: 'Peloponnesian War',
    period: '431-404 BC',
    yearsAgo: 2455,
    keywords: ['hegemonic rivalry', 'alliance systems', 'democracy vs oligarchy', 'naval power'],
    causalPattern: 'Rising power threatens established power → preventive war',
    outcome: 'Mutual exhaustion, third-party gains (Persia, later Macedon)',
  },
  {
    id: 'fall-of-rome',
    name: 'Fall of Western Roman Empire',
    period: '376-476 AD',
    yearsAgo: 1549,
    keywords: ['imperial overstretch', 'border pressure', 'economic decline', 'currency debasement'],
    causalPattern: 'Overextension + migration pressure + fiscal crisis → collapse',
    outcome: 'Fragmentation, dark age, long-term institutional memory loss',
  },
  {
    id: 'mongol-invasions',
    name: 'Mongol Invasions',
    period: '1206-1368',
    yearsAgo: 818,
    keywords: ['rapid conquest', 'steppe warfare', 'psychological warfare', 'disease spread'],
    causalPattern: 'Unified nomadic power + military innovation → rapid expansion',
    outcome: 'Eurasian trade integration, population collapse, institutional destruction',
  },
  {
    id: 'black-death',
    name: 'Black Death',
    period: '1346-1353',
    yearsAgo: 678,
    keywords: ['pandemic', 'trade routes', 'social upheaval', 'labor shortage'],
    causalPattern: 'Disease + trade networks + urban density → mass mortality',
    outcome: 'Labor power shift, religious crisis, long-term wage gains',
  },
  {
    id: 'ottoman-rise',
    name: 'Ottoman Rise and Fall of Constantinople',
    period: '1299-1453',
    yearsAgo: 571,
    keywords: ['siege warfare', 'gunpowder', 'religious conflict', 'trade route control'],
    causalPattern: 'Military innovation + declining opponent → regime change',
    outcome: 'Trade route disruption, Renaissance acceleration, religious polarization',
  },
  // Renaissance & Early Modern
  {
    id: 'italian-wars',
    name: 'Italian Wars',
    period: '1494-1559',
    yearsAgo: 531,
    keywords: ['balance of power', 'mercenaries', 'foreign intervention', 'city-state rivalry'],
    causalPattern: 'Power vacuum + external intervention → prolonged conflict',
    outcome: 'Habsburg dominance, end of Italian independence, diplomatic innovation',
  },
  {
    id: 'reformation',
    name: 'Protestant Reformation',
    period: '1517-1555',
    yearsAgo: 508,
    keywords: ['religious schism', 'printing press', 'political fragmentation', 'ideological warfare'],
    causalPattern: 'Information technology + elite dissatisfaction + popular grievance → revolution',
    outcome: 'Permanent religious split, wars of religion, state-church reconfiguration',
  },
  {
    id: 'ming-treasure-voyages',
    name: 'Ming Treasure Voyages & Withdrawal',
    period: '1405-1433',
    yearsAgo: 620,
    keywords: ['naval power', 'isolationism', 'bureaucratic politics', 'strategic choice'],
    causalPattern: 'Capability + choice not to use → strategic withdrawal',
    outcome: 'Power vacuum in Indian Ocean, European opportunity',
  },
  {
    id: 'spanish-conquest',
    name: 'Spanish Conquest of Americas',
    period: '1492-1572',
    yearsAgo: 533,
    keywords: ['technological asymmetry', 'disease', 'alliance exploitation', 'resource extraction'],
    causalPattern: 'Technology gap + disease + local divisions → rapid conquest',
    outcome: 'Demographic collapse, silver inflation, global trade transformation',
  },
  // Economic & Financial
  {
    id: 'tulip-mania',
    name: 'Dutch Tulip Mania',
    period: '1634-1637',
    yearsAgo: 391,
    keywords: ['speculative bubble', 'futures contracts', 'irrational exuberance', 'crash'],
    causalPattern: 'Easy credit + novel asset + social mania → bubble → crash',
    outcome: 'Modest economic impact, cultural memory of speculation',
  },
  {
    id: 'south-sea-bubble',
    name: 'South Sea Bubble',
    period: '1720',
    yearsAgo: 305,
    keywords: ['debt swap', 'stock manipulation', 'government complicity', 'panic'],
    causalPattern: 'Debt crisis + financial innovation + fraud → crash',
    outcome: 'Regulatory response, long-term investor caution',
  },
];

// ============================================================================
// SIMULATION PSEUDOCODE FRAMEWORK
// ============================================================================

/**
 * EPISTEMIC SIMULATION PIPELINE (Pseudocode)
 *
 * 1. INITIALIZE
 *    - Load current signals from all domain agents
 *    - Query historian agent for correlates (500+ year filter)
 *    - Build variable graph from extracted features
 *
 * 2. HYPOTHESIS GENERATION
 *    FOR each signal S:
 *      - Extract key variables V = {v1, v2, ..., vn}
 *      - Find historical correlates H = historian.query(S)
 *      - Generate hypotheses:
 *        H1: "S follows pattern of [historical event]"
 *        H2: "S diverges from history due to [novel factor]"
 *        H3: "S represents unknown unknown category"
 *
 * 3. DERIVATIVE CALCULATION
 *    FOR each variable v in V:
 *      - Calculate dv/dt (first derivative)
 *      - Calculate d²v/dt² (second derivative)
 *      - Identify inflection points
 *      - Estimate time-to-threshold
 *
 * 4. CAUSAL CHAIN CONSTRUCTION
 *    - Build DAG from V with edges E = {(vi, vj, strength)}
 *    - Apply Granger causality tests where temporal data available
 *    - Identify feedback loops
 *    - Calculate cascade probabilities
 *
 * 5. SYNTHETIC DATA GENERATION
 *    FOR each hypothesis H:
 *      - Generate synthetic scenarios S' = perturb(S, ε)
 *      - Vary key parameters across fuzzy ranges
 *      - Create 1000 Monte Carlo samples
 *
 * 6. SIMULATION EXECUTION
 *    FOR each synthetic scenario S':
 *      - Run forward simulation 12 time steps
 *      - Record outcome trajectories
 *      - Track which variables drive outcomes
 *
 * 7. ABLATION TESTING
 *    FOR each variable v in V:
 *      - Remove v from model
 *      - Re-run simulations
 *      - Measure ΔPrediction
 *      - Classify essentiality
 *
 * 8. KNOWLEDGE GAP DETECTION
 *    - Compare simulation variance to historical variance
 *    - Flag high-variance regions as "unknown unknowns"
 *    - Extract implicit patterns as "unknown knowns"
 *    - Log unanswered questions as "known unknowns"
 *
 * 9. PROOF VERIFICATION
 *    FOR each prediction P:
 *      - Verify P.confidence ≤ EPISTEMIC_PROOFS['max-confidence'].bound
 *      - Apply temporal decay: P.confidence *= e^(-λt)
 *      - Apply cascade amplification if applicable
 *
 * 10. OUTPUT SYNTHESIS
 *     - Aggregate simulation results
 *     - Apply epistemic bounds
 *     - Generate actionable recommendations
 *     - Flag knowledge gaps for investigation
 */

export interface SimulationConfig {
  signalId: string;
  timeSteps: number;
  monteCarloSamples: number;
  ablationEnabled: boolean;
  historianEnabled: boolean;
  decayLambda: number;
}

export interface SimulationResult {
  configId: string;
  executedAt: string;

  // Trajectory distribution
  outcomeDistribution: {
    scenario: string;
    probability: FuzzyNumber;
    keyDrivers: string[];
  }[];

  // Epistemic state
  knowledgeState: {
    knownKnowns: EpistemicClaim[];
    knownUnknowns: KnowledgeGap[];
    unknownUnknowns: string[]; // anomaly descriptions
    unknownKnowns: string[]; // implicit patterns detected
  };

  // Confidence after all proofs applied
  boundedConfidence: number;

  // Recommended next steps
  recommendations: {
    action: string;
    priority: 'critical' | 'high' | 'medium' | 'low';
    fillsGap?: string;
  }[];
}

// ============================================================================
// INTEGRATION HOOKS
// ============================================================================

export function applyEpistemicBounds(confidence: number, context: {
  timeHorizonMonths: number;
  cascadeSteps: number;
  domainComplexity: 'simple' | 'complex' | 'chaotic';
}): number {
  // Apply maximum confidence bound
  let bounded = Math.min(confidence, 0.95);

  // Apply temporal decay
  const lambda = 0.1; // per month
  bounded *= Math.exp(-lambda * context.timeHorizonMonths);

  // Apply cascade amplification
  if (context.cascadeSteps > 1) {
    const cascadeUncertainty = 1 - Math.pow(0.9, context.cascadeSteps);
    bounded *= (1 - cascadeUncertainty);
  }

  // Apply domain complexity factor
  const complexityFactor = {
    simple: 1.0,
    complex: 0.85,
    chaotic: 0.7,
  };
  bounded *= complexityFactor[context.domainComplexity];

  return Math.max(0.05, bounded); // minimum 5% confidence
}

export function detectKnowledgeQuadrant(claim: string, context: {
  hasEvidence: boolean;
  isFormalized: boolean;
  isAnomalous: boolean;
}): KnowledgeQuadrant {
  if (context.hasEvidence && context.isFormalized) {
    return 'known_known';
  }
  if (!context.hasEvidence && context.isFormalized) {
    return 'known_unknown';
  }
  if (!context.hasEvidence && !context.isFormalized && context.isAnomalous) {
    return 'unknown_unknown';
  }
  // Has evidence but not formalized = implicit knowledge
  return 'unknown_known';
}
