/**
 * Dempster-Shafer Evidence Theory Engine
 *
 * THEORETICAL BASIS:
 * Dempster-Shafer theory extends Bayesian probability to handle
 * uncertainty and ignorance explicitly through "belief" and "plausibility".
 *
 * Frame of Discernment: Θ = {θ₁, θ₂, ..., θₙ}
 * Mass Function: m: 2^Θ → [0,1], Σm(A) = 1
 *
 * Belief: Bel(A) = Σ_{B⊆A} m(B)  (certainty that A is true)
 * Plausibility: Pl(A) = 1 - Bel(¬A) (maximum possible belief)
 *
 * Dempster's Rule: m₁⊕m₂(A) = (Σ_{B∩C=A} m₁(B)m₂(C)) / (1 - K)
 * where K = Σ_{B∩C=∅} m₁(B)m₂(C) (conflict)
 *
 * APPLICATION: Intelligence fusion from multiple uncertain sources
 * - Each source provides mass on regime states
 * - Combination produces fused assessment with conflict measure
 */

export type FrameElement = string;
export type Subset = Set<FrameElement>;

export interface MassAssignment {
  subset: FrameElement[];
  mass: number;
}

export interface BeliefFunction {
  frame: FrameElement[];
  masses: MassAssignment[];
  totalMass: number;
}

export interface FusionResult {
  fused: BeliefFunction;
  conflict: number;          // K: degree of conflict between sources
  uncertainty: number;       // m(Θ): mass on full frame
  sources: number;           // Number of sources combined
}

export interface BeliefInterval {
  element: FrameElement;
  belief: number;           // Lower bound (certainty)
  plausibility: number;     // Upper bound (possibility)
  uncertainty: number;      // Pl - Bel (ignorance)
}

/**
 * Create a belief function from mass assignments
 */
export function createBeliefFunction(
  frame: FrameElement[],
  assignments: Array<{ elements: FrameElement[]; mass: number }>
): BeliefFunction {
  const masses: MassAssignment[] = assignments.map(a => ({
    subset: a.elements,
    mass: a.mass,
  }));

  const totalMass = masses.reduce((sum, m) => sum + m.mass, 0);

  // Normalize if needed
  if (Math.abs(totalMass - 1) > 0.01) {
    masses.forEach(m => {
      m.mass = m.mass / totalMass;
    });
  }

  return { frame, masses, totalMass: 1 };
}

/**
 * Check if subset A is contained in subset B
 */
function isSubset(A: FrameElement[], B: FrameElement[]): boolean {
  return A.every(e => B.includes(e));
}

/**
 * Compute intersection of two subsets
 */
function intersection(A: FrameElement[], B: FrameElement[]): FrameElement[] {
  return A.filter(e => B.includes(e));
}

/**
 * Compute Belief: Bel(A) = Σ_{B⊆A} m(B)
 */
export function computeBelief(bf: BeliefFunction, A: FrameElement[]): number {
  return bf.masses
    .filter(m => isSubset(m.subset, A))
    .reduce((sum, m) => sum + m.mass, 0);
}

/**
 * Compute Plausibility: Pl(A) = Σ_{B∩A≠∅} m(B)
 */
export function computePlausibility(bf: BeliefFunction, A: FrameElement[]): number {
  return bf.masses
    .filter(m => intersection(m.subset, A).length > 0)
    .reduce((sum, m) => sum + m.mass, 0);
}

/**
 * Compute belief intervals for all singletons
 */
export function computeBeliefIntervals(bf: BeliefFunction): BeliefInterval[] {
  return bf.frame.map(element => {
    const singleton = [element];
    const bel = computeBelief(bf, singleton);
    const pl = computePlausibility(bf, singleton);

    return {
      element,
      belief: bel,
      plausibility: pl,
      uncertainty: pl - bel,
    };
  });
}

/**
 * Dempster's Rule of Combination: m₁ ⊕ m₂
 *
 * Combines two independent sources of evidence
 */
export function dempsterCombine(
  bf1: BeliefFunction,
  bf2: BeliefFunction
): FusionResult {
  // Verify same frame
  if (bf1.frame.length !== bf2.frame.length) {
    throw new Error('Belief functions must have same frame of discernment');
  }

  // Compute all pairwise products
  const products: Array<{ subset: FrameElement[]; mass: number }> = [];
  let conflict = 0;

  for (const m1 of bf1.masses) {
    for (const m2 of bf2.masses) {
      const inter = intersection(m1.subset, m2.subset);
      const product = m1.mass * m2.mass;

      if (inter.length === 0) {
        conflict += product;
      } else {
        products.push({ subset: inter, mass: product });
      }
    }
  }

  // Normalize by (1 - K)
  const normalization = 1 - conflict;

  if (normalization < 0.01) {
    // High conflict - sources disagree strongly
    console.warn('High conflict in Dempster combination:', conflict);
  }

  // Aggregate masses for same subsets
  const massMap = new Map<string, number>();
  for (const p of products) {
    const key = p.subset.sort().join(',');
    massMap.set(key, (massMap.get(key) || 0) + p.mass / normalization);
  }

  const fusedMasses: MassAssignment[] = Array.from(massMap.entries()).map(
    ([key, mass]) => ({
      subset: key.split(',').filter(s => s.length > 0),
      mass,
    })
  );

  // Compute uncertainty (mass on full frame)
  const uncertainty = fusedMasses
    .filter(m => m.subset.length === bf1.frame.length)
    .reduce((sum, m) => sum + m.mass, 0);

  return {
    fused: {
      frame: bf1.frame,
      masses: fusedMasses,
      totalMass: 1,
    },
    conflict,
    uncertainty,
    sources: 2,
  };
}

/**
 * Combine multiple sources iteratively
 */
export function fuseMultipleSources(
  beliefFunctions: BeliefFunction[]
): FusionResult {
  if (beliefFunctions.length === 0) {
    throw new Error('At least one belief function required');
  }

  if (beliefFunctions.length === 1) {
    const bf = beliefFunctions[0];
    const uncertainty = bf.masses
      .filter(m => m.subset.length === bf.frame.length)
      .reduce((sum, m) => sum + m.mass, 0);

    return {
      fused: bf,
      conflict: 0,
      uncertainty,
      sources: 1,
    };
  }

  let result = dempsterCombine(beliefFunctions[0], beliefFunctions[1]);
  let totalConflict = result.conflict;

  for (let i = 2; i < beliefFunctions.length; i++) {
    result = dempsterCombine(result.fused, beliefFunctions[i]);
    totalConflict = Math.max(totalConflict, result.conflict);
  }

  return {
    ...result,
    conflict: totalConflict,
    sources: beliefFunctions.length,
  };
}

/**
 * Pignistic Probability Transform: Convert belief to probability
 *
 * BetP(x) = Σ_{A∋x} m(A) / |A|
 *
 * Used for decision-making when a single probability is needed
 */
export function pignisticTransform(bf: BeliefFunction): Map<FrameElement, number> {
  const probs = new Map<FrameElement, number>();

  // Initialize
  for (const element of bf.frame) {
    probs.set(element, 0);
  }

  // Distribute mass
  for (const m of bf.masses) {
    const share = m.mass / m.subset.length;
    for (const element of m.subset) {
      probs.set(element, (probs.get(element) || 0) + share);
    }
  }

  return probs;
}

/**
 * Compute entropy of belief function
 *
 * Measures total uncertainty including ignorance
 */
export function beliefEntropy(bf: BeliefFunction): number {
  const intervals = computeBeliefIntervals(bf);

  // Aggregate uncertainty across all elements
  const totalUncertainty = intervals.reduce((sum, i) => sum + i.uncertainty, 0);

  // Shannon-like entropy from pignistic probabilities
  const pignistic = pignisticTransform(bf);
  let shannonEntropy = 0;
  for (const p of pignistic.values()) {
    if (p > 0) {
      shannonEntropy -= p * Math.log2(p);
    }
  }

  // Combined metric
  return shannonEntropy + 0.5 * totalUncertainty;
}

/**
 * Create belief function for regime assessment
 *
 * Frame: {STABLE, VOLATILE, CRISIS}
 */
export const REGIME_FRAME: FrameElement[] = ['STABLE', 'VOLATILE', 'CRISIS'];

export interface SourceAssessment {
  sourceId: string;
  reliability: number;       // 0-1: Source credibility
  stableConfidence: number;  // 0-1: Confidence in stable regime
  volatileConfidence: number;
  crisisConfidence: number;
  uncertainty: number;       // Remaining mass on Θ
}

export function assessmentToBeliefFunction(
  assessment: SourceAssessment
): BeliefFunction {
  const { reliability, stableConfidence, volatileConfidence, crisisConfidence, uncertainty } = assessment;

  // Discount by reliability
  const discount = reliability;

  const assignments: Array<{ elements: FrameElement[]; mass: number }> = [
    { elements: ['STABLE'], mass: stableConfidence * discount },
    { elements: ['VOLATILE'], mass: volatileConfidence * discount },
    { elements: ['CRISIS'], mass: crisisConfidence * discount },
    // Remaining goes to full frame (ignorance)
    { elements: REGIME_FRAME, mass: uncertainty * discount + (1 - discount) },
  ];

  return createBeliefFunction(REGIME_FRAME, assignments);
}

/**
 * Fuse multiple intelligence source assessments
 */
export function fuseIntelligenceAssessments(
  assessments: SourceAssessment[]
): {
  fusedBelief: BeliefFunction;
  beliefIntervals: BeliefInterval[];
  probabilities: Map<FrameElement, number>;
  conflict: number;
  recommendation: FrameElement;
  confidence: number;
} {
  const beliefFunctions = assessments.map(assessmentToBeliefFunction);
  const { fused, conflict } = fuseMultipleSources(beliefFunctions);

  const intervals = computeBeliefIntervals(fused);
  const probs = pignisticTransform(fused);

  // Find most probable state
  let maxProb = 0;
  let recommendation: FrameElement = 'STABLE';
  for (const [element, prob] of probs.entries()) {
    if (prob > maxProb) {
      maxProb = prob;
      recommendation = element;
    }
  }

  // Confidence is belief in recommendation
  const confidence = computeBelief(fused, [recommendation]);

  return {
    fusedBelief: fused,
    beliefIntervals: intervals,
    probabilities: probs,
    conflict,
    recommendation,
    confidence,
  };
}

/**
 * Compute conflict between two specific sources
 */
export function pairwiseConflict(
  assessment1: SourceAssessment,
  assessment2: SourceAssessment
): number {
  const bf1 = assessmentToBeliefFunction(assessment1);
  const bf2 = assessmentToBeliefFunction(assessment2);
  const { conflict } = dempsterCombine(bf1, bf2);
  return conflict;
}

/**
 * Identify which sources are in conflict
 */
export function identifyConflictingSources(
  assessments: SourceAssessment[],
  threshold: number = 0.3
): Array<{ source1: string; source2: string; conflict: number }> {
  const conflicts: Array<{ source1: string; source2: string; conflict: number }> = [];

  for (let i = 0; i < assessments.length; i++) {
    for (let j = i + 1; j < assessments.length; j++) {
      const conflict = pairwiseConflict(assessments[i], assessments[j]);
      if (conflict > threshold) {
        conflicts.push({
          source1: assessments[i].sourceId,
          source2: assessments[j].sourceId,
          conflict,
        });
      }
    }
  }

  return conflicts.sort((a, b) => b.conflict - a.conflict);
}
