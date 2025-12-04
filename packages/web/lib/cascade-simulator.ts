/**
 * LatticeForge Cascade Effects Simulator
 *
 * Higher-order chain-of-effects step sequencer for simulating:
 * - Historical events (what happened and why)
 * - Current events (what's transpiring)
 * - Future scenarios (what might happen based on risk inputs)
 *
 * WARNING: This is a ROUGH simulation with significant limitations.
 * Real-world outcomes depend on countless variables we cannot model.
 * Use for analytical exploration, not prediction.
 */

// ============================================================================
// TYPES
// ============================================================================

export interface RiskInputs {
  /** User's risk tolerance (0-1): 0 = very risk averse, 1 = high risk acceptance */
  riskTolerance: number;
  /** Confidence in data quality (0-1) */
  dataConfidence: number;
  /** Time horizon in days */
  timeHorizon: number;
  /** Consider black swan events */
  includeBlackSwans: boolean;
  /** Consider cascade/contagion effects */
  includeCascades: boolean;
  /** Custom scenario assumptions */
  assumptions?: string[];
}

export interface CascadeStep {
  /** Step number in the chain */
  order: number;
  /** Type of effect */
  effectType: 'political' | 'economic' | 'social' | 'military' | 'technological' | 'environmental';
  /** Who is affected */
  who: string[];
  /** What happens */
  what: string;
  /** When (relative to trigger or absolute date) */
  when: string;
  /** Where (geographic scope) */
  where: string[];
  /** Why this follows from previous step */
  why: string;
  /** How the mechanism works */
  how: string;
  /** Confidence in this step (0-1) */
  confidence: number;
  /** Verbal confidence description */
  confidenceLabel: 'Very Low' | 'Low' | 'Moderate' | 'High' | 'Very High';
  /** Time delay from previous step */
  delayDays: { min: number; max: number };
  /** Can this step be prevented/mitigated? */
  preventable: boolean;
  /** Mitigation strategies if preventable */
  mitigations?: string[];
}

export interface Scenario {
  /** Unique identifier */
  id: string;
  /** Scenario name */
  name: string;
  /** Probability of this scenario path */
  probability: number;
  /** Verbal probability */
  probabilityLabel: string;
  /** Chain of effects */
  steps: CascadeStep[];
  /** Overall impact assessment */
  impact: {
    economic: 'Negligible' | 'Minor' | 'Moderate' | 'Severe' | 'Catastrophic';
    political: 'Negligible' | 'Minor' | 'Moderate' | 'Severe' | 'Catastrophic';
    social: 'Negligible' | 'Minor' | 'Moderate' | 'Severe' | 'Catastrophic';
    military: 'Negligible' | 'Minor' | 'Moderate' | 'Severe' | 'Catastrophic';
  };
  /** Key uncertainties that could change this scenario */
  uncertainties: string[];
  /** Historical analogues */
  analogues?: { event: string; year: number; similarity: number }[];
}

export interface SimulationResult {
  /** Trigger event that starts the cascade */
  trigger: {
    what: string;
    when: string;
    where: string;
    confidence: number;
  };
  /** Multiple possible scenario paths */
  scenarios: Scenario[];
  /** Risk inputs used */
  inputs: RiskInputs;
  /** Simulation timestamp */
  timestamp: string;
  /** Disclaimers and warnings */
  warnings: string[];
  /** Model limitations */
  limitations: string[];
  /** Confidence calibration note */
  calibrationNote: string;
}

export type SimulationMode = 'historical' | 'current' | 'future' | 'fusion';

// ============================================================================
// CORE SIMULATION ENGINE
// ============================================================================

/**
 * Main cascade simulation function
 */
export function simulateCascade(
  mode: SimulationMode,
  triggerEvent: string,
  affectedRegions: string[],
  inputs: RiskInputs
): SimulationResult {
  const timestamp = new Date().toISOString();

  // Base warnings that always apply
  const warnings = getWarnings(mode, inputs);
  const limitations = getLimitations();
  const calibrationNote = getCalibrationNote(inputs.dataConfidence);

  // Generate scenarios based on mode
  const scenarios = generateScenarios(mode, triggerEvent, affectedRegions, inputs);

  return {
    trigger: {
      what: triggerEvent,
      when: mode === 'historical' ? 'Past event' : mode === 'current' ? 'Ongoing' : `Next ${inputs.timeHorizon} days`,
      where: affectedRegions.join(', '),
      confidence: inputs.dataConfidence * 0.9, // Slightly discount trigger confidence
    },
    scenarios,
    inputs,
    timestamp,
    warnings,
    limitations,
    calibrationNote,
  };
}

// ============================================================================
// SCENARIO GENERATION
// ============================================================================

function generateScenarios(
  mode: SimulationMode,
  trigger: string,
  regions: string[],
  inputs: RiskInputs
): Scenario[] {
  // Generate 2-4 scenarios based on complexity
  const numScenarios = inputs.includeBlackSwans ? 4 : 3;
  const scenarios: Scenario[] = [];

  // Scenario A: Most likely path (base case)
  scenarios.push(generateBaseScenario(trigger, regions, inputs));

  // Scenario B: Escalation path
  scenarios.push(generateEscalationScenario(trigger, regions, inputs));

  // Scenario C: De-escalation/resolution path
  scenarios.push(generateResolutionScenario(trigger, regions, inputs));

  // Scenario D: Black swan (if enabled)
  if (inputs.includeBlackSwans) {
    scenarios.push(generateBlackSwanScenario(trigger, regions, inputs));
  }

  // Normalize probabilities
  const totalProb = scenarios.reduce((sum, s) => sum + s.probability, 0);
  scenarios.forEach(s => {
    s.probability = s.probability / totalProb;
    s.probabilityLabel = getProbabilityLabel(s.probability);
  });

  return scenarios.sort((a, b) => b.probability - a.probability);
}

function generateBaseScenario(trigger: string, regions: string[], inputs: RiskInputs): Scenario {
  const steps = generateCascadeSteps('base', trigger, regions, inputs);

  return {
    id: 'scenario-a-base',
    name: 'Base Case: Managed Tension',
    probability: 0.45 * inputs.dataConfidence,
    probabilityLabel: 'Likely',
    steps,
    impact: {
      economic: 'Moderate',
      political: 'Moderate',
      social: 'Minor',
      military: 'Negligible',
    },
    uncertainties: [
      'Leadership decision-making under pressure',
      'Effectiveness of diplomatic channels',
      'Economic resilience to shocks',
      'Media/information environment',
    ],
    analogues: [
      { event: 'Cuban Missile Crisis resolution', year: 1962, similarity: 0.6 },
      { event: 'Taiwan Strait Crisis management', year: 1996, similarity: 0.5 },
    ],
  };
}

function generateEscalationScenario(trigger: string, regions: string[], inputs: RiskInputs): Scenario {
  const steps = generateCascadeSteps('escalation', trigger, regions, inputs);

  return {
    id: 'scenario-b-escalation',
    name: 'Escalation: Spiral Dynamics',
    probability: 0.25 * (1 - inputs.riskTolerance * 0.3),
    probabilityLabel: 'Possible',
    steps,
    impact: {
      economic: 'Severe',
      political: 'Severe',
      social: 'Moderate',
      military: 'Moderate',
    },
    uncertainties: [
      'Miscalculation or accident risk',
      'Domestic political pressures on leaders',
      'Third-party intervention',
      'Information fog and misperception',
    ],
    analogues: [
      { event: 'July Crisis 1914', year: 1914, similarity: 0.4 },
      { event: 'Russo-Ukrainian escalation 2022', year: 2022, similarity: 0.7 },
    ],
  };
}

function generateResolutionScenario(trigger: string, regions: string[], inputs: RiskInputs): Scenario {
  const steps = generateCascadeSteps('resolution', trigger, regions, inputs);

  return {
    id: 'scenario-c-resolution',
    name: 'Resolution: Diplomatic Off-Ramp',
    probability: 0.20 * (1 + inputs.riskTolerance * 0.2),
    probabilityLabel: 'Possible',
    steps,
    impact: {
      economic: 'Minor',
      political: 'Minor',
      social: 'Negligible',
      military: 'Negligible',
    },
    uncertainties: [
      'Willingness of parties to compromise',
      'Face-saving mechanisms availability',
      'Third-party mediator effectiveness',
      'Economic incentives alignment',
    ],
    analogues: [
      { event: 'Iran Nuclear Deal (JCPOA)', year: 2015, similarity: 0.5 },
      { event: 'Korean Armistice negotiations', year: 1953, similarity: 0.4 },
    ],
  };
}

function generateBlackSwanScenario(trigger: string, regions: string[], inputs: RiskInputs): Scenario {
  const steps = generateCascadeSteps('blackswan', trigger, regions, inputs);

  return {
    id: 'scenario-d-blackswan',
    name: 'Black Swan: Discontinuous Shock',
    probability: 0.05,
    probabilityLabel: 'Unlikely but Consequential',
    steps,
    impact: {
      economic: 'Catastrophic',
      political: 'Catastrophic',
      social: 'Severe',
      military: 'Severe',
    },
    uncertainties: [
      'By definition, black swans are unpredictable',
      'Second and third-order effects highly uncertain',
      'Historical analogues may not apply',
      'System behavior in extreme stress unknown',
    ],
    analogues: [
      { event: 'COVID-19 pandemic onset', year: 2020, similarity: 0.3 },
      { event: 'Soviet Union collapse', year: 1991, similarity: 0.3 },
      { event: 'Lehman Brothers collapse', year: 2008, similarity: 0.4 },
    ],
  };
}

// ============================================================================
// CASCADE STEP GENERATION
// ============================================================================

function generateCascadeSteps(
  scenarioType: 'base' | 'escalation' | 'resolution' | 'blackswan',
  trigger: string,
  regions: string[],
  inputs: RiskInputs
): CascadeStep[] {
  const steps: CascadeStep[] = [];
  const numSteps = scenarioType === 'blackswan' ? 6 : scenarioType === 'escalation' ? 5 : 4;

  // Step templates based on scenario type
  const templates = getStepTemplates(scenarioType, trigger, regions);

  for (let i = 0; i < Math.min(numSteps, templates.length); i++) {
    const template = templates[i];
    const baseConfidence = template.confidence * inputs.dataConfidence;
    const decayFactor = Math.pow(0.85, i); // Confidence decays with each step

    steps.push({
      order: i + 1,
      effectType: template.effectType,
      who: template.who,
      what: template.what,
      when: template.when,
      where: template.where.length > 0 ? template.where : regions,
      why: template.why,
      how: template.how,
      confidence: baseConfidence * decayFactor,
      confidenceLabel: getConfidenceLabel(baseConfidence * decayFactor),
      delayDays: template.delayDays,
      preventable: template.preventable,
      mitigations: template.mitigations,
    });
  }

  return steps;
}

function getStepTemplates(
  scenarioType: string,
  trigger: string,
  regions: string[]
): Array<Omit<CascadeStep, 'order' | 'confidence' | 'confidenceLabel'> & { confidence: number }> {
  // Generic templates - in production, these would be domain-specific
  const baseTemplates = {
    base: [
      {
        effectType: 'political' as const,
        who: ['National leadership', 'Foreign ministries'],
        what: 'Diplomatic communications intensify',
        when: 'Within 24-72 hours',
        where: regions,
        why: 'Standard crisis response protocol',
        how: 'Back-channel communications, public statements, ally consultations',
        confidence: 0.8,
        delayDays: { min: 1, max: 3 },
        preventable: false,
      },
      {
        effectType: 'economic' as const,
        who: ['Financial markets', 'Investors', 'Corporations'],
        what: 'Market volatility increases, risk-off positioning',
        when: 'Within 1-2 trading days',
        where: ['Global financial centers'],
        why: 'Uncertainty pricing into asset valuations',
        how: 'Flight to safety (treasuries, gold), sector rotation, hedging',
        confidence: 0.75,
        delayDays: { min: 1, max: 2 },
        preventable: false,
      },
      {
        effectType: 'social' as const,
        who: ['Media', 'Public opinion'],
        what: 'Public attention and discourse shifts to event',
        when: 'Within 24 hours',
        where: regions,
        why: 'Media coverage drives public awareness',
        how: 'News cycles, social media amplification, expert commentary',
        confidence: 0.7,
        delayDays: { min: 0, max: 1 },
        preventable: false,
      },
      {
        effectType: 'political' as const,
        who: ['International organizations', 'Third-party states'],
        what: 'Third parties offer mediation, express concern',
        when: 'Within 1-2 weeks',
        where: ['UN', 'Regional bodies'],
        why: 'Standard multilateral response to tensions',
        how: 'Statements, resolutions, envoy dispatches',
        confidence: 0.65,
        delayDays: { min: 5, max: 14 },
        preventable: true,
        mitigations: ['Early bilateral resolution', 'Private face-saving deals'],
      },
    ],
    escalation: [
      {
        effectType: 'military' as const,
        who: ['Armed forces', 'Defense establishments'],
        what: 'Military readiness increases, force positioning',
        when: 'Within 48-96 hours',
        where: regions,
        why: 'Prudent preparation for potential conflict',
        how: 'Troop movements, alert level changes, logistics staging',
        confidence: 0.7,
        delayDays: { min: 2, max: 4 },
        preventable: true,
        mitigations: ['De-escalation signaling', 'Communication hotlines'],
      },
      {
        effectType: 'economic' as const,
        who: ['Central banks', 'Trade partners'],
        what: 'Economic countermeasures or sanctions considered',
        when: 'Within 1-2 weeks',
        where: ['Major economies'],
        why: 'Economic tools as coercion/deterrence',
        how: 'Sanctions packages, trade restrictions, financial measures',
        confidence: 0.6,
        delayDays: { min: 7, max: 14 },
        preventable: true,
        mitigations: ['Negotiated settlement', 'Phased de-escalation'],
      },
      {
        effectType: 'political' as const,
        who: ['Alliances', 'Treaty partners'],
        what: 'Alliance solidarity tested, commitments invoked',
        when: 'Within 2-4 weeks',
        where: ['Allied capitals'],
        why: 'Security commitments become relevant',
        how: 'Consultations, solidarity statements, capability offers',
        confidence: 0.55,
        delayDays: { min: 14, max: 28 },
        preventable: true,
        mitigations: ['Localization of dispute', 'Off-ramp acceptance'],
      },
      {
        effectType: 'military' as const,
        who: ['Military commands'],
        what: 'Limited military operations or incidents',
        when: 'If no de-escalation within 1 month',
        where: regions,
        why: 'Escalation dynamics, miscalculation, or deliberate probe',
        how: 'Border incidents, airspace violations, naval encounters',
        confidence: 0.4,
        delayDays: { min: 21, max: 45 },
        preventable: true,
        mitigations: ['Crisis communication protocols', 'Third-party intervention'],
      },
      {
        effectType: 'economic' as const,
        who: ['Global supply chains', 'Commodity markets'],
        what: 'Supply chain disruptions, commodity price spikes',
        when: 'Following any military action',
        where: ['Global'],
        why: 'Risk premia and actual disruption',
        how: 'Route avoidance, stockpiling, substitution',
        confidence: 0.5,
        delayDays: { min: 1, max: 7 },
        preventable: false,
      },
    ],
    resolution: [
      {
        effectType: 'political' as const,
        who: ['Leaders', 'Negotiators'],
        what: 'Face-saving formula found, tensions ease',
        when: 'Within 2-4 weeks',
        where: regions,
        why: 'Neither party benefits from continued standoff',
        how: 'Back-channel deals, phased steps, symbolic gestures',
        confidence: 0.5,
        delayDays: { min: 14, max: 30 },
        preventable: false,
      },
      {
        effectType: 'economic' as const,
        who: ['Markets', 'Investors'],
        what: 'Risk appetite recovers, volatility subsides',
        when: 'Following credible de-escalation',
        where: ['Global markets'],
        why: 'Uncertainty reduction repriced',
        how: 'Risk-on rotation, spread compression',
        confidence: 0.6,
        delayDays: { min: 1, max: 7 },
        preventable: false,
      },
      {
        effectType: 'social' as const,
        who: ['Public', 'Media'],
        what: 'Attention shifts, narrative moves to other issues',
        when: 'Within 1-2 weeks of resolution',
        where: ['Global'],
        why: 'News cycle dynamics',
        how: 'Coverage drops, new stories dominate',
        confidence: 0.7,
        delayDays: { min: 7, max: 14 },
        preventable: false,
      },
      {
        effectType: 'political' as const,
        who: ['Institutions', 'Policy makers'],
        what: 'Lessons learned, policy adjustments considered',
        when: 'Months after resolution',
        where: regions,
        why: 'Institutional adaptation',
        how: 'Reviews, reforms, new frameworks',
        confidence: 0.5,
        delayDays: { min: 30, max: 180 },
        preventable: false,
      },
    ],
    blackswan: [
      {
        effectType: 'military' as const,
        who: ['Unknown actors', 'Rogue elements'],
        what: 'Unexpected shock event fundamentally changes dynamics',
        when: 'Unpredictable',
        where: ['Uncertain'],
        why: 'Complex systems produce emergent behaviors',
        how: 'Mechanism unclear - that is the nature of black swans',
        confidence: 0.2,
        delayDays: { min: 1, max: 365 },
        preventable: false,
      },
      {
        effectType: 'political' as const,
        who: ['All actors'],
        what: 'Existing frameworks and assumptions invalidated',
        when: 'Immediately following shock',
        where: ['Global'],
        why: 'Paradigm shift in operating environment',
        how: 'Old playbooks no longer apply',
        confidence: 0.3,
        delayDays: { min: 0, max: 7 },
        preventable: false,
      },
      {
        effectType: 'economic' as const,
        who: ['All market participants'],
        what: 'Severe market dislocation, possible circuit breakers',
        when: 'Immediately',
        where: ['Global financial system'],
        why: 'Extreme uncertainty, liquidity crisis',
        how: 'Flight to cash, deleveraging, margin calls',
        confidence: 0.4,
        delayDays: { min: 0, max: 3 },
        preventable: false,
      },
      {
        effectType: 'social' as const,
        who: ['Global population'],
        what: 'Societal structures under unprecedented stress',
        when: 'Days to weeks',
        where: ['Global'],
        why: 'Institutional capacity tested beyond design parameters',
        how: 'Varies by shock type',
        confidence: 0.25,
        delayDays: { min: 3, max: 30 },
        preventable: false,
      },
      {
        effectType: 'political' as const,
        who: ['Governments', 'International system'],
        what: 'Emergency governance measures, potential regime changes',
        when: 'Weeks to months',
        where: ['Multiple countries'],
        why: 'Crisis exposes and exacerbates existing weaknesses',
        how: 'Emergency powers, political realignments',
        confidence: 0.2,
        delayDays: { min: 14, max: 90 },
        preventable: false,
      },
      {
        effectType: 'technological' as const,
        who: ['Technology sector', 'Critical infrastructure'],
        what: 'Cascading infrastructure failures or adaptations',
        when: 'Variable',
        where: ['Dependent on shock type'],
        why: 'Interconnected systems propagate effects',
        how: 'Supply chain breaks, grid stress, cyber effects',
        confidence: 0.15,
        delayDays: { min: 1, max: 60 },
        preventable: true,
        mitigations: ['Redundancy', 'Emergency protocols', 'Resilience planning'],
      },
    ],
  };

  return baseTemplates[scenarioType as keyof typeof baseTemplates] || baseTemplates.base;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function getConfidenceLabel(confidence: number): CascadeStep['confidenceLabel'] {
  if (confidence >= 0.8) return 'Very High';
  if (confidence >= 0.6) return 'High';
  if (confidence >= 0.4) return 'Moderate';
  if (confidence >= 0.2) return 'Low';
  return 'Very Low';
}

function getProbabilityLabel(probability: number): string {
  if (probability >= 0.7) return 'Very Likely';
  if (probability >= 0.5) return 'Likely';
  if (probability >= 0.3) return 'Possible';
  if (probability >= 0.1) return 'Unlikely';
  return 'Very Unlikely';
}

function getWarnings(mode: SimulationMode, inputs: RiskInputs): string[] {
  const warnings = [
    'CRITICAL DISCLAIMER: This is a ROUGH SIMULATION with significant limitations.',
    'Real-world outcomes depend on countless variables that cannot be fully modeled.',
    'Do NOT use this for actual prediction or decision-making without expert validation.',
    'Confidence levels are estimates and may be systematically biased.',
  ];

  if (mode === 'future') {
    warnings.push('Future scenarios are inherently speculative and highly uncertain.');
  }

  if (inputs.dataConfidence < 0.5) {
    warnings.push('LOW DATA CONFIDENCE: Results are particularly unreliable.');
  }

  if (inputs.includeBlackSwans) {
    warnings.push('Black swan events are by definition unpredictable - treat with extreme caution.');
  }

  if (inputs.timeHorizon > 90) {
    warnings.push('Long time horizons exponentially increase uncertainty.');
  }

  return warnings;
}

function getLimitations(): string[] {
  return [
    'Model does not account for individual leader psychology',
    'Incomplete data on private communications and intentions',
    'Cannot predict genuine innovation or "unknown unknowns"',
    'Assumes some degree of rational actor behavior',
    'Does not fully model feedback loops and emergent behavior',
    'Historical analogues may not apply to novel situations',
    'Confidence calibration not validated against outcomes',
    'Many relevant variables not included in model',
  ];
}

function getCalibrationNote(dataConfidence: number): string {
  if (dataConfidence >= 0.8) {
    return 'Calibration: Based on high-quality data sources. Still subject to model limitations.';
  } else if (dataConfidence >= 0.5) {
    return 'Calibration: Moderate data quality. Confidence intervals should be widened by user judgment.';
  } else {
    return 'Calibration: Low data quality. Treat all estimates as highly provisional.';
  }
}

// ============================================================================
// HISTORICAL EVENT LIBRARY
// ============================================================================

export const HISTORICAL_CASCADES = [
  {
    id: 'wwi-july-crisis',
    name: 'WWI July Crisis (1914)',
    trigger: 'Assassination of Archduke Franz Ferdinand',
    regions: ['Austria-Hungary', 'Serbia', 'Germany', 'Russia', 'France', 'Britain'],
    description: 'How a political assassination cascaded into world war through alliance obligations, mobilization timelines, and miscalculation.',
  },
  {
    id: 'asian-financial-crisis',
    name: 'Asian Financial Crisis (1997)',
    trigger: 'Thai baht devaluation',
    regions: ['Thailand', 'Indonesia', 'South Korea', 'Malaysia', 'Philippines'],
    description: 'Currency crisis contagion across East Asia, demonstrating financial cascade dynamics.',
  },
  {
    id: 'arab-spring',
    name: 'Arab Spring (2011)',
    trigger: 'Tunisian street vendor self-immolation',
    regions: ['Tunisia', 'Egypt', 'Libya', 'Syria', 'Yemen', 'Bahrain'],
    description: 'How a single protest cascaded into regional revolutionary wave.',
  },
  {
    id: 'covid-19',
    name: 'COVID-19 Pandemic (2020)',
    trigger: 'Novel coronavirus outbreak in Wuhan',
    regions: ['Global'],
    description: 'Black swan event demonstrating cascading health, economic, and political effects.',
  },
  {
    id: 'lehman-collapse',
    name: 'Global Financial Crisis (2008)',
    trigger: 'Lehman Brothers bankruptcy',
    regions: ['United States', 'Europe', 'Global'],
    description: 'Financial contagion demonstrating systemic risk and cascade failures.',
  },
];

export default {
  simulateCascade,
  HISTORICAL_CASCADES,
};
