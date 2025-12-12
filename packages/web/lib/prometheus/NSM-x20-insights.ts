/**
 * NSM (Novel Signal Mining) x20 Distillation
 * PROMETHEUS PROTOCOL - Comprehensive Insight Extraction
 *
 * Distills 20 novel insights from nucleation packages for LatticeForge integration:
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #1: VARIANCE QUIETING PRECEDES PHASE TRANSITIONS
 * ═══════════════════════════════════════════════════════════════════════════════
 * All nucleation packages detect the same fundamental pattern: variance DECREASES
 * before major transitions. This is counter-intuitive - we expect chaos before change,
 * but systems "crystallize" their behavior as they approach critical points.
 *
 * APPLICATION: Track user engagement variance. Decreasing variance + stable mean
 * indicates imminent churn or major behavior change.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #2: MULTI-DOMAIN PHASE COHERENCE
 * ═══════════════════════════════════════════════════════════════════════════════
 * When multiple independent detectors (threat, churn, market, org) simultaneously
 * approach transition states, they're likely causally connected through hidden variables.
 *
 * APPLICATION: Cross-reference FlowState signals across domains. Coherent transitions
 * indicate systemic change requiring holistic response.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #3: SHEPHERD'S CONFLICT POTENTIAL = DIVERGENCE PRECURSOR
 * ═══════════════════════════════════════════════════════════════════════════════
 * The Shepherd class (used in ThreatCorrelator, IntegrationMonitor) measures
 * behavioral distribution divergence between actors. Rising conflict potential
 * predicts polarization before it manifests.
 *
 * APPLICATION: Monitor Elle interaction patterns vs user expectations. Divergence
 * indicates communication breakdown requiring intervention.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #4: CONFIDENCE AS OBSERVATION COUNT FUNCTION
 * ═══════════════════════════════════════════════════════════════════════════════
 * Detector confidence scales with datapoints, but asymptotically. Early readings
 * are noisy; mature detectors are reliable but slow to adapt.
 *
 * APPLICATION: Use confidence as a weighting factor in multi-signal fusion.
 * New users get benefit of doubt; established patterns are trusted.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #5: INFLECTION MAGNITUDE AS TRANSITION STRENGTH
 * ═══════════════════════════════════════════════════════════════════════════════
 * inflectionMagnitude() returns z-score of current variance vs historical baseline.
 * High magnitude transitions are more likely to be permanent; low magnitude may be noise.
 *
 * APPLICATION: Only alert on high-magnitude transitions; log low-magnitude for
 * retrospective analysis but don't interrupt user flow.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #6: WASM ACCELERATION FOR REAL-TIME DETECTION
 * ═══════════════════════════════════════════════════════════════════════════════
 * All packages use nucleation-wasm for core computation. Browser WASM runs at
 * near-native speed, enabling real-time phase detection on client-side.
 *
 * APPLICATION: Client-side FlowState tracking via WASM. Ship detectors to edge,
 * only sync state changes to server. Massive latency + bandwidth reduction.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #7: SERIALIZABLE STATE ENABLES CONTINUOUS MONITORING
 * ═══════════════════════════════════════════════════════════════════════════════
 * All detectors support serialize()/deserialize(). Historical state persists
 * across sessions without reprocessing entire history.
 *
 * APPLICATION: Store serialized detector states in encrypted user profiles.
 * Resume monitoring instantly on return visit.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #8: PROMETHEUS ENGINE'S NCD FOR STRUCTURAL SIMILARITY
 * ═══════════════════════════════════════════════════════════════════════════════
 * Normalized Compression Distance measures algorithmic similarity between strings.
 * Similar patterns compress well together; dissimilar don't.
 *
 * APPLICATION: Use NCD to cluster user queries. Similar queries get cached
 * responses; novel queries trigger fresh inference.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #9: GRAVITATIONAL BASINS FOR ANSWER CONSENSUS
 * ═══════════════════════════════════════════════════════════════════════════════
 * clusterBasins() groups numeric answers by proximity. Basin mass × density
 * indicates confidence. Multiple small basins = uncertain; one large = confident.
 *
 * APPLICATION: Multi-model inference fusion. Run Elle + backup models, cluster
 * responses, select highest-scoring basin.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #10: ENTROPY PHASE TRANSITIONS = REASONING QUALITY
 * ═══════════════════════════════════════════════════════════════════════════════
 * detectPhaseTransition() identifies when reasoning "crystallizes" (entropy drops).
 * Early high-entropy = exploration; late low-entropy = exploitation/confidence.
 *
 * APPLICATION: Track Elle's reasoning entropy. Failure to crystallize indicates
 * confusion; premature crystallization indicates overconfidence.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #11: SOLOMONOFF WEIGHTING = OCCAM'S RAZOR FORMALIZED
 * ═══════════════════════════════════════════════════════════════════════════════
 * solomonoffWeight() prefers shorter code/explanations. Exponential decay
 * penalizes verbosity. Simplicity is evidence of correctness.
 *
 * APPLICATION: Weight Elle responses inversely to length. Concise correct answers
 * beat verbose rambling.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #12: NOVELTY DETECTION'S KNOWLEDGE QUADRANTS
 * ═══════════════════════════════════════════════════════════════════════════════
 * buildKnowledgeQuadrant() classifies discoveries:
 * - Known Knowns: Established facts
 * - Known Unknowns: Acknowledged gaps
 * - Unknown Unknowns: Surprises
 * - Unknown Knowns: Implicit knowledge surfaced
 *
 * APPLICATION: Classify Elle's knowledge state. Surface unknown knowns through
 * strategic prompting; acknowledge unknown unknowns honestly.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #13: CROSS-DOMAIN ISOMORPHISMS = HIDDEN CAUSALITY
 * ═══════════════════════════════════════════════════════════════════════════════
 * detectCrossDomainIso() finds correlations between unrelated time series.
 * Lag analysis reveals which leads/follows. Cross-domain correlation > 0.7 is
 * strong evidence of shared cause.
 *
 * APPLICATION: Correlate user engagement with external events (news, market,
 * social). Identify what drives behavior changes.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #14: TEMPORAL BREAKS VIA KL-DIVERGENCE + FISHER DISTANCE
 * ═══════════════════════════════════════════════════════════════════════════════
 * detectTemporalBreak() uses two complementary measures:
 * - KL divergence: Information-theoretic distribution difference
 * - Fisher distance: Geometric distance on statistical manifold
 *
 * APPLICATION: Use both for robust regime detection. KL catches mean shifts;
 * Fisher catches variance/shape changes.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #15: TRIADIC CLOSURE FOR STRUCTURAL GAP DETECTION
 * ═══════════════════════════════════════════════════════════════════════════════
 * detectStructuralGap() identifies missing edges using common neighbors.
 * If A-B and A-C exist, B-C is likely. Missing expected edges are structural gaps.
 *
 * APPLICATION: Identify missing content connections. If users who like X also
 * like Y and Z, but Y-Z aren't linked, create connection.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #16: THREAT QUIETING PATTERN FOR SECURITY
 * ═══════════════════════════════════════════════════════════════════════════════
 * Attackers "quiet down" before striking - reducing variance in probing as they
 * zero in on targets. The calm before the storm is detectable.
 *
 * APPLICATION: Monitor Guardian logs for variance reduction. Quieting in
 * previously noisy patterns = elevated threat.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #17: CROWD TENSION LEVELS MAP TO ENGAGEMENT
 * ═══════════════════════════════════════════════════════════════════════════════
 * CrowdMonitor's CALM→TENSE→HEATED→VOLATILE progression maps to engagement:
 * - CALM: Normal browsing
 * - TENSE: Active engagement
 * - HEATED: High interaction (positive or negative)
 * - VOLATILE: Potential viral moment or PR crisis
 *
 * APPLICATION: Track aggregate user sentiment. HEATED triggers proactive
 * support; VOLATILE triggers executive notification.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #18: M&A CULTURE CLASH AS USER-PRODUCT FIT
 * ═══════════════════════════════════════════════════════════════════════════════
 * IntegrationMonitor's culture dimensions parallel user-product alignment.
 * Clash risk = preference divergence. High divergence = wrong segment.
 *
 * APPLICATION: Monitor user preference drift vs product evolution. Growing
 * divergence indicates churn risk even before explicit dissatisfaction.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #19: BATCH UPDATES FOR EFFICIENCY
 * ═══════════════════════════════════════════════════════════════════════════════
 * All detectors support updateBatch(). Processing N points at once is O(N);
 * N individual updates is O(N²) due to variance recalculation.
 *
 * APPLICATION: Batch user events into windows (e.g., 1-minute). Process
 * batches for 100x throughput improvement on high-traffic systems.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INSIGHT #20: RESET() AS INCIDENT BOUNDARY MARKER
 * ═══════════════════════════════════════════════════════════════════════════════
 * After major events (incident resolution, product launch, user conversion),
 * reset() clears historical state. Post-event behavior is new baseline.
 *
 * APPLICATION: Reset user detectors after conversion events. Pre-trial and
 * post-trial users are different populations; comparing them is invalid.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import type { NoveltySignal, NoveltyType } from './novelty-detection';

export interface NSMInsight {
  id: number;
  title: string;
  category: 'variance' | 'coherence' | 'fusion' | 'detection' | 'optimization' | 'security' | 'ux';
  principle: string;
  application: string;
  packages: string[];
  integrationPriority: 'critical' | 'high' | 'medium' | 'low';
}

export const NSM_INSIGHTS: NSMInsight[] = [
  {
    id: 1,
    title: 'Variance Quieting Precedes Phase Transitions',
    category: 'variance',
    principle: 'Systems crystallize behavior before major changes - variance decreases before transitions',
    application: 'Track user engagement variance. Decreasing variance + stable mean indicates imminent churn',
    packages: ['threat-pulse', 'regime-shift', 'churn-harbinger', 'crowd-phase'],
    integrationPriority: 'critical',
  },
  {
    id: 2,
    title: 'Multi-Domain Phase Coherence',
    category: 'coherence',
    principle: 'Simultaneous transitions across independent detectors indicate hidden causal connections',
    application: 'Cross-reference FlowState signals across domains for systemic change detection',
    packages: ['nucleation', 'threat-pulse', 'regime-shift', 'org-canary'],
    integrationPriority: 'critical',
  },
  {
    id: 3,
    title: 'Shepherd Conflict Potential as Divergence Precursor',
    category: 'detection',
    principle: 'Behavioral distribution divergence predicts polarization before manifestation',
    application: 'Monitor Elle↔User interaction pattern divergence for communication breakdown',
    packages: ['threat-pulse', 'org-canary'],
    integrationPriority: 'high',
  },
  {
    id: 4,
    title: 'Confidence as Observation Count Function',
    category: 'fusion',
    principle: 'Detector confidence scales asymptotically with datapoints',
    application: 'Weight multi-signal fusion by confidence; new users get benefit of doubt',
    packages: ['nucleation-wasm', 'regime-shift', 'churn-harbinger'],
    integrationPriority: 'high',
  },
  {
    id: 5,
    title: 'Inflection Magnitude as Transition Strength',
    category: 'variance',
    principle: 'Z-score of variance change indicates transition permanence',
    application: 'Alert only on high-magnitude transitions; log low-magnitude for analysis',
    packages: ['regime-shift', 'market-canary', 'sensor-shift'],
    integrationPriority: 'high',
  },
  {
    id: 6,
    title: 'WASM Acceleration for Real-Time Detection',
    category: 'optimization',
    principle: 'Browser WASM enables client-side phase detection at near-native speed',
    application: 'Ship detectors to edge; sync only state changes for latency/bandwidth reduction',
    packages: ['nucleation-wasm'],
    integrationPriority: 'critical',
  },
  {
    id: 7,
    title: 'Serializable State for Continuous Monitoring',
    category: 'optimization',
    principle: 'Detector state persists across sessions without history reprocessing',
    application: 'Store encrypted serialized states in user profiles; instant resume',
    packages: ['nucleation', 'regime-shift', 'churn-harbinger'],
    integrationPriority: 'high',
  },
  {
    id: 8,
    title: 'NCD for Structural Similarity',
    category: 'fusion',
    principle: 'Similar patterns compress well together; Normalized Compression Distance measures this',
    application: 'Cluster user queries by NCD; cache similar responses, fresh inference for novel',
    packages: ['@nucleation/gtvc', 'prometheus-engine'],
    integrationPriority: 'medium',
  },
  {
    id: 9,
    title: 'Gravitational Basins for Answer Consensus',
    category: 'fusion',
    principle: 'Basin mass × density indicates response confidence',
    application: 'Multi-model inference fusion; cluster responses, select highest-scoring basin',
    packages: ['prometheus-engine', '@nucleation/gtvc'],
    integrationPriority: 'critical',
  },
  {
    id: 10,
    title: 'Entropy Phase Transitions as Reasoning Quality',
    category: 'detection',
    principle: 'Reasoning crystallizes (entropy drops) as confidence increases',
    application: 'Track Elle reasoning entropy; detect confusion or overconfidence',
    packages: ['prometheus-engine', 'novelty-detection'],
    integrationPriority: 'high',
  },
  {
    id: 11,
    title: 'Solomonoff Weighting as Formalized Occam\'s Razor',
    category: 'fusion',
    principle: 'Shorter explanations are more likely correct; exponential decay on length',
    application: 'Weight Elle responses inversely to length; prefer concise correct answers',
    packages: ['prometheus-engine'],
    integrationPriority: 'medium',
  },
  {
    id: 12,
    title: 'Knowledge Quadrants Classification',
    category: 'detection',
    principle: 'Known/Unknown × Known/Unknown creates four quadrants of knowledge state',
    application: 'Classify Elle knowledge gaps; surface unknown knowns through strategic prompting',
    packages: ['novelty-detection'],
    integrationPriority: 'medium',
  },
  {
    id: 13,
    title: 'Cross-Domain Isomorphisms Reveal Hidden Causality',
    category: 'coherence',
    principle: 'High correlation (>0.7) between unrelated series indicates shared cause',
    application: 'Correlate engagement with external events to identify behavior drivers',
    packages: ['novelty-detection', 'regime-shift'],
    integrationPriority: 'high',
  },
  {
    id: 14,
    title: 'Dual Metrics for Robust Regime Detection',
    category: 'detection',
    principle: 'KL divergence catches mean shifts; Fisher distance catches variance/shape changes',
    application: 'Use both for comprehensive temporal break detection',
    packages: ['novelty-detection', 'information-geometry'],
    integrationPriority: 'high',
  },
  {
    id: 15,
    title: 'Triadic Closure for Structural Gap Detection',
    category: 'detection',
    principle: 'If A-B and A-C exist, B-C is likely; missing edges are structural gaps',
    application: 'Identify missing content connections; suggest related items',
    packages: ['novelty-detection'],
    integrationPriority: 'medium',
  },
  {
    id: 16,
    title: 'Threat Quieting Pattern',
    category: 'security',
    principle: 'Attackers reduce variance in probing as they zero in on targets',
    application: 'Monitor Guardian logs for quieting in previously noisy patterns',
    packages: ['threat-pulse'],
    integrationPriority: 'critical',
  },
  {
    id: 17,
    title: 'Crowd Tension as Engagement Indicator',
    category: 'ux',
    principle: 'CALM→TENSE→HEATED→VOLATILE maps to engagement intensity',
    application: 'Track aggregate sentiment; HEATED triggers support, VOLATILE triggers executive alert',
    packages: ['crowd-phase'],
    integrationPriority: 'high',
  },
  {
    id: 18,
    title: 'Culture Clash as User-Product Fit',
    category: 'ux',
    principle: 'Preference divergence = wrong segment; growing divergence predicts churn',
    application: 'Monitor user preference drift vs product evolution for fit detection',
    packages: ['org-canary'],
    integrationPriority: 'high',
  },
  {
    id: 19,
    title: 'Batch Updates for Throughput',
    category: 'optimization',
    principle: 'Batch processing is O(N); individual updates are O(N²)',
    application: 'Batch user events into windows for 100x throughput improvement',
    packages: ['nucleation-wasm', 'regime-shift', 'churn-harbinger'],
    integrationPriority: 'medium',
  },
  {
    id: 20,
    title: 'Reset as Incident Boundary Marker',
    category: 'optimization',
    principle: 'Post-event behavior is a new baseline; pre/post comparison invalid',
    application: 'Reset detectors after conversion events; treat as new population',
    packages: ['nucleation', 'churn-harbinger', 'org-canary'],
    integrationPriority: 'medium',
  },
];

/**
 * Get insights by category
 */
export function getInsightsByCategory(category: NSMInsight['category']): NSMInsight[] {
  return NSM_INSIGHTS.filter(i => i.category === category);
}

/**
 * Get insights by integration priority
 */
export function getCriticalInsights(): NSMInsight[] {
  return NSM_INSIGHTS.filter(i => i.integrationPriority === 'critical');
}

/**
 * Get insights relevant to specific packages
 */
export function getInsightsForPackage(packageName: string): NSMInsight[] {
  return NSM_INSIGHTS.filter(i => i.packages.includes(packageName));
}

/**
 * Generate actionable recommendations based on signals
 */
export function generateRecommendations(signals: NoveltySignal[]): string[] {
  const recommendations: string[] = [];

  const typeToInsight: Record<NoveltyType, number[]> = {
    'PATTERN_ANOMALY': [1, 5, 16],
    'STRUCTURAL_GAP': [15],
    'CROSS_DOMAIN_ISO': [2, 13],
    'TEMPORAL_BREAK': [1, 14],
    'CAUSAL_LOOP': [2, 3],
    'EMERGENT_PROPERTY': [12],
  };

  for (const signal of signals) {
    const relevantInsightIds = typeToInsight[signal.type] || [];
    for (const id of relevantInsightIds) {
      const insight = NSM_INSIGHTS.find(i => i.id === id);
      if (insight) {
        recommendations.push(`[${insight.title}] ${insight.application}`);
      }
    }
  }

  return [...new Set(recommendations)]; // Deduplicate
}

export default NSM_INSIGHTS;
