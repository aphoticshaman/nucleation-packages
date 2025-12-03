/**
 * Multi-Dimensional Geopolitical Analysis Framework
 *
 * Supports filtering and analysis across:
 * - Temporal states (past/present/future)
 * - Entity types (nations, orgs, individuals)
 * - Ideological positions
 * - Alliance/hostility networks
 * - Religious/cultural factors
 * - Military standings
 * - Historical relationships
 * - Resource dependencies
 * - Conflict trajectories
 * - Population/consumption trends
 */

// ============================================
// Core Entity Types
// ============================================

export type EntityType = 'nation' | 'organization' | 'individual' | 'alliance' | 'movement';

export interface Entity {
  id: string;
  name: string;
  type: EntityType;
  code?: string; // ISO code for nations
  aliases?: string[];
  parent_entities?: string[]; // e.g., nation belongs to alliance
}

// ============================================
// Temporal Dimension
// ============================================

export type TemporalState = 'historical' | 'current' | 'projected';

export interface TemporalRange {
  start: Date;
  end: Date;
  granularity: 'day' | 'week' | 'month' | 'quarter' | 'year' | 'decade';
}

export interface TemporalDataPoint<T> {
  timestamp: Date;
  state: TemporalState;
  confidence: number; // 0-1, lower for projections
  data: T;
}

// ============================================
// Ideological Dimension
// ============================================

export type IdeologicalAxis =
  | 'economic' // Left <-> Right
  | 'social' // Libertarian <-> Authoritarian
  | 'foreign_policy' // Isolationist <-> Interventionist
  | 'governance' // Democratic <-> Autocratic
  | 'globalism'; // Nationalist <-> Globalist

export interface IdeologicalPosition {
  axis: IdeologicalAxis;
  value: number; // -1 to 1
  confidence: number;
  trend: number; // Change rate
}

export interface IdeologicalProfile {
  entity_id: string;
  positions: IdeologicalPosition[];
  dominant_ideology?: string;
}

// ============================================
// Alliance & Hostility Networks
// ============================================

export type RelationType =
  | 'alliance' // Formal alliance
  | 'partnership' // Strategic partnership
  | 'friendly' // Generally positive relations
  | 'neutral' // No significant relationship
  | 'tension' // Diplomatic tension
  | 'rivalry' // Active competition
  | 'hostile' // Open hostility
  | 'conflict'; // Active military conflict

export interface EntityRelation {
  source_id: string;
  target_id: string;
  type: RelationType;
  strength: number; // 0-1
  basis: RelationBasis[];
  historical_context?: string;
  since?: Date;
  treaties?: string[];
}

export type RelationBasis =
  | 'military'
  | 'economic'
  | 'ideological'
  | 'religious'
  | 'cultural'
  | 'historical'
  | 'territorial'
  | 'resource';

// ============================================
// Religious & Cultural Dimension
// ============================================

export interface ReligiousDemographic {
  religion: string;
  denomination?: string;
  percentage: number;
  trend: 'growing' | 'stable' | 'declining';
  influence_level: 'dominant' | 'significant' | 'moderate' | 'minor';
}

export interface CulturalFactors {
  entity_id: string;
  primary_language: string;
  languages: string[];
  religions: ReligiousDemographic[];
  cultural_sphere: string; // e.g., 'Western', 'Sinic', 'Islamic', etc.
  cultural_influence_score: number;
  soft_power_index: number;
}

// ============================================
// Military Dimension
// ============================================

export type MilitaryPosture = 'defensive' | 'neutral' | 'forward' | 'aggressive';

export interface MilitaryCapabilities {
  entity_id: string;
  active_personnel: number;
  reserve_personnel: number;
  nuclear_capable: boolean;
  nuclear_warheads?: number;
  defense_budget_usd: number;
  defense_budget_gdp_percent: number;
  power_index: number; // Global Firepower style index
  posture: MilitaryPosture;
  major_systems: string[];
  alliances: string[];
  bases_abroad: number;
  deployment_readiness: 'high' | 'medium' | 'low';
}

export interface MilitaryActivity {
  id: string;
  entity_id: string;
  type: 'exercise' | 'deployment' | 'buildup' | 'withdrawal' | 'incident';
  location: { lat: number; lng: number; name: string };
  timestamp: Date;
  description: string;
  involved_entities: string[];
  threat_level: 'routine' | 'elevated' | 'concerning' | 'critical';
}

// ============================================
// Historical Feuds & Standings
// ============================================

export interface HistoricalEvent {
  id: string;
  name: string;
  date: Date;
  type: 'war' | 'treaty' | 'incident' | 'revolution' | 'independence' | 'annexation' | 'genocide';
  entities_involved: string[];
  outcome?: string;
  lasting_impact: number; // How much it still affects relations (-1 to 1)
  description: string;
}

export interface HistoricalFeud {
  entity_a: string;
  entity_b: string;
  origin_events: string[]; // Event IDs
  current_status: 'active' | 'dormant' | 'resolved' | 'escalating';
  intensity: number; // 0-1
  territorial_claims?: string[];
  casualties_historical?: number;
  reconciliation_attempts?: string[];
}

// ============================================
// Resource Dependencies
// ============================================

export type ResourceType =
  | 'oil'
  | 'natural_gas'
  | 'coal'
  | 'uranium'
  | 'rare_earths'
  | 'lithium'
  | 'cobalt'
  | 'copper'
  | 'iron'
  | 'aluminum'
  | 'food_grains'
  | 'water'
  | 'semiconductors'
  | 'pharmaceuticals';

export interface ResourceDependency {
  consumer_id: string;
  supplier_id: string;
  resource: ResourceType;
  dependency_percent: number; // What % of consumer's needs come from supplier
  criticality: 'essential' | 'important' | 'moderate' | 'low';
  alternatives_available: boolean;
  strategic_reserve_days?: number;
}

export interface ResourceProfile {
  entity_id: string;
  production: Record<ResourceType, number>;
  consumption: Record<ResourceType, number>;
  exports: ResourceDependency[];
  imports: ResourceDependency[];
  self_sufficiency: Record<ResourceType, number>; // 0-1
}

// Venn diagram data for overlap analysis
export interface ResourceOverlap {
  entities: string[];
  shared_dependencies: ResourceType[];
  competition_resources: ResourceType[];
  complementary_resources: ResourceType[];
  vulnerability_score: number;
}

// ============================================
// Conflict Trajectories & Vectors
// ============================================

export type ConflictPhase =
  | 'latent' // Underlying tensions exist
  | 'emerging' // Tensions becoming visible
  | 'escalating' // Active escalation
  | 'crisis' // Immediate crisis
  | 'active_conflict' // Ongoing conflict
  | 'de_escalating' // Moving toward resolution
  | 'post_conflict'; // Resolution/reconstruction

export interface ConflictVector {
  id: string;
  name: string;
  parties: string[];
  phase: ConflictPhase;
  drivers: ConflictDriver[];
  trajectory: TrajectoryPoint[];
  probability_escalation: number;
  probability_resolution: number;
  potential_triggers: string[];
  potential_offramps: string[];
  estimated_casualties_if_escalation?: { low: number; mid: number; high: number };
}

export interface ConflictDriver {
  type: 'territorial' | 'resource' | 'ideological' | 'ethnic' | 'religious' | 'economic' | 'historical' | 'proxy';
  description: string;
  weight: number; // How much this drives the conflict
  addressable: boolean;
}

export interface TrajectoryPoint {
  timestamp: Date;
  tension_level: number; // 0-1
  phase: ConflictPhase;
  key_events?: string[];
  projected: boolean;
}

// ============================================
// Population & Consumption Trends
// ============================================

export interface DemographicTrends {
  entity_id: string;
  population: number;
  population_growth_rate: number;
  median_age: number;
  urbanization_rate: number;
  fertility_rate: number;
  life_expectancy: number;
  youth_bulge: boolean; // Large young population = instability risk
  aging_population: boolean;
  migration_net: number;
  ethnic_composition: Array<{ group: string; percentage: number }>;
  projected_population: Record<number, number>; // Year -> population
}

export interface ConsumptionTrends {
  entity_id: string;
  gdp_per_capita: number;
  gdp_growth_rate: number;
  energy_consumption_per_capita: number;
  carbon_emissions_per_capita: number;
  food_security_index: number;
  water_stress_index: number;
  consumption_trajectory: 'growing' | 'stable' | 'declining';
  middle_class_percentage: number;
  inequality_gini: number;
}

// ============================================
// Societal & Governmental Factors
// ============================================

export interface GovernanceMetrics {
  entity_id: string;
  regime_type: 'democracy' | 'hybrid' | 'authoritarian' | 'failed_state';
  democracy_index: number; // 0-10
  corruption_index: number; // 0-100 (higher = less corrupt)
  press_freedom_index: number;
  rule_of_law_index: number;
  government_effectiveness: number;
  political_stability: number;
  succession_risk: 'low' | 'moderate' | 'high' | 'imminent';
  leadership_tenure_years: number;
  election_schedule?: Date;
}

export interface SocietalStability {
  entity_id: string;
  social_cohesion: number;
  ethnic_tension: number;
  religious_tension: number;
  class_tension: number;
  protest_activity: 'none' | 'sporadic' | 'frequent' | 'widespread';
  civil_unrest_risk: number;
  separatist_movements: string[];
  internal_displacement: number;
}

// ============================================
// Advancement & Technology Metrics
// ============================================

export interface TechnologyProfile {
  entity_id: string;
  tech_readiness_index: number;
  r_and_d_spending_gdp_percent: number;
  patent_filings_annual: number;
  stem_graduates_annual: number;
  internet_penetration: number;
  ai_capability: 'leading' | 'advanced' | 'developing' | 'limited';
  cyber_capability: 'tier1' | 'tier2' | 'tier3' | 'limited';
  space_capability: 'full' | 'partial' | 'aspirant' | 'none';
  strategic_tech_sectors: string[];
  tech_dependencies: string[]; // What they import
}

// ============================================
// Composite Analysis Types
// ============================================

export interface EntityAnalysis {
  entity: Entity;
  temporal_span: TemporalRange;
  ideological: IdeologicalProfile;
  cultural: CulturalFactors;
  military: MilitaryCapabilities;
  resources: ResourceProfile;
  demographics: DemographicTrends;
  consumption: ConsumptionTrends;
  governance: GovernanceMetrics;
  societal: SocietalStability;
  technology: TechnologyProfile;
  relations: EntityRelation[];
  historical_context: HistoricalEvent[];
  active_conflicts: ConflictVector[];
}

// ============================================
// Filter System
// ============================================

export interface AnalysisFilters {
  // Temporal
  temporal_states?: TemporalState[];
  date_range?: TemporalRange;

  // Entities
  entity_types?: EntityType[];
  entity_ids?: string[];
  entity_groups?: string[]; // Alliances, regions, etc.

  // Ideological
  ideological_range?: Partial<Record<IdeologicalAxis, [number, number]>>;

  // Relations
  relation_types?: RelationType[];
  min_relation_strength?: number;

  // Religious/Cultural
  religions?: string[];
  cultural_spheres?: string[];

  // Military
  military_postures?: MilitaryPosture[];
  nuclear_only?: boolean;
  min_military_power?: number;

  // Resources
  resource_types?: ResourceType[];
  dependency_criticality?: ResourceDependency['criticality'][];

  // Conflicts
  conflict_phases?: ConflictPhase[];
  min_escalation_probability?: number;

  // Demographics
  population_range?: [number, number];
  growth_rate_range?: [number, number];

  // Governance
  regime_types?: GovernanceMetrics['regime_type'][];
  min_democracy_index?: number;
  max_corruption?: number;

  // Risk
  min_risk_score?: number;
  max_stability?: number;
}

// ============================================
// Utility Functions
// ============================================

export function computeResourceOverlap(
  entityA: ResourceProfile,
  entityB: ResourceProfile
): ResourceOverlap {
  const aImports = new Set(entityA.imports.map((i) => i.resource));
  const bImports = new Set(entityB.imports.map((i) => i.resource));

  const shared = [...aImports].filter((r) => bImports.has(r)) as ResourceType[];

  // Resources they both need but one produces
  const aProduces = new Set(Object.keys(entityA.production).filter(
    (k) => entityA.production[k as ResourceType] > 0
  ));
  const bProduces = new Set(Object.keys(entityB.production).filter(
    (k) => entityB.production[k as ResourceType] > 0
  ));

  const competition = shared.filter((r) => !aProduces.has(r) && !bProduces.has(r));
  const complementary = [...aImports].filter((r) => bProduces.has(r)) as ResourceType[];

  return {
    entities: [entityA.entity_id, entityB.entity_id],
    shared_dependencies: shared,
    competition_resources: competition,
    complementary_resources: complementary,
    vulnerability_score: competition.length / (shared.length || 1),
  };
}

export function projectConflictTrajectory(
  conflict: ConflictVector,
  daysAhead: number
): TrajectoryPoint[] {
  const lastPoint = conflict.trajectory[conflict.trajectory.length - 1];
  const projections: TrajectoryPoint[] = [];

  let currentTension = lastPoint?.tension_level ?? 0.5;
  let currentPhase = lastPoint?.phase ?? conflict.phase;

  // Simple projection based on escalation probability
  for (let day = 1; day <= daysAhead; day += 7) {
    // Weekly projections
    const escalationDelta = (Math.random() - 0.5) * 0.1 +
      (conflict.probability_escalation - 0.5) * 0.05;

    currentTension = Math.max(0, Math.min(1, currentTension + escalationDelta));

    // Phase transitions
    if (currentTension > 0.8 && currentPhase !== 'active_conflict') {
      currentPhase = 'crisis';
    } else if (currentTension > 0.6 && currentPhase === 'latent') {
      currentPhase = 'escalating';
    } else if (currentTension < 0.3 && currentPhase !== 'post_conflict') {
      currentPhase = 'de_escalating';
    }

    projections.push({
      timestamp: new Date(Date.now() + day * 24 * 60 * 60 * 1000),
      tension_level: currentTension,
      phase: currentPhase,
      projected: true,
    });
  }

  return projections;
}

export function computeRelationStrength(
  relations: EntityRelation[],
  entityA: string,
  entityB: string
): { type: RelationType; strength: number } | null {
  const direct = relations.find(
    (r) =>
      (r.source_id === entityA && r.target_id === entityB) ||
      (r.source_id === entityB && r.target_id === entityA)
  );

  if (direct) {
    return { type: direct.type, strength: direct.strength };
  }

  return null;
}

export function identifyAlliances(
  relations: EntityRelation[],
  entities: Entity[]
): Map<string, string[]> {
  const alliances = new Map<string, string[]>();

  // Group by alliance relations
  const allianceRelations = relations.filter(
    (r) => r.type === 'alliance' || r.type === 'partnership'
  );

  // Build alliance clusters using union-find style grouping
  const parent = new Map<string, string>();

  const find = (x: string): string => {
    if (!parent.has(x)) parent.set(x, x);
    if (parent.get(x) !== x) {
      parent.set(x, find(parent.get(x)!));
    }
    return parent.get(x)!;
  };

  const union = (x: string, y: string) => {
    const px = find(x);
    const py = find(y);
    if (px !== py) {
      parent.set(px, py);
    }
  };

  allianceRelations.forEach((r) => union(r.source_id, r.target_id));

  // Group entities by their root alliance
  const groups = new Map<string, string[]>();
  entities.forEach((e) => {
    const root = find(e.id);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(e.id);
  });

  // Filter to only multi-member alliances
  groups.forEach((members, root) => {
    if (members.length > 1) {
      alliances.set(root, members);
    }
  });

  return alliances;
}
