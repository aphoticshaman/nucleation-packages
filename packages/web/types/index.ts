// Nation types
export interface Nation {
  id: string;
  code: string;
  name: string;
  lat: number;
  lon: number;
  position: number[];
  velocity: number[];
  basin_strength: number;
  transition_risk: number;
  regime: number;
  influence_radius: number;
  // Extended fields for territorial status
  sovereignty_status?: 'recognized' | 'disputed' | 'unrecognized' | 'occupied';
  de_facto_controller?: string;  // ISO code if different from legal sovereign
  is_contested?: boolean;        // Active conflict
  government_type?: string;
  last_transition?: string;      // Date of last government change
}

// Re-export territorial status types for convenience
export type {
  TerritorialStatus,
  TerritorialClaim,
  GovernmentStatus,
  MilitaryAsset,
  ContestedZone,
  SovereigntyStatus,
  ControlStatus,
  GovernmentType,
  TransitionType,
  MilitaryBranch,
  UnitType,
  UnitSize,
  ThreatLevel,
  ForceDisposition,
} from '@/lib/territorial-status';

export interface InfluenceEdge {
  id: string;
  source_id: string;
  target_id: string;
  source_code: string;
  target_code: string;
  strength: number;
  geodesic_distance: number;
  esteem: number;
}

export interface EsteemRelation {
  source_id: string;
  target_id: string;
  esteem: number;
}

// Simulation types
export interface SimulationConfig {
  n_dims: number;
  interaction_decay: number;
  min_influence: number;
  dt: number;
  diffusion: number;
}

export interface SimulationSnapshot {
  time_step: number;
  nations_state: Record<string, NationState>;
  n_edges: number;
  persistent_entropy?: number;
  alert_level?: AlertLevel;
  phase_transition_probability?: number;
}

export interface NationState {
  position: number[];
  velocity: number[];
  basin_strength: number;
  transition_risk: number;
  regime: number;
}

// Alert levels
export type AlertLevel = 'normal' | 'elevated' | 'warning' | 'critical';

// Map layer types
export type MapLayer = 'basin' | 'risk' | 'influence' | 'regime';

// Regime definitions
export const REGIMES: Record<number, { name: string; color: string }> = {
  0: { name: 'Liberal Democracy', color: '#3B82F6' },
  1: { name: 'State Capitalism', color: '#EF4444' },
  2: { name: 'Social Democracy', color: '#10B981' },
  3: { name: 'Authoritarian', color: '#6B7280' },
  4: { name: 'Transitional', color: '#F59E0B' },
};

// GeoJSON types for map
export interface GeoJsonFeature {
  type: 'Feature';
  geometry: {
    type: 'Point' | 'LineString';
    coordinates: number[] | number[][];
  };
  properties: Record<string, unknown>;
}

export interface GeoJsonCollection {
  type: 'FeatureCollection';
  features: GeoJsonFeature[];
  metadata?: Record<string, unknown>;
}
