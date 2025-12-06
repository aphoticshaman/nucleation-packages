
export enum ContextLayer {
  SURFACE = 'SURFACE (Geopol)',
  DEEP = 'DEEP (Econ/Crypto)',
  DARK = 'DARK (Cyber/Intel)'
}

export type IntelCategory = 
  | 'DEFENSE' | 'CYBER' | 'TECH' | 'POLITICS' | 'HEALTH' 
  | 'FINANCE' | 'SPACE' | 'ENTERTAINMENT' | 'CORP' 
  | 'AGRI' | 'RESOURCES' | 'HOUSING' | 'EDU' | 'CRIME';

// --- LatticeForge Core Types ---

export interface CausalNode {
  id: string;
  label: string;
  type: 'ACTOR' | 'EVENT' | 'RESOURCE' | 'HYPOTHESIS';
  level: 'MICRO' | 'MESO' | 'MACRO'; // Individual -> Org -> Nation
  beliefMass: number; // 0-1 (Dempster-Shafer mass)
  activity: number; // 0-1 (Pulse intensity)
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface CausalEdge {
  source: string;
  target: string;
  transferEntropy: number; // 0-1 (Flow magnitude)
  lag: number; // Time lag (color/dash)
  type: 'INFLUENCE' | 'CAUSALITY' | 'CORRELATION';
}

export interface FoldNode {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  targetX: number;
  targetY: number;
  color: string;
  size: number;
  connections: number[];
  data: string;
}

export interface RegimeData {
  timestamp: string;
  stable: number;
  volatile: number;
  crisis: number;
  transitionProbability: number;
}

export interface CICMetrics {
  phi: number; // Integrated Information
  entropy: number; // H(T|X)
  causalMulti: number; // C_multi
  freeEnergy: number; // F[T]
}

// ------------------------------

export interface EntropySource {
  id: string;
  source: 'CoinGecko' | 'OpenMeteo' | 'System';
  value: number; 
  label: string;
  delta: number;
}

export interface IntelligencePacket {
  id: string;
  timestamp: string;
  context: ContextLayer;
  category: IntelCategory;
  header: string;
  summary: string;
  body: string;
  coherence: number;
  source: string;
}

export interface Coordinates {
  lat: number;
  lng: number;
}

export enum RiskLevel {
  LOW = 'LOW',
  MODERATE = 'MODERATE',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}
