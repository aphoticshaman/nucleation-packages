
import { CausalNode, CausalEdge, RegimeData, CICMetrics } from './types';

// --- CAUSAL GRAPH DATA ---
export const MOCK_NODES: CausalNode[] = [
  { id: 'n1', label: 'US_FED_RES', type: 'ACTOR', level: 'MACRO', beliefMass: 0.95, activity: 0.8 },
  { id: 'n2', label: 'PRC_PLAN', type: 'ACTOR', level: 'MACRO', beliefMass: 0.88, activity: 0.9 },
  { id: 'n3', label: 'TSMC_FAB', type: 'RESOURCE', level: 'MESO', beliefMass: 0.92, activity: 0.6 },
  { id: 'n4', label: 'STRAIT_BLOCKADE', type: 'EVENT', level: 'MACRO', beliefMass: 0.45, activity: 0.95 },
  { id: 'n5', label: 'GLOBAL_GPU_SUPPLY', type: 'RESOURCE', level: 'MESO', beliefMass: 0.85, activity: 0.7 },
  { id: 'n6', label: 'NVDA_STOCK', type: 'ACTOR', level: 'MICRO', beliefMass: 0.98, activity: 0.85 },
  { id: 'n7', label: 'EU_CHIPS_ACT', type: 'EVENT', level: 'MACRO', beliefMass: 0.75, activity: 0.4 },
  { id: 'n8', label: 'LITHIUM_SPOT', type: 'RESOURCE', level: 'MICRO', beliefMass: 0.82, activity: 0.5 },
  { id: 'n9', label: 'ASEAN_TRADE', type: 'ACTOR', level: 'MACRO', beliefMass: 0.70, activity: 0.3 },
  { id: 'n10', label: 'CYBER_ATTACK_SCADA', type: 'EVENT', level: 'MESO', beliefMass: 0.30, activity: 0.9 },
];

export const MOCK_EDGES: CausalEdge[] = [
  { source: 'n2', target: 'n4', transferEntropy: 0.9, lag: 2, type: 'CAUSALITY' },
  { source: 'n4', target: 'n3', transferEntropy: 0.85, lag: 1, type: 'INFLUENCE' },
  { source: 'n3', target: 'n5', transferEntropy: 0.95, lag: 0, type: 'CAUSALITY' },
  { source: 'n5', target: 'n6', transferEntropy: 0.8, lag: 0, type: 'CORRELATION' },
  { source: 'n1', target: 'n6', transferEntropy: 0.6, lag: 1, type: 'INFLUENCE' },
  { source: 'n7', target: 'n3', transferEntropy: 0.4, lag: 5, type: 'INFLUENCE' },
  { source: 'n2', target: 'n9', transferEntropy: 0.5, lag: 3, type: 'INFLUENCE' },
  { source: 'n10', target: 'n3', transferEntropy: 0.7, lag: 0, type: 'CAUSALITY' },
  { source: 'n8', target: 'n5', transferEntropy: 0.65, lag: 2, type: 'CAUSALITY' },
];

// --- REGIME DATA (Markov-Switching) ---
export const GENERATE_REGIME_DATA = (): RegimeData[] => {
  const data: RegimeData[] = [];
  const steps = 50;
  for (let i = 0; i < steps; i++) {
    // Simulate a transition from Stable -> Volatile -> Crisis
    let stable = 0, volatile = 0, crisis = 0;
    
    if (i < 20) {
      stable = 0.8 + Math.random() * 0.1;
      volatile = 0.15 + Math.random() * 0.05;
      crisis = 0.05;
    } else if (i < 35) {
      stable = 0.3 + Math.random() * 0.1;
      volatile = 0.6 + Math.random() * 0.1;
      crisis = 0.1;
    } else {
      stable = 0.1;
      volatile = 0.2 + Math.random() * 0.1;
      crisis = 0.7 + Math.random() * 0.1;
    }
    
    // Normalize
    const total = stable + volatile + crisis;
    
    data.push({
      timestamp: `T-${steps - i}`,
      stable: stable / total,
      volatile: volatile / total,
      crisis: crisis / total,
      transitionProbability: i === 19 || i === 34 ? 0.9 : 0.1
    });
  }
  return data;
};

// --- CIC METRICS ---
export const MOCK_CIC_METRICS: CICMetrics = {
  phi: 4.25, // High integration
  entropy: 0.32, // Low conditional entropy (good prediction)
  causalMulti: 0.88, // Strong multi-scale causality
  freeEnergy: 12.4 // F[T]
};

export const MENU_ITEMS = [
  { id: 'causal', label: 'Causal Canvas', icon: 'share_2' },
  { id: 'workbench', label: 'Intel Workbench', icon: 'layout_dashboard' },
  { id: 'predictive', label: 'Predictive Ops', icon: 'trending_up' },
  { id: 'cic', label: 'CIC Theory', icon: 'activity' },
];
