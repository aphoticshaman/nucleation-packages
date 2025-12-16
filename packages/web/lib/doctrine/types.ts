/**
 * Doctrine Registry - Core types and utilities for rule management
 *
 * Doctrines define how LatticeForge interprets signals and produces judgments.
 * This is a key differentiator for Stewardship tier customers.
 */

export interface DoctrineRule {
  id: string;
  name: string;
  category: 'signal_interpretation' | 'analytic_judgment' | 'policy_logic' | 'narrative';
  description: string;
  rule_definition: {
    type: 'threshold' | 'weight' | 'mapping' | 'template';
    parameters: Record<string, unknown>;
  };
  rationale: string;
  version: number;
  effective_from: string;
  deprecated_at?: string;
  created_by: string;
  created_at: string;
  updated_at: string;
}

export interface DoctrineChangeLog {
  id: string;
  doctrine_id: string;
  version: number;
  change_type: 'created' | 'updated' | 'deprecated';
  previous_definition?: DoctrineRule['rule_definition'];
  new_definition: DoctrineRule['rule_definition'];
  change_rationale: string;
  changed_by: string;
  changed_at: string;
}

export interface ShadowEvaluation {
  id: string;
  doctrine_id: string;
  proposed_changes: Partial<DoctrineRule['rule_definition']>;
  evaluation_period: {
    start: string;
    end: string;
  };
  results: {
    total_events_evaluated: number;
    current_outputs: ShadowOutput[];
    proposed_outputs: ShadowOutput[];
    divergence_count: number;
    divergence_rate: number;
  };
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
}

export interface ShadowOutput {
  event_id: string;
  timestamp: string;
  domain: string;
  output_value: number | string;
  confidence: number;
}

// Default doctrine rules that ship with LatticeForge
export const DEFAULT_DOCTRINES: Omit<DoctrineRule, 'id' | 'created_at' | 'updated_at'>[] = [
  {
    name: 'Basin Strength Threshold',
    category: 'signal_interpretation',
    description: 'Minimum institutional resilience score to classify a nation as stable',
    rule_definition: {
      type: 'threshold',
      parameters: {
        stable_threshold: 0.7,
        unstable_threshold: 0.3,
        metric: 'basin_strength'
      }
    },
    rationale: 'Based on Polity V dataset analysis showing 0.7 as inflection point for democratic stability',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  },
  {
    name: 'Transition Risk Weights',
    category: 'analytic_judgment',
    description: 'Weight factors for computing transition risk from multiple signals',
    rule_definition: {
      type: 'weight',
      parameters: {
        conflict_events: 0.35,
        economic_indicators: 0.25,
        media_tone: 0.20,
        governance_score: 0.20
      }
    },
    rationale: 'Conflict events are leading indicators per UCDP research; economic stress amplifies but rarely causes transitions alone',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  },
  {
    name: 'Phase Classification',
    category: 'policy_logic',
    description: 'Rules for classifying nations into stability phases',
    rule_definition: {
      type: 'mapping',
      parameters: {
        CRYSTALLINE: { basin_min: 0.7, transition_max: 0.2 },
        SUPERCOOLED: { basin_min: 0.5, transition_max: 0.4 },
        NUCLEATING: { basin_min: 0.3, transition_max: 0.6 },
        PLASMA: { basin_max: 0.3, transition_min: 0.6 },
        ANNEALING: { basin_trend: 'increasing', transition_trend: 'decreasing' }
      }
    },
    rationale: 'Phase terminology from materials science; maps intuitively to political stability states',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  },
  {
    name: 'Risk Narrative Template',
    category: 'narrative',
    description: 'Template for generating human-readable risk assessments',
    rule_definition: {
      type: 'template',
      parameters: {
        high_risk_prefix: 'Elevated risk conditions detected',
        medium_risk_prefix: 'Moderate risk indicators present',
        low_risk_prefix: 'Baseline stability maintained',
        include_confidence: true,
        include_sources: true
      }
    },
    rationale: 'Narrative framing affects policy response; neutral language prevents alarm fatigue',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  },
  {
    name: 'GDELT Tone Normalization',
    category: 'signal_interpretation',
    description: 'How to normalize GDELT average tone scores to risk values',
    rule_definition: {
      type: 'mapping',
      parameters: {
        tone_min: -10,
        tone_max: 10,
        risk_floor: 0,
        risk_ceiling: 1,
        invert: true  // More negative tone = higher risk
      }
    },
    rationale: 'GDELT tone ranges approximately -10 to +10; inversion maps negative sentiment to risk',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  },
  {
    name: 'Alert Threshold Configuration',
    category: 'policy_logic',
    description: 'When to trigger alerts based on computed risk scores',
    rule_definition: {
      type: 'threshold',
      parameters: {
        critical_threshold: 0.8,
        warning_threshold: 0.6,
        info_threshold: 0.4,
        cooldown_hours: 24
      }
    },
    rationale: 'Thresholds calibrated to minimize false positives while catching significant events',
    version: 1,
    effective_from: '2024-01-01',
    created_by: 'system'
  }
];

// Tier access control
export type PricingTier = 'observer' | 'operational' | 'integrated' | 'stewardship';

export function mapUserTierToPricing(userTier: string): PricingTier {
  switch (userTier) {
    case 'enterprise_tier':
      return 'stewardship';
    case 'pro':
      return 'integrated';
    case 'starter':
      return 'operational';
    default:
      return 'observer';
  }
}

export const TIER_CAPABILITIES: Record<PricingTier, {
  doctrine_read: boolean;
  doctrine_propose: boolean;
  shadow_evaluate: boolean;
  api_access: boolean;
  webhook_access: boolean;
  audit_logs: boolean;
  rate_limit: number;  // requests per minute
}> = {
  observer: {
    doctrine_read: false,
    doctrine_propose: false,
    shadow_evaluate: false,
    api_access: false,
    webhook_access: false,
    audit_logs: false,
    rate_limit: 10
  },
  operational: {
    doctrine_read: false,
    doctrine_propose: false,
    shadow_evaluate: false,
    api_access: true,
    webhook_access: false,
    audit_logs: false,
    rate_limit: 100
  },
  integrated: {
    doctrine_read: true,
    doctrine_propose: false,
    shadow_evaluate: false,
    api_access: true,
    webhook_access: true,
    audit_logs: true,
    rate_limit: 1000
  },
  stewardship: {
    doctrine_read: true,
    doctrine_propose: true,
    shadow_evaluate: true,
    api_access: true,
    webhook_access: true,
    audit_logs: true,
    rate_limit: 10000
  }
};
