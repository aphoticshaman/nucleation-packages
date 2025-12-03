'use client';

import { useState, useMemo } from 'react';
import {
  KnowledgeQuadrant,
  EpistemicClaim,
  KnowledgeGap,
  FuzzyNumber,
  defuzzify,
  applyEpistemicBounds,
  EPISTEMIC_PROOFS,
} from '@/lib/epistemic-engine';

interface EpistemicDashboardProps {
  claims?: EpistemicClaim[];
  gaps?: KnowledgeGap[];
  onClaimClick?: (claim: EpistemicClaim) => void;
  onGapClick?: (gap: KnowledgeGap) => void;
}

// Epistemic Knowledge Quadrant Visualization
export function EpistemicDashboard({
  claims = [],
  gaps = [],
  onClaimClick,
  onGapClick,
}: EpistemicDashboardProps) {
  const [selectedQuadrant, setSelectedQuadrant] = useState<KnowledgeQuadrant | null>(null);
  const [showProofs, setShowProofs] = useState(false);

  // Group claims by quadrant
  const quadrantCounts = useMemo(() => {
    const counts: Record<KnowledgeQuadrant, number> = {
      known_known: 0,
      known_unknown: 0,
      unknown_unknown: 0,
      unknown_known: 0,
    };
    claims.forEach(c => {
      counts[c.quadrant]++;
    });
    return counts;
  }, [claims]);

  // Calculate epistemic health score
  const epistemicHealth = useMemo(() => {
    const total = claims.length || 1;
    const knownKnownRatio = quadrantCounts.known_known / total;
    const uncertaintyRatio = (quadrantCounts.unknown_unknown + quadrantCounts.known_unknown) / total;

    // Higher known_known and lower uncertainty = better health
    return Math.max(0, Math.min(1, knownKnownRatio - uncertaintyRatio * 0.5 + 0.3));
  }, [claims, quadrantCounts]);

  const quadrantConfig: Record<KnowledgeQuadrant, {
    label: string;
    description: string;
    color: string;
    bgColor: string;
    borderColor: string;
    icon: string;
  }> = {
    known_known: {
      label: 'Known Knowns',
      description: 'Verified facts with high confidence',
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
      borderColor: 'border-green-500/50',
      icon: '✓',
    },
    known_unknown: {
      label: 'Known Unknowns',
      description: 'Identified gaps requiring investigation',
      color: 'text-amber-400',
      bgColor: 'bg-amber-500/10',
      borderColor: 'border-amber-500/50',
      icon: '?',
    },
    unknown_unknown: {
      label: 'Unknown Unknowns',
      description: 'Blind spots detected via anomaly',
      color: 'text-red-400',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/50',
      icon: '⚠',
    },
    unknown_known: {
      label: 'Unknown Knowns',
      description: 'Implicit knowledge awaiting formalization',
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
      borderColor: 'border-purple-500/50',
      icon: '◐',
    },
  };

  return (
    <div className="space-y-6">
      {/* Header with epistemic health */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium text-slate-200">Epistemic State</h2>
          <p className="text-xs text-slate-500">Knowledge quadrant analysis</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowProofs(!showProofs)}
            className="px-3 py-1 text-xs text-slate-400 hover:text-cyan-400 transition-colors"
          >
            {showProofs ? 'Hide' : 'Show'} Proofs
          </button>
          <EpistemicHealthGauge health={epistemicHealth} />
        </div>
      </div>

      {/* Quadrant Grid */}
      <div className="grid grid-cols-2 gap-4">
        {(Object.keys(quadrantConfig) as KnowledgeQuadrant[]).map(quadrant => {
          const config = quadrantConfig[quadrant];
          const count = quadrantCounts[quadrant];
          const isSelected = selectedQuadrant === quadrant;
          const quadrantClaims = claims.filter(c => c.quadrant === quadrant);

          return (
            <button
              key={quadrant}
              onClick={() => setSelectedQuadrant(isSelected ? null : quadrant)}
              className={`
                p-4 rounded-lg border transition-all text-left
                ${config.bgColor} ${config.borderColor}
                ${isSelected ? 'ring-2 ring-offset-2 ring-offset-slate-900' : ''}
                hover:scale-[1.02]
              `}
              style={{
                boxShadow: isSelected ? `0 0 20px ${config.color.replace('text-', 'rgb(').replace('-400', ' / 0.3)')}` : undefined,
              }}
            >
              <div className="flex items-start justify-between">
                <div>
                  <span className={`text-2xl ${config.color}`}>{config.icon}</span>
                  <h3 className={`text-sm font-medium mt-2 ${config.color}`}>
                    {config.label}
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    {config.description}
                  </p>
                </div>
                <span className={`text-3xl font-bold font-mono ${config.color}`}>
                  {count}
                </span>
              </div>

              {/* Mini preview of claims */}
              {quadrantClaims.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-700/50">
                  <div className="text-xs text-slate-400 truncate">
                    {quadrantClaims[0].claim.slice(0, 50)}...
                  </div>
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Selected quadrant detail */}
      {selectedQuadrant && (
        <QuadrantDetail
          quadrant={selectedQuadrant}
          claims={claims.filter(c => c.quadrant === selectedQuadrant)}
          gaps={gaps.filter(g => g.quadrant === selectedQuadrant)}
          config={quadrantConfig[selectedQuadrant]}
          onClaimClick={onClaimClick}
          onGapClick={onGapClick}
        />
      )}

      {/* Knowledge gaps priority list */}
      {gaps.length > 0 && (
        <KnowledgeGapsList gaps={gaps} onGapClick={onGapClick} />
      )}

      {/* Epistemic proofs reference */}
      {showProofs && <EpistemicProofsPanel />}
    </div>
  );
}

// Epistemic Health Gauge
function EpistemicHealthGauge({ health }: { health: number }) {
  const color = health >= 0.7 ? 'text-green-400' :
                health >= 0.4 ? 'text-amber-400' :
                'text-red-400';

  const label = health >= 0.7 ? 'Strong' :
                health >= 0.4 ? 'Moderate' :
                'Weak';

  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            health >= 0.7 ? 'bg-green-500' :
            health >= 0.4 ? 'bg-amber-500' :
            'bg-red-500'
          }`}
          style={{ width: `${health * 100}%` }}
        />
      </div>
      <span className={`text-xs font-medium ${color}`}>{label}</span>
    </div>
  );
}

// Quadrant detail panel
function QuadrantDetail({
  quadrant,
  claims,
  gaps,
  config,
  onClaimClick,
  onGapClick,
}: {
  quadrant: KnowledgeQuadrant;
  claims: EpistemicClaim[];
  gaps: KnowledgeGap[];
  config: { label: string; color: string; bgColor: string; borderColor: string };
  onClaimClick?: (claim: EpistemicClaim) => void;
  onGapClick?: (gap: KnowledgeGap) => void;
}) {
  return (
    <div className={`p-4 rounded-lg border ${config.bgColor} ${config.borderColor}`}>
      <h3 className={`text-sm font-medium mb-3 ${config.color}`}>
        {config.label} ({claims.length})
      </h3>

      {claims.length === 0 ? (
        <p className="text-xs text-slate-500">No items in this quadrant</p>
      ) : (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {claims.map(claim => (
            <button
              key={claim.id}
              onClick={() => onClaimClick?.(claim)}
              className="w-full text-left p-2 rounded bg-slate-800/50 hover:bg-slate-800 transition-colors"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-xs text-slate-300 line-clamp-2">{claim.claim}</p>
                <div className="flex-shrink-0">
                  <ConfidenceBadge confidence={claim.confidence} uncertainty={claim.uncertainty} />
                </div>
              </div>
              {claim.historicalCorrelates.length > 0 && (
                <div className="mt-1 text-xs text-purple-400">
                  ↳ {claim.historicalCorrelates[0].eventName}
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {gaps.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700/50">
          <h4 className="text-xs text-slate-400 mb-2">Related Gaps</h4>
          {gaps.slice(0, 3).map(gap => (
            <button
              key={gap.id}
              onClick={() => onGapClick?.(gap)}
              className="w-full text-left p-2 rounded bg-slate-800/30 hover:bg-slate-800/50 mb-1"
            >
              <p className="text-xs text-slate-400">{gap.description}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Confidence badge with uncertainty
function ConfidenceBadge({ confidence, uncertainty }: { confidence: number; uncertainty: number }) {
  const color = confidence >= 0.8 ? 'bg-green-500/20 text-green-400' :
                confidence >= 0.5 ? 'bg-amber-500/20 text-amber-400' :
                'bg-red-500/20 text-red-400';

  return (
    <div className={`px-1.5 py-0.5 rounded text-xs font-mono ${color}`}>
      {(confidence * 100).toFixed(0)}%
      {uncertainty > 0.2 && (
        <span className="text-slate-500 ml-1">±{(uncertainty * 100).toFixed(0)}</span>
      )}
    </div>
  );
}

// Knowledge gaps priority list
function KnowledgeGapsList({
  gaps,
  onGapClick,
}: {
  gaps: KnowledgeGap[];
  onGapClick?: (gap: KnowledgeGap) => void;
}) {
  const priorityConfig = {
    critical: { color: 'text-red-400', bg: 'bg-red-500/10', icon: '🚨' },
    high: { color: 'text-orange-400', bg: 'bg-orange-500/10', icon: '⚡' },
    medium: { color: 'text-amber-400', bg: 'bg-amber-500/10', icon: '◉' },
    low: { color: 'text-slate-400', bg: 'bg-slate-500/10', icon: '○' },
  };

  const sortedGaps = [...gaps].sort((a, b) => {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    return order[a.priority] - order[b.priority];
  });

  return (
    <div className="bg-slate-900/50 rounded-lg border border-slate-700 p-4">
      <h3 className="text-sm font-medium text-slate-200 mb-3">
        Knowledge Gaps ({gaps.length})
      </h3>
      <div className="space-y-2">
        {sortedGaps.slice(0, 5).map(gap => {
          const config = priorityConfig[gap.priority];
          return (
            <button
              key={gap.id}
              onClick={() => onGapClick?.(gap)}
              className={`w-full text-left p-3 rounded-lg border border-slate-700/50 ${config.bg} hover:border-slate-600 transition-colors`}
            >
              <div className="flex items-start gap-2">
                <span>{config.icon}</span>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm ${config.color}`}>{gap.description}</p>
                  <p className="text-xs text-slate-500 mt-1">{gap.proposedResolution}</p>
                </div>
                <span className="text-xs text-slate-500">
                  ~{gap.estimatedEffortHours}h
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Epistemic proofs reference panel
function EpistemicProofsPanel() {
  return (
    <div className="bg-slate-900/50 rounded-lg border border-cyan-500/30 p-4">
      <h3 className="text-sm font-medium text-cyan-400 mb-3">
        Epistemic Humility Proofs
      </h3>
      <div className="space-y-4">
        {EPISTEMIC_PROOFS.map(proof => (
          <div key={proof.id} className="border-l-2 border-cyan-500/30 pl-3">
            <h4 className="text-xs font-medium text-slate-300">{proof.name}</h4>
            <p className="text-xs text-slate-400 mt-1">{proof.statement}</p>
            <div className="mt-2 text-xs text-slate-500 font-mono">
              Max confidence bound: {(proof.confidenceBound * 100).toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Fuzzy number visualization
export function FuzzyDisplay({ fuzzy, label }: { fuzzy: FuzzyNumber; label?: string }) {
  const center = defuzzify(fuzzy);
  const width = fuzzy.high - fuzzy.low;

  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-xs text-slate-400">{label}:</span>}
      <div className="relative w-24 h-4">
        {/* Range bar */}
        <div
          className="absolute top-1 h-2 bg-cyan-500/30 rounded"
          style={{
            left: `${fuzzy.low * 100}%`,
            width: `${width * 100}%`,
          }}
        />
        {/* Peak marker */}
        <div
          className="absolute top-0 w-0.5 h-4 bg-cyan-400"
          style={{ left: `${fuzzy.peak * 100}%` }}
        />
        {/* Center marker */}
        <div
          className="absolute top-1 w-1 h-2 bg-white rounded-full"
          style={{ left: `${center * 100}%`, transform: 'translateX(-50%)' }}
        />
      </div>
      <span className="text-xs font-mono text-cyan-400">
        {(center * 100).toFixed(0)}%
      </span>
    </div>
  );
}

// Confidence with epistemic bounds applied
export function BoundedConfidence({
  rawConfidence,
  timeHorizonMonths = 0,
  cascadeSteps = 1,
  domainComplexity = 'complex',
}: {
  rawConfidence: number;
  timeHorizonMonths?: number;
  cascadeSteps?: number;
  domainComplexity?: 'simple' | 'complex' | 'chaotic';
}) {
  const bounded = applyEpistemicBounds(rawConfidence, {
    timeHorizonMonths,
    cascadeSteps,
    domainComplexity,
  });

  const reduction = rawConfidence - bounded;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 line-through">{(rawConfidence * 100).toFixed(0)}%</span>
      <span className="text-xs text-cyan-400 font-mono">{(bounded * 100).toFixed(0)}%</span>
      {reduction > 0.05 && (
        <span className="text-xs text-amber-400">
          (-{(reduction * 100).toFixed(0)}% epistemic bounds)
        </span>
      )}
    </div>
  );
}

// Mock data for demo
export const mockEpistemicClaims: EpistemicClaim[] = [
  {
    id: '1',
    claim: 'Russia will maintain pressure on Ukraine through winter 2024-2025',
    quadrant: 'known_known',
    confidence: 0.85,
    uncertainty: 0.1,
    sources: ['reuters-1', 'nato-2'],
    historicalCorrelates: [
      {
        eventId: 'hundred-years-war',
        eventName: 'Hundred Years War attritional phases',
        period: '1337-1453',
        yearsAgo: 687,
        correlationStrength: 0.72,
        correlationType: 'structural',
        keyVariables: ['siege warfare', 'economic attrition', 'alliance fatigue'],
        divergences: ['nuclear dimension', 'global economic integration'],
      },
    ],
    hypotheses: [],
    derivatives: [],
    lastUpdated: new Date().toISOString(),
  },
  {
    id: '2',
    claim: 'China-Taiwan tensions will not escalate to kinetic conflict in 2025',
    quadrant: 'known_unknown',
    confidence: 0.55,
    uncertainty: 0.35,
    sources: ['csis-1'],
    historicalCorrelates: [],
    hypotheses: [
      {
        id: 'h1',
        statement: 'Economic interdependence prevents escalation',
        predictedOutcome: 'Continued status quo',
        probabilityIfTrue: 0.8,
        priorProbability: 0.6,
        posteriorProbability: 0.65,
        testableImplications: ['Trade volumes stable', 'No major exercises'],
        falsificationCriteria: ['Amphibious buildup detected', 'Trade restrictions >50%'],
        status: 'testing',
      },
    ],
    derivatives: [],
    lastUpdated: new Date().toISOString(),
  },
  {
    id: '3',
    claim: 'Unidentified cascade risk in semiconductor supply chain',
    quadrant: 'unknown_unknown',
    confidence: 0.25,
    uncertainty: 0.6,
    sources: [],
    historicalCorrelates: [],
    hypotheses: [],
    derivatives: [],
    lastUpdated: new Date().toISOString(),
  },
  {
    id: '4',
    claim: 'Pattern: Debt-driven military expansion historically precedes fiscal crisis',
    quadrant: 'unknown_known',
    confidence: 0.78,
    uncertainty: 0.15,
    sources: [],
    historicalCorrelates: [
      {
        eventId: 'spanish-bankruptcies',
        eventName: 'Spanish Imperial Bankruptcies',
        period: '1557-1627',
        yearsAgo: 468,
        correlationStrength: 0.85,
        correlationType: 'causal',
        keyVariables: ['military spending', 'debt service', 'currency pressure'],
        divergences: ['fiat currency', 'central bank intervention'],
      },
    ],
    hypotheses: [],
    derivatives: [],
    lastUpdated: new Date().toISOString(),
  },
];

export const mockKnowledgeGaps: KnowledgeGap[] = [
  {
    id: 'g1',
    description: 'Iranian nuclear breakout timeline uncertainty',
    quadrant: 'known_unknown',
    priority: 'critical',
    proposedResolution: 'Cross-reference IAEA reports with commercial satellite imagery',
    estimatedEffortHours: 8,
  },
  {
    id: 'g2',
    description: 'Chinese rare earth export restriction probability',
    quadrant: 'known_unknown',
    priority: 'high',
    proposedResolution: 'Monitor trade policy signals and inventory levels',
    estimatedEffortHours: 4,
  },
  {
    id: 'g3',
    description: 'Unmodeled second-order effects of AI on labor markets',
    quadrant: 'unknown_unknown',
    priority: 'medium',
    proposedResolution: 'Build agent-based model with historical automation data',
    estimatedEffortHours: 40,
  },
];
