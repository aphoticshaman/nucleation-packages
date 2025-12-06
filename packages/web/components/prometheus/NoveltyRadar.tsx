'use client';

import React, { useState } from 'react';
import {
  AlertTriangle,
  Zap,
  GitBranch,
  Clock,
  Network,
  Sparkles,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  type LucideIcon,
} from 'lucide-react';
import type { NoveltySignal, NoveltyType, KnowledgeQuadrant } from '@/lib/prometheus/novelty-detection';

interface NoveltyRadarProps {
  signals: NoveltySignal[];
  quadrant?: KnowledgeQuadrant;
  maxDisplay?: number;
  onSignalClick?: (signal: NoveltySignal) => void;
}

const NOVELTY_CONFIG: Record<
  NoveltyType,
  { icon: LucideIcon; color: string; label: string }
> = {
  PATTERN_ANOMALY: {
    icon: AlertTriangle,
    color: 'text-amber-400',
    label: 'Pattern Anomaly',
  },
  STRUCTURAL_GAP: {
    icon: Network,
    color: 'text-blue-400',
    label: 'Structural Gap',
  },
  CROSS_DOMAIN_ISO: {
    icon: GitBranch,
    color: 'text-purple-400',
    label: 'Cross-Domain Isomorphism',
  },
  TEMPORAL_BREAK: {
    icon: Clock,
    color: 'text-red-400',
    label: 'Temporal Break',
  },
  CAUSAL_LOOP: {
    icon: Zap,
    color: 'text-cyan-400',
    label: 'Causal Loop',
  },
  EMERGENT_PROPERTY: {
    icon: Sparkles,
    color: 'text-green-400',
    label: 'Emergent Property',
  },
};

/**
 * Novelty Radar - Unknown Knowns Detection Display
 *
 * Visualizes detected novelty signals from the PROMETHEUS
 * novelty detection engine with epistemic classification.
 */
export function NoveltyRadar({
  signals,
  quadrant,
  maxDisplay = 10,
  onSignalClick,
}: NoveltyRadarProps) {
  const [expandedSignal, setExpandedSignal] = useState<string | null>(null);
  const [activeQuadrant, setActiveQuadrant] = useState<keyof KnowledgeQuadrant | null>(null);

  const displaySignals = signals.slice(0, maxDisplay);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500';
    if (confidence >= 0.6) return 'bg-amber-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-500/20 border border-purple-500/30">
            <Sparkles size={20} className="text-purple-400" />
          </div>
          <div>
            <h3 className="font-bold text-white font-mono">Novelty Radar</h3>
            <p className="text-xs text-slate-400">Unknown Knowns Detection</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs font-mono text-slate-400">
            {signals.length} signals detected
          </span>
        </div>
      </div>

      {/* Knowledge Quadrant (if provided) */}
      {quadrant && (
        <div className="grid grid-cols-2 gap-2">
          {(Object.entries(quadrant) as [keyof KnowledgeQuadrant, string[]][]).map(
            ([key, items]) => {
              const isActive = activeQuadrant === key;
              const labels: Record<keyof KnowledgeQuadrant, { label: string; color: string }> = {
                knownKnowns: { label: 'Known Knowns', color: 'bg-green-500/20 border-green-500/30' },
                knownUnknowns: { label: 'Known Unknowns', color: 'bg-blue-500/20 border-blue-500/30' },
                unknownUnknowns: { label: 'Unknown Unknowns', color: 'bg-red-500/20 border-red-500/30' },
                unknownKnowns: { label: 'Unknown Knowns', color: 'bg-purple-500/20 border-purple-500/30' },
              };

              return (
                <button
                  key={key}
                  onClick={() => setActiveQuadrant(isActive ? null : key)}
                  className={`p-3 rounded-lg border text-left transition-all ${
                    labels[key].color
                  } ${isActive ? 'ring-2 ring-white/20' : ''}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-slate-300">
                      {labels[key].label}
                    </span>
                    <span className="text-xs font-mono text-slate-400">
                      {items.length}
                    </span>
                  </div>
                  {isActive && items.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {items.slice(0, 3).map((item, i) => (
                        <p key={i} className="text-[10px] text-slate-400 truncate">
                          • {item}
                        </p>
                      ))}
                      {items.length > 3 && (
                        <p className="text-[10px] text-slate-500">
                          +{items.length - 3} more
                        </p>
                      )}
                    </div>
                  )}
                </button>
              );
            }
          )}
        </div>
      )}

      {/* Signals List */}
      <div className="space-y-2">
        {displaySignals.map(signal => {
          const config = NOVELTY_CONFIG[signal.type];
          const Icon = config.icon;
          const isExpanded = expandedSignal === signal.id;

          return (
            <div
              key={signal.id}
              className="rounded-lg bg-slate-800/50 border border-slate-700 overflow-hidden"
            >
              {/* Signal Header */}
              <button
                onClick={() => setExpandedSignal(isExpanded ? null : signal.id)}
                className="w-full p-3 flex items-center gap-3 hover:bg-slate-700/30 transition-colors"
              >
                <Icon size={16} className={config.color} />

                <div className="flex-1 text-left">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-mono ${config.color}`}>
                      {config.label}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 line-clamp-1">
                    {signal.description}
                  </p>
                </div>

                {/* Confidence/Significance bars */}
                <div className="flex items-center gap-2">
                  <div className="text-right">
                    <div className="text-[10px] text-slate-500">Conf</div>
                    <div className="w-12 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${getConfidenceColor(signal.confidence)}`}
                        style={{ width: `${signal.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] text-slate-500">Sig</div>
                    <div className="w-12 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500"
                        style={{ width: `${signal.significance * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {isExpanded ? (
                  <ChevronDown size={16} className="text-slate-400" />
                ) : (
                  <ChevronRight size={16} className="text-slate-400" />
                )}
              </button>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="px-3 pb-3 border-t border-slate-700">
                  {/* Evidence */}
                  <div className="mt-3">
                    <div className="text-xs font-mono text-slate-400 mb-2">Evidence</div>
                    <ul className="space-y-1">
                      {signal.evidence.map((e, i) => (
                        <li
                          key={i}
                          className="text-xs text-slate-300 flex items-start gap-2"
                        >
                          <span className="text-slate-500">•</span>
                          {e}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Suggested Action */}
                  <div className="mt-3 p-2 rounded bg-cyan-500/10 border border-cyan-500/20">
                    <div className="text-xs font-mono text-cyan-400 mb-1">
                      Suggested Action
                    </div>
                    <p className="text-xs text-slate-300">{signal.suggestedAction}</p>
                  </div>

                  {/* Action Button */}
                  {onSignalClick && (
                    <button
                      onClick={() => onSignalClick(signal)}
                      className="mt-3 w-full flex items-center justify-center gap-2 p-2 rounded bg-slate-700 hover:bg-slate-600 text-xs font-mono text-slate-300 transition-colors"
                    >
                      Investigate
                      <ExternalLink size={12} />
                    </button>
                  )}
                </div>
              )}
            </div>
          );
        })}

        {signals.length > maxDisplay && (
          <div className="text-center text-xs font-mono text-slate-500 py-2">
            +{signals.length - maxDisplay} more signals
          </div>
        )}

        {signals.length === 0 && (
          <div className="text-center py-8 text-slate-500">
            <Sparkles size={24} className="mx-auto mb-2 opacity-50" />
            <p className="text-sm">No novelty signals detected</p>
            <p className="text-xs mt-1">The system is monitoring for anomalies</p>
          </div>
        )}
      </div>
    </div>
  );
}
