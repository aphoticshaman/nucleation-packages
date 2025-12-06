'use client';

import React from 'react';
import { Shield, AlertTriangle, Activity, HelpCircle, Minus } from 'lucide-react';
import type { BeliefInterval } from '@/lib/physics/dempster-shafer';

interface BeliefFusionVizProps {
  beliefIntervals: BeliefInterval[];
  probabilities: Map<string, number>;
  recommendation: string;
  confidence: number;
  conflict: number;
  sourceCount: number;
}

/**
 * Dempster-Shafer Belief Fusion Visualization
 *
 * Displays fused intelligence assessments with explicit uncertainty bounds
 * using belief (lower) and plausibility (upper) intervals.
 */
export function BeliefFusionViz({
  beliefIntervals,
  probabilities,
  recommendation,
  confidence,
  conflict,
  sourceCount,
}: BeliefFusionVizProps) {
  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'STABLE':
        return { bg: 'bg-blue-500', text: 'text-blue-400', border: 'border-blue-500' };
      case 'VOLATILE':
        return { bg: 'bg-amber-500', text: 'text-amber-400', border: 'border-amber-500' };
      case 'CRISIS':
        return { bg: 'bg-red-500', text: 'text-red-400', border: 'border-red-500' };
      default:
        return { bg: 'bg-slate-500', text: 'text-slate-400', border: 'border-slate-500' };
    }
  };

  const getRegimeIcon = (regime: string) => {
    switch (regime) {
      case 'STABLE':
        return <Shield size={16} />;
      case 'VOLATILE':
        return <Activity size={16} />;
      case 'CRISIS':
        return <AlertTriangle size={16} />;
      default:
        return <HelpCircle size={16} />;
    }
  };

  const conflictLevel = conflict < 0.2 ? 'low' : conflict < 0.5 ? 'medium' : 'high';

  return (
    <div className="space-y-6">
      {/* Header: Recommendation */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${getRegimeColor(recommendation).bg}/20`}>
            {getRegimeIcon(recommendation)}
          </div>
          <div>
            <div className="text-xs font-mono text-slate-400 uppercase tracking-wider">
              Fused Assessment
            </div>
            <div className={`text-2xl font-bold ${getRegimeColor(recommendation).text}`}>
              {recommendation}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs font-mono text-slate-400">Confidence</div>
          <div className="text-xl font-mono font-bold text-white">
            {(confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Conflict Indicator */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700">
        <div className="flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              conflictLevel === 'low'
                ? 'bg-green-500'
                : conflictLevel === 'medium'
                ? 'bg-amber-500'
                : 'bg-red-500 animate-pulse'
            }`}
          />
          <span className="text-xs font-mono text-slate-400">
            Source Conflict: {conflictLevel.toUpperCase()}
          </span>
        </div>
        <span className="text-xs font-mono text-slate-500">
          {sourceCount} sources • K={conflict.toFixed(2)}
        </span>
      </div>

      {/* Belief Intervals */}
      <div className="space-y-4">
        <div className="text-xs font-mono text-slate-400 uppercase tracking-wider flex items-center gap-2">
          <span>Belief Intervals</span>
          <HelpCircle size={12} className="text-slate-500" />
        </div>

        {beliefIntervals.map((interval) => {
          const colors = getRegimeColor(interval.element);
          const prob = probabilities.get(interval.element) || 0;

          return (
            <div key={interval.element} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded ${colors.bg}`} />
                  <span className="text-sm font-mono text-slate-300">
                    {interval.element}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-xs font-mono">
                  <span className="text-slate-500">
                    Bel: {(interval.belief * 100).toFixed(0)}%
                  </span>
                  <span className="text-slate-500">
                    Pl: {(interval.plausibility * 100).toFixed(0)}%
                  </span>
                  <span className={colors.text}>
                    P: {(prob * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {/* Visual Bar */}
              <div className="relative h-6 bg-slate-800 rounded overflow-hidden">
                {/* Full plausibility range (background) */}
                <div
                  className={`absolute inset-y-0 left-0 ${colors.bg} opacity-20`}
                  style={{ width: `${interval.plausibility * 100}%` }}
                />

                {/* Belief (certain) range */}
                <div
                  className={`absolute inset-y-0 left-0 ${colors.bg} opacity-60`}
                  style={{ width: `${interval.belief * 100}%` }}
                />

                {/* Pignistic probability marker */}
                <div
                  className="absolute top-1 bottom-1 w-1 bg-white rounded"
                  style={{ left: `${prob * 100}%` }}
                />

                {/* Labels */}
                <div className="absolute inset-0 flex items-center justify-between px-2">
                  <span className="text-[10px] font-mono text-white/70">
                    {(interval.belief * 100).toFixed(0)}%
                  </span>
                  <span className="text-[10px] font-mono text-white/50">
                    {(interval.uncertainty * 100).toFixed(0)}% uncertain
                  </span>
                  <span className="text-[10px] font-mono text-white/70">
                    {(interval.plausibility * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 pt-4 border-t border-slate-700">
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
          <div className="w-4 h-3 bg-slate-600 rounded" />
          <span>Plausibility (max)</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
          <div className="w-4 h-3 bg-slate-400 rounded" />
          <span>Belief (min)</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500">
          <div className="w-1 h-3 bg-white rounded" />
          <span>Probability</span>
        </div>
      </div>
    </div>
  );
}
