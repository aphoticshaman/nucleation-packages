'use client';

import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  TooltipProps,
} from 'recharts';
import { Activity } from 'lucide-react';
import type { RegimeState } from '@/lib/types/causal';

interface RegimeChartProps {
  /** Regime state data over time */
  data: RegimeState[];
  /** Whether a phase transition is imminent */
  transitionImminent?: boolean;
  /** Time until transition (human readable) */
  transitionETA?: string;
  /** Optional className */
  className?: string;
}

/**
 * RegimeChart - Markov-Switching Regime Probability Visualization
 *
 * Displays stacked area chart of regime state probabilities over time:
 * - Stable (blue): Normal operating conditions
 * - Volatile (amber): Elevated uncertainty/risk
 * - Crisis (red): Extreme stress conditions
 *
 * Based on Markov-Switching models commonly used in:
 * - Financial econometrics (Hamilton 1989)
 * - Geopolitical risk modeling
 * - Infrastructure resilience assessment
 */
export function RegimeChart({
  data,
  transitionImminent = false,
  transitionETA = 'T-48H',
  className = '',
}: RegimeChartProps) {
  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: TooltipProps<number, string>) => {
    if (!active || !payload) return null;

    return (
      <div className="bg-slate-900/95 border border-slate-700 rounded-lg p-3 shadow-xl">
        <p className="text-xs font-mono text-slate-400 mb-2">{label}</p>
        <div className="space-y-1">
          {payload.map((entry, index) => (
            <div key={index} className="flex items-center justify-between gap-4">
              <span className="text-xs font-mono capitalize" style={{ color: entry.color }}>
                {entry.name}
              </span>
              <span className="text-xs font-mono text-slate-300">
                {((entry.value as number) * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Calculate current regime (last data point)
  const currentRegime = useMemo(() => {
    if (data.length === 0) return null;
    const last = data[data.length - 1];
    if (last.crisis > last.volatile && last.crisis > last.stable) return 'crisis';
    if (last.volatile > last.stable) return 'volatile';
    return 'stable';
  }, [data]);

  const regimeColors = {
    stable: '#3b82f6',
    volatile: '#f59e0b',
    crisis: '#ef4444',
  };

  return (
    <div
      className={`bg-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-lg p-4 ${className}`}
    >
      {/* Header */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
          <div
            className={`w-2 h-2 rounded-full ${
              currentRegime === 'crisis'
                ? 'bg-red-500 animate-pulse'
                : currentRegime === 'volatile'
                ? 'bg-amber-500 animate-pulse'
                : 'bg-blue-500'
            }`}
          />
          Markov-Switching Regime Probability
        </h3>
        {transitionImminent && (
          <span className="text-[10px] font-mono text-red-500 border border-red-500/30 px-2 py-0.5 rounded bg-red-500/10 animate-pulse">
            PHASE TRANSITION: {transitionETA}
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorStable" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={regimeColors.stable} stopOpacity={0.3} />
                <stop offset="95%" stopColor={regimeColors.stable} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorVolatile" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={regimeColors.volatile} stopOpacity={0.3} />
                <stop offset="95%" stopColor={regimeColors.volatile} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorCrisis" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={regimeColors.crisis} stopOpacity={0.3} />
                <stop offset="95%" stopColor={regimeColors.crisis} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="timestamp"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 1]}
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="stable"
              stackId="1"
              stroke={regimeColors.stable}
              fill="url(#colorStable)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="volatile"
              stackId="1"
              stroke={regimeColors.volatile}
              fill="url(#colorVolatile)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="crisis"
              stackId="1"
              stroke={regimeColors.crisis}
              fill="url(#colorCrisis)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-slate-700">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-blue-500" />
          <span className="text-[10px] font-mono text-slate-400">STABLE</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-amber-500" />
          <span className="text-[10px] font-mono text-slate-400">VOLATILE</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-sm bg-red-500" />
          <span className="text-[10px] font-mono text-slate-400">CRISIS</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Generate demo regime data simulating a transition from stable → volatile → crisis
 */
export function generateDemoRegimeData(steps = 50): RegimeState[] {
  const data: RegimeState[] = [];

  for (let i = 0; i < steps; i++) {
    let stable = 0,
      volatile = 0,
      crisis = 0;

    if (i < 20) {
      // Stable regime
      stable = 0.8 + Math.random() * 0.1;
      volatile = 0.15 + Math.random() * 0.05;
      crisis = 0.05;
    } else if (i < 35) {
      // Volatile transition
      stable = 0.3 + Math.random() * 0.1;
      volatile = 0.6 + Math.random() * 0.1;
      crisis = 0.1;
    } else {
      // Crisis regime
      stable = 0.1;
      volatile = 0.2 + Math.random() * 0.1;
      crisis = 0.7 + Math.random() * 0.1;
    }

    // Normalize to sum to 1
    const total = stable + volatile + crisis;

    data.push({
      timestamp: `T-${steps - i}`,
      stable: stable / total,
      volatile: volatile / total,
      crisis: crisis / total,
      transitionProbability: i === 19 || i === 34 ? 0.9 : 0.1,
    });
  }

  return data;
}

export default RegimeChart;
