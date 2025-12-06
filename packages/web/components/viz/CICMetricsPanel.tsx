'use client';

import React from 'react';
import { Activity, GitBranch, Zap, Layers, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { CICMetrics } from '@/lib/types/causal';

interface CICMetricsPanelProps {
  metrics: CICMetrics;
  className?: string;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: React.ElementType;
  trend?: 'up' | 'down' | 'stable';
  color: string;
  description: string;
  tooltip?: string;
}

function MetricCard({ label, value, icon: Icon, trend, color, description }: MetricCardProps) {
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;

  return (
    <div className="bg-slate-900/40 hover:bg-slate-900/60 border border-white/5 rounded-lg p-3 relative overflow-hidden transition-colors">
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
          {label}
        </span>
        <Icon size={14} className={`${color} opacity-80`} />
      </div>
      <div className={`text-2xl font-mono font-bold ${color} mb-1`}>
        {typeof value === 'number' ? value.toFixed(2) : value}
      </div>
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-slate-400 truncate">{description}</span>
        <span className="text-[10px] font-mono text-slate-500 flex items-center gap-1">
          <TrendIcon size={10} />
          {trend === 'up' ? '▲' : trend === 'down' ? '▼' : '−'}
        </span>
      </div>
    </div>
  );
}

interface ProgressBarProps {
  label: string;
  value: number; // 0-100
  color: string;
}

function ProgressBar({ label, value, color }: ProgressBarProps) {
  return (
    <div>
      <div className="flex justify-between text-[10px] font-mono text-slate-500 mb-1">
        <span>{label}</span>
        <span>{value.toFixed(0)}%</span>
      </div>
      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
        />
      </div>
    </div>
  );
}

/**
 * CICMetricsPanel - Causal Information Coherence Theory Metrics
 *
 * Displays key analytical confidence metrics:
 * - Φ (Phi): Integrated Information - System irreducibility
 * - H(T|X): Conditional Entropy - Compression quality
 * - C_multi: Multi-scale Causal Power - Effective causation
 * - F[T]: Free Energy - Optimization state
 *
 * Based on:
 * - Integrated Information Theory (Tononi)
 * - Free Energy Principle (Friston)
 * - Transfer Entropy / Causal Inference
 */
export function CICMetricsPanel({ metrics, className = '' }: CICMetricsPanelProps) {
  // Derive trend indicators based on typical "good" directions
  const phiTrend = metrics.phi > 3 ? 'up' : metrics.phi < 1 ? 'down' : 'stable';
  const entropyTrend = metrics.entropy < 0.3 ? 'up' : metrics.entropy > 0.7 ? 'down' : 'stable';
  const causalTrend = metrics.causalMulti > 0.7 ? 'up' : metrics.causalMulti < 0.3 ? 'down' : 'stable';
  const feTrend = metrics.freeEnergy < 15 ? 'up' : metrics.freeEnergy > 30 ? 'down' : 'stable';

  // Calculate epistemic confidence bars
  const sourceReliability = Math.min(100, metrics.causalMulti * 100 + 15);
  const modelConfidence = Math.min(100, 100 - metrics.entropy * 100);
  const evidenceConflict = Math.min(100, (1 - metrics.causalMulti) * 40);
  const cicCoherence = Math.min(100, (metrics.phi / 5) * 100);

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Main CIC Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="INTEGRATION (Φ)"
          value={metrics.phi}
          icon={Activity}
          trend={phiTrend}
          color="text-blue-400"
          description="System Coherence"
          tooltip="Integrated Information - measures irreducibility of the system. Higher = more unified."
        />
        <MetricCard
          label="ENTROPY H(T|X)"
          value={metrics.entropy}
          icon={Layers}
          trend={entropyTrend}
          color="text-cyan-400"
          description="Compression Quality"
          tooltip="Conditional entropy of Theory given Evidence. Lower = better prediction."
        />
        <MetricCard
          label="CAUSAL (C_multi)"
          value={metrics.causalMulti}
          icon={GitBranch}
          trend={causalTrend}
          color="text-green-400"
          description="Effective Power"
          tooltip="Multi-scale causal power. Higher = stronger causal relationships."
        />
        <MetricCard
          label="FREE ENERGY F[T]"
          value={metrics.freeEnergy}
          icon={Zap}
          trend={feTrend}
          color="text-amber-400"
          description="Optimization State"
          tooltip="Variational free energy. Lower = better model fit, less surprise."
        />
      </div>

      {/* Epistemic Humidity Bars */}
      <div className="bg-slate-900/40 border border-white/5 rounded-lg p-4">
        <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest mb-4">
          Epistemic Humidity
        </h3>
        <div className="space-y-3">
          <ProgressBar label="SOURCE RELIABILITY" value={sourceReliability} color="bg-blue-500" />
          <ProgressBar label="MODEL CONFIDENCE" value={modelConfidence} color="bg-amber-500" />
          <ProgressBar label="EVIDENCE CONFLICT" value={evidenceConflict} color="bg-red-500" />
          <ProgressBar label="CIC COHERENCE" value={cicCoherence} color="bg-cyan-500" />
        </div>
      </div>
    </div>
  );
}

/**
 * Generate demo CIC metrics
 */
export function generateDemoCICMetrics(): CICMetrics {
  return {
    phi: 4.25, // High integration
    entropy: 0.32, // Low conditional entropy (good compression)
    causalMulti: 0.88, // Strong multi-scale causality
    freeEnergy: 12.4, // Moderate free energy
  };
}

export default CICMetricsPanel;
