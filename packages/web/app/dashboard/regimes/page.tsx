'use client';

import { PhaseBasinViz } from '@/components/viz/PhaseBasinViz';
import { RegimeChart, generateDemoRegimeData } from '@/components/viz/RegimeChart';
import { CICMetricsPanel, generateDemoCICMetrics } from '@/components/viz/CICMetricsPanel';
import { Activity, AlertTriangle, TrendingUp, Zap } from 'lucide-react';
import Link from 'next/link';

export default function RegimesPage() {
  // Demo data - in production, fetch from API
  const regimeData = generateDemoRegimeData(50);
  const cicMetrics = generateDemoCICMetrics();

  // Calculate current risk state from last data point
  const lastState = regimeData[regimeData.length - 1];
  const dominantRegime = lastState.crisis > 0.5 ? 'CRISIS' : lastState.volatile > 0.4 ? 'VOLATILE' : 'STABLE';
  const stability = 1 - (lastState.crisis + lastState.volatile * 0.5);
  const isCritical = lastState.crisis > 0.5;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="text-slate-400 hover:text-white transition-colors">
                ← Dashboard
              </Link>
              <div className="h-4 w-px bg-slate-700" />
              <h1 className="text-lg font-mono font-bold flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-400" />
                Regime Detection
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <Link
                href="/dashboard/causal"
                className="text-sm text-slate-400 hover:text-blue-400 transition-colors"
              >
                Causal Graph →
              </Link>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Alert Banner */}
        {isCritical && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 animate-pulse" />
            <div>
              <p className="text-sm font-medium text-red-400">Phase Transition Imminent</p>
              <p className="text-xs text-red-400/70">System approaching critical regime. Monitor closely.</p>
            </div>
          </div>
        )}

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className={`p-4 rounded-lg border ${
            dominantRegime === 'CRISIS' ? 'bg-red-500/10 border-red-500/30' :
            dominantRegime === 'VOLATILE' ? 'bg-amber-500/10 border-amber-500/30' :
            'bg-blue-500/10 border-blue-500/30'
          }`}>
            <p className="text-xs text-slate-400 mb-1">Current Regime</p>
            <p className={`text-xl font-mono font-bold ${
              dominantRegime === 'CRISIS' ? 'text-red-400' :
              dominantRegime === 'VOLATILE' ? 'text-amber-400' :
              'text-blue-400'
            }`}>{dominantRegime}</p>
          </div>
          <div className="p-4 rounded-lg border bg-slate-800/50 border-slate-700">
            <p className="text-xs text-slate-400 mb-1">Stability Index</p>
            <p className="text-xl font-mono font-bold text-cyan-400">{(stability * 100).toFixed(1)}%</p>
          </div>
          <div className="p-4 rounded-lg border bg-slate-800/50 border-slate-700">
            <p className="text-xs text-slate-400 mb-1">Crisis Probability</p>
            <p className="text-xl font-mono font-bold text-red-400">{(lastState.crisis * 100).toFixed(1)}%</p>
          </div>
          <div className="p-4 rounded-lg border bg-slate-800/50 border-slate-700">
            <p className="text-xs text-slate-400 mb-1">Transition Risk</p>
            <p className="text-xl font-mono font-bold text-amber-400">{(lastState.transitionProbability * 100).toFixed(1)}%</p>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Regime Chart - 2 cols */}
          <div className="lg:col-span-2">
            <RegimeChart
              data={regimeData}
              transitionImminent={isCritical}
              transitionETA="T-48H"
            />
          </div>

          {/* Phase Basin - 1 col */}
          <div className="space-y-4">
            <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-4">
              <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Phase Space Navigator
              </h3>
              <PhaseBasinViz
                stability={stability}
                isCritical={isCritical}
              />
            </div>

            {/* Phase Description */}
            <div className="bg-slate-900/50 border border-slate-700 rounded-lg p-4">
              <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-widest mb-3">
                Model Description
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Landau-Ginzburg potential V(x) = x⁴ - ax² models bistable systems near phase transitions.
                The ball represents current system state oscillating in a potential well.
                As stability decreases, wells become shallower and transitions more likely.
              </p>
            </div>
          </div>
        </div>

        {/* CIC Metrics */}
        <div className="mt-8">
          <h2 className="text-sm font-mono font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Causal Information Coherence Metrics
          </h2>
          <CICMetricsPanel metrics={cicMetrics} />
        </div>

        {/* Navigation Links */}
        <div className="mt-8 pt-8 border-t border-slate-800">
          <div className="flex flex-wrap gap-4">
            <Link
              href="/dashboard/causal"
              className="px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 border border-blue-500/30 rounded-lg text-sm font-mono transition-colors"
            >
              View Causal Graph →
            </Link>
            <Link
              href="/dashboard/intelligence"
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-sm font-mono transition-colors"
            >
              Intelligence Feed
            </Link>
            <Link
              href="/dashboard"
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg text-sm font-mono transition-colors"
            >
              Dashboard Home
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
