'use client';

import React, { useState, useMemo } from 'react';
import { Activity, Zap, TrendingUp, AlertTriangle, Play, Pause, RotateCcw } from 'lucide-react';
import { EntropyLandscape } from '@/components/viz/EntropyLandscape';
import { usePhaseTransition } from '@/hooks/usePhaseTransition';
import { useRegimeDetection } from '@/hooks/useRegimeDetection';
import { REGIME_NAMES } from '@/lib/physics/markov-switching';

// Generate demo observations
function generateDemoObservations(length: number = 200): number[] {
  const observations: number[] = [];
  let regime = 0; // Start stable

  for (let i = 0; i < length; i++) {
    // Regime-dependent parameters
    const params = [
      { mean: 0.02, std: 0.1 },   // Stable
      { mean: 0.0, std: 0.2 },    // Volatile
      { mean: -0.05, std: 0.3 },  // Crisis
    ][regime];

    // Generate observation
    const u1 = Math.random();
    const u2 = Math.random();
    const gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    observations.push(params.mean + params.std * gaussian);

    // Regime transition
    const rand = Math.random();
    if (regime === 0 && rand < 0.02) regime = 1;
    else if (regime === 1 && rand < 0.1) regime = rand < 0.05 ? 0 : 2;
    else if (regime === 2 && rand < 0.05) regime = 1;
  }

  return observations;
}

export default function PhaseDynamicsPage() {
  const [stressFactors, setStressFactors] = useState({
    economic: 0.3,
    military: 0.2,
    political: 0.25,
  });

  const demoObservations = useMemo(() => generateDemoObservations(300), []);

  const phaseTransition = usePhaseTransition({
    stressFactors,
    autoSimulate: false,
    simulationSpeed: 30,
    onTransition: (event) => {
      console.log('Phase transition:', event);
    },
  });

  const regimeDetection = useRegimeDetection(demoObservations, {
    forecastHorizons: [1, 5, 10, 30],
  });

  const getRegimeColor = (regime: number | string) => {
    const r = typeof regime === 'string' ? regime : REGIME_NAMES[regime as 0 | 1 | 2];
    switch (r) {
      case 'STABLE':
        return 'text-blue-400 bg-blue-400/10 border-blue-400/30';
      case 'VOLATILE':
        return 'text-amber-400 bg-amber-400/10 border-amber-400/30';
      case 'CRISIS':
        return 'text-red-400 bg-red-400/10 border-red-400/30';
      default:
        return 'text-slate-400 bg-slate-400/10 border-slate-400/30';
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30">
            <Activity size={28} className="text-cyan-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white font-mono">
              Phase Dynamics
            </h1>
            <p className="text-sm text-slate-400">
              Landau-Ginzburg potential & Markov-switching regime detection
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() =>
              phaseTransition.isSimulating
                ? phaseTransition.stopSimulation()
                : phaseTransition.startSimulation()
            }
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-mono transition-colors ${
              phaseTransition.isSimulating
                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                : 'bg-green-500/20 text-green-400 border border-green-500/30'
            }`}
          >
            {phaseTransition.isSimulating ? (
              <>
                <Pause size={16} /> Stop
              </>
            ) : (
              <>
                <Play size={16} /> Simulate
              </>
            )}
          </button>
          <button
            onClick={phaseTransition.reset}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 transition-colors"
          >
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <div className={`p-4 rounded-xl border ${getRegimeColor(regimeDetection.currentRegime)}`}>
          <div className="text-xs font-mono text-slate-400 mb-1">Current Regime</div>
          <div className="text-2xl font-bold font-mono">{regimeDetection.regimeName}</div>
        </div>

        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700">
          <div className="text-xs font-mono text-slate-400 mb-1">Transition Risk</div>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {(regimeDetection.transitionRisk * 100).toFixed(1)}%
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700">
          <div className="text-xs font-mono text-slate-400 mb-1">Escape Rate</div>
          <div className="text-2xl font-bold font-mono text-cyan-400">
            {phaseTransition.escapeRate.toFixed(4)}
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700">
          <div className="text-xs font-mono text-slate-400 mb-1">Basin Stability</div>
          <div className="text-2xl font-bold font-mono text-green-400">
            {(phaseTransition.currentState.stability * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Main Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Landau-Ginzburg Potential */}
        <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700">
          <h3 className="text-sm font-mono font-bold text-slate-300 mb-4 flex items-center gap-2">
            <Zap size={16} className="text-cyan-400" />
            Landau-Ginzburg Potential Landscape
          </h3>
          <EntropyLandscape
            config={{
              a: phaseTransition.config.a,
              b: phaseTransition.config.b,
            }}
            initialState={phaseTransition.currentState}
            height={350}
          />
        </div>

        {/* Stress Factor Controls */}
        <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700">
          <h3 className="text-sm font-mono font-bold text-slate-300 mb-4 flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-400" />
            Stress Factors
          </h3>

          <div className="space-y-6">
            {Object.entries(stressFactors).map(([key, value]) => (
              <div key={key}>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-mono text-slate-400 capitalize">
                    {key} Stress
                  </span>
                  <span className="text-sm font-mono text-white">
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={value * 100}
                  onChange={(e) => {
                    const newValue = parseInt(e.target.value) / 100;
                    setStressFactors((prev) => ({ ...prev, [key]: newValue }));
                    phaseTransition.setStress(
                      key === 'economic' ? newValue : stressFactors.economic,
                      key === 'military' ? newValue : stressFactors.military,
                      key === 'political' ? newValue : stressFactors.political
                    );
                  }}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>
            ))}

            {/* Combined Stress Indicator */}
            <div className="pt-4 border-t border-slate-700">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-mono text-slate-400">Combined Stress</span>
                <span
                  className={`text-sm font-mono ${
                    phaseTransition.config.a > 0.5
                      ? 'text-red-400'
                      : phaseTransition.config.a > 0
                      ? 'text-amber-400'
                      : 'text-green-400'
                  }`}
                >
                  {phaseTransition.config.a > 0.5
                    ? 'HIGH'
                    : phaseTransition.config.a > 0
                    ? 'ELEVATED'
                    : 'LOW'}
                </span>
              </div>
              <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    phaseTransition.config.a > 0.5
                      ? 'bg-red-500'
                      : phaseTransition.config.a > 0
                      ? 'bg-amber-500'
                      : 'bg-green-500'
                  }`}
                  style={{
                    width: `${Math.abs((phaseTransition.config.a + 1) / 2) * 100}%`,
                  }}
                />
              </div>
              <p className="mt-2 text-xs text-slate-500">
                Landau coefficient a = {phaseTransition.config.a.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Regime Probabilities & Forecasts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Current Regime Probabilities */}
        <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700">
          <h3 className="text-sm font-mono font-bold text-slate-300 mb-4">
            Regime Probabilities
          </h3>
          <div className="space-y-3">
            {regimeDetection.regimeProbabilities.map((prob, i) => {
              const name = REGIME_NAMES[i as 0 | 1 | 2];
              const colors = ['bg-blue-500', 'bg-amber-500', 'bg-red-500'];

              return (
                <div key={name}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-mono text-slate-400">{name}</span>
                    <span className="text-xs font-mono text-white">
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${colors[i]}`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Forecasts */}
        <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700 col-span-2">
          <h3 className="text-sm font-mono font-bold text-slate-300 mb-4 flex items-center gap-2">
            <TrendingUp size={16} className="text-green-400" />
            Regime Forecasts
          </h3>
          <div className="grid grid-cols-4 gap-4">
            {[1, 5, 10, 30].map((horizon) => {
              const forecast = regimeDetection.getForecast(horizon);

              return (
                <div key={horizon} className="p-3 rounded-lg bg-slate-900/50">
                  <div className="text-xs font-mono text-slate-500 mb-2">
                    T+{horizon} periods
                  </div>
                  <div className="space-y-1">
                    {forecast.map((prob, i) => {
                      const name = REGIME_NAMES[i as 0 | 1 | 2];
                      const colors = ['text-blue-400', 'text-amber-400', 'text-red-400'];

                      return (
                        <div key={name} className="flex items-center justify-between">
                          <span className="text-[10px] font-mono text-slate-400">
                            {name.slice(0, 3)}
                          </span>
                          <span className={`text-xs font-mono ${colors[i]}`}>
                            {(prob * 100).toFixed(0)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Transition History */}
      {phaseTransition.transitions.length > 0 && (
        <div className="p-6 rounded-xl bg-slate-800/30 border border-slate-700">
          <h3 className="text-sm font-mono font-bold text-slate-300 mb-4">
            Phase Transition History
          </h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {phaseTransition.transitions.slice(-10).reverse().map((t, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-2 rounded bg-slate-900/50"
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-slate-500">t={t.timestamp}</span>
                  <span className="text-xs font-mono text-blue-400">{t.fromBasin.toUpperCase()}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-xs font-mono text-amber-400">{t.toBasin.toUpperCase()}</span>
                </div>
                <span className="text-xs font-mono text-slate-400">
                  φ = {t.orderParameter.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
