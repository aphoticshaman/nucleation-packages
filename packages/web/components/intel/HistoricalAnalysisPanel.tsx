'use client';

import { useState } from 'react';

/**
 * Historical Analysis Panel
 *
 * Granular controls for meta-analysis using Claude's verified historical knowledge.
 * Three modes:
 * - Realtime: Pure metric translation (default, cheap)
 * - Historical: Meta-analysis of historical patterns (leverages Claude's training data)
 * - Hybrid: Current metrics + historical context overlay
 */

// Historical eras Claude can confidently analyze
const HISTORICAL_ERAS = {
  ancient: { label: 'Ancient (3000 BCE - 500 CE)', description: 'Egypt, Rome, Greece, Persia' },
  medieval: { label: 'Medieval (500 - 1500)', description: 'Byzantine, Mongol Empire, Crusades' },
  earlyModern: { label: 'Early Modern (1500 - 1800)', description: 'Colonialism, Enlightenment' },
  industrial: { label: 'Industrial (1800 - 1914)', description: 'Industrial Revolution, Imperialism' },
  worldWars: { label: 'World Wars (1914 - 1945)', description: 'WWI, Interwar, WWII' },
  coldWar: { label: 'Cold War (1945 - 1991)', description: 'Nuclear Age, Proxy Wars' },
  postColdWar: { label: 'Post-Cold War (1991 - 2010)', description: 'Globalization, War on Terror' },
  modern: { label: 'Recent (2010 - 2024)', description: 'Arab Spring, Syria, Ukraine' },
} as const;

// Focus areas for analysis
const FOCUS_AREAS = [
  { id: 'geopolitical', label: 'Geopolitical Transitions', description: 'Rise/fall of powers, alliance shifts' },
  { id: 'economic', label: 'Economic Patterns', description: 'Trade, currency crises, sanctions' },
  { id: 'military', label: 'Military Doctrine', description: 'Conflict patterns, escalation dynamics' },
  { id: 'revolutionary', label: 'Revolutionary Movements', description: 'Coups, revolutions, regime change' },
  { id: 'pandemic', label: 'Pandemic/Crisis', description: 'Disease, famine, disaster response' },
  { id: 'technology', label: 'Technology Disruption', description: 'Innovation diffusion, tech competition' },
] as const;

// GDELT date ranges (goes back to 2015)
// All presets have both start and end functions for consistent typing
const today = () => new Date().toISOString().split('T')[0];

const GDELT_PRESETS: Array<{ id: string; label: string; start: () => string; end: () => string }> = [
  { id: 'last48h', label: 'Last 48 Hours', start: () => new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString().split('T')[0], end: today },
  { id: 'lastWeek', label: 'Last Week', start: () => new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], end: today },
  { id: 'lastMonth', label: 'Last Month', start: () => new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], end: today },
  { id: 'last3Months', label: 'Last 3 Months', start: () => new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], end: today },
  { id: 'lastYear', label: 'Last Year', start: () => new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], end: today },
  { id: '2024', label: '2024', start: () => '2024-01-01', end: () => '2024-12-31' },
  { id: '2023', label: '2023', start: () => '2023-01-01', end: () => '2023-12-31' },
  { id: '2022', label: '2022 (Ukraine War Start)', start: () => '2022-01-01', end: () => '2022-12-31' },
  { id: '2020', label: '2020 (COVID)', start: () => '2020-01-01', end: () => '2020-12-31' },
  { id: 'custom', label: 'Custom Range', start: () => '', end: () => '' },
];

interface HistoricalAnalysisPanelProps {
  onAnalyze: (config: AnalysisConfig) => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export interface AnalysisConfig {
  mode: 'realtime' | 'historical' | 'hybrid';
  depth: 'quick' | 'standard' | 'deep';
  gdeltPeriod?: {
    start: string;
    end: string;
  };
  historicalFocus?: string;
  selectedEras?: string[];
}

export function HistoricalAnalysisPanel({ onAnalyze, isLoading, disabled }: HistoricalAnalysisPanelProps) {
  const [mode, setMode] = useState<'realtime' | 'historical' | 'hybrid'>('realtime');
  const [depth, setDepth] = useState<'quick' | 'standard' | 'deep'>('standard');
  const [selectedEras, setSelectedEras] = useState<string[]>(['modern', 'postColdWar']);
  const [focusArea, setFocusArea] = useState('geopolitical');
  const [gdeltPreset, setGdeltPreset] = useState('lastWeek');
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');

  const handleAnalyze = () => {
    const preset = GDELT_PRESETS.find(p => p.id === gdeltPreset);
    const today = new Date().toISOString().split('T')[0];

    const config: AnalysisConfig = {
      mode,
      depth,
      ...(mode !== 'realtime' && {
        gdeltPeriod: gdeltPreset === 'custom'
          ? { start: customStart, end: customEnd || today() }
          : { start: preset?.start() || '', end: preset?.end() || today() },
        historicalFocus: FOCUS_AREAS.find(f => f.id === focusArea)?.description || focusArea,
        selectedEras,
      }),
    };

    onAnalyze(config);
  };

  // Cost estimates
  const costEstimate = {
    realtime: '$0.25-0.50',
    'historical-quick': '$0.25-0.50',
    'historical-standard': '$0.40-0.75',
    'historical-deep': '$1.00-2.00',
    'hybrid-quick': '$0.30-0.60',
    'hybrid-standard': '$0.50-1.00',
    'hybrid-deep': '$1.50-2.50',
  };
  const costKey = mode === 'realtime' ? 'realtime' : `${mode}-${depth}`;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Analysis Configuration</h3>
        <span className="text-xs text-gray-400">
          Est. cost: <span className="text-green-400">{costEstimate[costKey as keyof typeof costEstimate]}</span>
        </span>
      </div>

      {/* Mode Selection */}
      <div className="space-y-2">
        <label className="text-sm text-gray-400">Analysis Mode</label>
        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => setMode('realtime')}
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              mode === 'realtime'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            Realtime
            <span className="block text-xs opacity-70">Metric Translation</span>
          </button>
          <button
            onClick={() => setMode('historical')}
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              mode === 'historical'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            Historical
            <span className="block text-xs opacity-70">Pattern Analysis</span>
          </button>
          <button
            onClick={() => setMode('hybrid')}
            className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
              mode === 'hybrid'
                ? 'bg-cyan-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            Hybrid
            <span className="block text-xs opacity-70">Current + Historical</span>
          </button>
        </div>
      </div>

      {/* Depth Selection (for non-realtime modes) */}
      {mode !== 'realtime' && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Analysis Depth</label>
          <div className="grid grid-cols-3 gap-2">
            {(['quick', 'standard', 'deep'] as const).map((d) => (
              <button
                key={d}
                onClick={() => setDepth(d)}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  depth === d
                    ? 'bg-gray-600 text-white'
                    : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {d.charAt(0).toUpperCase() + d.slice(1)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Historical Era Selection (for historical/hybrid modes) */}
      {mode !== 'realtime' && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Historical Eras to Analyze</label>
          <div className="grid grid-cols-2 gap-1.5 max-h-40 overflow-y-auto">
            {Object.entries(HISTORICAL_ERAS).map(([key, era]) => (
              <label
                key={key}
                className={`flex items-center space-x-2 px-2 py-1.5 rounded cursor-pointer transition-colors ${
                  selectedEras.includes(key)
                    ? 'bg-purple-900/50 border border-purple-500'
                    : 'bg-gray-800 border border-transparent hover:border-gray-600'
                }`}
              >
                <input
                  type="checkbox"
                  checked={selectedEras.includes(key)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedEras([...selectedEras, key]);
                    } else {
                      setSelectedEras(selectedEras.filter(k => k !== key));
                    }
                  }}
                  className="rounded border-gray-600 bg-gray-700 text-purple-500 focus:ring-purple-500"
                />
                <div className="text-xs">
                  <div className="text-gray-200">{era.label}</div>
                  <div className="text-gray-500">{era.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* Focus Area */}
      {mode !== 'realtime' && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">Analysis Focus</label>
          <select
            value={focusArea}
            onChange={(e) => setFocusArea(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white"
          >
            {FOCUS_AREAS.map((focus) => (
              <option key={focus.id} value={focus.id}>
                {focus.label} - {focus.description}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* GDELT Period */}
      {mode !== 'realtime' && (
        <div className="space-y-2">
          <label className="text-sm text-gray-400">GDELT Data Period</label>
          <select
            value={gdeltPreset}
            onChange={(e) => setGdeltPreset(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-white"
          >
            {GDELT_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.label}
              </option>
            ))}
          </select>

          {gdeltPreset === 'custom' && (
            <div className="grid grid-cols-2 gap-2 mt-2">
              <input
                type="date"
                value={customStart}
                onChange={(e) => setCustomStart(e.target.value)}
                min="2015-02-18"
                max={new Date().toISOString().split('T')[0]}
                className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
                placeholder="Start date"
              />
              <input
                type="date"
                value={customEnd}
                onChange={(e) => setCustomEnd(e.target.value)}
                min={customStart || '2015-02-18'}
                max={new Date().toISOString().split('T')[0]}
                className="bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-sm text-white"
                placeholder="End date"
              />
            </div>
          )}
        </div>
      )}

      {/* Mode Description */}
      <div className="text-xs text-gray-500 bg-gray-800/50 rounded p-2">
        {mode === 'realtime' && (
          <>
            <strong className="text-blue-400">Realtime Mode:</strong> Translates current pipeline metrics into prose.
            Fast and cost-effective. No historical analysis.
          </>
        )}
        {mode === 'historical' && (
          <>
            <strong className="text-purple-400">Historical Mode:</strong> Meta-analysis using Claude&apos;s verified historical
            knowledge (3000 BCE → Jan 2025). Identifies patterns, precedents, and cycles. Zero hallucination risk on historical events.
          </>
        )}
        {mode === 'hybrid' && (
          <>
            <strong className="text-cyan-400">Hybrid Mode:</strong> Combines current metrics with historical context.
            Shows what numbers mean NOW and what similar patterns meant HISTORICALLY.
          </>
        )}
      </div>

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={isLoading || disabled}
        className={`w-full py-2.5 rounded font-medium transition-colors ${
          isLoading || disabled
            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
            : mode === 'realtime'
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : mode === 'historical'
                ? 'bg-purple-600 hover:bg-purple-500 text-white'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white'
        }`}
      >
        {isLoading ? (
          <span className="flex items-center justify-center space-x-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Analyzing...</span>
          </span>
        ) : (
          `Run ${mode.charAt(0).toUpperCase() + mode.slice(1)} Analysis`
        )}
      </button>
    </div>
  );
}

export default HistoricalAnalysisPanel;
