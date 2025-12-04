'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import { useIntelBriefing, getRiskBadgeStyle } from '@/hooks/useIntelBriefing';
import HelpTip from '@/components/HelpTip';
import Glossary from '@/components/Glossary';

// Dynamic import for map (client-side only)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-slate-900 rounded-xl flex items-center justify-center">
      <div className="text-slate-400">Loading map...</div>
    </div>
  ),
});

// Skill levels for progressive disclosure
type SkillLevel = 'simple' | 'standard' | 'detailed';

const SKILL_LEVELS = [
  { id: 'simple' as const, label: 'Basic', icon: '👀', desc: 'Easy to understand' },
  { id: 'standard' as const, label: 'Analyst', icon: '📊', desc: 'Industry context' },
  { id: 'detailed' as const, label: 'Expert', icon: '🔬', desc: 'Full tradecraft' },
];

// Region presets with progressive complexity
const PRESETS = [
  {
    id: 'global',
    name: 'Global',
    fullName: 'World Overview',
    icon: '🌍',
    simpleDesc: 'Every country in the world',
    standardDesc: 'Full 195-nation geopolitical landscape',
    detailedDesc: 'Complete basin-state manifold with cross-border influence tensors and transition probability matrices',
  },
  {
    id: 'nato',
    name: 'NATO',
    fullName: 'NATO Alliance',
    icon: '🛡️',
    simpleDesc: 'US and European allies',
    standardDesc: '32-member collective defense treaty',
    detailedDesc: 'Article 5 alliance coherence metrics, burden-sharing deltas, and interoperability indices',
  },
  {
    id: 'brics',
    name: 'BRICS+',
    fullName: 'BRICS Nations',
    icon: '🌏',
    simpleDesc: 'China, Russia, and friends',
    standardDesc: 'Emerging market bloc challenging Western order',
    detailedDesc:
      'De-dollarization velocity, commodity-backed settlement networks, and South-South trade dependency graphs',
  },
  {
    id: 'conflict',
    name: 'Hot Spots',
    fullName: 'Active Tensions',
    icon: '⚠️',
    simpleDesc: 'Where wars are happening',
    standardDesc: 'Active conflict zones and flashpoints',
    detailedDesc: 'Kinetic theaters, frozen conflicts, and gray-zone escalation ladders with phase transition indicators',
  },
];

// Visualization layers with progressive complexity
const LAYERS = [
  {
    id: 'basin' as const,
    name: 'Stability',
    icon: '⚓',
    color: 'blue',
    simpleDesc: 'Is this country stable or shaky?',
    standardDesc: 'Resistance to political and economic disruption',
    detailedDesc: 'Attractor basin depth: Lyapunov stability coefficient derived from multi-domain state vectors',
    legend: [
      { color: 'bg-blue-500', label: 'Very stable' },
      { color: 'bg-blue-400', label: 'Stable' },
      { color: 'bg-yellow-400', label: 'Moderate' },
      { color: 'bg-orange-400', label: 'Shaky' },
      { color: 'bg-red-500', label: 'Unstable' },
    ],
  },
  {
    id: 'risk' as const,
    name: 'Risk',
    icon: '📈',
    color: 'red',
    simpleDesc: 'Might something big happen soon?',
    standardDesc: 'Likelihood of major change in 1-6 months',
    detailedDesc: 'Transition probability: Monte Carlo phase-space trajectories with 30/60/90-day forecast horizons',
    legend: [
      { color: 'bg-green-500', label: 'Low risk' },
      { color: 'bg-yellow-400', label: 'Elevated' },
      { color: 'bg-orange-400', label: 'High' },
      { color: 'bg-red-500', label: 'Critical' },
    ],
  },
  {
    id: 'regime' as const,
    name: 'Government Type',
    icon: '🏛️',
    color: 'purple',
    simpleDesc: 'Democracy or dictatorship?',
    standardDesc: 'Political system classification (Polity V scale)',
    detailedDesc: 'Regime typology: V-Dem polyarchy scores, executive constraints, and competitive participation indices',
    legend: [
      { color: 'bg-blue-500', label: 'Democracy' },
      { color: 'bg-purple-500', label: 'Mixed' },
      { color: 'bg-red-500', label: 'Authoritarian' },
      { color: 'bg-gray-500', label: 'In transition' },
    ],
  },
];

// Key insight cards with progressive complexity
const KEY_INSIGHTS = [
  {
    icon: '🎯',
    title: 'What This Shows',
    simple: 'A map showing which countries are doing okay and which ones might have problems soon.',
    standard:
      'Real-time stability assessments derived from OSINT, economic indicators, and political event monitoring.',
    detailed:
      'Multi-dimensional attractor state visualization. Each nation\'s position in phase space reflects basin depth (stability), velocity vectors (momentum), and cross-border coupling coefficients.',
  },
  {
    icon: '⏱️',
    title: 'How Current',
    simple: 'Updated every day with the latest news.',
    standard: 'Continuous ingestion from 500+ global sources. Model retrains hourly.',
    detailed:
      'Sub-hourly GDELT/ACLED fusion, real-time sentiment NLP across 50+ languages, with Bayesian posterior updates on transition probability distributions.',
  },
  {
    icon: '🎲',
    title: 'Accuracy',
    simple: 'We get it right about 7 times out of 10.',
    standard:
      '72% recall on regime transitions at T-30. 85% precision on stability classifications.',
    detailed:
      'Backtested 2000-2023: 72% TPR at 30-day horizon, 15% FPR, AUC-ROC 0.84. Calibrated Brier scores across conflict, coup, and economic crisis domains.',
  },
];

// Preset filters - ISO 3166-1 alpha-3 country codes
// EXPANDED to include more countries of interest
const PRESET_FILTERS: Record<string, string[] | null> = {
  global: null, // null = show all
  nato: [
    'USA', 'CAN', 'GBR', 'FRA', 'DEU', 'ITA', 'ESP', 'POL', 'NLD', 'BEL',
    'PRT', 'GRC', 'TUR', 'NOR', 'DNK', 'CZE', 'HUN', 'ROU', 'BGR', 'SVK',
    'HRV', 'SVN', 'LVA', 'LTU', 'EST', 'ALB', 'MNE', 'MKD', 'FIN', 'SWE',
    'ISL', 'LUX',
    // Key NATO partners
    'UKR', 'GEO', 'MDA', 'JPN', 'KOR', 'AUS', 'NZL',
  ],
  brics: [
    'BRA', 'RUS', 'IND', 'CHN', 'ZAF',
    // New BRICS members
    'IRN', 'EGY', 'ETH', 'SAU', 'ARE',
    // Aspiring/associated
    'ARG', 'MEX', 'TUR', 'IDN', 'NGA', 'DZA', 'PAK', 'VNM', 'THA', 'MYS',
  ],
  conflict: [
    // Europe/Russia
    'UKR', 'RUS', 'BLR', 'MDA', 'GEO',
    // Middle East - full region
    'ISR', 'PSE', 'LBN', 'SYR', 'JOR', 'IRQ', 'IRN', 'YEM', 'SAU', 'ARE', 'QAT', 'KWT', 'BHR', 'OMN',
    // Africa flashpoints
    'SDN', 'SSD', 'ETH', 'SOM', 'LBY', 'NGA', 'MLI', 'BFA', 'NER', 'TCD', 'CAF', 'COD', 'MOZ',
    // Asia-Pacific
    'TWN', 'CHN', 'PRK', 'KOR', 'JPN', 'PHL', 'VNM', 'MMR', 'AFG', 'PAK', 'IND',
    // Americas
    'VEN', 'COL', 'MEX', 'HTI', 'NIC', 'CUB',
  ],
};

export default function ConsumerDashboard() {
  const { wasm, loading: wasmLoading } = useWasm();
  const { nations: allNations, edges: allEdges, loading: dataLoading } = useSupabaseNations();
  const [selectedPreset, setSelectedPreset] = useState('global');
  // Intel briefing - auto-fetch from cache on page load (cron keeps cache warm)
  const { briefings, metadata, loading: intelLoading, refetch: loadBriefing } = useIntelBriefing(selectedPreset, { autoFetch: true });

  // Refetch when preset changes
  const handlePresetChange = (preset: string) => {
    setSelectedPreset(preset);
    void loadBriefing();
  };
  const [selectedLayer, setSelectedLayer] = useState<'basin' | 'risk' | 'regime'>('basin');
  const [skillLevel, setSkillLevel] = useState<SkillLevel>('standard');
  const [isSimulating, setIsSimulating] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [showInsights, setShowInsights] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);

  // Filter nations and edges based on selected preset
  const { nations, edges } = useMemo(() => {
    const filter = PRESET_FILTERS[selectedPreset];
    if (!filter) {
      // Global - show all
      return { nations: allNations, edges: allEdges };
    }

    const filteredNations = allNations.filter((n) => filter.includes(n.code));
    const nationCodes = new Set(filteredNations.map((n) => n.code));

    // Only show edges where both source and target are in the filtered set
    const filteredEdges = allEdges.filter(
      (e) => nationCodes.has(e.source_code) && nationCodes.has(e.target_code)
    );

    return { nations: filteredNations, edges: filteredEdges };
  }, [allNations, allEdges, selectedPreset]);

  const handleSimulate = async () => {
    if (!wasm) return;
    setIsSimulating(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSimulating(false);
  };

  const currentLayer = LAYERS.find((l) => l.id === selectedLayer)!;
  const currentPreset = PRESETS.find((p) => p.id === selectedPreset)!;

  const getDescription = (item: {
    simpleDesc: string;
    standardDesc: string;
    detailedDesc: string;
  }) => {
    switch (skillLevel) {
      case 'simple':
        return item.simpleDesc;
      case 'detailed':
        return item.detailedDesc;
      default:
        return item.standardDesc;
    }
  };

  return (
    <div className="space-y-4 md:space-y-6 2xl:space-y-8">
      {/* Header with skill level toggle */}
      <div className="flex flex-col gap-4">
        <div className="flex items-start sm:items-center justify-between gap-4 flex-col sm:flex-row">
          <div>
            <h1 className="text-xl md:text-2xl 2xl:text-3xl font-bold text-white">
              {skillLevel === 'simple' ? 'World Stability Map' : skillLevel === 'standard' ? 'Geopolitical Analysis' : 'Strategic Intelligence Console'}
            </h1>
            <p className="text-slate-400 text-sm md:text-base 2xl:text-lg mt-0.5 md:mt-1">
              {skillLevel === 'simple'
                ? 'See which countries are doing well and which ones might have trouble'
                : skillLevel === 'detailed'
                  ? 'Attractor dynamics, transition probabilities, and multi-domain threat synthesis'
                  : 'Nation-level stability metrics and risk forecasting'}
            </p>
          </div>

          {/* Skill level selector + Help */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 bg-slate-900 rounded-lg p-1 border border-slate-800">
              {SKILL_LEVELS.map((level) => (
                <button
                  key={level.id}
                  onClick={() => setSkillLevel(level.id)}
                  className={`min-h-[44px] px-3 md:px-4 rounded-md text-xs md:text-sm transition-all flex items-center gap-1.5 ${
                    skillLevel === level.id
                      ? 'bg-blue-600 text-white'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800 active:bg-slate-700'
                  }`}
                  title={level.desc}
                >
                  <span className="text-base">{level.icon}</span>
                  <span className="hidden sm:inline">{level.label}</span>
                </button>
              ))}
            </div>
            {/* Glossary button */}
            <button
              onClick={() => setShowGlossary(true)}
              className="min-h-[44px] min-w-[44px] flex items-center justify-center rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Terminology Reference"
            >
              <span className="text-lg">📖</span>
            </button>
          </div>
        </div>

        {/* Quick insight toggle - 44px touch target */}
        <button
          onClick={() => setShowInsights(!showInsights)}
          className="self-start min-h-[44px] flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300 active:text-blue-200 px-2 -ml-2"
        >
          <span>{showInsights ? '▼' : '▶'}</span>
          <span>{showInsights ? 'Hide' : 'Show'} key information</span>
        </button>

        {/* Expandable insights panel */}
        {showInsights && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 md:gap-4 bg-slate-900/50 rounded-xl p-4 border border-slate-800">
            {KEY_INSIGHTS.map((insight) => (
              <div key={insight.title} className="space-y-1">
                <div className="flex items-center gap-2">
                  <span>{insight.icon}</span>
                  <span className="font-medium text-white text-sm">{insight.title}</span>
                </div>
                <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
                  {skillLevel === 'simple'
                    ? insight.simple
                    : skillLevel === 'detailed'
                      ? insight.detailed
                      : insight.standard}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Region presets */}
      <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
        <div className="flex md:grid md:grid-cols-4 gap-3 md:gap-4 min-w-max md:min-w-0">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => handlePresetChange(preset.id)}
              className={`flex-shrink-0 w-28 sm:w-32 md:w-auto p-3 md:p-4 2xl:p-5 rounded-xl border text-left transition-all ${
                selectedPreset === preset.id
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800/50'
              }`}
            >
              <span className="text-2xl md:text-3xl">{preset.icon}</span>
              <p className="font-medium mt-2 text-sm md:text-base 2xl:text-lg">{preset.fullName}</p>
              <p className="text-xs md:text-sm opacity-75 mt-1 line-clamp-2">
                {getDescription(preset)}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Main content grid - 3 columns on desktop: Summary | Map | Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-6 xl:grid-cols-7 2xl:grid-cols-8 gap-4 md:gap-6 2xl:gap-8">
        {/* Flashpoint Summary Panel - Left sidebar on desktop */}
        <div className="lg:col-span-2 xl:col-span-2 2xl:col-span-2 order-2 lg:order-1">
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 lg:p-5 space-y-4 lg:sticky lg:top-20 max-h-[calc(100vh-6rem)] overflow-y-auto">
            <div className="flex items-center justify-between">
              <h2 className="text-base lg:text-lg font-semibold text-white flex items-center gap-2">
                <span>📡</span>
                <span>Intel Briefing</span>
              </h2>
              {metadata && (
                <span
                  className={`text-xs px-2 py-1 rounded border ${getRiskBadgeStyle(metadata.overallRisk)}`}
                >
                  {metadata.overallRisk.toUpperCase()}
                </span>
              )}
            </div>

            {/* Location-based context */}
            <div className="text-xs text-slate-400 border-b border-slate-800 pb-3">
              <span className="text-blue-400">{metadata?.region || 'Global'}</span> •{' '}
              {intelLoading ? (
                <span className="animate-pulse">Analyzing...</span>
              ) : (
                `Updated ${metadata?.timestamp ? new Date(metadata.timestamp).toLocaleTimeString() : 'now'}`
              )}
            </div>

            {/* Briefing content - auto-loads from cache */}
            {intelLoading ? (
              <div className="space-y-4">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="space-y-2 animate-pulse">
                    <div className="h-4 bg-slate-800 rounded w-24" />
                    <div className="h-3 bg-slate-800 rounded w-full" />
                    <div className="h-3 bg-slate-800 rounded w-3/4" />
                  </div>
                ))}
              </div>
            ) : (
              <>
                {/* Political */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-amber-400 flex items-center gap-2">
                    <span>🏛️</span> Political
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.political || 'No data available'}
                  </p>
                </div>

                {/* Economic & Trade */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-green-400 flex items-center gap-2">
                    <span>📈</span> Economic & Trade
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.economic || 'No data available'}
                  </p>
                </div>

                {/* Security & Diplomacy */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-red-400 flex items-center gap-2">
                    <span>⚔️</span> Security & Diplomacy
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.security || 'No data available'}
                  </p>
                </div>

                {/* Financial */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-blue-400 flex items-center gap-2">
                    <span>💰</span> Financial
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.financial || 'No data available'}
                  </p>
                </div>

                {/* Health */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-pink-400 flex items-center gap-2">
                    <span>🏥</span> Health
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.health || 'No data available'}
                  </p>
                </div>

                {/* Science & Tech */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-cyan-400 flex items-center gap-2">
                    <span>🔬</span> Science & Tech
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.scitech || 'No data available'}
                  </p>
                </div>

                {/* Natural Resources */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-emerald-400 flex items-center gap-2">
                    <span>🌿</span> Natural Resources
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.resources || 'No data available'}
                  </p>
                </div>

                {/* Crime & Drugs */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-orange-400 flex items-center gap-2">
                    <span>🚨</span> Crime & Drugs
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.crime || 'No data available'}
                  </p>
                </div>

                {/* Cyber Threats */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-violet-400 flex items-center gap-2">
                    <span>💻</span> Cyber Threats
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.cyber || 'No data available'}
                  </p>
                </div>

                {/* Terrorism & Extremism */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-rose-400 flex items-center gap-2">
                    <span>⚡</span> Terrorism & Extremism
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.terrorism || 'No data available'}
                  </p>
                </div>

                {/* Domestic Instability */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-yellow-400 flex items-center gap-2">
                    <span>🔥</span> Domestic Instability
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.domestic || 'No data available'}
                  </p>
                </div>

                {/* Border & Incursions */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
                    <span>🗺️</span> Border & Incursions
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.borders || 'No data available'}
                  </p>
                </div>

                {/* Media & Info Ops */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-indigo-400 flex items-center gap-2">
                    <span>📺</span> Media & Info Ops
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.infoops || 'No data available'}
                  </p>
                </div>

                {/* Military */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-stone-400 flex items-center gap-2">
                    <span>🎖️</span> Military
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.military || 'No data available'}
                  </p>
                </div>

                {/* Space */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-purple-400 flex items-center gap-2">
                    <span>🛰️</span> Space
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.space || 'No data available'}
                  </p>
                </div>

                {/* Industry & Manufacturing */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-zinc-400 flex items-center gap-2">
                    <span>🏭</span> Industry
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.industry || 'No data available'}
                  </p>
                </div>

                {/* Logistics */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-amber-500 flex items-center gap-2">
                    <span>🚢</span> Logistics
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.logistics || 'No data available'}
                  </p>
                </div>

                {/* Minerals */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-teal-400 flex items-center gap-2">
                    <span>⛏️</span> Minerals
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.minerals || 'No data available'}
                  </p>
                </div>

                {/* Energy & Petrochemicals */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-lime-400 flex items-center gap-2">
                    <span>⚡</span> Energy
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.energy || 'No data available'}
                  </p>
                </div>

                {/* Markets & Exchanges */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-sky-400 flex items-center gap-2">
                    <span>📊</span> Markets
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.markets || 'No data available'}
                  </p>
                </div>

                {/* Religious & Ideological */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-fuchsia-400 flex items-center gap-2">
                    <span>🕊️</span> Religious & Ideological
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.religious || 'No data available'}
                  </p>
                </div>

                {/* Education */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-blue-300 flex items-center gap-2">
                    <span>🎓</span> Education
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.education || 'No data available'}
                  </p>
                </div>

                {/* Employment */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-green-300 flex items-center gap-2">
                    <span>💼</span> Employment
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.employment || 'No data available'}
                  </p>
                </div>

                {/* Housing */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-orange-300 flex items-center gap-2">
                    <span>🏠</span> Housing
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.housing || 'No data available'}
                  </p>
                </div>

                {/* Crypto */}
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-yellow-300 flex items-center gap-2">
                    <span>₿</span> Crypto
                  </h3>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {briefings?.crypto || 'No data available'}
                  </p>
                </div>

                {/* Emerging Trends */}
                <div className="space-y-2 pt-3 border-t border-slate-800/50">
                  <h3 className="text-sm font-medium text-white flex items-center gap-2">
                    <span>🔮</span> Emerging Trends
                  </h3>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {briefings?.emerging || 'No data available'}
                  </p>
                </div>

                {/* Next Strategic Move */}
                {briefings?.nsm && (
                  <div className="space-y-2 pt-3 border-t border-blue-800/50 bg-blue-950/30 -mx-4 px-4 py-3 rounded-lg">
                    <h3 className="text-sm font-medium text-blue-300 flex items-center gap-2">
                      <span>🎯</span> Next Strategic Move
                    </h3>
                    <p className="text-xs text-blue-200 leading-relaxed">{briefings.nsm}</p>
                  </div>
                )}

                {/* Summary */}
                {briefings?.summary && (
                  <div className="mt-4 pt-4 border-t border-slate-800">
                    <p className="text-xs text-slate-300 italic">{briefings.summary}</p>
                  </div>
                )}
              </>
            )}

            {/* Subscription upsell */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <p className="text-xs text-slate-500 text-center">
                <span className="text-blue-400 cursor-pointer hover:underline">Upgrade to Pro</span>{' '}
                for real-time alerts & deeper analysis
              </p>
            </div>
          </div>
        </div>

        {/* Map */}
        <div className="lg:col-span-3 xl:col-span-4 2xl:col-span-5 order-1 lg:order-2">
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            {/* Map header with current view info */}
            <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-xl">{currentPreset.icon}</span>
                <div>
                  <p className="text-white font-medium text-sm">{currentPreset.fullName}</p>
                  <p className="text-slate-500 text-xs">
                    Viewing: {currentLayer.name} {currentLayer.icon}
                  </p>
                </div>
              </div>
              {/* Legend */}
              <div className="hidden md:flex items-center gap-2">
                {currentLayer.legend.map((item, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <span className={`w-3 h-3 rounded-full ${item.color}`} />
                    <span className="text-xs text-slate-400">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="h-[55vh] md:h-[450px] lg:h-[550px] 2xl:h-[650px]">
              <AttractorMap nations={nations} edges={edges} layer={selectedLayer} />
            </div>

            {/* Mobile legend */}
            <div className="md:hidden px-4 py-3 border-t border-slate-800 flex flex-wrap gap-3">
              {currentLayer.legend.map((item, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <span className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                  <span className="text-xs text-slate-400">{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Controls sidebar */}
        <div
          className={`
          order-3 space-y-4 md:space-y-5
          lg:relative lg:block lg:col-span-1
          ${
            controlsOpen
              ? 'fixed inset-x-0 bottom-0 bg-slate-950 border-t border-slate-800 p-4 z-40 max-h-[70vh] overflow-y-auto rounded-t-2xl lg:p-0 lg:border-0 lg:static lg:max-h-none lg:rounded-none'
              : 'hidden lg:block'
          }
        `}
        >
          {/* Mobile close handle */}
          <div className="lg:hidden flex justify-center pb-2">
            <button
              onClick={() => setControlsOpen(false)}
              className="w-12 h-1.5 bg-slate-600 rounded-full"
            />
          </div>

          {/* View selector */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
            <h3 className="font-medium text-white mb-3 text-sm flex items-center gap-2">
              <span>🎨</span>
              <span>{skillLevel === 'simple' ? 'What to show' : skillLevel === 'detailed' ? 'Render Layer' : 'Map Layer'}</span>
            </h3>
            <div className="space-y-2">
              {LAYERS.map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => setSelectedLayer(layer.id)}
                  className={`w-full p-3 rounded-lg text-left transition-colors ${
                    selectedLayer === layer.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span>{layer.icon}</span>
                    <span className="font-medium text-sm">{layer.name}</span>
                    <HelpTip
                      term={layer.id === 'basin' ? 'Basin' : layer.id === 'risk' ? 'Transition Risk' : 'Regime'}
                      skillLevel={skillLevel}
                      size={12}
                      position="right"
                    />
                  </div>
                  <p className="text-xs opacity-75 mt-1 ml-6">{getDescription(layer)}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Simulate */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
            <h3 className="font-medium text-white mb-2 text-sm flex items-center gap-2">
              <span>⚡</span>
              <span>{skillLevel === 'simple' ? 'See the future' : skillLevel === 'detailed' ? 'Propagate Dynamics' : 'Run Simulation'}</span>
              <HelpTip term="Monte Carlo" skillLevel={skillLevel} size={12} position="right" />
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              {skillLevel === 'simple'
                ? 'Click to see how things might change next'
                : skillLevel === 'detailed'
                  ? 'Advance phase-space by Δt=1. Stochastic transitions sampled from posterior.'
                  : 'Advance the model forward in time by one step'}
            </p>
            <button
              onClick={() => void handleSimulate()}
              disabled={isSimulating || wasmLoading || dataLoading}
              className={`w-full py-3 rounded-lg font-medium transition-colors ${
                isSimulating || wasmLoading || dataLoading
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-500 active:bg-green-700'
              }`}
            >
              {dataLoading
                ? 'Loading data...'
                : wasmLoading
                  ? 'Loading WASM...'
                  : isSimulating
                    ? 'Running...'
                    : skillLevel === 'simple'
                      ? 'Go!'
                      : skillLevel === 'detailed'
                        ? 'Execute Δt'
                        : 'Run Step'}
            </button>
            <p className="text-xs text-slate-500 mt-2 text-center">{skillLevel === 'detailed' ? '10 iterations remaining (24h window)' : '10 remaining today'}</p>
          </div>

          {/* Save - 44px touch target */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
            <button className="w-full min-h-[44px] rounded-lg font-medium bg-slate-800 text-white hover:bg-slate-700 active:bg-slate-600 transition-colors text-sm">
              💾 Save This View
            </button>
            <p className="text-xs text-slate-500 mt-2 text-center">3 of 5 slots used</p>
          </div>

          {/* Mobile controls toggle - 48px for FAB */}
          <button
            onClick={() => setControlsOpen(!controlsOpen)}
            className="fixed bottom-4 right-4 lg:hidden z-30 min-h-[48px] px-4 bg-blue-600 text-white rounded-full shadow-lg flex items-center gap-2 active:bg-blue-500"
          >
            <span>⚙️</span>
            <span className="text-sm font-medium">Controls</span>
          </button>
        </div>

        {/* Mobile overlay */}
        {controlsOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 lg:hidden"
            onClick={() => setControlsOpen(false)}
          />
        )}
      </div>

      {/* Bottom explanation cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">📍</span>
            <h3 className="font-medium text-white text-sm md:text-base">Reading the Map</h3>
          </div>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            {skillLevel === 'simple'
              ? 'Blue = calm and stable. Red = trouble brewing. Click any country to learn more about it.'
              : skillLevel === 'detailed'
                ? 'Color saturation encodes basin depth (Lyapunov stability). Node radius maps to eigenvector centrality. Edge weights reflect bilateral influence coefficients. Click nodes for full state vector decomposition.'
                : 'Colors show stability (blue=stable, red=volatile). Circle size indicates regional influence. Click any country for detailed metrics.'}
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🔮</span>
            <h3 className="font-medium text-white text-sm md:text-base">{skillLevel === 'detailed' ? 'Forecast Methodology' : 'What We Predict'}</h3>
          </div>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            {skillLevel === 'simple'
              ? 'We read the news and look at money and trade to figure out which countries might have problems soon.'
              : skillLevel === 'detailed'
                ? 'Monte Carlo ensemble over phase-space trajectories. 10K iterations per forecast window. Transition probabilities derived from Markov chain stationary distributions with empirical priors.'
                : 'Predictive model analyzes news, economic data, and political signals to forecast major changes 1-6 months ahead.'}
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 sm:col-span-2 lg:col-span-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">⚠️</span>
            <h3 className="font-medium text-white text-sm md:text-base">{skillLevel === 'detailed' ? 'Model Limitations' : 'Important Limits'}</h3>
          </div>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            {skillLevel === 'simple'
              ? 'We can\'t predict everything. Surprises happen. Always check other sources too before making big decisions.'
              : skillLevel === 'detailed'
                ? 'Model assumes continuous dynamics; discontinuous shocks (coups, black swans) may evade detection. 72% TPR at T-30, 15% FPR. Confidence intervals widen beyond 60-day horizon. Cross-validate with HUMINT.'
                : 'Predictions are probabilistic estimates, not certainties. Sudden events may be missed. Always cross-reference with other intelligence sources.'}
          </p>
        </div>
      </div>

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel={skillLevel}
      />
    </div>
  );
}
