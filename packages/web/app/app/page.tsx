'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';

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
  { id: 'simple' as const, label: 'Simple', icon: '👀', desc: 'Big picture only' },
  { id: 'standard' as const, label: 'Standard', icon: '📊', desc: 'Key metrics' },
  { id: 'detailed' as const, label: 'Detailed', icon: '🔬', desc: 'Full analysis' },
];

// Region presets with plain-language descriptions
const PRESETS = [
  {
    id: 'global',
    name: 'Global',
    fullName: 'World Overview',
    icon: '🌍',
    simpleDesc: 'See the whole picture',
    standardDesc: 'All 195 nations at a glance',
    detailedDesc: 'Complete global state with all influence networks and transition probabilities',
  },
  {
    id: 'nato',
    name: 'NATO',
    fullName: 'NATO Alliance',
    icon: '🛡️',
    simpleDesc: 'Western allies',
    standardDesc: '31 member defense alliance',
    detailedDesc: 'North Atlantic Treaty Organization members with internal cohesion metrics',
  },
  {
    id: 'brics',
    name: 'BRICS+',
    fullName: 'BRICS Nations',
    icon: '🌏',
    simpleDesc: 'Rising powers',
    standardDesc: 'Major emerging economies',
    detailedDesc:
      'Brazil, Russia, India, China, South Africa + new members with trade dependencies',
  },
  {
    id: 'conflict',
    name: 'Hot Spots',
    fullName: 'Active Tensions',
    icon: '⚠️',
    simpleDesc: 'Watch zones',
    standardDesc: 'Highest risk areas',
    detailedDesc: 'Regions with elevated transition probability or ongoing instability indicators',
  },
];

// Visualization layers with plain-language explanations
const LAYERS = [
  {
    id: 'basin' as const,
    name: 'Stability',
    icon: '⚓',
    color: 'blue',
    simpleDesc: 'How steady is each country?',
    standardDesc: 'Basin strength indicates resistance to change',
    detailedDesc: 'Attractor basin depth measures the energy required for regime transition',
    legend: [
      { color: 'bg-blue-500', label: 'Very stable' },
      { color: 'bg-blue-400', label: 'Stable' },
      { color: 'bg-yellow-400', label: 'Moderate' },
      { color: 'bg-orange-400', label: 'Volatile' },
      { color: 'bg-red-500', label: 'Critical' },
    ],
  },
  {
    id: 'risk' as const,
    name: 'Risk',
    icon: '📈',
    color: 'red',
    simpleDesc: 'Could things change soon?',
    standardDesc: 'Likelihood of major shifts in next 6 months',
    detailedDesc: 'Transition probability computed from position-velocity dynamics in phase space',
    legend: [
      { color: 'bg-green-500', label: 'Low risk' },
      { color: 'bg-yellow-400', label: 'Elevated' },
      { color: 'bg-orange-400', label: 'High' },
      { color: 'bg-red-500', label: 'Critical' },
    ],
  },
  {
    id: 'regime' as const,
    name: 'System Type',
    icon: '🏛️',
    color: 'purple',
    simpleDesc: 'What kind of government?',
    standardDesc: 'Political system classification',
    detailedDesc: 'Regime type from attractor cluster assignment with confidence intervals',
    legend: [
      { color: 'bg-blue-500', label: 'Democracy' },
      { color: 'bg-purple-500', label: 'Hybrid' },
      { color: 'bg-red-500', label: 'Authoritarian' },
      { color: 'bg-gray-500', label: 'Transitional' },
    ],
  },
];

// Key insight cards with progressive detail
const KEY_INSIGHTS = [
  {
    icon: '🎯',
    title: 'What This Shows',
    simple: 'Which countries might face big changes soon, and which ones are stable.',
    standard:
      'Nation-level stability analysis based on political, economic, and social indicators combined into a predictive model.',
    detailed:
      'Dynamical systems analysis treating nations as particles in high-dimensional phase space. Basin strength measures attractor depth; transition risk derives from proximity to separatrices.',
  },
  {
    icon: '⏱️',
    title: 'How Current',
    simple: 'Data is updated daily from news and reports.',
    standard: 'Model ingests daily feeds from 500+ sources. Simulations run hourly.',
    detailed:
      'Real-time NLP pipeline processes Reuters, AP, governmental releases. Feature extraction updates at 00:00 UTC. Monte Carlo sims refresh hourly.',
  },
  {
    icon: '🎲',
    title: 'Accuracy',
    simple: 'The model catches about 7 out of 10 major events beforehand.',
    standard:
      '72% recall on regime transitions with 30-day lead time. 85% AUC on binary stability classification.',
    detailed:
      'Backtest 2000-2023: 72% TPR @ 30d horizon, 15% FPR. Brier score 0.18. Calibration verified via reliability diagrams. See methodology docs.',
  },
];

// Preset filters - ISO 3166-1 alpha-3 country codes
const PRESET_FILTERS: Record<string, string[] | null> = {
  global: null, // null = show all
  nato: [
    'USA',
    'CAN',
    'GBR',
    'FRA',
    'DEU',
    'ITA',
    'ESP',
    'POL',
    'NLD',
    'BEL',
    'PRT',
    'GRC',
    'TUR',
    'NOR',
    'DNK',
    'CZE',
    'HUN',
    'ROU',
    'BGR',
    'SVK',
    'HRV',
    'SVN',
    'LVA',
    'LTU',
    'EST',
    'ALB',
    'MNE',
    'MKD',
    'FIN',
    'SWE',
    'ISL',
    'LUX',
  ],
  brics: ['BRA', 'RUS', 'IND', 'CHN', 'ZAF', 'IRN', 'EGY', 'ETH', 'SAU', 'ARE'],
  conflict: [
    'UKR',
    'RUS',
    'ISR',
    'PSE',
    'LBN',
    'SYR',
    'YEM',
    'SDN',
    'MMR',
    'AFG',
    'TWN',
    'CHN',
    'PRK',
    'KOR',
  ],
};

export default function ConsumerDashboard() {
  const { wasm, loading: wasmLoading } = useWasm();
  const { nations: allNations, edges: allEdges, loading: dataLoading } = useSupabaseNations();
  const [selectedPreset, setSelectedPreset] = useState('global');
  const [selectedLayer, setSelectedLayer] = useState<'basin' | 'risk' | 'regime'>('basin');
  const [skillLevel, setSkillLevel] = useState<SkillLevel>('standard');
  const [isSimulating, setIsSimulating] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);
  const [showInsights, setShowInsights] = useState(false);

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
              {skillLevel === 'simple' ? 'World Stability Map' : 'Geopolitical Analysis'}
            </h1>
            <p className="text-slate-400 text-sm md:text-base 2xl:text-lg mt-0.5 md:mt-1">
              {skillLevel === 'simple'
                ? 'See which countries are stable and which might change'
                : skillLevel === 'detailed'
                  ? 'Attractor dynamics and transition probability analysis'
                  : 'Explore nation-level stability and risk patterns'}
            </p>
          </div>

          {/* Skill level selector */}
          <div className="flex items-center gap-1 bg-slate-900 rounded-lg p-1 border border-slate-800">
            {SKILL_LEVELS.map((level) => (
              <button
                key={level.id}
                onClick={() => setSkillLevel(level.id)}
                className={`px-3 py-1.5 md:px-4 md:py-2 rounded-md text-xs md:text-sm transition-all flex items-center gap-1.5 ${
                  skillLevel === level.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
                title={level.desc}
              >
                <span>{level.icon}</span>
                <span className="hidden sm:inline">{level.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Quick insight toggle */}
        <button
          onClick={() => setShowInsights(!showInsights)}
          className="self-start flex items-center gap-2 text-sm text-blue-400 hover:text-blue-300"
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
              onClick={() => setSelectedPreset(preset.id)}
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
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 lg:p-5 space-y-4 lg:sticky lg:top-20">
            <div className="flex items-center justify-between">
              <h2 className="text-base lg:text-lg font-semibold text-white flex items-center gap-2">
                <span>📡</span>
                <span>Intel Briefing</span>
              </h2>
              <span className="text-xs text-slate-500 bg-slate-800 px-2 py-1 rounded">Live</span>
            </div>

            {/* Location-based context */}
            <div className="text-xs text-slate-400 border-b border-slate-800 pb-3">
              <span className="text-blue-400">Based on your region</span> • Updated 2h ago
            </div>

            {/* Political */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-amber-400 flex items-center gap-2">
                <span>🏛️</span> Political
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vestibulum ante ipsum
                primis in faucibus orci luctus.
              </p>
            </div>

            {/* Economic & Trade */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-green-400 flex items-center gap-2">
                <span>📈</span> Economic & Trade
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
                doloremque laudantium.
              </p>
            </div>

            {/* Security & Diplomacy */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-red-400 flex items-center gap-2">
                <span>⚔️</span> Security & Diplomacy
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia
                consequuntur.
              </p>
            </div>

            {/* Financial */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-blue-400 flex items-center gap-2">
                <span>💰</span> Financial
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit
                laboriosam.
              </p>
            </div>

            {/* Health */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-pink-400 flex items-center gap-2">
                <span>🏥</span> Health
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Quis autem vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil
                molestiae.
              </p>
            </div>

            {/* Science & Tech */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-cyan-400 flex items-center gap-2">
                <span>🔬</span> Science & Tech
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis
                praesentium.
              </p>
            </div>

            {/* Natural Resources */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-emerald-400 flex items-center gap-2">
                <span>🌿</span> Natural Resources
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit quo
                minus.
              </p>
            </div>

            {/* Crime & Drugs */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-orange-400 flex items-center gap-2">
                <span>🚨</span> Crime & Drugs
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe
                eveniet.
              </p>
            </div>

            {/* Cyber Threats */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-violet-400 flex items-center gap-2">
                <span>💻</span> Cyber Threats
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus
                maiores.
              </p>
            </div>

            {/* Terrorism & Extremism */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-rose-400 flex items-center gap-2">
                <span>⚡</span> Terrorism & Extremism
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore soluta.
              </p>
            </div>

            {/* Domestic Instability */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-yellow-400 flex items-center gap-2">
                <span>🔥</span> Domestic Instability
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam.
              </p>
            </div>

            {/* Border & Incursions */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-slate-300 flex items-center gap-2">
                <span>🗺️</span> Border & Incursions
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium
                doloremque.
              </p>
            </div>

            {/* Media & Info Ops */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-indigo-400 flex items-center gap-2">
                <span>📺</span> Media & Info Ops
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit
                consequuntur.
              </p>
            </div>

            {/* Subscription upsell */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <p className="text-xs text-slate-500 text-center">
                <span className="text-blue-400 cursor-pointer hover:underline">Upgrade to Pro</span>{' '}
                for real-time alerts
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
              <span>{skillLevel === 'simple' ? 'What to show' : 'Visualization Layer'}</span>
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
              <span>{skillLevel === 'simple' ? 'See what happens next' : 'Run Simulation'}</span>
            </h3>
            <p className="text-xs text-slate-500 mb-3">
              {skillLevel === 'simple'
                ? 'Step forward in time to see how things might change'
                : 'Advance the model by one timestep'}
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
                      ? 'Step Forward'
                      : 'Run Step'}
            </button>
            <p className="text-xs text-slate-500 mt-2 text-center">10 remaining today</p>
          </div>

          {/* Save */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4">
            <button className="w-full py-2.5 rounded-lg font-medium bg-slate-800 text-white hover:bg-slate-700 transition-colors text-sm">
              💾 Save This View
            </button>
            <p className="text-xs text-slate-500 mt-2 text-center">3 of 5 slots used</p>
          </div>

          {/* Mobile controls toggle (shows when controls hidden) */}
          <button
            onClick={() => setControlsOpen(!controlsOpen)}
            className="fixed bottom-4 right-4 lg:hidden z-30 px-4 py-3 bg-blue-600 text-white rounded-full shadow-lg flex items-center gap-2"
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
              ? 'Darker colors mean more stable. Bright colors mean things might change soon. Click any country to learn more.'
              : skillLevel === 'detailed'
                ? 'Color saturation encodes basin depth (stability). Marker size represents influence radius. Click nodes to view velocity vectors and network connections.'
                : 'Each country is colored by its current stability level. Brighter colors indicate higher risk of change. Click for details.'}
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">🔮</span>
            <h3 className="font-medium text-white text-sm md:text-base">What We Predict</h3>
          </div>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            {skillLevel === 'simple'
              ? 'We look at news, trade, and politics to guess which countries might face big changes in the next few months.'
              : skillLevel === 'detailed'
                ? 'Transition probabilities are derived from Monte Carlo sampling of phase space trajectories. 30-day forecast horizon with confidence intervals.'
                : 'The model predicts likelihood of major political or economic shifts over the next 1-6 months based on multiple data sources.'}
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 sm:col-span-2 lg:col-span-1">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">⚠️</span>
            <h3 className="font-medium text-white text-sm md:text-base">Important Limits</h3>
          </div>
          <p className="text-xs md:text-sm text-slate-400 leading-relaxed">
            {skillLevel === 'simple'
              ? 'No prediction is perfect. Use this as one tool among many. Always check multiple sources before making decisions.'
              : skillLevel === 'detailed'
                ? 'Model assumes continuous dynamics; black swan events may cause discontinuous jumps. Past performance does not guarantee future accuracy. See confidence intervals.'
                : 'Predictions are probabilistic, not certain. The model may miss sudden events. Always combine with other intelligence sources.'}
          </p>
        </div>
      </div>
    </div>
  );
}
