'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';

// Dynamic import for map (client-side only)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full bg-slate-900 rounded-xl flex items-center justify-center">
      <div className="text-slate-400">Loading map...</div>
    </div>
  ),
});

// Preset configurations for consumer
const PRESETS = [
  {
    id: 'global',
    name: 'Global',
    fullName: 'Global Overview',
    description: 'See all nation attractors',
    icon: '🌍',
  },
  {
    id: 'nato',
    name: 'NATO',
    fullName: 'NATO Alliance',
    description: 'Western democratic nations',
    icon: '🛡️',
  },
  {
    id: 'brics',
    name: 'BRICS',
    fullName: 'BRICS Nations',
    description: 'Emerging economies bloc',
    icon: '🌏',
  },
  {
    id: 'conflict',
    name: 'Conflict',
    fullName: 'Conflict Zones',
    description: 'High transition risk areas',
    icon: '⚠️',
  },
];

// Layer options for consumer (simplified)
const LAYERS = [
  { id: 'basin', name: 'Stability', description: 'How stable is each nation?' },
  { id: 'risk', name: 'Risk', description: 'Transition probability' },
  { id: 'regime', name: 'Regimes', description: 'Political categories' },
];

export default function ConsumerDashboard() {
  const { wasm, loading: wasmLoading } = useWasm();
  const [selectedPreset, setSelectedPreset] = useState('global');
  const [selectedLayer, setSelectedLayer] = useState<'basin' | 'risk' | 'regime'>('basin');
  const [isSimulating, setIsSimulating] = useState(false);
  const [controlsOpen, setControlsOpen] = useState(false);

  // Placeholder nation data (would come from Supabase in real app)
  const nations: never[] = [];
  const edges: never[] = [];

  const handleSimulate = async () => {
    if (!wasm) return;
    setIsSimulating(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSimulating(false);
  };

  return (
    <div className="space-y-4 md:space-y-6 2xl:space-y-8">
      {/* Header - responsive sizing */}
      <div className="flex items-start sm:items-center justify-between gap-4 flex-col sm:flex-row">
        <div>
          <h1 className="text-xl md:text-2xl 2xl:text-3xl font-bold text-white">Explore Attractors</h1>
          <p className="text-slate-400 text-sm md:text-base 2xl:text-lg mt-0.5 md:mt-1">
            <span className="hidden sm:inline">Visualize nation-level dynamics and stability patterns</span>
            <span className="sm:hidden">Nation dynamics visualization</span>
          </p>
        </div>
        {/* Mobile controls toggle */}
        <button
          onClick={() => setControlsOpen(!controlsOpen)}
          className="lg:hidden px-4 py-2.5 bg-slate-800 text-white rounded-lg text-sm flex items-center gap-2 shrink-0"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
          </svg>
          <span>Controls</span>
        </button>
      </div>

      {/* Preset selector - horizontal scroll on mobile, grid on larger screens */}
      <div className="overflow-x-auto -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
        <div className="flex md:grid md:grid-cols-4 2xl:grid-cols-4 gap-3 md:gap-4 min-w-max md:min-w-0">
          {PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setSelectedPreset(preset.id)}
              className={`flex-shrink-0 w-24 sm:w-28 md:w-auto p-3 md:p-4 2xl:p-5 rounded-xl border text-left transition-all ${
                selectedPreset === preset.id
                  ? 'bg-blue-600 border-blue-500 text-white'
                  : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800/50'
              }`}
            >
              <span className="text-xl md:text-2xl 2xl:text-3xl">{preset.icon}</span>
              <p className="font-medium mt-1.5 md:mt-2 text-sm md:text-base 2xl:text-lg">
                <span className="md:hidden">{preset.name}</span>
                <span className="hidden md:inline">{preset.fullName}</span>
              </p>
              <p className="text-xs md:text-sm 2xl:text-base opacity-75 mt-0.5 md:mt-1 hidden md:block">
                {preset.description}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Main content - responsive grid */}
      {/* Mobile: stacked, Tablet: 2-col, Desktop: 3+1, Ultrawide: 5+1 with larger map */}
      <div className="grid grid-cols-1 lg:grid-cols-4 2xl:grid-cols-6 gap-4 md:gap-6 2xl:gap-8">
        {/* Map - expands more on ultrawide */}
        <div className="lg:col-span-3 2xl:col-span-5 order-1">
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            {/* Responsive height: mobile 60vh, tablet 500px, desktop 600px, ultrawide 700px */}
            <div className="h-[60vh] md:h-[500px] lg:h-[600px] 2xl:h-[700px] 3xl:h-[800px]">
              <AttractorMap
                nations={nations}
                edges={edges}
                layer={selectedLayer}
              />
            </div>
          </div>
        </div>

        {/* Controls sidebar - slides up from bottom on mobile */}
        <div className={`
          order-2 space-y-4 md:space-y-6
          lg:relative lg:block lg:col-span-1
          ${controlsOpen
            ? 'fixed inset-x-0 bottom-0 bg-slate-950 border-t border-slate-800 p-4 z-40 max-h-[70vh] overflow-y-auto rounded-t-2xl lg:p-0 lg:border-0 lg:static lg:max-h-none lg:rounded-none'
            : 'hidden lg:block'
          }
        `}>
          {/* Close handle for mobile */}
          <div className="lg:hidden flex justify-center pb-2">
            <button
              onClick={() => setControlsOpen(false)}
              className="w-12 h-1.5 bg-slate-600 rounded-full"
              aria-label="Close controls"
            />
          </div>

          {/* Layer selector */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6">
            <h3 className="font-medium text-white mb-3 md:mb-4 text-sm md:text-base 2xl:text-lg">Visualization</h3>
            <div className="grid grid-cols-3 lg:grid-cols-1 gap-2">
              {LAYERS.map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => setSelectedLayer(layer.id as 'basin' | 'risk' | 'regime')}
                  className={`p-2.5 md:p-3 2xl:p-4 rounded-lg text-center lg:text-left transition-colors ${
                    selectedLayer === layer.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <p className="font-medium text-xs md:text-sm 2xl:text-base">{layer.name}</p>
                  <p className="text-xs 2xl:text-sm opacity-75 hidden lg:block mt-0.5">{layer.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex lg:flex-col gap-3 md:gap-4">
            {/* Simulate button */}
            <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6">
              <h3 className="font-medium text-white mb-2 md:mb-3 2xl:mb-4 text-sm md:text-base 2xl:text-lg hidden lg:block">
                Simulation
              </h3>
              <button
                onClick={handleSimulate}
                disabled={isSimulating || wasmLoading}
                className={`w-full py-2.5 md:py-3 2xl:py-4 rounded-lg font-medium transition-colors text-sm md:text-base 2xl:text-lg ${
                  isSimulating || wasmLoading
                    ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                    : 'bg-green-600 text-white hover:bg-green-500 active:bg-green-700'
                }`}
              >
                {wasmLoading ? 'Loading...' : isSimulating ? 'Running...' : 'Run Step'}
              </button>
              <p className="text-xs 2xl:text-sm text-slate-500 mt-2 md:mt-3 text-center hidden lg:block">
                10 simulations remaining today
              </p>
            </div>

            {/* Save */}
            <div className="flex-1 bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6">
              <button className="w-full py-2.5 md:py-3 2xl:py-4 rounded-lg font-medium bg-slate-800 text-white hover:bg-slate-700 active:bg-slate-600 transition-colors text-sm md:text-base 2xl:text-lg">
                Save State
              </button>
              <p className="text-xs 2xl:text-sm text-slate-500 mt-2 md:mt-3 text-center hidden lg:block">
                3 of 5 save slots used
              </p>
            </div>
          </div>

          {/* Upgrade prompt - hidden on mobile sheet, visible on desktop */}
          <div className="hidden lg:block bg-gradient-to-br from-blue-900/50 to-purple-900/50 rounded-xl border border-blue-800/50 p-5 2xl:p-6">
            <h3 className="font-medium text-white mb-2 text-base 2xl:text-lg">Need more?</h3>
            <p className="text-sm 2xl:text-base text-slate-300 mb-4">
              Upgrade for unlimited simulations, API access, and team features.
            </p>
            <a
              href="/pricing"
              className="block w-full py-2.5 2xl:py-3 text-center rounded-lg bg-white text-slate-900 font-medium hover:bg-slate-100 transition-colors text-sm 2xl:text-base"
            >
              View Plans
            </a>
          </div>
        </div>

        {/* Mobile controls overlay backdrop */}
        {controlsOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-30 lg:hidden"
            onClick={() => setControlsOpen(false)}
          />
        )}
      </div>

      {/* Info cards - responsive grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-3 gap-4 md:gap-6">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6">
          <h3 className="font-medium text-white mb-1.5 md:mb-2 text-sm md:text-base 2xl:text-lg">
            What are Attractors?
          </h3>
          <p className="text-xs md:text-sm 2xl:text-base text-slate-400 leading-relaxed">
            Attractor basins represent stable states that nations tend toward.
            Higher basin strength means more stability.
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6">
          <h3 className="font-medium text-white mb-1.5 md:mb-2 text-sm md:text-base 2xl:text-lg">
            Understanding Risk
          </h3>
          <p className="text-xs md:text-sm 2xl:text-base text-slate-400 leading-relaxed">
            Transition risk shows the probability of a nation shifting to a
            different political or economic state.
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 md:p-5 2xl:p-6 sm:col-span-2 lg:col-span-1">
          <h3 className="font-medium text-white mb-1.5 md:mb-2 text-sm md:text-base 2xl:text-lg">
            Regime Types
          </h3>
          <p className="text-xs md:text-sm 2xl:text-base text-slate-400 leading-relaxed">
            Nations are categorized by political structure: democracies,
            authoritarian states, and transitional regimes.
          </p>
        </div>
      </div>
    </div>
  );
}
