'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';

// Dynamic import for map (client-side only)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[500px] bg-slate-900 rounded-xl flex items-center justify-center">
      <div className="text-slate-400">Loading map...</div>
    </div>
  ),
});

// Preset configurations for consumer
const PRESETS = [
  {
    id: 'global',
    name: 'Global Overview',
    description: 'See all nation attractors',
    icon: '🌍',
  },
  {
    id: 'nato',
    name: 'NATO Alliance',
    description: 'Western democratic nations',
    icon: '🛡️',
  },
  {
    id: 'brics',
    name: 'BRICS Nations',
    description: 'Emerging economies bloc',
    icon: '🌏',
  },
  {
    id: 'conflict',
    name: 'Conflict Zones',
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

  // Placeholder nation data (would come from Supabase in real app)
  const nations: never[] = [];
  const edges: never[] = [];

  const handleSimulate = async () => {
    if (!wasm) return;
    setIsSimulating(true);
    // Simulate for consumer is rate-limited and simplified
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSimulating(false);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Explore Attractors</h1>
        <p className="text-slate-400 mt-1">
          Visualize nation-level dynamics and stability patterns
        </p>
      </div>

      {/* Preset selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => setSelectedPreset(preset.id)}
            className={`p-4 rounded-xl border text-left transition-all ${
              selectedPreset === preset.id
                ? 'bg-blue-600 border-blue-500 text-white'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <span className="text-2xl">{preset.icon}</span>
            <p className="font-medium mt-2">{preset.name}</p>
            <p className="text-sm opacity-75 mt-1">{preset.description}</p>
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Map */}
        <div className="lg:col-span-3">
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            <div className="h-[500px]">
              <AttractorMap
                nations={nations}
                edges={edges}
                layer={selectedLayer}
              />
            </div>
          </div>
        </div>

        {/* Controls sidebar */}
        <div className="space-y-6">
          {/* Layer selector */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <h3 className="font-medium text-white mb-4">Visualization</h3>
            <div className="space-y-2">
              {LAYERS.map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => setSelectedLayer(layer.id as 'basin' | 'risk' | 'regime')}
                  className={`w-full p-3 rounded-lg text-left transition-colors ${
                    selectedLayer === layer.id
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  <p className="font-medium text-sm">{layer.name}</p>
                  <p className="text-xs opacity-75">{layer.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Simulate button */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <h3 className="font-medium text-white mb-4">Simulation</h3>
            <button
              onClick={handleSimulate}
              disabled={isSimulating || wasmLoading}
              className={`w-full py-3 rounded-lg font-medium transition-colors ${
                isSimulating || wasmLoading
                  ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-500'
              }`}
            >
              {wasmLoading ? 'Loading...' : isSimulating ? 'Simulating...' : 'Run Step'}
            </button>
            <p className="text-xs text-slate-500 mt-3 text-center">
              10 simulations remaining today
            </p>
          </div>

          {/* Save */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <button className="w-full py-3 rounded-lg font-medium bg-slate-800 text-white hover:bg-slate-700 transition-colors">
              Save Current State
            </button>
            <p className="text-xs text-slate-500 mt-3 text-center">
              3 of 5 save slots used
            </p>
          </div>

          {/* Upgrade prompt */}
          <div className="bg-gradient-to-br from-blue-900/50 to-purple-900/50 rounded-xl border border-blue-800/50 p-6">
            <h3 className="font-medium text-white mb-2">Need more?</h3>
            <p className="text-sm text-slate-300 mb-4">
              Upgrade to Enterprise for unlimited simulations, API access, and team features.
            </p>
            <a
              href="/pricing"
              className="block w-full py-2 text-center rounded-lg bg-white text-slate-900 font-medium hover:bg-slate-100 transition-colors"
            >
              View Plans
            </a>
          </div>
        </div>
      </div>

      {/* Info cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-medium text-white mb-2">What are Attractors?</h3>
          <p className="text-sm text-slate-400">
            Attractor basins represent stable states that nations tend toward.
            Higher basin strength means more stability.
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-medium text-white mb-2">Understanding Risk</h3>
          <p className="text-sm text-slate-400">
            Transition risk shows the probability of a nation shifting to a
            different political or economic state.
          </p>
        </div>
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
          <h3 className="font-medium text-white mb-2">Regime Types</h3>
          <p className="text-sm text-slate-400">
            Nations are categorized by political structure: democracies,
            authoritarian states, and transitional regimes.
          </p>
        </div>
      </div>
    </div>
  );
}
