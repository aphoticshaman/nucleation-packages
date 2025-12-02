'use client';

import { MapLayer, REGIMES } from '@/types';

interface ControlPanelProps {
  layer: MapLayer;
  onLayerChange: (layer: MapLayer) => void;
  onStep: () => void;
  isSimulating: boolean;
  alertLevel: string;
}

const LAYER_OPTIONS: { value: MapLayer; label: string; description: string }[] = [
  { value: 'basin', label: 'Basin Strength', description: 'Attractor stability' },
  { value: 'risk', label: 'Transition Risk', description: 'Phase transition probability' },
  { value: 'influence', label: 'Influence Flow', description: 'Cross-national influence' },
  { value: 'regime', label: 'Regime Clusters', description: 'Political regime types' },
];

export default function ControlPanel({
  layer,
  onLayerChange,
  onStep,
  isSimulating,
  alertLevel,
}: ControlPanelProps) {
  return (
    <div className="control-panel">
      <h2 className="text-lg font-bold mb-4">LatticeForge</h2>

      {/* Layer selector */}
      <div className="mb-4">
        <label className="block text-sm text-slate-400 mb-2">Visualization Layer</label>
        <div className="space-y-2">
          {LAYER_OPTIONS.map((option) => (
            <button
              key={option.value}
              onClick={() => onLayerChange(option.value)}
              className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                layer === option.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              <div className="font-medium">{option.label}</div>
              <div className="text-xs opacity-75">{option.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Simulation controls */}
      <div className="mb-4">
        <label className="block text-sm text-slate-400 mb-2">Simulation</label>
        <button
          onClick={onStep}
          disabled={isSimulating}
          className={`w-full px-4 py-2 rounded-md font-medium transition-colors ${
            isSimulating
              ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-500'
          }`}
        >
          {isSimulating ? 'Running...' : 'Step Forward'}
        </button>
      </div>

      {/* Status indicator */}
      <div className="mb-4">
        <label className="block text-sm text-slate-400 mb-2">System Status</label>
        <div
          className={`px-3 py-2 rounded-md text-sm font-medium ${
            alertLevel === 'normal'
              ? 'bg-green-900/50 text-green-400'
              : alertLevel === 'elevated'
                ? 'bg-yellow-900/50 text-yellow-400'
                : alertLevel === 'warning'
                  ? 'bg-orange-900/50 text-orange-400'
                  : 'bg-red-900/50 text-red-400 animate-pulse'
          }`}
        >
          {alertLevel.toUpperCase()}
        </div>
      </div>

      {/* Legend */}
      {layer === 'regime' && (
        <div className="mb-4">
          <label className="block text-sm text-slate-400 mb-2">Regime Types</label>
          <div className="space-y-1">
            {Object.entries(REGIMES).map(([id, { name, color }]) => (
              <div key={id} className="flex items-center gap-2 text-xs">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-slate-300">{name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {layer === 'basin' && (
        <div className="mb-4">
          <label className="block text-sm text-slate-400 mb-2">Basin Strength</label>
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-500">Low</span>
            <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500" />
            <span className="text-xs text-slate-500">High</span>
          </div>
        </div>
      )}

      {layer === 'risk' && (
        <div className="mb-4">
          <label className="block text-sm text-slate-400 mb-2">Transition Risk</label>
          <div className="flex items-center gap-1">
            <span className="text-xs text-slate-500">Low</span>
            <div className="flex-1 h-2 rounded-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500" />
            <span className="text-xs text-slate-500">High</span>
          </div>
        </div>
      )}
    </div>
  );
}
