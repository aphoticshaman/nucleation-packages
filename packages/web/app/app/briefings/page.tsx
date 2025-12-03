'use client';

import { useState } from 'react';
import { useIntelBriefing, getRiskBadgeStyle } from '@/hooks/useIntelBriefing';

const PRESETS = [
  { id: 'global', name: 'Global Overview', icon: '🌍', desc: 'All 195 nations' },
  { id: 'nato', name: 'NATO Alliance', icon: '🛡️', desc: '32 member states' },
  { id: 'brics', name: 'BRICS+', icon: '🌏', desc: 'Emerging powers bloc' },
  { id: 'conflict', name: 'Hot Spots', icon: '⚠️', desc: 'Active tension zones' },
];

export default function BriefingsPage() {
  const [selectedPreset, setSelectedPreset] = useState('global');
  const { briefings, metadata, loading, refetch } = useIntelBriefing(selectedPreset);
  const [hasLoaded, setHasLoaded] = useState(false);

  const handleLoad = async () => {
    setHasLoaded(true);
    await refetch();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Intel Briefings</h1>
        <p className="text-slate-400 mt-1">AI-generated intelligence summaries across all domains</p>
      </div>

      {/* Preset selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            onClick={() => {
              setSelectedPreset(preset.id);
              setHasLoaded(false);
            }}
            className={`p-4 rounded-xl border text-left transition-all ${
              selectedPreset === preset.id
                ? 'bg-blue-600/20 border-blue-500 text-white'
                : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <span className="text-2xl">{preset.icon}</span>
            <p className="font-medium mt-2">{preset.name}</p>
            <p className="text-xs text-slate-400 mt-1">{preset.desc}</p>
          </button>
        ))}
      </div>

      {/* Load button or briefing content */}
      {!hasLoaded ? (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center">
          <p className="text-slate-400 mb-4">Ready to generate briefing for {PRESETS.find(p => p.id === selectedPreset)?.name}</p>
          <button
            onClick={() => void handleLoad()}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg font-medium"
          >
            Generate Intel Briefing
          </button>
          <p className="text-xs text-slate-500 mt-3">Cached for 10 minutes • Uses AI analysis</p>
        </div>
      ) : loading ? (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-8">
          <div className="animate-pulse space-y-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-slate-800 rounded w-32" />
                <div className="h-3 bg-slate-800 rounded w-full" />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-slate-900 rounded-xl border border-slate-800 p-6 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-lg font-semibold text-white">
                {PRESETS.find(p => p.id === selectedPreset)?.name} Briefing
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Generated {metadata?.timestamp ? new Date(metadata.timestamp).toLocaleString() : 'now'}
              </p>
            </div>
            {metadata && (
              <span className={`px-3 py-1 rounded text-sm font-medium ${getRiskBadgeStyle(metadata.overallRisk)}`}>
                {metadata.overallRisk?.toUpperCase()} RISK
              </span>
            )}
          </div>

          {/* Briefing sections */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {briefings && Object.entries(briefings).filter(([key]) => !['summary', 'nsm'].includes(key)).map(([key, value]) => (
              <div key={key} className="space-y-2">
                <h3 className="text-sm font-medium text-blue-400 capitalize">{key.replace(/_/g, ' ')}</h3>
                <p className="text-sm text-slate-300">{value as string}</p>
              </div>
            ))}
          </div>

          {/* Summary & NSM */}
          {briefings?.summary && (
            <div className="border-t border-slate-800 pt-4">
              <p className="text-slate-400 italic">{briefings.summary}</p>
            </div>
          )}
          {briefings?.nsm && (
            <div className="bg-blue-950/30 border border-blue-800/50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-blue-300 mb-2">🎯 Next Strategic Move</h3>
              <p className="text-sm text-blue-200">{briefings.nsm}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
