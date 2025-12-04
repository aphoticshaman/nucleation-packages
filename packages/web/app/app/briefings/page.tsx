'use client';

import { useState } from 'react';
import { useIntelBriefing, getRiskBadgeStyle } from '@/hooks/useIntelBriefing';
import { Globe, Shield, TrendingUp, AlertTriangle, RefreshCw, Target, BookOpen } from 'lucide-react';
import Glossary from '@/components/Glossary';
import HelpTip from '@/components/HelpTip';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

const PRESETS = [
  { id: 'global', name: 'Global Overview', icon: Globe, desc: 'All 195 nations' },
  { id: 'nato', name: 'NATO Alliance', icon: Shield, desc: '32 member states' },
  { id: 'brics', name: 'BRICS+', icon: TrendingUp, desc: 'Emerging powers bloc' },
  { id: 'conflict', name: 'Hot Spots', icon: AlertTriangle, desc: 'Active tension zones' },
];

export default function BriefingsPage() {
  const [selectedPreset, setSelectedPreset] = useState('global');
  const { briefings, metadata, loading, refetch } = useIntelBriefing(selectedPreset);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);

  const handleLoad = async () => {
    setHasLoaded(true);
    await refetch();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Intel Briefings</h1>
          <p className="text-slate-400 mt-1">AI-generated intelligence summaries across all domains</p>
        </div>
        <button
          onClick={() => setShowGlossary(true)}
          className="flex items-center gap-2 px-3 py-2 min-h-[44px] bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/[0.12] transition-all"
        >
          <BookOpen className="w-4 h-4" />
          <span className="text-sm">Terms</span>
        </button>
      </div>

      {/* Preset selector */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {PRESETS.map((preset) => {
          const Icon = preset.icon;
          return (
            <button
              key={preset.id}
              onClick={() => {
                setSelectedPreset(preset.id);
                setHasLoaded(false);
              }}
              className={`p-4 rounded-xl border text-left transition-all min-h-[100px] ${
                selectedPreset === preset.id
                  ? 'bg-blue-500/20 border-blue-500/50 text-white'
                  : 'bg-[rgba(18,18,26,0.7)] backdrop-blur-sm border-white/[0.06] text-slate-300 hover:border-white/[0.12] hover:bg-[rgba(18,18,26,0.8)]'
              }`}
            >
              <Icon className={`w-6 h-6 mb-2 ${selectedPreset === preset.id ? 'text-blue-400' : 'text-slate-500'}`} />
              <p className="font-medium">{preset.name}</p>
              <p className="text-xs text-slate-400 mt-1">{preset.desc}</p>
            </button>
          );
        })}
      </div>

      {/* Load button or briefing content */}
      {!hasLoaded ? (
        <GlassCard blur="heavy" className="p-8 text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-blue-500/20 flex items-center justify-center">
            <RefreshCw className="w-8 h-8 text-blue-400" />
          </div>
          <p className="text-slate-400 mb-4">Ready to generate briefing for {PRESETS.find(p => p.id === selectedPreset)?.name}</p>
          <GlassButton
            variant="primary"
            glow
            onClick={() => void handleLoad()}
          >
            Generate Intel Briefing
          </GlassButton>
          <p className="text-xs text-slate-500 mt-3">Cached for 10 minutes • Uses AI analysis</p>
        </GlassCard>
      ) : loading ? (
        <GlassCard blur="heavy" className="p-8">
          <div className="animate-pulse space-y-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-white/10 rounded w-32" />
                <div className="h-3 bg-white/10 rounded w-full" />
              </div>
            ))}
          </div>
        </GlassCard>
      ) : (
        <GlassCard blur="heavy" className="p-6 space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-4">
            <div>
              <h2 className="text-lg font-semibold text-white">
                {PRESETS.find(p => p.id === selectedPreset)?.name} Briefing
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Generated {metadata?.timestamp ? new Date(metadata.timestamp).toLocaleString() : 'now'}
              </p>
            </div>
            {metadata && (
              <div className="flex items-center gap-1">
                <span className={`px-3 py-1.5 rounded-lg text-sm font-medium ${getRiskBadgeStyle(metadata.overallRisk)}`}>
                  {metadata.overallRisk?.toUpperCase()} RISK
                </span>
                <HelpTip term="Transition Risk" skillLevel="standard" size={10} />
              </div>
            )}
          </div>

          {/* Briefing sections */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {briefings && Object.entries(briefings).filter(([key]) => !['summary', 'nsm'].includes(key)).map(([key, value]) => (
              <div key={key} className="bg-black/20 rounded-lg border border-white/[0.04] p-4">
                <h3 className="text-sm font-medium text-blue-400 capitalize mb-2">{key.replace(/_/g, ' ')}</h3>
                <p className="text-sm text-slate-300">{value as string}</p>
              </div>
            ))}
          </div>

          {/* Summary & NSM */}
          {briefings?.summary && (
            <div className="border-t border-white/[0.06] pt-4">
              <p className="text-slate-400 italic">{briefings.summary}</p>
            </div>
          )}
          {briefings?.nsm && (
            <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-medium text-blue-300">
                  Next Strategic Move
                  <HelpTip term="NSM (Next Strategic Move)" skillLevel="standard" size={10} />
                </h3>
              </div>
              <p className="text-sm text-blue-200">{briefings.nsm}</p>
            </div>
          )}
        </GlassCard>
      )}

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel="standard"
      />
    </div>
  );
}
