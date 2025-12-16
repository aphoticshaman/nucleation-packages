'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Radio, Filter } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { DataFreshness } from '@/components/DataFreshness';

interface Signal {
  timestamp: string;
  source: string;
  domain: string;
  [key: string]: unknown;
}

const sourceColors: Record<string, string> = {
  gdelt: 'text-blue-400 bg-blue-500/10',
  usgs: 'text-amber-400 bg-amber-500/10',
  sentiment: 'text-purple-400 bg-purple-500/10',
  worldbank: 'text-green-400 bg-green-500/10',
};

export default function SignalsPage() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState('all');

  useEffect(() => {
    fetchSignals();
  }, [source]);

  async function fetchSignals() {
    setLoading(true);
    try {
      const res = await fetch(`/api/query/signals?source=${source}&limit=100`, {
        headers: { 'x-user-tier': 'enterprise_tier' },
      });
      const data = await res.json();
      setSignals(data.signals || []);
    } catch (e) {
      console.error('Failed to fetch signals:', e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Signals</h1>
          <p className="text-slate-400 mt-1">Real-time signal feed from all sources</p>
        </div>
        <div className="flex items-center gap-3">
          <DataFreshness compact />
          <GlassButton variant="secondary" size="sm" onClick={fetchSignals}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </GlassButton>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Filter className="w-4 h-4 text-slate-500" />
        {['all', 'gdelt', 'usgs', 'sentiment', 'worldbank'].map((s) => (
          <button
            key={s}
            onClick={() => setSource(s)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              source === s ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {s === 'all' ? 'All Sources' : s.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
          </div>
        ) : signals.length === 0 ? (
          <GlassCard className="p-8 text-center">
            <Radio className="w-10 h-10 text-slate-500 mx-auto mb-3" />
            <p className="text-slate-400">No signals found</p>
          </GlassCard>
        ) : (
          signals.map((signal, i) => (
            <GlassCard key={i} className="p-4">
              <div className="flex items-start gap-4">
                <div className={`px-2 py-1 rounded text-xs font-mono ${sourceColors[signal.source] || 'text-slate-400 bg-slate-500/10'}`}>
                  {signal.source}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-white font-medium">{signal.domain || 'Global'}</span>
                    <span className="text-xs text-slate-500">
                      {new Date(signal.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-sm text-slate-400 line-clamp-2">
                    {String(signal.title || signal.summary || JSON.stringify(signal.numeric_features || {}))}
                  </div>
                </div>
              </div>
            </GlassCard>
          ))
        )}
      </div>
    </div>
  );
}
