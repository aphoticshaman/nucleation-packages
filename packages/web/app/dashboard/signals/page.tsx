'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Radio, Filter } from 'lucide-react';
import { Card, Button, EmptyState, SkeletonList } from '@/components/ui';
import { DataFreshness } from '@/components/DataFreshness';

interface Signal {
  timestamp: string;
  source: string;
  domain: string;
  [key: string]: unknown;
}

const sourceStyles: Record<string, string> = {
  gdelt: 'text-blue-400 border-blue-600/30 bg-blue-600/5',
  usgs: 'text-amber-400 border-amber-600/30 bg-amber-600/5',
  sentiment: 'text-purple-400 border-purple-600/30 bg-purple-600/5',
  worldbank: 'text-emerald-400 border-emerald-600/30 bg-emerald-600/5',
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
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Signal Feed</h1>
          <p className="text-sm text-slate-500 mt-0.5">Aggregated signals from all data sources</p>
        </div>
        <div className="flex items-center gap-3">
          <DataFreshness compact />
          <Button variant="secondary" size="sm" onClick={fetchSignals} loading={loading}>
            <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-1.5 border-b border-slate-800 pb-3">
        <Filter className="w-3.5 h-3.5 text-slate-600 mr-1" />
        {['all', 'gdelt', 'usgs', 'sentiment', 'worldbank'].map((s) => (
          <button
            key={s}
            onClick={() => setSource(s)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              source === s
                ? 'bg-slate-800 text-slate-200'
                : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'
            }`}
          >
            {s === 'all' ? 'All' : s.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="space-y-2">
        {loading ? (
          <SkeletonList items={8} />
        ) : signals.length === 0 ? (
          <EmptyState
            icon={Radio}
            title="No signals available"
            description="No signals match the current filter. Adjust source filters or check back when new data arrives."
            action={{ label: 'Clear filters', onClick: () => setSource('all') }}
          />
        ) : (
          signals.map((signal, i) => (
            <Card key={i} padding="sm" interactive>
              <div className="flex items-start gap-3">
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-mono uppercase border ${
                    sourceStyles[signal.source] || 'text-slate-400 border-slate-700 bg-slate-800'
                  }`}
                >
                  {signal.source}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-sm text-slate-200 font-medium">{signal.domain || 'Global'}</span>
                    <span className="text-xs text-slate-600">
                      {new Date(signal.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 line-clamp-2">
                    {String(signal.title || signal.summary || JSON.stringify(signal.numeric_features || {}))}
                  </p>
                </div>
              </div>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}
