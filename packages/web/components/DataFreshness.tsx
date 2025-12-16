'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, Clock, CheckCircle, AlertCircle } from 'lucide-react';

interface FreshnessData {
  [source: string]: {
    last_update: string | null;
    count_24h: number;
  };
}

function formatAge(timestamp: string | null): { text: string; status: 'fresh' | 'stale' | 'old' } {
  if (!timestamp) return { text: 'No data', status: 'old' };

  const age = Date.now() - new Date(timestamp).getTime();
  const minutes = Math.floor(age / 60000);
  const hours = Math.floor(age / 3600000);

  if (minutes < 30) return { text: `${minutes}m ago`, status: 'fresh' };
  if (hours < 4) return { text: `${hours}h ago`, status: 'fresh' };
  if (hours < 24) return { text: `${hours}h ago`, status: 'stale' };
  return { text: `${Math.floor(hours / 24)}d ago`, status: 'old' };
}

const sourceLabels: Record<string, string> = {
  gdelt: 'GDELT News',
  usgs: 'USGS Earthquakes',
  sentiment: 'Market Sentiment',
};

export function DataFreshness({ compact = false }: { compact?: boolean }) {
  const [freshness, setFreshness] = useState<FreshnessData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchFreshness() {
      try {
        const res = await fetch('/api/query/signals?source=freshness');
        const data = await res.json();
        setFreshness(data.freshness);
      } catch (e) {
        console.error('Failed to fetch freshness:', e);
      } finally {
        setLoading(false);
      }
    }

    fetchFreshness();
    const interval = setInterval(fetchFreshness, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-slate-500 text-sm">
        <RefreshCw className="w-4 h-4 animate-spin" />
        <span>Loading...</span>
      </div>
    );
  }

  if (!freshness) {
    return null;
  }

  if (compact) {
    const allFresh = Object.values(freshness).every(
      (f) => f.last_update && formatAge(f.last_update).status === 'fresh'
    );

    return (
      <div className="flex items-center gap-2 text-sm">
        {allFresh ? (
          <CheckCircle className="w-4 h-4 text-green-400" />
        ) : (
          <AlertCircle className="w-4 h-4 text-amber-400" />
        )}
        <span className={allFresh ? 'text-green-400' : 'text-amber-400'}>
          {allFresh ? 'Data Fresh' : 'Data Stale'}
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-4 text-sm">
      {Object.entries(freshness).map(([source, data]) => {
        const { text, status } = formatAge(data.last_update);
        const statusColors = {
          fresh: 'text-green-400',
          stale: 'text-amber-400',
          old: 'text-red-400',
        };

        return (
          <div key={source} className="flex items-center gap-2">
            <Clock className={`w-3 h-3 ${statusColors[status]}`} />
            <span className="text-slate-400">{sourceLabels[source] || source}:</span>
            <span className={statusColors[status]}>{text}</span>
            {data.count_24h > 0 && (
              <span className="text-slate-500">({data.count_24h} events)</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
