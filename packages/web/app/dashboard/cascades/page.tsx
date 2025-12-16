'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, GitBranch, Activity } from 'lucide-react';
import { Card, Button, EmptyState, Skeleton } from '@/components/ui';

interface CascadeEdge {
  trigger_domain: string;
  effect_domain: string;
  co_occurrences: number;
}

interface ActiveDomain {
  domain: string;
  eventCount: number;
}

export default function CascadesPage() {
  const [cascades, setCascades] = useState<CascadeEdge[]>([]);
  const [activeDomains, setActiveDomains] = useState<ActiveDomain[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCascades();
  }, []);

  async function fetchCascades() {
    setLoading(true);
    try {
      const [matrixRes, recentRes] = await Promise.all([
        fetch('/api/query/cascades?mode=summary', {
          headers: { 'x-user-tier': 'enterprise_tier' },
        }),
        fetch('/api/query/cascades?mode=recent', {
          headers: { 'x-user-tier': 'enterprise_tier' },
        }),
      ]);

      const matrixData = await matrixRes.json();
      const recentData = await recentRes.json();

      setCascades(matrixData.topCascades || []);
      setActiveDomains(recentData.activeDomains || []);
    } catch (e) {
      console.error('Failed to fetch cascades:', e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Cascade Analysis</h1>
          <p className="text-sm text-slate-500 mt-0.5">Cross-domain event propagation patterns</p>
        </div>
        <Button variant="secondary" size="sm" onClick={fetchCascades} loading={loading}>
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Active Domains */}
        <Card padding="md">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-4 h-4 text-slate-400" />
            <h2 className="text-sm font-semibold text-slate-200">Active Domains (24h)</h2>
          </div>

          {loading ? (
            <div className="space-y-2">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-10 rounded" />
              ))}
            </div>
          ) : activeDomains.length === 0 ? (
            <div className="py-6 text-center">
              <p className="text-sm text-slate-500">No significant domain activity detected</p>
            </div>
          ) : (
            <div className="space-y-2">
              {activeDomains.map((d) => (
                <div key={d.domain} className="flex items-center justify-between py-2 px-3 bg-slate-800/50 rounded">
                  <span className="text-sm text-slate-300">{d.domain}</span>
                  <span className="text-xs text-slate-500">{d.eventCount} events</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Top Cascade Paths */}
        <Card padding="md">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-4 h-4 text-slate-400" />
            <h2 className="text-sm font-semibold text-slate-200">Top Cascade Paths</h2>
          </div>

          {loading ? (
            <div className="space-y-2">
              {[...Array(4)].map((_, i) => (
                <Skeleton key={i} className="h-10 rounded" />
              ))}
            </div>
          ) : cascades.length === 0 ? (
            <div className="py-6 text-center">
              <p className="text-sm text-slate-500">Cascade matrix not yet computed</p>
            </div>
          ) : (
            <div className="space-y-2">
              {cascades.map((c, i) => (
                <div key={i} className="flex items-center gap-2 py-2 px-3 bg-slate-800/50 rounded text-sm">
                  <span className="text-slate-300">{c.trigger_domain}</span>
                  <span className="text-slate-600">→</span>
                  <span className="text-slate-300">{c.effect_domain}</span>
                  <span className="ml-auto text-xs text-slate-500">{c.co_occurrences}x</span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>

      {/* Info Footer */}
      <Card padding="sm" className="border-dashed border-slate-700">
        <div className="flex items-center gap-3 text-slate-500">
          <GitBranch className="w-4 h-4 shrink-0" />
          <p className="text-xs">
            Cascade detection identifies when events in one domain trigger correlated events in another.
            Used for early warning and risk propagation modeling.
          </p>
        </div>
      </Card>
    </div>
  );
}
