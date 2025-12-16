'use client';

import { useState, useEffect } from 'react';
import { RefreshCw, GitBranch, Activity } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

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
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Cascade Analysis</h1>
          <p className="text-slate-400 mt-1">Cross-domain event propagation patterns</p>
        </div>
        <GlassButton variant="secondary" size="sm" onClick={fetchCascades}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </GlassButton>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <GlassCard blur="heavy">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">Active Domains (24h)</h2>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-32">
              <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
            </div>
          ) : activeDomains.length === 0 ? (
            <p className="text-slate-400 text-center py-8">No significant domain activity</p>
          ) : (
            <div className="space-y-2">
              {activeDomains.map((d) => (
                <div key={d.domain} className="flex items-center justify-between p-3 bg-black/20 rounded-lg">
                  <span className="text-white font-medium">{d.domain}</span>
                  <span className="text-cyan-400">{d.eventCount} events</span>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        <GlassCard blur="heavy">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-5 h-5 text-purple-400" />
            <h2 className="text-lg font-bold text-white">Top Cascade Paths</h2>
          </div>

          {loading ? (
            <div className="flex items-center justify-center h-32">
              <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
            </div>
          ) : cascades.length === 0 ? (
            <p className="text-slate-400 text-center py-8">Cascade matrix not yet computed</p>
          ) : (
            <div className="space-y-2">
              {cascades.map((c, i) => (
                <div key={i} className="flex items-center gap-3 p-3 bg-black/20 rounded-lg">
                  <span className="text-white">{c.trigger_domain}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-white">{c.effect_domain}</span>
                  <span className="ml-auto text-purple-400">{c.co_occurrences}x</span>
                </div>
              ))}
            </div>
          )}
        </GlassCard>
      </div>

      <GlassCard className="p-4 border-dashed">
        <div className="flex items-center gap-3 text-slate-400">
          <GitBranch className="w-5 h-5" />
          <div>
            <p className="text-sm font-medium">Cascade Detection</p>
            <p className="text-xs">
              Identifies when events in one domain trigger correlated events in another.
              Used for early warning and risk propagation modeling.
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
