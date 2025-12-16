'use client';

import { useState, useEffect } from 'react';
import { GitBranch, Activity, RefreshCw, ArrowRight, Clock, Zap } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

interface CascadeRelation {
  trigger_domain: string;
  effect_domain: string;
  co_occurrences: number;
  avg_lag_hours?: number;
}

interface ActiveDomain {
  domain: string;
  eventCount: number;
}

export default function CascadesPage() {
  const [cascades, setCascades] = useState<CascadeRelation[]>([]);
  const [activeDomains, setActiveDomains] = useState<ActiveDomain[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    setLoading(true);
    try {
      const [summaryRes, recentRes] = await Promise.all([
        fetch('/api/query/cascades?mode=summary'),
        fetch('/api/query/cascades?mode=recent'),
      ]);

      if (summaryRes.ok) {
        const summary = await summaryRes.json();
        setCascades(summary.topCascades || []);
        setLastUpdate(summary.lastUpdate);
      }

      if (recentRes.ok) {
        const recent = await recentRes.json();
        setActiveDomains(recent.activeDomains || []);
      }
    } catch (e) {
      console.error('Failed to fetch cascade data:', e);
    } finally {
      setLoading(false);
    }
  }

  const formatTime = (ts: string | null) => {
    if (!ts) return 'Never';
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Cascade Analysis</h1>
          <p className="text-slate-400 mt-1">Domain-to-domain event propagation patterns</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Clock className="w-4 h-4" />
            <span>Updated: {formatTime(lastUpdate)}</span>
          </div>
          <GlassButton variant="secondary" size="sm" onClick={fetchData}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </GlassButton>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Domains */}
        <GlassCard blur="heavy">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-green-400" />
            <h2 className="text-lg font-semibold text-white">Active Domains (24h)</h2>
          </div>

          {loading ? (
            <div className="h-40 flex items-center justify-center">
              <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
            </div>
          ) : activeDomains.length === 0 ? (
            <div className="h-40 flex items-center justify-center text-slate-500 text-sm">
              No active domains in the past 24 hours
            </div>
          ) : (
            <div className="space-y-2">
              {activeDomains.map((domain, i) => (
                <div
                  key={domain.domain}
                  className="flex items-center justify-between p-3 bg-black/20 rounded-xl border border-white/[0.04]"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center text-xs font-bold text-white">
                      {i + 1}
                    </span>
                    <span className="text-white font-medium">{domain.domain}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-4 h-4 text-amber-400" />
                    <span className="text-slate-300">{domain.eventCount} events</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        {/* Cascade Relationships */}
        <div className="lg:col-span-2">
          <GlassCard blur="heavy">
            <div className="flex items-center gap-2 mb-4">
              <GitBranch className="w-5 h-5 text-purple-400" />
              <h2 className="text-lg font-semibold text-white">Top Cascade Patterns</h2>
            </div>

            {loading ? (
              <div className="h-64 flex items-center justify-center">
                <RefreshCw className="w-6 h-6 text-slate-500 animate-spin" />
              </div>
            ) : cascades.length === 0 ? (
              <div className="h-64 flex flex-col items-center justify-center border border-dashed border-white/[0.08] rounded-xl bg-black/20">
                <GitBranch className="w-10 h-10 text-slate-500 mb-3" />
                <p className="text-slate-400">No cascade patterns computed yet</p>
                <p className="text-slate-500 text-sm mt-1">
                  Cascade analysis runs daily at 00:30 UTC
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {cascades.map((cascade, i) => (
                  <div
                    key={`${cascade.trigger_domain}-${cascade.effect_domain}`}
                    className="p-4 bg-black/20 rounded-xl border border-white/[0.04] hover:border-white/[0.08] transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      {/* Trigger */}
                      <div className="flex-1">
                        <p className="text-xs text-slate-500 mb-1">Trigger Domain</p>
                        <p className="text-white font-medium">{cascade.trigger_domain}</p>
                      </div>

                      {/* Arrow */}
                      <div className="flex items-center gap-2">
                        <div className="w-12 h-0.5 bg-gradient-to-r from-purple-500 to-cyan-500" />
                        <ArrowRight className="w-5 h-5 text-cyan-400" />
                      </div>

                      {/* Effect */}
                      <div className="flex-1">
                        <p className="text-xs text-slate-500 mb-1">Effect Domain</p>
                        <p className="text-white font-medium">{cascade.effect_domain}</p>
                      </div>

                      {/* Stats */}
                      <div className="text-right">
                        <p className="text-2xl font-bold text-cyan-400">{cascade.co_occurrences}</p>
                        <p className="text-xs text-slate-500">co-occurrences</p>
                      </div>
                    </div>

                    {cascade.avg_lag_hours !== undefined && (
                      <div className="mt-3 pt-3 border-t border-white/[0.04] flex items-center gap-2 text-sm text-slate-400">
                        <Clock className="w-4 h-4" />
                        <span>Average lag: {cascade.avg_lag_hours.toFixed(1)} hours</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </GlassCard>
        </div>
      </div>

      {/* Explanation */}
      <GlassCard className="p-4 border-dashed">
        <div className="flex items-center gap-3 text-slate-400">
          <GitBranch className="w-5 h-5" />
          <div>
            <p className="text-sm font-medium">What is cascade analysis?</p>
            <p className="text-xs">
              Cascade patterns identify how events in one domain tend to trigger events in another.
              For example, political instability often cascades into economic volatility.
              This analysis is computed daily from historical signal correlations.
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
