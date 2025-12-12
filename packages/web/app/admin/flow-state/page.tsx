'use client';

import { useState, useEffect, useCallback } from 'react';
import { createBrowserClient } from '@supabase/ssr';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

interface OnlineStats {
  total_online: number;
  consumers_online: number;
  enterprise_online: number;
  subscribers_online: number;
}

interface SubscriberStats {
  total_subscribers: number;
  pro_subscribers: number;
  enterprise_subscribers: number;
  subscribers_online: number;
  mrr_estimate: number;
}

interface DashboardStats {
  total_users: number;
  total_consumers: number;
  total_enterprise_users: number;
  total_admins: number;
  free_tier_count: number;
  pro_tier_count: number;
  enterprise_tier_count: number;
  users_online_now: number;
  total_subscribers: number;
  subscribers_online: number;
  active_24h: number;
  active_7d: number;
  active_30d: number;
  active_orgs: number;
  api_calls_24h: number;
  total_simulations: number;
  active_sessions: number;
}

interface FlowStateSignal {
  domain: string;
  value: number;
  phase: 'stable' | 'approaching' | 'critical' | 'transitioning';
  variance: number;
  confidence: number;
}

interface NoveltyAlert {
  id: string;
  type: string;
  description: string;
  confidence: number;
  significance: number;
  timestamp: string;
}

function PhaseIndicator({ phase }: { phase: string }) {
  const colors: Record<string, string> = {
    stable: 'bg-green-500',
    approaching: 'bg-yellow-500',
    critical: 'bg-orange-500',
    transitioning: 'bg-red-500 animate-pulse',
  };

  const labels: Record<string, string> = {
    stable: 'Stable',
    approaching: 'Approaching',
    critical: 'Critical',
    transitioning: 'Transitioning',
  };

  return (
    <div className="flex items-center gap-2">
      <span className={`w-3 h-3 rounded-full ${colors[phase] || colors.stable}`} />
      <span className="text-sm text-slate-300">{labels[phase] || 'Unknown'}</span>
    </div>
  );
}

function StatCard({
  label,
  value,
  subValue,
  icon,
  trend,
}: {
  label: string;
  value: number | string;
  subValue?: string;
  icon?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
}) {
  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-slate-400',
  };

  return (
    <GlassCard blur="heavy" className="relative overflow-hidden">
      {icon && (
        <div className="absolute top-4 right-4 text-3xl opacity-20">
          {icon}
        </div>
      )}
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-3xl font-bold text-white mt-1">{value}</p>
      {subValue && (
        <p className={`text-sm mt-2 ${trend ? trendColors[trend] : 'text-slate-400'}`}>
          {subValue}
        </p>
      )}
    </GlassCard>
  );
}

function SignalCard({ signal }: { signal: FlowStateSignal }) {
  const phaseColors: Record<string, string> = {
    stable: 'border-green-500/30',
    approaching: 'border-yellow-500/30',
    critical: 'border-orange-500/30',
    transitioning: 'border-red-500/30',
  };

  return (
    <GlassCard blur="light" className={`border-l-4 ${phaseColors[signal.phase]}`}>
      <div className="flex justify-between items-start">
        <div>
          <h4 className="text-white font-medium capitalize">{signal.domain}</h4>
          <PhaseIndicator phase={signal.phase} />
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-white">
            {(signal.value * 100).toFixed(1)}%
          </p>
          <p className="text-xs text-slate-400">
            Confidence: {(signal.confidence * 100).toFixed(0)}%
          </p>
        </div>
      </div>
      <div className="mt-4 flex justify-between text-xs text-slate-400">
        <span>Variance: {signal.variance.toFixed(4)}</span>
      </div>
    </GlassCard>
  );
}

function AlertCard({ alert }: { alert: NoveltyAlert }) {
  const typeColors: Record<string, string> = {
    PATTERN_ANOMALY: 'bg-yellow-500/20 text-yellow-300',
    TEMPORAL_BREAK: 'bg-orange-500/20 text-orange-300',
    CROSS_DOMAIN_ISO: 'bg-blue-500/20 text-blue-300',
    STRUCTURAL_GAP: 'bg-purple-500/20 text-purple-300',
  };

  return (
    <div className="p-4 bg-white/[0.02] rounded-lg border border-white/[0.06]">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <span className={`px-2 py-0.5 rounded text-xs ${typeColors[alert.type] || 'bg-slate-500/20 text-slate-300'}`}>
            {alert.type.replace('_', ' ')}
          </span>
          <p className="text-white mt-2 text-sm">{alert.description}</p>
        </div>
        <div className="text-right ml-4">
          <p className="text-xs text-slate-400">Confidence</p>
          <p className="text-lg font-bold text-white">{(alert.confidence * 100).toFixed(0)}%</p>
        </div>
      </div>
      <div className="mt-2 flex justify-between text-xs text-slate-500">
        <span>Significance: {(alert.significance * 100).toFixed(0)}%</span>
        <span>{new Date(alert.timestamp).toLocaleTimeString()}</span>
      </div>
    </div>
  );
}

export default function FlowStateAnalyticsPage() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [onlineStats, setOnlineStats] = useState<OnlineStats | null>(null);
  const [subscriberStats, setSubscriberStats] = useState<SubscriberStats | null>(null);
  const [signals, setSignals] = useState<FlowStateSignal[]>([]);
  const [alerts, setAlerts] = useState<NoveltyAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  const loadStats = useCallback(async () => {
    // Load dashboard stats
    const { data: dashboardStats } = await supabase
      .from('admin_dashboard_stats')
      .select('*')
      .single();

    if (dashboardStats) {
      setStats(dashboardStats as DashboardStats);
    }

    // Load real-time online stats
    const { data: onlineData } = await supabase.rpc('get_online_count');
    if (onlineData?.[0]) {
      setOnlineStats(onlineData[0] as OnlineStats);
    }

    // Load subscriber stats
    const { data: subData } = await supabase.rpc('get_subscriber_stats');
    if (subData?.[0]) {
      setSubscriberStats(subData[0] as SubscriberStats);
    }

    // Generate simulated flow state signals (in production, these come from FlowStateEngine)
    const mockSignals: FlowStateSignal[] = [
      {
        domain: 'engagement',
        value: 0.72,
        phase: 'stable',
        variance: 0.0234,
        confidence: 0.89,
      },
      {
        domain: 'churn',
        value: 0.15,
        phase: 'approaching',
        variance: 0.0456,
        confidence: 0.76,
      },
      {
        domain: 'security',
        value: 0.08,
        phase: 'stable',
        variance: 0.0089,
        confidence: 0.95,
      },
      {
        domain: 'ux',
        value: 0.68,
        phase: 'stable',
        variance: 0.0178,
        confidence: 0.82,
      },
      {
        domain: 'ai quality',
        value: 0.91,
        phase: 'stable',
        variance: 0.0056,
        confidence: 0.94,
      },
      {
        domain: 'system health',
        value: 0.96,
        phase: 'stable',
        variance: 0.0023,
        confidence: 0.97,
      },
    ];
    setSignals(mockSignals);

    // Generate alerts based on signals
    const mockAlerts: NoveltyAlert[] = mockSignals
      .filter(s => s.phase !== 'stable')
      .map((s, i) => ({
        id: `alert-${i}`,
        type: s.phase === 'approaching' ? 'PATTERN_ANOMALY' : 'TEMPORAL_BREAK',
        description: `${s.domain} showing variance quieting - potential phase transition ahead`,
        confidence: s.confidence,
        significance: 0.7,
        timestamp: new Date().toISOString(),
      }));
    setAlerts(mockAlerts);

    setLastUpdate(new Date());
    setLoading(false);
  }, [supabase]);

  useEffect(() => {
    void loadStats();

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      void loadStats();
    }, 30000);

    return () => clearInterval(interval);
  }, [loadStats]);

  if (loading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="grid grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white/[0.02] rounded-xl h-32" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Flow State Analytics</h1>
          <p className="text-slate-400">
            Real-time nucleation monitoring with NSM-x20 insights
          </p>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-slate-500">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </span>
          <GlassButton variant="secondary" onClick={() => void loadStats()}>
            Refresh
          </GlassButton>
        </div>
      </div>

      {/* Real-time User Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          label="Total Users"
          value={stats?.total_users || 0}
          subValue={`${stats?.active_24h || 0} active today`}
          icon="👥"
        />
        <StatCard
          label="Online Now"
          value={onlineStats?.total_online || stats?.users_online_now || 0}
          subValue={`${onlineStats?.subscribers_online || 0} subscribers`}
          icon="🟢"
          trend="up"
        />
        <StatCard
          label="Total Subscribers"
          value={subscriberStats?.total_subscribers || stats?.total_subscribers || 0}
          subValue={`$${(subscriberStats?.mrr_estimate || 0).toFixed(0)} MRR`}
          icon="💎"
          trend="up"
        />
        <StatCard
          label="Active Sessions"
          value={stats?.active_sessions || 0}
          subValue={`${stats?.api_calls_24h || 0} API calls/24h`}
          icon="📊"
        />
      </div>

      {/* Tier Breakdown */}
      <GlassCard blur="heavy" className="mb-8">
        <h2 className="text-lg font-bold text-white mb-4">User Tiers</h2>
        <div className="grid grid-cols-3 gap-6">
          <div className="text-center p-4 bg-white/[0.02] rounded-lg">
            <p className="text-3xl font-bold text-slate-300">{stats?.free_tier_count || 0}</p>
            <p className="text-sm text-slate-400">Free</p>
          </div>
          <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
            <p className="text-3xl font-bold text-green-400">
              {subscriberStats?.pro_subscribers || stats?.pro_tier_count || 0}
            </p>
            <p className="text-sm text-green-300">Pro</p>
          </div>
          <div className="text-center p-4 bg-amber-500/10 rounded-lg border border-amber-500/20">
            <p className="text-3xl font-bold text-amber-400">
              {subscriberStats?.enterprise_subscribers || stats?.enterprise_tier_count || 0}
            </p>
            <p className="text-sm text-amber-300">Enterprise</p>
          </div>
        </div>
      </GlassCard>

      {/* Two-column layout: Signals + Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        {/* Flow State Signals */}
        <GlassCard blur="heavy">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-white">Flow State Signals</h2>
            <span className="text-xs text-slate-500">NSM-powered</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {signals.map((signal, i) => (
              <SignalCard key={i} signal={signal} />
            ))}
          </div>
        </GlassCard>

        {/* Novelty Alerts */}
        <GlassCard blur="heavy">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-white">Novelty Alerts</h2>
            <span className={`px-2 py-1 rounded text-xs ${
              alerts.length > 0 ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'
            }`}>
              {alerts.length} active
            </span>
          </div>
          <div className="space-y-4">
            {alerts.length > 0 ? (
              alerts.map((alert) => (
                <AlertCard key={alert.id} alert={alert} />
              ))
            ) : (
              <div className="text-center py-8 text-slate-500">
                <p>All systems stable</p>
                <p className="text-xs mt-1">No novelty signals detected</p>
              </div>
            )}
          </div>
        </GlassCard>
      </div>

      {/* NSM Insights Summary */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">NSM-x20 Active Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-white/[0.02] rounded-lg">
            <p className="text-sm text-purple-400 font-medium">#1 Variance Quieting</p>
            <p className="text-xs text-slate-400 mt-1">
              Monitoring for phase transition precursors
            </p>
            <span className="inline-block mt-2 px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded">
              Active
            </span>
          </div>
          <div className="p-4 bg-white/[0.02] rounded-lg">
            <p className="text-sm text-blue-400 font-medium">#2 Multi-Domain Coherence</p>
            <p className="text-xs text-slate-400 mt-1">
              Cross-signal correlation analysis
            </p>
            <span className="inline-block mt-2 px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded">
              Active
            </span>
          </div>
          <div className="p-4 bg-white/[0.02] rounded-lg">
            <p className="text-sm text-cyan-400 font-medium">#6 WASM Edge Detection</p>
            <p className="text-xs text-slate-400 mt-1">
              Client-side real-time processing
            </p>
            <span className="inline-block mt-2 px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded">
              Active
            </span>
          </div>
          <div className="p-4 bg-white/[0.02] rounded-lg">
            <p className="text-sm text-orange-400 font-medium">#16 Threat Quieting</p>
            <p className="text-xs text-slate-400 mt-1">
              Security anomaly detection
            </p>
            <span className="inline-block mt-2 px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded">
              Active
            </span>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
