'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import {
  ArrowLeft,
  RefreshCw,
  Settings,
  Share2,
  Download,
  LayoutDashboard,
  Globe,
  TrendingUp,
  Shield,
  Zap,
  BarChart3,
  Activity,
  AlertTriangle,
  Clock,
  Database,
} from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { useIntelBriefing } from '@/hooks/useIntelBriefing';

// Mock dashboard data - in production this would come from the database
const MOCK_DASHBOARDS: Record<string, {
  name: string;
  description: string;
  presetId: string;
  focusTemplate?: string;
  widgets: string[];
}> = {
  'dash-1': {
    name: 'Global Risk Monitor',
    description: 'My primary intel dashboard tracking global threats',
    presetId: 'intelligence-officer',
    widgets: ['threat-matrix', 'signal-feed', 'risk-heatmap', 'cascade-detector'],
  },
  'dash-2': {
    name: 'US Markets Deep Dive',
    description: 'Real-time US equity and treasury tracking',
    presetId: 'economic-analyst',
    focusTemplate: 'us-equities',
    widgets: ['market-pulse', 'sector-rotation', 'yield-curve', 'sentiment-gauge'],
  },
  'dash-3': {
    name: 'China Watch',
    description: 'Tracking China-US relations and economic signals',
    presetId: 'regional-specialist',
    focusTemplate: 'china-watch',
    widgets: ['bilateral-tracker', 'trade-flows', 'policy-monitor', 'risk-radar'],
  },
};

const PRESET_ICONS: Record<string, React.ReactNode> = {
  'intelligence-officer': <Shield className="w-5 h-5" />,
  'economic-analyst': <TrendingUp className="w-5 h-5" />,
  'crisis-monitor': <Zap className="w-5 h-5" />,
  'executive-briefer': <BarChart3 className="w-5 h-5" />,
  'regional-specialist': <Globe className="w-5 h-5" />,
};

export default function DashboardDetailPage() {
  const params = useParams();
  const router = useRouter();
  const dashboardId = params.id as string;

  const dashboard = MOCK_DASHBOARDS[dashboardId];
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Fetch briefing data based on focus template
  const preset = dashboard?.focusTemplate === 'china-watch'
    ? 'brics'
    : dashboard?.focusTemplate === 'us-equities'
    ? 'nato'
    : 'global';

  const { briefings, metadata, loading, refresh } = useIntelBriefing(preset, { autoFetch: true });

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refresh();
    setLastRefresh(new Date());
    setIsRefreshing(false);
  };

  if (!dashboard) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
        <LayoutDashboard className="w-16 h-16 text-slate-600 mb-4" />
        <h1 className="text-2xl font-bold text-white mb-2">Dashboard Not Found</h1>
        <p className="text-slate-400 mb-6">
          The dashboard you're looking for doesn't exist or has been deleted.
        </p>
        <GlassButton variant="primary" onClick={() => router.push('/app/dashboards')}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Dashboard Hub
        </GlassButton>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/app/dashboards')}
            className="p-2 rounded-lg hover:bg-white/[0.06] text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-cyan-500/20 rounded-xl text-cyan-400">
              {PRESET_ICONS[dashboard.presetId] || <LayoutDashboard className="w-5 h-5" />}
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">{dashboard.name}</h1>
              <p className="text-sm text-slate-400">{dashboard.description}</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <GlassButton
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span className="ml-2 hidden sm:inline">Refresh</span>
          </GlassButton>
          <GlassButton variant="ghost" size="sm">
            <Share2 className="w-4 h-4" />
            <span className="ml-2 hidden sm:inline">Share</span>
          </GlassButton>
          <GlassButton variant="ghost" size="sm">
            <Download className="w-4 h-4" />
            <span className="ml-2 hidden sm:inline">Export</span>
          </GlassButton>
          <GlassButton variant="secondary" size="sm">
            <Settings className="w-4 h-4" />
            <span className="ml-2 hidden sm:inline">Configure</span>
          </GlassButton>
        </div>
      </div>

      {/* Status Bar */}
      <div className="flex items-center gap-6 text-xs text-slate-400 px-1">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4" />
          Last updated: {lastRefresh.toLocaleTimeString()}
        </div>
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          {metadata?.sources || 4} sources active
        </div>
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-green-400" />
          Live data streaming
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Primary Widget - 2 cols */}
        <GlassCard blur="heavy" className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              Situational Overview
            </h2>
            <span className="text-xs text-slate-500">
              {loading ? 'Updating...' : 'Real-time'}
            </span>
          </div>

          {loading ? (
            <div className="animate-pulse space-y-3">
              <div className="h-4 bg-white/5 rounded w-3/4"></div>
              <div className="h-4 bg-white/5 rounded w-1/2"></div>
              <div className="h-4 bg-white/5 rounded w-2/3"></div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-slate-300 leading-relaxed">
                {briefings?.lead || 'Loading intelligence briefing...'}
              </p>
              {briefings?.context && (
                <p className="text-sm text-slate-400 leading-relaxed">
                  {briefings.context}
                </p>
              )}
            </div>
          )}
        </GlassCard>

        {/* Risk Indicators */}
        <GlassCard blur="heavy">
          <h3 className="text-sm font-medium text-slate-400 mb-4">Risk Indicators</h3>
          <div className="space-y-3">
            <RiskIndicator
              label="Geopolitical"
              value={metadata?.risk_level === 'elevated' ? 72 : 45}
              color="amber"
            />
            <RiskIndicator
              label="Economic"
              value={dashboard.focusTemplate === 'us-equities' ? 58 : 34}
              color="blue"
            />
            <RiskIndicator
              label="Security"
              value={metadata?.risk_level === 'high' ? 85 : 42}
              color="red"
            />
            <RiskIndicator
              label="Market"
              value={dashboard.focusTemplate === 'us-equities' ? 67 : 28}
              color="purple"
            />
          </div>
        </GlassCard>

        {/* Market Signals Widget */}
        <GlassCard blur="heavy">
          <h3 className="text-sm font-medium text-slate-400 mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            Market Signals
          </h3>
          {loading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-white/5 rounded w-full"></div>
              <div className="h-4 bg-white/5 rounded w-3/4"></div>
            </div>
          ) : (
            <p className="text-sm text-slate-300 leading-relaxed">
              {briefings?.markets || 'Market data not available for this dashboard configuration.'}
            </p>
          )}
        </GlassCard>

        {/* Key Developments */}
        <GlassCard blur="heavy">
          <h3 className="text-sm font-medium text-slate-400 mb-4 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            Key Developments
          </h3>
          {loading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-white/5 rounded w-full"></div>
              <div className="h-4 bg-white/5 rounded w-2/3"></div>
            </div>
          ) : (
            <p className="text-sm text-slate-300 leading-relaxed">
              {briefings?.security || briefings?.economic || 'No significant developments at this time.'}
            </p>
          )}
        </GlassCard>

        {/* Cascade Watch */}
        <GlassCard blur="heavy">
          <h3 className="text-sm font-medium text-slate-400 mb-4 flex items-center gap-2">
            <Zap className="w-4 h-4 text-cyan-400" />
            Cascade Watch
          </h3>
          {loading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-white/5 rounded w-full"></div>
              <div className="h-4 bg-white/5 rounded w-1/2"></div>
            </div>
          ) : (
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-400">
                <span>Active Cascades</span>
                <span className="text-white">{metadata?.cascade_count || 3}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Risk Level</span>
                <span className={`${
                  metadata?.risk_level === 'high' ? 'text-red-400' :
                  metadata?.risk_level === 'elevated' ? 'text-amber-400' :
                  'text-green-400'
                }`}>
                  {(metadata?.risk_level || 'moderate').toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Confidence</span>
                <span className="text-white">{((metadata?.confidence || 0.75) * 100).toFixed(0)}%</span>
              </div>
            </div>
          )}
        </GlassCard>
      </div>

      {/* Widget Grid - Placeholder for actual widgets */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {dashboard.widgets.map((widget) => (
          <GlassCard key={widget} blur="light" compact>
            <div className="text-center py-6">
              <div className="w-10 h-10 mx-auto mb-3 rounded-lg bg-white/5 flex items-center justify-center">
                <LayoutDashboard className="w-5 h-5 text-slate-500" />
              </div>
              <p className="text-sm text-slate-400">
                {widget.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </p>
              <p className="text-xs text-slate-600 mt-1">Widget placeholder</p>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
}

function RiskIndicator({
  label,
  value,
  color
}: {
  label: string;
  value: number;
  color: 'amber' | 'blue' | 'red' | 'purple' | 'green';
}) {
  const colorClasses = {
    amber: 'bg-amber-500',
    blue: 'bg-blue-500',
    red: 'bg-red-500',
    purple: 'bg-purple-500',
    green: 'bg-green-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-400">{label}</span>
        <span className="text-white">{value}%</span>
      </div>
      <div className="h-1.5 bg-black/30 rounded-full overflow-hidden">
        <div
          className={`h-full ${colorClasses[color]} transition-all duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}
