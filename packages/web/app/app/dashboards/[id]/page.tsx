'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState } from 'react';
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
    : 'global'; // US markets uses global preset for broader coverage

  const { briefings, metadata, loading, refetch } = useIntelBriefing(preset, { autoFetch: true });

  // Determine which briefing domains to show based on dashboard focus
  const getRelevantDomains = () => {
    if (dashboard?.focusTemplate === 'us-equities') {
      return {
        primary: ['markets', 'financial', 'economic'],
        secondary: ['employment', 'housing', 'crypto', 'energy'],
      };
    }
    if (dashboard?.focusTemplate === 'china-watch') {
      return {
        primary: ['political', 'economic', 'security'],
        secondary: ['military', 'cyber', 'industry', 'minerals'],
      };
    }
    // Global/default
    return {
      primary: ['political', 'security', 'economic'],
      secondary: ['military', 'cyber', 'resources', 'emerging'],
    };
  };

  const domains = getRelevantDomains();

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetch();
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
            onClick={() => void handleRefresh()}
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
          4 sources active
        </div>
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-green-400" />
          Live data streaming
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Primary Widget - 2 cols - Deep Dive Content */}
        <GlassCard blur="heavy" className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              {dashboard.focusTemplate === 'us-equities' ? 'Market Intelligence Deep Dive' :
               dashboard.focusTemplate === 'china-watch' ? 'Strategic Analysis Deep Dive' :
               'Geopolitical Intelligence Deep Dive'}
            </h2>
            <span className="text-xs text-slate-500">
              {loading ? 'Updating...' : 'Real-time Analysis'}
            </span>
          </div>

          {loading ? (
            <div className="animate-pulse space-y-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="h-3 bg-white/5 rounded w-24"></div>
                  <div className="h-4 bg-white/5 rounded w-full"></div>
                  <div className="h-4 bg-white/5 rounded w-5/6"></div>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-5">
              {/* Executive Summary */}
              <div>
                <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider mb-2">
                  Executive Summary
                </h4>
                <p className="text-slate-300 leading-relaxed">
                  {briefings?.summary || 'Loading intelligence briefing...'}
                </p>
              </div>

              {/* Primary Domain Briefings */}
              {domains.primary.map((domain) => {
                const content = briefings?.[domain as keyof typeof briefings];
                if (!content) return null;
                const domainLabels: Record<string, string> = {
                  markets: 'Market Dynamics',
                  financial: 'Financial Systems',
                  economic: 'Economic Indicators',
                  political: 'Political Landscape',
                  security: 'Security Environment',
                  military: 'Military Assessment',
                  employment: 'Labor Markets',
                  housing: 'Real Estate & Housing',
                  crypto: 'Digital Assets',
                  energy: 'Energy Sector',
                  cyber: 'Cyber Threats',
                  industry: 'Industrial Activity',
                  minerals: 'Critical Minerals',
                };
                return (
                  <div key={domain} className="border-t border-white/5 pt-4">
                    <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
                      {domainLabels[domain] || domain}
                    </h4>
                    <p className="text-sm text-slate-300 leading-relaxed">
                      {content}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </GlassCard>

        {/* Risk Indicators */}
        <GlassCard blur="heavy">
          <h3 className="text-sm font-medium text-slate-400 mb-4">Risk Indicators</h3>
          <div className="space-y-3">
            <RiskIndicator
              label="Geopolitical"
              value={metadata?.overallRisk === 'elevated' ? 72 : 45}
              color="amber"
            />
            <RiskIndicator
              label="Economic"
              value={dashboard.focusTemplate === 'us-equities' ? 58 : 34}
              color="blue"
            />
            <RiskIndicator
              label="Security"
              value={metadata?.overallRisk === 'high' ? 85 : 42}
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
                <span className="text-white">3</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Risk Level</span>
                <span className={`${
                  metadata?.overallRisk === 'high' || metadata?.overallRisk === 'critical' ? 'text-red-400' :
                  metadata?.overallRisk === 'elevated' ? 'text-amber-400' :
                  'text-green-400'
                }`}>
                  {(metadata?.overallRisk || 'moderate').toUpperCase()}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Confidence</span>
                <span className="text-white">75%</span>
              </div>
            </div>
          )}
        </GlassCard>
      </div>

      {/* Secondary Intelligence Domains */}
      <GlassCard blur="heavy">
        <h3 className="text-sm font-medium text-slate-400 mb-4">
          Extended Analysis
        </h3>
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="animate-pulse space-y-2 p-4 bg-white/[0.02] rounded-lg">
                <div className="h-3 bg-white/5 rounded w-20"></div>
                <div className="h-4 bg-white/5 rounded w-full"></div>
                <div className="h-4 bg-white/5 rounded w-3/4"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {domains.secondary.map((domain) => {
              const content = briefings?.[domain as keyof typeof briefings];
              if (!content) return null;
              const domainLabels: Record<string, { label: string; icon: string }> = {
                employment: { label: 'Labor Markets', icon: '👥' },
                housing: { label: 'Real Estate', icon: '🏠' },
                crypto: { label: 'Digital Assets', icon: '₿' },
                energy: { label: 'Energy Sector', icon: '⚡' },
                military: { label: 'Military', icon: '🎖️' },
                cyber: { label: 'Cyber', icon: '🔒' },
                industry: { label: 'Industry', icon: '🏭' },
                minerals: { label: 'Critical Minerals', icon: '⛏️' },
                resources: { label: 'Resources', icon: '🌍' },
                emerging: { label: 'Emerging Tech', icon: '🔬' },
              };
              const meta = domainLabels[domain] || { label: domain, icon: '📊' };
              return (
                <div key={domain} className="p-4 bg-white/[0.02] rounded-lg border border-white/5">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-lg">{meta.icon}</span>
                    <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                      {meta.label}
                    </h4>
                  </div>
                  <p className="text-sm text-slate-300 leading-relaxed">
                    {content}
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </GlassCard>

      {/* Next Strategic Move */}
      {briefings?.nsm && (
        <GlassCard blur="heavy" className="border-l-4 border-cyan-500">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <Zap className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-2">
                Next Strategic Move
              </h3>
              <p className="text-slate-300 leading-relaxed">
                {briefings.nsm}
              </p>
            </div>
          </div>
        </GlassCard>
      )}
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
