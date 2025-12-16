import { createClient, requireEnterprise } from '@/lib/auth';
import { Suspense } from 'react';
import { Card, Button } from '@/components/ui';
import { TrendingUp, TrendingDown, Key, Link2, Users, FileText, Share2, Activity, Zap, type LucideIcon } from 'lucide-react';
import Link from 'next/link';

// Stat card - responsive with glass design
function StatCard({
  label,
  value,
  subtitle,
  trend,
}: {
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: { value: string; up: boolean };
}) {
  return (
    <Card className="p-4">
      <p className="text-xs md:text-sm text-slate-400">{label}</p>
      <p className="text-2xl md:text-3xl font-bold text-white mt-1.5">
        {value}
      </p>
      <div className="flex items-center justify-between mt-1.5 gap-2">
        {subtitle && (
          <p className="text-xs md:text-sm text-slate-500 truncate">{subtitle}</p>
        )}
        {trend && (
          <p className={`text-xs md:text-sm shrink-0 flex items-center gap-1 ${trend.up ? 'text-green-400' : 'text-red-400'}`}>
            {trend.up ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            {trend.value}
          </p>
        )}
      </div>
    </Card>
  );
}

// Endpoint card for data streams - responsive with glass design
function EndpointCard({
  method,
  endpoint,
  description,
  streaming,
}: {
  method: 'GET' | 'POST' | 'WS' | 'SSE';
  endpoint: string;
  description: string;
  streaming?: boolean;
}) {
  const methodColors = {
    GET: 'bg-green-500/20 text-green-400 border-green-500/30',
    POST: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    WS: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    SSE: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  };

  return (
    <div className="flex flex-col sm:flex-row items-start gap-3 md:gap-4 p-3 md:p-4 bg-black/20 rounded-md border border-white/[0.04]">
      <span className={`px-2 py-1 rounded-md text-xs font-mono border ${methodColors[method]}`}>
        {method}
      </span>
      <div className="flex-1 min-w-0">
        <code className="text-xs md:text-sm text-blue-400 font-mono break-all">
          {endpoint}
        </code>
        <p className="text-xs md:text-sm text-slate-400 mt-1">{description}</p>
        {streaming && (
          <span className="inline-flex items-center gap-1 mt-2 text-xs text-emerald-400">
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            Real-time streaming
          </span>
        )}
      </div>
    </div>
  );
}

// Quick action button - responsive with glass design
function QuickAction({ icon: Icon, label, href }: { icon: LucideIcon; label: string; href: string }) {
  return (
    <a
      href={href}
      className="flex items-center gap-3 p-3 md:p-4 bg-black/20 hover:bg-black/30 border border-white/[0.04] hover:border-white/[0.08] rounded-md transition-all"
    >
      <div className="w-10 h-10 rounded-md bg-blue-500/10 flex items-center justify-center">
        <Icon className="w-5 h-5 text-blue-400" />
      </div>
      <span className="text-white text-sm md:text-base">{label}</span>
    </a>
  );
}

async function DashboardContent() {
  const user = await requireEnterprise();
  const supabase = await createClient();

  // Get org and usage data
  const { data: org } = await supabase
    .from('organizations')
    .select('*')
    .eq('id', user.organization_id)
    .single();

  // Get API keys count
  const { count: apiKeyCount } = await supabase
    .from('api_keys')
    .select('*', { count: 'exact', head: true })
    .eq('organization_id', user.organization_id)
    .eq('is_active', true);

  // Get team size
  const { count: teamSize } = await supabase
    .from('profiles')
    .select('*', { count: 'exact', head: true })
    .eq('organization_id', user.organization_id);

  // Get recent API calls
  const { count: recentCalls } = await supabase
    .from('api_usage')
    .select('*', { count: 'exact', head: true })
    .eq('organization_id', user.organization_id)
    .gte('created_at', new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString());

  return (
    <>
      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4 lg:gap-6 mb-6 md:mb-8">
        <StatCard
          label="API Calls (24h)"
          value={(recentCalls || 0).toLocaleString()}
          subtitle={`of ${org?.api_calls_limit?.toLocaleString() || 0} limit`}
          trend={{ value: '12%', up: true }}
        />
        <StatCard label="Active API Keys" value={apiKeyCount || 0} subtitle="Configured" />
        <StatCard
          label="Team Members"
          value={teamSize || 0}
          subtitle={`of ${org?.team_seats_limit || 5} seats`}
        />
        <StatCard label="Avg Response" value="142ms" trend={{ value: '8ms', up: false }} />
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6 lg:gap-8">
        {/* Data Streams / Endpoints */}
        <Card className="lg:col-span-2 p-6">
          <div className="border-b border-white/[0.06] pb-4 mb-4">
            <h2 className="text-base md:text-lg font-bold text-white">Data Endpoints</h2>
            <p className="text-xs md:text-sm text-slate-400 mt-0.5">
              Connect your pipelines to these endpoints
            </p>
          </div>
          <div className="space-y-3 md:space-y-4">
            <EndpointCard
              method="GET"
              endpoint="/api/v1/nations"
              description="Fetch all nation attractor states"
            />
            <EndpointCard
              method="POST"
              endpoint="/api/v1/simulate"
              description="Run simulation step and get results"
            />
            <EndpointCard
              method="SSE"
              endpoint="/api/v1/stream/attractors"
              description="Real-time attractor position updates"
              streaming
            />
            <EndpointCard
              method="WS"
              endpoint="/api/v1/ws/simulation"
              description="WebSocket for bidirectional simulation control"
              streaming
            />
            <EndpointCard
              method="GET"
              endpoint="/api/v1/export/geojson"
              description="Export current state as GeoJSON for mapping"
            />
          </div>
          <div className="pt-4 mt-4 border-t border-white/[0.06]">
            <a href="/docs/api" className="text-blue-400 hover:text-blue-300 text-sm">
              View full API documentation →
            </a>
          </div>
        </Card>

        {/* Quick Actions + Executive Reports */}
        <div className="space-y-4 md:space-y-6">
          {/* Quick Actions */}
          <Card className="p-6">
            <h2 className="text-base md:text-lg font-bold text-white mb-3 md:mb-4">
              Quick Actions
            </h2>
            <div className="space-y-2 md:space-y-3">
              <QuickAction icon={Key} label="Generate API Key" href="/dashboard/api-keys" />
              <QuickAction icon={Link2} label="Configure Webhook" href="/dashboard/webhooks" />
              <QuickAction icon={Users} label="Invite Team Member" href="/dashboard/team" />
            </div>
          </Card>

          {/* Executive Reports */}
          <Card className="p-6">
            <h2 className="text-base md:text-lg font-bold text-white mb-1 md:mb-2">
              Executive Reports
            </h2>
            <p className="text-xs md:text-sm text-slate-400 mb-3 md:mb-4">
              Export usage data for leadership
            </p>
            <div className="space-y-2 md:space-y-3">
              {[
                { label: 'Monthly Usage Report', format: 'PDF' },
                { label: 'API Performance', format: 'CSV' },
                { label: 'ROI Analysis', format: 'PDF' },
              ].map((report) => (
                <button
                  key={report.label}
                  className="w-full flex items-center justify-between px-3 md:px-4 py-2.5 md:py-3 bg-black/20 hover:bg-black/30 border border-white/[0.04] hover:border-white/[0.08] rounded-md transition-all"
                >
                  <span className="text-white text-xs md:text-sm flex items-center gap-2">
                    <FileText className="w-4 h-4 text-slate-500" />
                    {report.label}
                  </span>
                  <span className="text-slate-500 text-xs">{report.format}</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Status */}
          <Card className="p-6">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 bg-green-500 rounded-full shrink-0 animate-pulse" />
              <div>
                <p className="text-white text-sm">All Systems Operational</p>
                <p className="text-xs text-slate-500">Last checked: just now</p>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Visualization Tools */}
      <div className="mt-6 md:mt-8">
        <h2 className="text-base md:text-lg font-bold text-white mb-4">Visualization Tools</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link
            href="/dashboard/intelligence"
            className="group p-4 bg-slate-900/50 hover:bg-slate-800/50 border border-slate-700 hover:border-blue-500/50 rounded-md transition-all"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-md bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                <Activity className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-white font-medium">Intelligence Feed</h3>
            </div>
            <p className="text-xs text-slate-400">Real-time streaming signals and alerts with domain filtering</p>
          </Link>

          <Link
            href="/dashboard/causal"
            className="group p-4 bg-slate-900/50 hover:bg-slate-800/50 border border-slate-700 hover:border-cyan-500/50 rounded-md transition-all"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-md bg-cyan-500/10 flex items-center justify-center group-hover:bg-cyan-500/20 transition-colors">
                <Share2 className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="text-white font-medium">Causal Graph</h3>
            </div>
            <p className="text-xs text-slate-400">Force-directed topology with transfer entropy weighting</p>
          </Link>

          <Link
            href="/dashboard/regimes"
            className="group p-4 bg-slate-900/50 hover:bg-slate-800/50 border border-slate-700 hover:border-amber-500/50 rounded-md transition-all"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center group-hover:bg-amber-500/20 transition-colors">
                <Zap className="w-5 h-5 text-amber-400" />
              </div>
              <h3 className="text-white font-medium">Regime Detection</h3>
            </div>
            <p className="text-xs text-slate-400">Markov-switching regimes with phase transition indicators</p>
          </Link>
        </div>
      </div>

      {/* Code Snippet */}
      <Card className="mt-6 md:mt-8 p-6">
        <div className="border-b border-white/[0.06] pb-4 mb-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base md:text-lg font-bold text-white">Quick Start</h2>
            <p className="text-xs md:text-sm text-slate-400">
              Copy this to start streaming data
            </p>
          </div>
          <div className="flex gap-2 shrink-0">
            <Button variant="secondary" size="sm">cURL</Button>
            <Button variant="ghost" size="sm">Python</Button>
            <Button variant="ghost" size="sm">Node.js</Button>
          </div>
        </div>
        <pre className="text-xs md:text-sm text-slate-300 font-mono overflow-x-auto p-4 bg-black/30 rounded-md">
          {`curl -X GET "https://api.latticeforge.io/v1/nations" \\
  -H "Authorization: Bearer <your-api-key>" \\
  -H "Content-Type: application/json"`}
        </pre>
      </Card>
    </>
  );
}

export default function EnterpriseDashboard() {
  return (
    <div>
      {/* Header */}
      <div className="mb-6 md:mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-white">Dashboard</h1>
          <p className="text-slate-400 text-sm md:text-base">
            Your API usage and data streams
          </p>
        </div>
        <div className="flex gap-2 md:gap-3 w-full sm:w-auto">
          <Button variant="secondary">
            View Docs
          </Button>
          <Button variant="secondary">
            New API Key
          </Button>
        </div>
      </div>

      <Suspense
        fallback={
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-6">
            {[...Array(4)].map((_, i) => (
              <Card key={i} className="animate-pulse h-24 md:h-32">
                <span className="sr-only">Loading...</span>
              </Card>
            ))}
          </div>
        }
      >
        <DashboardContent />
      </Suspense>
    </div>
  );
}
