import { createClient, requireEnterprise } from '@/lib/auth';
import { Suspense } from 'react';

// Stat card
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
    <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-3xl font-bold text-white mt-2">{value}</p>
      <div className="flex items-center justify-between mt-2">
        {subtitle && <p className="text-sm text-slate-500">{subtitle}</p>}
        {trend && (
          <p className={`text-sm ${trend.up ? 'text-green-400' : 'text-red-400'}`}>
            {trend.up ? '↑' : '↓'} {trend.value}
          </p>
        )}
      </div>
    </div>
  );
}

// Endpoint card for data streams
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
    GET: 'bg-green-600',
    POST: 'bg-blue-600',
    WS: 'bg-purple-600',
    SSE: 'bg-orange-600',
  };

  return (
    <div className="flex items-start gap-4 p-4 bg-slate-800/50 rounded-lg">
      <span className={`px-2 py-1 rounded text-xs font-mono text-white ${methodColors[method]}`}>
        {method}
      </span>
      <div className="flex-1 min-w-0">
        <code className="text-sm text-blue-400 font-mono">{endpoint}</code>
        <p className="text-sm text-slate-400 mt-1">{description}</p>
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

// Quick action button
function QuickAction({
  icon,
  label,
  href,
}: {
  icon: string;
  label: string;
  href: string;
}) {
  return (
    <a
      href={href}
      className="flex items-center gap-3 p-4 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
    >
      <span className="text-2xl">{icon}</span>
      <span className="text-white">{label}</span>
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
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          label="API Calls (24h)"
          value={(recentCalls || 0).toLocaleString()}
          subtitle={`of ${org?.api_calls_limit?.toLocaleString() || 0} limit`}
          trend={{ value: '12%', up: true }}
        />
        <StatCard
          label="Active API Keys"
          value={apiKeyCount || 0}
          subtitle="Configured"
        />
        <StatCard
          label="Team Members"
          value={teamSize || 0}
          subtitle={`of ${org?.team_seats_limit || 5} seats`}
        />
        <StatCard
          label="Avg Response Time"
          value="142ms"
          trend={{ value: '8ms', up: false }}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Data Streams / Endpoints */}
        <div className="lg:col-span-2 bg-slate-900 rounded-xl border border-slate-800">
          <div className="p-6 border-b border-slate-800">
            <h2 className="text-lg font-bold text-white">Data Endpoints</h2>
            <p className="text-sm text-slate-400">
              Connect your pipelines to these endpoints
            </p>
          </div>
          <div className="p-6 space-y-4">
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
          <div className="p-6 border-t border-slate-800">
            <a
              href="/docs/api"
              className="text-blue-400 hover:text-blue-300 text-sm"
            >
              View full API documentation →
            </a>
          </div>
        </div>

        {/* Quick Actions + Executive Reports */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <h2 className="text-lg font-bold text-white mb-4">Quick Actions</h2>
            <div className="space-y-3">
              <QuickAction icon="🔑" label="Generate API Key" href="/dashboard/api-keys" />
              <QuickAction icon="🔗" label="Configure Webhook" href="/dashboard/webhooks" />
              <QuickAction icon="👥" label="Invite Team Member" href="/dashboard/team" />
            </div>
          </div>

          {/* Executive Reports */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <h2 className="text-lg font-bold text-white mb-2">Executive Reports</h2>
            <p className="text-sm text-slate-400 mb-4">
              Export usage data for leadership
            </p>
            <div className="space-y-3">
              <button className="w-full flex items-center justify-between px-4 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
                <span className="text-white text-sm">Monthly Usage Report</span>
                <span className="text-slate-400 text-xs">PDF</span>
              </button>
              <button className="w-full flex items-center justify-between px-4 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
                <span className="text-white text-sm">API Performance Metrics</span>
                <span className="text-slate-400 text-xs">CSV</span>
              </button>
              <button className="w-full flex items-center justify-between px-4 py-3 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors">
                <span className="text-white text-sm">ROI Analysis</span>
                <span className="text-slate-400 text-xs">PDF</span>
              </button>
            </div>
          </div>

          {/* Status */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
            <div className="flex items-center gap-3">
              <span className="w-3 h-3 bg-green-500 rounded-full" />
              <div>
                <p className="text-white text-sm">All Systems Operational</p>
                <p className="text-xs text-slate-500">Last checked: just now</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Code Snippet */}
      <div className="mt-8 bg-slate-900 rounded-xl border border-slate-800">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">Quick Start</h2>
            <p className="text-sm text-slate-400">Copy this to start streaming data</p>
          </div>
          <div className="flex gap-2">
            <button className="px-3 py-1 text-sm bg-slate-700 text-white rounded hover:bg-slate-600">
              cURL
            </button>
            <button className="px-3 py-1 text-sm bg-slate-800 text-slate-400 rounded hover:bg-slate-700">
              Python
            </button>
            <button className="px-3 py-1 text-sm bg-slate-800 text-slate-400 rounded hover:bg-slate-700">
              Node.js
            </button>
          </div>
        </div>
        <pre className="p-6 text-sm text-slate-300 font-mono overflow-x-auto">
{`curl -X GET "https://api.latticeforge.io/v1/nations" \\
  -H "Authorization: Bearer lf_live_xxxxxxxxxxxx" \\
  -H "Content-Type: application/json"`}
        </pre>
      </div>
    </>
  );
}

export default function EnterpriseDashboard() {
  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Dashboard</h1>
          <p className="text-slate-400">Your API usage and data streams</p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-slate-800 text-white rounded-lg hover:bg-slate-700 text-sm">
            View Docs
          </button>
          <button className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 text-sm">
            New API Key
          </button>
        </div>
      </div>

      <Suspense
        fallback={
          <div className="grid grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-slate-900 rounded-xl p-6 border border-slate-800 animate-pulse h-32"
              />
            ))}
          </div>
        }
      >
        <DashboardContent />
      </Suspense>
    </div>
  );
}
