import { createClient } from '@/lib/auth';
import { Suspense } from 'react';
import Link from 'next/link';
import { GlassCard } from '@/components/ui/GlassCard';

function StatCard({
  label,
  value,
  change,
  changeType = 'neutral',
}: {
  label: string;
  value: string | number;
  change?: string;
  changeType?: 'up' | 'down' | 'neutral';
}) {
  const changeColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    neutral: 'text-slate-400',
  };

  return (
    <GlassCard blur="heavy">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="text-3xl font-bold text-white mt-2">{value}</p>
      {change && <p className={`text-sm mt-2 ${changeColors[changeType]}`}>{change}</p>}
    </GlassCard>
  );
}

function StatusBadge({ status }: { status: 'healthy' | 'warning' | 'critical' }) {
  const colors = {
    healthy: 'bg-green-500',
    warning: 'bg-yellow-500',
    critical: 'bg-red-500 animate-pulse',
  };

  return <span className={`w-2 h-2 rounded-full ${colors[status]}`} />;
}

function EnterpriseRow({
  org,
}: {
  org: {
    name: string;
    plan: string;
    api_calls_24h: number;
    api_calls_limit: number;
    team_size: number;
    last_activity: string | null;
    is_active: boolean;
  };
}) {
  const usagePercent = (org.api_calls_24h / org.api_calls_limit) * 100;
  const status = !org.is_active ? 'critical' : usagePercent > 80 ? 'warning' : 'healthy';

  return (
    <tr className="border-b border-white/[0.06] hover:bg-white/[0.02]">
      <td className="py-4 px-4">
        <div className="flex items-center gap-3">
          <StatusBadge status={status} />
          <span className="text-white font-medium">{org.name}</span>
        </div>
      </td>
      <td className="py-4 px-4">
        <span className="px-2 py-1 bg-white/[0.06] rounded text-xs text-slate-300 uppercase">
          {org.plan}
        </span>
      </td>
      <td className="py-4 px-4">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-white/[0.06] rounded-full overflow-hidden max-w-[100px]">
            <div
              className={`h-full ${usagePercent > 80 ? 'bg-yellow-500' : 'bg-blue-500'}`}
              style={{ width: `${Math.min(usagePercent, 100)}%` }}
            />
          </div>
          <span className="text-sm text-slate-400">
            {org.api_calls_24h.toLocaleString()} / {org.api_calls_limit.toLocaleString()}
          </span>
        </div>
      </td>
      <td className="py-4 px-4 text-slate-400">{org.team_size}</td>
      <td className="py-4 px-4 text-slate-400 text-sm">
        {org.last_activity ? new Date(org.last_activity).toLocaleDateString() : 'Never'}
      </td>
    </tr>
  );
}

function ConsumerRow({
  user,
}: {
  user: {
    email: string;
    full_name: string | null;
    simulation_count: number;
    actions_7d: number;
    last_seen_at: string | null;
    is_active: boolean;
  };
}) {
  const isOnline =
    user.last_seen_at && new Date(user.last_seen_at) > new Date(Date.now() - 15 * 60 * 1000);

  return (
    <tr className="border-b border-white/[0.06] hover:bg-white/[0.02]">
      <td className="py-3 px-4">
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-slate-600'}`} />
          <div>
            <p className="text-white text-sm">{user.full_name || user.email}</p>
            {user.full_name && <p className="text-xs text-slate-500">{user.email}</p>}
          </div>
        </div>
      </td>
      <td className="py-3 px-4 text-slate-400 text-sm">{user.simulation_count}</td>
      <td className="py-3 px-4 text-slate-400 text-sm">{user.actions_7d}</td>
      <td className="py-3 px-4 text-slate-400 text-sm">
        {user.last_seen_at ? new Date(user.last_seen_at).toLocaleString() : 'Never'}
      </td>
    </tr>
  );
}

async function DashboardContent() {
  const supabase = await createClient();

  const { data: stats } = await supabase.from('admin_dashboard_stats').select('*').single();

  const { data: enterprises } = await supabase
    .from('admin_enterprise_overview')
    .select('*')
    .order('api_calls_24h', { ascending: false })
    .limit(10);

  const { data: consumers } = await supabase
    .from('admin_consumer_overview')
    .select('*')
    .order('last_seen_at', { ascending: false, nullsFirst: false })
    .limit(10);

  return (
    <>
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          label="Total Consumers"
          value={stats?.total_consumers || 0}
          change="+12% this week"
          changeType="up"
        />
        <StatCard
          label="Enterprise Orgs"
          value={stats?.total_enterprise || 0}
          change={`${stats?.active_orgs || 0} active`}
        />
        <StatCard
          label="Active (24h)"
          value={stats?.active_24h || 0}
          change={`${stats?.active_7d || 0} this week`}
        />
        <StatCard label="API Calls (24h)" value={(stats?.api_calls_24h || 0).toLocaleString()} />
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Enterprise Customers */}
        <GlassCard blur="heavy" className="p-0 overflow-hidden">
          <div className="p-6 border-b border-white/[0.06] flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white">Enterprise Customers</h2>
              <p className="text-sm text-slate-400">API usage and status</p>
            </div>
            <Link
              href="/admin/customers?type=enterprise"
              className="px-3 py-1.5 text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] rounded-lg transition-colors"
            >
              View all
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wide">
                  <th className="py-3 px-4">Organization</th>
                  <th className="py-3 px-4">Plan</th>
                  <th className="py-3 px-4">API Usage (24h)</th>
                  <th className="py-3 px-4">Team</th>
                  <th className="py-3 px-4">Last Active</th>
                </tr>
              </thead>
              <tbody>
                {enterprises?.map((org) => <EnterpriseRow key={org.id} org={org} />) || (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-slate-500">
                      No enterprise customers yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </GlassCard>

        {/* Consumer Users */}
        <GlassCard blur="heavy" className="p-0 overflow-hidden">
          <div className="p-6 border-b border-white/[0.06] flex items-center justify-between">
            <div>
              <h2 className="text-lg font-bold text-white">Consumer Users</h2>
              <p className="text-sm text-slate-400">Recent activity</p>
            </div>
            <Link
              href="/admin/customers?type=consumer"
              className="px-3 py-1.5 text-sm text-slate-300 hover:text-white hover:bg-white/[0.06] rounded-lg transition-colors"
            >
              View all
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-xs text-slate-500 uppercase tracking-wide">
                  <th className="py-3 px-4">User</th>
                  <th className="py-3 px-4">Sims</th>
                  <th className="py-3 px-4">Actions (7d)</th>
                  <th className="py-3 px-4">Last Seen</th>
                </tr>
              </thead>
              <tbody>
                {consumers?.map((user) => <ConsumerRow key={user.id} user={user} />) || (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-slate-500">
                      No consumer users yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </GlassCard>
      </div>

      {/* System Health */}
      <GlassCard blur="heavy" className="mt-8">
        <h2 className="text-lg font-bold text-white mb-4">System Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <div>
              <p className="text-white">WASM Core</p>
              <p className="text-sm text-slate-400">Operational</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <div>
              <p className="text-white">Supabase</p>
              <p className="text-sm text-slate-400">Operational</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <div>
              <p className="text-white">API Gateway</p>
              <p className="text-sm text-slate-400">Operational</p>
            </div>
          </div>
        </div>
      </GlassCard>
    </>
  );
}

export default function AdminDashboard() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400">System overview and customer status</p>
      </div>

      <Suspense
        fallback={
          <div className="grid grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-[rgba(18,18,26,0.7)] backdrop-blur-xl rounded-xl p-6 border border-white/[0.06] animate-pulse h-32"
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
