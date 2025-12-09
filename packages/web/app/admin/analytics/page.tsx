import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { BarChart3, Users, Activity, Globe } from 'lucide-react';

export default async function AnalyticsPage() {
  await requireAdmin();

  return (
    <div className="p-4 lg:pl-72 lg:p-8">
      <div className="mb-6 lg:mb-8">
        <h1 className="text-xl lg:text-2xl font-bold text-white">Analytics</h1>
        <p className="text-slate-400 text-sm lg:text-base">Platform usage and engagement metrics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-6 lg:mb-8">
        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Users className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Daily Active Users</span>
          </div>
          <p className="text-3xl font-bold text-white">1,247</p>
          <p className="text-sm text-green-400 mt-1">+8% from yesterday</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">API Requests (24h)</span>
          </div>
          <p className="text-3xl font-bold text-white">847K</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Avg Session Duration</span>
          </div>
          <p className="text-3xl font-bold text-white">12m 34s</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Globe className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Countries</span>
          </div>
          <p className="text-3xl font-bold text-white">47</p>
        </GlassCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
        {/* Usage Chart */}
        <GlassCard blur="heavy">
          <h2 className="text-lg font-bold text-white mb-4">API Usage Trends</h2>
          <div className="h-64 flex items-center justify-center bg-black/20 rounded-xl border border-white/[0.04]">
            <p className="text-slate-500">Usage chart visualization</p>
          </div>
        </GlassCard>

        {/* Top Endpoints */}
        <GlassCard blur="heavy">
          <h2 className="text-lg font-bold text-white mb-4">Top Endpoints</h2>
          <div className="space-y-3">
            {[
              { endpoint: '/api/v1/nations', calls: '312K', pct: 37 },
              { endpoint: '/api/v1/simulate', calls: '189K', pct: 22 },
              { endpoint: '/api/v1/stream/attractors', calls: '156K', pct: 18 },
              { endpoint: '/api/v1/export/geojson', calls: '98K', pct: 12 },
              { endpoint: '/api/v1/ws/simulation', calls: '92K', pct: 11 },
            ].map((ep, i) => (
              <div key={i} className="p-3 bg-black/20 rounded-xl border border-white/[0.04]">
                <div className="flex justify-between mb-2">
                  <code className="text-sm text-blue-400">{ep.endpoint}</code>
                  <span className="text-sm text-slate-400">{ep.calls}</span>
                </div>
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500" style={{ width: `${ep.pct}%` }} />
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
