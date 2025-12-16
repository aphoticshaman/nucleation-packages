import { requireEnterprise, createClient } from '@/lib/auth';
import { Card } from '@/components/ui';
import { TrendingUp, BarChart3, Calendar } from 'lucide-react';

export default async function UsagePage() {
  const user = await requireEnterprise();
  const supabase = await createClient();

  const { data: org } = await supabase
    .from('organizations')
    .select('*')
    .eq('id', user.organization_id)
    .single();

  const usagePercent = org ? (org.api_calls_used / org.api_calls_limit) * 100 : 0;

  return (
    <div>
      <div className="mb-6 md:mb-8">
        <h1 className="text-lg font-bold text-white">Usage & Analytics</h1>
        <p className="text-slate-400 text-sm md:text-base">
          Monitor your API usage and performance metrics
        </p>
      </div>

      {/* Usage Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6 mb-6 md:mb-8">
        <Card className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <BarChart3 className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">API Calls This Month</span>
          </div>
          <p className="text-2xl md:text-3xl font-bold text-white">
            {org?.api_calls_used?.toLocaleString() || 0}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            of {org?.api_calls_limit?.toLocaleString() || 0} limit
          </p>
          <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${usagePercent > 80 ? 'bg-yellow-500' : 'bg-blue-500'}`}
              style={{ width: `${Math.min(usagePercent, 100)}%` }}
            />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <TrendingUp className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Avg Response Time</span>
          </div>
          <p className="text-2xl md:text-3xl font-bold text-white">142ms</p>
          <p className="text-sm text-green-400 mt-1">↓ 8ms from last week</p>
        </Card>

        <Card className="p-4">
          <div className="flex items-center gap-3 mb-3">
            <Calendar className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Billing Period</span>
          </div>
          <p className="text-2xl md:text-3xl font-bold text-white">Day 15</p>
          <p className="text-sm text-slate-500 mt-1">Resets in 15 days</p>
        </Card>
      </div>

      {/* Usage Chart Placeholder */}
      <Card className="p-6">
        <h2 className="text-lg font-bold text-white mb-4">Usage Over Time</h2>
        <div className="h-64 flex items-center justify-center bg-black/20 rounded-md border border-white/[0.04]">
          <p className="text-slate-500">Usage chart visualization</p>
        </div>
      </Card>
    </div>
  );
}
