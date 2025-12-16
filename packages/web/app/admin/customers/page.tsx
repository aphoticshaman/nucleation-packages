import { requireAdmin, createClient } from '@/lib/auth';
import { Card } from '@/components/ui';
import { Users, Building2, TrendingUp, DollarSign } from 'lucide-react';

export default async function CustomersPage() {
  await requireAdmin();
  const supabase = await createClient();

  // Get all organizations (customers)
  const { data: organizations, count } = await supabase
    .from('organizations')
    .select('*', { count: 'exact' })
    .order('created_at', { ascending: false });

  return (
    <div className="p-4 lg:pl-72 lg:p-8">
      <div className="mb-6 lg:mb-8">
        <h1 className="text-lg font-bold text-white">Customers</h1>
        <p className="text-slate-400 text-sm lg:text-base">Manage customer organizations and subscriptions</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-6 lg:mb-8">
        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Building2 className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Total Customers</span>
          </div>
          <p className="text-3xl font-bold text-white">{count || 0}</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Users className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Active This Month</span>
          </div>
          <p className="text-3xl font-bold text-white">{organizations?.filter(o => o.api_calls_used > 0).length || 0}</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Enterprise</span>
          </div>
          <p className="text-3xl font-bold text-white">{organizations?.filter(o => o.plan === 'enterprise').length || 0}</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">MRR</span>
          </div>
          <p className="text-3xl font-bold text-white">$12.4k</p>
        </Card>
      </div>

      {/* Customer List */}
      <Card>
        <h2 className="text-lg font-bold text-white mb-4">All Customers</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-sm text-slate-400 border-b border-white/[0.06]">
                <th className="pb-3 font-medium">Organization</th>
                <th className="pb-3 font-medium">Plan</th>
                <th className="pb-3 font-medium">API Usage</th>
                <th className="pb-3 font-medium">Created</th>
                <th className="pb-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {organizations?.map((org) => (
                <tr key={org.id} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                  <td className="py-4">
                    <div>
                      <p className="text-white font-medium">{org.name}</p>
                      <p className="text-slate-500 text-xs">{org.id}</p>
                    </div>
                  </td>
                  <td className="py-4">
                    <span className={`px-2 py-1 rounded text-xs uppercase ${
                      org.plan === 'enterprise' ? 'bg-purple-500/20 text-purple-400' :
                      org.plan === 'pro' ? 'bg-blue-500/20 text-blue-400' :
                      'bg-slate-500/20 text-slate-400'
                    }`}>
                      {org.plan || 'free'}
                    </span>
                  </td>
                  <td className="py-4 text-slate-300">
                    {org.api_calls_used?.toLocaleString() || 0} / {org.api_calls_limit?.toLocaleString() || 0}
                  </td>
                  <td className="py-4 text-slate-400">
                    {new Date(org.created_at).toLocaleDateString()}
                  </td>
                  <td className="py-4">
                    <span className="flex items-center gap-1 text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full" />
                      Active
                    </span>
                  </td>
                </tr>
              )) || (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-400">
                    No customers found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
