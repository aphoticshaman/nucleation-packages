import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { DollarSign, CreditCard, TrendingUp, Receipt } from 'lucide-react';

export default async function BillingPage() {
  await requireAdmin();

  return (
    <div className="pl-72 p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Billing</h1>
        <p className="text-slate-400">Revenue metrics and subscription management</p>
      </div>

      {/* Revenue Stats */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">MRR</span>
          </div>
          <p className="text-3xl font-bold text-white">$12,450</p>
          <p className="text-sm text-green-400 mt-1">+12% from last month</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">ARR</span>
          </div>
          <p className="text-3xl font-bold text-white">$149,400</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <CreditCard className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Active Subscriptions</span>
          </div>
          <p className="text-3xl font-bold text-white">47</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Receipt className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Avg Revenue/Customer</span>
          </div>
          <p className="text-3xl font-bold text-white">$265</p>
        </GlassCard>
      </div>

      {/* Revenue Chart Placeholder */}
      <GlassCard blur="heavy" className="mb-8">
        <h2 className="text-lg font-bold text-white mb-4">Revenue Over Time</h2>
        <div className="h-64 flex items-center justify-center bg-black/20 rounded-xl border border-white/[0.04]">
          <p className="text-slate-500">Revenue chart visualization</p>
        </div>
      </GlassCard>

      {/* Recent Transactions */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">Recent Transactions</h2>
        <div className="space-y-3">
          {[
            { customer: 'Acme Corp', amount: 499, type: 'subscription', date: '2024-03-15' },
            { customer: 'Tech Startup', amount: 199, type: 'subscription', date: '2024-03-14' },
            { customer: 'Finance Inc', amount: 999, type: 'enterprise', date: '2024-03-13' },
          ].map((tx, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]">
              <div>
                <p className="text-white font-medium">{tx.customer}</p>
                <p className="text-sm text-slate-400">{tx.type} - {tx.date}</p>
              </div>
              <p className="text-green-400 font-bold">+${tx.amount}</p>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
