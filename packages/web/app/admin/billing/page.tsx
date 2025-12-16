import { requireAdmin } from '@/lib/auth';
import { Card } from '@/components/ui';
import { DollarSign, CreditCard, TrendingUp, Receipt } from 'lucide-react';

export default async function BillingPage() {
  await requireAdmin();

  return (
    <div className="p-4 lg:pl-72 lg:p-8">
      <div className="mb-6 lg:mb-8">
        <h1 className="text-lg font-bold text-white">Billing</h1>
        <p className="text-slate-400 text-sm lg:text-base">Revenue metrics and subscription management</p>
      </div>

      {/* Revenue Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-6 lg:mb-8">
        <Card>
          <div className="flex items-center gap-3 mb-2">
            <DollarSign className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">MRR</span>
          </div>
          <p className="text-3xl font-bold text-white">$12,450</p>
          <p className="text-sm text-green-400 mt-1">+12% from last month</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">ARR</span>
          </div>
          <p className="text-3xl font-bold text-white">$149,400</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <CreditCard className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Active Subscriptions</span>
          </div>
          <p className="text-3xl font-bold text-white">47</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Receipt className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Avg Revenue/Customer</span>
          </div>
          <p className="text-3xl font-bold text-white">$265</p>
        </Card>
      </div>

      {/* Revenue Chart Placeholder */}
      <Card className="mb-8">
        <h2 className="text-lg font-bold text-white mb-4">Revenue Over Time</h2>
        <div className="h-64 flex items-center justify-center bg-black/20 rounded-md border border-white/[0.04]">
          <p className="text-slate-500">Revenue chart visualization</p>
        </div>
      </Card>

      {/* Recent Transactions */}
      <Card>
        <h2 className="text-lg font-bold text-white mb-4">Recent Transactions</h2>
        <div className="space-y-3">
          {[
            { customer: 'Acme Corp', amount: 499, type: 'subscription', date: '2024-03-15' },
            { customer: 'Tech Startup', amount: 199, type: 'subscription', date: '2024-03-14' },
            { customer: 'Finance Inc', amount: 999, type: 'enterprise', date: '2024-03-13' },
          ].map((tx, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-md border border-white/[0.04]">
              <div>
                <p className="text-white font-medium">{tx.customer}</p>
                <p className="text-sm text-slate-400">{tx.type} - {tx.date}</p>
              </div>
              <p className="text-green-400 font-bold">+${tx.amount}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
