import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { BarChart3, Construction } from 'lucide-react';

export default async function AnalyticsPage() {
  await requireAdmin();

  return (
    <div className="p-4 lg:pl-72 lg:p-8">
      <div className="mb-6 lg:mb-8">
        <h1 className="text-xl lg:text-2xl font-bold text-white">Analytics</h1>
        <p className="text-slate-400 text-sm lg:text-base">Platform usage and engagement metrics</p>
      </div>

      <GlassCard blur="heavy" className="max-w-2xl">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="w-16 h-16 rounded-full bg-amber-500/10 flex items-center justify-center mb-4">
            <Construction className="w-8 h-8 text-amber-400" />
          </div>
          <h2 className="text-xl font-bold text-white mb-2">Analytics Coming Soon</h2>
          <p className="text-slate-400 max-w-md">
            Usage analytics will be available in a future release. This feature is currently under development.
          </p>
          <div className="mt-6 flex items-center gap-2 text-sm text-slate-500">
            <BarChart3 className="w-4 h-4" />
            <span>Tracking API requests, user sessions, and endpoint usage</span>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
