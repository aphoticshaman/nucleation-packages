import { requireEnterprise } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Link2, Plus, CheckCircle, XCircle } from 'lucide-react';

export default async function WebhooksPage() {
  await requireEnterprise();

  return (
    <div>
      <div className="mb-6 md:mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white">Webhooks</h1>
          <p className="text-slate-400 text-sm md:text-base">
            Configure endpoints to receive real-time event notifications
          </p>
        </div>
        <GlassButton variant="primary" glow>
          <Plus className="w-4 h-4 mr-2" />
          Add Webhook
        </GlassButton>
      </div>

      <GlassCard blur="heavy">
        <div className="space-y-4">
          {/* Example Webhook */}
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-green-500/10 flex items-center justify-center">
                <Link2 className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <p className="text-white font-medium">Production Webhook</p>
                <code className="text-sm text-slate-400 font-mono">https://api.yourapp.com/webhooks/lattice</code>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1 text-sm text-green-400">
                <CheckCircle className="w-4 h-4" />
                Active
              </span>
            </div>
          </div>

          {/* Inactive Webhook */}
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04] opacity-60">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-slate-500/10 flex items-center justify-center">
                <Link2 className="w-5 h-5 text-slate-400" />
              </div>
              <div>
                <p className="text-white font-medium">Test Webhook</p>
                <code className="text-sm text-slate-400 font-mono">https://webhook.site/test-123</code>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1 text-sm text-slate-400">
                <XCircle className="w-4 h-4" />
                Disabled
              </span>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-6 border-t border-white/[0.06]">
          <h3 className="text-sm font-medium text-white mb-2">Available Events</h3>
          <div className="flex flex-wrap gap-2">
            {['simulation.complete', 'attractor.shift', 'alert.triggered', 'export.ready'].map((event) => (
              <span key={event} className="px-2 py-1 bg-blue-500/10 text-blue-400 rounded text-xs font-mono">
                {event}
              </span>
            ))}
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
