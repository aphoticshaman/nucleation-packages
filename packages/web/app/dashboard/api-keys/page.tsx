import { requireEnterprise } from '@/lib/auth';
import { Card, Button } from '@/components/ui';
import { Key, Plus, Copy, Trash2 } from 'lucide-react';

export default async function ApiKeysPage() {
  await requireEnterprise();

  return (
    <div>
      <div className="mb-6 md:mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-white">API Keys</h1>
          <p className="text-slate-400 text-sm md:text-base">
            Manage your API keys for authentication
          </p>
        </div>
        <Button variant="secondary">
          <Plus className="w-4 h-4 mr-2" />
          Generate New Key
        </Button>
      </div>

      <Card className="p-6">
        <div className="space-y-4">
          {/* Example API Key */}
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-md border border-white/[0.04]">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-md bg-blue-500/10 flex items-center justify-center">
                <Key className="w-5 h-5 text-blue-400" />
              </div>
              <div>
                <p className="text-white font-medium">Production Key</p>
                <code className="text-sm text-slate-400 font-mono">lf_live_sk_••••••••••••••••</code>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.05] rounded-md transition-all">
                <Copy className="w-4 h-4" />
              </button>
              <button className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-all">
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Test Key */}
          <div className="flex items-center justify-between p-4 bg-black/20 rounded-md border border-white/[0.04]">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-md bg-amber-500/10 flex items-center justify-center">
                <Key className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <p className="text-white font-medium">Test Key</p>
                <code className="text-sm text-slate-400 font-mono">lf_test_sk_••••••••••••••••</code>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.05] rounded-md transition-all">
                <Copy className="w-4 h-4" />
              </button>
              <button className="p-2 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-md transition-all">
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="mt-6 pt-6 border-t border-white/[0.06]">
          <p className="text-sm text-slate-400">
            API keys are used to authenticate requests to the LatticeForge API. Keep your keys secure and never share them publicly.
          </p>
        </div>
      </Card>
    </div>
  );
}
