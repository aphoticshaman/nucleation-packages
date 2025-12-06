import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Database, RefreshCw, CheckCircle, AlertCircle, Clock, Play } from 'lucide-react';

export default async function PipelinesPage() {
  await requireAdmin();

  const pipelines = [
    { name: 'Geopolitical Events Ingestion', status: 'running', lastRun: '2 min ago', schedule: 'Every 5 min' },
    { name: 'Economic Indicators ETL', status: 'completed', lastRun: '15 min ago', schedule: 'Every 15 min' },
    { name: 'Social Sentiment Analysis', status: 'completed', lastRun: '1 hour ago', schedule: 'Every hour' },
    { name: 'Attractor State Recalculation', status: 'running', lastRun: 'Running...', schedule: 'Every 10 min' },
    { name: 'Historical Data Backfill', status: 'failed', lastRun: '3 hours ago', schedule: 'Daily' },
    { name: 'Model Retraining Pipeline', status: 'pending', lastRun: 'Never', schedule: 'Weekly' },
  ];

  return (
    <div className="pl-72 p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Data Pipelines</h1>
          <p className="text-slate-400">Monitor and manage data ingestion pipelines</p>
        </div>
        <GlassButton variant="primary" glow>
          <RefreshCw className="w-4 h-4 mr-2" />
          Run All
        </GlassButton>
      </div>

      {/* Pipeline Stats */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Database className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Total Pipelines</span>
          </div>
          <p className="text-3xl font-bold text-white">{pipelines.length}</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <RefreshCw className="w-5 h-5 text-green-400 animate-spin" />
            <span className="text-sm text-slate-400">Running</span>
          </div>
          <p className="text-3xl font-bold text-white">{pipelines.filter(p => p.status === 'running').length}</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <span className="text-sm text-slate-400">Completed (24h)</span>
          </div>
          <p className="text-3xl font-bold text-white">142</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-slate-400">Failed</span>
          </div>
          <p className="text-3xl font-bold text-white">{pipelines.filter(p => p.status === 'failed').length}</p>
        </GlassCard>
      </div>

      {/* Pipeline List */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">All Pipelines</h2>
        <div className="space-y-3">
          {pipelines.map((pipeline, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                  pipeline.status === 'running' ? 'bg-blue-500/10' :
                  pipeline.status === 'completed' ? 'bg-green-500/10' :
                  pipeline.status === 'failed' ? 'bg-red-500/10' :
                  'bg-slate-500/10'
                }`}>
                  {pipeline.status === 'running' ? <RefreshCw className="w-5 h-5 text-blue-400 animate-spin" /> :
                   pipeline.status === 'completed' ? <CheckCircle className="w-5 h-5 text-green-400" /> :
                   pipeline.status === 'failed' ? <AlertCircle className="w-5 h-5 text-red-400" /> :
                   <Clock className="w-5 h-5 text-slate-400" />}
                </div>
                <div>
                  <p className="text-white font-medium">{pipeline.name}</p>
                  <p className="text-sm text-slate-400">{pipeline.schedule} • Last: {pipeline.lastRun}</p>
                </div>
              </div>
              <button className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.05] rounded-lg transition-all">
                <Play className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </GlassCard>
    </div>
  );
}
