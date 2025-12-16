import { requireAdmin } from '@/lib/auth';
import { Card, Button } from '@/components/ui';
import { Brain, Cpu, TrendingUp, Clock, Play, Pause, Settings } from 'lucide-react';

export default async function ModelsPage() {
  await requireAdmin();

  const models = [
    {
      name: 'Attractor State Predictor v3.2',
      type: 'transformer',
      status: 'active',
      accuracy: 94.2,
      latency: '45ms',
      lastTrained: '2024-03-10',
      requests: '1.2M'
    },
    {
      name: 'Geopolitical Event Classifier',
      type: 'ensemble',
      status: 'active',
      accuracy: 91.8,
      latency: '23ms',
      lastTrained: '2024-03-08',
      requests: '890K'
    },
    {
      name: 'Economic Indicator Forecaster',
      type: 'lstm',
      status: 'active',
      accuracy: 88.5,
      latency: '67ms',
      lastTrained: '2024-03-12',
      requests: '456K'
    },
    {
      name: 'Social Sentiment Analyzer',
      type: 'transformer',
      status: 'training',
      accuracy: 86.3,
      latency: '34ms',
      lastTrained: 'Training...',
      requests: '2.1M'
    },
    {
      name: 'Attractor State Predictor v4.0 (beta)',
      type: 'moe',
      status: 'staging',
      accuracy: 96.1,
      latency: '52ms',
      lastTrained: '2024-03-14',
      requests: '12K'
    },
  ];

  return (
    <div className="pl-72 p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white">ML Models</h1>
          <p className="text-slate-400">Manage and monitor deployed models</p>
        </div>
        <Button variant="secondary">
          <Brain className="w-4 h-4 mr-2" />
          Deploy New Model
        </Button>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Brain className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Total Models</span>
          </div>
          <p className="text-3xl font-bold text-white">{models.length}</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Cpu className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Active</span>
          </div>
          <p className="text-3xl font-bold text-white">{models.filter(m => m.status === 'active').length}</p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Avg Accuracy</span>
          </div>
          <p className="text-3xl font-bold text-white">
            {(models.reduce((acc, m) => acc + m.accuracy, 0) / models.length).toFixed(1)}%
          </p>
        </Card>

        <Card>
          <div className="flex items-center gap-3 mb-2">
            <Clock className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Avg Latency</span>
          </div>
          <p className="text-3xl font-bold text-white">44ms</p>
        </Card>
      </div>

      {/* Model List */}
      <Card>
        <h2 className="text-lg font-bold text-white mb-4">Deployed Models</h2>
        <div className="space-y-3">
          {models.map((model, i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-md border border-white/[0.04]">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-md flex items-center justify-center ${
                  model.status === 'active' ? 'bg-green-500/10' :
                  model.status === 'training' ? 'bg-blue-500/10' :
                  'bg-amber-500/10'
                }`}>
                  <Brain className={`w-5 h-5 ${
                    model.status === 'active' ? 'text-green-400' :
                    model.status === 'training' ? 'text-blue-400' :
                    'text-amber-400'
                  }`} />
                </div>
                <div>
                  <p className="text-white font-medium">{model.name}</p>
                  <p className="text-sm text-slate-400">
                    <span className="uppercase">{model.type}</span> • {model.requests} requests • Last trained: {model.lastTrained}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <div className="text-right">
                  <p className="text-white font-medium">{model.accuracy}%</p>
                  <p className="text-xs text-slate-400">accuracy</p>
                </div>
                <div className="text-right">
                  <p className="text-white font-medium">{model.latency}</p>
                  <p className="text-xs text-slate-400">latency</p>
                </div>
                <span className={`px-2 py-1 rounded text-xs uppercase ${
                  model.status === 'active' ? 'bg-green-500/20 text-green-400' :
                  model.status === 'training' ? 'bg-blue-500/20 text-blue-400' :
                  'bg-amber-500/20 text-amber-400'
                }`}>
                  {model.status}
                </span>
                <div className="flex gap-1">
                  <button className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.05] rounded-md transition-all">
                    {model.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.05] rounded-md transition-all">
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
