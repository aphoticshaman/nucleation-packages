'use client';

import { useState } from 'react';
import {
  Plus,
  LayoutDashboard,
  Star,
  StarOff,
  ChevronRight,
  Clock,
  Database,
  Activity,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Globe,
  TrendingUp,
  Shield,
  Zap,
  BarChart3,
  LineChart,
  X,
  BookOpen,
} from 'lucide-react';
import { DASHBOARD_PRESETS, type PresetId } from '@/lib/config/dashboardPresets';
import { FOCUS_TEMPLATES } from '@/lib/pipeline/DataPipeline';
import Glossary from '@/components/Glossary';
import { Card } from '@/components/ui';

interface UserDashboard {
  id: string;
  name: string;
  description: string;
  presetId: PresetId | 'custom';
  focusTemplate?: string;
  createdAt: string;
  updatedAt: string;
  isFavorite: boolean;
  dataQuality: {
    score: number;
    sources: number;
    freshness: 'fresh' | 'stale' | 'mixed';
  };
}

const MOCK_USER_DASHBOARDS: UserDashboard[] = [
  {
    id: 'dash-1',
    name: 'Global Risk Monitor',
    description: 'My primary intel dashboard tracking global threats',
    presetId: 'intelligence-officer',
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-03-01T14:30:00Z',
    isFavorite: true,
    dataQuality: { score: 92, sources: 8, freshness: 'fresh' },
  },
  {
    id: 'dash-2',
    name: 'US Markets Deep Dive',
    description: 'Real-time US equity and treasury tracking',
    presetId: 'economic-analyst',
    focusTemplate: 'us-equities',
    createdAt: '2024-02-10T08:00:00Z',
    updatedAt: '2024-03-02T09:15:00Z',
    isFavorite: true,
    dataQuality: { score: 88, sources: 6, freshness: 'fresh' },
  },
  {
    id: 'dash-3',
    name: 'China Watch',
    description: 'Tracking China-US relations and economic signals',
    presetId: 'regional-specialist',
    focusTemplate: 'china-watch',
    createdAt: '2024-02-20T12:00:00Z',
    updatedAt: '2024-03-01T16:45:00Z',
    isFavorite: false,
    dataQuality: { score: 75, sources: 5, freshness: 'mixed' },
  },
];

export default function DashboardHub() {
  const [dashboards, setDashboards] = useState<UserDashboard[]>(MOCK_USER_DASHBOARDS);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<PresetId | null>(null);
  const [selectedFocus, setSelectedFocus] = useState<string | null>(null);
  const [createStep, setCreateStep] = useState<'preset' | 'focus' | 'configure'>('preset');
  const [showGlossary, setShowGlossary] = useState(false);

  const favoriteDashboards = dashboards.filter(d => d.isFavorite);
  const otherDashboards = dashboards.filter(d => !d.isFavorite);

  const toggleFavorite = (id: string) => {
    setDashboards(prev =>
      prev.map(d =>
        d.id === id ? { ...d, isFavorite: !d.isFavorite } : d
      )
    );
  };

  const getFreshnessIcon = (freshness: string) => {
    switch (freshness) {
      case 'fresh':
        return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'stale':
        return <AlertCircle className="w-4 h-4 text-red-400" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-400" />;
    }
  };

  const getPresetIcon = (presetId: PresetId) => {
    switch (presetId) {
      case 'intelligence-officer':
        return <Shield className="w-5 h-5" />;
      case 'economic-analyst':
        return <TrendingUp className="w-5 h-5" />;
      case 'crisis-monitor':
        return <Zap className="w-5 h-5" />;
      case 'executive-briefer':
        return <BarChart3 className="w-5 h-5" />;
      case 'regional-specialist':
        return <Globe className="w-5 h-5" />;
      default:
        return <LayoutDashboard className="w-5 h-5" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-white flex items-center gap-3">
            <LayoutDashboard className="w-7 h-7 text-cyan-400" />
            Dashboard Hub
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Create, customize, and manage your intelligence dashboards
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowGlossary(true)}
            className="flex items-center gap-2 px-3 py-2 min-h-[44px] bg-[rgba(18,18,26,0.7)] backdrop-blur-sm rounded-xl border border-white/[0.06] text-slate-400 hover:text-white hover:border-white/[0.12] transition-all"
          >
            <BookOpen className="w-4 h-4" />
            <span className="text-sm">Terms</span>
          </button>
          <button
            onClick={() => {
              setShowCreateModal(true);
              setCreateStep('preset');
            }}
            className="flex items-center gap-2 px-4 py-2.5 min-h-[44px] bg-gradient-to-r from-cyan-600 to-blue-500 text-white rounded-xl font-medium
              hover:shadow-[0_0_20px_rgba(6,182,212,0.4)] active:scale-[0.98] transition-all"
          >
            <Plus className="w-5 h-5" />
            New Dashboard
          </button>
        </div>
      </div>

      {/* Data Health Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card padding="sm">
          <div className="flex items-center gap-3 mb-2">
            <Database className="w-5 h-5 text-cyan-400" />
            <span className="text-sm text-slate-400">Active Sources</span>
          </div>
          <div className="text-lg font-bold text-white">12</div>
          <div className="text-xs text-green-400 mt-1">All connected</div>
        </Card>
        <Card padding="sm">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-5 h-5 text-emerald-400" />
            <span className="text-sm text-slate-400">Data Freshness</span>
          </div>
          <div className="text-lg font-bold text-white">94%</div>
          <div className="text-xs text-slate-500 mt-1">Last 24 hours</div>
        </Card>
        <Card padding="sm">
          <div className="flex items-center gap-3 mb-2">
            <RefreshCw className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Last Refresh</span>
          </div>
          <div className="text-lg font-bold text-white">2m ago</div>
          <div className="text-xs text-slate-500 mt-1">Auto-refresh: ON</div>
        </Card>
        <Card padding="sm">
          <div className="flex items-center gap-3 mb-2">
            <LineChart className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-slate-400">Data Points</span>
          </div>
          <div className="text-lg font-bold text-white">1.2M</div>
          <div className="text-xs text-slate-500 mt-1">This month</div>
        </Card>
      </div>

      {/* Favorite Dashboards */}
      {favoriteDashboards.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
            <Star className="w-5 h-5 text-amber-400 fill-amber-400" />
            Favorites
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {favoriteDashboards.map(dashboard => (
              <DashboardCard
                key={dashboard.id}
                dashboard={dashboard}
                onToggleFavorite={toggleFavorite}
                getPresetIcon={getPresetIcon}
                getFreshnessIcon={getFreshnessIcon}
              />
            ))}
          </div>
        </section>
      )}

      {/* All Dashboards */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
          <LayoutDashboard className="w-5 h-5 text-slate-400" />
          All Dashboards ({dashboards.length})
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {otherDashboards.map(dashboard => (
            <DashboardCard
              key={dashboard.id}
              dashboard={dashboard}
              onToggleFavorite={toggleFavorite}
              getPresetIcon={getPresetIcon}
              getFreshnessIcon={getFreshnessIcon}
            />
          ))}
          {/* Empty State / Quick Create */}
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex flex-col items-center justify-center gap-3 p-8 border-2 border-dashed border-white/[0.08] rounded-xl hover:border-cyan-500/50 hover:bg-[rgba(18,18,26,0.5)] transition-all group min-h-[200px]"
          >
            <Plus className="w-10 h-10 text-slate-600 group-hover:text-cyan-400 transition-colors" />
            <span className="text-slate-500 group-hover:text-slate-300 transition-colors">
              Create New Dashboard
            </span>
          </button>
        </div>
      </section>

      {/* Quick Start Templates */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white">
          <Zap className="w-5 h-5 text-cyan-400" />
          Quick Start Templates
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(FOCUS_TEMPLATES).slice(0, 8).map(([key, focus]) => (
            <button
              key={key}
              onClick={() => {
                setSelectedFocus(key);
                setShowCreateModal(true);
                setCreateStep('configure');
              }}
              className="text-left p-4 bg-[rgba(18,18,26,0.5)] backdrop-blur-sm border border-white/[0.06] rounded-xl hover:border-cyan-500/50 hover:bg-[rgba(18,18,26,0.7)] transition-all group"
            >
              <div className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors">
                {key.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {focus.type} focus • {focus.topics?.length || focus.sectors?.length || 0} topics
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* Create Modal */}
      {showCreateModal && (
        <CreateDashboardModal
          step={createStep}
          onStepChange={setCreateStep}
          selectedPreset={selectedPreset}
          onSelectPreset={setSelectedPreset}
          selectedFocus={selectedFocus}
          onSelectFocus={setSelectedFocus}
          onClose={() => {
            setShowCreateModal(false);
            setSelectedPreset(null);
            setSelectedFocus(null);
            setCreateStep('preset');
          }}
          onCreate={(config) => {
            const newDash: UserDashboard = {
              id: `dash-${Date.now()}`,
              name: config.name,
              description: config.description,
              presetId: config.presetId,
              focusTemplate: config.focusTemplate,
              createdAt: new Date().toISOString(),
              updatedAt: new Date().toISOString(),
              isFavorite: false,
              dataQuality: { score: 100, sources: 0, freshness: 'fresh' },
            };
            setDashboards(prev => [...prev, newDash]);
            setShowCreateModal(false);
          }}
        />
      )}

      {/* Glossary Modal */}
      <Glossary
        isOpen={showGlossary}
        onClose={() => setShowGlossary(false)}
        skillLevel="standard"
      />
    </div>
  );
}

function DashboardCard({
  dashboard,
  onToggleFavorite,
  getPresetIcon,
  getFreshnessIcon,
}: {
  dashboard: UserDashboard;
  onToggleFavorite: (id: string) => void;
  getPresetIcon: (id: PresetId) => React.ReactNode;
  getFreshnessIcon: (freshness: string) => React.ReactNode;
}) {
  return (
    <div className="bg-[rgba(18,18,26,0.7)] backdrop-blur-xl border border-white/[0.06] rounded-xl p-4 hover:border-cyan-500/30 transition-all group">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-black/30 rounded-lg text-cyan-400">
            {getPresetIcon(dashboard.presetId as PresetId)}
          </div>
          <div>
            <h3 className="font-medium text-white group-hover:text-cyan-400 transition-colors">
              {dashboard.name}
            </h3>
            <p className="text-xs text-slate-500">
              {DASHBOARD_PRESETS[dashboard.presetId as PresetId]?.name || 'Custom'}
            </p>
          </div>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite(dashboard.id);
          }}
          className="text-slate-500 hover:text-amber-400 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center -mr-2 -mt-2"
        >
          {dashboard.isFavorite ? (
            <Star className="w-5 h-5 fill-amber-400 text-amber-400" />
          ) : (
            <StarOff className="w-5 h-5" />
          )}
        </button>
      </div>

      <p className="text-sm text-slate-400 mb-4 line-clamp-2">
        {dashboard.description}
      </p>

      {/* Data Quality Indicator */}
      <div className="flex items-center justify-between pt-3 border-t border-white/[0.06]">
        <div className="flex items-center gap-2 text-xs">
          {getFreshnessIcon(dashboard.dataQuality.freshness)}
          <span className="text-slate-400">
            {dashboard.dataQuality.sources} sources • {dashboard.dataQuality.score}% quality
          </span>
        </div>
        <a
          href={`/app/dashboards/${dashboard.id}`}
          className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors min-h-[44px]"
        >
          Open <ChevronRight className="w-4 h-4" />
        </a>
      </div>
    </div>
  );
}

function CreateDashboardModal({
  step,
  onStepChange,
  selectedPreset,
  onSelectPreset,
  selectedFocus,
  onSelectFocus,
  onClose,
  onCreate,
}: {
  step: 'preset' | 'focus' | 'configure';
  onStepChange: (step: 'preset' | 'focus' | 'configure') => void;
  selectedPreset: PresetId | null;
  onSelectPreset: (id: PresetId) => void;
  selectedFocus: string | null;
  onSelectFocus: (id: string) => void;
  onClose: () => void;
  onCreate: (config: { name: string; description: string; presetId: PresetId; focusTemplate?: string }) => void;
}) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-[rgba(18,18,26,0.95)] backdrop-blur-xl border border-white/[0.08] rounded-2xl w-full max-w-4xl max-h-[80vh] overflow-hidden shadow-2xl">
        {/* Header */}
        <div className="p-6 border-b border-white/[0.06] flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Create New Dashboard</h2>
            <div className="flex items-center gap-2 mt-4">
              {['preset', 'focus', 'configure'].map((s, i) => (
                <div key={s} className="flex items-center">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                      step === s
                        ? 'bg-cyan-600 text-white'
                        : i < ['preset', 'focus', 'configure'].indexOf(step)
                        ? 'bg-green-600 text-white'
                        : 'bg-white/[0.06] text-slate-500'
                    }`}
                  >
                    {i + 1}
                  </div>
                  {i < 2 && (
                    <div className={`w-16 h-0.5 mx-2 ${
                      i < ['preset', 'focus', 'configure'].indexOf(step)
                        ? 'bg-green-600'
                        : 'bg-white/[0.06]'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white min-w-[44px] min-h-[44px] flex items-center justify-center -mr-2 -mt-2"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {step === 'preset' && (
            <div>
              <h3 className="text-sm font-medium text-slate-400 mb-4">
                Choose a starting template
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(DASHBOARD_PRESETS).map(([id, preset]) => (
                  <button
                    key={id}
                    onClick={() => {
                      onSelectPreset(id as PresetId);
                      onStepChange('focus');
                    }}
                    className={`text-left p-4 rounded-xl border transition-all min-h-[100px] ${
                      selectedPreset === id
                        ? 'border-cyan-500/50 bg-cyan-500/10'
                        : 'border-white/[0.06] hover:border-white/[0.12] bg-black/20'
                    }`}
                  >
                    <div className="font-medium text-white">{preset.name}</div>
                    <div className="text-xs text-slate-400 mt-1">{preset.tagline}</div>
                    <div className="text-xs text-slate-500 mt-2">
                      {preset.widgets.length} widgets • {preset.recommendedFor[0]}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 'focus' && (
            <div>
              <h3 className="text-sm font-medium text-slate-400 mb-4">
                Choose a data focus (optional)
              </h3>
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => onStepChange('configure')}
                  className={`text-left p-4 rounded-xl border ${
                    !selectedFocus
                      ? 'border-cyan-500/50 bg-cyan-500/10'
                      : 'border-white/[0.06] hover:border-white/[0.12] bg-black/20'
                  }`}
                >
                  <div className="font-medium text-white">Global (Default)</div>
                  <div className="text-xs text-slate-400 mt-1">
                    All regions, all sectors
                  </div>
                </button>
                {Object.entries(FOCUS_TEMPLATES).map(([id, focus]) => (
                  <button
                    key={id}
                    onClick={() => {
                      onSelectFocus(id);
                      onStepChange('configure');
                    }}
                    className={`text-left p-4 rounded-xl border transition-all ${
                      selectedFocus === id
                        ? 'border-cyan-500/50 bg-cyan-500/10'
                        : 'border-white/[0.06] hover:border-white/[0.12] bg-black/20'
                    }`}
                  >
                    <div className="font-medium text-white">
                      {id.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      {focus.type} • {focus.topics?.slice(0, 2).join(', ') || 'Custom focus'}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {step === 'configure' && (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-slate-400 block mb-2">
                  Dashboard Name
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="My Intelligence Dashboard"
                  className="w-full bg-black/30 border border-white/[0.08] rounded-xl px-4 py-3 min-h-[48px] text-white placeholder:text-slate-500
                    focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-400 block mb-2">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is this dashboard for?"
                  rows={3}
                  className="w-full bg-black/30 border border-white/[0.08] rounded-xl px-4 py-3 text-white placeholder:text-slate-500 resize-none
                    focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20"
                />
              </div>

              {/* Summary */}
              <div className="bg-black/20 border border-white/[0.06] rounded-xl p-4">
                <div className="text-sm text-slate-400 mb-2">Configuration Summary</div>
                <div className="text-sm">
                  <span className="text-slate-500">Template: </span>
                  <span className="text-white">
                    {selectedPreset ? DASHBOARD_PRESETS[selectedPreset].name : 'Custom'}
                  </span>
                </div>
                <div className="text-sm mt-1">
                  <span className="text-slate-500">Focus: </span>
                  <span className="text-white">
                    {selectedFocus
                      ? selectedFocus.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                      : 'Global'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/[0.06] flex justify-between">
          <button
            onClick={onClose}
            className="px-4 py-2.5 min-h-[44px] text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <div className="flex gap-2">
            {step !== 'preset' && (
              <button
                onClick={() =>
                  onStepChange(step === 'configure' ? 'focus' : 'preset')
                }
                className="px-4 py-2.5 min-h-[44px] bg-white/[0.06] text-white rounded-xl hover:bg-white/[0.1] transition-colors"
              >
                Back
              </button>
            )}
            {step === 'configure' && (
              <button
                onClick={() =>
                  onCreate({
                    name: name || 'Untitled Dashboard',
                    description,
                    presetId: selectedPreset || 'custom' as PresetId,
                    focusTemplate: selectedFocus || undefined,
                  })
                }
                className="px-4 py-2.5 min-h-[44px] bg-gradient-to-r from-cyan-600 to-blue-500 text-white rounded-xl font-medium
                  hover:shadow-[0_0_20px_rgba(6,182,212,0.4)] active:scale-[0.98] transition-all"
              >
                Create Dashboard
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
