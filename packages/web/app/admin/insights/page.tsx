'use client';

import { useState, useEffect, useCallback } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import {
  Brain,
  Sparkles,
  FlaskConical,
  Code2,
  FileCheck,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  XCircle,
  RefreshCw,
  X,
} from 'lucide-react';

// Types
interface InsightReport {
  id: string;
  title: string;
  summary: string;
  target_subject: string;
  current_stage: string;
  status: string;
  confidence_score: number;
  confidence_type: string;
  archaeology_data: Record<string, unknown>;
  nsm_data: Record<string, unknown>;
  theoretical_validation: Record<string, unknown>;
  xyza_data: Record<string, unknown>;
  code_artifacts: Array<{
    filename: string;
    language: string;
    content: string;
    tests_passed: boolean;
  }>;
  impact_analysis: Record<string, unknown>;
  tags: string[];
  created_at: string;
  updated_at: string;
  admin_notes?: string;
  admin_rating?: number;
}

interface InsightStats {
  total_insights: number;
  in_progress: number;
  awaiting_review: number;
  validated: number;
  avg_confidence: number;
}

// Stage config
const STAGE_CONFIG = {
  latent_archaeology: {
    icon: Brain,
    label: 'Archaeology',
    color: 'bg-purple-500',
    description: 'Finding the gradient of ignorance',
  },
  novel_synthesis: {
    icon: Sparkles,
    label: 'Synthesis',
    color: 'bg-blue-500',
    description: 'Creating the fusion',
  },
  theoretical_validation: {
    icon: FlaskConical,
    label: 'Validation',
    color: 'bg-green-500',
    description: 'Proving with math',
  },
  xyza_operationalization: {
    icon: Code2,
    label: 'XYZA',
    color: 'bg-orange-500',
    description: 'Writing the code',
  },
  output_generation: {
    icon: FileCheck,
    label: 'Output',
    color: 'bg-red-500',
    description: 'Final dossier',
  },
};

const STATUS_CONFIG = {
  in_progress: { label: 'In Progress', color: 'bg-yellow-500', textColor: 'text-yellow-300', icon: Clock },
  awaiting_review: { label: 'Awaiting Review', color: 'bg-blue-500', textColor: 'text-blue-300', icon: AlertCircle },
  validated: { label: 'Validated', color: 'bg-green-500', textColor: 'text-green-300', icon: CheckCircle },
  rejected: { label: 'Rejected', color: 'bg-red-500', textColor: 'text-red-300', icon: XCircle },
  needs_revision: { label: 'Needs Revision', color: 'bg-orange-500', textColor: 'text-orange-300', icon: RefreshCw },
};

function Badge({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${className}`}>
      {children}
    </span>
  );
}

function Modal({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/70" onClick={onClose} />
      <div className="relative bg-slate-900 rounded-xl border border-white/10 max-w-4xl w-full max-h-[90vh] overflow-y-auto m-4">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white"
        >
          <X className="w-5 h-5" />
        </button>
        {children}
      </div>
    </div>
  );
}

export default function InsightsDashboard() {
  const [insights, setInsights] = useState<InsightReport[]>([]);
  const [stats, setStats] = useState<InsightStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedInsight, setSelectedInsight] = useState<InsightReport | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterStage, setFilterStage] = useState<string>('all');
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [adminNotes, setAdminNotes] = useState('');

  const fetchInsights = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filterStatus !== 'all') params.set('status', filterStatus);
      if (filterStage !== 'all') params.set('stage', filterStage);

      const response = await fetch(`/api/elle/insights?${params}`);
      const data = await response.json();

      setInsights(data.insights || []);
      setStats(data.stats);
    } catch (error) {
      console.error('Failed to fetch insights:', error);
    } finally {
      setLoading(false);
    }
  }, [filterStatus, filterStage]);

  useEffect(() => {
    void fetchInsights();
  }, [fetchInsights]);

  const handleReview = async (
    insightId: string,
    action: 'validate' | 'reject' | 'revision'
  ) => {
    const statusMap = {
      validate: 'validated',
      reject: 'rejected',
      revision: 'needs_revision',
    };

    try {
      await fetch('/api/elle/insights', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          insight_id: insightId,
          status: statusMap[action],
          admin_notes: adminNotes,
        }),
      });
      void fetchInsights();
      setSelectedInsight(null);
      setAdminNotes('');
    } catch (error) {
      console.error('Review action failed:', error);
    }
  };

  return (
    <div className="text-white">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">Elle&apos;s Research Insights</h1>
        <p className="text-slate-400">
          Autonomous research capture via PROMETHEUS protocol
        </p>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <GlassCard blur="heavy">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Total Insights</p>
                <p className="text-2xl font-bold">{stats.total_insights}</p>
              </div>
              <Brain className="h-8 w-8 text-purple-400" />
            </div>
          </GlassCard>

          <GlassCard blur="heavy">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">In Progress</p>
                <p className="text-2xl font-bold">{stats.in_progress}</p>
              </div>
              <Clock className="h-8 w-8 text-yellow-400" />
            </div>
          </GlassCard>

          <GlassCard blur="heavy">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Awaiting Review</p>
                <p className="text-2xl font-bold">{stats.awaiting_review}</p>
              </div>
              <AlertCircle className="h-8 w-8 text-blue-400" />
            </div>
          </GlassCard>

          <GlassCard blur="heavy">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm">Avg Confidence</p>
                <p className="text-2xl font-bold">
                  {(stats.avg_confidence * 100).toFixed(0)}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-400" />
            </div>
          </GlassCard>
        </div>
      )}

      {/* Pipeline Visualization */}
      <GlassCard blur="heavy" className="mb-8">
        <h2 className="text-lg font-bold mb-4">PROMETHEUS Pipeline</h2>
        <div className="flex items-center justify-between overflow-x-auto">
          {Object.entries(STAGE_CONFIG).map(([stage, config], index) => {
            const Icon = config.icon;
            const count = insights.filter((i) => i.current_stage === stage).length;

            return (
              <div key={stage} className="flex items-center">
                <div className="flex flex-col items-center min-w-[80px]">
                  <div className={`${config.color} p-3 rounded-full mb-2`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <span className="text-sm font-medium">{config.label}</span>
                  <span className="text-xs text-slate-400">{count} active</span>
                </div>
                {index < Object.keys(STAGE_CONFIG).length - 1 && (
                  <div className="w-8 md:w-16 h-0.5 bg-slate-600 mx-2" />
                )}
              </div>
            );
          })}
        </div>
      </GlassCard>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="bg-slate-800 text-white rounded-lg px-4 py-2 border border-white/10"
        >
          <option value="all">All Statuses</option>
          <option value="in_progress">In Progress</option>
          <option value="awaiting_review">Awaiting Review</option>
          <option value="validated">Validated</option>
          <option value="rejected">Rejected</option>
        </select>

        <select
          value={filterStage}
          onChange={(e) => setFilterStage(e.target.value)}
          className="bg-slate-800 text-white rounded-lg px-4 py-2 border border-white/10"
        >
          <option value="all">All Stages</option>
          {Object.entries(STAGE_CONFIG).map(([stage, config]) => (
            <option key={stage} value={stage}>
              {config.label}
            </option>
          ))}
        </select>

        <GlassButton variant="secondary" onClick={() => void fetchInsights()}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </GlassButton>
      </div>

      {/* Insights List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin h-8 w-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-slate-400">Loading insights...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {insights.map((insight) => {
            const stageConfig = STAGE_CONFIG[insight.current_stage as keyof typeof STAGE_CONFIG];
            const statusConfig = STATUS_CONFIG[insight.status as keyof typeof STATUS_CONFIG];
            const StageIcon = stageConfig?.icon || Brain;
            const StatusIcon = statusConfig?.icon || Clock;

            return (
              <GlassCard
                key={insight.id}
                blur="light"
                className="cursor-pointer hover:border-white/20 transition-colors"
                onClick={() => {
                  setSelectedInsight(insight);
                  setActiveTab('overview');
                }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2 flex-wrap">
                      <div className={`${stageConfig?.color || 'bg-slate-500'} p-2 rounded`}>
                        <StageIcon className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-lg">{insight.title}</h3>
                      <Badge className={`${statusConfig?.color || 'bg-slate-500'} text-white flex items-center gap-1`}>
                        <StatusIcon className="h-3 w-3" />
                        {statusConfig?.label || insight.status}
                      </Badge>
                    </div>

                    <p className="text-slate-400 text-sm mb-3">
                      {insight.summary || 'No summary available'}
                    </p>

                    <div className="flex items-center gap-4 text-sm text-slate-500 flex-wrap">
                      <span>Target: {insight.target_subject}</span>
                      <span>•</span>
                      <span>
                        Confidence:{' '}
                        {insight.confidence_score
                          ? `${(insight.confidence_score * 100).toFixed(0)}%`
                          : 'N/A'}
                      </span>
                      <span>•</span>
                      <span>{new Date(insight.created_at).toLocaleDateString()}</span>
                    </div>

                    {insight.tags.length > 0 && (
                      <div className="flex gap-2 mt-3 flex-wrap">
                        {insight.tags.map((tag) => (
                          <Badge key={tag} className="bg-slate-700 text-slate-300">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  {insight.code_artifacts.length > 0 && (
                    <Badge className="bg-green-600 text-white flex items-center gap-1">
                      <Code2 className="h-3 w-3" />
                      {insight.code_artifacts.length} artifact{insight.code_artifacts.length > 1 ? 's' : ''}
                    </Badge>
                  )}
                </div>
              </GlassCard>
            );
          })}

          {insights.length === 0 && (
            <div className="text-center py-12">
              <Brain className="h-12 w-12 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400">No insights found</p>
              <p className="text-slate-500 text-sm">
                Elle is working silently. Insights will appear when validated.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Insight Detail Modal */}
      <Modal open={!!selectedInsight} onClose={() => setSelectedInsight(null)}>
        {selectedInsight && (
          <div className="p-6">
            <h2 className="text-xl font-bold mb-2">{selectedInsight.title}</h2>
            <p className="text-slate-400 mb-6">{selectedInsight.summary}</p>

            {/* Tabs */}
            <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
              {['overview', 'archaeology', 'nsm', 'validation', 'code', 'impact'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium capitalize whitespace-nowrap ${
                    activeTab === tab
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="min-h-[200px]">
              {activeTab === 'overview' && (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-slate-400">Stage</label>
                    <p className="font-medium">
                      {STAGE_CONFIG[selectedInsight.current_stage as keyof typeof STAGE_CONFIG]?.label}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Status</label>
                    <p className="font-medium">{selectedInsight.status}</p>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Confidence</label>
                    <p className="font-medium">
                      {selectedInsight.confidence_score
                        ? `${(selectedInsight.confidence_score * 100).toFixed(0)}%`
                        : 'Not assessed'}
                      {selectedInsight.confidence_type && (
                        <span className="text-slate-500 ml-2">({selectedInsight.confidence_type})</span>
                      )}
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-slate-400">Target Subject</label>
                    <p className="font-medium">{selectedInsight.target_subject}</p>
                  </div>
                </div>
              )}

              {activeTab === 'archaeology' && (
                <pre className="bg-slate-800 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(selectedInsight.archaeology_data, null, 2)}
                </pre>
              )}

              {activeTab === 'nsm' && (
                <pre className="bg-slate-800 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(selectedInsight.nsm_data, null, 2)}
                </pre>
              )}

              {activeTab === 'validation' && (
                <pre className="bg-slate-800 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(selectedInsight.theoretical_validation, null, 2)}
                </pre>
              )}

              {activeTab === 'code' && (
                <div className="space-y-4">
                  {selectedInsight.code_artifacts.length > 0 ? (
                    selectedInsight.code_artifacts.map((artifact, idx) => (
                      <div key={idx} className="border border-slate-700 rounded">
                        <div className="bg-slate-700 px-4 py-2 flex items-center justify-between">
                          <span className="font-mono text-sm">{artifact.filename}</span>
                          <Badge className={artifact.tests_passed ? 'bg-green-600' : 'bg-red-600'}>
                            {artifact.tests_passed ? 'Tests Passed' : 'Tests Failed'}
                          </Badge>
                        </div>
                        <pre className="p-4 overflow-x-auto text-sm bg-slate-800">
                          {artifact.content}
                        </pre>
                      </div>
                    ))
                  ) : (
                    <p className="text-slate-400">No code artifacts yet</p>
                  )}
                </div>
              )}

              {activeTab === 'impact' && (
                <pre className="bg-slate-800 p-4 rounded overflow-x-auto text-sm">
                  {JSON.stringify(selectedInsight.impact_analysis, null, 2)}
                </pre>
              )}
            </div>

            {/* Review Actions */}
            {selectedInsight.status === 'awaiting_review' && (
              <div className="mt-6 pt-6 border-t border-slate-700">
                <h4 className="font-medium mb-4">Admin Review</h4>
                <textarea
                  value={adminNotes}
                  onChange={(e) => setAdminNotes(e.target.value)}
                  placeholder="Add notes (optional)..."
                  className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 mb-4 text-white"
                  rows={3}
                />
                <div className="flex gap-3 flex-wrap">
                  <GlassButton
                    variant="primary"
                    onClick={() => void handleReview(selectedInsight.id, 'validate')}
                  >
                    <CheckCircle className="h-4 w-4 mr-2" />
                    Validate
                  </GlassButton>
                  <GlassButton
                    variant="secondary"
                    onClick={() => void handleReview(selectedInsight.id, 'revision')}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Request Revision
                  </GlassButton>
                  <button
                    onClick={() => void handleReview(selectedInsight.id, 'reject')}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center text-white"
                  >
                    <XCircle className="h-4 w-4 mr-2" />
                    Reject
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
