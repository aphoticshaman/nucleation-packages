'use client';

import { useState, useEffect, useCallback } from 'react';
import { GlassCard } from '@/components/ui/GlassCard';
import { createBrowserClient } from '@supabase/ssr';

// Types for Guardian data
interface GuardianMetrics {
  global: {
    total_evaluations?: number;
    accuracy_pct?: number;
    hallucination_rate_pct?: number;
    factualness_pct?: number;
    informativeness_pct?: number;
    scope_coverage_pct?: number;
    depth_score?: number;
    empiricality_pct?: number;
  };
  byDomain: Array<{
    domain: string;
    accuracy_pct?: number;
    total_evaluations?: number;
  }>;
}

interface RuleProposal {
  id: string;
  proposed_rule_name: string;
  proposed_domain: string;
  proposed_config: Record<string, unknown>;
  rationale: string;
  elle_summary: string;
  guardian_summary: string;
  confidence_score: number;
  supporting_evidence: string[];
  created_at: string;
}

interface Disagreement {
  id: string;
  domain: string;
  input_summary: string;
  elle_decision: string;
  elle_reasoning: string;
  guardian_decision: string;
  guardian_reasoning: string;
  created_at: string;
}

interface ActiveRule {
  id: string;
  rule_name: string;
  domain: string;
  version: number;
  rule_config: Record<string, unknown>;
  description: string;
  activated_at: string;
}

interface AuditLogEntry {
  id: string;
  action: string;
  entity_type: string;
  entity_id: string;
  old_value?: Record<string, unknown>;
  new_value?: Record<string, unknown>;
  reason?: string;
  performed_by: string;
  performed_at: string;
}

interface DashboardData {
  activeRules: ActiveRule[];
  ruleCount: number;
  metrics: GuardianMetrics;
  proposals: {
    pending: RuleProposal[];
    pendingCount: number;
  };
  disagreements: Disagreement[];
  auditLog: AuditLogEntry[];
  summary: {
    totalEvaluations24h: number;
    accuracy: number | null;
    hallucinationRate: number | null;
    pendingReviews: number;
  };
}

// Metric Card Component
function MetricCard({
  label,
  value,
  unit = '%',
  threshold,
  inverse = false,
}: {
  label: string;
  value: number | null;
  unit?: string;
  threshold?: { warning: number; critical: number };
  inverse?: boolean;
}) {
  const getColor = () => {
    if (value === null) return 'text-slate-500';
    if (!threshold) return 'text-white';

    const isGood = inverse
      ? value < threshold.warning
      : value > threshold.warning;
    const isBad = inverse
      ? value > threshold.critical
      : value < threshold.critical;

    if (isBad) return 'text-red-400';
    if (!isGood) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className="bg-black/20 rounded-xl p-4 border border-white/[0.06]">
      <p className="text-xs text-slate-500 uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${getColor()}`}>
        {value !== null ? `${value.toFixed(1)}${unit}` : 'N/A'}
      </p>
    </div>
  );
}

// Status Badge Component
function StatusBadge({ status }: { status: 'pending' | 'accepted' | 'rejected' }) {
  const colors = {
    pending: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    accepted: 'bg-green-500/20 text-green-400 border-green-500/30',
    rejected: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  return (
    <span className={`px-2 py-0.5 text-xs rounded-full border ${colors[status]}`}>
      {status}
    </span>
  );
}

export default function GuardianDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [selectedProposal, setSelectedProposal] = useState<RuleProposal | null>(null);
  const [selectedDisagreement, setSelectedDisagreement] = useState<Disagreement | null>(null);
  const [reviewNotes, setReviewNotes] = useState('');

  const supabase = createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );

  // Fetch dashboard data
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/admin/guardian?view=dashboard');
      if (!response.ok) throw new Error('Failed to fetch dashboard data');
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Action handlers
  const handleAction = async (action: string, payload: Record<string, unknown>) => {
    const actionId = `${action}-${JSON.stringify(payload)}`;
    setActionLoading(actionId);
    try {
      const response = await fetch('/api/admin/guardian', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...payload }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Action failed');

      // Refresh data
      await fetchData();
      setSelectedProposal(null);
      setSelectedDisagreement(null);
      setReviewNotes('');
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Action failed');
    } finally {
      setActionLoading(null);
    }
  };

  // Export training log
  const handleExport = async () => {
    if (!data) return;

    const exportData = {
      exportedAt: new Date().toISOString(),
      summary: data.summary,
      metrics: data.metrics,
      activeRules: data.activeRules,
      pendingProposals: data.proposals.pending,
      recentDisagreements: data.disagreements,
      auditLog: data.auditLog,
      trainingRecommendations: generateTrainingRecommendations(data),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `guardian-training-log-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Generate training recommendations based on metrics
  function generateTrainingRecommendations(data: DashboardData): string[] {
    const recommendations: string[] = [];
    const { metrics, disagreements } = data;

    if (metrics.global.accuracy_pct && metrics.global.accuracy_pct < 85) {
      recommendations.push('Accuracy below 85% - review disagreement patterns for systematic errors');
    }
    if (metrics.global.hallucination_rate_pct && metrics.global.hallucination_rate_pct > 5) {
      recommendations.push('Hallucination rate above 5% - strengthen grounding in factual sources');
    }
    if (metrics.global.factualness_pct && metrics.global.factualness_pct < 90) {
      recommendations.push('Factualness below 90% - increase emphasis on source verification');
    }
    if (disagreements.length > 10) {
      recommendations.push(`${disagreements.length} unresolved disagreements - prioritize human review`);
    }

    // Domain-specific recommendations
    const weakDomains = metrics.byDomain
      .filter(d => d.accuracy_pct && d.accuracy_pct < 80)
      .map(d => d.domain);
    if (weakDomains.length > 0) {
      recommendations.push(`Weak performance in domains: ${weakDomains.join(', ')} - focus training data`);
    }

    return recommendations.length > 0 ? recommendations : ['System performing within acceptable parameters'];
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-white">Guardian Governance</h1>
            <p className="text-slate-400">Human-in-the-loop AI oversight</p>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-black/20 rounded-xl p-4 border border-white/[0.06] animate-pulse h-24" />
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <GlassCard blur="heavy" className="text-center py-12">
        <p className="text-red-400 text-lg">Error loading dashboard</p>
        <p className="text-slate-500 mt-2">{error}</p>
        <button
          onClick={fetchData}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500"
        >
          Retry
        </button>
      </GlassCard>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold text-white">Guardian Governance</h1>
          <p className="text-slate-400">Human-in-the-loop AI oversight dashboard</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-white/[0.06] text-slate-300 rounded-lg hover:bg-white/[0.1] transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export Training Log
          </button>
          <button
            onClick={fetchData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <GlassCard blur="heavy">
          <p className="text-sm text-slate-400">Evaluations (24h)</p>
          <p className="text-3xl font-bold text-white mt-2">{data.summary.totalEvaluations24h.toLocaleString()}</p>
        </GlassCard>
        <GlassCard blur="heavy">
          <p className="text-sm text-slate-400">Accuracy</p>
          <p className={`text-3xl font-bold mt-2 ${
            data.summary.accuracy && data.summary.accuracy >= 90 ? 'text-green-400' :
            data.summary.accuracy && data.summary.accuracy >= 80 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {data.summary.accuracy ? `${data.summary.accuracy.toFixed(1)}%` : 'N/A'}
          </p>
        </GlassCard>
        <GlassCard blur="heavy">
          <p className="text-sm text-slate-400">Hallucination Rate</p>
          <p className={`text-3xl font-bold mt-2 ${
            data.summary.hallucinationRate && data.summary.hallucinationRate <= 2 ? 'text-green-400' :
            data.summary.hallucinationRate && data.summary.hallucinationRate <= 5 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {data.summary.hallucinationRate ? `${data.summary.hallucinationRate.toFixed(1)}%` : 'N/A'}
          </p>
        </GlassCard>
        <GlassCard blur="heavy">
          <p className="text-sm text-slate-400">Pending Reviews</p>
          <p className={`text-3xl font-bold mt-2 ${
            data.summary.pendingReviews === 0 ? 'text-green-400' :
            data.summary.pendingReviews < 5 ? 'text-yellow-400' : 'text-orange-400'
          }`}>
            {data.summary.pendingReviews}
          </p>
        </GlassCard>
      </div>

      {/* Detailed Metrics Grid */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">Quality Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          <MetricCard
            label="Accuracy"
            value={data.metrics.global.accuracy_pct ?? null}
            threshold={{ warning: 85, critical: 75 }}
          />
          <MetricCard
            label="Hallucination"
            value={data.metrics.global.hallucination_rate_pct ?? null}
            threshold={{ warning: 5, critical: 10 }}
            inverse
          />
          <MetricCard
            label="Factualness"
            value={data.metrics.global.factualness_pct ?? null}
            threshold={{ warning: 90, critical: 80 }}
          />
          <MetricCard
            label="Informativeness"
            value={data.metrics.global.informativeness_pct ?? null}
            threshold={{ warning: 80, critical: 70 }}
          />
          <MetricCard
            label="Scope Coverage"
            value={data.metrics.global.scope_coverage_pct ?? null}
            threshold={{ warning: 85, critical: 75 }}
          />
          <MetricCard
            label="Depth Score"
            value={data.metrics.global.depth_score ?? null}
            unit=""
            threshold={{ warning: 7, critical: 5 }}
          />
          <MetricCard
            label="Empiricality"
            value={data.metrics.global.empiricality_pct ?? null}
            threshold={{ warning: 85, critical: 75 }}
          />
        </div>
      </GlassCard>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pending Proposals */}
        <GlassCard blur="heavy" className="p-0 overflow-hidden">
          <div className="p-6 border-b border-white/[0.06]">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-bold text-white">Elle&apos;s Proposals</h2>
                <p className="text-sm text-slate-400">Rule changes awaiting approval</p>
              </div>
              {data.proposals.pendingCount > 0 && (
                <span className="px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm font-medium">
                  {data.proposals.pendingCount} pending
                </span>
              )}
            </div>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            {data.proposals.pending.length === 0 ? (
              <div className="p-8 text-center text-slate-500">
                No pending proposals
              </div>
            ) : (
              data.proposals.pending.map((proposal) => (
                <div
                  key={proposal.id}
                  className="p-4 border-b border-white/[0.06] hover:bg-white/[0.02] cursor-pointer"
                  onClick={() => setSelectedProposal(proposal)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-white">{proposal.proposed_rule_name}</h3>
                    <span className="text-xs text-slate-500">
                      {new Date(proposal.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 line-clamp-2">{proposal.rationale}</p>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="px-2 py-0.5 bg-white/[0.06] text-xs text-slate-300 rounded">
                      {proposal.proposed_domain || 'global'}
                    </span>
                    <span className="text-xs text-slate-500">
                      Confidence: {(proposal.confidence_score * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </GlassCard>

        {/* Disagreements */}
        <GlassCard blur="heavy" className="p-0 overflow-hidden">
          <div className="p-6 border-b border-white/[0.06]">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-lg font-bold text-white">Disagreements</h2>
                <p className="text-sm text-slate-400">Elle vs Guardian conflicts</p>
              </div>
              {data.disagreements.length > 0 && (
                <span className="px-3 py-1 bg-orange-500/20 text-orange-400 rounded-full text-sm font-medium">
                  {data.disagreements.length} unresolved
                </span>
              )}
            </div>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            {data.disagreements.length === 0 ? (
              <div className="p-8 text-center text-slate-500">
                No disagreements to review
              </div>
            ) : (
              data.disagreements.map((d) => (
                <div
                  key={d.id}
                  className="p-4 border-b border-white/[0.06] hover:bg-white/[0.02] cursor-pointer"
                  onClick={() => setSelectedDisagreement(d)}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="px-2 py-0.5 bg-white/[0.06] text-xs text-slate-300 rounded">
                      {d.domain}
                    </span>
                    <span className="text-xs text-slate-500">
                      {new Date(d.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-400 line-clamp-2">{d.input_summary}</p>
                  <div className="grid grid-cols-2 gap-4 mt-3">
                    <div className="text-xs">
                      <span className="text-purple-400">Elle:</span>
                      <span className="text-slate-400 ml-1">{d.elle_decision}</span>
                    </div>
                    <div className="text-xs">
                      <span className="text-blue-400">Guardian:</span>
                      <span className="text-slate-400 ml-1">{d.guardian_decision}</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </GlassCard>
      </div>

      {/* Active Rules */}
      <GlassCard blur="heavy" className="p-0 overflow-hidden">
        <div className="p-6 border-b border-white/[0.06] flex justify-between items-center">
          <div>
            <h2 className="text-lg font-bold text-white">Active Rules</h2>
            <p className="text-sm text-slate-400">{data.ruleCount} rules currently active</p>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left text-xs text-slate-500 uppercase tracking-wide border-b border-white/[0.06]">
                <th className="py-3 px-4">Rule</th>
                <th className="py-3 px-4">Domain</th>
                <th className="py-3 px-4">Version</th>
                <th className="py-3 px-4">Description</th>
                <th className="py-3 px-4">Activated</th>
                <th className="py-3 px-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {data.activeRules.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500">
                    No active rules
                  </td>
                </tr>
              ) : (
                data.activeRules.map((rule) => (
                  <tr key={rule.id} className="border-b border-white/[0.06] hover:bg-white/[0.02]">
                    <td className="py-3 px-4 text-white font-medium">{rule.rule_name}</td>
                    <td className="py-3 px-4">
                      <span className="px-2 py-0.5 bg-white/[0.06] text-xs text-slate-300 rounded">
                        {rule.domain || 'global'}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400">v{rule.version}</td>
                    <td className="py-3 px-4 text-slate-400 text-sm max-w-[300px] truncate">
                      {rule.description}
                    </td>
                    <td className="py-3 px-4 text-slate-500 text-sm">
                      {new Date(rule.activated_at).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => handleAction('rollback_rule', { ruleName: rule.rule_name })}
                        disabled={actionLoading !== null || rule.version <= 1}
                        className="px-2 py-1 text-xs text-orange-400 hover:bg-orange-500/20 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        Rollback
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Audit Log */}
      <GlassCard blur="heavy" className="p-0 overflow-hidden">
        <div className="p-6 border-b border-white/[0.06]">
          <h2 className="text-lg font-bold text-white">Audit Log</h2>
          <p className="text-sm text-slate-400">Recent governance actions</p>
        </div>
        <div className="max-h-[300px] overflow-y-auto">
          {data.auditLog.length === 0 ? (
            <div className="p-8 text-center text-slate-500">
              No audit entries yet
            </div>
          ) : (
            data.auditLog.map((entry) => (
              <div key={entry.id} className="p-4 border-b border-white/[0.06] hover:bg-white/[0.02]">
                <div className="flex justify-between items-start">
                  <div>
                    <span className={`px-2 py-0.5 text-xs rounded ${
                      entry.action.includes('accept') ? 'bg-green-500/20 text-green-400' :
                      entry.action.includes('reject') ? 'bg-red-500/20 text-red-400' :
                      entry.action.includes('rollback') ? 'bg-orange-500/20 text-orange-400' :
                      'bg-blue-500/20 text-blue-400'
                    }`}>
                      {entry.action.replace(/_/g, ' ')}
                    </span>
                    <span className="text-slate-500 text-sm ml-2">
                      {entry.entity_type} {entry.entity_id.slice(0, 8)}...
                    </span>
                  </div>
                  <span className="text-xs text-slate-500">
                    {new Date(entry.performed_at).toLocaleString()}
                  </span>
                </div>
                {entry.reason && (
                  <p className="text-sm text-slate-400 mt-2">{entry.reason}</p>
                )}
              </div>
            ))
          )}
        </div>
      </GlassCard>

      {/* Training Recommendations */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">Training Recommendations</h2>
        <div className="space-y-3">
          {generateTrainingRecommendations(data).map((rec, i) => (
            <div key={i} className="flex items-start gap-3 p-3 bg-black/20 rounded-lg border border-white/[0.06]">
              <span className="text-yellow-400 mt-0.5">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </span>
              <p className="text-sm text-slate-300">{rec}</p>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Proposal Detail Modal */}
      {selectedProposal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/[0.06]">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-xl font-bold text-white">{selectedProposal.proposed_rule_name}</h2>
                  <p className="text-sm text-slate-400 mt-1">Proposed by Elle</p>
                </div>
                <button
                  onClick={() => setSelectedProposal(null)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded-lg"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6 space-y-6">
              {/* Summaries from both parties */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-purple-400 mb-2">Elle&apos;s Summary</h3>
                  <p className="text-sm text-slate-300">{selectedProposal.elle_summary || 'No summary provided'}</p>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-blue-400 mb-2">Guardian&apos;s Analysis</h3>
                  <p className="text-sm text-slate-300">{selectedProposal.guardian_summary || 'Pending analysis'}</p>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-2">Rationale</h3>
                <p className="text-slate-300">{selectedProposal.rationale}</p>
              </div>

              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-2">Proposed Configuration</h3>
                <pre className="bg-black/40 p-4 rounded-lg text-sm text-slate-300 overflow-x-auto">
                  {JSON.stringify(selectedProposal.proposed_config, null, 2)}
                </pre>
              </div>

              {selectedProposal.supporting_evidence && selectedProposal.supporting_evidence.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium text-slate-400 mb-2">Supporting Evidence</h3>
                  <ul className="space-y-2">
                    {selectedProposal.supporting_evidence.map((evidence, i) => (
                      <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                        <span className="text-green-400 mt-1">
                          <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        </span>
                        {evidence}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-2">Review Notes</h3>
                <textarea
                  value={reviewNotes}
                  onChange={(e) => setReviewNotes(e.target.value)}
                  placeholder="Add notes for the audit log..."
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg p-3 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
                  rows={3}
                />
              </div>
            </div>

            <div className="p-6 border-t border-white/[0.06] flex justify-end gap-3">
              <button
                onClick={() => handleAction('reject_proposal', { proposalId: selectedProposal.id, notes: reviewNotes })}
                disabled={actionLoading !== null}
                className="px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 disabled:opacity-50"
              >
                Reject
              </button>
              <button
                onClick={() => handleAction('accept_proposal', { proposalId: selectedProposal.id, notes: reviewNotes })}
                disabled={actionLoading !== null}
                className="px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 disabled:opacity-50"
              >
                Accept & Create Rule
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Disagreement Review Modal */}
      {selectedDisagreement && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[rgba(18,18,26,0.95)] border border-white/[0.1] rounded-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/[0.06]">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-xl font-bold text-white">Review Disagreement</h2>
                  <p className="text-sm text-slate-400 mt-1">Domain: {selectedDisagreement.domain}</p>
                </div>
                <button
                  onClick={() => setSelectedDisagreement(null)}
                  className="p-2 text-slate-400 hover:text-white hover:bg-white/[0.06] rounded-lg"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-2">Input</h3>
                <p className="text-slate-300 bg-black/40 p-4 rounded-lg">{selectedDisagreement.input_summary}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-purple-400 mb-2">Elle&apos;s Decision</h3>
                  <p className="text-white font-medium mb-2">{selectedDisagreement.elle_decision}</p>
                  <p className="text-sm text-slate-300">{selectedDisagreement.elle_reasoning}</p>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4">
                  <h3 className="text-sm font-medium text-blue-400 mb-2">Guardian&apos;s Decision</h3>
                  <p className="text-white font-medium mb-2">{selectedDisagreement.guardian_decision}</p>
                  <p className="text-sm text-slate-300">{selectedDisagreement.guardian_reasoning}</p>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-2">Review Notes</h3>
                <textarea
                  value={reviewNotes}
                  onChange={(e) => setReviewNotes(e.target.value)}
                  placeholder="Explain your ground truth determination..."
                  className="w-full bg-black/40 border border-white/[0.1] rounded-lg p-3 text-white placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
                  rows={3}
                />
              </div>
            </div>

            <div className="p-6 border-t border-white/[0.06]">
              <p className="text-sm text-slate-400 mb-3">Who was correct?</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <button
                  onClick={() => handleAction('review_evaluation', { evaluationId: selectedDisagreement.id, groundTruth: 'elle_correct', notes: reviewNotes })}
                  disabled={actionLoading !== null}
                  className="px-4 py-3 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 disabled:opacity-50 text-sm"
                >
                  Elle Correct
                </button>
                <button
                  onClick={() => handleAction('review_evaluation', { evaluationId: selectedDisagreement.id, groundTruth: 'guardian_correct', notes: reviewNotes })}
                  disabled={actionLoading !== null}
                  className="px-4 py-3 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 disabled:opacity-50 text-sm"
                >
                  Guardian Correct
                </button>
                <button
                  onClick={() => handleAction('review_evaluation', { evaluationId: selectedDisagreement.id, groundTruth: 'both_correct', notes: reviewNotes })}
                  disabled={actionLoading !== null}
                  className="px-4 py-3 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 disabled:opacity-50 text-sm"
                >
                  Both Correct
                </button>
                <button
                  onClick={() => handleAction('review_evaluation', { evaluationId: selectedDisagreement.id, groundTruth: 'both_wrong', notes: reviewNotes })}
                  disabled={actionLoading !== null}
                  className="px-4 py-3 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 disabled:opacity-50 text-sm"
                >
                  Both Wrong
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
