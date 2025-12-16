'use client';

import { useState, useEffect } from 'react';
import { BookOpen, Shield, GitBranch, FlaskConical, ChevronRight, Lock, RefreshCw, AlertTriangle } from 'lucide-react';
import { Card, Button, EmptyState, Skeleton, SkeletonList } from '@/components/ui';

interface DoctrineRule {
  id: string;
  name: string;
  category: 'signal_interpretation' | 'analytic_judgment' | 'policy_logic' | 'narrative';
  description: string;
  rule_definition: {
    type: string;
    parameters: Record<string, unknown>;
  };
  rationale: string;
  version: number;
  effective_from: string;
}

interface ShadowResult {
  doctrine: string;
  events_evaluated: number;
  divergence_count: number;
  divergence_rate: string;
  recommendation: string;
}

const categoryIcons: Record<string, typeof BookOpen> = {
  signal_interpretation: BookOpen,
  analytic_judgment: GitBranch,
  policy_logic: Shield,
  narrative: BookOpen
};

const categoryStyles: Record<string, string> = {
  signal_interpretation: 'text-blue-400 border-blue-600/30 bg-blue-600/5',
  analytic_judgment: 'text-purple-400 border-purple-600/30 bg-purple-600/5',
  policy_logic: 'text-emerald-400 border-emerald-600/30 bg-emerald-600/5',
  narrative: 'text-amber-400 border-amber-600/30 bg-amber-600/5'
};

export default function DoctrinePage() {
  const [doctrines, setDoctrines] = useState<DoctrineRule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedDoctrine, setSelectedDoctrine] = useState<DoctrineRule | null>(null);
  const [shadowResult, setShadowResult] = useState<ShadowResult | null>(null);
  const [shadowLoading, setShadowLoading] = useState(false);

  useEffect(() => {
    fetchDoctrines();
  }, []);

  async function fetchDoctrines() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/doctrine', {
        headers: {
          'x-user-tier': 'enterprise_tier'
        }
      });

      if (res.status === 403) {
        setError('Doctrine access requires Integrated or Stewardship tier');
        return;
      }

      const data = await res.json();
      setDoctrines(data.doctrines || []);
    } catch (e) {
      setError('Failed to load doctrines');
    } finally {
      setLoading(false);
    }
  }

  async function runShadowEvaluation(doctrine: DoctrineRule) {
    setShadowLoading(true);
    setShadowResult(null);
    try {
      const res = await fetch('/api/doctrine/shadow', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-user-tier': 'enterprise_tier'
        },
        body: JSON.stringify({
          doctrine_id: doctrine.id,
          proposed_parameters: {
            ...Object.fromEntries(
              Object.entries(doctrine.rule_definition.parameters).map(([k, v]) => [
                k,
                typeof v === 'number' ? v * 1.1 : v
              ])
            )
          },
          evaluation_days: 7
        })
      });

      const data = await res.json();
      setShadowResult(data.summary);
    } catch (e) {
      console.error('Shadow evaluation failed:', e);
    } finally {
      setShadowLoading(false);
    }
  }

  const categories = [...new Set(doctrines.map(d => d.category))];
  const filteredDoctrines = selectedCategory
    ? doctrines.filter(d => d.category === selectedCategory)
    : doctrines;

  if (loading) {
    return (
      <div className="space-y-5">
        <div>
          <Skeleton className="h-5 w-40 mb-2" />
          <Skeleton className="h-4 w-64" />
        </div>
        <SkeletonList items={5} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-5">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Doctrine Registry</h1>
          <p className="text-sm text-slate-500 mt-0.5">Rule sets governing intelligence computation</p>
        </div>

        <Card padding="lg" className="text-center">
          <Lock className="w-10 h-10 text-amber-500 mx-auto mb-4" />
          <h2 className="text-base font-semibold text-slate-200 mb-2">Access Restricted</h2>
          <p className="text-sm text-slate-500 max-w-md mx-auto mb-2">{error}</p>
          <p className="text-xs text-slate-600">
            Doctrine access is available to Integrated and Stewardship tier customers.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Doctrine Registry</h1>
          <p className="text-sm text-slate-500 mt-0.5">Rule sets governing intelligence computation</p>
        </div>
        <Button variant="secondary" size="sm" onClick={fetchDoctrines} loading={loading}>
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
          Refresh
        </Button>
      </div>

      {/* Category Filter */}
      <div className="flex gap-1.5 flex-wrap border-b border-slate-800 pb-3">
        <button
          onClick={() => setSelectedCategory(null)}
          className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
            !selectedCategory ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
          }`}
        >
          All ({doctrines.length})
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
              selectedCategory === cat ? 'bg-slate-800 text-slate-200' : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {cat.replace(/_/g, ' ')} ({doctrines.filter(d => d.category === cat).length})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Doctrine List */}
        <div className="lg:col-span-2 space-y-2">
          {filteredDoctrines.length === 0 ? (
            <EmptyState
              icon={BookOpen}
              title="No doctrines found"
              description="No doctrines match the current filter criteria."
              action={{ label: 'Clear filter', onClick: () => setSelectedCategory(null) }}
            />
          ) : (
            filteredDoctrines.map((doctrine) => {
              const Icon = categoryIcons[doctrine.category] || BookOpen;
              const styleClass = categoryStyles[doctrine.category] || 'text-slate-400 border-slate-700 bg-slate-800';

              return (
                <Card
                  key={doctrine.id}
                  padding="sm"
                  interactive
                  selected={selectedDoctrine?.id === doctrine.id}
                  onClick={() => setSelectedDoctrine(doctrine)}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-1.5 rounded border ${styleClass}`}>
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-sm font-medium text-slate-200">{doctrine.name}</span>
                        <span className="text-[10px] text-slate-600">v{doctrine.version}</span>
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-2">{doctrine.description}</p>
                      <div className="flex items-center gap-3 mt-1.5 text-[10px] text-slate-600">
                        <span>{doctrine.category.replace(/_/g, ' ')}</span>
                        <span>Since {new Date(doctrine.effective_from).toLocaleDateString()}</span>
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-600 shrink-0" />
                  </div>
                </Card>
              );
            })
          )}
        </div>

        {/* Detail Panel */}
        <div className="space-y-4">
          {selectedDoctrine ? (
            <>
              <Card padding="md">
                <h2 className="text-sm font-semibold text-slate-200 mb-4">{selectedDoctrine.name}</h2>

                <div className="space-y-4">
                  <div>
                    <p className="text-[10px] text-slate-600 uppercase tracking-wide mb-1">Description</p>
                    <p className="text-xs text-slate-400">{selectedDoctrine.description}</p>
                  </div>

                  <div>
                    <p className="text-[10px] text-slate-600 uppercase tracking-wide mb-1">Rationale</p>
                    <p className="text-xs text-slate-400">{selectedDoctrine.rationale}</p>
                  </div>

                  <div>
                    <p className="text-[10px] text-slate-600 uppercase tracking-wide mb-2">Parameters</p>
                    <div className="bg-slate-800/50 rounded p-2.5 font-mono text-[10px]">
                      {Object.entries(selectedDoctrine.rule_definition.parameters).map(([key, value]) => (
                        <div key={key} className="flex justify-between py-0.5">
                          <span className="text-slate-500">{key}:</span>
                          <span className="text-slate-300">
                            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-slate-600 pt-2 border-t border-slate-800">
                    <span>Type: {selectedDoctrine.rule_definition.type}</span>
                    <span>Version {selectedDoctrine.version}</span>
                  </div>
                </div>
              </Card>

              {/* Shadow Evaluation */}
              <Card padding="md">
                <div className="flex items-center gap-2 mb-3">
                  <FlaskConical className="w-4 h-4 text-slate-400" />
                  <h3 className="text-sm font-semibold text-slate-200">Shadow Evaluation</h3>
                </div>

                <p className="text-xs text-slate-500 mb-3">
                  Test parameter changes against historical outputs without affecting production.
                </p>

                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => runShadowEvaluation(selectedDoctrine)}
                  loading={shadowLoading}
                  className="w-full"
                >
                  <FlaskConical className="w-3.5 h-3.5 mr-1.5" />
                  Run +10% Sensitivity Test
                </Button>

                {shadowResult && (
                  <div className="mt-3 p-2.5 bg-slate-800/50 rounded">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className={`w-3.5 h-3.5 ${
                        shadowResult.recommendation.includes('HIGH') ? 'text-red-500' :
                        shadowResult.recommendation.includes('MODERATE') ? 'text-amber-500' :
                        'text-emerald-500'
                      }`} />
                      <span className="text-xs font-medium text-slate-300">
                        {shadowResult.recommendation.split(' - ')[0]}
                      </span>
                    </div>
                    <div className="space-y-1 text-[10px]">
                      <div className="flex justify-between">
                        <span className="text-slate-500">Events evaluated:</span>
                        <span className="text-slate-300">{shadowResult.events_evaluated}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Divergent outputs:</span>
                        <span className="text-slate-300">{shadowResult.divergence_count}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-500">Divergence rate:</span>
                        <span className="text-slate-300">{shadowResult.divergence_rate}</span>
                      </div>
                    </div>
                  </div>
                )}
              </Card>
            </>
          ) : (
            <Card padding="md" className="text-center py-8">
              <BookOpen className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-sm text-slate-500">Select a doctrine to view details</p>
            </Card>
          )}
        </div>
      </div>

      {/* Info Footer */}
      <Card padding="sm" className="border-dashed border-slate-700">
        <div className="flex items-center gap-3 text-slate-500">
          <Shield className="w-4 h-4 shrink-0" />
          <p className="text-xs">
            Doctrines define how LatticeForge interprets signals and produces judgments.
            Changes require review cycles per your governance agreement.
          </p>
        </div>
      </Card>
    </div>
  );
}
