'use client';

import { useState, useEffect } from 'react';
import { BookOpen, Shield, GitBranch, FlaskConical, ChevronRight, Lock, RefreshCw, AlertTriangle } from 'lucide-react';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';

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

const categoryColors: Record<string, string> = {
  signal_interpretation: 'text-blue-400 bg-blue-500/10',
  analytic_judgment: 'text-purple-400 bg-purple-500/10',
  policy_logic: 'text-green-400 bg-green-500/10',
  narrative: 'text-amber-400 bg-amber-500/10'
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
      // In a real app, this would include auth headers
      const res = await fetch('/api/doctrine', {
        headers: {
          'x-user-tier': 'enterprise_tier' // Demo: simulate enterprise tier
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
            // Example: adjust threshold by 10%
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
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 text-slate-500 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-white">Doctrine Registry</h1>
          <p className="text-slate-400 mt-1">Rule sets governing intelligence computation</p>
        </div>

        <GlassCard className="p-8 text-center">
          <Lock className="w-12 h-12 text-amber-400 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">Access Restricted</h2>
          <p className="text-slate-400 max-w-md mx-auto mb-4">{error}</p>
          <p className="text-sm text-slate-500">
            Doctrine access is available to Integrated and Stewardship tier customers.
          </p>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Doctrine Registry</h1>
          <p className="text-slate-400 mt-1">Rule sets governing intelligence computation</p>
        </div>
        <GlassButton variant="secondary" size="sm" onClick={fetchDoctrines}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </GlassButton>
      </div>

      {/* Category Filter */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={() => setSelectedCategory(null)}
          className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
            !selectedCategory ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'
          }`}
        >
          All ({doctrines.length})
        </button>
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
              selectedCategory === cat ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            {cat.replace(/_/g, ' ')} ({doctrines.filter(d => d.category === cat).length})
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Doctrine List */}
        <div className="lg:col-span-2 space-y-3">
          {filteredDoctrines.map((doctrine) => {
            const Icon = categoryIcons[doctrine.category] || BookOpen;
            const colorClass = categoryColors[doctrine.category] || 'text-slate-400 bg-slate-500/10';

            return (
              <GlassCard
                key={doctrine.id}
                className={`p-4 cursor-pointer transition-all hover:bg-white/5 ${
                  selectedDoctrine?.id === doctrine.id ? 'border-cyan-500/50' : ''
                }`}
                onClick={() => setSelectedDoctrine(doctrine)}
              >
                <div className="flex items-start gap-4">
                  <div className={`p-2 rounded-lg ${colorClass}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-white">{doctrine.name}</span>
                      <span className="text-xs text-slate-500">v{doctrine.version}</span>
                    </div>
                    <p className="text-sm text-slate-400 line-clamp-2">{doctrine.description}</p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
                      <span>{doctrine.category.replace(/_/g, ' ')}</span>
                      <span>Since {new Date(doctrine.effective_from).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-slate-500" />
                </div>
              </GlassCard>
            );
          })}
        </div>

        {/* Detail Panel */}
        <div className="space-y-4">
          {selectedDoctrine ? (
            <>
              <GlassCard blur="heavy">
                <h2 className="text-lg font-bold text-white mb-4">{selectedDoctrine.name}</h2>

                <div className="space-y-4">
                  <div>
                    <p className="text-xs text-slate-500 mb-1">Description</p>
                    <p className="text-sm text-slate-300">{selectedDoctrine.description}</p>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500 mb-1">Rationale</p>
                    <p className="text-sm text-slate-300">{selectedDoctrine.rationale}</p>
                  </div>

                  <div>
                    <p className="text-xs text-slate-500 mb-2">Parameters</p>
                    <div className="bg-black/20 rounded-lg p-3 font-mono text-xs">
                      {Object.entries(selectedDoctrine.rule_definition.parameters).map(([key, value]) => (
                        <div key={key} className="flex justify-between py-1">
                          <span className="text-slate-400">{key}:</span>
                          <span className="text-cyan-400">
                            {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>Type: {selectedDoctrine.rule_definition.type}</span>
                    <span>Version {selectedDoctrine.version}</span>
                  </div>
                </div>
              </GlassCard>

              {/* Shadow Evaluation */}
              <GlassCard blur="heavy">
                <div className="flex items-center gap-2 mb-4">
                  <FlaskConical className="w-5 h-5 text-purple-400" />
                  <h3 className="text-lg font-bold text-white">Shadow Evaluation</h3>
                </div>

                <p className="text-sm text-slate-400 mb-4">
                  Test how parameter changes would affect historical outputs without changing production.
                </p>

                <GlassButton
                  variant="secondary"
                  size="sm"
                  onClick={() => runShadowEvaluation(selectedDoctrine)}
                  disabled={shadowLoading}
                  className="w-full"
                >
                  {shadowLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      Evaluating...
                    </>
                  ) : (
                    <>
                      <FlaskConical className="w-4 h-4 mr-2" />
                      Run +10% Sensitivity Test
                    </>
                  )}
                </GlassButton>

                {shadowResult && (
                  <div className="mt-4 p-3 bg-black/20 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className={`w-4 h-4 ${
                        shadowResult.recommendation.includes('HIGH') ? 'text-red-400' :
                        shadowResult.recommendation.includes('MODERATE') ? 'text-amber-400' :
                        'text-green-400'
                      }`} />
                      <span className="text-sm font-medium text-white">
                        {shadowResult.recommendation.split(' - ')[0]}
                      </span>
                    </div>
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between text-slate-400">
                        <span>Events evaluated:</span>
                        <span className="text-white">{shadowResult.events_evaluated}</span>
                      </div>
                      <div className="flex justify-between text-slate-400">
                        <span>Divergent outputs:</span>
                        <span className="text-white">{shadowResult.divergence_count}</span>
                      </div>
                      <div className="flex justify-between text-slate-400">
                        <span>Divergence rate:</span>
                        <span className="text-white">{shadowResult.divergence_rate}</span>
                      </div>
                    </div>
                  </div>
                )}
              </GlassCard>
            </>
          ) : (
            <GlassCard blur="heavy" className="p-8 text-center">
              <BookOpen className="w-10 h-10 text-slate-500 mx-auto mb-3" />
              <p className="text-slate-400">Select a doctrine to view details</p>
            </GlassCard>
          )}
        </div>
      </div>

      {/* Info */}
      <GlassCard className="p-4 border-dashed">
        <div className="flex items-center gap-3 text-slate-400">
          <Shield className="w-5 h-5" />
          <div>
            <p className="text-sm font-medium">Doctrine Governance</p>
            <p className="text-xs">
              Doctrines define how LatticeForge interprets signals and produces judgments.
              Changes require review cycles per your governance agreement.
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
