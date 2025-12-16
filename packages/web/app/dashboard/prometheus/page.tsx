'use client';

import React, { useState } from 'react';
import { Sparkles, Play, RefreshCw, AlertCircle } from 'lucide-react';
import { CognitivePipeline } from '@/components/prometheus/CognitivePipeline';
import { NoveltyRadar } from '@/components/prometheus/NoveltyRadar';
import {
  executePipeline,
  performArchaeology,
  type PipelineState,
  type ConceptPrimitive,
} from '@/lib/prometheus/cognitive-pipeline';
import {
  detectPatternAnomaly,
  detectTemporalBreak,
  buildKnowledgeQuadrant,
  prioritizeNovelties,
  type NoveltySignal,
} from '@/lib/prometheus/novelty-detection';

// Demo data
const DEMO_OBSERVATIONS = Array.from({ length: 200 }, (_, i) => {
  const base = Math.sin(i * 0.1) * 10;
  const noise = (Math.random() - 0.5) * 5;
  const anomaly = i === 150 ? 30 : 0; // Inject anomaly
  return base + noise + anomaly;
});

const EXISTING_KNOWLEDGE = [
  'Regime transitions follow power-law distributions',
  'Economic stress correlates with political instability',
  'Military buildups precede conflict escalation',
];

export default function PROMETHEUSPage() {
  const [pipelineState, setPipelineState] = useState<PipelineState | null>(null);
  const [noveltySignals, setNoveltySignals] = useState<NoveltySignal[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [targetSubject, setTargetSubject] = useState('Geopolitical Regime Transitions');
  const [activeTab, setActiveTab] = useState<'pipeline' | 'novelty'>('pipeline');

  const runPipeline = async () => {
    setIsRunning(true);

    try {
      const catalyst: ConceptPrimitive = {
        id: 'catalyst_thermodynamics',
        domain: 'Statistical Mechanics',
        name: 'Phase Transition Dynamics',
        definition: 'The study of abrupt changes in system state driven by order parameters crossing critical thresholds',
        axioms: [
          'Systems minimize free energy',
          'Phase transitions occur at critical points',
          'Order parameters characterize symmetry breaking',
        ],
        connections: ['Landau-Ginzburg theory', 'Critical phenomena', 'Spontaneous symmetry breaking'],
      };

      const state = await executePipeline(
        targetSubject,
        ['Information Theory', 'Network Science', 'Dynamical Systems'],
        catalyst
      );

      setPipelineState(state);
    } catch (error) {
      console.error('Pipeline error:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const runNoveltyDetection = () => {
    const patternAnomalies = detectPatternAnomaly(DEMO_OBSERVATIONS);
    const temporalBreaks = detectTemporalBreak(DEMO_OBSERVATIONS);
    const allSignals = [...patternAnomalies, ...temporalBreaks];
    const prioritized = prioritizeNovelties(allSignals);
    setNoveltySignals(prioritized);
  };

  const quadrant = buildKnowledgeQuadrant(noveltySignals, EXISTING_KNOWLEDGE);

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-md bg-gradient-to-br from-purple-500/20 to-cyan-500/20 border border-purple-500/30">
            <Sparkles size={28} className="text-purple-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white font-mono">
              P.R.O.M.E.T.H.E.U.S.
            </h1>
            <p className="text-sm text-slate-400">
              Protocol for Recursive Optimization & Meta-Enhanced Theoretical Extraction
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={runNoveltyDetection}
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-600 text-sm font-mono text-slate-300 transition-colors"
          >
            <RefreshCw size={16} />
            Scan Novelty
          </button>
          <button
            onClick={() => void runPipeline()}
            disabled={isRunning}
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-sm font-mono text-white transition-all disabled:opacity-50"
          >
            {isRunning ? (
              <RefreshCw size={16} className="animate-spin" />
            ) : (
              <Play size={16} />
            )}
            Execute Pipeline
          </button>
        </div>
      </div>

      {/* Target Subject Input */}
      <div className="p-4 rounded-md bg-slate-800/50 border border-slate-700">
        <label className="block text-xs font-mono text-slate-400 uppercase tracking-wider mb-2">
          Target Subject
        </label>
        <input
          type="text"
          value={targetSubject}
          onChange={(e) => setTargetSubject(e.target.value)}
          className="w-full px-4 py-3 rounded-md bg-slate-900 border border-slate-700 text-white font-mono focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none"
          placeholder="Enter subject for knowledge extraction..."
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-700">
        <button
          onClick={() => setActiveTab('pipeline')}
          className={`px-4 py-2 text-sm font-mono border-b-2 transition-colors ${
            activeTab === 'pipeline'
              ? 'text-cyan-400 border-cyan-400'
              : 'text-slate-400 border-transparent hover:text-slate-300'
          }`}
        >
          Cognitive Pipeline
        </button>
        <button
          onClick={() => setActiveTab('novelty')}
          className={`px-4 py-2 text-sm font-mono border-b-2 transition-colors ${
            activeTab === 'novelty'
              ? 'text-purple-400 border-purple-400'
              : 'text-slate-400 border-transparent hover:text-slate-300'
          }`}
        >
          Novelty Radar
        </button>
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {activeTab === 'pipeline' ? (
          <>
            {/* Pipeline Visualization */}
            <div className="p-6 rounded-md bg-slate-800/30 border border-slate-700">
              {pipelineState ? (
                <CognitivePipeline state={pipelineState} />
              ) : (
                <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                  <Sparkles size={48} className="mb-4 opacity-30" />
                  <p className="text-sm font-mono">Execute pipeline to begin extraction</p>
                  <p className="text-xs mt-2 text-slate-600">
                    The PROMETHEUS protocol will analyze the target subject
                  </p>
                </div>
              )}
            </div>

            {/* Gradients of Ignorance */}
            <div className="p-6 rounded-md bg-slate-800/30 border border-slate-700">
              <h3 className="text-sm font-mono font-bold text-slate-300 mb-4 flex items-center gap-2">
                <AlertCircle size={16} className="text-amber-400" />
                Gradients of Ignorance
              </h3>
              {pipelineState ? (
                <div className="space-y-3">
                  {performArchaeology(targetSubject, ['Information Theory', 'Network Science']).map((gradient, i) => (
                    <div
                      key={i}
                      className="p-3 rounded-md bg-slate-900/50 border border-slate-700"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-mono text-slate-300">
                          {gradient.topic}
                        </span>
                        <span className="text-xs font-mono text-amber-400">
                          Priority: {(gradient.excavationPriority * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-500"
                          style={{ width: `${gradient.excavationPriority * 100}%` }}
                        />
                      </div>
                      <div className="mt-2 text-xs text-slate-500">
                        {gradient.unknownKnowns.length} unknown knowns identified
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-500 text-sm">
                  Run pipeline to identify knowledge gaps
                </div>
              )}
            </div>
          </>
        ) : (
          <>
            {/* Novelty Radar */}
            <div className="p-6 rounded-md bg-slate-800/30 border border-slate-700">
              <NoveltyRadar
                signals={noveltySignals}
                quadrant={quadrant}
                maxDisplay={8}
                onSignalClick={(signal) => {
                  console.log('Investigating signal:', signal);
                }}
              />
            </div>

            {/* Knowledge Quadrant Details */}
            <div className="p-6 rounded-md bg-slate-800/30 border border-slate-700">
              <h3 className="text-sm font-mono font-bold text-slate-300 mb-4">
                Epistemic Quadrants
              </h3>
              <div className="space-y-4">
                {(Object.entries(quadrant) as [string, string[]][]).map(([key, items]) => {
                  const labels: Record<string, { title: string; color: string }> = {
                    knownKnowns: { title: 'Known Knowns', color: 'text-green-400' },
                    knownUnknowns: { title: 'Known Unknowns', color: 'text-blue-400' },
                    unknownUnknowns: { title: 'Unknown Unknowns', color: 'text-red-400' },
                    unknownKnowns: { title: 'Unknown Knowns', color: 'text-purple-400' },
                  };
                  const config = labels[key];

                  return (
                    <div key={key} className="p-3 rounded-md bg-slate-900/50">
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-sm font-mono ${config.color}`}>
                          {config.title}
                        </span>
                        <span className="text-xs font-mono text-slate-500">
                          {items.length} items
                        </span>
                      </div>
                      {items.length > 0 ? (
                        <ul className="space-y-1">
                          {items.slice(0, 3).map((item, i) => (
                            <li key={i} className="text-xs text-slate-400 truncate">
                              • {item}
                            </li>
                          ))}
                          {items.length > 3 && (
                            <li className="text-xs text-slate-500">
                              +{items.length - 3} more
                            </li>
                          )}
                        </ul>
                      ) : (
                        <p className="text-xs text-slate-500 italic">No items</p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Protocol Description */}
      <div className="p-6 rounded-md bg-gradient-to-br from-purple-900/20 to-cyan-900/20 border border-purple-700/30">
        <h3 className="text-sm font-mono font-bold text-purple-400 mb-3">
          About P.R.O.M.E.T.H.E.U.S.
        </h3>
        <p className="text-sm text-slate-400 leading-relaxed">
          The <strong className="text-purple-300">Protocol for Recursive Optimization, Meta-Enhanced
          Theoretical Heuristic Extraction, and Universal Synthesis</strong> is a 5-stage cognitive
          pipeline designed to extract "Unknown Knowns" - insights that exist implicitly in the
          high-dimensional relationships between data points but have never been explicitly surfaced.
        </p>
        <div className="mt-4 grid grid-cols-5 gap-2">
          {['Archaeology', 'Synthesis', 'Validation', 'Operationalization', 'Output'].map((stage, i) => (
            <div key={stage} className="text-center">
              <div className="w-8 h-8 mx-auto rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-xs font-mono text-purple-400">
                {i + 1}
              </div>
              <div className="mt-1 text-[10px] font-mono text-slate-500">{stage}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
