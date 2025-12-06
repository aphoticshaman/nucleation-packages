'use client';

import React from 'react';
import {
  Search,
  Zap,
  CheckCircle,
  Code,
  FileText,
  ChevronRight,
  AlertCircle,
  Loader2,
  type LucideIcon,
} from 'lucide-react';
import type { PROMETHEUSStage, PipelineState } from '@/lib/prometheus/cognitive-pipeline';

interface CognitivePipelineProps {
  state: PipelineState;
  onStageClick?: (stage: PROMETHEUSStage) => void;
}

const STAGES: {
  id: PROMETHEUSStage;
  name: string;
  description: string;
  icon: LucideIcon;
}[] = [
  {
    id: 'ARCHAEOLOGY',
    name: 'Latent Space Archaeology',
    description: 'Deep scan for unknown knowns',
    icon: Search,
  },
  {
    id: 'SYNTHESIS',
    name: 'Novel Synthesis Method',
    description: 'Force-fusion of concepts',
    icon: Zap,
  },
  {
    id: 'VALIDATION',
    name: 'Theoretical Validation',
    description: 'Mathematical proof',
    icon: CheckCircle,
  },
  {
    id: 'OPERATIONALIZATION',
    name: 'XYZA Operationalization',
    description: 'Code implementation',
    icon: Code,
  },
  {
    id: 'OUTPUT',
    name: 'Output Generation',
    description: 'Deliverable package',
    icon: FileText,
  },
];

/**
 * PROMETHEUS Cognitive Pipeline Visualization
 *
 * Displays the 5-stage cognitive pipeline progress
 * for novel knowledge extraction.
 */
export function CognitivePipeline({ state, onStageClick }: CognitivePipelineProps) {
  const currentStageIndex = STAGES.findIndex(s => s.id === state.currentStage);

  const getStageStatus = (stageIndex: number): 'completed' | 'active' | 'pending' => {
    if (stageIndex < currentStageIndex) return 'completed';
    if (stageIndex === currentStageIndex) return 'active';
    return 'pending';
  };

  const getStatusColor = (status: 'completed' | 'active' | 'pending') => {
    switch (status) {
      case 'completed':
        return 'text-green-400 border-green-400 bg-green-400/10';
      case 'active':
        return 'text-cyan-400 border-cyan-400 bg-cyan-400/10';
      case 'pending':
        return 'text-slate-500 border-slate-600 bg-slate-800/50';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white font-mono">
            P.R.O.M.E.T.H.E.U.S.
          </h2>
          <p className="text-xs text-slate-400 font-mono">
            Protocol for Recursive Optimization & Meta-Enhanced Synthesis
          </p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold text-cyan-400 font-mono">
            {state.progress}%
          </div>
          <div className="text-xs text-slate-500">Progress</div>
        </div>
      </div>

      {/* Target Subject */}
      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
        <div className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-1">
          Target Subject
        </div>
        <div className="text-lg font-bold text-white">
          {state.targetSubject || 'Not specified'}
        </div>
      </div>

      {/* Pipeline Stages */}
      <div className="space-y-3">
        {STAGES.map((stage, index) => {
          const status = getStageStatus(index);
          const Icon = stage.icon;

          return (
            <div
              key={stage.id}
              className={`relative flex items-center gap-4 p-4 rounded-lg border transition-all ${
                getStatusColor(status)
              } ${onStageClick ? 'cursor-pointer hover:brightness-110' : ''}`}
              onClick={() => onStageClick?.(stage.id)}
            >
              {/* Stage Number */}
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center border-2 ${
                  status === 'completed'
                    ? 'border-green-400 bg-green-400/20'
                    : status === 'active'
                    ? 'border-cyan-400 bg-cyan-400/20'
                    : 'border-slate-600 bg-slate-800'
                }`}
              >
                {status === 'completed' ? (
                  <CheckCircle size={20} className="text-green-400" />
                ) : status === 'active' ? (
                  <Loader2 size={20} className="text-cyan-400 animate-spin" />
                ) : (
                  <span className="text-sm font-mono text-slate-500">{index + 1}</span>
                )}
              </div>

              {/* Stage Info */}
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Icon size={16} className={status === 'pending' ? 'text-slate-500' : ''} />
                  <span className="font-mono font-bold text-sm">{stage.name}</span>
                </div>
                <p className="text-xs text-slate-400 mt-1">{stage.description}</p>
              </div>

              {/* Status indicator */}
              {status === 'active' && (
                <div className="flex items-center gap-2 text-xs font-mono text-cyan-400">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  Processing
                </div>
              )}

              {/* Connector line */}
              {index < STAGES.length - 1 && (
                <div
                  className={`absolute left-[2.25rem] top-[4rem] w-0.5 h-3 ${
                    status === 'completed' ? 'bg-green-400' : 'bg-slate-700'
                  }`}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Artifacts Summary */}
      {state.artifacts.length > 0 && (
        <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
          <div className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">
            Generated Artifacts
          </div>
          <div className="space-y-2">
            {state.artifacts.map(artifact => (
              <div
                key={artifact.id}
                className="flex items-center justify-between p-2 rounded bg-slate-900/50"
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      artifact.validationStatus === 'validated'
                        ? 'bg-green-400'
                        : artifact.validationStatus === 'rejected'
                        ? 'bg-red-400'
                        : 'bg-amber-400'
                    }`}
                  />
                  <span className="text-sm font-mono text-slate-300">
                    {artifact.name}
                  </span>
                </div>
                <span
                  className={`text-xs font-mono px-2 py-0.5 rounded ${
                    artifact.epistemicLabel === 'VALIDATED'
                      ? 'bg-green-400/10 text-green-400'
                      : artifact.epistemicLabel === 'DERIVED'
                      ? 'bg-blue-400/10 text-blue-400'
                      : artifact.epistemicLabel === 'HYPOTHETICAL'
                      ? 'bg-amber-400/10 text-amber-400'
                      : 'bg-red-400/10 text-red-400'
                  }`}
                >
                  {artifact.epistemicLabel}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Outputs */}
      {state.outputs.length > 0 && (
        <div className="p-4 rounded-lg bg-gradient-to-br from-cyan-900/20 to-blue-900/20 border border-cyan-700/50">
          <div className="flex items-center gap-2 text-xs font-mono text-cyan-400 uppercase tracking-wider mb-3">
            <FileText size={14} />
            Generated Outputs
          </div>
          {state.outputs.map((output, i) => (
            <div key={i} className="p-3 rounded bg-slate-900/50 mb-2 last:mb-0">
              <div className="font-bold text-white mb-1">
                {output.breakthrough.name}
                {output.breakthrough.acronym && (
                  <span className="text-cyan-400 ml-2">
                    ({output.breakthrough.acronym})
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mb-2">
                {output.breakthrough.definition}
              </p>
              <div className="flex items-center gap-4 text-xs font-mono">
                <span className="text-slate-500">
                  {output.code.filename}
                </span>
                <span className="text-slate-500">
                  {output.code.language}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Metadata */}
      <div className="flex items-center justify-between text-xs font-mono text-slate-500">
        <span>Started: {new Date(state.metadata.startedAt).toLocaleTimeString()}</span>
        <span>Iterations: {state.metadata.iterationCount}</span>
      </div>
    </div>
  );
}
