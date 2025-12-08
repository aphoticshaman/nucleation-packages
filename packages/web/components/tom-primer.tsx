/**
 * ToM Primer Component
 * Theory of Mind scaffolding for high-complexity AI queries
 *
 * Based on Riedl & Weidmann (2025) insights:
 * - Insight 3: ToM fluctuates moment-to-moment
 * - Insight 22: ToM is orthogonal to intelligence but predicts AI collaboration success
 * - Insight 23: Prompt engineering is cognitive state management
 *
 * This component mechanically forces users into high-ToM states before complex queries.
 */

'use client';

import React, { useState, useCallback } from 'react';
import { X, Brain, AlertTriangle, CheckCircle, ChevronRight } from 'lucide-react';

// =============================================================================
// TYPES
// =============================================================================

export interface ToMPrimerProps {
  /** Whether the primer modal is open */
  isOpen: boolean;
  /** Callback when primer is completed or dismissed */
  onComplete: (responses: ToMResponses) => void;
  /** Callback to dismiss without completing */
  onDismiss: () => void;
  /** The query that triggered the primer */
  queryContext?: string;
  /** Complexity level that triggered the primer */
  complexityLevel?: 'HIGH' | 'CRITICAL';
}

export interface ToMResponses {
  /** What the user thinks the model assumes */
  modelAssumption: string;
  /** What the reality actually is */
  realityCorrection: string;
  /** Predicted model error */
  predictedError: string;
  /** Optional persona shift */
  personaShift?: string;
  /** Time spent on primer (ms) */
  timeSpent: number;
  /** Whether user skipped any steps */
  stepsCompleted: number;
}

type PrimerPhase = 'intro' | 'context' | 'prediction' | 'persona' | 'complete';

// =============================================================================
// COMPONENT
// =============================================================================

export function ToMPrimer({
  isOpen,
  onComplete,
  onDismiss,
  queryContext,
  complexityLevel = 'HIGH',
}: ToMPrimerProps) {
  const [phase, setPhase] = useState<PrimerPhase>('intro');
  const [startTime] = useState(Date.now());
  const [stepsCompleted, setStepsCompleted] = useState(0);

  // Form state
  const [modelAssumption, setModelAssumption] = useState('');
  const [realityCorrection, setRealityCorrection] = useState('');
  const [predictedError, setPredictedError] = useState('');
  const [personaShift, setPersonaShift] = useState('');

  const handleNext = useCallback(() => {
    setStepsCompleted((prev) => prev + 1);

    switch (phase) {
      case 'intro':
        setPhase('context');
        break;
      case 'context':
        setPhase('prediction');
        break;
      case 'prediction':
        setPhase('persona');
        break;
      case 'persona':
        setPhase('complete');
        // Auto-complete after persona
        onComplete({
          modelAssumption,
          realityCorrection,
          predictedError,
          personaShift: personaShift || undefined,
          timeSpent: Date.now() - startTime,
          stepsCompleted: stepsCompleted + 1,
        });
        break;
    }
  }, [
    phase,
    modelAssumption,
    realityCorrection,
    predictedError,
    personaShift,
    startTime,
    stepsCompleted,
    onComplete,
  ]);

  const handleSkip = useCallback(() => {
    onComplete({
      modelAssumption: modelAssumption || '[SKIPPED]',
      realityCorrection: realityCorrection || '[SKIPPED]',
      predictedError: predictedError || 'none',
      personaShift: undefined,
      timeSpent: Date.now() - startTime,
      stepsCompleted,
    });
  }, [
    modelAssumption,
    realityCorrection,
    predictedError,
    startTime,
    stepsCompleted,
    onComplete,
  ]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onDismiss}
      />

      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-[#0a0a0f] border border-cyan-500/30 rounded-lg shadow-2xl shadow-cyan-500/10">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-cyan-500/20">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/10 rounded-lg">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">
                Theory of Mind Primer
              </h2>
              <p className="text-sm text-gray-400">
                {complexityLevel === 'CRITICAL'
                  ? 'Critical complexity detected'
                  : 'High complexity query detected'}
              </p>
            </div>
          </div>
          <button
            onClick={onDismiss}
            className="p-2 text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress */}
        <div className="px-4 py-2 border-b border-cyan-500/10">
          <div className="flex gap-1">
            {['intro', 'context', 'prediction', 'persona'].map((p, i) => (
              <div
                key={p}
                className={`h-1 flex-1 rounded-full transition-colors ${
                  i <= ['intro', 'context', 'prediction', 'persona'].indexOf(phase)
                    ? 'bg-cyan-500'
                    : 'bg-gray-700'
                }`}
              />
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 min-h-[300px]">
          {/* Phase: Intro */}
          {phase === 'intro' && (
            <div className="space-y-4">
              <div className="flex items-start gap-3 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-amber-200 font-medium">
                    Collaboration Optimization Required
                  </p>
                  <p className="text-sm text-amber-200/70 mt-1">
                    Research shows that perspective-taking (Theory of Mind) predicts
                    AI collaboration success, while raw intelligence does not.
                  </p>
                </div>
              </div>

              <div className="space-y-3 text-gray-300">
                <p>
                  This primer will help you enter a{' '}
                  <span className="text-cyan-400">high-ToM state</span> for optimal
                  results.
                </p>
                <p className="text-sm text-gray-400">
                  Based on:{' '}
                  <span className="italic">
                    Riedl & Weidmann (2025) "Quantifying Human-AI Synergy"
                  </span>
                </p>
              </div>

              {queryContext && (
                <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
                  <p className="text-xs text-gray-500 mb-1">Your Query:</p>
                  <p className="text-sm text-gray-300 line-clamp-2">
                    {queryContext}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Phase: Context Isolation */}
          {phase === 'context' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-white font-medium mb-2">
                  Phase 1: Context Isolation
                </h3>
                <p className="text-sm text-gray-400">
                  The AI has training data up to early 2025. It does{' '}
                  <span className="text-red-400">not</span> know your current
                  operational context.
                </p>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    The model assumes...
                  </label>
                  <input
                    type="text"
                    value={modelAssumption}
                    onChange={(e) => setModelAssumption(e.target.value)}
                    placeholder="e.g., 'Standard market conditions apply'"
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-colors"
                  />
                </div>

                <div>
                  <label className="block text-sm text-gray-300 mb-2">
                    ...but the reality is...
                  </label>
                  <input
                    type="text"
                    value={realityCorrection}
                    onChange={(e) => setRealityCorrection(e.target.value)}
                    placeholder="e.g., 'We're in a post-conflict reconstruction phase'"
                    className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-colors"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Phase: Error Prediction */}
          {phase === 'prediction' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-white font-medium mb-2">
                  Phase 2: Error Prediction
                </h3>
                <p className="text-sm text-gray-400">
                  Anticipating model failures activates predictive ToM circuits.
                </p>
              </div>

              <div>
                <label className="block text-sm text-gray-300 mb-3">
                  Given the complexity, the model is most likely to:
                </label>
                <div className="space-y-2">
                  {[
                    { id: 'dates', label: 'Hallucinate specific dates or figures' },
                    { id: 'actors', label: 'Conflate causal actors or events' },
                    { id: 'simplify', label: 'Over-simplify multi-factor dynamics' },
                    { id: 'none', label: 'None - I have provided full context' },
                  ].map((option) => (
                    <label
                      key={option.id}
                      className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        predictedError === option.id
                          ? 'bg-cyan-500/10 border-cyan-500/50'
                          : 'bg-gray-800/50 border-gray-700 hover:border-gray-600'
                      }`}
                    >
                      <input
                        type="radio"
                        name="predictedError"
                        value={option.id}
                        checked={predictedError === option.id}
                        onChange={(e) => setPredictedError(e.target.value)}
                        className="sr-only"
                      />
                      <div
                        className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                          predictedError === option.id
                            ? 'border-cyan-500 bg-cyan-500'
                            : 'border-gray-500'
                        }`}
                      >
                        {predictedError === option.id && (
                          <div className="w-1.5 h-1.5 rounded-full bg-white" />
                        )}
                      </div>
                      <span className="text-gray-300">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Phase: Persona Shift (Optional) */}
          {phase === 'persona' && (
            <div className="space-y-4">
              <div>
                <h3 className="text-white font-medium mb-2">
                  Phase 3: Persona Alignment (Optional)
                </h3>
                <p className="text-sm text-gray-400">
                  Explicit perspective assignment further increases ToM engagement.
                </p>
              </div>

              <div>
                <label className="block text-sm text-gray-300 mb-2">
                  To optimize output, the AI should act as...
                </label>
                <input
                  type="text"
                  value={personaShift}
                  onChange={(e) => setPersonaShift(e.target.value)}
                  placeholder="e.g., 'A Red Team analyst looking for failure modes'"
                  className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none transition-colors"
                />
                <p className="text-xs text-gray-500 mt-2">
                  Leave blank to skip persona assignment
                </p>
              </div>

              <div className="flex items-start gap-3 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-green-200 font-medium">ToM State Elevated</p>
                  <p className="text-sm text-green-200/70 mt-1">
                    You've activated perspective-taking circuits. Your next query
                    will benefit from enhanced human-AI synergy.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-cyan-500/10">
          <button
            onClick={handleSkip}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors text-sm"
          >
            Skip Primer
          </button>

          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500">
              Step {['intro', 'context', 'prediction', 'persona'].indexOf(phase) + 1}{' '}
              of 4
            </span>
            <button
              onClick={handleNext}
              disabled={
                (phase === 'context' && (!modelAssumption || !realityCorrection)) ||
                (phase === 'prediction' && !predictedError)
              }
              className="flex items-center gap-2 px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 disabled:bg-gray-700 disabled:text-gray-500 text-black font-medium rounded-lg transition-colors"
            >
              {phase === 'persona' ? 'Complete' : 'Continue'}
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// HOOK: useToMPrimer
// =============================================================================

export interface UseToMPrimerOptions {
  /** Complexity threshold to trigger primer */
  complexityThreshold?: number;
  /** Whether primer is enabled */
  enabled?: boolean;
}

export function useToMPrimer(options: UseToMPrimerOptions = {}) {
  const { complexityThreshold = 0.7, enabled = true } = options;
  const [isOpen, setIsOpen] = useState(false);
  const [lastResponses, setLastResponses] = useState<ToMResponses | null>(null);

  const triggerPrimer = useCallback(
    (complexity: number): boolean => {
      if (!enabled) return false;
      if (complexity >= complexityThreshold) {
        setIsOpen(true);
        return true;
      }
      return false;
    },
    [enabled, complexityThreshold]
  );

  const handleComplete = useCallback((responses: ToMResponses) => {
    setLastResponses(responses);
    setIsOpen(false);
  }, []);

  const handleDismiss = useCallback(() => {
    setIsOpen(false);
  }, []);

  return {
    isOpen,
    triggerPrimer,
    handleComplete,
    handleDismiss,
    lastResponses,
  };
}

export default ToMPrimer;
