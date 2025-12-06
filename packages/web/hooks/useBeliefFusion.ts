'use client';

import { useState, useCallback, useMemo } from 'react';
import {
  fuseIntelligenceAssessments,
  identifyConflictingSources,
  pignisticTransform,
  computeBeliefIntervals,
  REGIME_FRAME,
  type SourceAssessment,
  type BeliefInterval,
  type BeliefFunction,
} from '@/lib/physics/dempster-shafer';

export interface IntelligenceSource {
  id: string;
  name: string;
  type: 'HUMINT' | 'SIGINT' | 'OSINT' | 'GEOINT' | 'MASINT' | 'TECHINT';
  reliability: number;  // 0-1
}

export interface SourceInput extends IntelligenceSource {
  assessment: {
    stable: number;
    volatile: number;
    crisis: number;
    uncertainty: number;
  };
}

export interface FusionState {
  recommendation: string;
  confidence: number;
  conflict: number;
  beliefIntervals: BeliefInterval[];
  probabilities: Map<string, number>;
  conflictingSources: Array<{ source1: string; source2: string; conflict: number }>;
  fusedBelief: BeliefFunction | null;
  sourceCount: number;
}

/**
 * Hook for Dempster-Shafer belief fusion
 *
 * Combines multiple intelligence source assessments into
 * a unified belief state with explicit uncertainty handling.
 */
export function useBeliefFusion(sources: SourceInput[]) {
  const [state, setState] = useState<FusionState>({
    recommendation: 'STABLE',
    confidence: 0,
    conflict: 0,
    beliefIntervals: [],
    probabilities: new Map(),
    conflictingSources: [],
    fusedBelief: null,
    sourceCount: 0,
  });

  // Convert sources to assessments
  const assessments = useMemo<SourceAssessment[]>(() => {
    return sources.map(source => ({
      sourceId: source.id,
      reliability: source.reliability,
      stableConfidence: source.assessment.stable,
      volatileConfidence: source.assessment.volatile,
      crisisConfidence: source.assessment.crisis,
      uncertainty: source.assessment.uncertainty,
    }));
  }, [sources]);

  // Fuse assessments
  const fuse = useCallback(() => {
    if (assessments.length === 0) {
      setState({
        recommendation: 'STABLE',
        confidence: 0,
        conflict: 0,
        beliefIntervals: REGIME_FRAME.map(e => ({
          element: e,
          belief: 0,
          plausibility: 1,
          uncertainty: 1,
        })),
        probabilities: new Map(REGIME_FRAME.map(e => [e, 1 / REGIME_FRAME.length])),
        conflictingSources: [],
        fusedBelief: null,
        sourceCount: 0,
      });
      return;
    }

    try {
      const result = fuseIntelligenceAssessments(assessments);
      const conflicts = identifyConflictingSources(assessments);

      setState({
        recommendation: result.recommendation,
        confidence: result.confidence,
        conflict: result.conflict,
        beliefIntervals: result.beliefIntervals,
        probabilities: result.probabilities,
        conflictingSources: conflicts,
        fusedBelief: result.fusedBelief,
        sourceCount: assessments.length,
      });
    } catch (error) {
      console.error('Belief fusion error:', error);
    }
  }, [assessments]);

  // Auto-fuse when sources change
  useMemo(() => {
    fuse();
  }, [fuse]);

  // Get belief interval for specific regime
  const getBeliefInterval = useCallback((regime: string): BeliefInterval | null => {
    return state.beliefIntervals.find(i => i.element === regime) || null;
  }, [state.beliefIntervals]);

  // Get probability for specific regime
  const getProbability = useCallback((regime: string): number => {
    return state.probabilities.get(regime) || 0;
  }, [state.probabilities]);

  // Check if specific sources conflict
  const checkConflict = useCallback((source1Id: string, source2Id: string): number => {
    const conflict = state.conflictingSources.find(
      c => (c.source1 === source1Id && c.source2 === source2Id) ||
           (c.source1 === source2Id && c.source2 === source1Id)
    );
    return conflict?.conflict || 0;
  }, [state.conflictingSources]);

  // Get most reliable source
  const getMostReliableSource = useCallback((): SourceInput | null => {
    if (sources.length === 0) return null;
    return sources.reduce((best, current) =>
      current.reliability > best.reliability ? current : best
    );
  }, [sources]);

  // Get uncertainty level
  const getUncertaintyLevel = useCallback((): 'low' | 'medium' | 'high' => {
    const avgUncertainty = state.beliefIntervals.reduce(
      (sum, i) => sum + i.uncertainty,
      0
    ) / state.beliefIntervals.length;

    if (avgUncertainty < 0.2) return 'low';
    if (avgUncertainty < 0.5) return 'medium';
    return 'high';
  }, [state.beliefIntervals]);

  return {
    ...state,
    fuse,
    getBeliefInterval,
    getProbability,
    checkConflict,
    getMostReliableSource,
    getUncertaintyLevel,
  };
}
