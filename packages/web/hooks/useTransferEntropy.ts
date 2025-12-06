'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  analyzeTransferEntropy,
  computeEdgeWeights,
  type TimeSeriesPoint,
  type TransferEntropyResult,
  type TEConfig,
} from '@/lib/physics/transfer-entropy';

export interface TimeSeriesInput {
  id: string;
  name: string;
  data: TimeSeriesPoint[];
}

export interface CausalEdge extends TransferEntropyResult {
  sourceName: string;
  targetName: string;
}

export interface UseTransferEntropyOptions {
  config?: Partial<TEConfig>;
  autoCompute?: boolean;
  updateInterval?: number;
  significanceThreshold?: number;
}

export interface TransferEntropyState {
  edges: CausalEdge[];
  matrix: number[][];  // Adjacency matrix of TE values
  strongestCauses: Array<{ source: string; target: string; te: number }>;
  isComputing: boolean;
  progress: number;
  lastUpdate: string;
}

/**
 * Hook for computing Transfer Entropy between multiple time series
 *
 * Enables real-time causal edge weighting for knowledge graphs
 * based on directed information flow.
 */
export function useTransferEntropy(
  series: TimeSeriesInput[],
  options: UseTransferEntropyOptions = {}
) {
  const {
    config = {},
    autoCompute = true,
    updateInterval = 30000,
    significanceThreshold = 0.05,
  } = options;

  const [state, setState] = useState<TransferEntropyState>({
    edges: [],
    matrix: [],
    strongestCauses: [],
    isComputing: false,
    progress: 0,
    lastUpdate: new Date().toISOString(),
  });

  const abortRef = useRef<AbortController | null>(null);

  // Compute all pairwise Transfer Entropy values
  const computeAll = useCallback(async () => {
    if (series.length < 2) {
      return;
    }

    // Abort any ongoing computation
    if (abortRef.current) {
      abortRef.current.abort();
    }
    abortRef.current = new AbortController();

    setState(prev => ({ ...prev, isComputing: true, progress: 0 }));

    try {
      const edges: CausalEdge[] = [];
      const n = series.length;
      const totalPairs = n * (n - 1);
      let completed = 0;

      // Create name lookup
      const nameMap = new Map(series.map(s => [s.id, s.name]));

      // Compute pairwise TE
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) continue;

          const result = await analyzeTransferEntropy(
            series[i].id,
            series[j].id,
            series[i].data,
            series[j].data,
            { ...config, alpha: significanceThreshold }
          );

          if (result.significance < significanceThreshold) {
            edges.push({
              ...result,
              sourceName: nameMap.get(result.sourceId) || result.sourceId,
              targetName: nameMap.get(result.targetId) || result.targetId,
            });
          }

          completed++;
          setState(prev => ({
            ...prev,
            progress: Math.round((completed / totalPairs) * 100),
          }));
        }
      }

      // Build adjacency matrix
      const matrix: number[][] = Array.from({ length: n }, () =>
        new Array(n).fill(0)
      );

      const idxMap = new Map(series.map((s, i) => [s.id, i]));
      for (const edge of edges) {
        const i = idxMap.get(edge.sourceId);
        const j = idxMap.get(edge.targetId);
        if (i !== undefined && j !== undefined) {
          matrix[i][j] = edge.normalizedTE;
        }
      }

      // Find strongest causal relationships
      const strongestCauses = edges
        .sort((a, b) => b.transferEntropy - a.transferEntropy)
        .slice(0, 10)
        .map(e => ({
          source: e.sourceName,
          target: e.targetName,
          te: e.transferEntropy,
        }));

      setState({
        edges,
        matrix,
        strongestCauses,
        isComputing: false,
        progress: 100,
        lastUpdate: new Date().toISOString(),
      });
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        console.error('Transfer Entropy computation error:', error);
      }
      setState(prev => ({ ...prev, isComputing: false }));
    }
  }, [series, config, significanceThreshold]);

  // Auto-compute on mount and when series change
  useEffect(() => {
    if (autoCompute && series.length >= 2) {
      computeAll();
    }
  }, [autoCompute, series, computeAll]);

  // Periodic updates
  useEffect(() => {
    if (!autoCompute || updateInterval <= 0) return;

    const interval = setInterval(() => {
      computeAll();
    }, updateInterval);

    return () => clearInterval(interval);
  }, [autoCompute, updateInterval, computeAll]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  // Get edges for a specific node
  const getEdgesForNode = useCallback((nodeId: string) => {
    const incoming = state.edges.filter(e => e.targetId === nodeId);
    const outgoing = state.edges.filter(e => e.sourceId === nodeId);
    return { incoming, outgoing };
  }, [state.edges]);

  // Get TE value between two specific nodes
  const getTE = useCallback((sourceId: string, targetId: string): number | null => {
    const edge = state.edges.find(
      e => e.sourceId === sourceId && e.targetId === targetId
    );
    return edge?.transferEntropy ?? null;
  }, [state.edges]);

  // Compute for a single pair on demand
  const computePair = useCallback(async (
    source: TimeSeriesInput,
    target: TimeSeriesInput
  ): Promise<TransferEntropyResult> => {
    return analyzeTransferEntropy(
      source.id,
      target.id,
      source.data,
      target.data,
      config
    );
  }, [config]);

  return {
    ...state,
    computeAll,
    getEdgesForNode,
    getTE,
    computePair,
  };
}
