'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Cognitive state from self-hosted LatticeForge inference.
 *
 * XYZA metrics measure cognitive coherence during generation:
 * - X (Coherence): Logical flow and consistency
 * - Y (Complexity): Appropriate depth without over-complication
 * - Z (Reflection): Self-awareness and metacognition
 * - A (Attunement): Alignment with user context/intent
 */
export interface XYZAMetrics {
  coherence_x: number;
  complexity_y: number;
  reflection_z: number;
  attunement_a: number;
  combined_score: number;
  cognitive_level: 'peak' | 'enhanced' | 'normal' | 'degraded';
}

/**
 * Flow state derived from Kuramoto order parameter.
 * Measures synchronization and cognitive engagement.
 */
export interface FlowState {
  level: 'NONE' | 'EMERGING' | 'BUILDING' | 'FLOW' | 'DEEP_FLOW';
  R: number;           // Order parameter [0,1]
  dR_dt?: number;      // Rate of change
  is_flow: boolean;
  is_deep_flow: boolean;
  stability: number;   // How stable is current state
  time_in_state_ms: number;
}

export interface CognitiveState {
  xyza: XYZAMetrics;
  flow_state: FlowState;
  diagnostics?: string[];
  generation_time_ms?: number;
}

interface UseCognitiveStateOptions {
  /** LatticeForge inference endpoint */
  endpoint?: string;
  /** Enable automatic polling */
  enablePolling?: boolean;
  /** Polling interval in ms (default: 5000) */
  pollInterval?: number;
}

interface UseCognitiveStateResult {
  state: CognitiveState | null;
  loading: boolean;
  error: Error | null;
  /** Fetch current cognitive state */
  refetch: () => Promise<void>;
  /** Check if inference server is healthy */
  checkHealth: () => Promise<boolean>;
  /** Generate with cognitive monitoring */
  generate: (prompt: string, options?: GenerateOptions) => Promise<GenerateResult>;
}

interface GenerateOptions {
  max_tokens?: number;
  temperature?: number;
  user_context?: string;
  system_prompt?: string;
}

interface GenerateResult {
  text: string;
  tokens_generated: number;
  generation_time_ms: number;
  cognitive_state: CognitiveState | null;
}

// Default endpoint - can be overridden
const DEFAULT_ENDPOINT = process.env.NEXT_PUBLIC_LATTICEFORGE_ENDPOINT || 'http://localhost:8000';

/**
 * Hook for managing cognitive state from self-hosted LatticeForge inference.
 *
 * Usage:
 * ```tsx
 * const { state, generate, loading } = useCognitiveState();
 *
 * const result = await generate("Analyze the geopolitical situation...");
 * console.log(result.cognitive_state?.xyza.combined_score);
 * ```
 */
export function useCognitiveState(
  options: UseCognitiveStateOptions = {}
): UseCognitiveStateResult {
  const {
    endpoint = DEFAULT_ENDPOINT,
    enablePolling = false,
    pollInterval = 5000,
  } = options;

  const [state, setState] = useState<CognitiveState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch current flow status from server
  const refetch = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${endpoint}/flow-status`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch cognitive state: ${response.status}`);
      }

      const data = await response.json();

      // Map server response to our interface
      if (data.flow_state && data.xyza) {
        setState({
          xyza: {
            coherence_x: data.xyza.coherence_x ?? 0,
            complexity_y: data.xyza.complexity_y ?? 0,
            reflection_z: data.xyza.reflection_z ?? 0,
            attunement_a: data.xyza.attunement_a ?? 0,
            combined_score: data.xyza.combined_score ?? 0,
            cognitive_level: data.xyza.cognitive_level ?? 'normal',
          },
          flow_state: {
            level: data.flow_state.level ?? 'NONE',
            R: data.flow_state.R ?? 0,
            dR_dt: data.flow_state.dR_dt,
            is_flow: data.flow_state.is_flow ?? false,
            is_deep_flow: data.flow_state.is_deep_flow ?? false,
            stability: data.flow_state.stability ?? 0,
            time_in_state_ms: data.flow_state.time_in_state_ms ?? 0,
          },
          diagnostics: data.diagnostics,
          generation_time_ms: data.generation_time_ms,
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch cognitive state'));
      console.error('[CognitiveState] Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  // Health check
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const response = await fetch(`${endpoint}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      return response.ok;
    } catch {
      return false;
    }
  }, [endpoint]);

  // Generate with cognitive monitoring
  const generate = useCallback(async (
    prompt: string,
    opts: GenerateOptions = {}
  ): Promise<GenerateResult> => {
    const {
      max_tokens = 1024,
      temperature = 0.7,
      user_context,
      system_prompt,
    } = opts;

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${endpoint}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens,
          temperature,
          return_cognitive: true,
          user_context,
          system_prompt,
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.status}`);
      }

      const data = await response.json();

      // Update local state with new cognitive metrics
      const cognitiveState: CognitiveState | null = (data.flow_state && data.xyza) ? {
        xyza: {
          coherence_x: data.xyza.coherence_x ?? 0,
          complexity_y: data.xyza.complexity_y ?? 0,
          reflection_z: data.xyza.reflection_z ?? 0,
          attunement_a: data.xyza.attunement_a ?? 0,
          combined_score: data.xyza.combined_score ?? 0,
          cognitive_level: data.xyza.cognitive_level ?? 'normal',
        },
        flow_state: {
          level: data.flow_state.level ?? 'NONE',
          R: data.flow_state.R ?? 0,
          dR_dt: data.flow_state.dR_dt,
          is_flow: data.flow_state.is_flow ?? false,
          is_deep_flow: data.flow_state.is_deep_flow ?? false,
          stability: data.flow_state.stability ?? 0,
          time_in_state_ms: data.flow_state.time_in_state_ms ?? 0,
        },
        diagnostics: data.diagnostics,
        generation_time_ms: data.generation_time_ms,
      } : null;

      if (cognitiveState) {
        setState(cognitiveState);
      }

      return {
        text: data.text ?? '',
        tokens_generated: data.tokens_generated ?? 0,
        generation_time_ms: data.generation_time_ms ?? 0,
        cognitive_state: cognitiveState,
      };
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Generation failed');
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [endpoint]);

  // Setup polling if enabled
  useEffect(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    if (enablePolling) {
      // Initial fetch
      void refetch();

      // Setup interval
      pollIntervalRef.current = setInterval(() => {
        void refetch();
      }, pollInterval);
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [enablePolling, pollInterval, refetch]);

  return {
    state,
    loading,
    error,
    refetch,
    checkHealth,
    generate,
  };
}

/**
 * Get color for cognitive level
 */
export function getCognitiveLevelColor(level: XYZAMetrics['cognitive_level']): string {
  switch (level) {
    case 'peak': return 'text-green-400';
    case 'enhanced': return 'text-blue-400';
    case 'normal': return 'text-amber-400';
    case 'degraded': return 'text-red-400';
    default: return 'text-slate-400';
  }
}

/**
 * Get color for flow state level
 */
export function getFlowLevelColor(level: FlowState['level']): string {
  switch (level) {
    case 'DEEP_FLOW': return 'text-purple-400';
    case 'FLOW': return 'text-green-400';
    case 'BUILDING': return 'text-blue-400';
    case 'EMERGING': return 'text-amber-400';
    case 'NONE': return 'text-slate-400';
    default: return 'text-slate-400';
  }
}
