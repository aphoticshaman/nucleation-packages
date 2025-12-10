'use client';

import { useState, useEffect } from 'react';
import { loadWasm, WasmCore, isWasmLoaded, getWasm } from '@/lib/wasm';

// Re-export for convenience
export type { WasmCore } from '@/lib/wasm';

interface UseWasmResult {
  wasm: WasmCore | null;
  loading: boolean;
  error: Error | null;
}

/**
 * React hook for loading and using LatticeForge WASM module.
 *
 * Provides:
 * - CIC Framework (Compression-Integration-Coherence)
 * - GTVC Clustering (Gauge-Theoretic Value Clustering)
 * - Q-matrix operations for regime transitions
 * - Geospatial system with nation dynamics
 */
export function useWasm(): UseWasmResult {
  const [wasm, setWasm] = useState<WasmCore | null>(() =>
    isWasmLoaded() ? ({ available: true, module: getWasm() ?? undefined } as WasmCore) : null
  );
  const [loading, setLoading] = useState(!isWasmLoaded());
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    // Already loaded
    if (isWasmLoaded()) {
      setWasm({ available: true, module: getWasm() ?? undefined } as WasmCore);
      setLoading(false);
      return;
    }

    // Load WASM module
    let cancelled = false;

    loadWasm()
      .then((result) => {
        if (!cancelled) {
          setWasm(result);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error('Failed to load WASM'));
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { wasm, loading, error };
}
