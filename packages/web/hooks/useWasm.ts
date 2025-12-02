'use client';

import { useState, useEffect } from 'react';
import { loadWasm, WasmCore } from '@/lib/wasm';

interface UseWasmResult {
  wasm: WasmCore | null;
  loading: boolean;
  error: Error | null;
}

export function useWasm(): UseWasmResult {
  const [wasm, setWasm] = useState<WasmCore | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let mounted = true;

    async function init() {
      try {
        const module = await loadWasm();
        if (mounted) {
          setWasm(module);
          setLoading(false);
        }
      } catch (err) {
        if (mounted) {
          setError(err instanceof Error ? err : new Error('Failed to load WASM'));
          setLoading(false);
        }
      }
    }

    void init();

    return () => {
      mounted = false;
    };
  }, []);

  return { wasm, loading, error };
}
