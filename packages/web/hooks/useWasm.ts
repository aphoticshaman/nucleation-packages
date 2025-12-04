'use client';

import { useState, useEffect } from 'react';

// Stub interface - WASM removed, but keep interface for future use
export interface WasmCore {
  // Placeholder - WASM functionality removed
  available: boolean;
}

interface UseWasmResult {
  wasm: WasmCore | null;
  loading: boolean;
  error: Error | null;
}

/**
 * WASM hook - currently returns stub since nucleation-wasm was removed.
 * The app runs without WASM; simulation features are disabled.
 */
export function useWasm(): UseWasmResult {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Simulate brief loading state for UI consistency
    const timer = setTimeout(() => setReady(true), 100);
    return () => clearTimeout(timer);
  }, []);

  // Return stub - WASM not available
  return {
    wasm: ready ? { available: false } : null,
    loading: !ready,
    error: null,
  };
}
