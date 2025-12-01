'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { useWasm } from '@/hooks/useWasm';
import { useSupabaseNations } from '@/hooks/useSupabaseNations';
import ControlPanel from '@/components/ControlPanel';
import AlertBanner from '@/components/AlertBanner';

// Dynamic import for Google Maps (no SSR)
const AttractorMap = dynamic(() => import('@/components/AttractorMap'), {
  ssr: false,
  loading: () => <div className="wasm-loader" />,
});

export default function Home() {
  const { wasm, loading: wasmLoading, error: wasmError } = useWasm();
  const { nations, edges, loading: dataLoading, refetch } = useSupabaseNations();
  const [layer, setLayer] = useState<'basin' | 'risk' | 'influence' | 'regime'>('basin');
  const [alertLevel, setAlertLevel] = useState<string>('normal');
  const [isSimulating, setIsSimulating] = useState(false);

  // Run simulation step
  const runStep = async () => {
    if (!wasm) return;

    setIsSimulating(true);
    try {
      // Call WASM geospatial step
      const result = wasm.geospatial_step(nations);

      // Update alert level based on TDA
      if (result.max_transition_risk > 0.75) {
        setAlertLevel('critical');
      } else if (result.max_transition_risk > 0.5) {
        setAlertLevel('warning');
      } else if (result.max_transition_risk > 0.25) {
        setAlertLevel('elevated');
      } else {
        setAlertLevel('normal');
      }
    } finally {
      setIsSimulating(false);
    }
  };

  if (wasmLoading || dataLoading) {
    return (
      <div className="wasm-loader">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-400">
            {wasmLoading ? 'Loading WASM core...' : 'Fetching nation data...'}
          </p>
        </div>
      </div>
    );
  }

  if (wasmError) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center text-red-400">
          <p className="text-xl mb-2">Failed to load WASM</p>
          <p className="text-sm text-slate-500">{wasmError.message}</p>
        </div>
      </div>
    );
  }

  return (
    <main className="relative h-screen w-full">
      {/* Alert Banner */}
      <AlertBanner level={alertLevel} />

      {/* Map */}
      <AttractorMap
        nations={nations}
        edges={edges}
        layer={layer}
      />

      {/* Control Panel */}
      <ControlPanel
        layer={layer}
        onLayerChange={setLayer}
        onStep={runStep}
        isSimulating={isSimulating}
        alertLevel={alertLevel}
      />

      {/* Info Panel (bottom right) */}
      <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur-sm rounded-lg p-3 text-xs text-slate-400">
        <p>{nations.length} nations tracked</p>
        <p>{edges.length} influence edges</p>
        <p className="text-blue-400">Powered by Rust/WASM</p>
      </div>
    </main>
  );
}
